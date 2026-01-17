//! Deep Research Agent implementation.
//!
//! The Deep Research pattern decomposes complex queries into sub-questions,
//! executes them in parallel, and synthesizes results into a comprehensive answer.

use crate::config::DeepResearchAgentConfig;
use crate::events::{EVENT_DECOMPOSITION_COMPLETE, EVENT_SUB_QUERY_COMPLETED};

use gemicro_core::agent::{remaining_time, timeout_error, with_timeout_and_cancellation};
use gemicro_core::{
    extract_total_tokens, Agent, AgentContext, AgentError, AgentStream, AgentUpdate, ResultMetadata,
};

use async_stream::try_stream;
use futures_util::{Stream, StreamExt};
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Semaphore};

/// Buffer size for the channel collecting parallel sub-query results.
///
/// This should be at least as large as max_sub_queries (default: 5) to avoid
/// blocking completed tasks while others are still running. Using 16 provides
/// headroom for configs with larger max_sub_queries values.
const PARALLEL_EXECUTION_CHANNEL_BUFFER: usize = 16;

// Event type constants (internal to this module)
const EVENT_DECOMPOSITION_STARTED: &str = "decomposition_started";
const EVENT_SUB_QUERY_STARTED: &str = "sub_query_started";
const EVENT_SUB_QUERY_FAILED: &str = "sub_query_failed";
const EVENT_SYNTHESIS_STARTED: &str = "synthesis_started";
const EVENT_SYNTHESIS_CHUNK: &str = "synthesis_chunk";

/// Deep Research Agent.
///
/// Implements the "decompose → parallel execute → synthesize" pattern:
///
/// 1. **Decomposition**: Break the query into focused sub-questions
/// 2. **Parallel Execution**: Answer each sub-question concurrently
/// 3. **Synthesis**: Combine findings into a comprehensive answer
///
/// # Events Emitted
///
/// - `decomposition_started` / `decomposition_complete`
/// - `sub_query_started` / `sub_query_completed` / `sub_query_failed`
/// - `synthesis_started`
/// - `final_result`
///
/// # Example
///
/// ```no_run
/// use gemicro_deep_research_agent::{DeepResearchAgent, DeepResearchAgentConfig};
/// use gemicro_core::{AgentContext, LlmClient, LlmConfig};
/// use futures_util::StreamExt;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let genai_client = genai_rs::Client::builder("api-key".to_string()).build()?;
/// let context = AgentContext::new(LlmClient::new(genai_client, LlmConfig::default()));
/// let agent = DeepResearchAgent::new(DeepResearchAgentConfig::default())?;
///
/// let stream = agent.execute("What is Rust?", context);
/// futures_util::pin_mut!(stream);
/// while let Some(update) = stream.next().await {
///     println!("{:?}", update?);
/// }
/// # Ok(())
/// # }
/// ```
pub struct DeepResearchAgent {
    config: DeepResearchAgentConfig,
}

impl DeepResearchAgent {
    /// Create a new Deep Research agent with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::InvalidConfig` if the configuration is invalid
    /// (e.g., min_sub_queries > max_sub_queries, zero timeout, empty prompts).
    pub fn new(config: DeepResearchAgentConfig) -> Result<Self, AgentError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Execute the deep research process.
    ///
    /// Returns a stream of [`AgentUpdate`] events.
    pub fn execute(
        &self,
        query: &str,
        context: AgentContext,
    ) -> impl Stream<Item = Result<AgentUpdate, AgentError>> + Send + '_ {
        let query = query.to_string();
        let config = self.config.clone();

        try_stream! {
            let start_time = Instant::now();

            // Phase 1: Decomposition (with timeout and cancellation)
            yield AgentUpdate::custom(
                EVENT_DECOMPOSITION_STARTED,
                "Decomposing query into sub-queries",
                json!({}),
            );
            let decomp_timeout = remaining_time(start_time, config.total_timeout, "decomposition")?;
            let (sub_queries, decomposition_tokens) = with_timeout_and_cancellation(
                decompose(&query, &context, &config),
                decomp_timeout,
                &context.cancellation_token,
                || timeout_error(start_time, config.total_timeout, "decomposition"),
            ).await?;
            yield AgentUpdate::custom(
                EVENT_DECOMPOSITION_COMPLETE,
                format!("Decomposed into {} sub-queries", sub_queries.len()),
                json!({ "sub_queries": sub_queries }),
            );

            // Phase 2: Parallel Execution (with timeout and cancellation)
            // Yield sub_query_started events before execution
            for (id, q) in sub_queries.iter().enumerate() {
                yield AgentUpdate::custom(
                    EVENT_SUB_QUERY_STARTED,
                    format!("Sub-query {} started", id),
                    json!({ "id": id, "query": q }),
                );
            }

            let exec_timeout = remaining_time(start_time, config.total_timeout, "parallel execution")?;
            let execution_result = with_timeout_and_cancellation(
                async { Ok(execute_parallel(&sub_queries, &context, &config).await) },
                exec_timeout,
                &context.cancellation_token,
                || timeout_error(start_time, config.total_timeout, "parallel execution"),
            ).await?;

            // Yield completion/failure events as they arrived
            // Note: These arrive in non-deterministic order due to parallel execution
            for update in execution_result.updates {
                yield update;
            }

            // Check if we aborted early due to failure
            if execution_result.aborted_early {
                Err(AgentError::AllSubQueriesFailed)?;
            }

            // Check if we have at least one result
            if execution_result.results.is_empty() {
                Err(AgentError::AllSubQueriesFailed)?;
            }

            // Phase 3: Synthesis (streaming with cancellation)
            yield AgentUpdate::custom(
                EVENT_SYNTHESIS_STARTED,
                "Synthesizing results",
                json!({}),
            );

            // Build synthesis prompt
            let findings = execution_result.results
                .iter()
                .enumerate()
                .map(|(i, r)| format!("Finding {}:\n{}", i + 1, r))
                .collect::<Vec<_>>()
                .join("\n\n---\n\n");
            let synthesis_prompt = config.prompts.render_synthesis(&query, &findings);

            // Stream synthesis response
            let builder = context
                .llm
                .client()
                .interaction()
                .with_model(&config.model)
                .with_system_instruction(&config.prompts.synthesis_system)
                .with_text(synthesis_prompt);

            let synth_timeout = remaining_time(start_time, config.total_timeout, "synthesis")?;
            let synth_deadline = Instant::now() + synth_timeout;

            let mut answer = String::new();
            let stream = context.llm.generate_stream_with_cancellation(
                builder,
                context.cancellation_token.clone(),
            );
            futures_util::pin_mut!(stream);

            while let Some(chunk_result) = stream.next().await {
                // Check deadline
                if Instant::now() > synth_deadline {
                    Err(timeout_error(start_time, config.total_timeout, "synthesis"))?;
                }

                let chunk = chunk_result.map_err(|e| AgentError::SynthesisFailed(e.to_string()))?;
                answer.push_str(&chunk.text);

                // Yield streaming chunk event
                yield AgentUpdate::custom(
                    EVENT_SYNTHESIS_CHUNK,
                    chunk.text.clone(),
                    json!({ "text": chunk.text }),
                );
            }

            if answer.trim().is_empty() {
                Err(AgentError::SynthesisFailed("Empty synthesis response".to_string()))?;
            }

            // Token count unavailable for streaming responses
            let synthesis_tokens: Option<u32> = None;

            // Calculate final metadata
            let mut total_tokens = execution_result.total_tokens;
            let mut tokens_unavailable = execution_result.tokens_unavailable_count;

            // Add decomposition tokens
            match decomposition_tokens {
                Some(tokens) => total_tokens = total_tokens.saturating_add(tokens),
                None => {
                    log::warn!("Token count unavailable for decomposition call");
                    tokens_unavailable += 1;
                }
            }

            // Add synthesis tokens
            match synthesis_tokens {
                Some(tokens) => total_tokens = total_tokens.saturating_add(tokens),
                None => {
                    log::warn!("Token count unavailable for synthesis call");
                    tokens_unavailable += 1;
                }
            }

            let duration_ms = start_time.elapsed().as_millis() as u64;

            let metadata = ResultMetadata::with_extra(
                total_tokens,
                tokens_unavailable,
                duration_ms,
                json!({
                    "steps_succeeded": execution_result.succeeded,
                    "steps_failed": execution_result.failed,
                }),
            );

            yield AgentUpdate::final_result(json!(answer), metadata);
        }
    }
}

impl Agent for DeepResearchAgent {
    fn name(&self) -> &str {
        "deep_research"
    }

    fn description(&self) -> &str {
        "Decomposes queries into sub-questions, executes them in parallel, and synthesizes results"
    }

    fn execute(&self, query: &str, context: AgentContext) -> AgentStream<'_> {
        Box::pin(DeepResearchAgent::execute(self, query, context))
    }

    fn create_tracker(&self) -> Box<dyn gemicro_core::ExecutionTracking> {
        Box::new(gemicro_core::DefaultTracker::default())
    }
}

// ============================================================================
// Private implementation details
// ============================================================================

/// Result of parallel sub-query execution
struct ExecutionResult {
    /// Successful sub-query results (text)
    results: Vec<String>,
    /// Number of successful sub-queries
    succeeded: usize,
    /// Number of failed sub-queries
    failed: usize,
    /// Total tokens from successful calls
    total_tokens: u32,
    /// Number of calls where token count was unavailable
    tokens_unavailable_count: usize,
    /// Updates to yield (sub_query_completed / sub_query_failed)
    updates: Vec<AgentUpdate>,
    /// Whether execution was aborted early due to failure
    aborted_early: bool,
}

/// Decompose a query into sub-queries using the LLM
async fn decompose(
    query: &str,
    context: &AgentContext,
    config: &DeepResearchAgentConfig,
) -> Result<(Vec<String>, Option<u32>), AgentError> {
    let prompt =
        config
            .prompts
            .render_decomposition(config.min_sub_queries, config.max_sub_queries, query);

    // Use structured output to guarantee valid JSON array
    let schema = serde_json::json!({
        "type": "array",
        "items": { "type": "string" }
    });

    let request = context
        .llm
        .client()
        .interaction()
        .with_model(&config.model)
        .with_system_instruction(&config.prompts.decomposition_system)
        .with_text(prompt)
        .with_response_format(schema)
        .build()
        .map_err(|e| AgentError::Other(e.to_string()))?;

    let response = context
        .llm
        .generate(request)
        .await
        .map_err(|e| AgentError::DecompositionFailed(e.to_string()))?;

    // Extract token count
    let tokens_used = extract_total_tokens(&response);

    // Parse JSON response (guaranteed valid by response_format)
    let response_text = response.as_text().unwrap_or("");
    let sub_queries: Vec<String> = serde_json::from_str(response_text)
        .map_err(|e| AgentError::ParseFailed(format!("Failed to parse decomposition: {}", e)))?;

    // Validate bounds
    if sub_queries.is_empty() {
        return Err(AgentError::DecompositionFailed(
            "No sub-queries generated".to_string(),
        ));
    }

    if sub_queries.len() < config.min_sub_queries {
        return Err(AgentError::DecompositionFailed(format!(
            "Generated {} sub-queries but minimum is {}",
            sub_queries.len(),
            config.min_sub_queries
        )));
    }

    // Truncate if too many (with warning)
    if sub_queries.len() > config.max_sub_queries {
        log::warn!(
            "Decomposition generated {} sub-queries, truncating to {}",
            sub_queries.len(),
            config.max_sub_queries
        );
        let truncated: Vec<String> = sub_queries
            .into_iter()
            .take(config.max_sub_queries)
            .collect();
        return Ok((truncated, tokens_used));
    }

    Ok((sub_queries, tokens_used))
}

/// Execute sub-queries in parallel, collecting results
async fn execute_parallel(
    sub_queries: &[String],
    context: &AgentContext,
    config: &DeepResearchAgentConfig,
) -> ExecutionResult {
    let (tx, mut rx) = mpsc::channel::<(usize, Result<(String, Option<u32>), String>)>(
        PARALLEL_EXECUTION_CHANNEL_BUFFER,
    );

    // Create semaphore for concurrency limiting (0 means unlimited)
    let semaphore = if config.max_concurrent_sub_queries > 0 {
        Some(Arc::new(Semaphore::new(config.max_concurrent_sub_queries)))
    } else {
        None
    };

    // Spawn all sub-query tasks
    for (id, query) in sub_queries.iter().enumerate() {
        let tx = tx.clone();
        let llm = context.llm.clone();
        let query = query.clone();
        let model = config.model.clone();
        let sub_query_system = config.prompts.sub_query_system.clone();
        let semaphore = semaphore.clone();
        let cancellation_token = context.cancellation_token.clone();
        let use_google_search = config.use_google_search;

        tokio::spawn(async move {
            // Acquire semaphore permit if concurrency is limited.
            let _permit = match &semaphore {
                Some(sem) => Some(sem.acquire().await.expect("semaphore closed unexpectedly")),
                None => None,
            };

            let mut builder = llm
                .client()
                .interaction()
                .with_model(&model)
                .with_system_instruction(&sub_query_system)
                .with_text(&query);
            if use_google_search {
                builder = builder.with_google_search();
            }
            let request = match builder.build() {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx
                        .send((id, Err(format!("Failed to build request: {e}"))))
                        .await;
                    return;
                }
            };

            // Execute LLM call with cancellation support
            let result = match llm
                .generate_with_cancellation(request, &cancellation_token)
                .await
            {
                Ok(response) => {
                    let text = response.as_text().unwrap_or("").to_string();
                    let tokens = extract_total_tokens(&response);
                    Ok((text, tokens))
                }
                Err(gemicro_core::LlmError::Cancelled) => {
                    log::debug!("Sub-query {} cancelled", id);
                    return; // Exit early, don't send result
                }
                Err(e) => Err(e.to_string()),
            };

            // Check cancellation again before sending
            if cancellation_token.is_cancelled() {
                log::debug!(
                    "Sub-query {} completed but cancelled, discarding result",
                    id
                );
                return;
            }

            if tx.send((id, result)).await.is_err() {
                log::debug!("Receiver dropped, sub-query {} result discarded", id);
            }
        });
    }

    // Drop sender so channel closes when all tasks complete
    drop(tx);

    // Collect results
    let mut results = Vec::new();
    let mut updates = Vec::new();
    let mut succeeded = 0;
    let mut failed = 0;
    let mut total_tokens = 0u32;
    let mut tokens_unavailable_count = 0usize;
    let mut aborted_early = false;

    loop {
        tokio::select! {
            biased;

            _ = context.cancellation_token.cancelled() => {
                log::debug!("Parallel execution cancelled, collecting in-flight results");

                // Grace period: give nearly-complete tasks a brief window to finish
                let grace_period = Duration::from_millis(100);
                let deadline = tokio::time::Instant::now() + grace_period;

                loop {
                    match tokio::time::timeout_at(deadline, rx.recv()).await {
                        Ok(Some((id, result))) => {
                            match result {
                                Ok((text, tokens)) => {
                                    match tokens {
                                        Some(t) => total_tokens = total_tokens.saturating_add(t),
                                        None => {
                                            log::warn!("Token count unavailable for sub-query {}", id);
                                            tokens_unavailable_count += 1;
                                        }
                                    }
                                    updates.push(AgentUpdate::custom(
                                        EVENT_SUB_QUERY_COMPLETED,
                                        format!("Sub-query {} completed", id),
                                        json!({
                                            "id": id,
                                            "result": &text,
                                            "tokens_used": tokens.unwrap_or(0),
                                        }),
                                    ));
                                    results.push(text);
                                    succeeded += 1;
                                }
                                Err(error) => {
                                    updates.push(AgentUpdate::custom(
                                        EVENT_SUB_QUERY_FAILED,
                                        format!("Sub-query {} failed", id),
                                        json!({ "id": id, "error": error }),
                                    ));
                                    failed += 1;
                                }
                            }
                        }
                        Ok(None) => break,
                        Err(_) => {
                            log::debug!("Grace period expired, {} results collected", succeeded);
                            break;
                        }
                    }
                }

                aborted_early = true;
                break;
            }

            recv_result = rx.recv() => {
                match recv_result {
                    Some((id, result)) => {
                        match result {
                            Ok((text, tokens)) => {
                                match tokens {
                                    Some(t) => total_tokens = total_tokens.saturating_add(t),
                                    None => {
                                        log::warn!("Token count unavailable for sub-query {}", id);
                                        tokens_unavailable_count += 1;
                                    }
                                }
                                updates.push(AgentUpdate::custom(
                                    EVENT_SUB_QUERY_COMPLETED,
                                    format!("Sub-query {} completed", id),
                                    json!({
                                        "id": id,
                                        "result": &text,
                                        "tokens_used": tokens.unwrap_or(0),
                                    }),
                                ));
                                results.push(text);
                                succeeded += 1;
                            }
                            Err(error) => {
                                updates.push(AgentUpdate::custom(
                                    EVENT_SUB_QUERY_FAILED,
                                    format!("Sub-query {} failed", id),
                                    json!({ "id": id, "error": error }),
                                ));
                                failed += 1;

                                // Abort early if configured to fail fast
                                if !config.continue_on_partial_failure {
                                    log::warn!(
                                        "Sub-query {} failed and continue_on_partial_failure=false, aborting",
                                        id
                                    );
                                    aborted_early = true;
                                    break;
                                }
                            }
                        }
                    }
                    None => {
                        // Channel closed, all tasks completed
                        break;
                    }
                }
            }
        }
    }

    ExecutionResult {
        results,
        succeeded,
        failed,
        total_tokens,
        tokens_unavailable_count,
        updates,
        aborted_early,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_creation_valid_config() {
        let config = DeepResearchAgentConfig::default();
        let agent = DeepResearchAgent::new(config);
        assert!(agent.is_ok());
    }

    #[test]
    fn test_agent_creation_invalid_config() {
        let config = DeepResearchAgentConfig {
            min_sub_queries: 10,
            max_sub_queries: 5, // Invalid: min > max
            ..Default::default()
        };

        let agent = DeepResearchAgent::new(config);
        assert!(agent.is_err());
        assert!(matches!(agent, Err(AgentError::InvalidConfig(_))));
    }

    #[test]
    fn test_agent_name_and_description() {
        let agent = DeepResearchAgent::new(DeepResearchAgentConfig::default()).unwrap();
        assert_eq!(agent.name(), "deep_research");
        assert!(!agent.description().is_empty());
    }
}
