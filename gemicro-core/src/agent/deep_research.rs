//! Deep Research Agent implementation.
//!
//! The Deep Research pattern decomposes complex queries into sub-questions,
//! executes them in parallel, and synthesizes results into a comprehensive answer.

use super::{remaining_time, timeout_error, with_timeout_and_cancellation, Agent, AgentContext};
use crate::config::ResearchConfig;
use crate::error::AgentError;
use crate::llm::LlmRequest;
use crate::update::{AgentUpdate, ResultMetadata};

use async_stream::try_stream;
use futures_util::Stream;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Semaphore};

/// Buffer size for the channel collecting parallel sub-query results.
///
/// This should be at least as large as max_sub_queries (default: 5) to avoid
/// blocking completed tasks while others are still running. Using 16 provides
/// headroom for configs with larger max_sub_queries values.
const PARALLEL_EXECUTION_CHANNEL_BUFFER: usize = 16;

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
/// use gemicro_core::{AgentContext, DeepResearchAgent, ResearchConfig, LlmClient, LlmConfig};
/// use futures_util::StreamExt;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let genai_client = rust_genai::Client::builder("api-key".to_string()).build();
/// let context = AgentContext::new(LlmClient::new(genai_client, LlmConfig::default()));
/// let agent = DeepResearchAgent::new(ResearchConfig::default())?;
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
    config: ResearchConfig,
}

impl DeepResearchAgent {
    /// Create a new Deep Research agent with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::InvalidConfig` if the configuration is invalid
    /// (e.g., min_sub_queries > max_sub_queries, zero timeout, empty prompts).
    pub fn new(config: ResearchConfig) -> Result<Self, AgentError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Execute the deep research process.
    ///
    /// This is the main entry point that orchestrates:
    /// 1. Query decomposition into sub-questions
    /// 2. Parallel execution of sub-queries
    /// 3. Synthesis of results into a final answer
    ///
    /// Returns a stream of [`AgentUpdate`] events:
    /// - `decomposition_started` / `decomposition_complete`
    /// - `sub_query_started` / `sub_query_completed` / `sub_query_failed`
    /// - `synthesis_started`
    /// - `final_result`
    ///
    /// # Arguments
    ///
    /// * `query` - The user's research query
    /// * `context` - Shared resources (LLM client)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gemicro_core::{AgentContext, DeepResearchAgent, ResearchConfig, LlmClient, LlmConfig};
    /// use futures_util::StreamExt;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let genai_client = rust_genai::Client::builder("api-key".to_string()).build();
    /// let context = AgentContext::new(LlmClient::new(genai_client, LlmConfig::default()));
    /// let agent = DeepResearchAgent::new(ResearchConfig::default())?;
    ///
    /// let stream = agent.execute("What is Rust?", context);
    /// futures_util::pin_mut!(stream);
    /// while let Some(update) = stream.next().await {
    ///     println!("{:?}", update?);
    /// }
    /// # Ok(())
    /// # }
    /// ```
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
            yield AgentUpdate::decomposition_started();
            let decomp_timeout = remaining_time(start_time, config.total_timeout, "decomposition")?;
            let (sub_queries, decomposition_tokens) = with_timeout_and_cancellation(
                decompose(&query, &context, &config),
                decomp_timeout,
                &context.cancellation_token,
                || timeout_error(start_time, config.total_timeout, "decomposition"),
            ).await?;
            yield AgentUpdate::decomposition_complete(sub_queries.clone());

            // Phase 2: Parallel Execution (with timeout and cancellation)
            // Yield sub_query_started events before execution
            for (id, q) in sub_queries.iter().enumerate() {
                yield AgentUpdate::sub_query_started(id, q.clone());
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

            // Phase 3: Synthesis (with timeout and cancellation)
            yield AgentUpdate::synthesis_started();
            let synth_timeout = remaining_time(start_time, config.total_timeout, "synthesis")?;
            let (answer, synthesis_tokens) = with_timeout_and_cancellation(
                synthesize(&query, &execution_result.results, &context, &config),
                synth_timeout,
                &context.cancellation_token,
                || timeout_error(start_time, config.total_timeout, "synthesis"),
            ).await?;

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

            let metadata = ResultMetadata {
                total_tokens,
                tokens_unavailable_count: tokens_unavailable,
                duration_ms,
                sub_queries_succeeded: execution_result.succeeded,
                sub_queries_failed: execution_result.failed,
            };

            yield AgentUpdate::final_result(answer, metadata);
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

    fn execute(&self, query: &str, context: AgentContext) -> super::AgentStream<'_> {
        Box::pin(DeepResearchAgent::execute(self, query, context))
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
    ///
    /// Note: These updates arrive in non-deterministic order due to parallel
    /// execution. The order depends on which sub-queries complete first.
    updates: Vec<AgentUpdate>,
    /// Whether execution was aborted early due to failure (when continue_on_partial_failure=false)
    aborted_early: bool,
}

/// Decompose a query into sub-queries using the LLM
///
/// Returns a tuple of (sub_queries, tokens_used) where tokens_used is the
/// token count from the decomposition LLM call, if available.
async fn decompose(
    query: &str,
    context: &AgentContext,
    config: &ResearchConfig,
) -> Result<(Vec<String>, Option<u32>), AgentError> {
    let prompt =
        config
            .prompts
            .render_decomposition(config.min_sub_queries, config.max_sub_queries, query);

    let request = LlmRequest::with_system(prompt, &config.prompts.decomposition_system);

    let response = context
        .llm
        .generate(request)
        .await
        .map_err(|e| AgentError::DecompositionFailed(e.to_string()))?;

    // Extract token count before consuming response text for parsing
    let tokens_used = response.tokens_used;

    // Parse JSON response
    let sub_queries: Vec<String> = parse_json_array(&response.text)
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

/// Parse a JSON array from LLM response, handling common formatting issues
fn parse_json_array(text: &str) -> Result<Vec<String>, String> {
    // Fast path: try direct parse first (avoids allocations for well-formed JSON)
    if let Ok(arr) = serde_json::from_str::<Vec<String>>(text) {
        return Ok(arr);
    }

    // Try with trimmed whitespace
    let trimmed = text.trim();
    if let Ok(arr) = serde_json::from_str::<Vec<String>>(trimmed) {
        return Ok(arr);
    }

    // Slow path: only extract from markdown code block if markers are present
    if trimmed.starts_with("```") {
        let lines: Vec<&str> = trimmed.lines().collect();
        if lines.len() >= 2 {
            let start = if lines[0].starts_with("```json") || lines[0] == "```" {
                1
            } else {
                0
            };
            let end = if lines.last() == Some(&"```") {
                lines.len() - 1
            } else {
                lines.len()
            };
            let json_text = lines[start..end].join("\n");
            return serde_json::from_str::<Vec<String>>(&json_text).map_err(|e| {
                format!(
                    "Invalid JSON: {}. Response was: {}",
                    e,
                    crate::utils::truncate_with_count(text, 200)
                )
            });
        }
    }

    Err(format!(
        "Invalid JSON: expected array of strings. Response was: {}",
        crate::utils::truncate_with_count(text, 200)
    ))
}

/// Execute sub-queries in parallel, collecting results
///
/// # Arguments
///
/// * `sub_queries` - The sub-queries to execute
/// * `context` - Agent context with LLM client and cancellation token
/// * `config` - Research configuration (for prompts, continue_on_partial_failure, and concurrency limit)
///
/// # Returns
///
/// Results with updates in non-deterministic order (depends on which queries complete first).
/// If cancelled, returns partial results collected so far with `aborted_early = true`.
async fn execute_parallel(
    sub_queries: &[String],
    context: &AgentContext,
    config: &ResearchConfig,
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
        let sub_query_system = config.prompts.sub_query_system.clone();
        let semaphore = semaphore.clone();
        let cancellation_token = context.cancellation_token.clone();
        let use_google_search = config.use_google_search;

        tokio::spawn(async move {
            // Acquire semaphore permit if concurrency is limited.
            // The permit is held in _permit and automatically released when
            // the task exits (via Drop), ensuring proper cleanup on all paths
            // including early returns from cancellation.
            let _permit = match &semaphore {
                Some(sem) => Some(sem.acquire().await.expect("semaphore closed unexpectedly")),
                None => None,
            };

            let mut request = LlmRequest::with_system(&query, &sub_query_system);
            if use_google_search {
                request = request.with_google_search();
            }

            // Execute LLM call with cancellation support
            let result = match llm
                .generate_with_cancellation(request, &cancellation_token)
                .await
            {
                Ok(response) => Ok((response.text, response.tokens_used)),
                Err(crate::LlmError::Cancelled) => {
                    log::debug!("Sub-query {} cancelled", id);
                    return; // Exit early, don't send result
                }
                Err(e) => Err(e.to_string()),
            };

            // Check cancellation again before sending - avoids queuing results
            // after cancellation was triggered during the LLM call
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
            // biased ensures cancellation is always checked first, providing
            // deterministic responsiveness to cancel requests
            biased;

            _ = context.cancellation_token.cancelled() => {
                log::debug!("Parallel execution cancelled, collecting in-flight results");

                // Grace period: give nearly-complete tasks a brief window to finish
                // This collects results that were in-flight when cancellation occurred
                let grace_period = Duration::from_millis(100);
                let deadline = tokio::time::Instant::now() + grace_period;

                loop {
                    match tokio::time::timeout_at(deadline, rx.recv()).await {
                        Ok(Some((id, result))) => {
                            // Process result normally
                            match result {
                                Ok((text, tokens)) => {
                                    match tokens {
                                        Some(t) => total_tokens = total_tokens.saturating_add(t),
                                        None => {
                                            log::warn!("Token count unavailable for sub-query {}", id);
                                            tokens_unavailable_count += 1;
                                        }
                                    }
                                    updates.push(AgentUpdate::sub_query_completed(
                                        id,
                                        text.clone(),
                                        tokens.unwrap_or(0),
                                    ));
                                    results.push(text);
                                    succeeded += 1;
                                }
                                Err(error) => {
                                    updates.push(AgentUpdate::sub_query_failed(id, error));
                                    failed += 1;
                                }
                            }
                        }
                        Ok(None) => {
                            // Channel closed, all tasks completed
                            break;
                        }
                        Err(_) => {
                            // Grace period expired
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
                                updates.push(AgentUpdate::sub_query_completed(
                                    id,
                                    text.clone(),
                                    tokens.unwrap_or(0),
                                ));
                                results.push(text);
                                succeeded += 1;
                            }
                            Err(error) => {
                                updates.push(AgentUpdate::sub_query_failed(id, error));
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

/// Synthesize sub-query results into a final answer
async fn synthesize(
    original_query: &str,
    results: &[String],
    context: &AgentContext,
    config: &ResearchConfig,
) -> Result<(String, Option<u32>), AgentError> {
    let findings = results
        .iter()
        .enumerate()
        .map(|(i, r)| format!("Finding {}:\n{}", i + 1, r))
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");

    let prompt = config.prompts.render_synthesis(original_query, &findings);

    let request = LlmRequest::with_system(prompt, &config.prompts.synthesis_system);

    let response = context
        .llm
        .generate(request)
        .await
        .map_err(|e| AgentError::SynthesisFailed(e.to_string()))?;

    if response.text.trim().is_empty() {
        return Err(AgentError::SynthesisFailed(
            "Empty synthesis response".to_string(),
        ));
    }

    Ok((response.text, response.tokens_used))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json_array_simple() {
        let input = r#"["Question 1?", "Question 2?"]"#;
        let result = parse_json_array(input).unwrap();
        assert_eq!(result, vec!["Question 1?", "Question 2?"]);
    }

    #[test]
    fn test_parse_json_array_with_markdown() {
        let input = r#"```json
["Question 1?", "Question 2?"]
```"#;
        let result = parse_json_array(input).unwrap();
        assert_eq!(result, vec!["Question 1?", "Question 2?"]);
    }

    #[test]
    fn test_parse_json_array_with_plain_markdown() {
        let input = r#"```
["Question 1?", "Question 2?"]
```"#;
        let result = parse_json_array(input).unwrap();
        assert_eq!(result, vec!["Question 1?", "Question 2?"]);
    }

    #[test]
    fn test_parse_json_array_invalid() {
        let input = "Not JSON at all";
        let result = parse_json_array(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_json_array_empty() {
        let input = "[]";
        let result = parse_json_array(input).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_json_array_with_whitespace() {
        let input = r#"  [  "Question 1?"  ,  "Question 2?"  ]  "#;
        let result = parse_json_array(input).unwrap();
        assert_eq!(result, vec!["Question 1?", "Question 2?"]);
    }

    #[test]
    fn test_parse_json_array_single_item() {
        let input = r#"["Only one question?"]"#;
        let result = parse_json_array(input).unwrap();
        assert_eq!(result, vec!["Only one question?"]);
    }

    #[test]
    fn test_parse_json_array_wrong_type() {
        // Array of numbers instead of strings
        let input = "[1, 2, 3]";
        let result = parse_json_array(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_json_array_object_instead() {
        let input = r#"{"question": "What?"}"#;
        let result = parse_json_array(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_agent_creation_valid_config() {
        let config = ResearchConfig::default();
        let agent = DeepResearchAgent::new(config);
        assert!(agent.is_ok());
    }

    #[test]
    fn test_agent_creation_invalid_config() {
        let config = ResearchConfig {
            min_sub_queries: 10,
            max_sub_queries: 5, // Invalid: min > max
            ..Default::default()
        };

        let agent = DeepResearchAgent::new(config);
        assert!(agent.is_err());
        assert!(matches!(agent, Err(AgentError::InvalidConfig(_))));
    }

    // parse_json_array edge case tests

    #[test]
    fn test_parse_json_array_nested_arrays() {
        // Nested arrays should fail (we expect Vec<String>, not Vec<Vec<...>>)
        let input = r#"[["nested"]]"#;
        let result = parse_json_array(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_json_array_mixed_types() {
        // Mixed strings and numbers should fail
        let input = r#"["question", 42, "another"]"#;
        let result = parse_json_array(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_json_array_null_values() {
        // Null values in array should fail
        let input = r#"["question", null, "another"]"#;
        let result = parse_json_array(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_json_array_large_array() {
        // Large array should work
        let items: Vec<String> = (0..100).map(|i| format!("Question {}?", i)).collect();
        let input = serde_json::to_string(&items).unwrap();
        let result = parse_json_array(&input).unwrap();
        assert_eq!(result.len(), 100);
        assert_eq!(result[0], "Question 0?");
        assert_eq!(result[99], "Question 99?");
    }

    #[test]
    fn test_parse_json_array_unicode_content() {
        let input = r#"["你好吗?", "こんにちは?", "🤔 What?"]"#;
        let result = parse_json_array(input).unwrap();
        assert_eq!(result, vec!["你好吗?", "こんにちは?", "🤔 What?"]);
    }

    #[test]
    fn test_parse_json_array_escaped_quotes() {
        let input = r#"["Question with \"quotes\"?", "Normal question?"]"#;
        let result = parse_json_array(input).unwrap();
        assert_eq!(
            result,
            vec!["Question with \"quotes\"?", "Normal question?"]
        );
    }

    #[test]
    fn test_parse_json_array_error_message_contains_response() {
        let input = "invalid json here";
        let result = parse_json_array(input);
        assert!(result.is_err());
        let error_msg = result.unwrap_err();
        assert!(
            error_msg.contains("invalid json here"),
            "Error should contain the invalid response text"
        );
    }
}
