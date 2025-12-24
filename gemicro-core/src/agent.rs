//! Deep Research Agent implementation
//!
//! This module provides the `DeepResearchAgent` which implements the Deep Research
//! pattern: decompose a complex query into sub-queries, execute them in parallel,
//! and synthesize the results.
//!
//! # Example
//!
//! ```no_run
//! use gemicro_core::{AgentContext, DeepResearchAgent, ResearchConfig, LlmClient, LlmConfig};
//! use futures_util::StreamExt;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let genai_client = rust_genai::Client::builder("api-key".to_string()).build();
//! let llm = LlmClient::new(genai_client, LlmConfig::default());
//!
//! let context = AgentContext::new(llm);
//! let agent = DeepResearchAgent::new(ResearchConfig::default())?;
//!
//! let stream = agent.execute("What are the trends in quantum computing?", context);
//! futures_util::pin_mut!(stream);
//!
//! while let Some(update) = stream.next().await {
//!     let update = update?;
//!     println!("[{}] {}", update.event_type, update.message);
//! }
//! # Ok(())
//! # }
//! ```

use crate::error::AgentError;
use crate::llm::{LlmClient, LlmRequest};
use crate::update::{AgentUpdate, ResultMetadata};
use crate::ResearchConfig;

use async_stream::try_stream;
use futures_util::stream::Stream;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Buffer size for the mpsc channel used in parallel sub-query execution.
/// This should be at least as large as the expected number of concurrent sub-queries
/// to avoid blocking on send operations.
const PARALLEL_EXECUTION_CHANNEL_BUFFER: usize = 16;

/// Calculate remaining time from total timeout, returning error if already exceeded
fn remaining_time(
    start: Instant,
    total_timeout: Duration,
    phase: &str,
) -> Result<Duration, AgentError> {
    let elapsed = start.elapsed();
    if elapsed >= total_timeout {
        Err(AgentError::Timeout {
            elapsed_ms: elapsed.as_millis() as u64,
            timeout_ms: total_timeout.as_millis() as u64,
            phase: phase.to_string(),
        })
    } else {
        Ok(total_timeout - elapsed)
    }
}

/// Convert a tokio timeout error to AgentError::Timeout
fn timeout_error(start: Instant, total_timeout: Duration, phase: &str) -> AgentError {
    AgentError::Timeout {
        elapsed_ms: start.elapsed().as_millis() as u64,
        timeout_ms: total_timeout.as_millis() as u64,
        phase: phase.to_string(),
    }
}

/// Truncate text for error messages to prevent huge output
fn truncate_for_error(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        text.to_string()
    } else {
        let truncated: String = text.chars().take(max_chars).collect();
        format!("{}... ({} chars total)", truncated, char_count)
    }
}

/// Shared resources for agent execution
///
/// Following [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) philosophy:
/// Contains ONLY cross-agent resources. Agent-specific config goes to constructors, not here.
#[derive(Clone)]
pub struct AgentContext {
    /// LLM client (Arc-wrapped for cloning across parallel tasks)
    pub llm: Arc<LlmClient>,
}

impl AgentContext {
    /// Create a new agent context from an LlmClient
    pub fn new(llm: LlmClient) -> Self {
        Self { llm: Arc::new(llm) }
    }

    /// Create from an existing Arc (useful when sharing across agents)
    pub fn from_arc(llm: Arc<LlmClient>) -> Self {
        Self { llm }
    }
}

/// Deep Research agent
///
/// Implements the Deep Research pattern:
/// 1. **Decomposition**: Break query into N sub-queries (configurable)
/// 2. **Parallel Execution**: Execute sub-queries concurrently, streaming updates
/// 3. **Synthesis**: Combine results into comprehensive final answer
///
/// # Design
///
/// - Agent-specific config (`ResearchConfig`) passed at construction
/// - Returns a stream of `AgentUpdate` events for real-time observability
/// - Gracefully handles partial failures (continues if some sub-queries fail)
pub struct DeepResearchAgent {
    config: ResearchConfig,
}

impl DeepResearchAgent {
    /// Create a new Deep Research agent with the given configuration
    ///
    /// # Errors
    ///
    /// Returns `AgentError::InvalidConfig` if the configuration is invalid.
    pub fn new(config: ResearchConfig) -> Result<Self, AgentError> {
        config.validate().map_err(AgentError::InvalidConfig)?;
        Ok(Self { config })
    }

    /// Execute the research, returning a stream of updates
    ///
    /// The stream yields `AgentUpdate` events as work progresses:
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

            // Phase 1: Decomposition (with timeout)
            yield AgentUpdate::decomposition_started();
            let decomp_timeout = remaining_time(start_time, config.total_timeout, "decomposition")?;
            let sub_queries = tokio::time::timeout(
                decomp_timeout,
                decompose(&query, &context, &config)
            )
            .await
            .map_err(|_| timeout_error(start_time, config.total_timeout, "decomposition"))??;
            yield AgentUpdate::decomposition_complete(sub_queries.clone());

            // Phase 2: Parallel Execution (with timeout)
            // Yield sub_query_started events before execution
            for (id, q) in sub_queries.iter().enumerate() {
                yield AgentUpdate::sub_query_started(id, q.clone());
            }

            let exec_timeout = remaining_time(start_time, config.total_timeout, "parallel execution")?;
            let execution_result = tokio::time::timeout(
                exec_timeout,
                execute_parallel(&sub_queries, &context, &config)
            )
            .await
            .map_err(|_| timeout_error(start_time, config.total_timeout, "parallel execution"))?;

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

            // Phase 3: Synthesis (with timeout)
            yield AgentUpdate::synthesis_started();
            let synth_timeout = remaining_time(start_time, config.total_timeout, "synthesis")?;
            let (answer, synthesis_tokens) = tokio::time::timeout(
                synth_timeout,
                synthesize(&query, &execution_result.results, &context, &config)
            )
            .await
            .map_err(|_| timeout_error(start_time, config.total_timeout, "synthesis"))??;

            // Calculate final metadata
            let mut total_tokens = execution_result.total_tokens;
            let mut tokens_unavailable = execution_result.tokens_unavailable_count;

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
async fn decompose(
    query: &str,
    context: &AgentContext,
    config: &ResearchConfig,
) -> Result<Vec<String>, AgentError> {
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
        return Ok(sub_queries
            .into_iter()
            .take(config.max_sub_queries)
            .collect());
    }

    Ok(sub_queries)
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
                    truncate_for_error(text, 200)
                )
            });
        }
    }

    Err(format!(
        "Invalid JSON: expected array of strings. Response was: {}",
        truncate_for_error(text, 200)
    ))
}

/// Execute sub-queries in parallel, collecting results
///
/// # Arguments
///
/// * `sub_queries` - The sub-queries to execute
/// * `context` - Agent context with LLM client
/// * `config` - Research configuration (for prompts and continue_on_partial_failure)
///
/// # Returns
///
/// Results with updates in non-deterministic order (depends on which queries complete first).
async fn execute_parallel(
    sub_queries: &[String],
    context: &AgentContext,
    config: &ResearchConfig,
) -> ExecutionResult {
    let (tx, mut rx) = mpsc::channel::<(usize, Result<(String, Option<u32>), String>)>(
        PARALLEL_EXECUTION_CHANNEL_BUFFER,
    );

    // Spawn all sub-query tasks
    for (id, query) in sub_queries.iter().enumerate() {
        let tx = tx.clone();
        let llm = context.llm.clone();
        let query = query.clone();
        let sub_query_system = config.prompts.sub_query_system.clone();

        tokio::spawn(async move {
            let request = LlmRequest::with_system(&query, &sub_query_system);

            let result = match llm.generate(request).await {
                Ok(response) => Ok((response.text, response.tokens_used)),
                Err(e) => Err(e.to_string()),
            };

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

    while let Some((id, result)) = rx.recv().await {
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
        let mut config = ResearchConfig::default();
        config.min_sub_queries = 10;
        config.max_sub_queries = 5; // Invalid: min > max

        let agent = DeepResearchAgent::new(config);
        assert!(agent.is_err());
        assert!(matches!(agent, Err(AgentError::InvalidConfig(_))));
    }

    #[test]
    fn test_agent_context_new() {
        // Just verify it compiles - we can't easily test without a real client
        let _context_fn = |llm: LlmClient| AgentContext::new(llm);
    }
}
