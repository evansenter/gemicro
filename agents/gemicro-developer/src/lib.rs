//! Developer agent with explicit function calling for real-time tool events.
//!
//! Uses an explicit function calling loop (rather than rust-genai's automatic mode)
//! to emit `tool_call_started` and `tool_result` events for real-time CLI feedback.
//!
//! # Architecture
//!
//! ```text
//! LLM Request → Response with function_calls
//!     ↓
//! For each function_call:
//!     yield tool_call_started event
//!     execute tool
//!     yield tool_result event
//!     ↓
//! Pass results back to LLM → loop until text response
//! ```
//!
//! # Example
//!
//! ```no_run
//! use gemicro_developer::{DeveloperAgent, DeveloperConfig};
//! use gemicro_core::{Agent, AgentContext, LlmClient, LlmConfig};
//! use futures_util::StreamExt;
//!
//! # async fn example() -> Result<(), gemicro_core::AgentError> {
//! let genai_client = rust_genai::Client::builder("your-api-key".to_string()).build();
//! let llm = LlmClient::new(genai_client, LlmConfig::default());
//! let context = AgentContext::new(llm);
//!
//! let config = DeveloperConfig::default();
//! let agent = DeveloperAgent::new(config)?;
//!
//! let stream = agent.execute("Read the CLAUDE.md file", context);
//! futures_util::pin_mut!(stream);
//!
//! while let Some(result) = stream.next().await {
//!     let update = result?;
//!     println!("[{}] {}", update.event_type, update.message);
//! }
//! # Ok(())
//! # }
//! ```

mod config;
mod events;

pub use config::DeveloperConfig;

use async_stream::try_stream;
use gemicro_core::{
    tool::{AutoDeny, ConfirmationHandler, ToolError},
    Agent, AgentContext, AgentError, AgentStream, AgentUpdate, DefaultTracker, ExecutionTracking,
    LlmError, ResultMetadata, MODEL,
};
use rust_genai::{
    function_result_content, FunctionDeclaration, InteractionContent, InteractionResponse,
    OwnedFunctionCallInfo,
};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;

/// Build the final result event with answer and metadata.
fn build_final_result(
    response: &rust_genai::InteractionResponse,
    start: Instant,
    total_tool_calls: usize,
    iteration: usize,
) -> AgentUpdate {
    let answer = response.text().map(|s| s.to_string()).unwrap_or_else(|| {
        log::warn!("LLM response contained no text content");
        String::new()
    });
    let total_tokens = gemicro_core::extract_total_tokens(response).unwrap_or_else(|| {
        log::debug!("Could not extract token count from response");
        0
    });
    let duration_ms = start.elapsed().as_millis() as u64;

    AgentUpdate::final_result(
        json!(answer),
        ResultMetadata::with_extra(
            total_tokens,
            0, // tokens_unavailable_count: we only make one final LLM call
            duration_ms,
            json!({
                "tool_call_count": total_tool_calls,
                "iterations": iteration,
            }),
        ),
    )
}

/// Build a final result for incomplete execution (cancellation, max iterations).
///
/// Per the event contract, `final_result` MUST always be emitted as the last event.
fn build_incomplete_result(
    reason: &str,
    start: Instant,
    total_tool_calls: usize,
    iteration: usize,
) -> AgentUpdate {
    let duration_ms = start.elapsed().as_millis() as u64;

    AgentUpdate::final_result(
        json!(format!("[Execution incomplete: {}]", reason)),
        ResultMetadata::with_extra(
            0, // total_tokens: unavailable for incomplete executions
            0, // tokens_unavailable_count: not tracked for incomplete
            duration_ms,
            json!({
                "tool_call_count": total_tool_calls,
                "iterations": iteration,
                "incomplete": true,
                "reason": reason,
            }),
        ),
    )
}

/// Developer agent with explicit function calling loop.
///
/// Provides real-time tool execution events for CLI display.
pub struct DeveloperAgent {
    config: DeveloperConfig,
}

impl DeveloperAgent {
    /// Create a new DeveloperAgent with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(config: DeveloperConfig) -> Result<Self, AgentError> {
        config.validate()?;
        Ok(Self { config })
    }
}

impl Agent for DeveloperAgent {
    fn name(&self) -> &str {
        "developer"
    }

    fn description(&self) -> &str {
        "Developer agent with real-time tool execution events"
    }

    fn execute(&self, query: &str, context: AgentContext) -> AgentStream<'_> {
        let start = Instant::now();
        let query = query.to_string();
        let max_iterations = self.config.max_iterations;
        let tool_filter = self.config.tool_filter.clone();
        let system_prompt = self.config.build_system_prompt();

        Box::pin(try_stream! {
            // ══════════════════════════════════════════════════════════════════
            // SECTION 1: Initialization
            // ══════════════════════════════════════════════════════════════════

            yield AgentUpdate::custom(
                events::EVENT_DEVELOPER_STARTED,
                "Starting developer agent",
                json!({ "query": &query }),
            );

            let tools = context.tools.clone().ok_or_else(|| {
                AgentError::InvalidConfig("DeveloperAgent requires tools in context".into())
            })?;

            let confirmation_handler: Arc<dyn ConfirmationHandler> =
                context.confirmation_handler.clone().unwrap_or_else(|| Arc::new(AutoDeny));

            let filtered_tools = tools.filter(&tool_filter);
            let function_declarations: Vec<FunctionDeclaration> = filtered_tools
                .iter()
                .map(|t| t.to_function_declaration())
                .collect();

            let genai_client = context.llm.client();

            let mut total_tool_calls = 0usize;
            let mut iteration = 0usize;
            let mut previous_interaction_id: Option<String> = None;
            let mut pending_function_calls: Vec<OwnedFunctionCallInfo> = Vec::new();

            // ══════════════════════════════════════════════════════════════════
            // SECTION 2: Main Function Calling Loop
            // ══════════════════════════════════════════════════════════════════
            loop {
                // Check for cancellation at start of each iteration
                if context.cancellation_token.is_cancelled() {
                    log::info!("DeveloperAgent cancelled");
                    yield build_incomplete_result("cancelled", start, total_tool_calls, iteration);
                    break;
                }

                iteration += 1;
                if iteration > max_iterations {
                    log::warn!("DeveloperAgent hit max iterations ({})", max_iterations);
                    yield build_incomplete_result(
                        &format!("max iterations ({}) reached", max_iterations),
                        start,
                        total_tool_calls,
                        iteration,
                    );
                    break;
                }

                // Determine what function calls to process this iteration
                let function_calls_to_process = if !pending_function_calls.is_empty() {
                    // Process pending function calls from previous iteration
                    std::mem::take(&mut pending_function_calls)
                } else {
                    // No pending calls - make LLM request
                    // Type-state pattern: FirstTurn vs Chained builders are different types
                    let response: InteractionResponse = match &previous_interaction_id {
                        Some(prev_id) => {
                            // Subsequent turns: chain to previous interaction
                            genai_client
                                .interaction()
                                .with_model(MODEL)
                                .with_functions(function_declarations.clone())
                                .with_store_enabled()
                                .with_previous_interaction(prev_id)
                                .create()
                                .await
                                .map_err(|e| AgentError::Llm(LlmError::GenAi(e)))?
                        }
                        None => {
                            // First turn: set system instruction and user query
                            genai_client
                                .interaction()
                                .with_model(MODEL)
                                .with_system_instruction(&system_prompt)
                                .with_functions(function_declarations.clone())
                                .with_store_enabled()
                                .with_text(&query)
                                .create()
                                .await
                                .map_err(|e| AgentError::Llm(LlmError::GenAi(e)))?
                        }
                    };

                    let calls = response.function_calls();

                    if calls.is_empty() {
                        // No function calls - extract final answer and exit
                        yield build_final_result(&response, start, total_tool_calls, iteration);
                        break;
                    }

                    // Store interaction ID for conversation context
                    previous_interaction_id = response.id.clone();

                    // Convert to owned for processing across iterations
                    calls.iter().map(|c| c.to_owned()).collect()
                };

                // ─────────────────────────────────────────────────────────────
                // SECTION 2a: Execute Function Calls with Real-Time Events
                // ─────────────────────────────────────────────────────────────

                let mut function_results: Vec<InteractionContent> = Vec::new();
                let mut cancelled = false;

                for fc in &function_calls_to_process {
                    let tool_name = &fc.name;
                    let call_id = fc.id.as_deref().unwrap_or_else(|| {
                        log::debug!("Function call missing ID, using 'unknown'");
                        "unknown"
                    });
                    let arguments = &fc.args;

                    // Emit tool_call_started event BEFORE execution
                    yield AgentUpdate::custom(
                        events::EVENT_TOOL_CALL_STARTED,
                        format!("Calling tool: {}", tool_name),
                        json!({
                            "tool_name": tool_name,
                            "call_id": call_id,
                            "arguments": arguments,
                        }),
                    );

                    let tool_start = Instant::now();

                    // Get the tool from registry
                    let tool_result = if let Some(tool) = tools.get(tool_name) {
                        // Check if tool requires confirmation
                        if tool.requires_confirmation(arguments) {
                            let message = tool.confirmation_message(arguments);
                            if !confirmation_handler.confirm(tool_name, &message, arguments).await {
                                Err(ToolError::ConfirmationDenied(message))
                            } else {
                                tool.execute(arguments.clone()).await
                            }
                        } else {
                            tool.execute(arguments.clone()).await
                        }
                    } else {
                        Err(ToolError::NotFound(tool_name.to_string()))
                    };

                    let tool_duration = tool_start.elapsed();
                    total_tool_calls += 1;

                    // Build result for function response
                    let result_json = match &tool_result {
                        Ok(result) => result.content.clone(),
                        Err(e) => json!({ "error": e.to_string() }),
                    };

                    // Emit tool_result event AFTER execution
                    yield AgentUpdate::custom(
                        events::EVENT_TOOL_RESULT,
                        format!("Tool {} completed", tool_name),
                        json!({
                            "tool_name": tool_name,
                            "call_id": call_id,
                            "result": result_json,
                            "success": tool_result.is_ok(),
                            "duration_ms": tool_duration.as_millis() as u64,
                        }),
                    );

                    // Check for cancellation after each tool execution
                    if context.cancellation_token.is_cancelled() {
                        log::info!("DeveloperAgent cancelled after tool execution");
                        cancelled = true;
                        break;
                    }

                    // Add function result for the next LLM call
                    function_results.push(function_result_content(
                        tool_name.to_string(),
                        call_id.to_string(),
                        result_json,
                    ));
                }

                if cancelled {
                    yield build_incomplete_result("cancelled during tool execution", start, total_tool_calls, iteration);
                    break;
                }

                // ─────────────────────────────────────────────────────────────
                // SECTION 2b: Send Results to LLM and Check for More Calls
                // ─────────────────────────────────────────────────────────────

                // Function result returns don't need tools re-sent - the model already
                // knows about available tools from the interaction that triggered the call.
                // System instruction is also inherited from the first turn.
                // Invariant: we have a previous_interaction_id because we got here by processing
                // function calls from a prior response.
                let prev_id = previous_interaction_id
                    .as_ref()
                    .expect("invariant: must have previous_interaction_id after processing function calls");

                let follow_up_response: InteractionResponse = genai_client
                    .interaction()
                    .with_model(MODEL)
                    .with_store_enabled()
                    .with_previous_interaction(prev_id)
                    .with_content(function_results)
                    .create()
                    .await
                    .map_err(|e| AgentError::Llm(LlmError::GenAi(e)))?;

                // Check if there are more function calls
                let more_calls = follow_up_response.function_calls();

                if more_calls.is_empty() {
                    // Done - extract final answer
                    yield build_final_result(&follow_up_response, start, total_tool_calls, iteration);
                    break;
                } else {
                    // More function calls - store them for next iteration
                    previous_interaction_id = follow_up_response.id.clone();
                    pending_function_calls = more_calls.iter().map(|c| c.to_owned()).collect();
                    log::debug!(
                        "Follow-up response contained {} more function calls, processing next iteration",
                        pending_function_calls.len()
                    );
                }
            }
        })
    }

    fn create_tracker(&self) -> Box<dyn ExecutionTracking> {
        Box::new(DefaultTracker::default())
    }
}
