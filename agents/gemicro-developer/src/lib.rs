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
//! # let genai_client = rust_genai::Client::builder("your-api-key".to_string()).build().unwrap();
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
    tool::{AutoDeny, ToolError},
    Agent, AgentContext, AgentError, AgentStream, AgentUpdate, BatchApproval,
    BatchConfirmationHandler, ContextLevel, ContextUsage, DefaultTracker, ExecutionTracking,
    LlmError, PendingToolCall, ResultMetadata, ToolBatch, MODEL,
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

/// Check if context level has changed and return the new level if it has.
///
/// This is used to detect when we should emit a context_usage event.
fn check_context_level_change(
    usage: &ContextUsage,
    last_level: ContextLevel,
) -> Option<ContextLevel> {
    let current_level = usage.level();
    if current_level != last_level {
        Some(current_level)
    } else {
        None
    }
}

/// Build a context usage event if the level has changed.
///
/// Updates `last_context_level` in place and returns an event to yield if needed.
fn maybe_emit_context_usage(
    context_usage: &ContextUsage,
    last_context_level: &mut ContextLevel,
) -> Option<AgentUpdate> {
    check_context_level_change(context_usage, *last_context_level).map(|new_level| {
        *last_context_level = new_level;
        AgentUpdate::custom(
            events::EVENT_CONTEXT_USAGE,
            format!(
                "Context usage: {:.1}% ({})",
                context_usage.usage_percent(),
                new_level
            ),
            json!({
                "tokens_used": context_usage.tokens_used(),
                "context_window": context_usage.context_window(),
                "usage_percent": context_usage.usage_percent(),
                "level": new_level.to_string(),
                "remaining": context_usage.remaining(),
            }),
        )
    })
}

/// Build denial results for tool calls that were rejected.
///
/// Used when a batch is denied - we send back error results to the LLM
/// so it knows the tools weren't executed.
fn build_denial_results(
    function_calls: &[OwnedFunctionCallInfo],
    reason: &str,
) -> Vec<InteractionContent> {
    function_calls
        .iter()
        .map(|fc| {
            let call_id = fc.id.as_deref().unwrap_or("unknown");
            function_result_content(
                fc.name.clone(),
                call_id.to_string(),
                json!({ "error": reason }),
            )
        })
        .collect()
}

/// Result of batch approval with events to emit.
#[derive(Debug)]
pub(crate) struct BatchApprovalResult {
    /// Whether the batch was approved for execution.
    pub approved: bool,
    /// Whether to review each tool individually (only when approved=true).
    pub review_individually: bool,
    /// Events to emit for this approval decision.
    pub events: Vec<AgentUpdate>,
}

/// Handle batch approval and return result with events.
///
/// This encapsulates the batch confirmation logic and event generation,
/// making the main execute loop cleaner.
fn handle_batch_approval(approval: BatchApproval, batch_len: usize) -> BatchApprovalResult {
    match approval {
        BatchApproval::Approved => BatchApprovalResult {
            approved: true,
            review_individually: false,
            events: vec![AgentUpdate::custom(
                events::EVENT_BATCH_APPROVED,
                "Batch approved",
                json!({ "total": batch_len }),
            )],
        },
        BatchApproval::Denied => BatchApprovalResult {
            approved: false,
            review_individually: false,
            events: vec![AgentUpdate::custom(
                events::EVENT_BATCH_DENIED,
                "Batch denied by user",
                json!({ "total": batch_len }),
            )],
        },
        BatchApproval::ReviewIndividually => BatchApprovalResult {
            approved: true,
            review_individually: true,
            events: vec![AgentUpdate::custom(
                events::EVENT_BATCH_REVIEW_INDIVIDUALLY,
                "User chose per-tool review",
                json!({ "total": batch_len }),
            )],
        },
        // Handle future BatchApproval variants - deny as safe default
        _ => {
            log::warn!("Unknown BatchApproval variant, denying batch");
            BatchApprovalResult {
                approved: false,
                review_individually: false,
                events: vec![AgentUpdate::custom(
                    events::EVENT_BATCH_DENIED,
                    "Batch denied (unknown approval type)",
                    json!({ "total": batch_len }),
                )],
            }
        }
    }
}

/// Result of executing a single tool.
///
/// Note: This helper is primarily for testing. In the main execute loop,
/// the started event must be yielded BEFORE execution, while this helper
/// assumes execution has already completed.
#[derive(Debug)]
#[cfg(test)]
pub(crate) struct ToolExecutionResult {
    /// The function result content to send back to LLM.
    pub function_result: InteractionContent,
    /// Event emitted when tool started.
    pub started_event: AgentUpdate,
    /// Event emitted when tool completed.
    pub completed_event: AgentUpdate,
    /// Whether the tool executed successfully.
    pub success: bool,
}

/// Execute a single tool and return result with events.
///
/// Note: This helper is primarily for testing. In the main execute loop,
/// the started event must be yielded BEFORE execution (while this helper
/// assumes execution has already completed), so the timing doesn't align.
#[cfg(test)]
pub(crate) fn execute_tool_sync_result(
    tool_name: &str,
    call_id: &str,
    arguments: &serde_json::Value,
    result: Result<gemicro_core::tool::ToolResult, ToolError>,
    duration: std::time::Duration,
) -> ToolExecutionResult {
    let started_event = AgentUpdate::custom(
        events::EVENT_TOOL_CALL_STARTED,
        format!("Calling tool: {}", tool_name),
        json!({
            "tool_name": tool_name,
            "call_id": call_id,
            "arguments": arguments,
        }),
    );

    let (result_json, success) = match &result {
        Ok(r) => (r.content.clone(), true),
        Err(e) => (json!({ "error": e.to_string() }), false),
    };

    let completed_event = AgentUpdate::custom(
        events::EVENT_TOOL_RESULT,
        format!("Tool {} completed", tool_name),
        json!({
            "tool_name": tool_name,
            "call_id": call_id,
            "result": result_json,
            "success": success,
            "duration_ms": duration.as_millis() as u64,
        }),
    );

    let function_result =
        function_result_content(tool_name.to_string(), call_id.to_string(), result_json);

    ToolExecutionResult {
        function_result,
        started_event,
        completed_event,
        success,
    }
}

/// Result of building subagent lifecycle events.
#[cfg(test)]
pub(crate) struct SubagentEvents {
    pub started_event: AgentUpdate,
    pub completed_event: AgentUpdate,
}

/// Builds subagent lifecycle events for a task tool call.
///
/// Returns `Some(SubagentEvents)` if the tool is "task", `None` otherwise.
/// This mirrors the event emission logic in the execute() loop but is
/// testable in isolation.
#[cfg(test)]
pub(crate) fn build_subagent_events(
    tool_name: &str,
    call_id: &str,
    arguments: &serde_json::Value,
    success: bool,
    duration_ms: u64,
    result_preview: Option<&str>,
) -> Option<SubagentEvents> {
    if tool_name != "task" {
        return None;
    }

    let agent_name = arguments
        .get("agent")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let query = arguments
        .get("query")
        .and_then(|v| v.as_str())
        .map(|s| gemicro_core::truncate(s, 80));

    let started_event = AgentUpdate::custom(
        events::EVENT_SUBAGENT_STARTED,
        format!("Spawning subagent: {}", agent_name),
        json!({
            "agent": agent_name,
            "query_preview": query,
            "call_id": call_id,
        }),
    );

    let completed_event = AgentUpdate::custom(
        events::EVENT_SUBAGENT_COMPLETED,
        format!("Subagent {} completed", agent_name),
        json!({
            "agent": agent_name,
            "call_id": call_id,
            "success": success,
            "duration_ms": duration_ms,
            "result_preview": result_preview.map(|s| gemicro_core::truncate(s, 100)),
        }),
    );

    Some(SubagentEvents {
        started_event,
        completed_event,
    })
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
        let approval_batching = self.config.approval_batching;

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

            let confirmation_handler: Arc<dyn BatchConfirmationHandler> =
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
            let mut context_usage = ContextUsage::new();
            let mut last_context_level = ContextLevel::Normal;

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

                    // Track context usage
                    if let Some(tokens) = gemicro_core::extract_total_tokens(&response) {
                        context_usage.add_tokens(tokens);
                        if let Some(event) = maybe_emit_context_usage(&context_usage, &mut last_context_level) {
                            yield event;
                        }
                    }

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
                // SECTION 2a: Build Batch and Handle Approval
                // ─────────────────────────────────────────────────────────────

                // Build a batch from all pending function calls
                let mut batch = ToolBatch::new();
                for fc in &function_calls_to_process {
                    let call_id = fc.id.clone().unwrap_or_else(|| "unknown".to_string());
                    let pending = PendingToolCall::new(call_id, &fc.name, fc.args.clone())
                        .with_tool_info(&tools);
                    batch.push(pending);
                }

                // Handle batch approval if enabled and required
                // review_individually tracks when user wants per-tool confirmation
                let (batch_approved, review_individually) = if approval_batching && batch.requires_confirmation() {
                    // Emit batch plan event for CLI display
                    let summary = batch.summary();
                    yield AgentUpdate::custom(
                        events::EVENT_BATCH_PLAN,
                        format!("Batch plan: {}", summary),
                        json!({
                            "total": summary.total(),
                            "requires_confirmation": summary.requires_confirmation(),
                            "tool_counts": summary.tool_counts(),
                            "calls": batch.iter().map(|c| json!({
                                "call_id": c.call_id(),
                                "tool_name": c.tool_name(),
                                "requires_confirmation": c.requires_confirmation(),
                                "confirmation_message": c.confirmation_message(),
                            })).collect::<Vec<_>>(),
                        }),
                    );

                    // Get batch confirmation and handle result
                    let approval = confirmation_handler.confirm_batch(&batch).await;
                    let batch_result = handle_batch_approval(approval, batch.len());
                    for event in batch_result.events {
                        yield event;
                    }
                    (batch_result.approved, batch_result.review_individually)
                } else {
                    // No batch confirmation needed
                    (true, false)
                };

                if !batch_approved {
                    // User denied the batch - send denial results back to LLM
                    let denial_results =
                        build_denial_results(&function_calls_to_process, "User denied batch execution");

                    // Send denial back to LLM and continue loop
                    // Type-state pattern: with_previous_interaction changes builder type
                    let denial_response: InteractionResponse = match &previous_interaction_id {
                        Some(prev_id) => {
                            genai_client
                                .interaction()
                                .with_model(MODEL)
                                .with_functions(function_declarations.clone())
                                .with_store_enabled()
                                .with_previous_interaction(prev_id)
                                .with_content(denial_results)
                                .create()
                                .await
                                .map_err(|e| AgentError::Llm(LlmError::GenAi(e)))?
                        }
                        None => {
                            genai_client
                                .interaction()
                                .with_model(MODEL)
                                .with_system_instruction(&system_prompt)
                                .with_functions(function_declarations.clone())
                                .with_store_enabled()
                                .with_content(denial_results)
                                .create()
                                .await
                                .map_err(|e| AgentError::Llm(LlmError::GenAi(e)))?
                        }
                    };

                    // Track context usage
                    if let Some(tokens) = gemicro_core::extract_total_tokens(&denial_response) {
                        context_usage.add_tokens(tokens);
                        if let Some(event) = maybe_emit_context_usage(&context_usage, &mut last_context_level) {
                            yield event;
                        }
                    }

                    let more_calls = denial_response.function_calls();
                    if more_calls.is_empty() {
                        yield build_final_result(&denial_response, start, total_tool_calls, iteration);
                        break;
                    } else {
                        previous_interaction_id = denial_response.id.clone();
                        pending_function_calls = more_calls.iter().map(|c| c.to_owned()).collect();
                        continue;
                    }
                }

                // ─────────────────────────────────────────────────────────────
                // SECTION 2b: Execute Function Calls with Real-Time Events
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

                    // For task tool, emit additional subagent_started event
                    let is_subagent_call = tool_name == "task";
                    let subagent_name = if is_subagent_call {
                        let agent_name = arguments.get("agent")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");
                        let query = arguments.get("query")
                            .and_then(|v| v.as_str())
                            .map(|s| gemicro_core::truncate(s, 80));
                        yield AgentUpdate::custom(
                            events::EVENT_SUBAGENT_STARTED,
                            format!("Spawning subagent: {}", agent_name),
                            json!({
                                "agent": agent_name,
                                "query_preview": query,
                                "call_id": call_id,
                            }),
                        );
                        Some(agent_name.to_string())
                    } else {
                        None
                    };

                    let tool_start = Instant::now();

                    // Get the tool from registry
                    // Individual confirmation needed when:
                    // - approval_batching is disabled (original behavior), OR
                    // - user chose ReviewIndividually for the batch
                    let tool_result = if let Some(tool) = tools.get(tool_name) {
                        let needs_individual_confirm = (!approval_batching || review_individually)
                            && tool.requires_confirmation(arguments);

                        if needs_individual_confirm {
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

                    // For task tool, emit additional subagent_completed event
                    if let Some(agent_name) = &subagent_name {
                        let result_preview = if tool_result.is_ok() {
                            result_json.as_str()
                                .map(|s| gemicro_core::truncate(s, 100))
                        } else {
                            None
                        };
                        yield AgentUpdate::custom(
                            events::EVENT_SUBAGENT_COMPLETED,
                            format!("Subagent {} completed", agent_name),
                            json!({
                                "agent": agent_name,
                                "call_id": call_id,
                                "success": tool_result.is_ok(),
                                "duration_ms": tool_duration.as_millis() as u64,
                                "result_preview": result_preview,
                            }),
                        );
                    }

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
                // SECTION 2c: Send Results to LLM and Check for More Calls
                // ─────────────────────────────────────────────────────────────

                // Function result returns don't need tools re-sent - the model already
                // knows about available tools from the interaction that triggered the call.
                // System instruction is also inherited from the first turn.
                // Invariant: we have a previous_interaction_id because we got here by processing
                // function calls from a prior response. If this is ever None, it's a bug.
                let prev_id = previous_interaction_id.as_ref().ok_or_else(|| {
                    AgentError::Other(
                        "previous_interaction_id was None after processing function calls".into(),
                    )
                })?;

                let follow_up_response: InteractionResponse = genai_client
                    .interaction()
                    .with_model(MODEL)
                    .with_store_enabled()
                    .with_previous_interaction(prev_id)
                    .with_content(function_results)
                    .create()
                    .await
                    .map_err(|e| AgentError::Llm(LlmError::GenAi(e)))?;

                // Track context usage
                if let Some(tokens) = gemicro_core::extract_total_tokens(&follow_up_response) {
                    context_usage.add_tokens(tokens);
                    if let Some(event) = maybe_emit_context_usage(&context_usage, &mut last_context_level) {
                        yield event;
                    }
                }

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

#[cfg(test)]
mod tests {
    use super::*;
    use gemicro_core::tool::ToolResult;
    use serde_json::json;
    use std::time::Duration;

    // =========================================================================
    // Tests for build_denial_results
    // =========================================================================

    #[test]
    fn test_build_denial_results_single_call() {
        let calls = vec![OwnedFunctionCallInfo {
            name: "bash".to_string(),
            id: Some("call_123".to_string()),
            args: json!({"command": "rm -rf /"}),
            thought_signature: None,
        }];

        let results = build_denial_results(&calls, "User denied execution");

        assert_eq!(results.len(), 1);
        // Verify it's a FunctionResult content type
        if let InteractionContent::FunctionResult {
            name,
            call_id,
            result,
        } = &results[0]
        {
            assert_eq!(name, "bash");
            assert_eq!(call_id, "call_123");
            assert_eq!(result, &json!({"error": "User denied execution"}));
        } else {
            panic!("Expected FunctionResult content");
        }
    }

    #[test]
    fn test_build_denial_results_multiple_calls() {
        let calls = vec![
            OwnedFunctionCallInfo {
                name: "file_write".to_string(),
                id: Some("call_1".to_string()),
                args: json!({"path": "/etc/passwd"}),
                thought_signature: None,
            },
            OwnedFunctionCallInfo {
                name: "bash".to_string(),
                id: Some("call_2".to_string()),
                args: json!({"command": "echo test"}),
                thought_signature: None,
            },
        ];

        let results = build_denial_results(&calls, "Batch denied");

        assert_eq!(results.len(), 2);
        // Check both are FunctionResult
        for (i, result) in results.iter().enumerate() {
            if let InteractionContent::FunctionResult {
                name: _,
                call_id,
                result,
            } = result
            {
                assert_eq!(result, &json!({"error": "Batch denied"}));
                assert_eq!(call_id, &format!("call_{}", i + 1));
            } else {
                panic!("Expected FunctionResult content at index {}", i);
            }
        }
    }

    #[test]
    fn test_build_denial_results_missing_id() {
        let calls = vec![OwnedFunctionCallInfo {
            name: "glob".to_string(),
            id: None, // No ID provided
            args: json!({"pattern": "*.rs"}),
            thought_signature: None,
        }];

        let results = build_denial_results(&calls, "Denied");

        assert_eq!(results.len(), 1);
        if let InteractionContent::FunctionResult {
            name: _,
            call_id,
            result: _,
        } = &results[0]
        {
            assert_eq!(call_id, "unknown"); // Falls back to "unknown"
        } else {
            panic!("Expected FunctionResult content");
        }
    }

    // =========================================================================
    // Tests for handle_batch_approval
    // =========================================================================

    #[test]
    fn test_handle_batch_approval_approved() {
        let result = handle_batch_approval(BatchApproval::Approved, 5);

        assert!(result.approved);
        assert!(!result.review_individually);
        assert_eq!(result.events.len(), 1);
        assert_eq!(result.events[0].event_type, events::EVENT_BATCH_APPROVED);
        assert_eq!(result.events[0].data["total"], 5);
    }

    #[test]
    fn test_handle_batch_approval_denied() {
        let result = handle_batch_approval(BatchApproval::Denied, 3);

        assert!(!result.approved);
        assert!(!result.review_individually);
        assert_eq!(result.events.len(), 1);
        assert_eq!(result.events[0].event_type, events::EVENT_BATCH_DENIED);
        assert_eq!(result.events[0].data["total"], 3);
    }

    #[test]
    fn test_handle_batch_approval_review_individually() {
        let result = handle_batch_approval(BatchApproval::ReviewIndividually, 7);

        assert!(result.approved); // Execution proceeds, but one-by-one
        assert!(result.review_individually);
        assert_eq!(result.events.len(), 1);
        assert_eq!(
            result.events[0].event_type,
            events::EVENT_BATCH_REVIEW_INDIVIDUALLY
        );
        assert_eq!(result.events[0].data["total"], 7);
    }

    // =========================================================================
    // Tests for execute_tool_sync_result
    // =========================================================================

    #[test]
    fn test_execute_tool_sync_result_success() {
        let tool_result = ToolResult::json(json!({"files": ["a.rs", "b.rs"]}));

        let result = execute_tool_sync_result(
            "glob",
            "call_abc",
            &json!({"pattern": "*.rs"}),
            Ok(tool_result),
            Duration::from_millis(42),
        );

        assert!(result.success);

        // Check started event
        assert_eq!(
            result.started_event.event_type,
            events::EVENT_TOOL_CALL_STARTED
        );
        assert_eq!(result.started_event.data["tool_name"], "glob");
        assert_eq!(result.started_event.data["call_id"], "call_abc");

        // Check completed event
        assert_eq!(result.completed_event.event_type, events::EVENT_TOOL_RESULT);
        assert_eq!(result.completed_event.data["success"], true);
        assert_eq!(result.completed_event.data["duration_ms"], 42);

        // Check function result
        if let InteractionContent::FunctionResult {
            name,
            call_id,
            result,
        } = &result.function_result
        {
            assert_eq!(name, "glob");
            assert_eq!(call_id, "call_abc");
            assert_eq!(result, &json!({"files": ["a.rs", "b.rs"]}));
        } else {
            panic!("Expected FunctionResult content");
        }
    }

    #[test]
    fn test_execute_tool_sync_result_error() {
        let error = ToolError::ExecutionFailed("Command timed out".to_string());

        let result = execute_tool_sync_result(
            "bash",
            "call_xyz",
            &json!({"command": "sleep 1000"}),
            Err(error),
            Duration::from_millis(5000),
        );

        assert!(!result.success);

        // Check completed event shows failure
        assert_eq!(result.completed_event.data["success"], false);

        // Check function result contains error
        if let InteractionContent::FunctionResult {
            name,
            call_id: _,
            result,
        } = &result.function_result
        {
            assert_eq!(name, "bash");
            // Error message should be in the content
            let error_str = result["error"].as_str().unwrap();
            assert!(error_str.contains("Command timed out"));
        } else {
            panic!("Expected FunctionResult content");
        }
    }

    #[test]
    fn test_execute_tool_sync_result_confirmation_denied() {
        let error = ToolError::ConfirmationDenied("file_write".to_string());

        let result = execute_tool_sync_result(
            "file_write",
            "call_denied",
            &json!({"path": "/etc/passwd"}),
            Err(error),
            Duration::from_millis(0),
        );

        assert!(!result.success);

        // Error should indicate denial
        if let InteractionContent::FunctionResult {
            name: _,
            call_id: _,
            result,
        } = &result.function_result
        {
            let error_str = result["error"].as_str().unwrap();
            assert!(error_str.contains("file_write"));
        } else {
            panic!("Expected FunctionResult content");
        }
    }

    // =========================================================================
    // Tests for DeveloperConfig
    // =========================================================================

    #[test]
    fn test_developer_config_default() {
        let config = DeveloperConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.max_iterations, 50);
        assert!(config.approval_batching);
    }

    #[test]
    fn test_developer_config_zero_iterations_invalid() {
        let config = DeveloperConfig {
            max_iterations: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_developer_config_builder() {
        let config = DeveloperConfig::default()
            .with_max_iterations(100)
            .with_approval_batching(false);
        assert_eq!(config.max_iterations, 100);
        assert!(!config.approval_batching);
    }

    // =========================================================================
    // Tests for build_subagent_events
    // =========================================================================

    #[test]
    fn test_subagent_events_task_tool() {
        let arguments = json!({
            "agent": "deep_research",
            "query": "What are the tradeoffs between async runtimes?"
        });

        let result = build_subagent_events(
            "task",
            "call_123",
            &arguments,
            true,
            5000,
            Some("Async runtimes comparison..."),
        );

        assert!(result.is_some());
        let events = result.unwrap();

        // Check started event
        assert_eq!(
            events.started_event.event_type,
            events::EVENT_SUBAGENT_STARTED
        );
        assert_eq!(events.started_event.data["agent"], "deep_research");
        assert_eq!(events.started_event.data["call_id"], "call_123");
        assert!(events.started_event.data["query_preview"]
            .as_str()
            .unwrap()
            .contains("async"));

        // Check completed event
        assert_eq!(
            events.completed_event.event_type,
            events::EVENT_SUBAGENT_COMPLETED
        );
        assert_eq!(events.completed_event.data["agent"], "deep_research");
        assert_eq!(events.completed_event.data["success"], true);
        assert_eq!(events.completed_event.data["duration_ms"], 5000);
        assert!(events.completed_event.data["result_preview"]
            .as_str()
            .unwrap()
            .contains("Async"));
    }

    #[test]
    fn test_subagent_events_non_task_tool_returns_none() {
        let arguments = json!({"pattern": "*.rs"});

        let result = build_subagent_events("glob", "call_456", &arguments, true, 100, None);

        assert!(result.is_none());
    }

    #[test]
    fn test_subagent_events_missing_agent_name() {
        let arguments = json!({"query": "some query"}); // Missing "agent" field

        let result = build_subagent_events("task", "call_789", &arguments, true, 1000, None);

        assert!(result.is_some());
        let events = result.unwrap();
        assert_eq!(events.started_event.data["agent"], "unknown");
        assert_eq!(events.completed_event.data["agent"], "unknown");
    }

    #[test]
    fn test_subagent_events_failure() {
        let arguments = json!({
            "agent": "tool_agent",
            "query": "Run failing task"
        });

        let result = build_subagent_events("task", "call_fail", &arguments, false, 250, None);

        assert!(result.is_some());
        let events = result.unwrap();
        assert_eq!(events.completed_event.data["success"], false);
        assert!(events.completed_event.data["result_preview"].is_null());
    }

    #[test]
    fn test_subagent_events_long_query_truncated() {
        let long_query = "a".repeat(200); // Longer than 80 char limit
        let arguments = json!({
            "agent": "deep_research",
            "query": long_query
        });

        let result = build_subagent_events("task", "call_long", &arguments, true, 100, None);

        assert!(result.is_some());
        let events = result.unwrap();
        let query_preview = events.started_event.data["query_preview"].as_str().unwrap();
        assert!(query_preview.len() <= 83); // 80 + "..."
        assert!(query_preview.ends_with("..."));
    }
}
