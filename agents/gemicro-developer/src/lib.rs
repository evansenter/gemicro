//! Developer agent with explicit function calling for real-time tool events.
//!
//! Unlike ToolAgent which delegates function calling to rust-genai's automatic
//! mode, DeveloperAgent uses an explicit FC loop to emit `tool_call_started` and
//! `tool_result` events for real-time CLI feedback.
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
//! let genai_client = rust_genai::Client::builder("key".to_string()).build();
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
use rust_genai::{function_result_content, FunctionDeclaration, InteractionContent};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;

/// Owned representation of a function call for cross-iteration processing.
///
/// `FunctionCallInfo<'a>` from rust-genai borrows from the response, so we need
/// an owned version to carry function calls across loop iterations.
struct OwnedFunctionCall {
    id: Option<String>,
    name: String,
    args: serde_json::Value,
}

impl OwnedFunctionCall {
    /// Convert a borrowed `FunctionCallInfo` to owned.
    fn from_info(info: &rust_genai::FunctionCallInfo<'_>) -> Self {
        Self {
            id: info.id.map(String::from),
            name: info.name.to_string(),
            args: info.args.clone(),
        }
    }
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
            // Emit started event
            yield AgentUpdate::custom(
                events::EVENT_DEVELOPER_STARTED,
                "Starting developer agent",
                json!({ "query": &query }),
            );

            // Get tools from context
            let tools = context.tools.clone().ok_or_else(|| {
                AgentError::InvalidConfig("DeveloperAgent requires tools in context".into())
            })?;

            // Get confirmation handler (default to AutoDeny for safety)
            let confirmation_handler: Arc<dyn ConfirmationHandler> =
                context.confirmation_handler.clone().unwrap_or_else(|| Arc::new(AutoDeny));

            // Build tool declarations for rust-genai
            let filtered_tools = tools.filter(&tool_filter);
            let function_declarations: Vec<FunctionDeclaration> = filtered_tools
                .iter()
                .map(|t| t.to_function_declaration())
                .collect();

            let genai_client = context.llm.client();

            let mut total_tool_calls = 0usize;
            let mut iteration = 0usize;
            let mut previous_interaction_id: Option<String> = None;
            // Track pending function calls from the LLM that need processing
            let mut pending_function_calls: Vec<OwnedFunctionCall> = Vec::new();

            // Explicit function calling loop
            loop {
                iteration += 1;
                if iteration > max_iterations {
                    log::warn!("DeveloperAgent hit max iterations ({})", max_iterations);
                    break;
                }

                // Determine what function calls to process this iteration
                let function_calls_to_process = if !pending_function_calls.is_empty() {
                    // Process pending function calls from previous iteration
                    std::mem::take(&mut pending_function_calls)
                } else {
                    // No pending calls - make initial LLM request
                    let mut builder = genai_client
                        .interaction()
                        .with_model(MODEL)
                        .with_system_instruction(&system_prompt)
                        .with_functions(function_declarations.clone())
                        .with_store(true);

                    // Add conversation context if we have it
                    if let Some(prev_id) = &previous_interaction_id {
                        builder = builder.with_previous_interaction(prev_id);
                    }

                    // First iteration gets the user query
                    if iteration == 1 {
                        builder = builder.with_text(&query);
                    }

                    let response = builder
                        .create()
                        .await
                        .map_err(|e| AgentError::Llm(LlmError::GenAi(e)))?;

                    let calls = response.function_calls();

                    if calls.is_empty() {
                        // No function calls - extract final answer and exit
                        let answer = response.text().map(|s| s.to_string()).unwrap_or_else(|| {
                            log::warn!("LLM response contained no text content");
                            String::new()
                        });
                        let total_tokens = gemicro_core::extract_total_tokens(&response).unwrap_or_else(|| {
                            log::debug!("Could not extract token count from response");
                            0
                        });
                        let duration_ms = start.elapsed().as_millis() as u64;

                        yield AgentUpdate::final_result(
                            json!(answer),
                            ResultMetadata::with_extra(
                                total_tokens,
                                0,
                                duration_ms,
                                json!({
                                    "tool_call_count": total_tool_calls,
                                    "iterations": iteration,
                                }),
                            ),
                        );
                        break;
                    }

                    // Store interaction ID for conversation context
                    previous_interaction_id = response.id.clone();

                    // Convert to owned for processing across iterations
                    calls.iter().map(OwnedFunctionCall::from_info).collect()
                };

                // Process each function call with real-time events
                let mut function_results: Vec<InteractionContent> = Vec::new();

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

                    // Add function result for the next LLM call
                    function_results.push(function_result_content(
                        tool_name.to_string(),
                        call_id.to_string(),
                        result_json,
                    ));
                }

                // Send function results back to the model
                let mut result_builder = genai_client
                    .interaction()
                    .with_model(MODEL)
                    .with_system_instruction(&system_prompt)
                    .with_functions(function_declarations.clone())
                    .with_store(true)
                    .with_content(function_results);

                if let Some(prev_id) = &previous_interaction_id {
                    result_builder = result_builder.with_previous_interaction(prev_id);
                }

                // Make the follow-up call with function results
                let follow_up_response = result_builder
                    .create()
                    .await
                    .map_err(|e| AgentError::Llm(LlmError::GenAi(e)))?;

                // Check if there are more function calls
                let more_calls = follow_up_response.function_calls();

                if more_calls.is_empty() {
                    // Done - extract final answer
                    let answer = follow_up_response.text().map(|s| s.to_string()).unwrap_or_else(|| {
                        log::warn!("LLM response contained no text content");
                        String::new()
                    });
                    let total_tokens = gemicro_core::extract_total_tokens(&follow_up_response).unwrap_or_else(|| {
                        log::debug!("Could not extract token count from response");
                        0
                    });
                    let duration_ms = start.elapsed().as_millis() as u64;

                    yield AgentUpdate::final_result(
                        json!(answer),
                        ResultMetadata::with_extra(
                            total_tokens,
                            0,
                            duration_ms,
                            json!({
                                "tool_call_count": total_tool_calls,
                                "iterations": iteration,
                            }),
                        ),
                    );
                    break;
                } else {
                    // More function calls - store them for next iteration
                    previous_interaction_id = follow_up_response.id.clone();
                    pending_function_calls = more_calls.iter().map(OwnedFunctionCall::from_info).collect();
                    log::debug!(
                        "Follow-up response contained {} more function calls, processing next iteration",
                        pending_function_calls.len()
                    );
                }
            }
        })
    }

    fn create_tracker(&self) -> Box<dyn ExecutionTracking> {
        // Use DefaultTracker for now - will implement custom DeveloperTracker later
        Box::new(DefaultTracker::default())
    }
}
