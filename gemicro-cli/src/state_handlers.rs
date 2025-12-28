//! Agent-specific state handlers for translating events to ExecutionState.
//!
//! These handlers bridge agent-specific events (from agent crates) with
//! the generic ExecutionState (from gemicro-runner). They live in gemicro-cli
//! because CLI is the only consumer that needs this integration.

use gemicro_core::AgentUpdate;
use gemicro_deep_research::DeepResearchEventExt;
use gemicro_runner::{phases, ExecutionState, ExecutionStep, FinalResultData, StateHandler};

/// First sentence extraction for previews.
fn first_sentence(text: &str) -> String {
    text.split_once(['.', '!', '?'])
        .map(|(first, _)| format!("{}.", first.trim()))
        .unwrap_or_else(|| text.to_string())
}

/// Handler for DeepResearch agent events.
///
/// Translates DeepResearch-specific events (decomposition_started, sub_query_completed, etc.)
/// into generic ExecutionState updates.
pub struct DeepResearchStateHandler;

impl StateHandler for DeepResearchStateHandler {
    fn handle(&self, state: &mut ExecutionState, event: &AgentUpdate) -> Option<String> {
        match event.event_type.as_str() {
            "decomposition_started" => {
                state.set_phase(phases::DECOMPOSING);
                None
            }

            "decomposition_complete" => {
                if let Some(queries) = event.as_decomposition_complete() {
                    let steps: Vec<ExecutionStep> = queries
                        .into_iter()
                        .enumerate()
                        .map(|(id, query)| ExecutionStep::new(id.to_string(), query))
                        .collect();
                    state.add_steps(steps);
                    state.set_phase(phases::EXECUTING);
                } else {
                    log::warn!(
                        "Received decomposition_complete event with malformed data: {:?}",
                        event.data
                    );
                }
                None
            }

            "sub_query_started" => {
                if let Some(id) = event.data.get("id").and_then(|v| v.as_u64()) {
                    let id_str = id.to_string();
                    if let Some(step) = state.step_by_index_mut(id as usize) {
                        step.start();
                        return Some(id_str);
                    }
                }
                None
            }

            "sub_query_completed" => {
                if let Some(result) = event.as_sub_query_completed() {
                    let id_str = result.id.to_string();
                    if let Some(step) = state.step_by_index_mut(result.id) {
                        let preview = first_sentence(&result.result);
                        step.complete(preview, Some(result.tokens_used));
                        return Some(id_str);
                    }
                } else {
                    log::warn!(
                        "Received sub_query_completed event with malformed data: {:?}",
                        event.data
                    );
                }
                None
            }

            "sub_query_failed" => {
                if let Some(id) = event.data.get("id").and_then(|v| v.as_u64()) {
                    let id_str = id.to_string();
                    if let Some(step) = state.step_by_index_mut(id as usize) {
                        let error = event
                            .data
                            .get("error")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Unknown error")
                            .to_string();
                        step.fail(error);
                        return Some(id_str);
                    }
                }
                None
            }

            "synthesis_started" => {
                state.set_phase(phases::SYNTHESIZING);
                None
            }

            "final_result" => {
                if let Some(result) = event.as_final_result() {
                    // Extract step counts from the extensible extra field
                    let steps_succeeded = result.metadata.extra["steps_succeeded"]
                        .as_u64()
                        .unwrap_or(0) as usize;
                    let steps_failed =
                        result.metadata.extra["steps_failed"].as_u64().unwrap_or(0) as usize;
                    state.set_final_result(FinalResultData {
                        answer: result.answer,
                        total_tokens: result.metadata.total_tokens,
                        tokens_unavailable_count: result.metadata.tokens_unavailable_count,
                        steps_succeeded,
                        steps_failed,
                    });
                } else {
                    log::warn!(
                        "Received final_result event with malformed data: {:?}",
                        event.data
                    );
                }
                None
            }

            _ => {
                log::debug!(
                    "Unknown event type in DeepResearchStateHandler: {}",
                    event.event_type
                );
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gemicro_core::ResultMetadata;
    use serde_json::json;

    #[test]
    fn test_deep_research_handler_decomposition() {
        let handler = DeepResearchStateHandler;
        let mut state = ExecutionState::new();

        let event = AgentUpdate::custom("decomposition_started", "Decomposing query", json!({}));
        handler.handle(&mut state, &event);
        assert_eq!(state.phase(), phases::DECOMPOSING);

        let event = AgentUpdate::custom(
            "decomposition_complete",
            "Decomposed into 3 sub-queries",
            json!({ "sub_queries": ["Query 1", "Query 2", "Query 3"] }),
        );
        handler.handle(&mut state, &event);

        assert_eq!(state.phase(), phases::EXECUTING);
        assert_eq!(state.steps().len(), 3);
        assert_eq!(state.step("0").unwrap().label, "Query 1");
    }

    #[test]
    fn test_deep_research_handler_sub_query_flow() {
        let handler = DeepResearchStateHandler;
        let mut state = ExecutionState::new();

        // Setup
        handler.handle(
            &mut state,
            &AgentUpdate::custom(
                "decomposition_complete",
                "Decomposed",
                json!({ "sub_queries": ["Q1"] }),
            ),
        );

        // Start
        let event = AgentUpdate::custom(
            "sub_query_started",
            "Sub-query 0 started",
            json!({ "id": 0, "query": "Q1" }),
        );
        let updated = handler.handle(&mut state, &event);
        assert_eq!(updated, Some("0".to_string()));

        // Complete
        let event = AgentUpdate::custom(
            "sub_query_completed",
            "Sub-query 0 completed",
            json!({ "id": 0, "result": "This is the result.", "tokens_used": 42 }),
        );
        let updated = handler.handle(&mut state, &event);
        assert_eq!(updated, Some("0".to_string()));
    }

    #[test]
    fn test_deep_research_handler_final_result() {
        let handler = DeepResearchStateHandler;
        let mut state = ExecutionState::new();

        let metadata = ResultMetadata::with_extra(
            100,
            0,
            5000,
            json!({
                "steps_succeeded": 3,
                "steps_failed": 1,
            }),
        );
        let event = AgentUpdate::final_result("Final answer".to_string(), metadata);
        handler.handle(&mut state, &event);

        assert_eq!(state.phase(), phases::COMPLETE);
        let result = state.final_result().unwrap();
        assert_eq!(result.answer, "Final answer");
        assert_eq!(result.total_tokens, 100);
        assert_eq!(result.steps_succeeded, 3);
    }

    #[test]
    fn test_unknown_event_logged() {
        let handler = DeepResearchStateHandler;
        let mut state = ExecutionState::new();

        let event = AgentUpdate::custom("unknown_event", "Unknown", json!({}));
        let result = handler.handle(&mut state, &event);

        assert!(result.is_none());
        assert_eq!(state.phase(), phases::NOT_STARTED);
    }
}
