//! Headless agent execution runner.
//!
//! Provides `AgentRunner` for executing agents without terminal dependencies,
//! returning structured `ExecutionMetrics` for programmatic consumption.

use crate::metrics::ExecutionMetrics;
use futures_util::StreamExt;
use gemicro_core::{
    enforce_final_result_contract, Agent, AgentContext, AgentError, AgentUpdate, Coordination,
    ExecutionTracking, ExternalEvent, LlmClient, LlmConfig, Trajectory, TrajectoryBuilder,
};
use serde_json::json;
use std::time::Instant;

/// Headless agent runner for programmatic execution.
///
/// Use this for:
/// - Executing agents without terminal rendering
/// - Building custom display backends
/// - Testing agent behavior
/// - Batch evaluation/benchmarking
///
/// # Example
///
/// ```text
/// use gemicro_runner::AgentRunner;
/// use gemicro_core::{AgentContext, LlmClient, LlmConfig};
/// // Assuming you have an agent that implements the Agent trait
///
/// async fn run_agent(agent: &dyn Agent, context: AgentContext) {
///     let runner = AgentRunner::new();
///
///     let result = runner.execute_with_tracking(
///         agent,
///         "What is Rust?",
///         context,
///         |_tracker, status| println!("Status: {}", status),
///     ).await;
///
///     match result {
///         Ok(metrics) => {
///             println!("Completed in {:?}", metrics.total_duration);
///             println!("Tokens used: {}", metrics.total_tokens);
///         }
///         Err((error, partial_metrics)) => {
///             println!("Failed: {}", error);
///             println!("Partial tokens used: {}", partial_metrics.total_tokens);
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AgentRunner;

impl AgentRunner {
    /// Create a new headless runner.
    pub fn new() -> Self {
        Self
    }

    /// Execute an agent and return final metrics.
    ///
    /// This is the preferred method for new code. It uses the agent's
    /// `create_tracker()` method instead of requiring external state handlers.
    ///
    /// # Returns
    ///
    /// - `Ok(metrics)` - Execution completed successfully with final metrics
    /// - `Err((error, partial_metrics))` - Execution failed, but partial metrics
    ///   are still returned. This allows callers to log duration, token usage,
    ///   and other stats even when the agent fails mid-execution.
    ///
    /// # Arguments
    ///
    /// * `agent` - The agent to execute
    /// * `query` - The user's query
    /// * `context` - Agent context with LLM client
    /// * `on_status` - Callback receiving status message updates
    ///
    /// # Example
    ///
    /// ```text
    /// use gemicro_runner::AgentRunner;
    /// use gemicro_core::{Agent, AgentContext};
    ///
    /// async fn example(agent: &dyn Agent, context: AgentContext) {
    ///     let runner = AgentRunner::new();
    ///
    ///     match runner.execute_with_tracking(
    ///         agent,
    ///         "query",
    ///         context,
    ///         |_tracker, status| println!("Status: {}", status),
    ///     ).await {
    ///         Ok(metrics) => println!("Success: {:?}", metrics.total_duration),
    ///         Err((error, partial)) => {
    ///             println!("Failed after {:?}: {}", partial.total_duration, error);
    ///         }
    ///     }
    /// }
    /// ```
    pub async fn execute_with_tracking<F>(
        &self,
        agent: &dyn Agent,
        query: &str,
        context: AgentContext,
        mut on_status: F,
    ) -> Result<ExecutionMetrics, (AgentError, ExecutionMetrics)>
    where
        F: FnMut(&dyn ExecutionTracking, &str),
    {
        let mut tracker = agent.create_tracker();
        let stream = agent.execute(query, context);
        // Wrap stream with contract enforcement to detect violations
        let stream = enforce_final_result_contract(stream);
        futures_util::pin_mut!(stream);
        let start = Instant::now();

        while let Some(result) = stream.next().await {
            match result {
                Ok(update) => {
                    tracker.handle_event(&update);
                    if let Some(msg) = tracker.status_message() {
                        on_status(tracker.as_ref(), msg);
                    }
                }
                Err(e) => {
                    // Return partial metrics along with the error
                    let partial_metrics =
                        ExecutionMetrics::from_tracker(tracker.as_ref(), start.elapsed());
                    return Err((e, partial_metrics));
                }
            }
        }

        Ok(ExecutionMetrics::from_tracker(
            tracker.as_ref(),
            start.elapsed(),
        ))
    }

    /// Execute an agent with optional event coordination.
    ///
    /// When `coordination` is provided, this method uses `tokio::select!` to
    /// interleave external events from the event bus with agent updates. External
    /// events are converted to `AgentUpdate` instances with `event_type = "external_event"`.
    ///
    /// # Returns
    ///
    /// - `Ok(metrics)` - Execution completed successfully with final metrics
    /// - `Err((error, partial_metrics))` - Execution failed, but partial metrics
    ///   are still returned for logging and debugging purposes.
    ///
    /// # Arguments
    ///
    /// * `agent` - The agent to execute
    /// * `query` - The user's query
    /// * `context` - Agent context with LLM client
    /// * `coordination` - Optional event bus coordination for real-time events
    /// * `on_update` - Callback receiving all updates (agent + external)
    ///
    /// # Event Handling
    ///
    /// External events are wrapped as `AgentUpdate::custom("external_event", ...)` with
    /// the full event data in the JSON payload. The callback receives both agent events
    /// and external events in interleaved order.
    ///
    /// # Example
    ///
    /// ```text
    /// use gemicro_runner::AgentRunner;
    /// use gemicro_core::{Agent, AgentContext, HubCoordination};
    ///
    /// async fn example(agent: &dyn Agent, context: AgentContext) {
    ///     let runner = AgentRunner::new();
    ///
    ///     // Connect to event bus (optional)
    ///     let coord = HubCoordination::connect("http://localhost:8765", "my-session").await.ok();
    ///     let coord_boxed = coord.map(|c| Box::new(c) as Box<dyn Coordination>);
    ///
    ///     match runner.execute_with_events(
    ///         agent,
    ///         "query",
    ///         context,
    ///         coord_boxed,
    ///         |update| {
    ///             match update.event_type.as_str() {
    ///                 "external_event" => println!("External: {}", update.message),
    ///                 _ => println!("Agent: {}", update.message),
    ///             }
    ///         },
    ///     ).await {
    ///         Ok(metrics) => println!("Completed in {:?}", metrics.total_duration),
    ///         Err((error, partial)) => {
    ///             println!("Failed after {:?}: {}", partial.total_duration, error);
    ///         }
    ///     }
    /// }
    /// ```
    pub async fn execute_with_events<F>(
        &self,
        agent: &dyn Agent,
        query: &str,
        context: AgentContext,
        coordination: Option<Box<dyn Coordination>>,
        mut on_update: F,
    ) -> Result<ExecutionMetrics, (AgentError, ExecutionMetrics)>
    where
        F: FnMut(&AgentUpdate),
    {
        let mut tracker = agent.create_tracker();
        let stream = agent.execute(query, context);
        let stream = enforce_final_result_contract(stream);
        futures_util::pin_mut!(stream);
        let start = Instant::now();

        // If no coordination, fall back to simple stream consumption
        let Some(mut coord) = coordination else {
            while let Some(result) = stream.next().await {
                match result {
                    Ok(update) => {
                        tracker.handle_event(&update);
                        on_update(&update);
                    }
                    Err(e) => {
                        let partial_metrics =
                            ExecutionMetrics::from_tracker(tracker.as_ref(), start.elapsed());
                        return Err((e, partial_metrics));
                    }
                }
            }
            return Ok(ExecutionMetrics::from_tracker(
                tracker.as_ref(),
                start.elapsed(),
            ));
        };

        // With coordination, use select! to interleave events
        loop {
            tokio::select! {
                // Agent stream has priority (biased) to avoid starving agent events
                biased;

                result = stream.next() => {
                    match result {
                        Some(Ok(update)) => {
                            tracker.handle_event(&update);
                            on_update(&update);
                            // Check if agent completed
                            if tracker.is_complete() {
                                break;
                            }
                        }
                        Some(Err(e)) => {
                            let partial_metrics =
                                ExecutionMetrics::from_tracker(tracker.as_ref(), start.elapsed());
                            return Err((e, partial_metrics));
                        }
                        None => break, // Stream exhausted
                    }
                }

                event = coord.recv_event() => {
                    if let Some(external_event) = event {
                        // Convert external event to AgentUpdate
                        let update = external_event_to_update(&external_event);
                        // Don't pass to tracker (external events don't affect agent state)
                        on_update(&update);
                    }
                    // If recv_event returns None, coordination is closed - continue with agent only
                }
            }
        }

        Ok(ExecutionMetrics::from_tracker(
            tracker.as_ref(),
            start.elapsed(),
        ))
    }

    /// Execute an agent and capture a full trajectory for offline replay.
    ///
    /// This method:
    /// - Creates a recording LLM client from the provided rust-genai client
    /// - Executes the agent while capturing all LLM interactions
    /// - Returns both execution metrics and a complete trajectory
    ///
    /// The trajectory can be saved for later replay or analysis.
    ///
    /// # Arguments
    ///
    /// * `agent` - The agent to execute
    /// * `query` - The user's query
    /// * `agent_config` - Agent configuration as JSON (for trajectory metadata)
    /// * `genai_client` - The rust-genai client to use
    /// * `llm_config` - LLM configuration
    ///
    /// # Returns
    ///
    /// - `Ok((metrics, trajectory))` - Execution completed successfully
    /// - `Err((error, partial_metrics, partial_trajectory))` - Execution failed, but partial
    ///   data is still returned for debugging and analysis.
    ///
    /// # Example
    ///
    /// ```text
    /// use gemicro_runner::AgentRunner;
    /// use gemicro_core::LlmConfig;
    /// use serde_json::json;
    ///
    /// async fn example(agent: &dyn Agent) -> Result<(), Box<dyn std::error::Error>> {
    ///     let runner = AgentRunner::new();
    ///     let genai_client = genai_rs::Client::builder("key".to_string()).build();
    ///
    ///     let (metrics, trajectory) = runner.execute_with_trajectory(
    ///         agent,
    ///         "What is Rust?",
    ///         json!({}),
    ///         genai_client,
    ///         LlmConfig::default(),
    ///     ).await?;
    ///
    ///     // Save for later replay
    ///     trajectory.save("runs/run_001.json")?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn execute_with_trajectory(
        &self,
        agent: &dyn Agent,
        query: &str,
        agent_config: serde_json::Value,
        genai_client: genai_rs::Client,
        llm_config: LlmConfig,
    ) -> Result<(ExecutionMetrics, Trajectory), (AgentError, ExecutionMetrics, Trajectory)> {
        // Create recording LLM client
        let llm = LlmClient::with_recording(genai_client, llm_config.clone());
        let context = AgentContext::new(llm);

        // Get a reference to the LLM client for extracting steps later
        let llm_ref = context.llm.clone();

        // Execute agent and collect events
        let mut tracker = agent.create_tracker();
        let stream = agent.execute(query, context);
        let stream = enforce_final_result_contract(stream);
        futures_util::pin_mut!(stream);

        let start = Instant::now();
        let mut events: Vec<AgentUpdate> = Vec::new();

        // Helper to build trajectory from current state
        let build_trajectory = |events: Vec<AgentUpdate>,
                                llm_ref: &std::sync::Arc<LlmClient>,
                                query: &str,
                                agent: &dyn Agent,
                                agent_config: &serde_json::Value,
                                duration_ms: u64,
                                final_answer: Option<String>| {
            let steps = llm_ref.take_steps();
            TrajectoryBuilder::default()
                .query(query)
                .agent_name(agent.name())
                .agent_config(agent_config.clone())
                .model(gemicro_core::MODEL)
                .build(
                    steps,
                    events,
                    duration_ms,
                    final_answer.map(serde_json::Value::String),
                )
        };

        while let Some(result) = stream.next().await {
            match result {
                Ok(update) => {
                    events.push(update.clone());
                    tracker.handle_event(&update);
                }
                Err(e) => {
                    // Return partial metrics and trajectory along with the error
                    let total_duration = start.elapsed();
                    let partial_metrics =
                        ExecutionMetrics::from_tracker(tracker.as_ref(), total_duration);
                    let partial_trajectory = build_trajectory(
                        events,
                        &llm_ref,
                        query,
                        agent,
                        &agent_config,
                        total_duration.as_millis() as u64,
                        partial_metrics.final_answer.clone(),
                    );
                    return Err((e, partial_metrics, partial_trajectory));
                }
            }
        }

        let total_duration = start.elapsed();
        let metrics = ExecutionMetrics::from_tracker(tracker.as_ref(), total_duration);

        let trajectory = build_trajectory(
            events,
            &llm_ref,
            query,
            agent,
            &agent_config,
            total_duration.as_millis() as u64,
            metrics.final_answer.clone(),
        );

        Ok((metrics, trajectory))
    }
}

impl Default for AgentRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Event type for external events injected from coordination.
pub const EVENT_EXTERNAL: &str = "external_event";

/// Convert an ExternalEvent to an AgentUpdate for stream injection.
fn external_event_to_update(event: &ExternalEvent) -> AgentUpdate {
    let message = format!(
        "[{}] {} (from: {})",
        event.event_type,
        event.payload,
        event.source_session.as_deref().unwrap_or("unknown")
    );

    AgentUpdate::custom(
        EVENT_EXTERNAL,
        message,
        json!({
            "id": event.id,
            "event_type": event.event_type,
            "payload": event.payload,
            "channel": event.channel,
            "source_session": event.source_session,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_stream::stream;
    use gemicro_core::{AgentStream, AgentUpdate, ResultMetadata};
    use std::sync::Arc;

    struct MockAgent {
        events: Vec<AgentUpdate>,
    }

    impl MockAgent {
        fn new(events: Vec<AgentUpdate>) -> Self {
            Self { events }
        }
    }

    impl Agent for MockAgent {
        fn name(&self) -> &str {
            "mock_agent"
        }

        fn description(&self) -> &str {
            "Mock agent for testing"
        }

        fn execute(&self, _query: &str, _context: AgentContext) -> AgentStream<'_> {
            let events = self.events.clone();
            Box::pin(stream! {
                for event in events {
                    yield Ok(event);
                }
            })
        }

        fn create_tracker(&self) -> Box<dyn gemicro_core::ExecutionTracking> {
            Box::new(gemicro_core::DefaultTracker::default())
        }
    }

    fn create_mock_context() -> AgentContext {
        // Create a minimal context for testing
        // Note: In real tests, you'd use a proper test helper
        use gemicro_core::{LlmClient, LlmConfig};

        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let llm = LlmClient::new(genai_client, LlmConfig::default());
        AgentContext::new(llm)
    }

    fn create_successful_events() -> Vec<AgentUpdate> {
        use serde_json::json;

        let metadata = ResultMetadata::new(100, 0, 1000);

        vec![
            AgentUpdate::custom("decomposition_started", "Decomposing query", json!({})),
            AgentUpdate::custom(
                "decomposition_complete",
                "Decomposed into 1 sub-query",
                json!({ "sub_queries": ["Q1"] }),
            ),
            AgentUpdate::custom(
                "sub_query_started",
                "Sub-query 0 started",
                json!({ "id": 0, "query": "Q1" }),
            ),
            AgentUpdate::custom(
                "sub_query_completed",
                "Sub-query 0 completed",
                json!({ "id": 0, "result": "Result", "tokens_used": 50 }),
            ),
            AgentUpdate::custom("synthesis_started", "Synthesizing results", json!({})),
            AgentUpdate::final_result(json!("Final answer"), metadata),
        ]
    }

    #[tokio::test]
    async fn test_runner_default() {
        let runner = AgentRunner;
        assert!(std::mem::size_of_val(&runner) == 0); // Zero-size struct
    }

    #[tokio::test]
    async fn test_runner_execute_with_tracking() {
        let runner = AgentRunner::new();
        let agent = MockAgent::new(create_successful_events());
        let context = create_mock_context();

        let status_messages = Arc::new(std::sync::Mutex::new(Vec::new()));
        let messages_clone = status_messages.clone();

        let metrics = runner
            .execute_with_tracking(&agent, "query", context, move |tracker, msg| {
                messages_clone.lock().unwrap().push(msg.to_string());
                // Verify tracker state is accessible during callback
                if tracker.is_complete() {
                    assert!(tracker.final_result().is_some());
                }
            })
            .await
            .unwrap();

        let messages = status_messages.lock().unwrap();
        // Should have received status updates (one per event with non-empty message)
        assert!(!messages.is_empty());
        // Metrics should reflect final result
        assert_eq!(metrics.total_tokens, 100);
        assert_eq!(metrics.final_answer, Some("Final answer".to_string()));
    }

    #[tokio::test]
    async fn test_runner_execute_with_trajectory() {
        let runner = AgentRunner::new();
        let agent = MockAgent::new(create_successful_events());
        let genai_client = genai_rs::Client::builder("test-key".to_string())
            .build()
            .unwrap();
        let llm_config = LlmConfig::default();

        let (metrics, trajectory) = runner
            .execute_with_trajectory(
                &agent,
                "test query",
                serde_json::json!({"temperature": 0.7}),
                genai_client,
                llm_config,
            )
            .await
            .unwrap();

        // Verify metrics
        assert_eq!(metrics.total_tokens, 100);
        assert_eq!(metrics.final_answer, Some("Final answer".to_string()));

        // Verify trajectory metadata
        assert_eq!(trajectory.query, "test query");
        assert_eq!(trajectory.agent_name, "mock_agent");
        assert_eq!(trajectory.agent_config["temperature"], 0.7);

        // Verify trajectory contains events
        assert_eq!(trajectory.events.len(), 6);

        // MockAgent doesn't make LLM calls, so steps should be empty
        // (In real tests with actual agents, steps would be populated)
        assert_eq!(trajectory.steps.len(), 0);
    }

    #[tokio::test]
    async fn test_runner_execute_with_events_no_coordination() {
        let runner = AgentRunner::new();
        let agent = MockAgent::new(create_successful_events());
        let context = create_mock_context();

        let updates = Arc::new(std::sync::Mutex::new(Vec::new()));
        let updates_clone = updates.clone();

        let metrics = runner
            .execute_with_events(&agent, "query", context, None, move |update| {
                updates_clone
                    .lock()
                    .unwrap()
                    .push(update.event_type.clone());
            })
            .await
            .unwrap();

        let events = updates.lock().unwrap();
        // Should have received all agent events
        assert_eq!(events.len(), 6);
        assert_eq!(events[0], "decomposition_started");
        assert_eq!(events[5], "final_result");

        // Metrics should reflect final result
        assert_eq!(metrics.total_tokens, 100);
        assert_eq!(metrics.final_answer, Some("Final answer".to_string()));
    }

    #[test]
    fn test_external_event_to_update() {
        let event = ExternalEvent::new(42, "task_completed", "Feature X done", "all")
            .with_source("happy-tiger");

        let update = external_event_to_update(&event);

        assert_eq!(update.event_type, EVENT_EXTERNAL);
        assert!(update.message.contains("[task_completed]"));
        assert!(update.message.contains("Feature X done"));
        assert!(update.message.contains("happy-tiger"));

        // Check data contains original event info
        assert_eq!(update.data["id"], 42);
        assert_eq!(update.data["event_type"], "task_completed");
        assert_eq!(update.data["payload"], "Feature X done");
    }

    #[test]
    fn test_external_event_to_update_unknown_source() {
        let event = ExternalEvent::new(1, "message", "Hello", "repo:gemicro");

        let update = external_event_to_update(&event);

        assert!(update.message.contains("unknown"));
        assert_eq!(update.data["source_session"], serde_json::Value::Null);
    }
}
