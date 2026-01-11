# Agent Authoring Guide

This guide walks you through implementing a new agent in Gemicro, following the [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) philosophy of soft-typed events and extensible design.

## Overview

### What is an Agent?

An agent in Gemicro is a streaming processor that:
1. Receives a query and shared context
2. Executes some logic (LLM calls, tool use, etc.)
3. Yields a stream of `AgentUpdate` events for real-time observability
4. Returns a final result

### Design Philosophy

Following Evergreen principles:

| Principle | Description |
|-----------|-------------|
| **Soft-typed events** | Event meaning lives in `event_type: String` + `data: JSON`, not rigid enums |
| **Agent-specific config** | Configuration belongs in the agent constructor, not shared context |
| **Graceful unknowns** | Consumers ignore unknown event types (log and continue, never error) |
| **Forward compatible** | New agents and events can be added without modifying core types |

## Quick Start Checklist

- [ ] Create new agent crate: `agents/gemicro-{agent-name}/`
- [ ] Add crate to workspace `Cargo.toml` members
- [ ] Create config struct with `#[non_exhaustive]`, `validate()`, and `with_*()` builder methods
- [ ] Define event types as strings (constants are internal, NOT exported)
- [ ] Implement `Agent` trait (`name`, `description`, `execute`)
- [ ] Use `async_stream::try_stream!` for streaming
- [ ] Handle timeouts and cancellation
- [ ] Add unit tests for config validation
- [ ] Add integration tests (`#[ignore]`) in `tests/integration.rs`

## Core Types

### Agent Trait

Location: `gemicro-core/src/agent.rs`

```rust
pub trait Agent: Send + Sync {
    /// Unique identifier for this agent type (e.g., "simple_qa", "deep_research")
    fn name(&self) -> &str;

    /// Human-readable description of what this agent does
    fn description(&self) -> &str;

    /// Execute the agent's logic and return a stream of updates
    fn execute(&self, query: &str, context: AgentContext) -> AgentStream<'_>;
}
```

### AgentStream

```rust
pub type AgentStream<'a> = Pin<Box<dyn Stream<Item = Result<AgentUpdate, AgentError>> + Send + 'a>>;
```

### AgentContext

Location: `gemicro-core/src/agent.rs`

Contains **only** cross-agent resources:

```rust
pub struct AgentContext {
    pub llm: Arc<LlmClient>,              // Shared LLM client
    pub cancellation_token: CancellationToken,  // For cooperative shutdown
    pub tools: Option<Arc<ToolRegistry>>,       // Shared tool registry
    pub confirmation_handler: Option<Arc<dyn ConfirmationHandler>>,  // Tool confirmation
    pub execution: ExecutionContext,      // Execution tracking for subagents
}
```

**Important:** Do NOT add agent-specific config here. Config belongs in the agent constructor.

### ExecutionContext (Subagent Tracking)

Location: `gemicro-core/src/agent/execution.rs`

Tracks parent-child relationships for observability and debugging:

```rust
#[derive(Clone, Debug)]
pub struct ExecutionContext {
    /// This execution's unique ID
    pub execution_id: ExecutionId,

    /// Parent execution ID (None for root)
    pub parent_id: Option<ExecutionId>,

    /// Depth in execution tree (0 for root)
    pub depth: usize,

    /// Path from root (for logging): ["deep_research", "simple_qa"]
    pub path: Vec<String>,
}

impl ExecutionContext {
    /// Create a root execution context (no parent)
    pub fn root() -> Self;

    /// Create a child context for spawning subagents
    pub fn child(&self, agent_name: &str) -> Self;

    /// Format path as string for logging: "deep_research → simple_qa"
    pub fn path_string(&self) -> String;
}
```

**Usage in subagent spawning:**

```rust
// When spawning a subagent via Task tool:
let parent_context = context.execution.clone();
let child_context = parent_context.child("simple_qa");

// child_context now has:
// - New unique execution_id
// - parent_id = parent_context.execution_id
// - depth = parent_context.depth + 1
// - path = [...parent_path, "simple_qa"]
```

### AgentUpdate

Location: `gemicro-core/src/update.rs`

```rust
pub struct AgentUpdate {
    pub event_type: String,       // Semantic identifier (e.g., "prompt_agent_result")
    pub message: String,          // Human-readable description
    pub timestamp: SystemTime,
    pub data: serde_json::Value,  // Arbitrary JSON payload
}
```

## Complete Example: PromptAgent

The `PromptAgent` is a minimal reference implementation. See the full source at `agents/gemicro-prompt-agent/src/lib.rs`.

### 1. Event Type Constants (Internal Only)

Constants are internal to prevent typos and ensure consistency within the agent. They are **not exported** to the public API - consumers match on string literals.

```rust
// Internal constants - private to this module
const EVENT_PROMPT_AGENT_STARTED: &str = "prompt_agent_started";
const EVENT_PROMPT_AGENT_RESULT: &str = "prompt_agent_result";
```

Use `AgentUpdate::custom()` with these constants:

```rust
yield AgentUpdate::custom(
    EVENT_PROMPT_AGENT_STARTED,
    format!("Processing query: {}", truncate(&query, 50)),
    json!({ "query": query }),
);
```

### 2. Configuration

Mark config structs with `#[non_exhaustive]` and provide builder methods:

```rust
// Location: agents/gemicro-prompt-agent/src/lib.rs

#[derive(Debug, Clone)]
#[non_exhaustive]  // Allows adding fields without breaking changes
pub struct PromptAgentConfig {
    pub timeout: Duration,
    pub system_prompt: String,
}

impl PromptAgentConfig {
    /// Set the timeout duration.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the system prompt.
    #[must_use]
    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = system_prompt.into();
        self
    }
}

impl PromptAgentConfig {
    pub fn validate(&self) -> Result<(), AgentError> {
        let mut errors = Vec::new();

        if self.timeout.is_zero() {
            errors.push("timeout must be greater than zero");
        }
        if self.system_prompt.trim().is_empty() {
            errors.push("system_prompt must not be empty");
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(AgentError::InvalidConfig(errors.join("; ")))
        }
    }
}
```

### 3. Agent Struct and Constructor

```rust
// Location: agents/gemicro-prompt-agent/src/lib.rs

pub struct PromptAgent {
    config: PromptAgentConfig,
}

impl PromptAgent {
    pub fn new(config: PromptAgentConfig) -> Result<Self, AgentError> {
        config.validate()?;  // Fail fast on invalid config
        Ok(Self { config })
    }
}
```

### 4. Implement Agent Trait

```rust
// Location: agents/gemicro-prompt-agent/src/lib.rs

impl Agent for PromptAgent {
    fn name(&self) -> &str {
        "prompt_agent"
    }

    fn description(&self) -> &str {
        "A prompt-based agent that makes a single LLM call with optional tools"
    }

    fn execute(&self, query: &str, context: AgentContext) -> AgentStream<'_> {
        let query = query.to_string();
        let config = self.config.clone();

        Box::pin(try_stream! {
            let start = Instant::now();

            // Emit start event
            yield AgentUpdate::custom(
                EVENT_PROMPT_AGENT_STARTED,
                format!("Processing query: {}", truncate(&query, 50)),
                json!({ "query": query }),
            );

            // Calculate remaining time and execute LLM call
            let timeout = remaining_time(start, config.timeout, "query")?;

            let request = context.llm.client().interaction()
                .system_instruction(&config.system_prompt)
                .user_text(&query)
                .build();

            // Wrap LLM call to convert LlmError -> AgentError
            let generate_future = async {
                context.llm
                    .generate(request)
                    .await
                    .map_err(|e| AgentError::Other(format!("LLM error: {}", e)))
            };

            let response = with_timeout_and_cancellation(
                generate_future,
                timeout,
                &context.cancellation_token,
                || timeout_error(start, config.timeout, "query"),
            ).await?;

            // Extract text and token count from InteractionResponse
            let answer = response.text().unwrap_or("").to_string();
            let tokens_used = extract_total_tokens(&response);

            // Emit result event
            yield AgentUpdate::custom(
                EVENT_PROMPT_AGENT_RESULT,
                answer.clone(),
                json!({
                    "answer": answer,
                    "tokens_used": tokens_used,
                    "duration_ms": start.elapsed().as_millis() as u64,
                }),
            );
        })
    }

    fn create_tracker(&self) -> Box<dyn ExecutionTracking> {
        Box::new(DefaultTracker::default())
    }
}
```

## Execution Tracking and Phases

Agents provide their own execution tracking logic via the `create_tracker()` method. This enables the CLI and runner to remain agent-agnostic while still providing meaningful progress updates.

### The ExecutionTracking Trait

Location: `gemicro-core/src/tracking.rs`

```rust
pub trait ExecutionTracking: Send + Sync {
    /// Process an event and update internal state.
    fn handle_event(&mut self, event: &AgentUpdate);

    /// Current status message for display.
    fn status_message(&self) -> Option<&str>;

    /// Whether execution is complete.
    fn is_complete(&self) -> bool;

    /// Final result data (available only when is_complete() is true).
    fn final_result(&self) -> Option<&FinalResult>;
}
```

### DefaultTracker

For simple agents, use `DefaultTracker` which extracts status from each event's `message` field:

```rust
// In your Agent trait implementation:
fn create_tracker(&self) -> Box<dyn ExecutionTracking> {
    Box::new(DefaultTracker::default())
}
```

`DefaultTracker`:
- Updates `status_message()` with each event's `message` field
- Captures `final_result` when `final_result` event arrives
- Returns `is_complete() = true` once the final result is set

This is sufficient for most agents since the event `message` field provides the human-readable status.

### Custom Trackers

For agents with complex execution patterns (e.g., tracking step counts, parallel progress), implement a custom tracker:

```rust
#[derive(Debug, Default)]
struct MyAgentTracker {
    current_step: usize,
    total_steps: usize,
    status: String,
    result: Option<FinalResult>,
}

impl ExecutionTracking for MyAgentTracker {
    fn handle_event(&mut self, event: &AgentUpdate) {
        match event.event_type.as_str() {
            "my_agent_step_started" => {
                self.current_step = event.data["step"].as_u64().unwrap_or(0) as usize;
                self.total_steps = event.data["total"].as_u64().unwrap_or(0) as usize;
                self.status = format!("Processing step {}/{}", self.current_step, self.total_steps);
            }
            "final_result" => {
                if let Some(result) = event.as_final_result() {
                    self.result = Some(result);
                }
            }
            _ => {
                // Unknown events: update status from message (graceful handling)
                self.status = event.message.clone();
            }
        }
    }

    fn status_message(&self) -> Option<&str> {
        if self.status.is_empty() { None } else { Some(&self.status) }
    }

    fn is_complete(&self) -> bool {
        self.result.is_some()
    }

    fn final_result(&self) -> Option<&FinalResult> {
        self.result.as_ref()
    }
}
```

### How Tracking Works

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         Execution Flow                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Agent.execute()           Tracker                   CLI/Runner     │
│        │                       │                           │         │
│        │  AgentUpdate          │                           │         │
│        ├──────────────────────>│                           │         │
│        │                       │  handle_event()           │         │
│        │                       ├────────────────────────>  │         │
│        │                       │                           │         │
│        │                       │  status_message()         │         │
│        │                       │<─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │         │
│        │                       │                           │         │
│        │                       │       Display             │         │
│        │                       │      "Step 2/5..."        │         │
│        │                       │                           │         │
│        │  final_result         │                           │         │
│        ├──────────────────────>│                           │         │
│        │                       │  final_result()           │         │
│        │                       │<─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │         │
│        │                       │                           │         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Phases vs Events

Gemicro has two related but distinct concepts:

| Concept | Location | Purpose |
|---------|----------|---------|
| **Events** | `AgentUpdate.event_type` | Semantic identifiers for what happened |
| **Phases** | `ExecutionState.phase` | High-level execution stage for state tracking |

**Events** are emitted by agents. **Phases** are used by `ExecutionState` (in gemicro-runner) for detailed state tracking, typically in rich UIs.

#### Well-Known Phases

Location: `gemicro-runner/src/state.rs`

```rust
pub mod phases {
    pub const NOT_STARTED: &str = "not_started";
    pub const COMPLETE: &str = "complete";

    // DeepResearch phases
    pub const DECOMPOSING: &str = "decomposing";
    pub const EXECUTING: &str = "executing";
    pub const SYNTHESIZING: &str = "synthesizing";

    // ReAct phases
    pub const THINKING: &str = "thinking";
    pub const ACTING: &str = "acting";
    pub const OBSERVING: &str = "observing";
}
```

Agents can use these or define their own phase names.

#### Event-to-Phase Mapping

For agents that integrate with `ExecutionState`, events map to phases:

**DeepResearchAgent:**
| Event | Phase |
|-------|-------|
| `decomposition_started` | `decomposing` |
| `decomposition_complete` | (still `decomposing`) |
| `sub_query_started` | `executing` |
| `sub_query_completed`/`sub_query_failed` | (still `executing`) |
| `synthesis_started` | `synthesizing` |
| `final_result` | `complete` |

**ReactAgent:**
| Event | Phase |
|-------|-------|
| `react_started` | `thinking` |
| `react_thought` | `thinking` |
| `react_action` | `acting` |
| `react_observation` | `observing` |
| `react_complete`/`react_max_iterations` | `complete` |
| `final_result` | `complete` |

### When to Use What

| Use Case | Mechanism |
|----------|-----------|
| **Simple CLI spinner** | `DefaultTracker` + event messages |
| **Step-by-step progress** | Custom tracker counting events |
| **Rich TUI with state** | `ExecutionState` with phase tracking |
| **Metrics collection** | `ExecutionMetrics.from_tracker()` |

### Status Message Best Practices

The `message` field in `AgentUpdate::custom()` becomes the status message in `DefaultTracker`:

```rust
// ✅ Good: Human-readable, informative
yield AgentUpdate::custom(
    "sub_query_started",
    "Researching: What are the benefits of Rust?",  // Displayed to user
    json!({ "id": 0, "query": "What are the benefits of Rust?" }),
);

// ✅ Good: Shows progress
yield AgentUpdate::custom(
    "sub_query_completed",
    "Sub-query 3/5 completed",  // User sees "Sub-query 3/5 completed"
    json!({ "id": 2, "result": "..." }),
);

// ❌ Avoid: Technical, not user-friendly
yield AgentUpdate::custom(
    "sub_query_completed",
    "SQ_COMPLETE",  // User sees "SQ_COMPLETE" 🤔
    json!({ "id": 2 }),
);
```

## Event System Patterns

### Creating Custom Events

Use `AgentUpdate::custom()` for new event types:

```rust
yield AgentUpdate::custom(
    "my_agent_step_completed",           // event_type
    "Completed step 1 of 3",             // human-readable message
    json!({                               // arbitrary data
        "step": 1,
        "total": 3,
        "result": "some value"
    }),
);
```

### Adding Typed Accessors (Optional)

For frequently-accessed event data, add accessor methods to `AgentUpdate`:

```rust
// In gemicro-core/src/update.rs
impl AgentUpdate {
    pub fn as_my_agent_result(&self) -> Option<MyAgentResult> {
        if self.event_type != "my_agent_result" {
            return None;
        }
        serde_json::from_value(self.data.clone()).ok()
    }
}
```

### What NOT to Do

```rust
// DON'T: Add variants to a shared enum (requires modifying core)
pub enum AgentUpdate {
    MyNewAgentEvent { ... },  // ❌ Breaks Evergreen philosophy
}

// DO: Use soft-typed events
AgentUpdate::custom("my_new_agent_event", ...)  // ✅ Extensible
```

## Timeout and Cancellation

### Helper Functions

Location: `gemicro-core/src/agent.rs`

```rust
// Calculate remaining time from a total budget
let timeout = remaining_time(start, total_timeout, "phase_name")?;

// Execute with both timeout and cancellation support
let result = with_timeout_and_cancellation(
    async_operation(),
    timeout,
    &context.cancellation_token,
    || timeout_error(start, total_timeout, "phase_name"),
).await?;
```

### Phase Budgeting

For multi-phase agents, track time across phases:

```rust
let start = Instant::now();
let total_timeout = config.total_timeout;

// Phase 1
let phase1_timeout = remaining_time(start, total_timeout, "decomposition")?;
// ... execute phase 1 ...

// Phase 2 (uses remaining time from budget)
let phase2_timeout = remaining_time(start, total_timeout, "execution")?;
// ... execute phase 2 ...
```

## Parallel Execution

For agents that execute work in parallel (like `DeepResearchAgent`):

```rust
// Semaphore for concurrency limiting
let semaphore = if config.max_concurrent > 0 {
    Some(Arc::new(Semaphore::new(config.max_concurrent)))
} else {
    None  // Unlimited
};

// Channel for collecting results
let (tx, mut rx) = mpsc::channel(queries.len());

// Spawn tasks
for (id, query) in queries.iter().enumerate() {
    let tx = tx.clone();
    let sem = semaphore.clone();

    tokio::spawn(async move {
        let _permit = if let Some(s) = sem {
            Some(s.acquire().await.ok())
        } else {
            None
        };

        let result = execute_query(query).await;
        let _ = tx.send((id, result)).await;
    });
}

// Important: Drop sender to signal completion
drop(tx);

// Collect results (non-deterministic order)
while let Some((id, result)) = rx.recv().await {
    // Process result
}
```

## Testing Patterns

### Unit Tests (Config Validation)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = PromptAgentConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_rejects_zero_timeout() {
        let config = PromptAgentConfig {
            timeout: Duration::ZERO,
            system_prompt: "Valid".to_string(),
        };
        assert!(config.validate().is_err());
    }
}
```

### Integration Tests

Location: `agents/gemicro-prompt-agent/tests/integration.rs`

```rust
#[tokio::test]
#[ignore]  // Requires GEMINI_API_KEY
async fn test_prompt_agent_full_flow() {
    let Some(api_key) = get_api_key() else {
        return;  // Skip if no API key
    };

    let context = create_test_context(&api_key);
    let agent = PromptAgent::new(PromptAgentConfig::default()).unwrap();

    let stream = agent.execute("What is 2 + 2?", context);
    futures_util::pin_mut!(stream);

    let mut events = Vec::new();
    while let Some(result) = stream.next().await {
        let update = result.expect("Should not error");
        events.push(update.event_type.clone());
    }

    // Use string literals to match events (constants are internal)
    assert_eq!(events, vec![
        "prompt_agent_started",
        "prompt_agent_result",
        "final_result",
    ]);
}
```

Run integration tests with:
```bash
cargo nextest run -p gemicro-prompt-agent --run-ignored all
```

## Registry Integration

Location: `gemicro-runner/src/registry.rs`

### Registering an Agent

```rust
use gemicro_runner::AgentRegistry;

let mut registry = AgentRegistry::new();

registry.register("prompt_agent", || {
    // unwrap() is safe here: PromptAgentConfig::default() is always valid
    Box::new(PromptAgent::new(PromptAgentConfig::default()).unwrap())
});

// Create fresh instance for each execution
let agent = registry.get("prompt_agent").unwrap();
```

### Why Factories?

- Fresh agent instance per execution (no shared state)
- Allows config changes between runs
- Each task gets independent agent

## Common Pitfalls

### 1. Config in execute() instead of constructor

```rust
// ❌ DON'T
fn execute(&self, query: &str, config: MyConfig, context: AgentContext) { ... }

// ✅ DO
fn new(config: MyConfig) -> Self { ... }
fn execute(&self, query: &str, context: AgentContext) { ... }
```

### 2. Ignoring cancellation token

```rust
// ❌ DON'T - blocks forever if cancelled
let result = long_operation().await;

// ✅ DO - respects cancellation
let result = with_timeout_and_cancellation(
    long_operation(),
    timeout,
    &context.cancellation_token,
    || timeout_error(...),
).await?;
```

### 3. Panicking in streaming code

```rust
// ❌ DON'T - panics abort the stream ungracefully
panic!("something went wrong");

// ✅ DO - return errors through the stream
return Err(AgentError::Other("something went wrong".into()));
```

### 4. Rigid event enums in core

```rust
// ❌ DON'T - requires modifying gemicro-core for each new agent
pub enum AgentEventType {
    DeepResearchStarted,
    ReactStepCompleted,
    // ... grows forever
}

// ✅ DO - soft-typed events with internal constants
// In your agent module (NOT exported):
const EVENT_REACT_STEP_COMPLETED: &str = "react_step_completed";
AgentUpdate::custom(EVENT_REACT_STEP_COMPLETED, ...)

// Consumers match on string literals:
match update.event_type.as_str() {
    "react_step_completed" => { /* handle */ }
    _ => { log::debug!("Unknown event"); }
}
```

## File Structure for New Agents

Each agent gets its own crate in the `agents/` subdirectory:

```
agents/
└── gemicro-my-agent/
    ├── Cargo.toml           # Depends on gemicro-core only
    ├── src/
    │   ├── lib.rs           # Public exports: MyAgent, MyConfig
    │   ├── agent.rs         # Agent implementation (optional split)
    │   └── config.rs        # Config struct (optional split)
    └── tests/
        └── integration.rs   # Integration tests (#[ignore])
```

**Cargo.toml template:**

```toml
[package]
name = "gemicro-my-agent"
version = "0.1.0"
edition = "2021"

[dependencies]
gemicro-core = { path = "../../gemicro-core" }
async-stream = "0.3"
futures-util = "0.3"
log = "0.4"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.0", features = ["time", "sync"] }

[dev-dependencies]
tokio = { version = "1.0", features = ["rt-multi-thread", "macros"] }
```

**Add to workspace Cargo.toml:**

```toml
[workspace]
members = [
    # ... existing members
    "agents/gemicro-my-agent",
]
```

**Note:** Each agent crate depends ONLY on gemicro-core, never on other agent crates. This enforces hermetic isolation at compile time.

## Trajectory Recording and Replay

Gemicro supports capturing full LLM interaction traces during agent execution for offline replay, evaluation, and debugging.

### Recording Trajectories

Use `AgentRunner::execute_with_trajectory()` to capture execution traces:

```rust
use gemicro_core::{LlmConfig, Trajectory};
use gemicro_runner::AgentRunner;
use serde_json::json;

async fn record_trajectory(agent: &impl Agent, query: &str) -> Result<Trajectory, AgentError> {
    let genai_client = rust_genai::Client::builder(api_key).build();
    let llm_config = LlmConfig::default();

    // Use AgentRunner for trajectory capture
    let runner = AgentRunner::new();
    let (metrics, trajectory) = runner.execute_with_trajectory(
        agent,
        query,
        json!({"temperature": 0.7}),  // Agent config as JSON
        genai_client,
        llm_config,
    ).await?;

    // Save for later replay
    trajectory.save("trajectories/run_001.json")?;
    Ok(trajectory)
}
```

### Replaying Trajectories

Use `MockLlmClient` to replay recorded trajectories without API calls:

```rust
use gemicro_core::{MockLlmClient, Trajectory};

async fn replay_trajectory() -> Result<(), Box<dyn std::error::Error>> {
    // Load previously recorded trajectory
    let trajectory = Trajectory::load("trajectories/run_001.json")?;

    // Create mock client that replays the recorded responses
    let mock = MockLlmClient::from_trajectory(&trajectory);

    // Use like a regular LlmClient - returns recorded responses in order
    let response = mock.generate("Any prompt").await?;
    println!("Replayed: {}", response["text"]);

    Ok(())
}
```

### Trajectory Structure

A trajectory contains:

| Field | Description |
|-------|-------------|
| `id` | Unique UUID |
| `query` | Original user query |
| `agent_name` | Agent that executed |
| `agent_config` | Agent configuration (soft-typed JSON) |
| `steps` | Raw LLM request/response pairs with timing |
| `events` | High-level `AgentUpdate` events |
| `metadata` | Tokens, duration, schema version |

### Use Cases

1. **Offline Testing**: Run agents without API calls for fast unit tests
2. **Evaluation**: Score agent responses against ground truth
3. **Debugging**: Inspect exact LLM requests and responses
4. **Dataset Creation**: Build evaluation datasets from production runs

### Integration with Eval Framework

Load trajectories as evaluation datasets with `TrajectoryDataset`:

```rust
use gemicro_eval::{TrajectoryDataset, Dataset};
use std::path::PathBuf;

async fn evaluate_from_trajectories() -> Result<(), Box<dyn std::error::Error>> {
    // Load trajectories from a directory
    let dataset = TrajectoryDataset::new(PathBuf::from("trajectories/"));
    let questions = dataset.load(Some(100)).await?;

    println!("Loaded {} questions from trajectories", questions.len());
    Ok(())
}
```

## Using Tools in Agents

Agents can use tools via the `AgentContext.tools` field, which provides access to a shared `ToolRegistry`.

### Basic Tool Usage

```rust
impl Agent for MyToolAgent {
    fn execute(&self, query: &str, context: AgentContext) -> AgentStream<'_> {
        Box::pin(try_stream! {
            // Get tool from registry
            let tools = context.tools
                .ok_or(AgentError::Other("No tools available".into()))?;

            let file_read = tools.get("file_read")
                .ok_or(AgentError::Other("file_read tool not found".into()))?;

            // Execute tool
            let result = file_read.execute(json!({"path": "/etc/hosts"}))
                .await
                .map_err(|e| AgentError::Other(format!("Tool error: {}", e)))?;

            yield AgentUpdate::custom("tool_result", result.content.to_string(), json!({}));
        })
    }
}
```

### Using genai-rs's Automatic Function Calling

For LLM-driven tool use (where the model decides which tools to call), use `GemicroToolService` with genai-rs:

```rust
use gemicro_core::tool::{GemicroToolService, ToolSet};
use std::sync::Arc;

impl Agent for PromptAgent {
    fn execute(&self, query: &str, context: AgentContext) -> AgentStream<'_> {
        Box::pin(try_stream! {
            let tools = context.tools.clone().unwrap();

            // Create tool service for genai-rs
            let service = Arc::new(
                GemicroToolService::new(tools)
                    .with_filter(ToolSet::All)
                    .with_confirmation_handler(
                        context.confirmation_handler.clone().unwrap_or_else(|| Arc::new(AutoDeny))
                    )
            );

            // Use with automatic function calling
            let response = context.llm.genai_client()
                .interaction()
                .with_model("gemini-2.0-flash")
                .with_system(&self.config.system_prompt)
                .with_user(query)
                .with_tool_service(service)
                .create_with_auto_functions()
                .await?;

            // LLM automatically called tools as needed
            let answer = response.text().unwrap_or("");
            yield AgentUpdate::final_result(answer.to_string(), ResultMetadata::default());
        })
    }
}
```

See `agents/gemicro-prompt-agent/` for a complete implementation.

## Subagent Orchestration

Agents can spawn subagents via the Task tool. This section covers the types that enable subagent orchestration.

### SubagentConfig

Location: `gemicro-core/src/agent/subagent.rs`

Controls what resources subagents can access:

```rust
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SubagentConfig {
    /// Which tools the subagent can access
    pub tools: ToolSet,

    /// Whether subagent inherits parent's hooks
    pub inherit_hooks: bool,

    /// Whether subagent can spawn its own subagents
    pub allow_nested: bool,

    /// Timeout for subagent execution
    pub timeout: Duration,
}
```

**Presets:**

```rust
// Inherit everything from parent (default)
SubagentConfig::default()

// Minimal permissions
SubagentConfig::restrictive()
    // tools: ToolSet::None
    // inherit_hooks: false
    // allow_nested: false

// Maximum permissions
SubagentConfig::permissive()
    // tools: ToolSet::All
    // inherit_hooks: true
    // allow_nested: true
```

**Builder pattern:**

```rust
let config = SubagentConfig::default()
    .with_tools(ToolSet::Specific(vec!["file_read".into(), "grep".into()]))
    .with_timeout(Duration::from_secs(30))
    .without_nested_subagents();
```

### ToolSet Inheritance

Location: `gemicro-core/src/tool/mod.rs`

When spawning subagents, tools can inherit from the parent:

```rust
pub enum ToolSet {
    All,                        // All registered tools
    None,                       // No tools
    Specific(Vec<String>),      // Only named tools
    Except(Vec<String>),        // All except named tools
    Inherit,                    // Inherit parent's tool set
    InheritExcept(Vec<String>), // Inherit parent's tools, minus these
}
```

**Resolution:**

```rust
// Parent has: ToolSet::Except(["bash"])
let parent_tools = ToolSet::Except(vec!["bash".into()]);

// Child wants to inherit but also exclude "file_write"
let child_tools = ToolSet::InheritExcept(vec!["file_write".into()]);

// Resolve to concrete set
let resolved = child_tools.resolve(&parent_tools);
// Result: ToolSet::Except(["bash", "file_write"])
```

**Important:** `Inherit` and `InheritExcept` must be resolved before use:

```rust
let tools = ToolSet::Inherit;
assert!(tools.needs_resolution());  // true

// Panics if used without resolution:
// tools.matches("grep")  // PANIC!

// Must resolve first:
let resolved = tools.resolve(&parent_tools);
resolved.matches("grep")  // OK
```

### Ephemeral Agents (PromptAgentDef)

Location: `gemicro-core/src/agent/ephemeral.rs`

Define agents inline without pre-registration:

```rust
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PromptAgentDef {
    /// Human-readable description
    pub description: String,

    /// System prompt for the agent
    pub system_prompt: String,

    /// Which tools this agent can use
    pub tools: ToolSet,

    /// Optional model override (e.g., use cheaper model)
    pub model: Option<String>,
}
```

**Builder pattern:**

```rust
use gemicro_core::agent::{PromptAgentDef, AgentSpec};

let def = PromptAgentDef::new("Python security reviewer")
    .with_system_prompt("You are an expert Python security auditor...")
    .with_tools(ToolSet::Specific(vec!["file_read".into(), "grep".into()]))
    .with_model("gemini-1.5-flash");
```

### AgentSpec

Unified type for referencing agents (named or ephemeral):

```rust
pub enum AgentSpec {
    /// Reference a pre-registered agent
    Named(String),

    /// Define a prompt-based agent inline
    Prompt(PromptAgentDef),
}
```

**Usage:**

```rust
// Reference a registered agent
let spec = AgentSpec::Named("deep_research".into());

// Define inline using builder
let spec = AgentSpec::Prompt(
    PromptAgentDef::new("Code reviewer")
        .with_system_prompt("Review code for quality...")
);

// Convenience method
let def = AgentSpec::prompt("Code reviewer")
    .with_system_prompt("Review code for quality...");
```

### Task Tool Integration

The Task tool (in `tools/gemicro-task/`) uses these types to spawn subagents:

```rust
// JSON input from LLM:
{
    "agent": "simple_qa",  // Named agent reference
    "query": "What is Rust?",
    "config": {
        "timeout_secs": 30,
        "allow_nested": false
    }
}

// Or with ephemeral agent:
{
    "agent": {
        "type": "prompt",
        "description": "Python security reviewer",
        "system_prompt": "You are an expert Python security auditor...",
        "tools": ["file_read", "grep"],
        "model": "gemini-1.5-flash"
    },
    "query": "Review src/auth.py for security issues"
}
```

**Depth limiting:**

The Task tool enforces a maximum subagent depth to prevent infinite recursion:

```rust
let task = Task::new(registry)
    .with_max_depth(3)  // Default is 3
    .with_parent_context(parent_execution);
```

### Convenience Constructors

`PromptAgent` provides constructors for use as an ephemeral agent backend:

```rust
// Create with custom system prompt (uses default timeout)
let agent = PromptAgent::with_system_prompt(
    "You are a Python security auditor. Review code for vulnerabilities."
)?;

// Create with custom prompt and timeout
let agent = PromptAgent::with_system_prompt_and_timeout(
    "You are a code reviewer.",
    Duration::from_secs(60),
)?;
```

These are used by the Task tool when executing `PromptAgentDef`:

```rust
// Task tool internally does:
let agent = PromptAgent::with_system_prompt(&prompt_def.system_prompt)?;
let stream = agent.execute(&query, child_context);
```

## See Also

- `agents/gemicro-prompt-agent/src/lib.rs` - Full reference implementation with tool support
- `agents/gemicro-deep-research/src/` - Complex multi-phase example
- `tools/gemicro-task/src/lib.rs` - Task tool for spawning subagents
- `agents/gemicro-prompt-agent/tests/integration.rs` - Integration test examples
- `agents/gemicro-prompt-agent/examples/trajectory_recording.rs` - Trajectory recording example
- `agents/gemicro-developer/examples/subagent_delegation.rs` - Subagent delegation with CritiqueAgent
- `docs/TOOL_AUTHORING.md` - Creating new tools
- `docs/INTERCEPTOR_AUTHORING.md` - Creating interceptors to intercept tools
- `CLAUDE.md` - Project design philosophy and crate responsibilities
