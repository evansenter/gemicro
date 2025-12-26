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

- [ ] Create agent-specific config struct with `validate()` method
- [ ] Define event type constants (`pub const EVENT_*: &str`)
- [ ] Implement `Agent` trait (`name`, `description`, `execute`)
- [ ] Use `async_stream::try_stream!` for streaming
- [ ] Handle timeouts and cancellation
- [ ] Add unit tests for config validation
- [ ] Add integration tests (`#[ignore]`)
- [ ] Export from `gemicro-core/src/agent/mod.rs` and `lib.rs`

## Core Types

### Agent Trait

Location: `gemicro-core/src/agent/mod.rs:87-99`

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

Location: `gemicro-core/src/agent/mod.rs:101-154`

Contains **only** cross-agent resources:

```rust
pub struct AgentContext {
    pub llm: Arc<LlmClient>,              // Shared LLM client
    pub cancellation_token: CancellationToken,  // For cooperative shutdown
}
```

**Important:** Do NOT add agent-specific config here. Config belongs in the agent constructor.

### AgentUpdate

Location: `gemicro-core/src/update.rs:39-58`

```rust
pub struct AgentUpdate {
    pub event_type: String,       // Semantic identifier (e.g., "simple_qa_result")
    pub message: String,          // Human-readable description
    pub timestamp: SystemTime,
    pub data: serde_json::Value,  // Arbitrary JSON payload
}
```

## Complete Example: SimpleQaAgent

The `SimpleQaAgent` is a minimal reference implementation. See the full source at `gemicro-core/src/agent/simple_qa.rs`.

### 1. Event Type Constants

```rust
// Location: gemicro-core/src/agent/simple_qa.rs:23-27

pub const EVENT_SIMPLE_QA_STARTED: &str = "simple_qa_started";
pub const EVENT_SIMPLE_QA_RESULT: &str = "simple_qa_result";
```

### 2. Configuration

```rust
// Location: gemicro-core/src/agent/simple_qa.rs:33-97

#[derive(Debug, Clone)]
pub struct SimpleQaConfig {
    pub timeout: Duration,
    pub system_prompt: String,
}

impl SimpleQaConfig {
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
// Location: gemicro-core/src/agent/simple_qa.rs:124-141

pub struct SimpleQaAgent {
    config: SimpleQaConfig,
}

impl SimpleQaAgent {
    pub fn new(config: SimpleQaConfig) -> Result<Self, AgentError> {
        config.validate()?;  // Fail fast on invalid config
        Ok(Self { config })
    }
}
```

### 4. Implement Agent Trait

```rust
// Location: gemicro-core/src/agent/simple_qa.rs:143-190

impl Agent for SimpleQaAgent {
    fn name(&self) -> &str {
        "simple_qa"
    }

    fn description(&self) -> &str {
        "A simple question-answering agent that makes a single LLM call"
    }

    fn execute(&self, query: &str, context: AgentContext) -> AgentStream<'_> {
        let query = query.to_string();
        let config = self.config.clone();

        Box::pin(try_stream! {
            let start = Instant::now();

            // Emit start event
            yield AgentUpdate::custom(
                EVENT_SIMPLE_QA_STARTED,
                format!("Processing query: {}", truncate(&query, 50)),
                json!({ "query": query }),
            );

            // Calculate remaining time and execute LLM call
            let timeout = remaining_time(start, config.timeout, "query")?;

            let request = LlmRequest::with_system(&query, &config.system_prompt);

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

            // Emit result event
            yield AgentUpdate::custom(
                EVENT_SIMPLE_QA_RESULT,
                response.text.clone(),
                json!({
                    "answer": response.text,
                    "tokens_used": response.tokens_used,
                    "duration_ms": start.elapsed().as_millis() as u64,
                }),
            );
        })
    }
}
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

Location: `gemicro-core/src/agent/mod.rs:160-214`

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
        let config = SimpleQaConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_rejects_zero_timeout() {
        let config = SimpleQaConfig {
            timeout: Duration::ZERO,
            system_prompt: "Valid".to_string(),
        };
        assert!(config.validate().is_err());
    }
}
```

### Integration Tests

Location: `gemicro-core/tests/simple_qa_integration.rs`

```rust
#[tokio::test]
#[ignore]  // Requires GEMINI_API_KEY
async fn test_simple_qa_full_flow() {
    let Some(api_key) = get_api_key() else {
        return;  // Skip if no API key
    };

    let context = create_test_context(&api_key);
    let agent = SimpleQaAgent::new(SimpleQaConfig::default()).unwrap();

    let stream = agent.execute("What is 2 + 2?", context);
    futures_util::pin_mut!(stream);

    let mut events = Vec::new();
    while let Some(result) = stream.next().await {
        let update = result.expect("Should not error");
        events.push(update.event_type.clone());
    }

    assert_eq!(events, vec![
        EVENT_SIMPLE_QA_STARTED,
        EVENT_SIMPLE_QA_RESULT,
    ]);
}
```

Run integration tests with:
```bash
cargo test --package gemicro-core -- --include-ignored
```

## Registry Integration

Location: `gemicro-runner/src/registry.rs`

### Registering an Agent

```rust
use gemicro_runner::AgentRegistry;

let mut registry = AgentRegistry::new();

registry.register("simple_qa", || {
    // unwrap() is safe here: SimpleQaConfig::default() is always valid
    Box::new(SimpleQaAgent::new(SimpleQaConfig::default()).unwrap())
});

// Create fresh instance for each execution
let agent = registry.get("simple_qa").unwrap();
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

// ✅ DO - soft-typed events
pub const EVENT_REACT_STEP_COMPLETED: &str = "react_step_completed";
AgentUpdate::custom(EVENT_REACT_STEP_COMPLETED, ...)
```

## File Structure for New Agents

```
gemicro-core/
├── src/
│   ├── agent/
│   │   ├── mod.rs          # Add: mod my_agent; pub use my_agent::*;
│   │   ├── deep_research.rs
│   │   ├── simple_qa.rs    # Reference implementation
│   │   └── my_agent.rs     # Your new agent
│   └── lib.rs              # Add exports
└── tests/
    └── my_agent_integration.rs  # Integration tests
```

## See Also

- `gemicro-core/src/agent/simple_qa.rs` - Full reference implementation
- `gemicro-core/src/agent/deep_research.rs` - Complex multi-phase example
- `gemicro-core/tests/simple_qa_integration.rs` - Integration test examples
- `CLAUDE.md` - Project design philosophy
