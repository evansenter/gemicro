# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Gemicro is a CLI agent exploration platform for experimenting with AI agent implementation patterns, powered by the Gemini API via the rust-genai library.

**Key Architecture**: Two-crate workspace (gemicro-core library + gemicro-cli binary, coming soon)

## Core Design Philosophy: Evergreen-Inspired Soft-Typing

**CRITICAL**: This project follows the [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) philosophy of **pragmatic flexibility over rigid typing**.

### Principles

| Principle | Description |
|-----------|-------------|
| **Soft-Typed Events** | `event_type: String` + `data: JSON` instead of rigid enums |
| **Semantic Meaning in Metadata** | Meaning lives in field names and event_type, not structure |
| **ID Opacity** | IDs are opaque identifiers—never encode semantics in ID values |
| **Named Parameters** | JSON fields are named; adding new fields is always non-breaking |
| **Graceful Unknown Handling** | Unknown event_types and data fields MUST be ignored, not errors |
| **Idempotent Events** | Events should be safely re-processable without side effects |
| **Agent-Specific Config Isolation** | Config belongs to agent constructors, not shared context |

### What This Means in Practice

#### ✅ DO: Soft-Typed Events (AgentUpdate)

```rust
// CORRECT: Flexible, extensible
pub struct AgentUpdate {
    pub event_type: String,          // "sub_query_completed"
    pub message: String,
    pub timestamp: SystemTime,
    pub data: serde_json::Value,     // Arbitrary JSON
}

// Helper constructors for ergonomics
impl AgentUpdate {
    pub fn sub_query_completed(id: usize, result: String, tokens: u32) -> Self {
        Self {
            event_type: "sub_query_completed".into(),
            data: json!({ "id": id, "result": result, "tokens_used": tokens }),
            // ...
        }
    }
}
```

#### ❌ DON'T: Rigid Enums for Extensible Types

```rust
// WRONG: Requires modifying core for each new agent type
pub enum AgentUpdate {
    SubQueryCompleted { id: usize, result: String, tokens: u32 },
    ReactStepCompleted { thought: String, action: String },      // Needs core change
    ReflexionCritiqueStarted { iteration: usize },               // Needs core change
    // ... would grow to 30+ variants
}
```

#### ✅ DO: Agent-Specific Config Passed to Constructors

```rust
// CORRECT: Agent owns its config
pub struct ResearchConfig {
    pub max_sub_queries: usize,
    // ... Deep Research specific fields
}

impl DeepResearchAgent {
    pub fn new(config: ResearchConfig) -> Self { /* ... */ }
}

// Core config contains ONLY cross-agent concerns
pub struct GemicroConfig {
    pub llm: LlmConfig,  // Shared by all agents
}
```

#### ❌ DON'T: Embed Agent-Specific Config in Core

```rust
// WRONG: Doesn't scale, violates Evergreen philosophy
pub struct GemicroConfig {
    pub llm: LlmConfig,
    pub research: ResearchConfig,        // ❌ Deep Research specific
    pub react: ReactConfig,              // ❌ Would need to add this
    pub reflexion: ReflexionConfig,      // ❌ And this
    pub planning: PlanningConfig,        // ❌ And this...
}
```

#### ✅ DO: Keep AgentContext Minimal

```rust
// CORRECT: AgentContext has only cross-agent shared resources
pub struct AgentContext {
    pub llm: Arc<LlmClient>,  // All agents need LLM access
    // NO agent-specific config here!
}

// Config goes to constructor
let agent = DeepResearchAgent::new(research_config);
let stream = agent.execute(query, context);
```

#### ✅ DO: Gracefully Ignore Unknown Events

```rust
match update.event_type.as_str() {
    "sub_query_completed" => { /* handle */ }
    "final_result" => { /* handle */ }
    _ => {
        log::warn!("Unknown event type: {}", update.event_type);
        // Continue - NOT an error
    }
}
```

### When to Use Rigid Types

Use strong typing for:
- **Error types**: `#[non_exhaustive]` enums with clear error categories
- **Cross-agent configuration**: Settings shared by all agents (like `LlmConfig`)
- **Internal implementation**: Agent internals can use whatever structure they want

### Future Evolution

If we later need more structure, these are easy to add non-breaking:

```rust
// Option 1: Soft-typed agent config registry
pub struct GemicroConfig {
    pub llm: LlmConfig,
    pub agent_configs: HashMap<String, serde_json::Value>,
}

// Option 2: Trait with associated config type (when we implement Agent trait)
trait Agent {
    type Config: Default + DeserializeOwned;
    fn new(config: Self::Config) -> Self;
}
```

## File Structure

```
gemicro/
├── Cargo.toml                    # Workspace manifest
├── CLAUDE.md                     # This file
├── IMPLEMENTATION_PLAN.md        # Detailed implementation roadmap
├── README.md                     # Public documentation
│
├── gemicro-core/                 # Platform-agnostic library
│   ├── src/
│   │   ├── lib.rs                # Public API exports
│   │   ├── update.rs             # Soft-typed AgentUpdate
│   │   ├── error.rs              # Error types (#[non_exhaustive])
│   │   ├── config.rs             # Cross-agent config only
│   │   ├── llm.rs                # LLM client (Phase 2)
│   │   └── agent.rs              # Agent trait (Phase 3)
│
└── gemicro-cli/                  # CLI binary (Phase 4)
    └── src/
        ├── main.rs               # Entry point
        ├── cli.rs                # Argument parsing
        └── display.rs            # Stream consumer with indicatif
```

## Testing Philosophy

- **Unit tests**: Every module has comprehensive tests
- **Doc tests**: Public API examples must compile
- **Integration tests**: Coming in Phase 2+ with actual LLM calls

## Common Patterns

### Adding a New Agent Type

1. Create agent-specific config struct (e.g., `ReactConfig`)
2. Define new event types (e.g., `"react_step_started"`, `"react_action_completed"`)
3. Add helper constructors to `AgentUpdate` (optional but recommended)
4. Implement agent using existing `Agent` trait
5. **NO CHANGES TO CORE TYPES REQUIRED** ✅

### Adding a New Event Type

```rust
// In your agent implementation
impl AgentUpdate {
    pub fn react_step_completed(thought: String, action: String) -> Self {
        Self {
            event_type: "react_step_completed".into(),
            message: format!("Completed ReAct step"),
            timestamp: SystemTime::now(),
            data: json!({ "thought": thought, "action": action }),
        }
    }
}
```

### Consuming Events in CLI

```rust
match update.event_type.as_str() {
    "sub_query_completed" => { /* handle Deep Research event */ }
    "react_step_completed" => { /* handle ReAct event */ }
    _ => {
        // Unknown events are logged but don't crash
        log::debug!("Unknown event type: {}", update.event_type);
    }
}
```

## Dependencies

- **rust-genai**: Git dependency from GitHub (`evansenter/rust-genai`, branch main)
- **tokio**: Async runtime
- **async-stream**: For streaming agent implementations
- **serde/serde_json**: Soft-typed data serialization

## Contributing Guidelines

1. **Follow Evergreen philosophy**: If adding something requires changing core types, rethink the design
2. **Write tests**: All new code needs comprehensive unit tests
3. **Document public APIs**: Use rustdoc for all public items
4. **Keep it simple**: Avoid premature abstraction; solve today's problem simply

## Questions?

See IMPLEMENTATION_PLAN.md for detailed phase-by-phase implementation details.
