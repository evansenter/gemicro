# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Gemicro is a CLI agent exploration platform for experimenting with AI agent implementation patterns, powered by the Gemini API via the rust-genai library.

**Key Architecture**: Four-crate workspace with layered dependencies

**Current Status**: Core implementation complete. Remaining work tracked in [GitHub Issues](https://github.com/evansenter/gemicro/issues).

## Build Commands

```bash
# Build entire workspace
cargo build --workspace

# Run tests (unit + doc tests)
cargo test --workspace

# Run ALL tests including LLM integration tests (requires GEMINI_API_KEY)
cargo test --workspace -- --include-ignored

# Run a single test
cargo test test_name

# Run tests in a specific module
cargo test agent::tests

# Linting
cargo clippy --workspace -- -D warnings

# Format check
cargo fmt --all -- --check

# Build docs
cargo doc --workspace --no-deps

# Run the deep research example
cargo run -p gemicro-core --example deep_research

# Run specific crate tests
cargo test -p gemicro-runner
cargo test -p gemicro-eval
```

## Environment Setup

```bash
export GEMINI_API_KEY="your-api-key"  # Required for integration tests and examples
```

## CLI Quick Reference

```bash
# Single query mode
gemicro "What are the latest developments in quantum computing?"

# Interactive REPL mode
gemicro --interactive

# With Google Search grounding for real-time web data
gemicro "What AI news happened this week?" --google-search

# Custom configuration
gemicro "Compare async runtimes" --min-sub-queries 3 --max-sub-queries 7 --timeout 120
```

REPL commands: `/help`, `/agent [name]`, `/history`, `/clear`, `/reload`, `/quit`

## Crate Layers

```text
gemicro-core (agents, events, LLM)
    ↓
gemicro-runner (execution state, metrics, runner)
    ↓
gemicro-eval (datasets, scorers, harness)
gemicro-cli (terminal rendering)
```

| Crate | Purpose |
|-------|---------|
| **gemicro-core** | Platform-agnostic library: Agent trait, AgentUpdate events, LlmClient, conversation history. Agents: DeepResearchAgent, ReActAgent, SimpleQaAgent |
| **gemicro-runner** | Headless execution runtime: ExecutionState, AgentRunner, AgentRegistry, metrics collection |
| **gemicro-eval** | Evaluation framework: HotpotQA/custom datasets, scorers (Contains, LLM Judge), LlmJudgeAgent |
| **gemicro-cli** | Terminal UI: indicatif progress display, rustyline REPL, markdown rendering |

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

## File Structure

```
gemicro/
├── Cargo.toml                    # Workspace manifest
├── CLAUDE.md                     # This file
├── README.md                     # Public documentation
├── docs/AGENT_AUTHORING.md       # Guide for implementing new agents
│
├── gemicro-core/                 # Platform-agnostic library
│   ├── src/
│   │   ├── lib.rs                # Public API exports
│   │   ├── agent/                # Agent implementations
│   │   │   ├── mod.rs            # Agent trait, AgentContext, helpers
│   │   │   ├── deep_research.rs  # DeepResearchAgent (decompose → parallel → synthesize)
│   │   │   ├── react.rs          # ReActAgent (Thought → Action → Observation loops)
│   │   │   └── simple_qa.rs      # SimpleQaAgent (reference implementation)
│   │   ├── update.rs             # Soft-typed AgentUpdate
│   │   ├── error.rs              # Error types (#[non_exhaustive])
│   │   ├── config.rs             # LlmConfig + agent-specific configs
│   │   ├── llm.rs                # LLM client (buffered + streaming)
│   │   ├── history.rs            # ConversationHistory, HistoryEntry
│   │   └── utils.rs              # Shared utilities (truncation, etc.)
│   ├── tests/
│   │   ├── llm_integration.rs    # LLM integration tests (#[ignore])
│   │   ├── agent_integration.rs  # Agent integration tests (#[ignore])
│   │   └── common/mod.rs         # Shared test helpers
│   └── examples/
│       └── deep_research.rs      # Full agent example with progress display
│
├── gemicro-runner/               # Headless execution runtime
│   └── src/
│       ├── lib.rs
│       ├── state.rs              # ExecutionState, Phase tracking
│       ├── metrics.rs            # ExecutionMetrics, SubQueryTiming
│       ├── runner.rs             # AgentRunner for headless execution
│       ├── registry.rs           # AgentRegistry, AgentFactory
│       └── utils.rs              # Formatting utilities
│
├── gemicro-eval/                 # Evaluation framework
│   └── src/
│       ├── lib.rs
│       ├── dataset.rs            # HotpotQA, JsonFileDataset
│       ├── scorer.rs             # Contains, LlmJudgeScorer
│       ├── harness.rs            # EvalHarness, EvalConfig
│       ├── results.rs            # EvalSummary, EvalResult
│       └── judge.rs              # LlmJudgeAgent for evaluation
│
└── gemicro-cli/                  # Terminal UI binary
    └── src/
        ├── main.rs               # Entry point, stream orchestration
        ├── cli.rs                # Clap argument parsing, OutputConfig
        ├── format.rs             # Text utilities, markdown rendering
        ├── error.rs              # CLI-specific error handling
        ├── display/              # State-renderer pattern
        │   ├── mod.rs
        │   ├── renderer.rs       # Renderer trait for swappable backends
        │   └── indicatif.rs      # IndicatifRenderer implementation
        └── repl/                 # Interactive REPL mode
            ├── mod.rs
            ├── session.rs        # REPL session management
            └── commands.rs       # /help, /agent, /history, etc.
```

## Key Architectural Decisions

1. **Streaming-first**: `DeepResearchAgent::execute()` returns `impl Stream<Item = Result<AgentUpdate, AgentError>>` for real-time observability

2. **Parallel execution with mpsc**: Sub-queries spawn via `tokio::spawn`, results stream through `mpsc::channel` as they complete (non-deterministic order)

3. **Timeout enforcement**: Uses `tokio::time::timeout` wrapping each phase (decompose, execute, synthesize) with remaining time calculation

4. **Graceful partial failure**: `continue_on_partial_failure` config controls whether to abort on first error or continue with partial results

## Testing Philosophy

- **Unit tests**: In-module `#[cfg(test)]` blocks for fast feedback
- **Doc tests**: Public API examples must compile (`cargo test --doc`)
- **Integration tests**: Marked `#[ignore]`, require `GEMINI_API_KEY`, run with `--include-ignored`
- **Shared test helpers**: `tests/common/mod.rs` provides `setup_test_context()`

## Common Patterns

### Adding a New Agent Type

See [`docs/AGENT_AUTHORING.md`](docs/AGENT_AUTHORING.md) for a complete walkthrough. Reference implementation: `SimpleQaAgent` in `gemicro-core/src/agent/simple_qa.rs`.

Quick checklist:
1. Create agent-specific config struct (e.g., `ReactConfig`) with `validate()` method
2. Define event types as strings (e.g., `"react_step"`) - no exports needed
3. Implement `Agent` trait using `async_stream::try_stream!`
4. Handle timeouts via `remaining_time()` and `with_timeout_and_cancellation()`
5. Add unit tests for config, integration tests (`#[ignore]`) for execution
6. Export agent struct and config from `mod.rs` and `lib.rs`
7. **NO CHANGES TO CORE TYPES REQUIRED** ✅

### Adding a New Event Type

Events are soft-typed strings - no exports or core changes needed. Use `AgentUpdate::custom()`:

```rust
// In your agent implementation - just use string literals
yield Ok(AgentUpdate::custom(
    "react_step_completed",
    "Completed ReAct step",
    json!({ "thought": thought, "action": action })
));

// Or define internal constants for consistency (NOT exported):
const EVENT_MY_STEP: &str = "my_step";
yield Ok(AgentUpdate::custom(EVENT_MY_STEP, "Step complete", json!({})));
```

### Standard Events

For interoperability, all agents should emit `final_result` when complete:

| Event Type | Purpose | Emitted By |
|------------|---------|------------|
| `final_result` | Signals completion with answer | All agents (required) |
| `decomposition_started` | Query decomposition begins | DeepResearchAgent |
| `decomposition_complete` | Sub-queries determined | DeepResearchAgent |
| `sub_query_started` | Individual query starts | DeepResearchAgent |
| `sub_query_completed` | Individual query finishes | DeepResearchAgent |
| `synthesis_started` | Result synthesis begins | DeepResearchAgent |
| `react_started` | ReAct loop begins | ReactAgent |
| `react_thought` | Agent reasoning | ReactAgent |
| `react_action` | Tool invocation | ReactAgent |
| `react_observation` | Tool result | ReactAgent |
| `react_complete` | Answer found | ReactAgent |
| `react_max_iterations` | Max iterations reached | ReactAgent |
| `simple_qa_started` | Query starts | SimpleQaAgent |
| `simple_qa_result` | Response received | SimpleQaAgent |
| `tool_agent_started` | Tool agent starts | ToolAgent |
| `tool_agent_complete` | Tool agent finishes | ToolAgent |

Consumers must handle unknown event types gracefully (log and ignore).

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

## Known Limitations & Tracked Issues

See [GitHub Issues](https://github.com/evansenter/gemicro/issues) for the full list.
