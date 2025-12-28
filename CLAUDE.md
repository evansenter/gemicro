# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Gemicro is a CLI agent exploration platform for experimenting with AI agent implementation patterns, powered by the Gemini API via the rust-genai library.

**Key Architecture**: Nine-crate workspace with layered dependencies (5 agent crates in `agents/` subdirectory)

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
cargo run -p gemicro-deep-research --example deep_research

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
gemicro-core (Agent trait, events, LLM - GENERIC ONLY)
    ↓
agents/* (one crate per agent - hermetic isolation)
    ↓
gemicro-runner (execution state, metrics, runner)
    ↓
gemicro-eval (datasets, scorers, harness)
gemicro-cli (terminal rendering)
```

| Crate | Purpose |
|-------|---------|
| **gemicro-core** | Platform-agnostic library: Agent trait, AgentContext, AgentUpdate events, LlmClient, conversation history. **No agent implementations.** |
| **agents/gemicro-deep-research** | DeepResearchAgent: query decomposition, parallel sub-query execution, synthesis |
| **agents/gemicro-react** | ReactAgent: Thought → Action → Observation reasoning loops |
| **agents/gemicro-simple-qa** | SimpleQaAgent: minimal reference implementation |
| **agents/gemicro-tool-agent** | ToolAgent: native function calling with calculator/datetime tools |
| **agents/gemicro-judge** | LlmJudgeAgent: LLM-based evaluation scoring |
| **gemicro-runner** | Headless execution runtime: ExecutionState, AgentRunner, AgentRegistry, metrics collection |
| **gemicro-eval** | Evaluation framework: HotpotQA/custom datasets, scorers (Contains, LLM Judge) |
| **gemicro-cli** | Terminal UI: indicatif progress display, rustyline REPL, markdown rendering |

## Crate Responsibilities

Each crate has a specific purpose. Before adding code, verify it belongs in that crate.

| Crate | Contains | Does NOT Contain |
|-------|----------|------------------|
| **gemicro-core** | Agent trait, AgentContext, AgentUpdate, LlmClient, LlmConfig, errors, utilities | Agent implementations, agent-specific configs |
| **agents/gemicro-deep-research** | DeepResearchAgent, ResearchConfig, DeepResearchEventExt | Other agents, core infrastructure |
| **agents/gemicro-react** | ReactAgent, ReactConfig | Other agents, core infrastructure |
| **agents/gemicro-simple-qa** | SimpleQaAgent, SimpleQaConfig | Other agents, core infrastructure |
| **agents/gemicro-tool-agent** | ToolAgent, ToolAgentConfig, ToolType | Other agents, core infrastructure |
| **agents/gemicro-judge** | LlmJudgeAgent, JudgeConfig | Other agents, core infrastructure |
| **gemicro-runner** | AgentRunner, AgentRegistry, generic execution infrastructure | Agent implementations |
| **gemicro-eval** | EvalHarness, Scorers, Datasets | Agent implementations |
| **gemicro-cli** | Terminal UI, REPL, argument parsing | Agent implementations |

### Checklist: Before Adding Code

- [ ] Is this a new agent? → Create new `agents/gemicro-{agent-name}` crate
- [ ] Is this agent-specific config/events? → Put in the agent's crate
- [ ] Is this cross-agent infrastructure? → gemicro-core
- [ ] Is this evaluation-specific? → gemicro-eval
- [ ] Is this CLI/rendering? → gemicro-cli
- [ ] Is this execution tracking? → gemicro-runner (keep generic)

### Agent Crate Independence

Each agent crate:
- Depends ONLY on gemicro-core (never on other agent crates)
- Contains its own config, prompts, event accessors
- Has its own tests and examples
- Can be versioned and released independently

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
| **No Agent/Dataset Leakage** | Agent-specific functionality (constructors, accessors, types, configs) belongs in agent modules, NOT in core. Dataset-specific logic belongs in eval, NOT in core |

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

// Core only provides two constructors: custom() and final_result()
impl AgentUpdate {
    // Universal constructor - use for ALL agent-specific events
    // Uses impl Into<String> for ergonomic &str or String arguments
    pub fn custom(
        event_type: impl Into<String>,
        message: impl Into<String>,
        data: Value,
    ) -> Self { /* ... */ }

    // Required completion signal - the ONLY cross-agent constructor
    pub fn final_result(answer: String, metadata: ResultMetadata) -> Self { /* ... */ }
}

// Agents define their own constants locally (NOT exported from core):
const EVENT_MY_STEP: &str = "my_step";  // in agent/my_agent.rs
yield Ok(AgentUpdate::custom(EVENT_MY_STEP, "Step complete", json!({})));
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
├── agents/                       # Agent crates (one per agent, hermetic)
│   ├── gemicro-deep-research/    # DeepResearchAgent
│   │   ├── src/
│   │   │   ├── lib.rs            # Public exports
│   │   │   ├── agent.rs          # Agent implementation
│   │   │   ├── config.rs         # ResearchConfig, ResearchPrompts
│   │   │   └── events.rs         # DeepResearchEventExt, SubQueryResult
│   │   └── tests/integration.rs
│   ├── gemicro-react/            # ReactAgent
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── agent.rs
│   │   │   └── config.rs         # ReactConfig, ReactPrompts
│   │   └── tests/integration.rs
│   ├── gemicro-simple-qa/        # SimpleQaAgent (reference impl)
│   │   ├── src/lib.rs
│   │   └── tests/integration.rs
│   ├── gemicro-tool-agent/       # ToolAgent (function calling)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── agent.rs
│   │   │   ├── config.rs
│   │   │   └── tools.rs          # calculator(), current_datetime()
│   │   └── tests/integration.rs
│   └── gemicro-judge/            # LlmJudgeAgent (evaluation)
│       └── src/lib.rs
│
├── gemicro-core/                 # Platform-agnostic library (GENERIC ONLY)
│   ├── src/
│   │   ├── lib.rs                # Public API exports
│   │   ├── agent.rs              # Agent trait, AgentContext, helpers (NO implementations)
│   │   ├── update.rs             # Soft-typed AgentUpdate
│   │   ├── error.rs              # Error types (#[non_exhaustive])
│   │   ├── config.rs             # LlmConfig only (no agent configs)
│   │   ├── llm.rs                # LLM client (buffered + streaming)
│   │   ├── history.rs            # ConversationHistory, HistoryEntry
│   │   └── utils.rs              # Shared utilities (truncation, etc.)
│   └── tests/
│       ├── llm_integration.rs    # LLM integration tests (#[ignore])
│       └── common/mod.rs         # Shared test helpers
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
│       └── results.rs            # EvalSummary, EvalResult
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

See [`docs/AGENT_AUTHORING.md`](docs/AGENT_AUTHORING.md) for a complete walkthrough. Reference implementation: `SimpleQaAgent` in `agents/gemicro-simple-qa/src/lib.rs`.

Quick checklist:
1. Create new crate: `agents/gemicro-{agent-name}/`
2. Add to workspace `Cargo.toml` members
3. Create agent-specific config struct with `validate()` method
4. Define event types as strings (e.g., `"react_step"`) - no exports needed
5. Implement `Agent` trait using `async_stream::try_stream!`
6. Handle timeouts via `remaining_time()` and `with_timeout_and_cancellation()`
7. Add unit tests for config, integration tests (`#[ignore]`) for execution
8. **NO CHANGES TO CORE TYPES REQUIRED** ✅

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

| Event Type | Purpose | Emitted By | Required |
|------------|---------|------------|----------|
| `final_result` | Signals completion with answer | All agents | **Yes** |
| `decomposition_started` | Query decomposition begins | DeepResearchAgent | No |
| `decomposition_complete` | Sub-queries determined | DeepResearchAgent | No |
| `sub_query_started` | Individual query starts | DeepResearchAgent | No |
| `sub_query_completed` | Individual query finishes | DeepResearchAgent | No |
| `sub_query_failed` | Individual query fails | DeepResearchAgent | No |
| `synthesis_started` | Result synthesis begins | DeepResearchAgent | No |
| `react_started` | ReAct loop begins | ReactAgent | No |
| `react_thought` | Agent reasoning | ReactAgent | No |
| `react_action` | Tool invocation | ReactAgent | No |
| `react_observation` | Tool result | ReactAgent | No |
| `react_complete` | Answer found | ReactAgent | No |
| `react_max_iterations` | Max iterations reached | ReactAgent | No |
| `simple_qa_started` | Query starts | SimpleQaAgent | No |
| `simple_qa_result` | Response received | SimpleQaAgent | No |
| `tool_agent_started` | Tool agent starts | ToolAgent | No |
| `tool_agent_complete` | Tool agent finishes | ToolAgent | No |

**Event Contract:**
- `final_result` **MUST** be the last event emitted by any agent
- All other events are informational/observability only
- Consumers **MUST** gracefully ignore unknown event types (log and continue)
- Event data schemas are soft-typed (JSON) and may evolve

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
