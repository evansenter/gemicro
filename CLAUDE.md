# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Gemicro is a CLI agent exploration platform for experimenting with AI agent implementation patterns, powered by the Gemini API via the rust-genai library.

**Key Architecture**: 23-crate workspace with layered dependencies (5 agent crates in `agents/`, 9 tool crates in `tools/`, 5 hook crates in `hooks/`)

**Current Status**: Core implementation complete. Remaining work tracked in [GitHub Issues](https://github.com/evansenter/gemicro/issues).

## Versioning Philosophy

Breaking changes are always permitted, and preferred when they:
- Simplify the API surface
- Remove unnecessary abstractions
- Align with Evergreen principles

Prefer clean breaks over backwards-compatibility shims. Don't add deprecation warnings or migration layers—just make the change.

## Build Commands

```bash
# Run all quality gates before pushing (format, clippy, tests)
make check

# Individual commands
make fmt        # Check formatting
make clippy     # Run clippy with -D warnings
make test       # Run unit + doc tests
make test-all   # Include LLM integration tests (requires GEMINI_API_KEY)
make docs       # Build documentation
make clean      # Clean build artifacts
```

### Direct Cargo Commands

```bash
# Run a single test
cargo test test_name

# Run tests in a specific module
cargo test agent::tests

# Run specific crate tests
cargo test -p gemicro-runner
cargo test -p gemicro-eval

# Run the deep research example
cargo run -p gemicro-deep-research --example deep_research
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
gemicro-core (Agent trait, Tool trait, ToolHook trait, events, LLM - GENERIC ONLY)
    ↓
tools/* (one crate per tool - file_read, web_fetch, task, web_search, glob, grep, file_write, file_edit, bash)
hooks/* (one crate per hook - audit_log, file_security, input_sanitizer, conditional_permission, metrics)
agents/* (one crate per agent - hermetic isolation)
    ↓
gemicro-runner (execution state, metrics, runner)
    ↓
gemicro-eval (datasets, scorers, harness)
gemicro-cli (terminal rendering)
```

## Crate Responsibilities

Each crate has a specific purpose. Before adding code, verify it belongs in that crate.

| Crate | Contains | Does NOT Contain |
|-------|----------|------------------|
| **gemicro-core** | Agent trait, Tool trait, ToolHook trait, HookRegistry, AgentContext, AgentUpdate, ToolRegistry, ToolSet, LlmClient, LlmConfig, errors | Agent/tool/hook implementations, agent-specific configs |
| **tools/*** | One tool per crate (FileRead, WebFetch, Bash, etc.) | Other tools, agent logic, hook logic |
| **hooks/*** | One hook per crate (AuditLog, FileSecurity, Metrics, etc.) | Other hooks, agent logic, tool logic |
| **agents/*** | One agent per crate with its config and events | Other agents, core infrastructure |
| **gemicro-runner** | AgentRunner, AgentRegistry, ExecutionState, metrics | Agent implementations |
| **gemicro-eval** | EvalHarness, Scorers, Datasets | Agent implementations |
| **gemicro-cli** | Terminal UI, REPL, argument parsing, InteractiveConfirmation | Agent implementations |

### Checklist: Before Adding Code

- [ ] Is this a new agent? → Create new `agents/gemicro-{agent-name}` crate
- [ ] Is this a new tool? → Create new `tools/gemicro-{tool-name}` crate
- [ ] Is this a new hook? → Create new `hooks/gemicro-{hook-name}` crate
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

### Import Principles: Single Source of Truth

**NEVER re-export types from other crates for "convenience".** Each type has exactly one canonical home:

| Type | Canonical Import |
|------|------------------|
| `Agent`, `AgentContext`, `AgentUpdate`, `AgentError` | `gemicro_core` |
| `Tool`, `ToolRegistry`, `ToolSet`, `ToolResult`, `ToolError` | `gemicro_core::tool` |
| `ToolHook`, `HookRegistry`, `HookDecision`, `HookError` | `gemicro_core::tool` |
| `ConfirmationHandler`, `AutoApprove`, `AutoDeny`, `GemicroToolService` | `gemicro_core::tool` |
| `Calculator`, `CurrentDatetime` | `gemicro_tool_agent::tools` |
| `FileRead` | `gemicro_file_read` |
| `WebFetch` | `gemicro_web_fetch` |
| `Task` | `gemicro_task` |
| `WebSearch` | `gemicro_web_search` |
| `Glob` | `gemicro_glob` |
| `Grep` | `gemicro_grep` |
| `FileWrite` | `gemicro_file_write` |
| `FileEdit` | `gemicro_file_edit` |
| `Bash` | `gemicro_bash` |
| `AuditLog` | `gemicro_audit_log` |
| `FileSecurity` | `gemicro_file_security` |
| `InputSanitizer` | `gemicro_input_sanitizer` |
| `ConditionalPermission` | `gemicro_conditional_permission` |
| `Metrics` | `gemicro_metrics` |
| `DeepResearchAgent`, `ResearchConfig` | `gemicro_deep_research` |
| `ReactAgent`, `ReactConfig` | `gemicro_react` |
| `LlmJudgeAgent`, `JudgeConfig` | `gemicro_judge` |
| `EvalHarness`, `Scorers` | `gemicro_eval` |

**Why?**
- One source of truth per type (no sync maintenance)
- Clear ownership (where does this type live?)
- No confusion about which import to use

**❌ DON'T: Re-export for convenience**
```rust
// In gemicro-deep-research/src/lib.rs - DON'T DO THIS
pub use gemicro_core::{Agent, AgentContext}; // Creates duplicate paths
```

**✅ DO: Import from canonical source**
```rust
// User code
use gemicro_deep_research::DeepResearchAgent;
use gemicro_core::{Agent, AgentContext}; // Always from core
```

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
    pub llm: Arc<LlmClient>,              // All agents need LLM access
    pub cancellation_token: CancellationToken,  // Cooperative shutdown
    pub tools: Option<Arc<ToolRegistry>>,       // Shared tool registry
    pub confirmation_handler: Option<Arc<dyn ConfirmationHandler>>,  // Tool confirmation
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

### `#[non_exhaustive]` Guidelines

**Always add `#[non_exhaustive]` to:**

| Type Category | Examples | Reason |
|---------------|----------|--------|
| **Error enums** | `AgentError`, `ToolError`, `LlmError` | New error variants are common |
| **Config structs** | `LlmConfig`, `ResearchConfig`, `EvalConfig` | Options grow over time |
| **Public data structs** | `AgentUpdate`, `ToolResult`, `LlmRequest` | Fields may be added |
| **Serialized types** | `Trajectory`, `TrajectoryStep` | Format evolution |
| **Progress/status enums** | `StepStatus`, `EvalProgress` | New states emerge |

**Skip `#[non_exhaustive]` for:**

| Type Category | Examples | Reason |
|---------------|----------|--------|
| **Closed enums** | `ToolSet` (All/None/Specific/Except) | Logically complete set |
| **External mirrors** | `GSM8KSplit` (Test/Train) | Mirrors external dataset |
| **Unit structs** | `FileRead`, `Glob`, `Calculator` | No fields to add |
| **Crate-internal types** | CLI `Args`, `Session` | Not public API |

**Checklist for new public types:**
- [ ] Is this a config struct users construct? → Add `#[non_exhaustive]`
- [ ] Is this an enum that might get new variants? → Add `#[non_exhaustive]`
- [ ] Is this serialized/deserialized? → Add `#[non_exhaustive]`
- [ ] Is this returned from public APIs? → Consider `#[non_exhaustive]`

## Key Files

| File | Purpose |
|------|---------|
| `Cargo.toml` | Workspace manifest with all crate members |
| `docs/AGENT_AUTHORING.md` | Complete guide for implementing new agents |
| `docs/TOOL_AUTHORING.md` | Complete guide for implementing new tools |
| `docs/HOOK_AUTHORING.md` | Complete guide for implementing new hooks |
| `agents/gemicro-simple-qa/` | Reference implementation for new agents |
| `tools/gemicro-file-read/` | Reference implementation for new tools |
| `hooks/gemicro-audit-log/` | Reference implementation for new hooks |
| `gemicro-core/src/agent.rs` | Agent trait, AgentContext, timeout helpers |
| `gemicro-core/src/update.rs` | Soft-typed AgentUpdate |
| `gemicro-core/src/tool/mod.rs` | Tool trait, ToolRegistry, GemicroToolService |
| `gemicro-core/src/tool/hooks.rs` | ToolHook trait, HookRegistry |

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

### Adding a New Tool

See [`docs/TOOL_AUTHORING.md`](docs/TOOL_AUTHORING.md) for a complete walkthrough. Reference implementation: `FileRead` in `tools/gemicro-file-read/src/lib.rs`.

Quick checklist:
1. Create new crate: `tools/gemicro-{tool-name}/`
2. Add to workspace `Cargo.toml` members
3. Implement `Tool` trait (`name`, `description`, `parameters_schema`, `execute`)
4. Implement `requires_confirmation()` and `confirmation_message()` for dangerous tools
5. Add unit tests for all code paths
6. **NO CHANGES TO CORE TYPES REQUIRED** ✅

### Adding a New Hook

See [`docs/HOOK_AUTHORING.md`](docs/HOOK_AUTHORING.md) for a complete walkthrough. Reference implementation: `AuditLog` in `hooks/gemicro-audit-log/src/lib.rs`.

Quick checklist:
1. Create new crate: `hooks/gemicro-{hook-name}/`
2. Add to workspace `Cargo.toml` members
3. Choose struct pattern (unit, config, builder, or stateful)
4. Implement `ToolHook` trait (`pre_tool_use`, `post_tool_use`)
5. Implement `Clone` and `Debug` traits
6. Add `#[non_exhaustive]` to public structs
7. Add unit tests for all decision paths
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

**Contract Enforcement:**

Use `enforce_final_result_contract()` to wrap agent streams with runtime validation:

```rust
use gemicro_core::enforce_final_result_contract;

let stream = agent.execute(query, context);
let validated = enforce_final_result_contract(Box::pin(stream));
// Violations are logged as warnings, events still pass through
```

This is already applied in `AgentRunner` and CLI stream consumption.

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

## rust-genai Integration

**Context:** rust-genai is our library (`evansenter/rust-genai`). We control both sides.

### Separation of Concerns

| Layer | Responsibility |
|-------|----------------|
| **rust-genai** | Gemini API client, function calling, streaming, request/response types |
| **gemicro** | Agent patterns, observability, trajectory recording, tool orchestration |

### When to Wrap rust-genai Types

| gemicro Type | Wraps | Why |
|--------------|-------|-----|
| `LlmRequest` | `InteractionBuilder` params | Serialized in trajectories |
| `LlmClient` | `Client` | Adds recording capability |
| `Tool` trait | `CallableFunction` | Adds metadata, confirmation |
| `GemicroToolService` | `ToolService` | Adds registry, filtering |

Use rust-genai types directly when just passing through (e.g., `InteractionResponse`, `FunctionDeclaration`).

### Error Propagation

```
rust_genai::GenaiError → gemicro_core::LlmError → gemicro_core::AgentError
```

### What NOT to Add to gemicro

- **Alternative LLM backends** - gemicro is Gemini-focused via rust-genai
- **Gemini API wrappers** - that's rust-genai's job
- **Complex workarounds** - fix rust-genai instead

## Tool Confirmation

Tools that perform dangerous operations require user confirmation via `ConfirmationHandler`.

```
ConfirmationHandler (trait)
    ├── AutoApprove              - Always approve (for tests)
    ├── AutoDeny                 - Always deny (safe default)
    └── InteractiveConfirmation  - CLI terminal prompts
```

Tools implement `requires_confirmation()` and `confirmation_message()`. When confirmation is needed, `GemicroToolService` calls the handler; denial returns `ToolError::ConfirmationDenied`.

| Handler | Behavior | Use Case |
|---------|----------|----------|
| `AutoApprove` | Always true | Tests, trusted automation |
| `AutoDeny` | Always false | Safe default |
| `InteractiveConfirmation` | Terminal prompt | CLI applications |

See [`docs/TOOL_AUTHORING.md`](docs/TOOL_AUTHORING.md) for implementation details.

## Hook System

Hooks intercept tool execution for validation, logging, and security without modifying tools.

### Architecture

```
ToolCallableAdapter (enforces hooks)
    ├─ Pre-hooks → Validate/modify/deny execution
    ├─ Confirmation → User approval for dangerous operations
    ├─ Tool::execute() → Actual tool logic
    └─ Post-hooks → Logging, metrics (observability only)
```

Hooks are enforced in `ToolCallableAdapter::call()` - the only interception point for rust-genai's automatic function calling. See `gemicro-core/src/tool/adapter.rs` for rationale.

### Execution Order

```
pre_hook_1 → pre_hook_2 → ... → EXECUTE → post_hook_1 → post_hook_2 → ...
```

- First `Deny` stops chain and prevents execution
- Post-hooks run even if earlier post-hooks fail (logged, not fatal)

### Built-in Hook Crates

| Hook Crate | Purpose |
|------------|---------|
| `gemicro-audit-log` | Log all tool invocations |
| `gemicro-file-security` | Block writes to sensitive paths |
| `gemicro-input-sanitizer` | Enforce input size limits |
| `gemicro-conditional-permission` | Request permission for dangerous operations |
| `gemicro-metrics` | Track tool usage metrics |

### Hook Compatibility

Hooks **only work** with automatic function calling:

| Pattern | Method | Hooks? |
|---------|--------|--------|
| Automatic | `create_with_auto_functions()` | ✅ Yes |
| Automatic streaming | `create_stream_with_auto_functions()` | ✅ Yes |
| Manual | `create()` + loop | ❌ No |

Manual FC bypasses hooks because you handle execution yourself. Use automatic FC for hook enforcement.

See [`docs/HOOK_AUTHORING.md`](docs/HOOK_AUTHORING.md) for implementation details, design guidelines, and examples.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `cargo build` fails with "can't find crate" | Add new crate to `[workspace.members]` in root `Cargo.toml` |
| Integration tests skipped silently | Set `GEMINI_API_KEY` env var; tests use `#[ignore]` |
| `cargo test` passes but `make check` fails | Run `cargo fmt --all` to fix formatting |
| "Unknown event type" warnings in CLI | Expected behavior - consumers ignore unknown events per Evergreen philosophy |
| Tool confirmation prompts hang in tests | Use `AutoApprove` handler in test contexts |
| Clippy errors on CI but not locally | CI runs `cargo clippy -- -D warnings`; run `make clippy` locally |

## ToolSet Reference

String-based tool filtering (replaced the old `ToolType` enum):

```rust
use gemicro_core::ToolSet;

ToolSet::All                              // Use all registered tools (default)
ToolSet::None                             // Use no tools
ToolSet::Specific(vec!["calc".into()])    // Use only named tools
ToolSet::Except(vec!["bash".into()])      // Use all except named tools
```

## Known Limitations & Tracked Issues

See [GitHub Issues](https://github.com/evansenter/gemicro/issues) for the full list.
