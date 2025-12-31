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

**Wrap when gemicro adds observability concerns:**
- Trajectory recording (type must serialize for replay)
- Metadata fields (e.g., `ToolResult.metadata`)
- Richer error categorization for agent-level handling

**Use rust-genai types directly when just passing through.**

| gemicro Type | Wraps | Why |
|--------------|-------|-----|
| `LlmRequest` | `InteractionBuilder` params | Serialized in trajectories |
| `LlmClient` | `Client` | Adds recording capability |
| `Tool` trait | `CallableFunction` | Adds metadata, confirmation, richer errors |
| `GemicroToolService` | `ToolService` | Adds registry, filtering, confirmation |
| `InteractionResponse` | (re-exported) | No additions needed |
| `FunctionDeclaration` | (used directly) | No additions needed |

### When to Request Features (Not Work Around)

Since we control rust-genai, **fix gaps at the source**:

1. **Don't** build complex gemicro abstractions to work around rust-genai limitations
2. **Do** file an issue or implement it in rust-genai
3. Keep gemicro's wrapper layer thin

**Examples:**
- Need structured output? → Added to rust-genai, gemicro just calls it
- Need Google Search grounding? → rust-genai exposes `with_google_search()`, gemicro forwards it
- Need async tool execution? → rust-genai's `CallableFunction::call` is async

### Where Concerns Live

| Concern | Layer | Rationale |
|---------|-------|-----------|
| Gemini API types | rust-genai | Single source of truth for API contract |
| Streaming primitives | rust-genai | `StreamChunk::Delta/Complete` pattern |
| Function calling | rust-genai | `CallableFunction`, `AutoFunctionResult` |
| Agent patterns | gemicro | DeepResearch, ReAct, etc. |
| Trajectory recording | gemicro | Agent-level observability |
| Tool orchestration | gemicro | Registry, filtering, metadata |
| Evaluation harness | gemicro | Datasets, scorers, harness |

### Error Propagation

```
rust_genai::GenaiError
    ↓ wrapped by
gemicro_core::LlmError
    ↓ wrapped by
gemicro_core::AgentError
```

Errors bubble up with context added at each layer. Don't swallow errors.

### Testing Strategy

| Test Type | Location | Runs When |
|-----------|----------|-----------|
| Unit tests | Both libraries | Always (`cargo test`) |
| Doc tests | Both libraries | Always |
| LLM integration | gemicro | `#[ignore]`, needs `GEMINI_API_KEY` |
| API canary tests | rust-genai | CI with secrets |

### What NOT to Add to gemicro

- **Alternative LLM backends** - gemicro is Gemini-focused via rust-genai
- **Gemini API wrappers** - that's rust-genai's job
- **Complex workarounds** - fix rust-genai instead

## Tool Confirmation

Tools that perform potentially dangerous operations (bash commands, file writes) require user confirmation before execution. This is managed through the `ConfirmationHandler` trait.

### Architecture

```
ConfirmationHandler (trait)       - Async confirmation interface
    ├── AutoApprove              - Always approve (for tests/trusted contexts)
    ├── AutoDeny                 - Always deny (safe default)
    └── InteractiveConfirmation  - CLI terminal prompts (dialoguer)

GemicroToolService               - rust-genai ToolService implementation
    ├── ToolRegistry             - Available tools
    ├── ToolSet filter           - Which tools to enable
    └── ConfirmationHandler      - How to confirm dangerous operations
```

### Usage

```rust
use gemicro_core::{AgentContext, ConfirmationHandler, GemicroToolService, ToolRegistry, ToolSet};
use std::sync::Arc;

// 1. Create tool registry with dangerous tools
let mut registry = ToolRegistry::new();
registry.register(Calculator);      // Safe - no confirmation
registry.register(Bash::default()); // Dangerous - requires confirmation

// 2. Create confirmation handler
let handler = Arc::new(InteractiveConfirmation::default());

// 3. Wire into GemicroToolService
let service = GemicroToolService::new(Arc::new(registry))
    .with_filter(ToolSet::All)
    .with_confirmation_handler(Arc::clone(&handler));

// 4. Use with rust-genai
client.interaction()
    .with_model(MODEL)
    .with_tool_service(Arc::new(service))
    .create_with_auto_functions()
    .await?;

// 5. Or wire into AgentContext for agent-managed tools
let context = AgentContext::new(llm)
    .with_confirmation_handler(handler);
```

### Tool Confirmation Protocol

Tools signal confirmation requirements through the `Tool` trait:

```rust
trait Tool {
    fn requires_confirmation(&self, args: &Value) -> bool;
    fn confirmation_message(&self, args: &Value) -> String;
}
```

When a tool requires confirmation:
1. `GemicroToolService` calls `handler.confirm(tool_name, message, args)`
2. If approved → tool executes normally
3. If denied → returns `ToolError::ConfirmationDenied`
4. LLM receives error message and may try alternative approach

### Built-in Handlers

| Handler | Behavior | Use Case |
|---------|----------|----------|
| `AutoApprove` | Always returns true | Tests, trusted automation |
| `AutoDeny` | Always returns false | Safe default when no handler set |
| `InteractiveConfirmation` | Terminal prompt | CLI applications |

## Hook System

The hook system intercepts tool execution for validation, logging, security controls, and custom logic without modifying tools themselves.

### Architecture

```
ToolCallableAdapter (enforces hooks)
    ├─ Pre-hooks → Validate/modify/deny execution
    ├─ Confirmation → User approval for dangerous operations
    ├─ Tool::execute() → Actual tool logic
    └─ Post-hooks → Logging, metrics (observability only)
```

**Critical Design:** Hooks are enforced in `ToolCallableAdapter::call()` because it's the **only interception point** when using rust-genai's automatic function calling (`create_with_auto_functions()` or `create_stream_with_auto_functions()`). The LLM calls `CallableFunction::call()` directly, bypassing `Tool` and `ToolRegistry` abstractions. See `gemicro-core/src/tool/adapter.rs` for detailed rationale.

**Streaming Support:** Hooks work identically in both streaming and non-streaming modes. Use `create_stream_with_auto_functions()` for real-time incremental text updates via `AutoFunctionStreamChunk` while maintaining full hook/confirmation enforcement. See `gemicro-tool-agent/examples/streaming_tool_agent.rs` for a complete example and `gemicro-tool-agent/tests/integration.rs::test_streaming_function_calling_with_hooks` for verification.

### Usage

```rust
use gemicro_audit_log::AuditLog;
use gemicro_file_security::FileSecurity;
use gemicro_metrics::Metrics;
use gemicro_core::tool::{HookRegistry, GemicroToolService, ToolRegistry};
use std::sync::Arc;
use std::path::PathBuf;

// 1. Create metrics first to retain a reference for later access
let metrics = Metrics::new();

// 2. Create hooks registry (clone metrics to share with registry)
let hooks = Arc::new(
    HookRegistry::new()
        .with_hook(AuditLog)  // Log all tool invocations
        .with_hook(FileSecurity::new(vec![
            PathBuf::from("/etc"),
            PathBuf::from("/var"),
        ]))  // Block writes to sensitive paths
        .with_hook(metrics.clone())  // Collect usage metrics
);

// 3. Wire into service
let mut registry = ToolRegistry::new();
// ... register tools ...

let service = GemicroToolService::new(Arc::new(registry))
    .with_hooks(hooks)
    .with_confirmation_handler(Arc::new(AutoApprove));

// 4. Use with rust-genai
// client.interaction()
//     .with_tool_service(Arc::new(service))
//     .create_with_auto_functions()
//     .await?;

// 5. Later: access metrics via the original reference
let snapshot = metrics.snapshot();
```

### Hook Interface

```rust
#[async_trait]
pub trait ToolHook: Send + Sync {
    /// Called before tool execution
    /// Returns: Allow | AllowWithModifiedInput(Value) | Deny { reason }
    async fn pre_tool_use(&self, tool_name: &str, input: &Value)
        -> Result<HookDecision, HookError>;

    /// Called after tool execution (observability only)
    async fn post_tool_use(&self, tool_name: &str, input: &Value, output: &ToolResult)
        -> Result<(), HookError>;
}
```

### Execution Order

Multiple hooks run in registration order:
```
pre_hook_1 → pre_hook_2 → ... → EXECUTE → ... → post_hook_2 → post_hook_1
```

- First `Deny` stops the chain and prevents execution
- First `AllowWithModifiedInput` modifies input for subsequent hooks
- If all return `Allow`, execution proceeds with original input
- Post-hooks run even if earlier post-hooks fail (logged, not fatal)

### Built-in Hook Crates

| Hook Crate | Purpose | Use Case |
|------------|---------|----------|
| `gemicro-audit-log` | Log all tool invocations | Compliance, debugging |
| `gemicro-file-security` | Block writes to sensitive paths | Security policy enforcement |
| `gemicro-input-sanitizer` | Enforce input size limits | Resource protection |
| `gemicro-conditional-permission` | Request permission for dangerous operations | Dynamic security controls |
| `gemicro-metrics` | Track tool usage metrics | Observability |

Each hook is a separate crate in `hooks/` following the same pattern as tools and agents.
See individual crate documentation for usage examples and configuration options.

### Custom Hooks

```rust
use gemicro_core::tool::{ToolHook, HookDecision, HookError, ToolResult};
use async_trait::async_trait;
use serde_json::Value;

#[derive(Debug)]
struct MyCustomHook;

#[async_trait]
impl ToolHook for MyCustomHook {
    async fn pre_tool_use(&self, tool_name: &str, input: &Value)
        -> Result<HookDecision, HookError>
    {
        // Custom validation logic
        if tool_name == "bash" && input["command"].as_str() == Some("rm -rf /") {
            return Ok(HookDecision::Deny {
                reason: "Dangerous command blocked".into()
            });
        }
        Ok(HookDecision::Allow)
    }

    async fn post_tool_use(&self, _: &str, _: &Value, _: &ToolResult)
        -> Result<(), HookError>
    {
        // Logging, metrics, etc.
        Ok(())
    }
}
```

### Hook Design Guidelines

When creating new hooks, follow these patterns for consistency:

#### **Struct Patterns**

| Hook Type | Pattern | Example | Rationale |
|-----------|---------|---------|-----------|
| **Stateless** | Unit struct + Default | `AuditLog` | No configuration needed |
| **Config** | Struct + pub fields + #[non_exhaustive] | `FileSecurity`, `InputSanitizer` | Simple config with field access |
| **Complex Config** | Struct + pub fields + builder + #[non_exhaustive] | `ConditionalPermission` | Multiple optional fields |
| **Stateful** | Struct + private fields + accessors + #[non_exhaustive] | `Metrics` | Runtime mutable state |

#### **Field Visibility Rules**

- **Config hooks**: Use `pub` fields + `#[non_exhaustive]`
  - Allows inspection and debugging
  - `#[non_exhaustive]` prevents struct literals, forces constructor/builder
  - Examples: `FileSecurity { pub blocked_paths }`, `InputSanitizer { pub max_input_size_bytes }`

- **Stateful hooks**: Use private fields + accessors
  - Encapsulates internal mutable state
  - Provides controlled access via methods
  - Example: `Metrics { tools: Arc<RwLock<...>> }` with `snapshot()` method

- **Unit structs**: No fields
  - For hooks with no configuration
  - Just derive `Default`
  - Example: `AuditLog`

#### **Required Traits**

All hooks must implement:
- `ToolHook` (async trait with `pre_tool_use` and `post_tool_use`)
- `Clone` (for sharing across registries)
- `Debug` (for observability)

#### **Cargo.toml Standards**

```toml
[package]
name = "gemicro-<hook-name>"
version.workspace = true     # Always use workspace version
edition.workspace = true     # Always use workspace edition
description = "Brief description"

[dependencies]
gemicro-core = { path = "../../gemicro-core" }
async-trait = { workspace = true }
serde_json = { workspace = true }
# ... hook-specific deps
```

#### **Naming Conventions**

- **Crate**: `gemicro-<kebab-case>`  (e.g., `gemicro-audit-log`)
- **Struct**: `<PascalCase>` (e.g., `AuditLog`)
- **No "Hook" suffix** - the `ToolHook` trait provides type context

#### **When to Use #[non_exhaustive]**

Always add `#[non_exhaustive]` to:
- Public hook structs (even if no fields currently)
- Public snapshot/result structs
- Config builders

This allows adding fields in the future without breaking semver.

### Function Calling Patterns and Hook Compatibility

rust-genai supports multiple function calling patterns. Hooks **only work** with patterns that use `ToolService`:

| Pattern | rust-genai Method | Hook Support | Confirmation Support | Status | Use Case |
|---------|-------------------|--------------|---------------------|--------|----------|
| **Automatic (non-streaming)** | `create_with_auto_functions()` | ✅ Full | ✅ Full | ✅ **Available** | Production agents, CLI tools |
| **Automatic (streaming)** | `create_stream_with_auto_functions()` | ✅ Full | ✅ Full | ✅ **Available** | Real-time UIs, progress updates |
| **Manual (non-streaming)** | `create()` + loop | ❌ No | ❌ No | ✅ Available | Custom control flow, specialized logic |
| **Manual (streaming)** | `create_stream()` + loop | ❌ No | ❌ No | ✅ Available | Custom streaming control |

**Why Manual FC is Incompatible with Hooks:**

Manual function calling gives you raw `FunctionCall` objects and expects you to handle execution yourself:

```rust
// Manual FC pattern (hooks NOT applied)
let response = client.interaction()
    .with_functions(vec![my_function_decl])  // FunctionDeclaration, not ToolService
    .create()
    .await?;

if let Some(calls) = response.function_calls() {
    for call in calls {
        // You write custom execution logic here
        // No Tool trait, no adapter, no hooks
        let result = my_custom_logic(&call.arguments);
    }
}
```

**Key differences:**
- **Automatic FC**: "LLM, here are my tools (ToolService), execute them automatically" → Hooks intercept at `CallableFunction::call()`
- **Manual FC**: "LLM, tell me what you want, I'll handle execution myself" → No tool abstraction, no interception point

**Recommendation:**
- **Use automatic FC** (`create_with_auto_functions()` or `create_stream_with_auto_functions()`) when you want:
  - Automatic hook enforcement (logging, validation, security)
  - Confirmation prompts for dangerous operations
  - The `Tool` trait abstraction for reusable tools
- **Use manual FC** when you:
  - Need raw control over execution flow
  - Have logic that doesn't fit the `Tool` trait
  - Want to bypass all abstractions for specialized cases

**Can you apply hooks in manual FC?** Technically yes, but it defeats the purpose:

```rust
// Possible but NOT recommended
if let Some(calls) = response.function_calls() {
    for call in calls {
        // You could manually get the tool and apply hooks...
        let tool = registry.get(&call.name)?;
        let adapter = ToolCallableAdapter::new(tool)
            .with_hooks(hooks);
        let result = adapter.call(&call.arguments).await?;
        // But if you're doing this, just use automatic FC!
    }
}
```

This is **opt-in** (not enforced) and if you're using the `Tool` trait anyway, you should use automatic FC. Manual FC is for users who explicitly want to bypass abstractions.

**Testing:** See `gemicro-tool-agent/tests/integration.rs`:
- `test_streaming_function_calling_with_hooks()` - Verifies streaming automatic FC with hooks and confirmation
- `test_tool_agent_calculator()` - Verifies non-streaming automatic FC
- Both streaming and non-streaming automatic FC are fully tested and working

**Example:** `gemicro-tool-agent/examples/streaming_tool_agent.rs` demonstrates streaming FC with hooks, showing real-time text updates via `AutoFunctionStreamChunk` while maintaining full hook enforcement.

### Design Trade-offs

**Why hooks live in the adapter:**
- rust-genai calls `CallableFunction::call()` directly
- No other interception point exists for auto-function calling
- Tools stay simple, adapter handles cross-cutting concerns

**Trade-off:**
- Direct calls to `tool.execute()` bypass hooks
- Acceptable because direct calls are for testing/manual use
- LLM function calling (primary use case) always goes through adapter

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
