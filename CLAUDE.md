# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Design Philosophy: LLM-First

**Don't build code to do what the LLM already does well.** Give it context instead of building infrastructure.

| Principle | Meaning |
|-----------|---------|
| **LLM-First** | Trust the model; don't over-engineer |
| **Explicit Over Implicit** | No magical defaults; 3 lines of clear code beats hidden behavior |
| **Soft-Typed Events** | `event_type: String` + `data: JSON`, not rigid enums |
| **Graceful Unknowns** | Unknown event types logged, not errors |
| **Agent Isolation** | Agents depend only on core, never on each other |
| **Config at Construction** | Agent-specific config in constructors, not shared context |

**Corollary**: Breaking changes are preferred over backwards-compatibility shims. Clean breaks, no deprecation warnings.

## Repository Overview

Gemicro is a CLI agent exploration platform for AI agent patterns, powered by Gemini API via rust-genai.

**Architecture**: 26-crate workspace (7 agents, 10 tools, 5 hooks, 4 core)

**Status**: Core complete. Remaining work in [GitHub Issues](https://github.com/evansenter/gemicro/issues).

## Build Commands

```bash
make check      # Format + clippy + tests (run before pushing)
make fmt        # Check formatting
make clippy     # Clippy with -D warnings
make test       # Unit + doc tests
make test-all   # Include LLM integration tests (requires GEMINI_API_KEY)
```

```bash
cargo test test_name                              # Single test
cargo test -p gemicro-runner                      # Specific crate
cargo run -p gemicro-deep-research --example deep_research  # Run example
```

## Environment

```bash
export GEMINI_API_KEY="your-api-key"  # Required for integration tests

# Debug rust-genai HTTP traffic
LOUD_WIRE=1 cargo run -p gemicro-developer --example developer
```

For tool execution debugging, use `gemicro-audit-log` (structured logging without HTTP noise).

## Crate Architecture

```
gemicro-core (Agent, Tool, Interceptor traits, LLM client - GENERIC ONLY)
    ↓
tools/* (one crate per tool)
hooks/* (one crate per hook)
agents/* (one crate per agent - hermetic isolation)
    ↓
gemicro-runner (execution, metrics)
    ↓
gemicro-eval (datasets, scorers)
gemicro-cli (terminal UI)
```

### Crate Boundaries

| Crate | Contains | Does NOT Contain |
|-------|----------|------------------|
| **gemicro-core** | Traits (Agent, Tool, Interceptor), AgentContext, LlmClient, errors | Implementations |
| **tools/*** | One tool per crate | Other tools, agent logic |
| **hooks/*** | One hook per crate | Other hooks, agent logic |
| **agents/*** | One agent + its config/events | Other agents, core infra |
| **gemicro-runner** | AgentRunner, ExecutionState | Agent implementations |
| **gemicro-eval** | EvalHarness, Scorers | Agent implementations |

### Before Adding Code

- New agent? → `agents/gemicro-{name}/`
- New tool? → `tools/gemicro-{name}/`
- New hook? → `hooks/gemicro-{name}/`
- Cross-agent infrastructure? → `gemicro-core`
- NO CHANGES TO CORE TYPES for new agents/tools/hooks

## Key Patterns

### Soft-Typed Events (AgentUpdate)

```rust
// Use AgentUpdate::custom() for agent-specific events
yield Ok(AgentUpdate::custom("my_step", "Step complete", json!({})));

// AgentUpdate::final_result() is the ONLY required event (signals completion)
yield Ok(AgentUpdate::final_result(answer, metadata));
```

Unknown event types must be logged and ignored, not treated as errors.

### Agent-Specific Config

Config belongs in agent constructors, not shared context:

```rust
let agent = DeepResearchAgent::new(research_config);  // Config here
let stream = agent.execute(query, context);            // Context is minimal
```

### Adding New Agents/Tools/Hooks

See `docs/AGENT_AUTHORING.md`, `docs/TOOL_AUTHORING.md`, `docs/INTERCEPTOR_AUTHORING.md`.

Reference implementations:
- Agent: `agents/gemicro-prompt-agent/`
- Tool: `tools/gemicro-file-read/`
- Hook: `hooks/gemicro-audit-log/`

## Key Architectural Decisions

1. **Streaming-first**: `execute()` returns `impl Stream<Item = Result<AgentUpdate>>` for real-time observability
2. **Parallel sub-queries**: Spawn via `tokio::spawn`, results stream through `mpsc::channel`
3. **Timeout enforcement**: `tokio::time::timeout` per phase with remaining time calculation
4. **Partial failure**: `continue_on_partial_failure` config controls abort vs continue

## Testing

- **Unit tests**: In-module `#[cfg(test)]` blocks
- **Doc tests**: Public API examples must compile
- **Integration tests**: `#[ignore]`, require `GEMINI_API_KEY`, run with `--include-ignored`
- **Test helpers**: Each crate has `tests/common/mod.rs` with `create_test_context()`

## `#[non_exhaustive]` Guidelines

**Add to**: Error enums, config structs, public data structs, serialized types

**Skip for**: Closed enums (`ToolSet`), unit structs, crate-internal types

## Dependencies

- **rust-genai**: Git dependency (`evansenter/rust-genai`, main branch)
- **tokio**: Async runtime
- **async-stream**: Streaming agent implementations

## rust-genai Integration

| Layer | Responsibility |
|-------|----------------|
| **rust-genai** | Gemini API client, function calling, streaming |
| **gemicro** | Agent patterns, observability, tool orchestration |

Use rust-genai types directly when passing through. Wrap when adding functionality (recording, metadata).

**Don't add to gemicro**: Alternative LLM backends, Gemini API wrappers, complex workarounds (fix rust-genai instead).

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "can't find crate" | Add to `[workspace.members]` in root Cargo.toml |
| Integration tests skipped | Set `GEMINI_API_KEY`; tests use `#[ignore]` |
| `make check` fails but `cargo test` passes | Run `cargo fmt --all` |
| "Unknown event type" warnings | Expected - consumers ignore unknowns |
| Tool confirmation hangs in tests | Use `AutoApprove` handler |

## README Maintenance

When adding cross-cutting features (observability, error handling, security), update the "Cross-Cutting Concerns" table in `README.md`.
