# Gemicro

**CLI agent exploration platform for experimenting with AI agent implementation patterns**

Gemicro allows you to explore and interact with different AI agent patterns through a command-line interface, powered by the Gemini API via the [rust-genai](https://github.com/evansenter/rust-genai) library.

## Features

- 🔬 **Deep Research Agent**: Query decomposition with parallel sub-query execution and synthesis
- 🔄 **ReAct Agent**: Reasoning and Acting loops with Thought → Action → Observation cycles
- 🛠️ **Tool Agent**: Native function calling with built-in tools (calculator, datetime, file operations)
- 🎯 **Simple QA Agent**: Minimal reference implementation for agent authoring
- 📊 **Real-time Observability**: Streaming updates show agent execution as it happens
- 🏗️ **Extensible Architecture**: Soft-typed events allow adding new agent types without protocol changes
- 📱 **Platform-Agnostic Core**: Library designed for future mobile and web support
- ⚡ **Parallel Execution**: Sub-queries fan out for faster results
- 🌐 **Google Search Grounding**: Enable real-time web search for current events and live data
- 🔐 **Tool Confirmation**: Interactive approval for dangerous operations (bash, file writes)
- 📈 **Evaluation Framework**: HotpotQA datasets, scorers (Contains, LLM Judge), and evaluation harness

## Architecture

### 23-Crate Workspace

```
gemicro-core (Agent trait, Tool trait, ToolHook trait, events, LLM - GENERIC ONLY)
    ↓
tools/* (9 tool crates - file_read, web_fetch, task, web_search, glob, grep, file_write, file_edit, bash)
hooks/* (5 hook crates - audit_log, file_security, input_sanitizer, conditional_permission, metrics)
agents/* (5 agent crates - hermetic isolation)
    ↓
gemicro-runner (execution state, metrics, runner)
    ↓
gemicro-eval (datasets, scorers, harness)
gemicro-cli (terminal rendering)
```

| Layer | Crates |
|-------|--------|
| **gemicro-core** | Agent/Tool/ToolHook traits, AgentContext, AgentUpdate events, LlmClient. **No implementations.** |
| **tools/** | file_read, web_fetch, task, web_search, glob, grep, file_write, file_edit, bash |
| **hooks/** | audit_log, file_security, input_sanitizer, conditional_permission, metrics |
| **agents/** | deep_research, react, simple_qa, tool_agent, judge |
| **gemicro-runner** | Headless execution: AgentRunner, AgentRegistry, ExecutionState, metrics |
| **gemicro-eval** | Evaluation: HotpotQA/custom datasets, scorers (Contains, LLM Judge) |
| **gemicro-cli** | Terminal UI: indicatif progress, rustyline REPL, markdown rendering |

### Design Philosophy

- **Streaming-first**: Agents return async streams of updates for real-time observability
- **Soft-typed events**: Following [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) philosophy for maximum extensibility
- **Interactions API**: Uses Gemini's unified Interactions API exclusively
- **Single model**: Hardcoded to `gemini-3-flash-preview` for consistency
- **Single source of truth**: Each type has one canonical crate—no convenience re-exports
- **Breaking changes welcome**: Simplicity over backwards compatibility

### Imports

Each type lives in exactly one crate. Import from the canonical source:

```rust
use gemicro_core::{Agent, AgentContext, AgentUpdate};     // Core types
use gemicro_deep_research::{DeepResearchAgent, ResearchConfig}; // Agent + config
use gemicro_judge::LlmJudgeAgent;                          // Judge agent
```

## Quick Start

```bash
# Set your API key
export GEMINI_API_KEY="your-api-key"

# Single query mode
gemicro "What are the latest developments in quantum computing?"

# Interactive REPL mode
gemicro --interactive
```

## Usage

### Single Query Mode

Run a single research query with real-time streaming output:

```bash
# Basic query
gemicro "What is Rust?"

# With custom configuration
gemicro "Compare async runtimes" \
    --min-sub-queries 3 \
    --max-sub-queries 7 \
    --timeout 120

# With Google Search grounding for real-time web data
gemicro "What are the latest AI developments this week?" --google-search

# Verbose mode (debug logging)
gemicro "Your query" --verbose
```

### Interactive REPL Mode

Start an interactive session for multiple queries with conversation context:

```bash
gemicro --interactive
# or
gemicro -i
```

**REPL Commands:**

| Command | Alias | Description |
|---------|-------|-------------|
| `/help` | `/?` | Show help message |
| `/agent [name]` | `/a` | Switch agent or list available agents |
| `/history` | `/h` | Show conversation history |
| `/clear` | | Clear conversation history |
| `/reload` | `/r` | Hot-reload agents (placeholder) |
| `/quit` | `/q`, `/exit` | Exit the REPL |

**Example session:**

```
gemicro REPL - Type /help for commands, /quit to exit

[deep_research] > What is Rust?
⠋ Decomposing query...
✓ Generated 4 sub-queries
⠋ Executing sub-queries...
...

[deep_research] > /agent
Available agents:
  deep_research * - Decomposes queries into sub-questions...

[deep_research] > /quit
Goodbye!
```

### CLI Options

```bash
gemicro [OPTIONS] [QUERY]

Arguments:
  [QUERY]  Research query (required unless using --interactive)

Options:
  -i, --interactive            Interactive REPL mode
      --api-key <KEY>          Gemini API key (or GEMINI_API_KEY env var)
      --min-sub-queries <N>    Minimum sub-queries [default: 3]
      --max-sub-queries <N>    Maximum sub-queries [default: 5]
      --max-concurrent <N>     Max parallel executions [default: 5]
      --timeout <SECS>         Total timeout [default: 180]
      --continue-on-failure    Continue if some sub-queries fail
      --llm-timeout <SECS>     Per-request timeout [default: 60]
      --max-tokens <N>         Max tokens per LLM request [default: 16384]
      --temperature <F>        Temperature 0.0-1.0 [default: 0.7]
      --google-search          Enable Google Search grounding
      --plain                  Plain text output (no markdown)
  -v, --verbose                Enable debug logging
  -h, --help                   Print help
  -V, --version                Print version
```

## Development

### Prerequisites

```bash
# Clone and build
git clone https://github.com/evansenter/gemicro.git
cd gemicro
cargo build --workspace
```

### Building and Testing

```bash
# Build workspace
cargo build --workspace

# Run all quality gates (format, clippy, tests)
make check

# Individual quality gates
make fmt        # Check formatting
make clippy     # Run clippy with -D warnings
make test       # Run unit + doc tests
make test-all   # Include LLM integration tests (requires GEMINI_API_KEY)
```

### Running Examples

```bash
# Deep research example (non-interactive)
cargo run -p gemicro-deep-research --example deep_research

# A/B comparison example (requires GEMINI_API_KEY)
cargo run -p gemicro-eval --example ab_comparison

# REPL demo script
./examples/repl_demo.sh
```

### Evaluation

Run evaluations against HotpotQA, GSM8K, or custom datasets:

```bash
# Basic evaluation with default scorers
gemicro-eval --dataset hotpotqa --sample 10

# With specific scorer
gemicro-eval --dataset gsm8k --scorer contains --sample 50

# Full evaluation with LLM judge
gemicro-eval --dataset hotpotqa --scorer contains,llm_judge --agent react
```

**Scorer Cost Considerations**

The `llm_judge` scorer makes an LLM API call for each evaluation:

| Scorer | Cost | Speed | Use Case |
|--------|------|-------|----------|
| `contains` | Free | Instant | Quick iterations, substring matching |
| `llm_judge` | API tokens | ~500ms-2s per call | Semantic accuracy, final evaluation |

For a 100-question dataset with `--scorer llm_judge`:
- 100 agent executions + 100 judge calls = **200 total LLM calls**

**Recommendation**: Use `--scorer contains` for rapid iteration during development, then add `llm_judge` for final accuracy assessment.

## Future Exploration Areas

See [GitHub Issues](https://github.com/evansenter/gemicro/issues) for the full roadmap. Key areas include:

- Additional agent patterns (Reflexion, Plan-and-Execute)
- Model Context Protocol (MCP) client support
- Hot-reload for agent development (`/reload --watch`)
- Persistent sessions across restarts
- Tab completion for REPL commands and agent names
- Performance benchmarks with criterion.rs

## License

MIT

## Security

See [SECURITY.md](SECURITY.md) for security policy, vulnerability reporting, and best practices.

## Trajectory Recording

Gemicro supports capturing full LLM interaction traces for offline replay and evaluation:

```rust
use gemicro_runner::AgentRunner;
use gemicro_core::{Trajectory, MockLlmClient};

// Record a trajectory during agent execution
let runner = AgentRunner::new();
let (metrics, trajectory) = runner.execute_with_trajectory(
    &agent, "What is Rust?", json!({}), genai_client, llm_config
).await?;

// Save for later
trajectory.save("trajectories/run_001.json")?;

// Replay without API calls
let loaded = Trajectory::load("trajectories/run_001.json")?;
let mock = MockLlmClient::from_trajectory(&loaded);
```

Use cases:
- **Offline testing** without API calls
- **Evaluation datasets** from production runs
- **Debugging** with exact request/response inspection

See the [Agent Authoring Guide](docs/AGENT_AUTHORING.md#trajectory-recording-and-replay) for details.

## Documentation

- [Agent Authoring Guide](docs/AGENT_AUTHORING.md) - Complete walkthrough for implementing new agents
- [Tool Authoring Guide](docs/TOOL_AUTHORING.md) - Complete walkthrough for implementing new tools
- [Hook Authoring Guide](docs/HOOK_AUTHORING.md) - Complete walkthrough for implementing new hooks
- [CLAUDE.md](CLAUDE.md) - Project design philosophy and architecture decisions

## Contributing

This is an experimental project for exploring agent implementation patterns. Feedback and contributions welcome!
