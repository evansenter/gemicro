# Gemicro

**CLI agent exploration platform for experimenting with AI agent implementation patterns**

Gemicro allows you to explore and interact with different AI agent patterns through a command-line interface, powered by the Gemini API via the [genai-rs](https://github.com/evansenter/genai-rs) library.

## Why Gemicro?

Building AI agents that call tools seems simple—until you need:

- **Real-time visibility** — What is the agent doing *right now*? Which tool is it calling?
- **Graceful cancellation** — User pressed Ctrl+C. Clean up without crashing or orphaned requests.
- **User confirmation** — "This will execute `rm -rf`. Proceed?" Interactive approval for dangerous operations.
- **Unified error handling** — Retry transient failures, surface permanent ones, categorize by type.
- **Evaluation** — Is this agent actually accurate? Run it against benchmarks with scoring.
- **Trajectory capture** — Debug failures by replaying exact LLM request/response sequences.
- **Security hooks** — Block writes to sensitive paths, sanitize inputs, log all tool calls.
- **Agent composability** — Switch agents at runtime, register new ones without code changes.

Gemicro provides these capabilities so you can focus on the reasoning pattern.

### Cross-Cutting Concerns

| Concern | DIY Approach | With Gemicro |
|---------|--------------|--------------|
| **Observability** | Black box until completion | Streaming `AgentUpdate` events (`tool_call_started`, `tool_result`, custom) |
| **Cancellation** | Kill the process, hope for the best | Cooperative `CancellationToken` checked at safe points |
| **Tool Confirmation** | Manual checks scattered per tool | `ConfirmationHandler` trait with `AutoApprove`/`AutoDeny`/`Interactive` |
| **Tool Discovery** | Hardcoded tool lists | `ToolRegistry` + `ToolSet` filtering (`All`/`None`/`Specific`/`Except`) |
| **Error Handling** | Ad-hoc `anyhow::Error` | Typed `AgentError` with `is_retriable()`, `is_timeout()`, `is_cancelled()` |
| **Result Metadata** | Parse it yourself | `FinalResult` with token counts, duration, agent-specific `extra` field |
| **CLI Integration** | Build progress bars from scratch | `Renderer` trait + `ExecutionTracking` for automatic display |
| **Evaluation** | Manual spot-checking | `EvalHarness` + scorers (`Contains`, `LLMJudge`) + datasets (`HotpotQA`, `GSM8K`) |
| **Trajectory Recording** | Hope you logged enough | `Trajectory` capture/replay with `MockLlmClient` for offline testing |
| **Security Hooks** | Audit logging? What's that? | `Interceptor` trait for pre/post tool execution (audit, security, metrics) |
| **Agent Switching** | Refactor main() | `AgentRegistry` with runtime agent selection (`--agent developer`) |
| **Extensibility** | Modify core types for each event | Soft-typed events per [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) |

## Quick Start

```bash
# Set your API key
export GEMINI_API_KEY="your-api-key"

# Single query mode
gemicro "What are the latest developments in quantum computing?"

# Interactive REPL mode
gemicro --interactive
```

## Library Usage

```rust
use gemicro_developer_agent::{DeveloperAgent, DeveloperAgentConfig};
use gemicro_core::{Agent, AgentContext, LlmClient, LlmConfig};

let llm = LlmClient::new(rust_genai::Client::builder(api_key).build(), LlmConfig::default());
let agent = DeveloperAgent::new(DeveloperAgentConfig::default())?;
let stream = agent.execute("Read CLAUDE.md and summarize it", AgentContext::new(llm));

while let Some(update) = stream.next().await {
    let update = update?;
    match update.event_type.as_str() {
        "tool_call_started" => println!("🔧 {}", update.message),
        "final_result" => println!("{}", update.as_final_result().unwrap().result),
        _ => {} // Ignore unknown events (Evergreen philosophy)
    }
}
```

See [`agents/gemicro-developer-agent/examples/developer.rs`](agents/gemicro-developer-agent/examples/developer.rs) for the full example with tools and confirmation handling.

## Available Agents

| Agent | Pattern | Use Case |
|-------|---------|----------|
| `deep_research` | Decompose → Parallel Execute → Synthesize | Multi-hop research questions |
| `react` | Thought → Action → Observation loops | Step-by-step reasoning with tools |
| `developer` | Explicit FC with real-time tool events | Code tasks with full visibility |
| `prompt_agent` | Single LLM call with optional tools | Simple prompts and tool use |
| `critique` | LLM-as-judge with verdicts | Evaluation and quality assessment |

## Architecture

### Workspace Structure

```
gemicro-core (Agent trait, Tool trait, Interceptor trait, Coordination trait, events, LLM)
    ↓
tools/* (10 crates)  ·  hooks/* (5 crates)  ·  agents/* (7 crates)
    ↓
gemicro-runner  ·  gemicro-eval  ·  gemicro-cli
```

| Layer | Contents |
|-------|----------|
| **gemicro-core** | Agent/Tool/Interceptor/Coordination traits, events, LlmClient |
| **tools/** | file ops, search (glob/grep), bash, web, task, event_bus |
| **hooks/** | audit_log, file_security, input_sanitizer, conditional_permission, metrics |
| **agents/** | deep_research, react, developer, prompt_agent, critique, echo |
| **gemicro-runner** | AgentRunner, AgentRegistry, ExecutionState |
| **gemicro-eval** | HotpotQA/GSM8K datasets, scorers |
| **gemicro-cli** | Terminal UI, REPL, markdown rendering |

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
use gemicro_core::{Agent, AgentContext, AgentUpdate};           // Core types
use gemicro_deep_research_agent::{DeepResearchAgent, DeepResearchAgentConfig}; // Agent + config
use gemicro_developer_agent::{DeveloperAgent, DeveloperAgentConfig};       // Developer agent
use gemicro_critique_agent::CritiqueAgent;                             // Critique agent
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

[deep_research] > /agent developer
Switched to: developer

[developer] > Read the CLAUDE.md file
🔧 FileRead: ./CLAUDE.md ...
  ✓ FileRead (0.1s) → # CLAUDE.md...
...

[developer] > /quit
Goodbye!
```

### CLI Options

Key options (run `gemicro --help` for full list):

| Option | Description |
|--------|-------------|
| `-i, --interactive` | REPL mode |
| `--agent <NAME>` | Agent to use (required) |
| `--google-search` | Enable web grounding |
| `--timeout <SECS>` | Total timeout [default: 180] |
| `-v, --verbose` | Debug logging |

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

# Run all quality gates (format, clippy, tests) - pre-push gate
make check

# Individual quality gates
make fmt        # Check formatting
make clippy     # Run clippy with -D warnings
make test       # Run unit + doc tests (uses cargo-nextest)
make test-all   # Include LLM integration tests (requires GEMINI_API_KEY)
```

**During development**, target only changed crates for faster feedback:

```bash
cargo nextest run -p gemicro-core              # Single crate (~3s vs ~15s full)
cargo nextest run -p gemicro-core -p gemicro-runner  # Multiple crates
cargo nextest run test_name                    # Single test by name
```

Install nextest: `cargo install cargo-nextest`

### Running Examples

```bash
# Deep research example (non-interactive)
cargo run -p gemicro-deep-research-agent --example deep_research

# A/B comparison example (requires GEMINI_API_KEY)
cargo run -p gemicro-eval --example ab_comparison

# REPL demo script
./examples/repl_demo.sh
```

### Evaluation

Run evaluations against built-in or custom datasets (requires `GEMINI_API_KEY`):

- **HotpotQA**: Multi-hop question answering benchmark
- **GSM8K**: Grade school math word problems

```bash
# Basic evaluation with default scorers
gemicro-eval --dataset hotpotqa --sample 10

# With specific scorer (GSM8K math problems)
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

## Future Exploration Areas

See [GitHub Issues](https://github.com/evansenter/gemicro/issues) for the full roadmap. Key areas include:

- Additional agent patterns (Reflexion, Plan-and-Execute)
- Model Context Protocol (MCP) client support
- Hot-reload for agent development (`/reload --watch`)
- Persistent sessions across restarts
- Tab completion for REPL commands and agent names
- Performance benchmarks with criterion.rs

## Documentation

- [Agent Authoring Guide](docs/AGENT_AUTHORING.md) - Complete walkthrough for implementing new agents
- [Tool Authoring Guide](docs/TOOL_AUTHORING.md) - Complete walkthrough for implementing new tools
- [Interceptor Authoring Guide](docs/INTERCEPTOR_AUTHORING.md) - Complete walkthrough for implementing new interceptors
- [CLAUDE.md](CLAUDE.md) - Project design philosophy and architecture decisions

## License

MIT

## Security

See [SECURITY.md](SECURITY.md) for security policy, vulnerability reporting, and best practices.

## Contributing

This is an experimental project for exploring agent implementation patterns. Feedback and contributions welcome!
