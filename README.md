# Gemicro

**CLI agent exploration platform for experimenting with AI agent implementation patterns**

Gemicro allows you to explore and interact with different AI agent patterns through a command-line interface, powered by the Gemini API via the [rust-genai](https://github.com/evansenter/rust-genai) library.

## Features

- 🔬 **Agent Pattern Exploration**: Experiment with Deep Research, ReAct, Reflexion, and other agent patterns
- 📊 **Real-time Observability**: Streaming updates show agent execution as it happens
- 🏗️ **Extensible Architecture**: Soft-typed events allow adding new agent types without protocol changes
- 📱 **iOS-Ready**: Platform-agnostic core library for future mobile support
- ⚡ **Parallel Execution**: Deep Research pattern fans out queries for faster results
- 🌐 **Google Search Grounding**: Enable real-time web search for current events and live data

## Architecture

### Two-Crate Workspace

- **gemicro-core**: Platform-agnostic library with Agent trait, streaming updates, and conversation history
- **gemicro-cli**: Terminal UI with indicatif progress display and rustyline REPL

### Design Philosophy

- **Streaming-first**: Agents return async streams of updates for real-time observability
- **Soft-typed events**: Following [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) philosophy for maximum extensibility
- **Interactions API**: Uses Gemini's unified Interactions API exclusively
- **Single model**: Hardcoded to `gemini-3-flash-preview` for consistency

## Project Status

🚧 **Active Development** - Core features complete, see [GitHub Issues](https://github.com/evansenter/gemicro/issues) for roadmap.

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
  -i, --interactive          Interactive REPL mode
      --api-key <KEY>        Gemini API key (or set GEMINI_API_KEY)
      --min-sub-queries <N>  Minimum sub-queries [default: 3]
      --max-sub-queries <N>  Maximum sub-queries [default: 5]
      --max-concurrent <N>   Max parallel executions [default: 5]
      --timeout <SECS>       Total timeout [default: 180]
      --llm-timeout <SECS>   Per-request timeout [default: 60]
      --temperature <F>      Generation temperature 0.0-1.0 [default: 0.7]
      --google-search        Enable Google Search grounding for real-time data
      --plain                Use plain text output (no markdown rendering)
  -v, --verbose              Enable debug logging
  -h, --help                 Print help
  -V, --version              Print version
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

# Run unit tests
cargo test --workspace

# Run ALL tests including LLM integration tests
cargo test --workspace -- --include-ignored

# Linting
cargo clippy --workspace -- -D warnings

# Format check
cargo fmt --all -- --check
```

### Running Examples

```bash
# Deep research example (non-interactive)
cargo run -p gemicro-core --example deep_research

# REPL demo script
./examples/repl_demo.sh
```

## Future Exploration Areas

- Hot-reload for agent development (`/reload --watch`)
- Persistent sessions across restarts
- Additional agent patterns (ReAct, Reflexion, Planning)
- Tab completion for commands
- iOS/mobile interface

## License

MIT

## Contributing

This is an experimental project for exploring agent implementation patterns. Feedback and contributions welcome!
