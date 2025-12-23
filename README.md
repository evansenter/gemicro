# Gemicro

**CLI agent exploration platform for experimenting with AI agent implementation patterns**

Gemicro allows you to explore and interact with different AI agent patterns through a command-line interface, powered by the Gemini API via the [rust-genai](https://github.com/evansenter/rust-genai) library.

## Features

- 🔬 **Agent Pattern Exploration**: Experiment with Deep Research, ReAct, Reflexion, and other agent patterns
- 📊 **Real-time Observability**: Streaming updates show agent execution as it happens
- 🏗️ **Extensible Architecture**: Soft-typed events allow adding new agent types without protocol changes
- 📱 **iOS-Ready**: Platform-agnostic core library for future mobile support
- ⚡ **Parallel Execution**: Deep Research pattern fans out queries for faster results

## Architecture

### Two-Crate Workspace

- **gemicro-core**: Platform-agnostic library with zero platform-specific dependencies
- **gemicro-cli**: Terminal UI using indicatif for progress display (coming soon)

### Design Philosophy

- **Streaming-first**: Agents return async streams of updates for real-time observability
- **Soft-typed events**: Following [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) philosophy for maximum extensibility
- **Interactions API**: Uses Gemini's unified Interactions API exclusively
- **Single model**: Hardcoded to `gemini-3-flash-preview` for consistency

## Project Status

🚧 **Early Development** - Phase 1 (Core Foundation) complete

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed implementation roadmap.

## Development

### Prerequisites

This project depends on the [rust-genai](https://github.com/evansenter/rust-genai) library (v0.2.0+) via git dependency.

```bash
# Clone and build
git clone https://github.com/evansenter/gemicro.git
cd gemicro
cargo build
```

### Building and Testing

```bash
# Build workspace
cargo build

# Run tests
cargo test

# Run tests with logging output
RUST_LOG=debug cargo test

# Build with all features
cargo build --all-features
```

### Dependency Note

The workspace `Cargo.toml` references rust-genai from GitHub:
```toml
rust-genai = { git = "https://github.com/evansenter/rust-genai", branch = "main" }
```

For local development of rust-genai, you can temporarily switch to a path dependency:
```toml
rust-genai = { path = "../rust-genai" }
```

## Planned Usage (CLI coming in Phase 4)

```bash
export GEMINI_API_KEY="your-api-key"

# Basic research query
gemicro "What are the latest developments in quantum computing?"

# Custom configuration
gemicro "Compare async runtimes" --max-queries 7 --timeout 45

# Verbose mode
gemicro "Your query" --verbose
```

## Future Exploration Areas

- Memory compression schemes (inspired by Claude Code's approach)
- Multi-turn conversation support
- Additional agent patterns (ReAct, Reflexion, Planning)
- iOS/mobile interface

## License

MIT

## Contributing

This is an experimental project for exploring agent implementation patterns. Feedback and contributions welcome!
