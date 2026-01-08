# Available Tools

This document lists all tools available for use with agents in Gemicro.

## Philosophy

Gemicro follows the **"Explicit Over Implicit"** principle. Tools are never automatically registered - callers must explicitly add them to a `ToolRegistry`.

```rust
use gemicro_core::ToolRegistry;
use gemicro_file_read::FileRead;
use gemicro_bash::Bash;

let mut registry = ToolRegistry::new();
registry.register(FileRead);
registry.register(Bash::new());
```

## Core Tools (`tools/`)

### File Operations

| Crate | Tool | Description |
|-------|------|-------------|
| `gemicro-file-read` | `FileRead` | Read file contents |
| `gemicro-file-write` | `FileWrite` | Write content to files |
| `gemicro-file-edit` | `FileEdit` | Edit portions of files |
| `gemicro-glob` | `Glob` | Find files by pattern |
| `gemicro-grep` | `Grep` | Search file contents |

### System

| Crate | Tool | Description |
|-------|------|-------------|
| `gemicro-bash` | `Bash` | Execute shell commands |

### Web

| Crate | Tool | Description |
|-------|------|-------------|
| `gemicro-web-fetch` | `WebFetch` | Fetch web pages |
| `gemicro-web-search` | `WebSearch` | Search the web |

### Agent Coordination

| Crate | Tool | Description |
|-------|------|-------------|
| `gemicro-task` | `Task` | Delegate to other agents |
| `gemicro-event-bus` | `EventBus` | Cross-agent communication |

## Utility Tools (`gemicro-prompt-agent/src/tools/`)

These lightweight tools are bundled with PromptAgent for convenience:

| Tool | Description |
|------|-------------|
| `Calculator` | Mathematical expression evaluation |
| `CurrentDatetime` | Get current date/time in any timezone |

```rust
use gemicro_prompt_agent::tools::{Calculator, CurrentDatetime};

registry.register(Calculator);
registry.register(CurrentDatetime);
```

## Using Tools with PromptAgent

```rust
use gemicro_prompt_agent::{PromptAgent, PromptAgentConfig};
use gemicro_core::{AgentContext, ToolRegistry, ToolSet};
use std::sync::Arc;

// 1. Create registry and register tools
let mut registry = ToolRegistry::new();
registry.register(Calculator);
registry.register(FileRead);

// 2. Create agent (optionally filter tools)
let agent = PromptAgent::new(
    PromptAgentConfig::default()
        .with_tool_filter(ToolSet::Specific(vec!["calculator".into()]))
)?;

// 3. Create context with tools
let context = AgentContext::new(llm)
    .with_tools_arc(Arc::new(registry));

// 4. Execute - agent will use function calling
let stream = agent.execute("What is 25 * 4?", context);
```

## Tool Filtering

Use `ToolSet` to control which tools an agent can access:

```rust
use gemicro_core::ToolSet;

// Allow all tools
ToolSet::All

// Allow no tools (simple prompt mode)
ToolSet::None

// Allow specific tools only
ToolSet::Specific(vec!["file_read".into(), "grep".into()])

// Allow all except these
ToolSet::Except(vec!["bash".into(), "file_write".into()])

// Inherit from parent context
ToolSet::Inherit
```

## Creating Custom Tools

See [TOOL_AUTHORING.md](./TOOL_AUTHORING.md) for guidance on creating new tools.
