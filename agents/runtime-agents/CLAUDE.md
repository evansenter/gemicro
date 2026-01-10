# Runtime Agents Directory

This directory contains markdown-defined agents that are loaded at runtime.

## Rules

1. **Only `.md` files** - No other file types are permitted in this directory
2. **Hermetic agents** - Each agent file must be completely self-contained:
   - No imports or includes from other files
   - No references to external state
   - All configuration in the YAML frontmatter
   - All prompt content in the markdown body
3. **Required frontmatter fields**:
   - `name`: Unique identifier for the agent
   - `description`: One-line summary of what the agent does
4. **Optional frontmatter fields**:
   - `model`: LLM model override (default: inherits from context)
   - `tools`: List of tool names the agent can use (default: inherit all tools)

## File Format

```markdown
---
name: agent-name
description: What this agent does
model: gemini-3.0-flash-preview  # optional
tools:                            # optional
  - file_read
  - glob
---

System prompt content goes here in markdown format.
```

## Tool Availability

Tools are filtered at runtime. If an agent requests a tool that doesn't exist, the tool call will fail gracefully with an error message.

Available tools (check `tools/*/src/lib.rs` for the full list):
- `file_read`, `file_write`, `file_edit` - File operations
- `glob`, `grep` - File discovery and search
- `bash` - Shell command execution
- `web_fetch`, `web_search` - Web access
- `task` - Spawn sub-agents
