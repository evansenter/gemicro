# Markdown Agent Authoring Guide

Define gemicro agents using markdown files with YAML frontmatter. This format separates configuration from the system prompt, making agents easy to read and maintain.

## Format

```markdown
---
name: my-agent
description: What this agent does
model: gemini-3-flash-preview  # optional
tools:                           # optional
  - file_read
  - grep
---

You are an expert assistant...

## Instructions

Your detailed system prompt goes here.
The entire markdown body becomes the agent's system prompt.
```

## Fields Reference

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `name` | Yes | - | Unique identifier for agent registration |
| `description` | No | Uses `name` | Human-readable description shown in `/agents` |
| `model` | No | Inherited | Model override (e.g., `gemini-3-flash-preview`) |
| `tools` | No | Inherit | Tool access configuration (see below) |

## Tool Configuration

The `tools` field controls which tools the agent can access:

| Configuration | YAML | Behavior |
|--------------|------|----------|
| **Inherit** | (omit field) | Use parent context's tools |
| **No tools** | `tools: []` | Agent cannot call any tools |
| **Specific tools** | `tools: [file_read, grep]` | Only listed tools available |

### Examples

**Inherit parent tools:**
```yaml
---
name: research-assistant
---
```

**No tool access (pure chat):**
```yaml
---
name: advisor
tools: []
---
```

**Specific tools only:**
```yaml
---
name: code-reviewer
tools:
  - file_read
  - glob
  - grep
---
```

## System Prompt

Everything after the closing `---` becomes the agent's system prompt. You can use full markdown formatting:

```markdown
---
name: code-reviewer
---

You are an expert code reviewer.

## Your Tasks

1. **Quality**: Check for bugs and logic errors
2. **Style**: Verify adherence to project conventions
3. **Security**: Flag potential vulnerabilities

## Output Format

For each issue found:
- **File**: `path/to/file.rs:42`
- **Severity**: Critical | Important | Suggestion
- **Issue**: Description of the problem
- **Fix**: Recommended solution
```

## Available Tools

Common tools you can specify:

| Tool | Description |
|------|-------------|
| `file_read` | Read file contents |
| `file_write` | Create/overwrite files |
| `file_edit` | Edit existing files |
| `glob` | Find files by pattern |
| `grep` | Search file contents |
| `bash` | Execute shell commands |
| `web_fetch` | Fetch URL contents |
| `web_search` | Search the web |

## Loading Agents

### Bundled Agents

Gemicro includes bundled agents defined in `session.rs`. These are loaded automatically at startup.

### Custom Agents (Future)

Phase 2 (#247) will support loading from:
- `~/.config/gemicro/agents/*.md` - User-defined agents
- `--agents-dir` CLI flag - Custom agent directories

## Validation

Agent definitions are validated at load time:

1. **Name required**: Must be non-empty
2. **System prompt required**: Markdown body cannot be empty
3. **Definition validation**: `PromptAgentDef::validate()` checks constraints

Parse errors are logged but don't crash the CLI - bundled agents should always succeed.

## Cross-Platform Support

The parser handles both Unix (`\n`) and Windows (`\r\n`) line endings automatically.

## See Also

- `gemicro-loader` crate - The parsing implementation
- `docs/AGENT_AUTHORING.md` - Rust-based agent authoring
- Issue #247 - Phase 2: Directory scanning, hot-reload
