# CLAUDE.md - gemicro-developer

## Core Principle

**Don't build code to do what the LLM already does well.**

Gemicro infrastructure should *enable* LLM capabilities (tools, I/O, workflows), not *duplicate* them (classification, parsing, understanding).

## What Belongs Here

| Feature | Why |
|---------|-----|
| Tool execution | LLM can't execute code |
| Approval batching | LLM can't prompt users - workflow infrastructure |
| Token tracking | LLM doesn't know its token count - system monitoring |
| File I/O | LLM can't read/write files directly |

## What Doesn't Belong

| Rejected Feature | Why Rejected |
|------------------|--------------|
| Intent classification | LLM understands "fix the bug" vs "explain this" from natural language |
| Crate graph parsing | CLAUDE.md describes crate structure; LLM can read it |
| CLAUDE.md structured parsing | Just include raw content in system prompt; LLM extracts what it needs |
| Query categorization | LLM naturally understands query types from context |

## Before Adding Features

Ask: "Could the LLM do this if I just gave it the right context?"

If yes, give it the context instead of building infrastructure.

## IMPORTANT: When in Doubt, ASK

If you're unsure whether a requested feature duplicates LLM capabilities, or if an Issue describes something that might violate this principle — **ASK THE USER before building it.**

Don't silently skip work. Don't assume. Just ask:
> "This feature seems like something the LLM could handle naturally. Should I build it anyway, or rely on the LLM?"
