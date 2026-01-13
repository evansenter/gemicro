# CLAUDE.md - gemicro-developer

See root CLAUDE.md for the **LLM-First Design** principle. This file provides agent-specific examples.

## What Infrastructure Enables vs Duplicates

| ✅ Build (LLM can't do) | ❌ Skip (LLM does well) |
|-------------------------|------------------------|
| Tool execution | Intent classification |
| File I/O | Query categorization |
| User prompts/confirmation | Crate graph understanding |
| Token tracking | CLAUDE.md parsing |

## Before Adding Features

Ask: *"Could the LLM do this if I just gave it the right context?"*

If yes → give it the context instead.

If unsure → ask the user before building.
