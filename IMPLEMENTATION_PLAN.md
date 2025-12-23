# Gemicro Implementation Plan

**Project**: CLI agent exploration platform with Deep Research pattern
**Backend**: rust-genai library (evansenter/rust-genai) - **Interactions API**
**Model**: gemini-3-flash-preview only
**Architecture**: Streaming-first design with pragmatic extensibility
**Timeline**: ~1 week for initial working version

---

## Executive Summary

Gemicro is a CLI application for exploring AI agent implementation patterns. The initial implementation focuses on a Deep Research agent that decomposes complex queries into parallel sub-queries, executes them concurrently, and synthesizes results.

**Key Design Decisions:**
- **Evergreen soft-typing philosophy**: Core types (AgentUpdate, GemicroConfig) use flexible structures that don't require modification when adding new agent types
- **Streaming-first architecture**: Agents emit real-time updates via async streams
- **Soft-typed events**: AgentUpdate uses flexible JSON data following [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) philosophy for extensibility
- **Agent-specific config passed to constructors**: GemicroConfig contains ONLY cross-agent concerns; agent-specific config (like ResearchConfig) passed directly to agent constructors
- **Two-crate workspace**: gemicro-core (library) + gemicro-cli (binary)
- **iOS-ready**: Core library has zero platform-specific dependencies
- **Interactions API**: Uses rust-genai's unified Interactions API
- **Single model**: gemini-3-flash-preview for all operations
- **Observable execution**: Stream updates show decomposition, parallel execution, synthesis in real-time
- **Error resilience**: Continue on sub-agent failure, aggregate partial results
- **Cost controls**: Configurable max tokens and timeouts

**Future Exploration Areas:**
- **Memory compression schemes**: Explore different context compression approaches (Claude Code does excellent context compression) for managing long conversations and research sessions efficiently

---

## Design Principles (MUST READ)

### Evergreen Philosophy

**Core Principle**: The gemicro-core library must support adding new agent types **without modifying core types**.

Inspired by the [Evergreen spec](https://github.com/google-deepmind/evergreen-spec), we adopt these principles:

| Principle | Description |
|-----------|-------------|
| **Soft-Typed Events** | `event_type: String` + `data: JSON` instead of rigid enums |
| **Semantic Meaning in Metadata** | Meaning lives in field names and event_type, not structure |
| **ID Opacity** | IDs are opaque identifiers—never encode semantics in ID values |
| **Named Parameters** | JSON fields are named; adding new fields is always non-breaking |
| **Graceful Unknown Handling** | Unknown event_types and data fields MUST be ignored, not errors |
| **Idempotent Events** | Events should be safely re-processable without side effects |
| **Agent-Specific Config Isolation** | Config belongs to agent constructors, not shared context |

### Applying These Principles

#### ✅ DO: Soft-Typed Events
```rust
// CORRECT: AgentUpdate uses flexible event_type string
pub struct AgentUpdate {
    pub event_type: String,           // "sub_query_completed", "react_step_completed", etc.
    pub data: serde_json::Value,      // Arbitrary agent-specific data
    // ...
}
```

#### ❌ DON'T: Rigid Enums for Extensible Types
```rust
// WRONG: Would require modifying core for each new agent type
pub enum AgentUpdate {
    SubQueryCompleted { /* Deep Research */ },
    ReactStepCompleted { /* ReAct */ },        // ❌ Core modification needed
    ReflexionCritique { /* Reflexion */ },     // ❌ Core modification needed
}
```

#### ✅ DO: Agent-Specific Config Passed to Constructors
```rust
// CORRECT: Agent owns its config
pub struct ResearchConfig { /* Deep Research specific */ }
impl DeepResearchAgent {
    pub fn new(config: ResearchConfig) -> Self { /* ... */ }
}

// Core config contains ONLY cross-agent concerns
pub struct GemicroConfig {
    pub llm: LlmConfig,  // ✅ Shared by all agents
}
```

#### ❌ DON'T: Embed Agent-Specific Config in Core
```rust
// WRONG: Doesn't scale, violates Evergreen philosophy
pub struct GemicroConfig {
    pub llm: LlmConfig,
    pub research: ResearchConfig,        // ❌ Deep Research specific
    pub react: ReactConfig,              // ❌ Would need to add
    pub reflexion: ReflexionConfig,      // ❌ Would need to add
    // ... grows indefinitely
}
```

#### ✅ DO: Keep AgentContext Minimal (Cross-Agent Resources Only)
```rust
// CORRECT: AgentContext has only shared resources
pub struct AgentContext {
    pub llm: Arc<LlmClient>,  // ✅ All agents need LLM access
    // NO agent-specific config here!
}

// Agent-specific config goes to constructor
let agent = DeepResearchAgent::new(research_config);
let stream = agent.execute(query, context);
```

#### ❌ DON'T: Put Agent Config in Context
```rust
// WRONG: Context becomes agent-specific
pub struct AgentContext {
    pub llm: Arc<LlmClient>,
    pub config: ResearchConfig,  // ❌ Only works for Deep Research
}
```

#### ✅ DO: Gracefully Ignore Unknown Events/Fields
```rust
// CORRECT: Unknown event types are logged, not errors
match update.event_type.as_str() {
    "sub_query_completed" => { /* handle */ }
    "final_result" => { /* handle */ }
    _ => {
        log::debug!("Unknown event type: {}", update.event_type);
        // Continue processing - NOT an error
    }
}

// CORRECT: Ignore unknown fields in data
// If future version adds "confidence_score" to sub_query_completed,
// older consumers just don't read it - no breakage
```

#### ✅ DO: Design Idempotent Events
```rust
// CORRECT: Events describe state, not commands
AgentUpdate {
    event_type: "sub_query_completed",
    data: json!({
        "id": 0,
        "result": "...",
        "tokens_used": 42,
    }),
}
// Re-processing this event multiple times has no side effects
// Consumer updates its view of sub-query 0's result
```

### When to Use Strong Typing

Use rigid types for:
- **Error hierarchies**: `#[non_exhaustive]` enums work well here
- **Cross-agent configuration**: Settings truly shared by all agents
- **Internal implementations**: Agent internals can use any structure

**See CLAUDE.md for comprehensive design philosophy documentation.**

---

## Architecture Overview

### Streaming-First Design Philosophy

**Traditional approach** (what we're NOT doing):
```rust
let result = agent.execute(query).await?;  // Blocks until complete
println!("{}", result);
```

**Streaming approach** (what we ARE doing):
```rust
let mut stream = agent.execute(query);
while let Some(update) = stream.next().await {
    let update = update?;
    match update.event_type.as_str() {
        "sub_query_started" => { /* show spinner */ }
        "sub_query_completed" => { /* show result */ }
        "final_result" => { /* show final answer */ }
        _ => { /* log unknown event */ }
    }
}
```

**Benefits:**
- Real-time observability (see what's happening as it happens)
- Can show partial results before completion
- Natural cancellation support (drop the stream)
- Backpressure handling (slow consumer → slow producer)
- Perfect for CLI progress bars AND future iOS live updates

---

### Workspace Structure

```
gemicro/
├── Cargo.toml                  # Workspace manifest
├── IMPLEMENTATION_PLAN.md      # This file
├── README.md                   # User-facing documentation
│
├── gemicro-core/               # Platform-agnostic library (iOS-ready)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs              # Public API and module exports
│       ├── agent.rs            # Agent trait (streaming) + DeepResearchAgent
│       ├── update.rs           # AgentUpdate enum
│       ├── llm.rs              # LLM client wrapper using Interactions API
│       ├── config.rs           # Configuration types
│       └── error.rs            # Error types
│
└── gemicro-cli/                # CLI application
    ├── Cargo.toml
    └── src/
        ├── main.rs             # Entry point
        ├── cli.rs              # Clap argument parsing
        ├── display.rs          # Stream consumer with indicatif
        └── output.rs           # Result formatting
```

### Core Abstractions

**1. Agent Trait (Streaming)** - gemicro-core/src/agent.rs
```rust
#[async_trait]
pub trait Agent {
    /// Execute the agent, returning a stream of updates
    async fn execute(
        &self,
        query: &str,
        context: AgentContext,
    ) -> impl Stream<Item = Result<AgentUpdate, AgentError>> + Send;

    fn name(&self) -> &str;
    fn description(&self) -> &str;
}
```

**2. AgentUpdate (Soft-Typed)** - gemicro-core/src/update.rs

Following the [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) philosophy of pragmatic flexibility over rigid typing, we use a soft-typed event structure:

```rust
/// Flexible event structure for agent updates
/// Inspired by Evergreen protocol's approach to extensibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentUpdate {
    /// Event type identifier (e.g., "decomposition_started", "sub_query_completed")
    /// Semantic meaning comes from this field, not from enum discriminants
    pub event_type: String,

    /// Human-readable message describing the event
    pub message: String,

    /// When the event occurred
    pub timestamp: std::time::SystemTime,

    /// Event-specific data as flexible JSON
    /// Each event type defines its own schema
    pub data: serde_json::Value,
}

/// Type-safe helper constructors for common Deep Research events
impl AgentUpdate {
    pub fn decomposition_started() -> Self {
        Self {
            event_type: "decomposition_started".into(),
            message: "Decomposing query into sub-queries".into(),
            timestamp: std::time::SystemTime::now(),
            data: json!({}),
        }
    }

    pub fn decomposition_complete(sub_queries: Vec<String>) -> Self {
        Self {
            event_type: "decomposition_complete".into(),
            message: format!("Decomposed into {} sub-queries", sub_queries.len()),
            timestamp: std::time::SystemTime::now(),
            data: json!({ "sub_queries": sub_queries }),
        }
    }

    pub fn sub_query_started(id: usize, query: String) -> Self {
        Self {
            event_type: "sub_query_started".into(),
            message: format!("Sub-query {} started", id),
            timestamp: std::time::SystemTime::now(),
            data: json!({ "id": id, "query": query }),
        }
    }

    pub fn sub_query_completed(id: usize, result: String, tokens_used: u32) -> Self {
        Self {
            event_type: "sub_query_completed".into(),
            message: format!("Sub-query {} completed", id),
            timestamp: std::time::SystemTime::now(),
            data: json!({
                "id": id,
                "result": result,
                "tokens_used": tokens_used,
            }),
        }
    }

    pub fn sub_query_failed(id: usize, error: String) -> Self {
        Self {
            event_type: "sub_query_failed".into(),
            message: format!("Sub-query {} failed", id),
            timestamp: std::time::SystemTime::now(),
            data: json!({ "id": id, "error": error }),
        }
    }

    pub fn synthesis_started() -> Self {
        Self {
            event_type: "synthesis_started".into(),
            message: "Synthesizing results".into(),
            timestamp: std::time::SystemTime::now(),
            data: json!({}),
        }
    }

    pub fn final_result(answer: String, metadata: ResultMetadata) -> Self {
        Self {
            event_type: "final_result".into(),
            message: "Research complete".into(),
            timestamp: std::time::SystemTime::now(),
            data: json!({
                "answer": answer,
                "metadata": metadata,
            }),
        }
    }

    /// Typed accessor for decomposition_complete events
    pub fn as_decomposition_complete(&self) -> Option<Vec<String>> {
        if self.event_type == "decomposition_complete" {
            self.data.get("sub_queries")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
        } else {
            None
        }
    }

    /// Typed accessor for sub_query_completed events
    pub fn as_sub_query_completed(&self) -> Option<SubQueryResult> {
        if self.event_type == "sub_query_completed" {
            serde_json::from_value(self.data.clone()).ok()
        } else {
            None
        }
    }

    /// Typed accessor for final_result events
    pub fn as_final_result(&self) -> Option<FinalResult> {
        if self.event_type == "final_result" {
            serde_json::from_value(self.data.clone()).ok()
        } else {
            None
        }
    }
}

/// Strongly-typed result structs for ergonomic access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubQueryResult {
    pub id: usize,
    pub result: String,
    pub tokens_used: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalResult {
    pub answer: String,
    pub metadata: ResultMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultMetadata {
    pub total_tokens: u32,
    pub duration_ms: u64,
    pub sub_queries_succeeded: usize,
    pub sub_queries_failed: usize,
}
```

**Design Rationale:**
- **Extensibility**: New agents can define new event types without modifying gemicro-core
- **Flexibility**: Event data can be arbitrarily complex JSON structures
- **Type Safety**: Helper constructors prevent typos; typed accessors provide ergonomics
- **Forward Compatibility**: Unknown event types are gracefully handled (just logged/ignored)
- **Evergreen-aligned**: "Semantic meaning in metadata, not structure" - event_type carries semantics

**3. LlmClient Wrapper** - gemicro-core/src/llm.rs
```rust
pub struct LlmClient {
    client: rust_genai::Client,
    config: LlmConfig,
}

impl LlmClient {
    // Non-streaming (buffered)
    pub async fn generate(&self, request: LlmRequest)
        -> Result<LlmResponse, LlmError>;

    // Optional: Streaming for long responses (Phase 2+)
    pub fn generate_stream(&self, request: LlmRequest)
        -> impl Stream<Item = Result<String, LlmError>>;
}
```

**Why Streaming in Agent but Not (Initially) in LlmClient?**
- Agent-level streaming: Shows progress across multiple LLM calls (decompose → parallel → synthesize)
- LLM-level streaming: Shows tokens as they generate within a single call
- Start with buffered LLM calls, add streaming when needed for long responses

**Note on Context Management:**
- Initial implementation: stateless LLM calls (no conversation history tracking)
- Future enhancement: Add context compression for multi-turn conversations
- See "Future Enhancements & Research Areas" section for compression strategies
- LlmClient would be natural place for `ContextCompressor` trait integration

---

## Deep Research Pattern Flow (Streaming)

```
User runs: gemicro "What are the latest developments in quantum computing?"
    ↓
let mut stream = agent.execute(query, context);
    ↓
┌─────────────────────────────────────────────────────────┐
│ Stream yields: AgentUpdate {                            │
│   event_type: "decomposition_started", ...              │
│ }                                                        │
│ CLI shows: [•] Decomposing query...                     │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ LLM Call (Interactions API): Decompose query            │
│ Response: ["Q1", "Q2", "Q3", "Q4", "Q5"]               │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Stream yields: AgentUpdate {                            │
│   event_type: "decomposition_complete",                 │
│   data: { "sub_queries": ["Q1", "Q2", ...] }           │
│ }                                                        │
│ CLI shows: [✓] Decomposed into 5 sub-queries           │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Stream yields 5 sub_query_started events                │
│ CLI shows:                                               │
│   [1/5] Quantum hardware advances... ⣾                  │
│   [2/5] Quantum algorithms research... ⣽               │
│   [3/5] Commercial applications... ⣻                    │
│   [4/5] Error correction methods... ⣺                   │
│   [5/5] Industry investments... ⣹                       │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ As each parallel LLM call completes, stream yields:     │
│   AgentUpdate { event_type: "sub_query_completed", ... }│
│   AgentUpdate { event_type: "sub_query_completed", ... }│
│   AgentUpdate { event_type: "sub_query_failed", ... }   │
│   AgentUpdate { event_type: "sub_query_completed", ... }│
│   AgentUpdate { event_type: "sub_query_completed", ... }│
│                                                          │
│ CLI updates progress bars in real-time as they arrive   │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Stream yields: AgentUpdate {                            │
│   event_type: "synthesis_started", ...                  │
│ }                                                        │
│ CLI shows: [•] Synthesizing results...                  │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ LLM Call: Synthesize findings                           │
│ Response: Final comprehensive answer                     │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Stream yields: AgentUpdate {                            │
│   event_type: "final_result",                           │
│   data: { "answer": "...", "metadata": {...} }         │
│ }                                                        │
│ CLI shows formatted final answer + stats                │
└─────────────────────────────────────────────────────────┘
    ↓
Stream ends, CLI exits
```

**User Experience (CLI Output):**
```
$ gemicro "What are the latest developments in quantum computing?"

[•] Decomposing query...
[✓] Decomposed into 5 sub-queries

Executing parallel research:
  [1/5] Quantum hardware advances... ⣾ (in progress)
  [2/5] Quantum algorithms research... ✓ Complete (234 tokens)
  [3/5] Commercial applications... ✗ Failed: timeout
  [4/5] Error correction methods... ⣽ (in progress)
  [5/5] Industry investments... ✓ Complete (189 tokens)

[•] Synthesizing results...
[✓] Synthesis complete

═══════════════════════════════════════════════════════════
ANSWER:

Recent developments in quantum computing include significant
advances in quantum hardware, with companies like IBM and Google
achieving quantum advantage in specific tasks...

[Full answer continues...]

───────────────────────────────────────────────────────────
Stats: 4/5 sub-queries succeeded | 1,247 total tokens | 23.4s
═══════════════════════════════════════════════════════════
```

---

## Implementation Phases

### Phase 1: Core Library Foundation (Days 1-2)

**Goal**: Set up workspace, streaming types, errors, config

**Tasks:**

1. **Create workspace structure**
   - `/Users/evansenter/Documents/projects/gemicro/Cargo.toml`
   - Define workspace members and shared dependencies

2. **Create gemicro-core crate**
   - `/Users/evansenter/Documents/projects/gemicro/gemicro-core/Cargo.toml`
   - Dependencies: rust-genai, tokio, async-trait, thiserror, serde, serde_json, futures-util, async-stream

3. **Implement AgentUpdate struct (soft-typed)**
   - `/Users/evansenter/Documents/projects/gemicro/gemicro-core/src/update.rs`
   - `AgentUpdate` struct with event_type, message, timestamp, data fields
   - Helper constructors for Deep Research events (decomposition_started, sub_query_completed, etc.)
   - Typed accessor methods (as_decomposition_complete, as_sub_query_completed, as_final_result)
   - Derive: Debug, Clone, Serialize, Deserialize
   - `ResultMetadata`, `SubQueryResult`, `FinalResult` structs

4. **Implement error types**
   - `/Users/evansenter/Documents/projects/gemicro/gemicro-core/src/error.rs`
   - `#[non_exhaustive]` on all public enums
   - `GemicroError` - top-level errors
   - `AgentError` - agent-specific errors
   - `LlmError` - LLM client errors
   - Convert from `rust_genai::GenaiError`

5. **Implement configuration types**
   - `/Users/evansenter/Documents/projects/gemicro/gemicro-core/src/config.rs`
   - `MODEL` constant: Hardcode `"gemini-3-flash-preview"`
   - `GemicroConfig`: Contains only cross-agent config (just `LlmConfig`)
   - `LlmConfig`: Shared LLM settings (timeouts, retries, temperature)
   - `ResearchConfig`: Agent-specific config (passed to DeepResearchAgent constructor, NOT in GemicroConfig)
   - **IMPORTANT**: Follow Evergreen philosophy - don't embed agent-specific config in GemicroConfig
   - Implement `Default` with sensible values

6. **Implement lib.rs**
   - `/Users/evansenter/Documents/projects/gemicro/gemicro-core/src/lib.rs`
   - Module declarations
   - Re-exports for public API

**Validation**: `cargo build` succeeds in gemicro-core

**Acceptance Criteria:**
- ✅ Workspace compiles without errors
- ✅ `AgentUpdate` struct with soft-typed design (event_type + data)
- ✅ Helper constructors for all Deep Research event types
- ✅ Typed accessors return Option<T> for ergonomic access
- ✅ Error types use `#[non_exhaustive]`
- ✅ Config is complete with defaults

---

### Phase 2: LLM Client Wrapper (Day 3)

**Goal**: Wrap rust-genai Interactions API with both buffered and streaming support

**Tasks:**

1. **Implement LlmClient**
   - `/Users/evansenter/Documents/projects/gemicro/gemicro-core/src/llm.rs`
   - `LlmClient::new(client, config)` constructor
   - `generate(request)` method using Interactions API (buffered)
   - `generate_stream(request)` method for streaming responses
   - Use `client.interaction().with_model("gemini-3-flash-preview")...`
   - Timeout enforcement with `tokio::time::timeout`
   - Extract text and metadata from `InteractionResponse`

2. **Define request/response types**
   - `LlmRequest { prompt, system_instruction }`
   - `LlmResponse { text, tokens_used, interaction_id }` (for buffered)
   - `LlmStreamChunk { text, is_final, usage }` (for streaming)

3. **Token counting**
   - Extract from `InteractionResponse.usage_metadata` if available
   - Return 0 and log warning if not available (graceful degradation)
   - For streaming: usage metadata available on final chunk only

4. **Error handling**
   - Map `rust_genai::GenaiError` to `LlmError`
   - Handle timeouts distinctly
   - Use `log` crate for warnings

**Validation**: Unit tests with mock responses

**Acceptance Criteria:**
- ✅ LlmClient wraps Interactions API
- ✅ Generates requests with system instructions
- ✅ Returns tokens_used (real or 0 with warning)
- ✅ Enforces timeouts
- ✅ Supports streaming via `generate_stream()`

---

### Phase 3: Streaming Deep Research Agent (Days 4-5)

**Goal**: Implement Deep Research pattern with streaming updates

**Tasks:**

1. **Define Agent trait and context**
   - `/Users/evansenter/Documents/projects/gemicro/gemicro-core/src/agent.rs`
   - `Agent` trait with streaming `execute()` method
   - `AgentContext { llm }` - ONLY cross-agent resources (see Design Principles)
   - Use `async_stream::try_stream!` macro for implementation

2. **Implement DeepResearchAgent structure**
   - `DeepResearchAgent::new(config: ResearchConfig)` - agent-specific config at construction
   - Store configuration in agent struct

3. **Implement streaming execute() method**
   ```rust
   async fn execute(
       &self,
       query: &str,
       context: AgentContext,
   ) -> impl Stream<Item = Result<AgentUpdate, AgentError>> + Send {
       use async_stream::try_stream;

       try_stream! {
           // Yield decomposition_started
           yield AgentUpdate::decomposition_started();

           // Call decompose, yield result
           let sub_queries = decompose(query, &context).await?;
           yield AgentUpdate::decomposition_complete(sub_queries.clone());

           // Yield sub_query_started for each
           for (id, q) in sub_queries.iter().enumerate() {
               yield AgentUpdate::sub_query_started(id, q.clone());
           }

           // Execute in parallel, yield as they complete
           // (More complex - see implementation details below)

           // Yield synthesis_started
           yield AgentUpdate::synthesis_started();

           // Call synthesize, yield final_result
           let answer = synthesize(...).await?;
           yield AgentUpdate::final_result(answer, metadata);
       }
   }
   ```

4. **Implement decomposition phase**
   - `async fn decompose(query: &str, context: &AgentContext) -> Result<Vec<String>>`
   - Craft prompt with system instruction
   - Call `context.llm.generate()`
   - Parse JSON response

5. **Implement parallel execution with streaming**
   - This is the tricky part! We need to:
     - Spawn N tasks with `tokio::spawn`
     - Yield updates as each completes (not all at once)
     - Use `futures::stream::SelectAll` or similar

   ```rust
   // Pseudocode for parallel streaming:
   let (tx, rx) = tokio::sync::mpsc::channel(10);

   for (id, query) in sub_queries.iter().enumerate() {
       let tx = tx.clone();
       let llm = context.llm.clone();
       let query = query.clone();

       tokio::spawn(async move {
           let result = llm.generate(/* ... */).await;
           let update = match result {
               Ok(r) => AgentUpdate::sub_query_completed(id, r.text, r.tokens_used),
               Err(e) => AgentUpdate::sub_query_failed(id, e.to_string()),
           };
           tx.send(update).await.ok();
       });
   }
   drop(tx); // Close channel when all spawned

   // Yield updates as they arrive
   while let Some(update) = rx.recv().await {
       yield update;
   }
   ```

6. **Implement synthesis phase**
   - `async fn synthesize(results, original_query, context) -> Result<String>`
   - Format prompt with findings
   - Call LLM

**Validation**: Integration tests with real API

**Acceptance Criteria:**
- ✅ Agent implements streaming trait
- ✅ Yields updates in real-time as work progresses
- ✅ Parallel sub-queries yield updates as they complete (not blocking)
- ✅ Handles sub-query failures gracefully
- ✅ Final result includes metadata

---

### Phase 4: CLI Application (Day 6)

**Goal**: Build CLI that consumes the stream

**Tasks:**

1. **Create gemicro-cli crate**
   - `/Users/evansenter/Documents/projects/gemicro/gemicro-cli/Cargo.toml`
   - Dependencies: gemicro-core, clap, indicatif, tokio, anyhow, futures-util

2. **Implement argument parsing**
   - `/Users/evansenter/Documents/projects/gemicro/gemicro-cli/src/cli.rs`
   - `Args` struct with clap
   - Fields: query, api_key (env), max_queries, timeout, verbose

3. **Implement stream display**
   - `/Users/evansenter/Documents/projects/gemicro/gemicro-cli/src/display.rs`
   - `StreamDisplay` struct wrapping `MultiProgress`
   - Consumes `Stream<AgentUpdate>` and updates indicatif in real-time

   ```rust
   pub struct StreamDisplay {
       multi: MultiProgress,
       main_bar: ProgressBar,
       sub_bars: HashMap<usize, ProgressBar>,
   }

   impl StreamDisplay {
       pub async fn consume(
           &mut self,
           stream: impl Stream<Item = Result<AgentUpdate, AgentError>>,
       ) -> Result<String> {
           futures::pin_mut!(stream);

           while let Some(update) = stream.next().await {
               let update = update?;
               match update.event_type.as_str() {
                   "decomposition_started" => {
                       self.main_bar.set_message("Decomposing...");
                   }
                   "sub_query_started" => {
                       if let Some(id) = update.data.get("id").and_then(|v| v.as_u64()) {
                           if let Some(query) = update.data.get("query").and_then(|v| v.as_str()) {
                               let pb = self.multi.add(ProgressBar::new_spinner());
                               pb.set_message(format!("[{}/{}] {}", id+1, total, query));
                               self.sub_bars.insert(id as usize, pb);
                           }
                       }
                   }
                   "sub_query_completed" => {
                       if let Some(result) = update.as_sub_query_completed() {
                           if let Some(pb) = self.sub_bars.get(&result.id) {
                               pb.finish_with_message(format!("✓ ({} tokens)", result.tokens_used));
                           }
                       }
                   }
                   "final_result" => {
                       if let Some(final_result) = update.as_final_result() {
                           self.main_bar.finish();
                           return Ok(final_result.answer);
                       }
                   }
                   _ => {
                       // Log unknown event types
                       log::debug!("Unknown event type: {}", update.event_type);
                   }
               }
           }

           Err(anyhow!("Stream ended without final result"))
       }
   }
   ```

4. **Implement main entry point**
   - `/Users/evansenter/Documents/projects/gemicro/gemicro-cli/src/main.rs`
   - Parse CLI args
   - Load API key from env
   - Create `rust_genai::Client`
   - Create `LlmClient`, `AgentContext`
   - Create `DeepResearchAgent`
   - Get stream from `agent.execute()`
   - Pass stream to `StreamDisplay::consume()`
   - Print final result

5. **Implement output formatting**
   - `/Users/evansenter/Documents/projects/gemicro/gemicro-cli/src/output.rs`
   - Format final answer with metadata
   - Show sub-query summary

**Validation**: Manual end-to-end testing

**Acceptance Criteria:**
- ✅ CLI consumes agent stream
- ✅ Progress bars update in real-time
- ✅ Sub-queries show as they complete (not all at once)
- ✅ Final result is well-formatted
- ✅ Errors are user-friendly

---

### Phase 5: Polish & Cost Controls (Day 7)

**Goal**: Production-ready features

**Tasks:**

1. **Add retry logic**
   - Exponential backoff in LlmClient
   - Configurable max retries

2. **Improve token counting**
   - Use real API metadata if available
   - Track cumulative tokens
   - Warn on approach to limits

3. **Add generation config**
   - Check InteractionBuilder support
   - File feature request if needed

4. **Improve error messages**
   - Context-rich errors
   - Suggestions for fixes

5. **Add logging**
   - Use `tracing` crate
   - `--verbose` flag
   - Log all LLM interactions

6. **Documentation**
   - README with examples
   - API docs
   - Inline rustdoc

**Validation**: Full test suite passes

**Acceptance Criteria:**
- ✅ Retry logic works
- ✅ Token tracking is accurate
- ✅ Errors are actionable
- ✅ Logging is useful
- ✅ Documentation is complete

---

## Key Implementation Details

### Streaming Parallel Execution (The Hard Part)

**Challenge**: We spawn N tasks that complete in any order, need to yield updates as they finish.

**Solution**: Use `tokio::sync::mpsc` channel

```rust
use tokio::sync::mpsc;
use async_stream::try_stream;

// In DeepResearchAgent::execute()
try_stream! {
    // ... decomposition yields ...

    // Create channel for sub-query results
    let (tx, mut rx) = mpsc::channel::<AgentUpdate>(10);

    // Spawn all sub-queries
    for (id, query) in sub_queries.iter().enumerate() {
        yield AgentUpdate::sub_query_started(id, query.clone());

        let tx = tx.clone();
        let llm = context.llm.clone();
        let query = query.clone();

        tokio::spawn(async move {
            let result = llm.generate(LlmRequest {
                prompt: query,
                system_instruction: None,
            }).await;

            let update = match result {
                Ok(response) => AgentUpdate::sub_query_completed(
                    id,
                    response.text,
                    response.tokens_used,
                ),
                Err(error) => AgentUpdate::sub_query_failed(
                    id,
                    error.to_string(),
                ),
            };

            // Send update through channel (ignore send errors)
            let _ = tx.send(update).await;
        });
    }

    // Drop original tx so channel closes when all tasks done
    drop(tx);

    // Yield updates as they arrive
    let mut results = Vec::new();
    while let Some(update) = rx.recv().await {
        // Extract results using typed accessor
        if let Some(completed) = update.as_sub_query_completed() {
            results.push(completed.result.clone());
        }
        // Note: failures are tracked in the event itself, just continue
        yield update;
    }

    // Check if we got at least one result
    if results.is_empty() {
        Err(AgentError::AllSubQueriesFailed)?;
    }

    // ... synthesis yields ...
}
```

### Error Propagation in Streams

**Important**: Errors in async_stream are yielded, not returned!

```rust
try_stream! {
    // This works:
    let result = some_fallible_operation().await?;  // Error yields and stream ends
    yield AgentUpdate::decomposition_started();

    // Conditional error:
    if results.is_empty() {
        Err(AgentError::AllSubQueriesFailed)?;  // Yields error, ends stream
    }
}
```

**Consuming errors in CLI:**
```rust
while let Some(update) = stream.next().await {
    match update {
        Ok(update) => { /* handle update */ }
        Err(error) => {
            eprintln!("Error: {}", error);
            break;  // Or continue, depending on error type
        }
    }
}
```

---

## File-by-File Implementation Guide

### Workspace Root

**File: `/Users/evansenter/Documents/projects/gemicro/Cargo.toml`**
```toml
[workspace]
members = ["gemicro-core", "gemicro-cli"]
resolver = "2"

[workspace.dependencies]
rust-genai = { path = "../rust-genai" }
tokio = { version = "1.48", features = ["full"] }
async-trait = "0.1"
async-stream = "0.3"
thiserror = "2.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
futures-util = "0.3"
anyhow = "1.0"
```

### gemicro-core Library

**File: `/Users/evansenter/Documents/projects/gemicro/gemicro-core/Cargo.toml`**
```toml
[package]
name = "gemicro-core"
version = "0.1.0"
edition = "2021"

[dependencies]
rust-genai = { workspace = true }
tokio = { workspace = true }
async-trait = { workspace = true }
async-stream = { workspace = true }
thiserror = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
futures-util = { workspace = true }
log = "0.4"
```

**File: `/Users/evansenter/Documents/projects/gemicro/gemicro-core/src/lib.rs`**
```rust
pub mod agent;
pub mod update;
pub mod llm;
pub mod config;
pub mod error;

// Re-exports
pub use agent::{Agent, AgentContext, DeepResearchAgent};
pub use update::{AgentUpdate, ResultMetadata};
pub use llm::{LlmClient, LlmRequest, LlmResponse};
pub use config::{GemicroConfig, ResearchConfig, LlmConfig, MODEL};
pub use error::{GemicroError, AgentError, LlmError};
```

**File: `/Users/evansenter/Documents/projects/gemicro/gemicro-core/src/update.rs`**
- Define `AgentUpdate` struct with soft-typed design (event_type, message, timestamp, data)
- Implement helper constructors (decomposition_started, sub_query_completed, etc.)
- Implement typed accessors (as_decomposition_complete, as_sub_query_completed, etc.)
- Define `ResultMetadata`, `SubQueryResult`, `FinalResult` structs
- Derive: `Debug, Clone, Serialize, Deserialize`

**File: `/Users/evansenter/Documents/projects/gemicro/gemicro-core/src/error.rs`**
- Define error enums with `#[non_exhaustive]`
- Implement conversions from rust_genai errors

**File: `/Users/evansenter/Documents/projects/gemicro/gemicro-core/src/config.rs`**
- Define config structs
- `pub const MODEL: &str = "gemini-3-flash-preview";`
- Implement `Default` trait

**File: `/Users/evansenter/Documents/projects/gemicro/gemicro-core/src/llm.rs`**
- `LlmClient` wrapping Interactions API
- `generate()` method
- Timeout and token tracking

**File: `/Users/evansenter/Documents/projects/gemicro/gemicro-core/src/agent.rs`**
- `Agent` trait with streaming `execute()`
- `AgentContext` struct
- `DeepResearchAgent` implementation using `async_stream::try_stream!`
- Helper functions: `decompose()`, `synthesize()`

### gemicro-cli Application

**File: `/Users/evansenter/Documents/projects/gemicro/gemicro-cli/Cargo.toml`**
```toml
[package]
name = "gemicro-cli"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "gemicro"
path = "src/main.rs"

[dependencies]
gemicro-core = { path = "../gemicro-core" }
rust-genai = { workspace = true }
tokio = { workspace = true }
anyhow = { workspace = true }
futures-util = { workspace = true }
clap = { version = "4.5", features = ["derive"] }
indicatif = "0.17"
tracing-subscriber = "0.3"
```

**File: `/Users/evansenter/Documents/projects/gemicro/gemicro-cli/src/main.rs`**
```rust
mod cli;
mod display;
mod output;

use anyhow::Result;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = cli::Args::parse();

    // Create client and context (cross-agent resources only)
    let client = rust_genai::Client::builder(args.api_key).build();
    let llm = gemicro_core::LlmClient::new(client, args.llm_config());
    let context = gemicro_core::AgentContext {
        llm: std::sync::Arc::new(llm),
        // NO agent-specific config here - follows Evergreen philosophy
    };

    // Create agent with its own config (agent-specific)
    let agent = gemicro_core::DeepResearchAgent::new(args.research_config());

    // Get stream
    let stream = agent.execute(&args.query, context).await;

    // Consume stream with display
    let mut display = display::StreamDisplay::new();
    let answer = display.consume(stream).await?;

    // Print final output
    output::print_result(&answer);

    Ok(())
}
```

**File: `/Users/evansenter/Documents/projects/gemicro/gemicro-cli/src/display.rs`**
- `StreamDisplay` struct with `MultiProgress`
- `consume()` method that processes stream updates
- Update indicatif bars in real-time

---

## Potential rust-genai Feature Requests

### Feature Request #1: Expose UsageMetadata
**Title**: Expose `usage_metadata` in `InteractionResponse`
**Status**: Check if already available, file if not

### Feature Request #2: GenerationConfig in InteractionBuilder
**Title**: Add `with_generation_config()` to `InteractionBuilder`
**Status**: File if not available

### Feature Request #3: Context Compression Utilities (Future)
**Title**: Add conversation history compression utilities
**Rationale**: Many agent patterns need efficient context management
**Potential API**:
- `ConversationHistory` type for managing multi-turn state
- `ContextCompressor` trait for pluggable compression strategies
- Built-in strategies: summarization, sliding window, semantic deduplication
**Priority**: Low (explore in gemicro first, upstream if patterns emerge)
**Note**: Claude Code provides excellent reference implementation

---

## Success Criteria

### MVP (End of Phase 4)
- ✅ Agent yields streaming updates
- ✅ CLI shows real-time progress
- ✅ Sub-queries update as they complete (not all at once)
- ✅ Final answer displays correctly
- ✅ Uses gemini-3-flash-preview via Interactions API

### Production Ready (End of Phase 5)
- ✅ All tests pass
- ✅ Error messages are clear
- ✅ Cost controls enforced
- ✅ Documentation complete
- ✅ Feature requests filed for rust-genai

---

## CLI Usage Examples

```bash
export GEMINI_API_KEY="your-api-key"

# Basic
gemicro "What are the pros and cons of Rust?"

# Custom config
gemicro "Compare async runtimes" --max-queries 7 --timeout 45

# Verbose
gemicro "Your query" --verbose
```

---

## Future Enhancements & Research Areas

### Memory Compression Schemes

**Motivation**: Long research sessions and multi-turn conversations can quickly consume context windows. Effective compression is critical for maintaining conversation coherence while staying within token limits.

**Areas to Explore:**

1. **Summarization-based compression**
   - Periodically summarize conversation history
   - Keep recent messages verbatim, compress older ones
   - Claude Code's approach: maintains semantic meaning while reducing tokens

2. **Hierarchical context management**
   - Different compression levels for different parts of history
   - Critical information (original query, key findings) preserved verbatim
   - Supporting details compressed or dropped

3. **Semantic deduplication**
   - Identify and merge redundant information across sub-queries
   - Remove overlapping findings from parallel research results

4. **Adaptive compression**
   - Monitor context usage in real-time
   - Dynamically adjust compression aggressiveness
   - Preserve more detail when context window allows

5. **Retrieval-based approaches**
   - Store full history externally
   - Retrieve relevant context on-demand
   - Similar to RAG patterns but for conversation history

**Implementation Considerations:**
- Add compression hooks in `LlmClient` or `AgentContext`
- Compression strategies could be agent-specific or shared
- Metrics: compression ratio, semantic preservation, token savings
- Could be exposed as pluggable `ContextCompressor` trait

**References:**
- Claude Code's context compression (exemplary approach)
- LangChain's memory management patterns
- Anthropic's prompt caching strategies

### Control Plane Capabilities

**Current State**: AgentUpdate events are purely observational—they describe what happened but don't control execution.

**Future Direction**: Add bidirectional communication for agent control.

#### Option 1: Separate Control Channel
```rust
// Keep AgentUpdate for observations (unidirectional)
pub struct AgentUpdate { /* existing */ }

// Add separate control type (bidirectional)
pub struct AgentCommand {
    pub command_type: String,           // "cancel", "pause", "adjust_config"
    pub command_id: String,             // For acknowledgment tracking
    pub data: serde_json::Value,
}

pub struct AgentAck {
    pub command_id: String,
    pub status: String,                 // "accepted", "rejected", "completed"
    pub data: serde_json::Value,
}
```

**Rationale**: Clean separation of concerns. Events remain idempotent and replayable. Commands are inherently non-idempotent (canceling twice shouldn't cancel twice).

#### Option 2: Unified Message Type with Direction
```rust
pub struct AgentMessage {
    pub direction: MessageDirection,     // Observation | Command | Ack
    pub message_type: String,
    pub correlation_id: Option<String>,  // Links commands to acks
    pub data: serde_json::Value,
}

pub enum MessageDirection {
    Observation,  // Agent → Consumer (current AgentUpdate)
    Command,      // Consumer → Agent
    Ack,          // Agent → Consumer (response to command)
}
```

**Trade-offs**:
- Option 1: Cleaner types, easier to reason about, more code
- Option 2: Unified handling, but mixes concerns

**Recommendation**: Start with Option 1. Add control plane only when needed (e.g., long-running agents that need cancellation).

### Schema Evolution & Versioning

**Current State**: Soft-typed JSON with implicit schema. Works for exploration but has risks.

#### Schema Evolution Strategies

**1. Additive-Only Changes (Safe)**
```rust
// v1: Original schema
json!({ "id": 0, "result": "..." })

// v2: Add optional field (backward compatible)
json!({ "id": 0, "result": "...", "confidence": 0.95 })

// Old consumers ignore "confidence" - no breakage
```

**2. Field Renaming (Breaking)**
```rust
// v1
json!({ "tokens_used": 42 })

// v2 - BREAKING! Old consumers won't find "tokens_used"
json!({ "token_count": 42 })

// Mitigation: Include both during transition
json!({ "tokens_used": 42, "token_count": 42 })
```

**3. Semantic Changes (Subtle Breaking)**
```rust
// v1: tokens_used is input tokens only
json!({ "tokens_used": 42 })

// v2: tokens_used is now input + output tokens
// Same field name, different meaning - dangerous!

// Mitigation: New field name for new semantics
json!({ "tokens_used": 42, "total_tokens": 150 })
```

#### Versioning Approaches

**Option A: Version in Event Type**
```rust
// Explicit version in event_type
"sub_query_completed.v2"

// Consumer can handle multiple versions
match update.event_type.as_str() {
    "sub_query_completed" | "sub_query_completed.v1" => { /* old */ }
    "sub_query_completed.v2" => { /* new */ }
    _ => { /* unknown */ }
}
```

**Option B: Schema Version Field**
```rust
pub struct AgentUpdate {
    pub event_type: String,
    pub schema_version: Option<u32>,  // None = v1, Some(2) = v2
    pub data: serde_json::Value,
}
```

**Option C: Content Negotiation**
```rust
// Producer advertises supported versions
// Consumer requests preferred version
// Similar to HTTP Accept headers
```

**Recommendation**: Use additive-only changes as long as possible. When breaking changes are unavoidable, use Option A (version in event_type) for simplicity.

### Schema Validation

**Current State**: No validation. Malformed events fail silently (return None from accessors, log warning).

#### Validation Levels

**Level 0: None (Current)**
- Pros: Maximum flexibility, fast
- Cons: Bugs hide until runtime, hard to debug

**Level 1: Accessor Validation (Recommended Near-Term)**
```rust
impl AgentUpdate {
    /// Validates and parses, returning detailed error
    pub fn try_as_sub_query_completed(&self) -> Result<SubQueryResult, ValidationError> {
        if self.event_type != EVENT_SUB_QUERY_COMPLETED {
            return Err(ValidationError::WrongEventType {
                expected: EVENT_SUB_QUERY_COMPLETED,
                actual: self.event_type.clone(),
            });
        }
        serde_json::from_value(self.data.clone())
            .map_err(|e| ValidationError::SchemaMismatch {
                event_type: self.event_type.clone(),
                error: e.to_string(),
            })
    }
}
```

**Level 2: JSON Schema Validation**
```rust
// Define schemas for each event type
const SUB_QUERY_COMPLETED_SCHEMA: &str = r#"{
    "type": "object",
    "required": ["id", "result", "tokens_used"],
    "properties": {
        "id": { "type": "integer", "minimum": 0 },
        "result": { "type": "string" },
        "tokens_used": { "type": "integer", "minimum": 0 }
    }
}"#;

impl AgentUpdate {
    pub fn validate(&self) -> Result<(), ValidationError> {
        let schema = get_schema_for_event_type(&self.event_type)?;
        jsonschema::validate(&schema, &self.data)?;
        Ok(())
    }
}
```

**Level 3: Type Registry**
```rust
// Runtime type registry for dynamic validation
pub struct EventRegistry {
    schemas: HashMap<String, JsonSchema>,
    validators: HashMap<String, Box<dyn Fn(&Value) -> Result<(), String>>>,
}

impl EventRegistry {
    pub fn register(&mut self, event_type: &str, schema: JsonSchema) { /* ... */ }
    pub fn validate(&self, update: &AgentUpdate) -> Result<(), ValidationError> { /* ... */ }
}
```

#### Production Validation Strategy

```rust
// Environment-aware validation
pub enum ValidationMode {
    Off,              // Development: maximum flexibility
    WarnOnly,         // Staging: log issues but don't fail
    Strict,           // Production: fail on invalid events
}

impl AgentUpdate {
    pub fn validate_with_mode(&self, mode: ValidationMode) -> Result<(), ValidationError> {
        match mode {
            ValidationMode::Off => Ok(()),
            ValidationMode::WarnOnly => {
                if let Err(e) = self.validate() {
                    log::warn!("Event validation failed: {}", e);
                }
                Ok(())
            }
            ValidationMode::Strict => self.validate(),
        }
    }
}
```

### Production Hardening Checklist

When moving from exploration to production:

| Area | Current | Production Requirement |
|------|---------|----------------------|
| **Validation** | None | Level 1 minimum, Level 2 for external APIs |
| **Schema Versioning** | Implicit v1 | Explicit versions for breaking changes |
| **Unknown Events** | log::warn + continue | Configurable: drop, queue, or fail |
| **Error Recovery** | Basic retry | Circuit breakers, dead letter queues |
| **Observability** | log crate | Structured logging, metrics, tracing |
| **Testing** | Unit tests | Contract tests, fuzzing, chaos testing |
| **Documentation** | Code comments | OpenAPI/AsyncAPI specs for event schemas |

**Migration Path**:
1. Add `try_as_*` methods alongside existing `as_*` methods
2. Add optional `schema_version` field to AgentUpdate
3. Build event registry with JSON schemas
4. Add validation mode configuration
5. Generate documentation from schemas

**References:**
- CloudEvents specification (event envelope patterns)
- AsyncAPI (event-driven API documentation)
- JSON Schema (validation)

---

**Document Version**: 6.0 (Control plane, schema evolution, validation roadmap)
**Last Updated**: 2025-12-23
**Status**: Ready for implementation
