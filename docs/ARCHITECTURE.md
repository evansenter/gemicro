# Architecture

This document explains the execution architecture of Gemicro, focusing on how agents, tools, and the LLM client interact.

## Execution Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                           AgentRunner                                │
│   Executes agents, manages metrics, handles cancellation            │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                              Agent                                   │
│   Implements business logic, orchestrates LLM calls and tools       │
│   (DeveloperAgent, PromptAgent, DeepResearchAgent, etc.)           │
└───────────────────┬─────────────────────────────┬───────────────────┘
                    │                             │
        ┌───────────┴───────────┐     ┌───────────┴───────────┐
        ▼                       │     ▼                       │
┌───────────────────┐           │ ┌───────────────────┐       │
│    LlmClient      │           │ │   ToolService     │       │
│   Talks to LLM    │           │ │  Tool execution   │       │
└────────┬──────────┘           │ └─────────┬─────────┘       │
         │                      │           │                 │
         ▼                      │           ▼                 │
┌───────────────────┐           │ ┌───────────────────┐       │
│    genai_rs       │           │ │ InterceptorChain  │       │
│  Gemini API SDK   │           │ │ Security/Logging  │       │
└───────────────────┘           │ └─────────┬─────────┘       │
                                │           │                 │
                                │           ▼                 │
                                │ ┌───────────────────┐       │
                                │ │  ToolRegistry     │       │
                                │ │  Individual Tools │       │
                                │ └───────────────────┘       │
                                │                             │
                                └─────────────────────────────┘
```

## Key Components

### LlmClient

The `LlmClient` wraps genai_rs to provide:

- **Logging**: All requests/responses logged at debug level
- **Retry**: Automatic retry on transient failures (rate limits, timeouts)
- **Recording**: Optional trajectory recording for replay/evaluation
- **Timeout enforcement**: Per-request timeout handling

```rust
// Standard usage - build with InteractionBuilder, execute with generate()
let request = context.llm.client().interaction()
    .with_text("What is 2+2?")
    .build();
let response = context.llm.generate(request).await?;

// With function calling (callback API - handles loop internally)
let request = context.llm.client().interaction()
    .with_system_instruction(system_prompt)
    .with_text(query)
    .add_functions(&function_declarations)
    .with_store_enabled()  // Enable continuation
    .build();
let result = context.llm.generate_with_tools(
    request,
    |fc| async move { execute_tool(&fc.name, &fc.args).await },
    10,  // max_turns
    &cancellation_token,
).await?;

// Or use primitives for fine-grained control (event emission, custom flow)
let request = context.llm.client().interaction()
    .with_system_instruction(system_prompt)
    .with_text(query)
    .add_functions(&function_declarations)
    .with_store_enabled()
    .build();
let response = context.llm.generate(request).await?;

// Continuation (sending function results back)
let continuation = context.llm.client().interaction()
    .with_previous_interaction(interaction_id)
    .add_functions(&function_declarations)
    .with_content(results)
    .with_store_enabled()
    .build();
let response = context.llm.generate(continuation).await?;
```

#### Function Calling API Levels

| API | Use When |
|-----|----------|
| `generate_with_tools()` | Simple callback-based tool execution, no per-tool events needed |
| `InteractionBuilder` + `generate()` | Streaming agents that emit per-tool events (PromptAgent) |
| Direct `client().interaction()` | Advanced optimizations like skipping function re-declaration |

#### Escape Hatch: `client()`

For advanced use cases needing direct genai_rs access:

```rust
let genai_client = context.llm.client();
genai_client.interaction()
    .with_model("gemini-3-flash-preview")
    // ... custom configuration
    .create()
    .await
```

**When to use `client()`:**
- Custom interaction builders with specialized options
- Streaming with manual control
- Optimizations that skip re-sending function declarations

**When NOT to use `client()`:**
- Standard LLM calls (use `generate()`)
- Function calling (use `add_functions()` and `continuation()`)
- Anything that benefits from logging/retry/recording

Currently only `DeveloperAgent` uses `client()` for its optimized function calling loop.

### InterceptorChain

Interceptors wrap tool execution, not LLM calls. They provide:

- **Pre-execution hooks**: Security checks, input sanitization
- **Post-execution hooks**: Audit logging, metrics
- **Transformation**: Modify tool inputs/outputs

```
Tool Call → InterceptorChain → Actual Tool → InterceptorChain → Result
```

Interceptors see tool calls, not LLM prompts. For LLM call interception, use LlmClient's recording/logging.

### ToolAdapter

Bridges the gap between genai_rs function calling and Gemicro tools:

```rust
let adapter = ToolCallableAdapter::new(tool)
    .with_confirmation_handler(handler)
    .with_interceptors(chain);

let result = adapter.call(arguments).await?;
```

The adapter:
1. Checks if confirmation is required
2. Runs pre-execution interceptors
3. Executes the tool
4. Runs post-execution interceptors
5. Returns the result

## Data Flow: Function Calling

```
┌─────────┐   1. Request with   ┌───────────┐   2. API call    ┌───────────┐
│  Agent  │ ─────functions────► │ LlmClient │ ───────────────► │  Gemini   │
└─────────┘                     └───────────┘                  └───────────┘
     ▲                                                              │
     │                                                              │
     │ 6. Next request                               3. Response with
     │    (more calls or final)                         function_calls
     │                                                              │
     │                                                              ▼
┌─────────┐   5. Results back   ┌───────────┐   4. Execute     ┌───────────┐
│  Agent  │ ◄─────────────────  │ LlmClient │ ◄──────────────  │   Tools   │
└─────────┘   (continuation)    └───────────┘    (via adapter) └───────────┘
```

## Agent Patterns

### Simple Prompt (PromptAgent)

Single or multi-turn LLM calls with optional tools:

```rust
let request = context.llm.client().interaction()
    .with_system_instruction(&system_prompt)
    .with_text(&query)
    .add_functions(&declarations)
    .with_store_enabled()
    .build();
let response = context.llm.generate(request).await?;

// If function calls returned, execute and continue
if !response.function_calls().is_empty() {
    // Execute tools, collect results
    let continuation = context.llm.client().interaction()
        .with_previous_interaction(id)
        .add_functions(&declarations)
        .with_content(results)
        .with_store_enabled()
        .build();
    let response = context.llm.generate(continuation).await?;
}
```

### Complex Orchestration (DeveloperAgent)

Fine-grained control with batch confirmation:

```rust
// Uses client() for optimized continuation (doesn't re-send functions)
let genai_client = context.llm.client();

loop {
    let response = genai_client.interaction()
        .with_previous_interaction(prev_id)
        .with_content(function_results)  // No functions re-sent
        .create()
        .await?;
    // ...
}
```

### Research Pattern (DeepResearchAgent)

Decompose, parallel execute, synthesize:

```rust
// 1. Decompose query into sub-queries
let sub_queries = decompose(query, context).await?;

// 2. Execute sub-queries in parallel
let results = join_all(sub_queries.iter().map(|q| execute(q, context))).await;

// 3. Synthesize final answer
let answer = synthesize(results, context).await?;
```

## Configuration Boundaries

| Layer | Configuration Source |
|-------|---------------------|
| LlmClient | `LlmConfig` (timeout, max_tokens, temperature) |
| Agent | Agent-specific config in constructor |
| Tool | Per-tool parameters in call arguments |
| Interceptor | Interceptor-specific config at registration |

Agents receive minimal `AgentContext`. Agent-specific config belongs in constructors, not context.

## See Also

- [Agent Authoring Guide](AGENT_AUTHORING.md) - How to build agents
- [Tool Authoring Guide](TOOL_AUTHORING.md) - How to build tools
- [Interceptor Authoring Guide](INTERCEPTOR_AUTHORING.md) - How to build interceptors
