# Interceptor Authoring Guide

This guide walks you through implementing a new interceptor in Gemicro, following the [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) philosophy of extensible design.

## Overview

### What is an Interceptor?

An interceptor in Gemicro intercepts execution for cross-cutting concerns:
- **Validation**: Block dangerous operations before execution
- **Transformation**: Sanitize or modify input before execution
- **Logging**: Record invocations for audit trails
- **Metrics**: Collect usage statistics
- **Security**: Enforce access controls

Interceptors are generic over input/output types, enabling reuse across tool calls, user messages, and external events.

### Design Philosophy

Following Evergreen principles:

| Principle | Description |
|-----------|-------------|
| **One interceptor per crate** | Each interceptor is an independent crate in `hooks/` |
| **intercept/observe lifecycle** | `intercept()` can deny or transform, `observe()` is observability-only |
| **Chain composition** | Multiple interceptors run in registration order |
| **Graceful degradation** | Interceptor failures should not crash the system |

## Quick Start Checklist

- [ ] Create new interceptor crate: `hooks/gemicro-{interceptor-name}/`
- [ ] Add crate to workspace `Cargo.toml` members
- [ ] Implement `Interceptor<ToolCall, ToolResult>` trait (`intercept`, optionally `observe`)
- [ ] Choose appropriate struct pattern (unit, config, stateful)
- [ ] Implement `Clone` and `Debug` traits
- [ ] Add `#[non_exhaustive]` to public structs
- [ ] Add unit tests for all decision paths
- [ ] Add doc tests for public API

## Core Types

### Interceptor Trait

Location: `gemicro-core/src/interceptor/mod.rs`

```rust
#[async_trait]
pub trait Interceptor<In, Out>: Send + Sync + fmt::Debug
where
    In: Send + Sync,
    Out: Send + Sync,
{
    /// Called before execution. Returns decision on how to proceed.
    async fn intercept(&self, input: &In) -> Result<InterceptDecision<In>, InterceptError>;

    /// Called after execution (observability only). Default: no-op.
    async fn observe(&self, input: &In, output: &Out) -> Result<(), InterceptError> {
        let _ = (input, output);
        Ok(())
    }
}
```

**Note:** While `Clone` is not required by the trait, all built-in interceptors implement it for sharing across chains. It is strongly recommended that custom interceptors also implement `Clone`.

### InterceptDecision

```rust
pub enum InterceptDecision<T> {
    /// Allow execution with original input
    Allow,

    /// Allow but transform the input first
    Transform(T),

    /// Request user confirmation before proceeding
    Confirm { message: String },

    /// Deny execution with a reason
    Deny { reason: String },
}
```

### InterceptError

```rust
pub enum InterceptError {
    /// Interceptor logic failed
    ExecutionFailed(String),

    /// Input modification failed
    InvalidModification(String),

    /// Other error
    Other(Box<dyn std::error::Error + Send + Sync>),
}
```

### ToolCall

The input type for tool interceptors:

```rust
pub struct ToolCall {
    /// The name of the tool being called
    pub name: String,

    /// The arguments passed to the tool
    pub arguments: Value,
}

impl ToolCall {
    pub fn new(name: impl Into<String>, arguments: Value) -> Self { ... }
}
```

## Interceptor Struct Patterns

Choose the pattern that fits your interceptor's needs:

### 1. Unit Struct (Stateless, No Config)

Use when the interceptor has no configuration or state.

```rust
// hooks/gemicro-audit-log/src/lib.rs

/// Logs all tool invocations for audit purposes.
#[derive(Debug, Clone, Copy, Default)]
pub struct AuditLog;

#[async_trait]
impl Interceptor<ToolCall, ToolResult> for AuditLog {
    async fn intercept(&self, input: &ToolCall)
        -> Result<InterceptDecision<ToolCall>, InterceptError>
    {
        log::info!("Tool '{}' invoked with input: {:?}", input.name, input.arguments);
        Ok(InterceptDecision::Allow)
    }

    async fn observe(&self, input: &ToolCall, output: &ToolResult)
        -> Result<(), InterceptError>
    {
        log::info!("Tool '{}' completed: {:?}", input.name, output.content);
        Ok(())
    }
}
```

### 2. Config Struct (Configuration, No State)

Use when the interceptor needs configuration but no runtime state.

```rust
// hooks/gemicro-file-security/src/lib.rs

/// Blocks writes to sensitive paths.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FileSecurity {
    /// Paths that are blocked from write operations.
    pub blocked_paths: Vec<PathBuf>,
}

impl FileSecurity {
    /// Create with blocked paths.
    ///
    /// # Panics
    /// Panics if `blocked_paths` is empty.
    pub fn new(blocked_paths: Vec<PathBuf>) -> Self {
        if blocked_paths.is_empty() {
            panic!("FileSecurity requires at least one blocked path");
        }
        Self { blocked_paths }
    }

    fn is_blocked(&self, path: &str) -> bool {
        let path = PathBuf::from(path);
        self.blocked_paths.iter().any(|blocked| path.starts_with(blocked))
    }
}

#[async_trait]
impl Interceptor<ToolCall, ToolResult> for FileSecurity {
    async fn intercept(&self, input: &ToolCall)
        -> Result<InterceptDecision<ToolCall>, InterceptError>
    {
        if !matches!(input.name.as_str(), "file_write" | "file_edit") {
            return Ok(InterceptDecision::Allow);
        }

        let path = match input.arguments.get("path").and_then(|p| p.as_str()) {
            Some(p) => p,
            None => {
                log::error!("FileSecurity: {} missing 'path' parameter - DENYING", input.name);
                return Ok(InterceptDecision::Deny {
                    reason: format!("Tool '{}' missing required 'path' parameter", input.name),
                });
            }
        };

        if self.is_blocked(path) {
            log::warn!("FileSecurity: BLOCKED write to '{}'", path);
            return Ok(InterceptDecision::Deny {
                reason: format!("Writing to '{}' is blocked by security policy", path),
            });
        }

        Ok(InterceptDecision::Allow)
    }
}
```

### 3. Builder Pattern (Complex Config)

Use when the interceptor has multiple optional configuration fields.

```rust
// hooks/gemicro-conditional-permission/src/lib.rs

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ConditionalPermission {
    pub dangerous_patterns: Vec<String>,
    pub fields_to_check: Vec<String>,
}

impl ConditionalPermission {
    pub fn new(dangerous_patterns: Vec<String>) -> Self {
        // ... validation ...
        Self {
            dangerous_patterns,
            fields_to_check: vec!["command".into(), "query".into()],
        }
    }

    pub fn builder() -> ConditionalPermissionBuilder {
        ConditionalPermissionBuilder::default()
    }
}

#[derive(Debug, Default)]
pub struct ConditionalPermissionBuilder {
    patterns: Vec<String>,
    fields: Vec<String>,
}

impl ConditionalPermissionBuilder {
    #[must_use]
    pub fn add_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.patterns.push(pattern.into());
        self
    }

    #[must_use]
    pub fn check_field(mut self, field: impl Into<String>) -> Self {
        self.fields.push(field.into());
        self
    }

    pub fn build(self) -> ConditionalPermission {
        // ... validation and construction ...
    }
}
```

### 4. Stateful Struct (Runtime State)

Use when the interceptor tracks state across invocations.

```rust
// hooks/gemicro-metrics/src/lib.rs

/// Collects tool usage metrics.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Metrics {
    // Private fields for internal state
    tools: Arc<RwLock<HashMap<String, ToolStats>>>,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get a snapshot of current metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        // Handle lock poisoning gracefully
        let tools = match self.tools.read() {
            Ok(guard) => guard,
            Err(poisoned) => {
                log::warn!("Metrics lock poisoned, recovering");
                poisoned.into_inner()
            }
        };
        // ... build snapshot ...
    }
}

#[async_trait]
impl Interceptor<ToolCall, ToolResult> for Metrics {
    async fn intercept(&self, input: &ToolCall)
        -> Result<InterceptDecision<ToolCall>, InterceptError>
    {
        self.record_invocation(&input.name);
        Ok(InterceptDecision::Allow)
    }

    async fn observe(&self, input: &ToolCall, output: &ToolResult)
        -> Result<(), InterceptError>
    {
        let is_error = output.metadata.get("error").is_some();
        if is_error {
            self.record_failure(&input.name);
        } else {
            self.record_success(&input.name);
        }
        Ok(())
    }
}
```

## Interceptor Execution Order

Multiple interceptors run in registration order:

```
intercept_1 → intercept_2 → ... → EXECUTE → observe_1 → observe_2 → ...
```

- First `Deny` or `Confirm` stops the chain
- `Transform` modifies input for subsequent interceptors
- `observe()` runs in the same order as `intercept()` (registration order)
- `observe()` runs even if earlier observers fail

## Registration and Usage

### Creating an InterceptorChain

```rust
use gemicro_core::interceptor::InterceptorChain;
use gemicro_audit_log::AuditLog;
use gemicro_file_security::FileSecurity;
use gemicro_metrics::Metrics;
use std::path::PathBuf;
use std::sync::Arc;

let metrics = Metrics::new();

let interceptors = Arc::new(
    InterceptorChain::new()
        .with(AuditLog)
        .with(FileSecurity::new(vec![PathBuf::from("/etc")]))
        .with(metrics.clone())
);

// Later: get metrics
let snapshot = metrics.snapshot();
```

### Integration with GemicroToolService

```rust
use gemicro_core::interceptor::InterceptorChain;
use gemicro_core::tool::{GemicroToolService, ToolRegistry, AutoApprove};
use std::sync::Arc;

let registry = Arc::new(ToolRegistry::new());
let interceptors = Arc::new(InterceptorChain::new().with(AuditLog));

let service = GemicroToolService::new(registry)
    .with_interceptors(interceptors)
    .with_confirmation_handler(Arc::new(AutoApprove));

// Use with rust-genai
client.interaction()
    .with_tool_service(Arc::new(service))
    .create_with_auto_functions()
    .await?;
```

## Testing Patterns

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use gemicro_core::interceptor::ToolCall;
    use serde_json::json;

    #[tokio::test]
    async fn test_allows_safe_path() {
        let interceptor = FileSecurity::new(vec![PathBuf::from("/etc")]);
        let input = ToolCall::new("file_write", json!({"path": "/home/user/safe.txt"}));

        let decision = interceptor.intercept(&input).await.unwrap();
        assert_eq!(decision, InterceptDecision::Allow);
    }

    #[tokio::test]
    async fn test_blocks_dangerous_path() {
        let interceptor = FileSecurity::new(vec![PathBuf::from("/etc")]);
        let input = ToolCall::new("file_write", json!({"path": "/etc/passwd"}));

        let decision = interceptor.intercept(&input).await.unwrap();
        assert!(matches!(decision, InterceptDecision::Deny { .. }));
    }

    #[tokio::test]
    async fn test_ignores_non_write_tools() {
        let interceptor = FileSecurity::new(vec![PathBuf::from("/etc")]);
        let input = ToolCall::new("file_read", json!({"path": "/etc/passwd"}));

        let decision = interceptor.intercept(&input).await.unwrap();
        assert_eq!(decision, InterceptDecision::Allow);
    }

    #[test]
    #[should_panic(expected = "requires at least one")]
    fn test_panics_on_empty_config() {
        FileSecurity::new(vec![]);
    }
}
```

### Testing Stateful Interceptors

```rust
#[tokio::test]
async fn test_metrics_records_invocation() {
    let metrics = Metrics::new();
    let input = ToolCall::new("test", json!({}));

    metrics.intercept(&input).await.unwrap();

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.total_invocations(), 1);
}

#[tokio::test]
async fn test_metrics_tracks_success_and_failure() {
    let metrics = Metrics::new();
    let input = ToolCall::new("tool", json!({}));

    // Success
    metrics.intercept(&input).await.unwrap();
    metrics.observe(&input, &ToolResult::text("ok")).await.unwrap();

    // Failure
    metrics.intercept(&input).await.unwrap();
    metrics.observe(&input,
        &ToolResult::text("err").with_metadata(json!({"error": true})))
        .await.unwrap();

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.get("tool").unwrap().successes, 1);
    assert_eq!(snapshot.get("tool").unwrap().failures, 1);
}
```

## Common Pitfalls

### 1. Panicking in interceptors

```rust
// DON'T - panics crash the whole system
.unwrap()

// DO - handle errors gracefully
match self.tools.read() {
    Ok(guard) => guard,
    Err(poisoned) => {
        log::warn!("Lock poisoned, recovering");
        poisoned.into_inner()
    }
}
```

### 2. Silent fallbacks

```rust
// DON'T - silently allow when path missing
let path = input.arguments.get("path").and_then(|p| p.as_str());
if path.is_none() { return Ok(InterceptDecision::Allow); }

// DO - log when falling back, or deny for security-critical checks
let path = match input.arguments.get("path").and_then(|p| p.as_str()) {
    Some(p) => p,
    None => {
        log::error!("Missing 'path' parameter, denying for safety");
        return Ok(InterceptDecision::Deny {
            reason: "Missing required 'path' parameter".into(),
        });
    }
};
```

### 3. Missing audit logging for security decisions

```rust
// DON'T - no record of blocked operations
return Ok(InterceptDecision::Deny { reason: "blocked".into() });

// DO - log security-relevant decisions
log::warn!("BLOCKED: write to '{}' by tool '{}'", path, input.name);
return Ok(InterceptDecision::Deny {
    reason: format!("Writing to '{}' is blocked", path),
});
```

### 4. Not implementing Clone

```rust
// DON'T - interceptors must be cloneable for sharing
pub struct MyInterceptor {
    data: Mutex<Data>,  // Mutex is !Clone
}

// DO - wrap in Arc for shared state
pub struct MyInterceptor {
    data: Arc<RwLock<Data>>,  // Arc is Clone
}

impl Clone for MyInterceptor {
    fn clone(&self) -> Self {
        Self { data: Arc::clone(&self.data) }
    }
}
```

## File Structure for New Interceptors

Each interceptor gets its own crate in the `hooks/` subdirectory:

```
hooks/
└── gemicro-my-interceptor/
    ├── Cargo.toml           # Depends on gemicro-core only
    └── src/
        └── lib.rs           # Interceptor implementation
```

**Cargo.toml template:**

```toml
[package]
name = "gemicro-my-interceptor"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true

description = "Brief description of what the interceptor does"

[dependencies]
gemicro-core = { path = "../../gemicro-core" }
async-trait.workspace = true
log.workspace = true
serde_json.workspace = true

[dev-dependencies]
tokio = { workspace = true, features = ["rt", "macros"] }
```

**Add to workspace Cargo.toml:**

```toml
[workspace]
members = [
    # ... existing members
    "hooks/gemicro-my-interceptor",
]
```

## Built-in Interceptor Crates

| Crate | Purpose | Pattern |
|-------|---------|---------|
| `gemicro-audit-log` | Log all tool invocations | Unit struct |
| `gemicro-file-security` | Block writes to sensitive paths | Config struct |
| `gemicro-input-sanitizer` | Enforce input size limits | Config struct |
| `gemicro-conditional-permission` | Dynamic permission prompts | Builder pattern |
| `gemicro-metrics` | Track tool usage metrics | Stateful struct |

## Security Documentation

When creating security-related interceptors, document limitations explicitly:

```rust
/// # Security Warnings
///
/// **This interceptor has known bypass vulnerabilities:**
///
/// ## Known Bypass Methods
///
/// 1. **Symlink attacks**: `/tmp/evil -> /etc/passwd` bypasses path checks
/// 2. **Path traversal**: `../../etc/passwd` may bypass prefix matching
/// 3. **Case sensitivity**: `/ETC/passwd` may bypass on some filesystems
///
/// ## Recommendations
///
/// - Canonicalize paths before checking
/// - Combine with OS-level controls
/// - Use as defense-in-depth, not sole protection
```

## See Also

- `hooks/gemicro-audit-log/src/lib.rs` - Unit struct example
- `hooks/gemicro-file-security/src/lib.rs` - Config struct example
- `hooks/gemicro-metrics/src/lib.rs` - Stateful struct example
- `hooks/gemicro-conditional-permission/src/lib.rs` - Builder pattern example
- `gemicro-core/src/interceptor/mod.rs` - Core trait definitions
- `docs/TOOL_AUTHORING.md` - Creating tools that interceptors intercept
- `CLAUDE.md` - Project design philosophy
