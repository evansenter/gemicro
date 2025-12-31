# Hook Authoring Guide

This guide walks you through implementing a new hook in Gemicro, following the [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) philosophy of extensible design.

## Overview

### What is a Hook?

A hook in Gemicro intercepts tool execution for cross-cutting concerns:
- **Validation**: Block dangerous operations before execution
- **Logging**: Record tool invocations for audit trails
- **Metrics**: Collect usage statistics
- **Security**: Enforce access controls

Hooks run at the adapter layer, intercepting `CallableFunction::call()` in rust-genai's automatic function calling flow.

### Design Philosophy

Following Evergreen principles:

| Principle | Description |
|-----------|-------------|
| **One hook per crate** | Each hook is an independent crate in `hooks/` |
| **Pre/Post execution** | Pre-hooks can deny or modify, post-hooks are observability-only |
| **Chain composition** | Multiple hooks run in registration order |
| **Graceful degradation** | Hook failures should not crash the system |

## Quick Start Checklist

- [ ] Create new hook crate: `hooks/gemicro-{hook-name}/`
- [ ] Add crate to workspace `Cargo.toml` members
- [ ] Implement `ToolHook` trait (`pre_tool_use`, `post_tool_use`)
- [ ] Choose appropriate struct pattern (unit, config, stateful)
- [ ] Implement `Clone` and `Debug` traits
- [ ] Add `#[non_exhaustive]` to public structs
- [ ] Add unit tests for all decision paths
- [ ] Add doc tests for public API

## Core Types

### ToolHook Trait

Location: `gemicro-core/src/tool/hooks.rs`

```rust
#[async_trait]
pub trait ToolHook: Send + Sync + fmt::Debug {
    /// Called before tool execution.
    /// Returns: Allow | AllowWithModifiedInput(Value) | Deny { reason } | RequestPermission { message }
    async fn pre_tool_use(
        &self,
        tool_name: &str,
        input: &Value,
    ) -> Result<HookDecision, HookError>;

    /// Called after tool execution (observability only).
    async fn post_tool_use(
        &self,
        tool_name: &str,
        input: &Value,
        output: &ToolResult,
    ) -> Result<(), HookError>;
}
```

**Note:** While `Clone` is not required by the trait, all built-in hooks implement it for sharing across registries. It is strongly recommended that custom hooks also implement `Clone`.

### HookDecision

```rust
pub enum HookDecision {
    /// Allow execution with original input
    Allow,

    /// Allow but modify the input first
    AllowWithModifiedInput(Value),

    /// Deny execution with a reason
    Deny { reason: String },

    /// Request user permission before proceeding
    RequestPermission { message: String },
}
```

### HookError

```rust
pub enum HookError {
    /// Hook logic failed
    ExecutionFailed(String),

    /// Other error
    Other(String),
}
```

## Hook Struct Patterns

Choose the pattern that fits your hook's needs:

### 1. Unit Struct (Stateless, No Config)

Use when the hook has no configuration or state.

```rust
// hooks/gemicro-audit-log/src/lib.rs

/// Logs all tool invocations for audit purposes.
#[derive(Debug, Clone, Copy, Default)]
pub struct AuditLog;

#[async_trait]
impl ToolHook for AuditLog {
    async fn pre_tool_use(&self, tool_name: &str, input: &Value)
        -> Result<HookDecision, HookError>
    {
        log::info!("Tool '{}' invoked with input: {:?}", tool_name, input);
        Ok(HookDecision::Allow)
    }

    async fn post_tool_use(&self, tool_name: &str, _input: &Value, output: &ToolResult)
        -> Result<(), HookError>
    {
        log::info!("Tool '{}' completed: {:?}", tool_name, output.content);
        Ok(())
    }
}
```

### 2. Config Struct (Configuration, No State)

Use when the hook needs configuration but no runtime state.

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
impl ToolHook for FileSecurity {
    async fn pre_tool_use(&self, tool_name: &str, input: &Value)
        -> Result<HookDecision, HookError>
    {
        if !matches!(tool_name, "file_write" | "file_edit") {
            return Ok(HookDecision::Allow);
        }

        let path = match input.get("path").and_then(|p| p.as_str()) {
            Some(p) => p,
            None => {
                log::warn!("FileSecurity: {} missing 'path' parameter", tool_name);
                return Ok(HookDecision::Allow);
            }
        };

        if self.is_blocked(path) {
            log::warn!("FileSecurity: BLOCKED write to '{}'", path);
            return Ok(HookDecision::Deny {
                reason: format!("Writing to '{}' is blocked by security policy", path),
            });
        }

        Ok(HookDecision::Allow)
    }

    async fn post_tool_use(&self, _: &str, _: &Value, _: &ToolResult)
        -> Result<(), HookError>
    {
        Ok(())
    }
}
```

### 3. Builder Pattern (Complex Config)

Use when the hook has multiple optional configuration fields.

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

Use when the hook tracks state across invocations.

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

    /// Reset all metrics.
    pub fn reset(&self) {
        let mut tools = match self.tools.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        tools.clear();
    }
}

#[async_trait]
impl ToolHook for Metrics {
    async fn pre_tool_use(&self, tool_name: &str, _input: &Value)
        -> Result<HookDecision, HookError>
    {
        self.record_invocation(tool_name);
        Ok(HookDecision::Allow)
    }

    async fn post_tool_use(&self, tool_name: &str, _input: &Value, output: &ToolResult)
        -> Result<(), HookError>
    {
        let is_error = output.metadata.get("error").is_some();
        if is_error {
            self.record_failure(tool_name);
        } else {
            self.record_success(tool_name);
        }
        Ok(())
    }
}
```

## Hook Execution Order

Multiple hooks run in registration order:

```
pre_hook_1 → pre_hook_2 → ... → EXECUTE → post_hook_1 → post_hook_2 → ...
```

- First `Deny` stops the chain and prevents execution
- First `AllowWithModifiedInput` modifies input for subsequent hooks
- Post-hooks run in the same order as pre-hooks (registration order)
- Post-hooks run even if earlier post-hooks fail

## Registration and Usage

### Creating a HookRegistry

```rust
use gemicro_core::tool::HookRegistry;
use gemicro_audit_log::AuditLog;
use gemicro_file_security::FileSecurity;
use gemicro_metrics::Metrics;
use std::path::PathBuf;
use std::sync::Arc;

let metrics = Metrics::new();

let hooks = Arc::new(
    HookRegistry::new()
        .with_hook(AuditLog)
        .with_hook(FileSecurity::new(vec![PathBuf::from("/etc")]))
        .with_hook(metrics.clone())
);

// Later: get metrics
let snapshot = metrics.snapshot();
```

### Integration with GemicroToolService

```rust
use gemicro_core::tool::{GemicroToolService, ToolRegistry, AutoApprove};
use std::sync::Arc;

let registry = Arc::new(ToolRegistry::new());
let hooks = Arc::new(HookRegistry::new().with_hook(AuditLog));

let service = GemicroToolService::new(registry)
    .with_hooks(hooks)
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
    use serde_json::json;

    #[tokio::test]
    async fn test_allows_safe_path() {
        let hook = FileSecurity::new(vec![PathBuf::from("/etc")]);
        let input = json!({"path": "/home/user/safe.txt"});

        let decision = hook.pre_tool_use("file_write", &input).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);
    }

    #[tokio::test]
    async fn test_blocks_dangerous_path() {
        let hook = FileSecurity::new(vec![PathBuf::from("/etc")]);
        let input = json!({"path": "/etc/passwd"});

        let decision = hook.pre_tool_use("file_write", &input).await.unwrap();
        assert!(matches!(decision, HookDecision::Deny { .. }));
    }

    #[tokio::test]
    async fn test_ignores_non_write_tools() {
        let hook = FileSecurity::new(vec![PathBuf::from("/etc")]);
        let input = json!({"path": "/etc/passwd"});

        let decision = hook.pre_tool_use("file_read", &input).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);
    }

    #[test]
    #[should_panic(expected = "requires at least one")]
    fn test_panics_on_empty_config() {
        FileSecurity::new(vec![]);
    }
}
```

### Testing Stateful Hooks

```rust
#[tokio::test]
async fn test_metrics_records_invocation() {
    let metrics = Metrics::new();

    metrics.pre_tool_use("test", &json!({})).await.unwrap();

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.total_invocations(), 1);
}

#[tokio::test]
async fn test_metrics_tracks_success_and_failure() {
    let metrics = Metrics::new();

    // Success
    metrics.pre_tool_use("tool", &json!({})).await.unwrap();
    metrics.post_tool_use("tool", &json!({}), &ToolResult::text("ok"))
        .await.unwrap();

    // Failure
    metrics.pre_tool_use("tool", &json!({})).await.unwrap();
    metrics.post_tool_use("tool", &json!({}),
        &ToolResult::text("err").with_metadata(json!({"error": true})))
        .await.unwrap();

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.get("tool").unwrap().successes, 1);
    assert_eq!(snapshot.get("tool").unwrap().failures, 1);
}
```

## Common Pitfalls

### 1. Panicking in hooks

```rust
// ❌ DON'T - panics crash the whole system
.unwrap()

// ✅ DO - handle errors gracefully
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
// ❌ DON'T - silently allow when path missing
let path = input.get("path").and_then(|p| p.as_str());
if path.is_none() { return Ok(HookDecision::Allow); }

// ✅ DO - log when falling back
let path = match input.get("path").and_then(|p| p.as_str()) {
    Some(p) => p,
    None => {
        log::warn!("Missing 'path' parameter, allowing");
        return Ok(HookDecision::Allow);
    }
};
```

### 3. Missing audit logging for security decisions

```rust
// ❌ DON'T - no record of blocked operations
return Ok(HookDecision::Deny { reason: "blocked" });

// ✅ DO - log security-relevant decisions
log::warn!("BLOCKED: write to '{}' by tool '{}'", path, tool_name);
return Ok(HookDecision::Deny {
    reason: format!("Writing to '{}' is blocked", path),
});
```

### 4. Not implementing Clone

```rust
// ❌ DON'T - hooks must be cloneable for sharing
pub struct MyHook {
    data: Mutex<Data>,  // Mutex is !Clone
}

// ✅ DO - wrap in Arc for shared state
pub struct MyHook {
    data: Arc<RwLock<Data>>,  // Arc is Clone
}

impl Clone for MyHook {
    fn clone(&self) -> Self {
        Self { data: Arc::clone(&self.data) }
    }
}
```

## File Structure for New Hooks

Each hook gets its own crate in the `hooks/` subdirectory:

```
hooks/
└── gemicro-my-hook/
    ├── Cargo.toml           # Depends on gemicro-core only
    └── src/
        └── lib.rs           # Hook implementation
```

**Cargo.toml template:**

```toml
[package]
name = "gemicro-my-hook"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true

description = "Brief description of what the hook does"

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
    "hooks/gemicro-my-hook",
]
```

## Built-in Hook Crates

| Crate | Purpose | Pattern |
|-------|---------|---------|
| `gemicro-audit-log` | Log all tool invocations | Unit struct |
| `gemicro-file-security` | Block writes to sensitive paths | Config struct |
| `gemicro-input-sanitizer` | Enforce input size limits | Config struct |
| `gemicro-conditional-permission` | Dynamic permission prompts | Builder pattern |
| `gemicro-metrics` | Track tool usage metrics | Stateful struct |

## Security Documentation

When creating security-related hooks, document limitations explicitly:

```rust
/// # Security Warnings
///
/// **This hook has known bypass vulnerabilities:**
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
- `gemicro-core/src/tool/hooks.rs` - Core trait definitions
- `docs/TOOL_AUTHORING.md` - Creating tools that hooks intercept
- `CLAUDE.md` - Project design philosophy
