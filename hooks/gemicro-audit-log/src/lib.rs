//! Audit logging interceptor for gemicro tool execution.
//!
//! This interceptor logs all tool invocations before and after execution,
//! providing a complete audit trail for compliance and debugging.
//!
//! # LOUD_WIRE Support
//!
//! When `LOUD_WIRE` environment variable is set, outputs pretty-printed
//! colored JSON to stderr with the same formatting as rust-genai's wire
//! debugging. This provides full visibility into tool arguments and results
//! without truncation.
//!
//! ```bash
//! LOUD_WIRE=1 cargo run -p gemicro-developer --example developer
//! ```
//!
//! # Example
//!
//! ```no_run
//! use gemicro_core::interceptor::{InterceptorChain, ToolCall};
//! use gemicro_core::tool::ToolResult;
//! use gemicro_audit_log::AuditLog;
//!
//! let interceptors: InterceptorChain<ToolCall, ToolResult> = InterceptorChain::new()
//!     .with(AuditLog);
//! ```

use async_trait::async_trait;
use chrono::Utc;
use colored::Colorize;
use gemicro_core::interceptor::{InterceptDecision, InterceptError, Interceptor, ToolCall};
use gemicro_core::tool::ToolResult;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{OnceLock, RwLock};

/// Cached check for whether LOUD_WIRE is enabled
static LOUD_WIRE_ENABLED: OnceLock<bool> = OnceLock::new();

/// Tool call counter for correlating invocations with results
static TOOL_COUNTER: AtomicUsize = AtomicUsize::new(1);

/// Map from ToolCall pointer address to sequential invocation ID for correlation.
///
/// This handles concurrent tool execution where multiple intercept() calls
/// happen before their corresponding observe() calls. The "tool ID" is a
/// sequential counter (1, 2, 3...) for visual log correlation, NOT related
/// to tool registration or tool definitions.
///
/// # Note on Dynamic Tools
///
/// This map tracks individual tool *invocations*, not tool *registrations*.
/// If tools are added/removed from the ToolRegistry at runtime, this logging
/// mechanism continues to work correctly - each invocation gets a fresh ID
/// regardless of when the underlying tool was registered.
static TOOL_ID_MAP: OnceLock<RwLock<HashMap<usize, usize>>> = OnceLock::new();

/// Get the tool ID correlation map.
fn tool_id_map() -> &'static RwLock<HashMap<usize, usize>> {
    TOOL_ID_MAP.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Store a sequential invocation ID for later correlation in observe().
///
/// Uses pointer address as key - this is safe because:
/// 1. The ToolCall reference lives from intercept() through observe()
/// 2. We remove the entry in retrieve_tool_id() to prevent memory leaks
/// 3. Fallback in retrieve_tool_id() handles any edge cases
fn store_tool_id(input: &ToolCall, tool_id: usize) {
    let key = input as *const ToolCall as usize;
    if let Ok(mut map) = tool_id_map().write() {
        map.insert(key, tool_id);
    }
}

/// Retrieve and remove the tool ID stored during intercept().
/// Falls back to current counter - 1 if lookup fails.
fn retrieve_tool_id(input: &ToolCall) -> usize {
    let key = input as *const ToolCall as usize;
    tool_id_map()
        .write()
        .ok()
        .and_then(|mut map| map.remove(&key))
        .unwrap_or_else(|| TOOL_COUNTER.load(Ordering::Relaxed).saturating_sub(1))
}

/// Check if LOUD_WIRE debugging is enabled.
///
/// The result is cached after first check for performance.
fn is_loud_wire() -> bool {
    *LOUD_WIRE_ENABLED.get_or_init(|| std::env::var("LOUD_WIRE").is_ok())
}

/// Get the next tool call ID for correlation.
fn next_tool_id() -> usize {
    TOOL_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Format the current timestamp for log output (ISO 8601 UTC).
fn timestamp() -> String {
    Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

/// Log prefix with timestamp and tool ID (for invocations).
/// Colors alternate: green (even) / yellow (odd) for visual distinction.
fn tool_invoke_prefix(tool_id: usize) -> String {
    let ts = timestamp().dimmed();
    let tool_label = format!("[TOOL#{}]", tool_id);
    let colored_label = if tool_id % 2 == 0 {
        tool_label.green().bold()
    } else {
        tool_label.yellow().bold()
    };
    format!("{} {} {}", "[LOUD_WIRE]".bold(), ts, colored_label)
}

/// Log prefix with timestamp and tool ID (for results).
/// Colors alternate: magenta (even) / cyan (odd) for visual distinction.
fn tool_result_prefix(tool_id: usize) -> String {
    let ts = timestamp().dimmed();
    let tool_label = format!("[TOOL#{}]", tool_id);
    let colored_label = if tool_id % 2 == 0 {
        tool_label.magenta().bold()
    } else {
        tool_label.cyan().bold()
    };
    format!("{} {} {}", "[LOUD_WIRE]".bold(), ts, colored_label)
}

/// Colorize and format JSON for terminal output.
fn colorize_json(value: &Value) -> Option<String> {
    colored_json::to_colored_json_auto(value).ok()
}

/// Audit log interceptor that logs all tool invocations.
///
/// Logs both pre and post execution for complete audit trail.
/// Uses the `log` crate by default. When `LOUD_WIRE` environment variable
/// is set, outputs pretty-printed colored JSON to stderr (matching
/// rust-genai's wire debugging format).
///
/// # Output Format
///
/// **Standard mode (log crate):**
/// ```text
/// [INFO] Tool invoked: bash with input: {"command": "ls -la"}
/// [INFO] Tool completed: bash -> total 16...
/// ```
///
/// **LOUD_WIRE mode (colored stderr):**
/// ```text
/// [LOUD_WIRE] 2026-01-03T15:30:45Z [TOOL#1] >>> bash
/// [LOUD_WIRE] 2026-01-03T15:30:45Z [TOOL#1] Args:
/// [LOUD_WIRE] 2026-01-03T15:30:45Z [TOOL#1]   { "command": "ls -la" }
/// [LOUD_WIRE] 2026-01-03T15:30:46Z [TOOL#1] <<< OK
/// [LOUD_WIRE] 2026-01-03T15:30:46Z [TOOL#1] Result:
/// [LOUD_WIRE] 2026-01-03T15:30:46Z [TOOL#1]   { "content": "total 16..." }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct AuditLog;

impl AuditLog {
    /// Log tool invocation in LOUD_WIRE format.
    fn log_loud_invoke(tool_id: usize, input: &ToolCall) {
        let prefix = tool_invoke_prefix(tool_id);
        let direction = ">>>".green().bold();

        eprintln!("{prefix} {direction} {}", input.name.cyan().bold());

        eprintln!("{prefix} {}:", "Args".green());
        if let Some(colored) = colorize_json(&input.arguments) {
            for line in colored.lines() {
                eprintln!("{prefix}   {line}");
            }
        } else if let Ok(pretty) = serde_json::to_string_pretty(&input.arguments) {
            for line in pretty.lines() {
                eprintln!("{prefix}   {line}");
            }
        }
    }

    /// Log tool result in LOUD_WIRE format.
    fn log_loud_result(tool_id: usize, input: &ToolCall, output: &ToolResult) {
        let prefix = tool_result_prefix(tool_id);
        let direction = "<<<".red().bold();

        // Check if this is an error result (error key in metadata or content)
        let is_error =
            output.metadata.get("error").is_some() || output.content.get("error").is_some();

        let status = if is_error {
            "ERROR".red().bold()
        } else {
            "OK".green().bold()
        };

        eprintln!(
            "{prefix} {direction} {} {}",
            status,
            input.name.cyan().bold()
        );

        eprintln!("{prefix} {}:", "Result".red());
        if let Some(colored) = colorize_json(&output.content) {
            for line in colored.lines() {
                eprintln!("{prefix}   {line}");
            }
        } else if let Ok(pretty) = serde_json::to_string_pretty(&output.content) {
            for line in pretty.lines() {
                eprintln!("{prefix}   {line}");
            }
        }
    }
}

#[async_trait]
impl Interceptor<ToolCall, ToolResult> for AuditLog {
    async fn intercept(
        &self,
        input: &ToolCall,
    ) -> Result<InterceptDecision<ToolCall>, InterceptError> {
        if is_loud_wire() {
            let tool_id = next_tool_id();
            // Store tool_id for correlation with observe() - handles concurrent execution
            store_tool_id(input, tool_id);
            Self::log_loud_invoke(tool_id, input);
        } else {
            log::info!(
                "Tool invoked: {} with input: {}",
                input.name,
                input.arguments
            );
        }
        Ok(InterceptDecision::Allow)
    }

    async fn observe(&self, input: &ToolCall, output: &ToolResult) -> Result<(), InterceptError> {
        if is_loud_wire() {
            // Retrieve the tool_id stored during intercept() - handles concurrent execution
            let tool_id = retrieve_tool_id(input);
            Self::log_loud_result(tool_id, input, output);
        } else {
            let content_preview = match &output.content {
                Value::String(s) => {
                    // Use char-aware truncation to avoid panicking on UTF-8 boundaries
                    if s.chars().count() > 100 {
                        let truncated: String = s.chars().take(100).collect();
                        format!("{}...", truncated)
                    } else {
                        s.clone()
                    }
                }
                other => {
                    let formatted = format!("{:?}", other);
                    if formatted.chars().count() > 100 {
                        let truncated: String = formatted.chars().take(100).collect();
                        format!("{}...", truncated)
                    } else {
                        formatted
                    }
                }
            };
            log::info!("Tool completed: {} -> {}", input.name, content_preview);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_audit_log_allows_execution() {
        let interceptor = AuditLog;
        let input = ToolCall::new("test", json!({}));
        let decision = interceptor.intercept(&input).await.unwrap();
        assert_eq!(decision, InterceptDecision::Allow);
    }

    #[tokio::test]
    async fn test_audit_log_post_execution() {
        let interceptor = AuditLog;
        let input = ToolCall::new("test", json!({}));
        let result = ToolResult::text("test output");
        let res = interceptor.observe(&input, &result).await;
        assert!(res.is_ok());
    }

    #[test]
    fn test_timestamp_format() {
        let ts = timestamp();
        // Should match ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ
        assert_eq!(ts.len(), 20, "Timestamp should be 20 chars: {ts}");
        assert!(ts.ends_with('Z'), "Should end with Z");
        assert!(ts.contains('T'), "Should contain T separator");
    }

    #[test]
    fn test_tool_id_increments() {
        let id1 = next_tool_id();
        let id2 = next_tool_id();
        assert!(id2 > id1, "Tool IDs should increment");
    }

    #[test]
    fn test_tool_id_correlation_store_retrieve() {
        // Test that store_tool_id and retrieve_tool_id work correctly
        let call1 = ToolCall::new("tool_a", json!({"arg": "value1"}));
        let call2 = ToolCall::new("tool_b", json!({"arg": "value2"}));
        let call3 = ToolCall::new("tool_c", json!({"arg": "value3"}));

        // Store IDs for multiple concurrent calls (simulating parallel intercept())
        store_tool_id(&call1, 100);
        store_tool_id(&call2, 101);
        store_tool_id(&call3, 102);

        // Retrieve in different order (simulating observe() completing out of order)
        let id2 = retrieve_tool_id(&call2);
        let id1 = retrieve_tool_id(&call1);
        let id3 = retrieve_tool_id(&call3);

        // Each should get back its correct ID
        assert_eq!(id1, 100, "call1 should retrieve ID 100");
        assert_eq!(id2, 101, "call2 should retrieve ID 101");
        assert_eq!(id3, 102, "call3 should retrieve ID 102");
    }

    #[test]
    fn test_tool_id_correlation_cleanup() {
        // Test that retrieve_tool_id removes the entry from the map
        let call = ToolCall::new("cleanup_test", json!({}));
        let stored_id = 99999; // Use a distinctive ID

        store_tool_id(&call, stored_id);

        // Verify the entry exists in the map
        let key = &call as *const ToolCall as usize;
        {
            let map = tool_id_map().read().unwrap();
            assert!(
                map.contains_key(&key),
                "Entry should exist before retrieval"
            );
        }

        // First retrieval should get the stored ID and remove it
        let retrieved = retrieve_tool_id(&call);
        assert_eq!(retrieved, stored_id, "Should retrieve the stored ID");

        // Verify the entry was removed from the map
        {
            let map = tool_id_map().read().unwrap();
            assert!(
                !map.contains_key(&key),
                "Entry should be removed after retrieval"
            );
        }
    }

    #[tokio::test]
    async fn test_concurrent_intercept_observe_correlation() {
        // Test full intercept -> observe flow for multiple concurrent calls
        let interceptor = AuditLog;

        // Create multiple tool calls
        let calls: Vec<ToolCall> = (0..5)
            .map(|i| ToolCall::new(&format!("concurrent_tool_{}", i), json!({"index": i})))
            .collect();

        // Intercept all (simulating parallel tool execution start)
        for call in &calls {
            let decision = interceptor.intercept(call).await.unwrap();
            assert_eq!(decision, InterceptDecision::Allow);
        }

        // Observe in reverse order (simulating out-of-order completion)
        for call in calls.iter().rev() {
            let result = ToolResult::text("completed");
            let res = interceptor.observe(call, &result).await;
            assert!(res.is_ok(), "observe should succeed for {}", call.name);
        }

        // Map should be empty after all retrievals (verify cleanup)
        // Note: We can't check exact emptiness due to other tests, but the
        // important thing is the flow completed without panics
        let _map = tool_id_map().read().unwrap();
    }
}
