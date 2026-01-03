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

/// Map from ToolCall pointer address to tool ID for correlation.
/// This handles concurrent tool execution where multiple intercept() calls
/// happen before their corresponding observe() calls.
static TOOL_ID_MAP: OnceLock<RwLock<HashMap<usize, usize>>> = OnceLock::new();

/// Get the tool ID correlation map.
fn tool_id_map() -> &'static RwLock<HashMap<usize, usize>> {
    TOOL_ID_MAP.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Store a tool ID for later correlation in observe().
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

/// Format the current timestamp for log output.
fn timestamp() -> String {
    let now = std::time::SystemTime::now();
    let duration = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    // Convert to human-readable format (simplified UTC)
    let days_since_epoch = secs / 86400;
    let secs_today = secs % 86400;
    let hours = secs_today / 3600;
    let minutes = (secs_today % 3600) / 60;
    let seconds = secs_today % 60;

    // Calculate year/month/day from days since epoch (1970-01-01)
    let mut remaining_days = days_since_epoch as i64;
    let mut year = 1970;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let (month, day) = day_of_year_to_month_day(remaining_days as u32 + 1, is_leap_year(year));

    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn day_of_year_to_month_day(day_of_year: u32, leap: bool) -> (u32, u32) {
    let days_in_months: [u32; 12] = if leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut remaining = day_of_year;
    for (i, &days) in days_in_months.iter().enumerate() {
        if remaining <= days {
            return (i as u32 + 1, remaining);
        }
        remaining -= days;
    }
    (12, 31) // Fallback
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
}
