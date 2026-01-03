//! Indicatif-based renderer for CLI progress display.
//!
//! Provides progress display with support for:
//! - Status messages from the agent's tracker
//! - Real-time tool call events (`tool_call_started`, `tool_result`)
//!
//! This design is agent-agnostic - any agent can provide its own tracker
//! with appropriate status messages, while tool-using agents can emit
//! tool events for more detailed display.

use super::renderer::Renderer;
use crate::format::{print_final_result, print_interrupted, truncate};
use anyhow::Result;
use gemicro_core::{AgentUpdate, ExecutionTracking};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::{Duration, Instant};

/// Spinner animation tick interval in milliseconds.
const SPINNER_TICK_MS: u64 = 120;

/// Maximum characters to show in tool argument previews.
const TOOL_ARGS_PREVIEW_CHARS: usize = 60;

/// Maximum characters to show in tool result previews.
const TOOL_RESULT_PREVIEW_CHARS: usize = 80;

/// Renderer using indicatif for progress bar display.
///
/// Uses a single spinner that displays status messages from the agent's tracker.
/// This is simpler and more flexible than tracking individual steps, as each
/// agent can provide its own appropriate status messages.
///
/// Also handles tool events (`tool_call_started`, `tool_result`) for real-time
/// feedback during tool execution.
pub struct IndicatifRenderer {
    spinner: ProgressBar,
    start_time: Instant,
    /// Whether to use plain text output (no markdown rendering).
    plain: bool,
    /// The last status message, for interrupted display.
    last_status: Option<String>,
    /// Whether a tool call is currently in progress.
    in_tool_call: bool,
    /// Count of tool calls in this session.
    tool_call_count: usize,
}

impl IndicatifRenderer {
    /// Create a new IndicatifRenderer.
    ///
    /// If `plain` is true, markdown rendering will be disabled for final output.
    pub fn new(plain: bool) -> Self {
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .expect("Invalid template"),
        );
        spinner.enable_steady_tick(Duration::from_millis(SPINNER_TICK_MS));
        spinner.set_message("Starting...");

        Self {
            spinner,
            start_time: Instant::now(),
            plain,
            last_status: None,
            in_tool_call: false,
            tool_call_count: 0,
        }
    }

    /// Format a tool call for display.
    fn format_tool_call(&self, tool_name: &str, args: &serde_json::Value) -> String {
        // Extract a meaningful preview from args
        let args_preview = self.format_args_preview(tool_name, args);

        if args_preview.is_empty() {
            format!("🔧 {} ...", tool_name)
        } else {
            format!("🔧 {}: {} ...", tool_name, args_preview)
        }
    }

    /// Format a tool result for display.
    fn format_tool_result(
        &self,
        tool_name: &str,
        success: bool,
        duration_ms: u64,
        result: &serde_json::Value,
    ) -> String {
        let status_icon = if success { "✓" } else { "✗" };
        let duration = format!("{:.1}s", duration_ms as f64 / 1000.0);

        // Get a preview of the result
        let result_preview = self.format_result_preview(tool_name, result);

        if result_preview.is_empty() {
            format!("  {} {} ({})", status_icon, tool_name, duration)
        } else {
            format!(
                "  {} {} ({}) → {}",
                status_icon, tool_name, duration, result_preview
            )
        }
    }

    /// Format a diff preview for FileEdit showing old → new.
    fn format_diff_preview(&self, args: &serde_json::Value) -> Option<String> {
        let old_str = args.get("old_string").and_then(|v| v.as_str())?;
        let new_str = args.get("new_string").and_then(|v| v.as_str())?;

        // Truncate both strings for preview
        let old_preview = truncate(old_str.lines().next().unwrap_or(old_str), 30);
        let new_preview = truncate(new_str.lines().next().unwrap_or(new_str), 30);

        Some(format!("\"{}\" → \"{}\"", old_preview, new_preview))
    }

    /// Extract a meaningful preview from tool arguments.
    fn format_args_preview(&self, tool_name: &str, args: &serde_json::Value) -> String {
        // Tool-specific argument formatting
        match tool_name {
            "bash" => args
                .get("command")
                .and_then(|v| v.as_str())
                .map(|s| truncate(s, TOOL_ARGS_PREVIEW_CHARS))
                .unwrap_or_default(),
            "file_read" | "FileRead" => args
                .get("path")
                .and_then(|v| v.as_str())
                .map(|s| truncate(s, TOOL_ARGS_PREVIEW_CHARS))
                .unwrap_or_default(),
            "file_write" | "FileWrite" => args
                .get("path")
                .and_then(|v| v.as_str())
                .map(|s| truncate(s, TOOL_ARGS_PREVIEW_CHARS))
                .unwrap_or_default(),
            "file_edit" | "FileEdit" => {
                // Show path + diff preview for edits
                let path = args
                    .get("path")
                    .and_then(|v| v.as_str())
                    .map(|s| truncate(s, 30))
                    .unwrap_or_default();
                if let Some(diff) = self.format_diff_preview(args) {
                    format!("{} {}", path, diff)
                } else {
                    path
                }
            }
            "glob" | "Glob" => args
                .get("pattern")
                .and_then(|v| v.as_str())
                .map(|s| truncate(s, TOOL_ARGS_PREVIEW_CHARS))
                .unwrap_or_default(),
            "grep" | "Grep" => args
                .get("pattern")
                .and_then(|v| v.as_str())
                .map(|s| truncate(s, TOOL_ARGS_PREVIEW_CHARS))
                .unwrap_or_default(),
            "web_fetch" | "WebFetch" => args
                .get("url")
                .and_then(|v| v.as_str())
                .map(|s| truncate(s, TOOL_ARGS_PREVIEW_CHARS))
                .unwrap_or_default(),
            "web_search" | "WebSearch" => args
                .get("query")
                .and_then(|v| v.as_str())
                .map(|s| truncate(s, TOOL_ARGS_PREVIEW_CHARS))
                .unwrap_or_default(),
            _ => {
                // Generic: try common field names or show truncated JSON
                if let Some(s) = args.get("input").and_then(|v| v.as_str()) {
                    truncate(s, TOOL_ARGS_PREVIEW_CHARS)
                } else if let Some(s) = args.get("query").and_then(|v| v.as_str()) {
                    truncate(s, TOOL_ARGS_PREVIEW_CHARS)
                } else if let Ok(json) = serde_json::to_string(args) {
                    truncate(&json, TOOL_ARGS_PREVIEW_CHARS)
                } else {
                    String::new()
                }
            }
        }
    }

    /// Extract a meaningful preview from tool result.
    fn format_result_preview(&self, _tool_name: &str, result: &serde_json::Value) -> String {
        // Check for error
        if let Some(err) = result.get("error").and_then(|v| v.as_str()) {
            return truncate(&format!("Error: {}", err), TOOL_RESULT_PREVIEW_CHARS);
        }

        // Try to extract a meaningful preview
        if let Some(s) = result.as_str() {
            // Simple string result
            let first_line = s.lines().next().unwrap_or(s);
            return truncate(first_line, TOOL_RESULT_PREVIEW_CHARS);
        }

        if let Some(content) = result.get("content") {
            if let Some(s) = content.as_str() {
                let first_line = s.lines().next().unwrap_or(s);
                return truncate(first_line, TOOL_RESULT_PREVIEW_CHARS);
            }
        }

        // For arrays (like file lists), show count
        if let Some(arr) = result.as_array() {
            return format!("{} items", arr.len());
        }

        String::new()
    }
}

impl Default for IndicatifRenderer {
    fn default() -> Self {
        Self::new(false)
    }
}

impl Renderer for IndicatifRenderer {
    fn on_event(&mut self, event: &AgentUpdate) -> Result<()> {
        match event.event_type.as_str() {
            "tool_call_started" => {
                // Extract tool info from event data
                let tool_name = event
                    .data
                    .get("tool_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let args = event
                    .data
                    .get("arguments")
                    .unwrap_or(&serde_json::Value::Null);

                // Track the tool call
                self.in_tool_call = true;
                self.tool_call_count += 1;

                // Update spinner with tool call message
                let msg = self.format_tool_call(tool_name, args);
                self.spinner.set_message(msg);
            }
            "tool_result" => {
                // Extract result info
                let tool_name = event
                    .data
                    .get("tool_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                // Default to false (fail-safe) when success field is missing
                let success = event
                    .data
                    .get("success")
                    .and_then(|v| v.as_bool())
                    .unwrap_or_else(|| {
                        log::warn!(
                            "tool_result event missing 'success' field, defaulting to false"
                        );
                        false
                    });
                let duration_ms = event
                    .data
                    .get("duration_ms")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let result = event.data.get("result").unwrap_or(&serde_json::Value::Null);

                // Clear spinner and print result line
                self.spinner.suspend(|| {
                    let msg = self.format_tool_result(tool_name, success, duration_ms, result);
                    println!("{}", msg);
                });

                // Clear tool call state
                self.in_tool_call = false;
            }
            "batch_approved" => {
                // Batch was approved - brief confirmation
                self.spinner.suspend(|| {
                    println!("✅ Batch approved");
                });
            }
            "batch_denied" => {
                // Batch was denied
                self.spinner.suspend(|| {
                    println!("❌ Batch denied");
                });
            }
            "batch_review_individually" => {
                // User chose to review individually
                self.spinner.suspend(|| {
                    println!("🔍 Reviewing tools individually...");
                });
            }
            "context_usage" => {
                // Context usage update - update spinner with usage info
                let percent = event
                    .data
                    .get("percent")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let level = event
                    .data
                    .get("level")
                    .and_then(|v| v.as_str())
                    .unwrap_or("normal");

                // Only show warning/critical levels prominently
                if level == "warning" || level == "critical" {
                    let icon = if level == "critical" { "🔴" } else { "🟡" };
                    self.spinner.suspend(|| {
                        println!("{} Context: {:.0}% used", icon, percent);
                    });
                }
            }
            _ => {
                // Other events are handled by on_status via the tracker
            }
        }
        Ok(())
    }

    fn on_status(&mut self, tracker: &dyn ExecutionTracking) -> Result<()> {
        // Only update spinner from tracker if not in a tool call
        // (tool calls manage the spinner directly)
        if !self.in_tool_call {
            if let Some(msg) = tracker.status_message() {
                self.spinner.set_message(msg.to_string());
                self.last_status = Some(msg.to_string());
            }
        } else if let Some(msg) = tracker.status_message() {
            // Still update last_status even during tool calls
            self.last_status = Some(msg.to_string());
        }
        Ok(())
    }

    fn on_complete(&mut self, tracker: &dyn ExecutionTracking) -> Result<()> {
        self.spinner.finish_and_clear();

        if let Some(result) = tracker.final_result() {
            print_final_result(result, self.start_time.elapsed(), self.plain);
        } else {
            log::warn!("on_complete called but tracker.final_result() returned None");
        }

        Ok(())
    }

    fn on_interrupted(&mut self, tracker: &dyn ExecutionTracking) -> Result<()> {
        self.spinner
            .finish_with_message("⚠️  Execution interrupted by user");

        // Use the tracker's status message if available, otherwise fall back to last known
        let status = tracker
            .status_message()
            .map(|s| s.to_string())
            .or_else(|| self.last_status.clone());

        print_interrupted(status.as_deref());

        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if !self.spinner.is_finished() {
            self.spinner.finish_and_clear();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time validation of constants
    const _: () = assert!(SPINNER_TICK_MS > 0);
    const _: () = assert!(SPINNER_TICK_MS <= 500);

    #[test]
    fn test_new_renderer() {
        let renderer = IndicatifRenderer::new(false);
        assert!(!renderer.plain);
        assert!(renderer.last_status.is_none());
    }

    #[test]
    fn test_default_renderer() {
        let renderer = IndicatifRenderer::default();
        assert!(!renderer.plain);
    }

    #[test]
    fn test_plain_mode() {
        let renderer = IndicatifRenderer::new(true);
        assert!(renderer.plain);
    }
}
