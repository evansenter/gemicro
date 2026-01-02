//! Approval batching for tool executions.
//!
//! Collects tool calls and presents them as a batch for single approval,
//! reducing user interaction overhead.

use gemicro_core::tool::{ConfirmationHandler, ToolRegistry};
use serde_json::Value;

/// A pending tool call waiting for batch approval.
#[derive(Debug, Clone)]
pub struct PendingToolCall {
    /// Unique identifier for this call.
    pub call_id: String,
    /// Name of the tool to execute.
    pub tool_name: String,
    /// Arguments for the tool.
    pub arguments: Value,
    /// Whether this tool requires confirmation.
    pub requires_confirmation: bool,
    /// Human-readable description of what the tool will do.
    pub confirmation_message: Option<String>,
}

impl PendingToolCall {
    /// Create a new pending tool call.
    pub fn new(call_id: impl Into<String>, tool_name: impl Into<String>, arguments: Value) -> Self {
        Self {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            arguments,
            requires_confirmation: false,
            confirmation_message: None,
        }
    }

    /// Populate confirmation info from the tool registry.
    pub fn with_tool_info(mut self, registry: &ToolRegistry) -> Self {
        if let Some(tool) = registry.get(&self.tool_name) {
            self.requires_confirmation = tool.requires_confirmation(&self.arguments);
            if self.requires_confirmation {
                self.confirmation_message = Some(tool.confirmation_message(&self.arguments));
            }
        }
        self
    }
}

/// A batch of pending tool calls for approval.
#[derive(Debug, Clone, Default)]
pub struct ToolBatch {
    /// The pending tool calls in this batch.
    pub calls: Vec<PendingToolCall>,
}

impl ToolBatch {
    /// Create a new empty batch.
    pub fn new() -> Self {
        Self { calls: Vec::new() }
    }

    /// Add a tool call to the batch.
    pub fn push(&mut self, call: PendingToolCall) {
        self.calls.push(call);
    }

    /// Check if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    /// Get the number of calls in the batch.
    pub fn len(&self) -> usize {
        self.calls.len()
    }

    /// Check if any call in the batch requires confirmation.
    pub fn requires_confirmation(&self) -> bool {
        self.calls.iter().any(|c| c.requires_confirmation)
    }

    /// Get only the calls that require confirmation.
    pub fn confirmation_required(&self) -> Vec<&PendingToolCall> {
        self.calls
            .iter()
            .filter(|c| c.requires_confirmation)
            .collect()
    }

    /// Get a summary of the batch for display.
    pub fn summary(&self) -> BatchSummary {
        let total = self.calls.len();
        let requires_confirmation = self
            .calls
            .iter()
            .filter(|c| c.requires_confirmation)
            .count();

        let tool_counts = self
            .calls
            .iter()
            .fold(std::collections::HashMap::new(), |mut acc, c| {
                *acc.entry(c.tool_name.clone()).or_insert(0usize) += 1;
                acc
            });

        BatchSummary {
            total,
            requires_confirmation,
            tool_counts,
        }
    }

    /// Clear the batch.
    pub fn clear(&mut self) {
        self.calls.clear();
    }
}

/// Summary of a tool batch for display.
#[derive(Debug, Clone)]
pub struct BatchSummary {
    /// Total number of tool calls.
    pub total: usize,
    /// Number that require confirmation.
    pub requires_confirmation: usize,
    /// Count by tool name.
    pub tool_counts: std::collections::HashMap<String, usize>,
}

impl std::fmt::Display for BatchSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} tool call(s)", self.total)?;

        if self.requires_confirmation > 0 {
            write!(f, " - {} require confirmation", self.requires_confirmation)?;
        }

        let tools: Vec<String> = self
            .tool_counts
            .iter()
            .map(|(name, count)| {
                if *count > 1 {
                    format!("{}×{}", name, count)
                } else {
                    name.clone()
                }
            })
            .collect();

        if !tools.is_empty() {
            write!(f, " - [{}]", tools.join(", "))?;
        }

        Ok(())
    }
}

/// Result of batch confirmation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchApproval {
    /// All tools approved for execution.
    Approved,
    /// User denied the batch.
    Denied,
    /// User requested individual review for each tool.
    ReviewIndividually,
}

/// Handler for batch confirmations.
///
/// Extends the standard ConfirmationHandler with batch-aware methods.
#[async_trait::async_trait]
pub trait BatchConfirmationHandler: ConfirmationHandler {
    /// Request approval for a batch of tool calls.
    ///
    /// Default implementation shows each requiring-confirmation call individually.
    async fn confirm_batch(&self, batch: &ToolBatch) -> BatchApproval {
        // Default: fall back to individual confirmations
        // If all pass, return Approved; if any fail, return Denied
        for call in batch.confirmation_required() {
            let message = call
                .confirmation_message
                .as_deref()
                .unwrap_or("Execute tool?");
            if !self
                .confirm(&call.tool_name, message, &call.arguments)
                .await
            {
                return BatchApproval::Denied;
            }
        }
        BatchApproval::Approved
    }
}

// Auto-implement BatchConfirmationHandler for any ConfirmationHandler
// (using the default fallback behavior)
impl<T: ConfirmationHandler + ?Sized> BatchConfirmationHandler for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_pending_tool_call() {
        let call = PendingToolCall::new("call-1", "file_read", json!({"path": "/tmp/test.txt"}));
        assert_eq!(call.call_id, "call-1");
        assert_eq!(call.tool_name, "file_read");
        assert!(!call.requires_confirmation);
    }

    #[test]
    fn test_tool_batch() {
        let mut batch = ToolBatch::new();
        assert!(batch.is_empty());

        batch.push(PendingToolCall::new("1", "file_read", json!({})));
        batch.push(PendingToolCall {
            call_id: "2".into(),
            tool_name: "file_write".into(),
            arguments: json!({}),
            requires_confirmation: true,
            confirmation_message: Some("Write to file?".into()),
        });

        assert_eq!(batch.len(), 2);
        assert!(batch.requires_confirmation());
        assert_eq!(batch.confirmation_required().len(), 1);
    }

    #[test]
    fn test_batch_summary() {
        let mut batch = ToolBatch::new();
        batch.push(PendingToolCall::new("1", "file_read", json!({})));
        batch.push(PendingToolCall::new("2", "file_read", json!({})));
        batch.push(PendingToolCall {
            call_id: "3".into(),
            tool_name: "file_write".into(),
            arguments: json!({}),
            requires_confirmation: true,
            confirmation_message: None,
        });

        let summary = batch.summary();
        assert_eq!(summary.total, 3);
        assert_eq!(summary.requires_confirmation, 1);
        assert_eq!(summary.tool_counts.get("file_read"), Some(&2));
        assert_eq!(summary.tool_counts.get("file_write"), Some(&1));

        let text = summary.to_string();
        assert!(text.contains("3 tool call(s)"));
        assert!(text.contains("1 require confirmation"));
    }
}
