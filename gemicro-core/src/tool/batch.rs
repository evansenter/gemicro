//! Approval batching for tool executions.
//!
//! Collects tool calls and presents them as a batch for single approval,
//! reducing user interaction overhead.

use super::{ConfirmationHandler, ToolRegistry};
use serde_json::Value;

/// A pending tool call waiting for batch approval.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PendingToolCall {
    /// Unique identifier for this call.
    call_id: String,
    /// Name of the tool to execute.
    tool_name: String,
    /// Arguments for the tool.
    arguments: Value,
    /// Whether this tool requires confirmation.
    requires_confirmation: bool,
    /// Human-readable description of what the tool will do.
    confirmation_message: Option<String>,
}

impl PendingToolCall {
    /// Get the call ID.
    pub fn call_id(&self) -> &str {
        &self.call_id
    }

    /// Get the tool name.
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    /// Get the arguments.
    pub fn arguments(&self) -> &Value {
        &self.arguments
    }

    /// Check if this call requires confirmation.
    pub fn requires_confirmation(&self) -> bool {
        self.requires_confirmation
    }

    /// Get the confirmation message, if any.
    pub fn confirmation_message(&self) -> Option<&str> {
        self.confirmation_message.as_deref()
    }
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
#[non_exhaustive]
pub struct ToolBatch {
    /// The pending tool calls in this batch.
    calls: Vec<PendingToolCall>,
}

impl ToolBatch {
    /// Get read-only access to the calls.
    pub fn calls(&self) -> &[PendingToolCall] {
        &self.calls
    }

    /// Iterate over the calls.
    pub fn iter(&self) -> impl Iterator<Item = &PendingToolCall> {
        self.calls.iter()
    }
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
        self.calls.iter().any(|c| c.requires_confirmation())
    }

    /// Get only the calls that require confirmation.
    pub fn confirmation_required(&self) -> Vec<&PendingToolCall> {
        self.calls
            .iter()
            .filter(|c| c.requires_confirmation())
            .collect()
    }

    /// Get a summary of the batch for display.
    pub fn summary(&self) -> BatchSummary {
        let total = self.calls.len();
        let requires_confirmation = self
            .calls
            .iter()
            .filter(|c| c.requires_confirmation())
            .count();

        let tool_counts = self
            .calls
            .iter()
            .fold(std::collections::HashMap::new(), |mut acc, c| {
                *acc.entry(c.tool_name().to_string()).or_insert(0usize) += 1;
                acc
            });

        BatchSummary::new(total, requires_confirmation, tool_counts)
    }

    /// Clear the batch.
    pub fn clear(&mut self) {
        self.calls.clear();
    }
}

/// Summary of a tool batch for display.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BatchSummary {
    /// Total number of tool calls.
    total: usize,
    /// Number that require confirmation.
    requires_confirmation: usize,
    /// Count by tool name.
    tool_counts: std::collections::HashMap<String, usize>,
}

impl BatchSummary {
    /// Create a new batch summary.
    pub fn new(
        total: usize,
        requires_confirmation: usize,
        tool_counts: std::collections::HashMap<String, usize>,
    ) -> Self {
        Self {
            total,
            requires_confirmation,
            tool_counts,
        }
    }

    /// Get the total number of tool calls.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Get the number of calls that require confirmation.
    pub fn requires_confirmation(&self) -> usize {
        self.requires_confirmation
    }

    /// Get the tool counts.
    pub fn tool_counts(&self) -> &std::collections::HashMap<String, usize> {
        &self.tool_counts
    }
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
                    format!("{}x{}", name, count)
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
/// The default implementation falls back to individual confirmations.
#[async_trait::async_trait]
pub trait BatchConfirmationHandler: ConfirmationHandler {
    /// Request approval for a batch of tool calls.
    ///
    /// Default implementation shows each requiring-confirmation call individually.
    async fn confirm_batch(&self, batch: &ToolBatch) -> BatchApproval;
}

/// Default batch confirmation behavior: fall back to individual confirmations.
///
/// This helper function can be used by BatchConfirmationHandler implementations
/// that want the default fallback behavior.
///
/// # Important
///
/// This function short-circuits on the first denial. If multiple tools require
/// confirmation and an earlier one is approved but a later one is denied, the
/// function returns `Denied`. The earlier approvals are not executed since the
/// entire batch is rejected. This is intentional: batches are all-or-nothing.
pub async fn default_batch_confirm<H: ConfirmationHandler + ?Sized>(
    handler: &H,
    batch: &ToolBatch,
) -> BatchApproval {
    for call in batch.confirmation_required() {
        let message = call.confirmation_message().unwrap_or("Execute tool?");
        if !handler
            .confirm(call.tool_name(), message, call.arguments())
            .await
        {
            return BatchApproval::Denied;
        }
    }
    BatchApproval::Approved
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{AutoApprove, AutoDeny};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_pending_tool_call() {
        let call = PendingToolCall::new("call-1", "file_read", json!({"path": "/tmp/test.txt"}));
        assert_eq!(call.call_id(), "call-1");
        assert_eq!(call.tool_name(), "file_read");
        assert!(!call.requires_confirmation());
    }

    #[test]
    fn test_pending_tool_call_with_confirmation() {
        let call = make_pending_call_with_confirmation(
            "call-2",
            "file_write",
            json!({"path": "/tmp/out.txt", "content": "hello"}),
            "Write to /tmp/out.txt?",
        );

        assert!(call.requires_confirmation());
        assert_eq!(call.confirmation_message(), Some("Write to /tmp/out.txt?"));
    }

    /// Test helper: create a pending call that requires confirmation.
    fn make_pending_call_with_confirmation(
        call_id: &str,
        tool_name: &str,
        arguments: Value,
        message: &str,
    ) -> PendingToolCall {
        // Use internal struct construction for test helpers only
        PendingToolCall {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            arguments,
            requires_confirmation: true,
            confirmation_message: Some(message.into()),
        }
    }

    #[test]
    fn test_tool_batch() {
        let mut batch = ToolBatch::new();
        assert!(batch.is_empty());

        batch.push(PendingToolCall::new("1", "file_read", json!({})));
        batch.push(make_pending_call_with_confirmation(
            "2",
            "file_write",
            json!({}),
            "Write to file?",
        ));

        assert_eq!(batch.len(), 2);
        assert!(batch.requires_confirmation());
        assert_eq!(batch.confirmation_required().len(), 1);
    }

    #[test]
    fn test_tool_batch_no_confirmation_required() {
        let mut batch = ToolBatch::new();
        batch.push(PendingToolCall::new("1", "file_read", json!({})));
        batch.push(PendingToolCall::new("2", "glob", json!({})));

        assert_eq!(batch.len(), 2);
        assert!(!batch.requires_confirmation());
        assert_eq!(batch.confirmation_required().len(), 0);
    }

    #[test]
    fn test_tool_batch_clear() {
        let mut batch = ToolBatch::new();
        batch.push(PendingToolCall::new("1", "file_read", json!({})));
        batch.push(PendingToolCall::new("2", "glob", json!({})));

        assert_eq!(batch.len(), 2);
        batch.clear();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_batch_summary() {
        let mut batch = ToolBatch::new();
        batch.push(PendingToolCall::new("1", "file_read", json!({})));
        batch.push(PendingToolCall::new("2", "file_read", json!({})));
        batch.push(make_pending_call_with_confirmation(
            "3",
            "file_write",
            json!({}),
            "Write file",
        ));

        let summary = batch.summary();
        assert_eq!(summary.total(), 3);
        assert_eq!(summary.requires_confirmation(), 1);
        assert_eq!(summary.tool_counts().get("file_read"), Some(&2));
        assert_eq!(summary.tool_counts().get("file_write"), Some(&1));

        let text = summary.to_string();
        assert!(text.contains("3 tool call(s)"));
        assert!(text.contains("1 require confirmation"));
    }

    #[test]
    fn test_batch_approval_variants() {
        assert_ne!(BatchApproval::Approved, BatchApproval::Denied);
        assert_ne!(BatchApproval::Approved, BatchApproval::ReviewIndividually);
        assert_ne!(BatchApproval::Denied, BatchApproval::ReviewIndividually);
    }

    // =========================================================================
    // BatchConfirmationHandler trait tests
    // =========================================================================

    /// Test handler that tracks confirm calls and returns configurable responses.
    #[derive(Debug)]
    struct TrackingHandler {
        approve: bool,
        confirm_calls: AtomicUsize,
    }

    impl TrackingHandler {
        fn new(approve: bool) -> Self {
            Self {
                approve,
                confirm_calls: AtomicUsize::new(0),
            }
        }

        fn confirm_call_count(&self) -> usize {
            self.confirm_calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl ConfirmationHandler for TrackingHandler {
        async fn confirm(&self, _tool_name: &str, _message: &str, _args: &Value) -> bool {
            self.confirm_calls.fetch_add(1, Ordering::SeqCst);
            self.approve
        }
    }

    #[async_trait::async_trait]
    impl BatchConfirmationHandler for TrackingHandler {
        async fn confirm_batch(&self, batch: &ToolBatch) -> BatchApproval {
            default_batch_confirm(self, batch).await
        }
    }

    #[tokio::test]
    async fn test_default_batch_handler_approves_when_all_confirm() {
        // Default batch handler falls back to individual confirms
        // When all individual confirms pass, returns Approved
        let handler = TrackingHandler::new(true);

        let mut batch = ToolBatch::new();
        batch.push(make_pending_call_with_confirmation(
            "1",
            "bash",
            json!({"command": "ls"}),
            "Run: ls",
        ));
        batch.push(make_pending_call_with_confirmation(
            "2",
            "file_write",
            json!({}),
            "Write file",
        ));

        let result = handler.confirm_batch(&batch).await;
        assert_eq!(result, BatchApproval::Approved);
        assert_eq!(handler.confirm_call_count(), 2); // Both tools confirmed
    }

    #[tokio::test]
    async fn test_default_batch_handler_denies_when_any_confirm_fails() {
        // Default batch handler: if any individual confirm fails, returns Denied
        let handler = TrackingHandler::new(false);

        let mut batch = ToolBatch::new();
        batch.push(make_pending_call_with_confirmation(
            "1",
            "bash",
            json!({}),
            "Run command",
        ));
        batch.push(make_pending_call_with_confirmation(
            "2",
            "file_write",
            json!({}),
            "Write file",
        ));

        let result = handler.confirm_batch(&batch).await;
        assert_eq!(result, BatchApproval::Denied);
        // Short-circuits on first denial
        assert_eq!(handler.confirm_call_count(), 1);
    }

    #[tokio::test]
    async fn test_default_batch_handler_skips_non_confirmation_tools() {
        // Default handler only confirms tools that require confirmation
        let handler = TrackingHandler::new(true);

        let mut batch = ToolBatch::new();
        batch.push(PendingToolCall::new("1", "file_read", json!({}))); // No confirm
        batch.push(make_pending_call_with_confirmation(
            "2",
            "file_write",
            json!({}),
            "Write file",
        ));
        batch.push(PendingToolCall::new("3", "glob", json!({}))); // No confirm

        let result = handler.confirm_batch(&batch).await;
        assert_eq!(result, BatchApproval::Approved);
        // Only the one tool requiring confirmation was confirmed
        assert_eq!(handler.confirm_call_count(), 1);
    }

    #[tokio::test]
    async fn test_default_batch_handler_empty_confirmation_required() {
        // If no tools require confirmation, approves immediately
        let handler = TrackingHandler::new(true);

        let mut batch = ToolBatch::new();
        batch.push(PendingToolCall::new("1", "file_read", json!({})));
        batch.push(PendingToolCall::new("2", "glob", json!({})));

        let result = handler.confirm_batch(&batch).await;
        assert_eq!(result, BatchApproval::Approved);
        // No individual confirms needed
        assert_eq!(handler.confirm_call_count(), 0);
    }

    #[tokio::test]
    async fn test_auto_approve_batch_handler() {
        // AutoApprove should use default batch behavior (approve all)
        let handler = AutoApprove;

        let mut batch = ToolBatch::new();
        batch.push(make_pending_call_with_confirmation(
            "1",
            "bash",
            json!({}),
            "Run command",
        ));

        let result = handler.confirm_batch(&batch).await;
        assert_eq!(result, BatchApproval::Approved);
    }

    #[tokio::test]
    async fn test_auto_deny_batch_handler() {
        // AutoDeny should use default batch behavior (deny on first)
        let handler = AutoDeny;

        let mut batch = ToolBatch::new();
        batch.push(make_pending_call_with_confirmation(
            "1",
            "bash",
            json!({}),
            "Run command",
        ));

        let result = handler.confirm_batch(&batch).await;
        assert_eq!(result, BatchApproval::Denied);
    }

    /// Custom handler that overrides confirm_batch to return a specific response.
    #[derive(Debug)]
    struct CustomBatchHandler {
        batch_response: BatchApproval,
        individual_response: bool,
        batch_calls: AtomicUsize,
        individual_calls: AtomicUsize,
    }

    impl CustomBatchHandler {
        fn new(batch_response: BatchApproval, individual_response: bool) -> Self {
            Self {
                batch_response,
                individual_response,
                batch_calls: AtomicUsize::new(0),
                individual_calls: AtomicUsize::new(0),
            }
        }

        fn batch_call_count(&self) -> usize {
            self.batch_calls.load(Ordering::SeqCst)
        }

        fn individual_call_count(&self) -> usize {
            self.individual_calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl ConfirmationHandler for CustomBatchHandler {
        async fn confirm(&self, _tool_name: &str, _message: &str, _args: &Value) -> bool {
            self.individual_calls.fetch_add(1, Ordering::SeqCst);
            self.individual_response
        }
    }

    #[async_trait::async_trait]
    impl BatchConfirmationHandler for CustomBatchHandler {
        async fn confirm_batch(&self, _batch: &ToolBatch) -> BatchApproval {
            self.batch_calls.fetch_add(1, Ordering::SeqCst);
            self.batch_response
        }
    }

    #[tokio::test]
    async fn test_custom_batch_handler_approve() {
        let handler = CustomBatchHandler::new(BatchApproval::Approved, false);

        let mut batch = ToolBatch::new();
        batch.push(make_pending_call_with_confirmation(
            "1",
            "bash",
            json!({}),
            "Run",
        ));

        let result = handler.confirm_batch(&batch).await;
        assert_eq!(result, BatchApproval::Approved);
        assert_eq!(handler.batch_call_count(), 1);
        assert_eq!(handler.individual_call_count(), 0); // Custom handler doesn't call individual
    }

    #[tokio::test]
    async fn test_custom_batch_handler_deny() {
        let handler = CustomBatchHandler::new(BatchApproval::Denied, true);

        let mut batch = ToolBatch::new();
        batch.push(make_pending_call_with_confirmation(
            "1",
            "bash",
            json!({}),
            "Run",
        ));

        let result = handler.confirm_batch(&batch).await;
        assert_eq!(result, BatchApproval::Denied);
        assert_eq!(handler.batch_call_count(), 1);
    }

    #[tokio::test]
    async fn test_custom_batch_handler_review_individually() {
        let handler = CustomBatchHandler::new(BatchApproval::ReviewIndividually, true);

        let mut batch = ToolBatch::new();
        batch.push(make_pending_call_with_confirmation(
            "1",
            "bash",
            json!({}),
            "Run",
        ));

        let result = handler.confirm_batch(&batch).await;
        assert_eq!(result, BatchApproval::ReviewIndividually);
        assert_eq!(handler.batch_call_count(), 1);
        // Individual confirms would be called separately by the agent
        assert_eq!(handler.individual_call_count(), 0);
    }

    #[tokio::test]
    async fn test_arc_dyn_batch_handler() {
        // Verify BatchConfirmationHandler works with Arc<dyn ...> via deref
        let handler: Arc<dyn ConfirmationHandler> = Arc::new(AutoApprove);

        let mut batch = ToolBatch::new();
        batch.push(make_pending_call_with_confirmation(
            "1",
            "bash",
            json!({}),
            "Run",
        ));

        // Deref coercion: Arc<dyn ConfirmationHandler> -> dyn ConfirmationHandler
        // BatchConfirmationHandler is implemented for dyn ConfirmationHandler + Send + Sync
        let result = (*handler).confirm_batch(&batch).await;
        assert_eq!(result, BatchApproval::Approved);
    }
}
