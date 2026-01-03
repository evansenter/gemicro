//! Event constants for DeveloperAgent.
//!
//! These are internal constants following Evergreen soft-typing principles.
//! They are NOT exported - use string literals in consuming code.

/// Emitted when the developer agent starts.
pub(crate) const EVENT_DEVELOPER_STARTED: &str = "developer_started";

/// Emitted before executing a tool call.
pub(crate) const EVENT_TOOL_CALL_STARTED: &str = "tool_call_started";

/// Emitted after a tool call completes.
pub(crate) const EVENT_TOOL_RESULT: &str = "tool_result";

/// Emitted when a batch of tool calls is ready for approval.
pub(crate) const EVENT_BATCH_PLAN: &str = "batch_plan";

/// Emitted when a batch is approved.
pub(crate) const EVENT_BATCH_APPROVED: &str = "batch_approved";

/// Emitted when a batch is denied.
pub(crate) const EVENT_BATCH_DENIED: &str = "batch_denied";

/// Emitted when user chooses to review batch tools individually.
pub(crate) const EVENT_BATCH_REVIEW_INDIVIDUALLY: &str = "batch_review_individually";

/// Emitted periodically to report context usage.
pub(crate) const EVENT_CONTEXT_USAGE: &str = "context_usage";
