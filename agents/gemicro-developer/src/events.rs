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
