//! Input/output types for interceptors.
//!
//! These types are used with the generic [`Interceptor<In, Out>`](super::Interceptor)
//! trait to provide type-safe interception for different operation types.

use serde_json::Value;

// ============================================================================
// ToolCall
// ============================================================================

/// Tool invocation context.
///
/// Represents a tool call that an interceptor can validate, transform, or deny.
///
/// # Example
///
/// ```
/// use gemicro_core::interceptor::ToolCall;
/// use serde_json::json;
///
/// let call = ToolCall::new("calculator", json!({"expression": "2 + 2"}));
/// assert_eq!(call.name, "calculator");
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ToolCall {
    /// The name of the tool being called.
    pub name: String,

    /// The arguments passed to the tool.
    pub arguments: Value,
}

impl ToolCall {
    /// Create a new tool call.
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }
}

// ============================================================================
// UserMessage
// ============================================================================

/// User message - the trigger that starts agent execution.
///
/// Designed for multimodal extension. Currently text-only, but the structure
/// allows adding image/audio/video support when rust-genai's multimodal
/// capabilities are integrated.
///
/// # Why This Is Not a Tool Call
///
/// Despite superficial similarities, user messages and tool calls have
/// different roles:
/// - Tools are capabilities the agent can invoke (LLM → System → LLM)
/// - User messages are triggers that start the agent (Human → Agent → Human)
///
/// The Interceptor pattern applies to both, but they remain distinct types.
///
/// # Example
///
/// ```
/// use gemicro_core::interceptor::UserMessage;
///
/// let msg = UserMessage::text("Hello, agent!");
/// assert_eq!(msg.as_text(), Some("Hello, agent!"));
///
/// // From string conversion
/// let msg: UserMessage = "Hello".into();
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct UserMessage {
    /// The content of the message.
    pub content: UserContent,
}

impl UserMessage {
    /// Create a text message (convenience constructor).
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: UserContent::Text(content.into()),
        }
    }

    /// Get the text content if this is a text message.
    pub fn as_text(&self) -> Option<&str> {
        match &self.content {
            UserContent::Text(s) => Some(s),
        }
    }
}

impl From<String> for UserMessage {
    fn from(s: String) -> Self {
        Self::text(s)
    }
}

impl From<&str> for UserMessage {
    fn from(s: &str) -> Self {
        Self::text(s)
    }
}

/// Content of a user message.
///
/// Currently text-only. When multimodal support is added, this will be
/// extended with additional variants (Image, Audio, etc.) or a Parts
/// variant containing mixed content.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum UserContent {
    /// Plain text content (common case).
    Text(String),
    // Future: Multimodal content
    // Parts(Vec<ContentPart>),
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tool_call_new() {
        let call = ToolCall::new("bash", json!({"command": "ls"}));
        assert_eq!(call.name, "bash");
        assert_eq!(call.arguments["command"], "ls");
    }

    #[test]
    fn test_tool_call_clone() {
        let call = ToolCall::new("test", json!({}));
        let cloned = call.clone();
        assert_eq!(call, cloned);
    }

    #[test]
    fn test_user_message_text() {
        let msg = UserMessage::text("Hello");
        assert_eq!(msg.as_text(), Some("Hello"));
    }

    #[test]
    fn test_user_message_from_string() {
        let msg: UserMessage = String::from("Hello").into();
        assert_eq!(msg.as_text(), Some("Hello"));
    }

    #[test]
    fn test_user_message_from_str() {
        let msg: UserMessage = "Hello".into();
        assert_eq!(msg.as_text(), Some("Hello"));
    }

    #[test]
    fn test_user_message_equality() {
        let msg1 = UserMessage::text("Hello");
        let msg2 = UserMessage::text("Hello");
        let msg3 = UserMessage::text("World");

        assert_eq!(msg1, msg2);
        assert_ne!(msg1, msg3);
    }
}
