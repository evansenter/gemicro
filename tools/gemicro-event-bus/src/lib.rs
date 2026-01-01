//! EventBus tool for agent coordination via claude-event-bus.
//!
//! Enables agents to coordinate with other agents and external processes
//! via the claude-event-bus polling mechanism.
//!
//! ## Actions
//!
//! - `register`: Register a session with the event bus
//! - `unregister`: Unregister from the event bus
//! - `publish`: Publish an event to a channel
//! - `get_events`: Poll for recent events
//! - `list_sessions`: List active sessions
//!
//! ## Example
//!
//! ```no_run
//! use gemicro_event_bus::EventBus;
//! use gemicro_core::tool::Tool;
//! use serde_json::json;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let tool = EventBus::new("http://localhost:8765");
//!
//! // Register with the event bus
//! let result = tool.execute(json!({
//!     "action": "register",
//!     "name": "my-agent"
//! })).await?;
//!
//! // Publish an event
//! let result = tool.execute(json!({
//!     "action": "publish",
//!     "event_type": "task_completed",
//!     "payload": "Feature X is done",
//!     "channel": "all"
//! })).await?;
//!
//! // Get events
//! let result = tool.execute(json!({
//!     "action": "get_events",
//!     "since_id": 0
//! })).await?;
//! # Ok(())
//! # }
//! ```

mod client;
mod error;

pub use client::{Event, EventBusClient, PublishResponse, Session};
pub use error::EventBusError;

use async_trait::async_trait;
use gemicro_core::tool::{Tool, ToolError, ToolResult};
use serde_json::{json, Value};
use std::sync::RwLock;

/// Default event bus URL.
const DEFAULT_EVENT_BUS_URL: &str = "http://localhost:8765";

/// Environment variable for event bus URL.
const EVENT_BUS_URL_ENV: &str = "GEMICRO_EVENT_BUS_URL";

/// EventBus tool for coordinating with other agents via the event bus.
///
/// Provides five actions:
/// - `register`: Register a session with the event bus
/// - `unregister`: Unregister the current session
/// - `publish`: Publish an event to a channel
/// - `get_events`: Poll for recent events
/// - `list_sessions`: List active sessions
///
/// # Session Management
///
/// The tool tracks its session ID internally after registration.
/// Subsequent calls to `publish` and `get_events` will include
/// the session ID for proper attribution and channel filtering.
#[derive(Debug)]
pub struct EventBus {
    client: EventBusClient,
    session_id: RwLock<Option<String>>,
}

impl EventBus {
    /// Create a new EventBus tool with the specified event bus URL.
    pub fn new(base_url: &str) -> Self {
        Self {
            client: EventBusClient::new(base_url),
            session_id: RwLock::new(None),
        }
    }

    /// Create a new EventBus tool using the URL from environment or default.
    ///
    /// Checks `GEMICRO_EVENT_BUS_URL` environment variable, falling back
    /// to `http://localhost:8765`.
    pub fn from_env() -> Self {
        let url = std::env::var(EVENT_BUS_URL_ENV).unwrap_or_else(|_| DEFAULT_EVENT_BUS_URL.into());
        Self::new(&url)
    }

    /// Create a new EventBus tool with an existing session ID.
    ///
    /// Use this when the session was registered externally.
    pub fn with_session(base_url: &str, session_id: String) -> Self {
        Self {
            client: EventBusClient::new(base_url),
            session_id: RwLock::new(Some(session_id)),
        }
    }

    /// Get the current session ID, if registered.
    pub fn session_id(&self) -> Option<String> {
        match self.session_id.read() {
            Ok(guard) => guard.clone(),
            Err(_) => {
                log::warn!("EventBus session_id lock poisoned, returning None");
                None
            }
        }
    }

    /// Handle the register action.
    async fn handle_register(&self, input: &Value) -> Result<ToolResult, ToolError> {
        let name = input["name"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("'name' required for register".into()))?;

        let machine = input["machine"].as_str();
        let cwd = input["cwd"].as_str();
        let pid = input["pid"].as_i64().map(|p| p as i32);

        let session = self
            .client
            .register(name, machine, cwd, pid)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        // Store session ID for future calls
        if let Ok(mut guard) = self.session_id.write() {
            *guard = Some(session.session_id.clone());
        }

        Ok(ToolResult::json(json!({
            "session_id": session.session_id,
            "name": session.name,
            "last_event_id": session.last_event_id
        })))
    }

    /// Handle the unregister action.
    async fn handle_unregister(&self) -> Result<ToolResult, ToolError> {
        let session_id = self
            .session_id()
            .ok_or_else(|| ToolError::ExecutionFailed("Not registered with event bus".into()))?;

        self.client
            .unregister(&session_id)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        // Clear stored session ID
        if let Ok(mut guard) = self.session_id.write() {
            *guard = None;
        }

        Ok(ToolResult::json(json!({
            "status": "unregistered",
            "session_id": session_id
        })))
    }

    /// Handle the publish action.
    async fn handle_publish(&self, input: &Value) -> Result<ToolResult, ToolError> {
        let event_type = input["event_type"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("'event_type' required for publish".into()))?;

        let payload = input["payload"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("'payload' required for publish".into()))?;

        let channel = input["channel"].as_str().unwrap_or("all");
        let session_id = self.session_id();

        let response = self
            .client
            .publish(event_type, payload, channel, session_id.as_deref())
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        Ok(ToolResult::json(json!({
            "event_id": response.id,
            "event_type": response.event_type,
            "channel": response.channel
        })))
    }

    /// Handle the get_events action.
    async fn handle_get_events(&self, input: &Value) -> Result<ToolResult, ToolError> {
        let since_id = input["since_id"].as_i64().unwrap_or(0);
        let limit = input["limit"].as_i64().unwrap_or(50) as i32;
        let session_id = self.session_id();

        let events = self
            .client
            .get_events(since_id, limit, session_id.as_deref())
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        // Convert events to JSON, extracting key fields
        let event_json: Vec<Value> = events
            .iter()
            .map(|e| {
                json!({
                    "id": e.id,
                    "event_type": e.event_type,
                    "payload": e.payload,
                    "channel": e.channel,
                    "session_name": e.session_name,
                    "created_at": e.created_at
                })
            })
            .collect();

        Ok(ToolResult::json(json!({
            "events": event_json,
            "count": events.len()
        })))
    }

    /// Handle the list_sessions action.
    async fn handle_list_sessions(&self) -> Result<ToolResult, ToolError> {
        let sessions = self
            .client
            .list_sessions()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        // Convert sessions to JSON
        let session_json: Vec<Value> = sessions
            .iter()
            .map(|s| {
                json!({
                    "session_id": s.session_id,
                    "name": s.name,
                    "machine": s.machine,
                    "cwd": s.cwd
                })
            })
            .collect();

        Ok(ToolResult::json(json!({
            "sessions": session_json,
            "count": sessions.len()
        })))
    }
}

// Manual Clone implementation since RwLock doesn't implement Clone
impl Clone for EventBus {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            session_id: RwLock::new(self.session_id()),
        }
    }
}

#[async_trait]
impl Tool for EventBus {
    fn name(&self) -> &str {
        "event_bus"
    }

    fn description(&self) -> &str {
        "Coordinate with other agents and sessions via the event bus. \
         Actions: register, unregister, publish, get_events, list_sessions. \
         Use to publish events, poll for updates, and discover active sessions."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["register", "unregister", "publish", "get_events", "list_sessions"],
                    "description": "The action to perform"
                },
                "name": {
                    "type": "string",
                    "description": "Session name (required for register)"
                },
                "machine": {
                    "type": "string",
                    "description": "Machine identifier (optional for register)"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory (optional for register)"
                },
                "pid": {
                    "type": "integer",
                    "description": "Process ID (optional for register)"
                },
                "event_type": {
                    "type": "string",
                    "description": "Event type (required for publish)"
                },
                "payload": {
                    "type": "string",
                    "description": "Event payload (required for publish)"
                },
                "channel": {
                    "type": "string",
                    "description": "Channel for publish (default: 'all'). Examples: 'all', 'session:id', 'repo:name'"
                },
                "since_id": {
                    "type": "integer",
                    "description": "Event ID to start from for get_events (default: 0 = recent events)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max events to return for get_events (default: 50)"
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let action = input["action"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("'action' is required".into()))?;

        match action {
            "register" => self.handle_register(&input).await,
            "unregister" => self.handle_unregister().await,
            "publish" => self.handle_publish(&input).await,
            "get_events" => self.handle_get_events(&input).await,
            "list_sessions" => self.handle_list_sessions().await,
            _ => Err(ToolError::InvalidInput(format!(
                "Unknown action: {}. Valid actions: register, unregister, publish, get_events, list_sessions",
                action
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_name_and_description() {
        let tool = EventBus::new("http://localhost:8765");
        assert_eq!(tool.name(), "event_bus");
        assert!(!tool.description().is_empty());
        assert!(tool.description().contains("Coordinate"));
    }

    #[test]
    fn test_parameters_schema() {
        let tool = EventBus::new("http://localhost:8765");
        let schema = tool.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["action"].is_object());
        assert!(schema["required"]
            .as_array()
            .unwrap()
            .contains(&json!("action")));
    }

    #[test]
    fn test_from_env_default() {
        // Clear env var to test default
        std::env::remove_var(EVENT_BUS_URL_ENV);
        let tool = EventBus::from_env();
        // Can't easily check the URL, but we can verify it was created
        assert!(tool.session_id().is_none());
    }

    #[test]
    fn test_with_session() {
        let tool = EventBus::with_session("http://localhost:8765", "test-session-123".into());
        assert_eq!(tool.session_id(), Some("test-session-123".to_string()));
    }

    #[test]
    fn test_clone() {
        let tool = EventBus::with_session("http://localhost:8765", "session-abc".into());
        let cloned = tool.clone();
        assert_eq!(cloned.session_id(), Some("session-abc".to_string()));
    }

    #[tokio::test]
    async fn test_missing_action() {
        let tool = EventBus::new("http://localhost:8765");
        let result = tool.execute(json!({})).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_invalid_action() {
        let tool = EventBus::new("http://localhost:8765");
        let result = tool.execute(json!({"action": "invalid_action"})).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        if let ToolError::InvalidInput(msg) = err {
            assert!(msg.contains("Unknown action"));
        }
    }

    #[tokio::test]
    async fn test_register_missing_name() {
        let tool = EventBus::new("http://localhost:8765");
        let result = tool.execute(json!({"action": "register"})).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_publish_missing_event_type() {
        let tool = EventBus::new("http://localhost:8765");
        let result = tool
            .execute(json!({"action": "publish", "payload": "test"}))
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_publish_missing_payload() {
        let tool = EventBus::new("http://localhost:8765");
        let result = tool
            .execute(json!({"action": "publish", "event_type": "test"}))
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_unregister_without_session() {
        let tool = EventBus::new("http://localhost:8765");
        let result = tool.execute(json!({"action": "unregister"})).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed(_)));
        if let ToolError::ExecutionFailed(msg) = err {
            assert!(msg.contains("Not registered"));
        }
    }

    // Integration tests (require running event bus)
    // Run with: cargo test --package gemicro-event-bus -- --ignored

    #[tokio::test]
    #[ignore = "requires running event bus"]
    async fn test_integration_register_and_unregister() {
        let tool = EventBus::from_env();

        // Register
        let result = tool
            .execute(json!({
                "action": "register",
                "name": "integration-test-agent"
            }))
            .await
            .expect("register should succeed");

        let content = result.content;
        assert!(content["session_id"].is_string());
        assert_eq!(content["name"], "integration-test-agent");

        // Verify session ID is stored
        assert!(tool.session_id().is_some());

        // Unregister
        let result = tool
            .execute(json!({"action": "unregister"}))
            .await
            .expect("unregister should succeed");

        assert_eq!(result.content["status"], "unregistered");

        // Verify session ID is cleared
        assert!(tool.session_id().is_none());
    }

    #[tokio::test]
    #[ignore = "requires running event bus"]
    async fn test_integration_publish_and_get_events() {
        let tool = EventBus::from_env();

        // Register first
        tool.execute(json!({
            "action": "register",
            "name": "publish-test-agent"
        }))
        .await
        .expect("register should succeed");

        // Publish an event
        let result = tool
            .execute(json!({
                "action": "publish",
                "event_type": "test_event",
                "payload": "Hello from integration test",
                "channel": "all"
            }))
            .await
            .expect("publish should succeed");

        let event_id = result.content["event_id"].as_i64().unwrap();
        assert!(event_id > 0);

        // Get events since before our event
        let result = tool
            .execute(json!({
                "action": "get_events",
                "since_id": event_id - 1,
                "limit": 10
            }))
            .await
            .expect("get_events should succeed");

        let events = result.content["events"].as_array().unwrap();
        assert!(!events.is_empty());

        // Find our event
        let our_event = events.iter().find(|e| e["id"].as_i64() == Some(event_id));
        assert!(our_event.is_some());
        assert_eq!(our_event.unwrap()["event_type"], "test_event");

        // Cleanup
        let _ = tool.execute(json!({"action": "unregister"})).await;
    }

    #[tokio::test]
    #[ignore = "requires running event bus"]
    async fn test_integration_list_sessions() {
        let tool = EventBus::from_env();

        // Register
        tool.execute(json!({
            "action": "register",
            "name": "list-test-agent"
        }))
        .await
        .expect("register should succeed");

        // List sessions
        let result = tool
            .execute(json!({"action": "list_sessions"}))
            .await
            .expect("list_sessions should succeed");

        let sessions = result.content["sessions"].as_array().unwrap();
        assert!(!sessions.is_empty());

        // Find our session
        let our_session = sessions
            .iter()
            .find(|s| s["name"].as_str() == Some("list-test-agent"));
        assert!(our_session.is_some());

        // Cleanup
        let _ = tool.execute(json!({"action": "unregister"})).await;
    }
}
