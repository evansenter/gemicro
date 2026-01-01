//! HTTP client for the claude-event-bus.

use crate::error::EventBusError;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// HTTP client for interacting with the claude-event-bus.
#[derive(Clone, Debug)]
pub struct EventBusClient {
    client: Client,
    base_url: String,
}

/// Session information returned by the event bus.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[non_exhaustive]
pub struct Session {
    /// Unique session identifier.
    pub session_id: String,

    /// Human-readable session name.
    pub name: String,

    /// Last event ID seen by this session (for polling).
    #[serde(default)]
    pub last_event_id: i64,

    /// Machine identifier.
    #[serde(default)]
    pub machine: Option<String>,

    /// Current working directory.
    #[serde(default)]
    pub cwd: Option<String>,
}

/// Event from the event bus.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[non_exhaustive]
pub struct Event {
    /// Unique event identifier.
    pub id: i64,

    /// Event type (e.g., "task_completed", "help_needed").
    pub event_type: String,

    /// Event payload (typically a message).
    pub payload: String,

    /// Channel the event was published to.
    pub channel: String,

    /// Name of the session that published the event.
    #[serde(default)]
    pub session_name: Option<String>,

    /// When the event was created.
    #[serde(default)]
    pub created_at: Option<String>,
}

/// Response when publishing an event.
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct PublishResponse {
    /// The created event's ID.
    pub id: i64,

    /// The event type.
    pub event_type: String,

    /// The channel it was published to.
    pub channel: String,
}

impl EventBusClient {
    /// Create a new client for the event bus.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL of the event bus (e.g., "http://localhost:8765")
    pub fn new(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Register a session with the event bus.
    ///
    /// Returns session info including the assigned session_id.
    pub async fn register(
        &self,
        name: &str,
        machine: Option<&str>,
        cwd: Option<&str>,
        pid: Option<i32>,
    ) -> Result<Session, EventBusError> {
        let mut body = json!({ "name": name });
        if let Some(m) = machine {
            body["machine"] = json!(m);
        }
        if let Some(c) = cwd {
            body["cwd"] = json!(c);
        }
        if let Some(p) = pid {
            body["pid"] = json!(p);
        }

        let resp = self
            .client
            .post(format!("{}/sessions", self.base_url))
            .json(&body)
            .send()
            .await?;

        self.handle_response(resp).await
    }

    /// Unregister a session from the event bus.
    pub async fn unregister(&self, session_id: &str) -> Result<(), EventBusError> {
        let resp = self
            .client
            .delete(format!("{}/sessions/{}", self.base_url, session_id))
            .send()
            .await?;

        if resp.status().is_success() {
            Ok(())
        } else {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            Err(EventBusError::Api {
                status,
                message: text,
            })
        }
    }

    /// Publish an event to the event bus.
    pub async fn publish(
        &self,
        event_type: &str,
        payload: &str,
        channel: &str,
        session_id: Option<&str>,
    ) -> Result<PublishResponse, EventBusError> {
        let mut body = json!({
            "event_type": event_type,
            "payload": payload,
            "channel": channel
        });
        if let Some(sid) = session_id {
            body["session_id"] = json!(sid);
        }

        let resp = self
            .client
            .post(format!("{}/events", self.base_url))
            .json(&body)
            .send()
            .await?;

        self.handle_response(resp).await
    }

    /// Get events from the event bus.
    ///
    /// # Arguments
    ///
    /// * `since_id` - Only return events with ID greater than this (0 for recent events)
    /// * `limit` - Maximum number of events to return
    /// * `session_id` - Optional session ID for channel filtering
    pub async fn get_events(
        &self,
        since_id: i64,
        limit: i32,
        session_id: Option<&str>,
    ) -> Result<Vec<Event>, EventBusError> {
        let mut url = format!(
            "{}/events?since_id={}&limit={}",
            self.base_url, since_id, limit
        );
        if let Some(sid) = session_id {
            url.push_str(&format!("&session_id={}", sid));
        }

        let resp = self.client.get(&url).send().await?;
        self.handle_response(resp).await
    }

    /// List all active sessions.
    pub async fn list_sessions(&self) -> Result<Vec<Session>, EventBusError> {
        let resp = self
            .client
            .get(format!("{}/sessions", self.base_url))
            .send()
            .await?;

        self.handle_response(resp).await
    }

    /// Handle an HTTP response, parsing JSON or returning an error.
    async fn handle_response<T: serde::de::DeserializeOwned>(
        &self,
        resp: reqwest::Response,
    ) -> Result<T, EventBusError> {
        if resp.status().is_success() {
            let text = resp.text().await?;
            serde_json::from_str(&text).map_err(|e| EventBusError::Parse(e.to_string()))
        } else {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            Err(EventBusError::Api {
                status,
                message: text,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_url_trimming() {
        let client = EventBusClient::new("http://localhost:8765/");
        assert_eq!(client.base_url, "http://localhost:8765");

        let client = EventBusClient::new("http://localhost:8765");
        assert_eq!(client.base_url, "http://localhost:8765");
    }

    #[test]
    fn test_session_deserialization() {
        let json = r#"{
            "session_id": "abc123",
            "name": "test-session",
            "last_event_id": 42,
            "machine": "laptop",
            "cwd": "/home/user"
        }"#;
        let session: Session = serde_json::from_str(json).unwrap();
        assert_eq!(session.session_id, "abc123");
        assert_eq!(session.name, "test-session");
        assert_eq!(session.last_event_id, 42);
        assert_eq!(session.machine, Some("laptop".to_string()));
    }

    #[test]
    fn test_session_deserialization_minimal() {
        // Test with only required fields
        let json = r#"{
            "session_id": "abc123",
            "name": "test-session"
        }"#;
        let session: Session = serde_json::from_str(json).unwrap();
        assert_eq!(session.session_id, "abc123");
        assert_eq!(session.last_event_id, 0);
        assert!(session.machine.is_none());
    }

    #[test]
    fn test_event_deserialization() {
        let json = r#"{
            "id": 100,
            "event_type": "task_completed",
            "payload": "Feature X is done",
            "channel": "all",
            "session_name": "happy-tiger",
            "created_at": "2026-01-01T12:00:00Z"
        }"#;
        let event: Event = serde_json::from_str(json).unwrap();
        assert_eq!(event.id, 100);
        assert_eq!(event.event_type, "task_completed");
        assert_eq!(event.payload, "Feature X is done");
        assert_eq!(event.session_name, Some("happy-tiger".to_string()));
    }

    #[test]
    fn test_event_deserialization_minimal() {
        let json = r#"{
            "id": 1,
            "event_type": "test",
            "payload": "msg",
            "channel": "all"
        }"#;
        let event: Event = serde_json::from_str(json).unwrap();
        assert_eq!(event.id, 1);
        assert!(event.session_name.is_none());
        assert!(event.created_at.is_none());
    }

    #[test]
    fn test_publish_response_deserialization() {
        let json = r#"{
            "id": 42,
            "event_type": "help_needed",
            "channel": "repo:gemicro"
        }"#;
        let resp: PublishResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, 42);
        assert_eq!(resp.event_type, "help_needed");
        assert_eq!(resp.channel, "repo:gemicro");
    }
}
