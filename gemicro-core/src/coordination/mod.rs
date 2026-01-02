//! Coordination infrastructure for cross-agent event handling.
//!
//! This module provides the [`Coordination`] trait for real-time event
//! coordination between agents and external systems. The primary implementation
//! is [`HubCoordination`] which connects to the claude-event-bus via SSE.
//!
//! ## Design Philosophy
//!
//! Coordination is opt-in and non-blocking:
//! - Events are received asynchronously via `recv_event()`
//! - Agents continue executing even if coordination is unavailable
//! - External events are injected into the agent stream via `AgentRunner`
//!
//! ## Usage
//!
//! ```text
//! use gemicro_core::coordination::{Coordination, HubCoordination};
//!
//! // Connect to the event bus
//! let coord = HubCoordination::connect("http://localhost:8765", "my-session").await?;
//!
//! // Pass to AgentRunner for event injection
//! let stream = runner.execute_with_events(&agent, query, context, Some(coord));
//! ```
//!
//! ## Event Flow
//!
//! ```text
//! ┌──────────────┐     SSE      ┌─────────────────┐
//! │ Event Bus    │◄────────────►│ HubCoordination │
//! └──────────────┘              └────────┬────────┘
//!                                        │
//!                                        ▼ recv_event()
//!                               ┌─────────────────┐
//!                               │  AgentRunner    │──► tokio::select!
//!                               └─────────────────┘         │
//!                                        ▲                  ▼
//!                                        │        ┌─────────────────┐
//!                                        └────────│ Agent Stream    │
//!                                    agent events └─────────────────┘
//! ```

mod error;
mod hub;

pub use error::CoordinationError;
pub use hub::HubCoordination;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// An external event received from the coordination system.
///
/// These events originate from other agents, sessions, or external
/// systems and are injected into the agent's update stream.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ExternalEvent {
    /// Unique event identifier from the event bus.
    pub id: i64,

    /// Event type (e.g., "task_completed", "help_needed").
    pub event_type: String,

    /// Event payload (typically a message or JSON).
    pub payload: String,

    /// Channel the event was published to (e.g., "all", "repo:gemicro").
    pub channel: String,

    /// Name of the session that published the event, if known.
    pub source_session: Option<String>,

    /// When the event was created.
    pub timestamp: SystemTime,
}

impl ExternalEvent {
    /// Create a new external event.
    pub fn new(
        id: i64,
        event_type: impl Into<String>,
        payload: impl Into<String>,
        channel: impl Into<String>,
    ) -> Self {
        Self {
            id,
            event_type: event_type.into(),
            payload: payload.into(),
            channel: channel.into(),
            source_session: None,
            timestamp: SystemTime::now(),
        }
    }

    /// Set the source session for this event.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source_session = Some(source.into());
        self
    }

    /// Set the timestamp for this event.
    pub fn with_timestamp(mut self, timestamp: SystemTime) -> Self {
        self.timestamp = timestamp;
        self
    }
}

/// Trait for coordination backends that provide real-time event handling.
///
/// Implementations connect to an event source (like SSE) and provide
/// methods to receive and publish events. The trait is designed to be
/// used with `tokio::select!` for non-blocking event injection.
///
/// # Example Implementation
///
/// ```text
/// struct MyCoordination {
///     rx: mpsc::Receiver<ExternalEvent>,
///     // ...
/// }
///
/// #[async_trait]
/// impl Coordination for MyCoordination {
///     async fn recv_event(&mut self) -> Option<ExternalEvent> {
///         self.rx.recv().await
///     }
///
///     async fn publish(
///         &self,
///         event_type: &str,
///         payload: &str,
///         channel: &str,
///     ) -> Result<(), CoordinationError> {
///         // Publish via HTTP API
///         Ok(())
///     }
///
///     fn is_connected(&self) -> bool {
///         // Check SSE connection health
///         true
///     }
/// }
/// ```
#[async_trait]
pub trait Coordination: Send + Sync {
    /// Receive the next external event.
    ///
    /// Returns `None` if the connection is lost or the coordination
    /// system is shutting down. Blocks until an event is available.
    ///
    /// This method is designed to be used in a `tokio::select!` branch
    /// alongside the agent's update stream.
    async fn recv_event(&mut self) -> Option<ExternalEvent>;

    /// Publish an event to the coordination system.
    ///
    /// # Arguments
    ///
    /// * `event_type` - The type of event (e.g., "task_completed")
    /// * `payload` - The event payload (typically a message)
    /// * `channel` - The target channel (e.g., "all", "repo:gemicro")
    ///
    /// # Errors
    ///
    /// Returns an error if publishing fails (network error, etc.).
    async fn publish(
        &self,
        event_type: &str,
        payload: &str,
        channel: &str,
    ) -> Result<(), CoordinationError>;

    /// Check if the coordination connection is healthy.
    ///
    /// Returns `false` if the SSE connection is lost or the
    /// coordination system is unavailable.
    fn is_connected(&self) -> bool;

    /// Get the session ID for this coordination instance.
    ///
    /// Returns `None` if not registered with the event bus.
    fn session_id(&self) -> Option<&str>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_external_event_new() {
        let event = ExternalEvent::new(1, "test_event", "Hello", "all");
        assert_eq!(event.id, 1);
        assert_eq!(event.event_type, "test_event");
        assert_eq!(event.payload, "Hello");
        assert_eq!(event.channel, "all");
        assert!(event.source_session.is_none());
    }

    #[test]
    fn test_external_event_with_source() {
        let event =
            ExternalEvent::new(2, "task_done", "Done", "repo:gemicro").with_source("happy-tiger");
        assert_eq!(event.source_session, Some("happy-tiger".to_string()));
    }

    #[test]
    fn test_external_event_with_timestamp() {
        let timestamp = SystemTime::UNIX_EPOCH;
        let event = ExternalEvent::new(3, "event", "payload", "all").with_timestamp(timestamp);
        assert_eq!(event.timestamp, timestamp);
    }

    #[test]
    fn test_external_event_serialization() {
        let event = ExternalEvent::new(42, "test", "payload", "all").with_source("source");
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"id\":42"));
        assert!(json.contains("\"event_type\":\"test\""));
        assert!(json.contains("\"source_session\":\"source\""));
    }

    #[test]
    fn test_external_event_deserialization() {
        let json = r#"{
            "id": 100,
            "event_type": "help_needed",
            "payload": "Need review",
            "channel": "repo:gemicro",
            "source_session": "brave-lion",
            "timestamp": {"secs_since_epoch": 0, "nanos_since_epoch": 0}
        }"#;
        let event: ExternalEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.id, 100);
        assert_eq!(event.event_type, "help_needed");
        assert_eq!(event.source_session, Some("brave-lion".to_string()));
    }
}
