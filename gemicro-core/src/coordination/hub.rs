//! Hub-based coordination using SSE for real-time events.
//!
//! This module provides [`HubCoordination`], which connects to the
//! claude-event-bus via Server-Sent Events (SSE) for push-based
//! event delivery.

use super::{Coordination, CoordinationError, ExternalEvent};
use async_trait::async_trait;
use eventsource_client::{Client as SseClient, SSE};
use futures_util::StreamExt;
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

/// Buffer size for the internal event channel.
const EVENT_CHANNEL_BUFFER: usize = 100;

/// Hub-based coordination using SSE for real-time event delivery.
///
/// Connects to the claude-event-bus and receives events via Server-Sent
/// Events. Events are buffered in an internal channel and delivered
/// via [`Coordination::recv_event()`].
///
/// # Lifecycle
///
/// 1. Call [`HubCoordination::connect()`] to register and start SSE listener
/// 2. Use [`Coordination::recv_event()`] in a `tokio::select!` to receive events
/// 3. Use [`Coordination::publish()`] to send events
/// 4. Drop the instance to unregister (best-effort cleanup)
///
/// # Example
///
/// ```text
/// use gemicro_core::coordination::HubCoordination;
///
/// let mut coord = HubCoordination::connect("http://localhost:8765", "my-session").await?;
///
/// // In a select loop:
/// tokio::select! {
///     Some(event) = coord.recv_event() => {
///         println!("Received: {:?}", event);
///     }
///     // ... other branches
/// }
/// ```
pub struct HubCoordination {
    /// HTTP client for API calls (register, publish, etc.).
    http_client: HttpClient,

    /// Base URL of the event bus.
    base_url: String,

    /// Session ID assigned by the event bus.
    session_id: String,

    /// Channel for receiving events from the SSE listener.
    event_rx: mpsc::Receiver<ExternalEvent>,

    /// Handle to the SSE listener task.
    sse_handle: JoinHandle<()>,

    /// Flag indicating if the connection is healthy.
    connected: Arc<AtomicBool>,
}

/// Session information from the event bus.
#[derive(Debug, Deserialize)]
struct SessionResponse {
    session_id: String,
    #[allow(dead_code)]
    name: String,
    #[serde(default)]
    last_event_id: i64,
}

/// Event from the SSE stream.
#[derive(Debug, Deserialize)]
struct SseEvent {
    id: i64,
    event_type: String,
    payload: String,
    channel: String,
    #[serde(default)]
    session_name: Option<String>,
    #[serde(default)]
    created_at: Option<String>,
}

/// Response from publishing an event.
#[derive(Debug, Deserialize, Serialize)]
#[allow(dead_code)]
struct PublishResponse {
    id: i64,
    event_type: String,
    channel: String,
}

impl HubCoordination {
    /// Connect to the event bus and start receiving events.
    ///
    /// Registers a session with the given name and starts an SSE listener
    /// for real-time event delivery.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL of the event bus (e.g., "http://localhost:8765")
    /// * `session_name` - Human-readable session name
    ///
    /// # Errors
    ///
    /// Returns an error if registration or SSE connection fails.
    pub async fn connect(base_url: &str, session_name: &str) -> Result<Self, CoordinationError> {
        let base_url = base_url.trim_end_matches('/').to_string();
        let http_client = HttpClient::new();

        // Register with the event bus
        let session = Self::register(&http_client, &base_url, session_name).await?;
        log::info!(
            "Registered with event bus as {} ({})",
            session_name,
            session.session_id
        );

        // Create channel for events
        let (tx, rx) = mpsc::channel(EVENT_CHANNEL_BUFFER);
        let connected = Arc::new(AtomicBool::new(true));

        // Start SSE listener
        let sse_url = format!(
            "{}/events/stream?session_id={}&since_id={}",
            base_url, session.session_id, session.last_event_id
        );
        let sse_handle = Self::spawn_sse_listener(sse_url, tx, Arc::clone(&connected));

        Ok(Self {
            http_client,
            base_url,
            session_id: session.session_id,
            event_rx: rx,
            sse_handle,
            connected,
        })
    }

    /// Register a session with the event bus.
    async fn register(
        client: &HttpClient,
        base_url: &str,
        name: &str,
    ) -> Result<SessionResponse, CoordinationError> {
        let url = format!("{}/sessions", base_url);
        let body = json!({
            "name": name,
            "cwd": std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
        });

        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| CoordinationError::RegistrationFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(CoordinationError::RegistrationFailed(format!(
                "HTTP {}: {}",
                status, text
            )));
        }

        let session: SessionResponse = resp
            .json()
            .await
            .map_err(|e| CoordinationError::Parse(e.to_string()))?;

        Ok(session)
    }

    /// Spawn the SSE listener task.
    fn spawn_sse_listener(
        url: String,
        tx: mpsc::Sender<ExternalEvent>,
        connected: Arc<AtomicBool>,
    ) -> JoinHandle<()> {
        tokio::spawn(async move {
            log::debug!("Starting SSE listener: {}", url);

            let client = match eventsource_client::ClientBuilder::for_url(&url) {
                Ok(builder) => builder.build(),
                Err(e) => {
                    log::error!("Failed to create SSE client: {}", e);
                    connected.store(false, Ordering::SeqCst);
                    return;
                }
            };

            let mut stream = client.stream();

            while let Some(event_result) = stream.next().await {
                match event_result {
                    Ok(SSE::Event(ev)) => {
                        // Parse the event data
                        match serde_json::from_str::<SseEvent>(&ev.data) {
                            Ok(sse_event) => {
                                let external = ExternalEvent {
                                    id: sse_event.id,
                                    event_type: sse_event.event_type,
                                    payload: sse_event.payload,
                                    channel: sse_event.channel,
                                    source_session: sse_event.session_name,
                                    timestamp: sse_event
                                        .created_at
                                        .and_then(|s| {
                                            chrono::DateTime::parse_from_rfc3339(&s).ok().map(
                                                |dt| {
                                                    SystemTime::UNIX_EPOCH
                                                        + std::time::Duration::from_secs(
                                                            dt.timestamp() as u64,
                                                        )
                                                },
                                            )
                                        })
                                        .unwrap_or_else(SystemTime::now),
                                };

                                if tx.send(external).await.is_err() {
                                    log::debug!("Event receiver dropped, stopping SSE listener");
                                    break;
                                }
                            }
                            Err(e) => {
                                log::warn!("Failed to parse SSE event: {} - data: {}", e, ev.data);
                            }
                        }
                    }
                    Ok(SSE::Connected(_)) => {
                        // Connection established, log and continue
                        log::debug!("SSE connection established");
                    }
                    Ok(SSE::Comment(_)) => {
                        // Heartbeat or comment, ignore
                    }
                    Err(e) => {
                        log::warn!("SSE error: {}", e);
                        connected.store(false, Ordering::SeqCst);
                        break;
                    }
                }
            }

            log::debug!("SSE listener stopped");
            connected.store(false, Ordering::SeqCst);
        })
    }

    /// Unregister from the event bus.
    #[allow(dead_code)]
    async fn unregister(&self) -> Result<(), CoordinationError> {
        let url = format!("{}/sessions/{}", self.base_url, self.session_id);
        self.http_client
            .delete(&url)
            .send()
            .await
            .map_err(|e| CoordinationError::Network(e.to_string()))?;
        Ok(())
    }
}

#[async_trait]
impl Coordination for HubCoordination {
    async fn recv_event(&mut self) -> Option<ExternalEvent> {
        self.event_rx.recv().await
    }

    async fn publish(
        &self,
        event_type: &str,
        payload: &str,
        channel: &str,
    ) -> Result<(), CoordinationError> {
        let url = format!("{}/events", self.base_url);
        let body = json!({
            "event_type": event_type,
            "payload": payload,
            "channel": channel,
            "session_id": self.session_id,
        });

        let resp = self
            .http_client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| CoordinationError::PublishFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(CoordinationError::PublishFailed(format!(
                "HTTP {}: {}",
                status, text
            )));
        }

        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst) && !self.sse_handle.is_finished()
    }

    fn session_id(&self) -> Option<&str> {
        Some(&self.session_id)
    }
}

impl Drop for HubCoordination {
    fn drop(&mut self) {
        // Abort the SSE listener
        self.sse_handle.abort();

        // Best-effort unregister (can't await in drop)
        let client = self.http_client.clone();
        let url = format!("{}/sessions/{}", self.base_url, self.session_id);
        tokio::spawn(async move {
            if let Err(e) = client.delete(&url).send().await {
                log::debug!("Failed to unregister on drop: {}", e);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_event_deserialization() {
        let json = r#"{
            "id": 42,
            "event_type": "task_completed",
            "payload": "Feature X done",
            "channel": "all",
            "session_name": "happy-tiger",
            "created_at": "2026-01-01T12:00:00Z"
        }"#;
        let event: SseEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.id, 42);
        assert_eq!(event.event_type, "task_completed");
        assert_eq!(event.session_name, Some("happy-tiger".to_string()));
    }

    #[test]
    fn test_sse_event_minimal_deserialization() {
        let json = r#"{
            "id": 1,
            "event_type": "test",
            "payload": "msg",
            "channel": "all"
        }"#;
        let event: SseEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.id, 1);
        assert!(event.session_name.is_none());
        assert!(event.created_at.is_none());
    }

    #[test]
    fn test_session_response_deserialization() {
        let json = r#"{
            "session_id": "abc123",
            "name": "test-session",
            "last_event_id": 100
        }"#;
        let session: SessionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(session.session_id, "abc123");
        assert_eq!(session.last_event_id, 100);
    }
}
