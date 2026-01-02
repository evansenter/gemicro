//! Error types for coordination operations.

use std::fmt;

/// Errors that can occur during coordination operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum CoordinationError {
    /// Failed to connect to the event bus.
    ConnectionFailed(String),

    /// SSE connection was lost.
    ConnectionLost(String),

    /// Failed to register with the event bus.
    RegistrationFailed(String),

    /// Failed to publish an event.
    PublishFailed(String),

    /// Network or HTTP error.
    Network(String),

    /// JSON parsing error.
    Parse(String),

    /// Coordination is not available (not configured or disabled).
    NotAvailable,
}

impl fmt::Display for CoordinationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConnectionFailed(msg) => write!(f, "Failed to connect to event bus: {}", msg),
            Self::ConnectionLost(msg) => write!(f, "Event bus connection lost: {}", msg),
            Self::RegistrationFailed(msg) => {
                write!(f, "Failed to register with event bus: {}", msg)
            }
            Self::PublishFailed(msg) => write!(f, "Failed to publish event: {}", msg),
            Self::Network(msg) => write!(f, "Network error: {}", msg),
            Self::Parse(msg) => write!(f, "Parse error: {}", msg),
            Self::NotAvailable => write!(f, "Coordination not available"),
        }
    }
}

impl std::error::Error for CoordinationError {}

impl From<reqwest::Error> for CoordinationError {
    fn from(err: reqwest::Error) -> Self {
        Self::Network(err.to_string())
    }
}

impl From<serde_json::Error> for CoordinationError {
    fn from(err: serde_json::Error) -> Self {
        Self::Parse(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CoordinationError::ConnectionFailed("timeout".to_string());
        assert!(err.to_string().contains("timeout"));
        assert!(err.to_string().contains("connect"));

        let err = CoordinationError::ConnectionLost("reset".to_string());
        assert!(err.to_string().contains("reset"));
        assert!(err.to_string().contains("lost"));

        let err = CoordinationError::NotAvailable;
        assert!(err.to_string().contains("not available"));
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CoordinationError>();
    }
}
