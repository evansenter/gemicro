//! Error types for the EventBus tool.

use thiserror::Error;

/// Errors that can occur when using the EventBus tool.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EventBusError {
    /// HTTP request failed.
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// Failed to parse response JSON.
    #[error("Failed to parse response: {0}")]
    Parse(String),

    /// Event bus returned an error response.
    #[error("Event bus error: {status} - {message}")]
    Api { status: u16, message: String },
}
