use thiserror::Error;

/// Top-level error type for the gemicro library
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GemicroError {
    /// Error from an agent
    #[error("Agent error: {0}")]
    Agent(#[from] AgentError),

    /// Error from the LLM client
    #[error("LLM error: {0}")]
    Llm(#[from] LlmError),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Errors that can occur during agent execution
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AgentError {
    /// LLM client error during agent execution
    #[error("LLM client error: {0}")]
    Llm(#[from] LlmError),

    /// Failed to decompose query
    #[error("Failed to decompose query: {0}")]
    DecompositionFailed(String),

    /// Failed to parse LLM response
    #[error("Failed to parse response: {0}")]
    ParseFailed(String),

    /// All sub-queries failed
    #[error("All sub-queries failed, cannot synthesize result")]
    AllSubQueriesFailed,

    /// Synthesis failed
    #[error("Failed to synthesize results: {0}")]
    SynthesisFailed(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Agent execution timed out
    #[error("Timeout after {elapsed_ms}ms (limit: {timeout_ms}ms) during {phase}")]
    Timeout {
        elapsed_ms: u64,
        timeout_ms: u64,
        phase: String,
    },

    /// Agent execution was cancelled
    #[error("Execution cancelled")]
    Cancelled,

    /// Other agent-specific error
    #[error("{0}")]
    Other(String),
}

impl AgentError {
    /// Check if this error is retriable (transient failures).
    ///
    /// Returns `true` for errors that might succeed on retry:
    /// - Timeouts (agent-level or LLM-level)
    /// - Rate limits
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_core::AgentError;
    ///
    /// let timeout = AgentError::Timeout {
    ///     elapsed_ms: 5000,
    ///     timeout_ms: 3000,
    ///     phase: "synthesis".into(),
    /// };
    /// assert!(timeout.is_retriable());
    ///
    /// let cancelled = AgentError::Cancelled;
    /// assert!(!cancelled.is_retriable());
    /// ```
    pub fn is_retriable(&self) -> bool {
        matches!(
            self,
            AgentError::Timeout { .. }
                | AgentError::Llm(LlmError::RateLimit(_))
                | AgentError::Llm(LlmError::Timeout(_))
        )
    }

    /// Check if this is a timeout error.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_core::AgentError;
    ///
    /// let timeout = AgentError::Timeout {
    ///     elapsed_ms: 5000,
    ///     timeout_ms: 3000,
    ///     phase: "synthesis".into(),
    /// };
    /// assert!(timeout.is_timeout());
    /// ```
    pub fn is_timeout(&self) -> bool {
        matches!(self, AgentError::Timeout { .. })
    }

    /// Check if execution was cancelled.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_core::AgentError;
    ///
    /// let cancelled = AgentError::Cancelled;
    /// assert!(cancelled.is_cancelled());
    /// ```
    pub fn is_cancelled(&self) -> bool {
        matches!(self, AgentError::Cancelled)
    }
}

/// Errors that can occur in the LLM client
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LlmError {
    /// Error from the underlying genai-rs library
    #[error("GenAI error: {0}")]
    GenAi(genai_rs::GenaiError),

    /// Request timed out
    #[error("Request timed out after {0}ms")]
    Timeout(u64),

    /// Invalid request
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Response processing error
    #[error("Failed to process response: {0}")]
    ResponseProcessing(String),

    /// No content in response
    #[error("No content in response")]
    NoContent,

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {0}")]
    RateLimit(String),

    /// Request was cancelled
    #[error("Request cancelled")]
    Cancelled,

    /// Other LLM error
    #[error("{0}")]
    Other(String),
}

impl LlmError {
    /// Get the retry-after duration if this is a rate limit error.
    ///
    /// Returns `Some(Duration)` if the underlying GenAI error has a Retry-After
    /// header (typically from a 429 response). Use this for smarter backoff.
    ///
    /// Returns `None` for non-rate-limit errors or if no Retry-After was provided.
    pub fn retry_after(&self) -> Option<std::time::Duration> {
        match self {
            LlmError::GenAi(e) => e.retry_after(),
            _ => None,
        }
    }

    /// Check if this error is retryable.
    ///
    /// Returns `true` for transient errors that might succeed on retry:
    /// - Timeouts
    /// - Rate limits
    /// - GenAI errors that are marked as retryable (5xx, network issues, etc.)
    pub fn is_retryable(&self) -> bool {
        match self {
            LlmError::Timeout(_) => true,
            LlmError::RateLimit(_) => true,
            LlmError::GenAi(e) => e.is_retryable(),
            _ => false,
        }
    }
}

impl From<genai_rs::GenaiError> for LlmError {
    fn from(error: genai_rs::GenaiError) -> Self {
        // Map GenaiError::Timeout to LlmError::Timeout for consistent API
        if let genai_rs::GenaiError::Timeout(duration) = &error {
            return LlmError::Timeout(duration.as_millis() as u64);
        }
        LlmError::GenAi(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    // Parameterized test for AgentError display messages
    #[rstest]
    #[case::decomposition_failed(
        AgentError::DecompositionFailed("Invalid JSON".into()),
        &["decompose", "Invalid JSON"]
    )]
    #[case::parse_failed(
        AgentError::ParseFailed("bad format".into()),
        &["parse", "bad format"]
    )]
    #[case::synthesis_failed(
        AgentError::SynthesisFailed("Empty response".into()),
        &["synthesize", "Empty response"]
    )]
    #[case::all_sub_queries_failed(
        AgentError::AllSubQueriesFailed,
        &["sub-queries failed"]
    )]
    #[case::cancelled(
        AgentError::Cancelled,
        &["cancelled"]
    )]
    #[case::timeout(
        AgentError::Timeout { elapsed_ms: 5000, timeout_ms: 3000, phase: "decomposition".into() },
        &["5000", "3000", "decomposition"]
    )]
    #[case::invalid_config(
        AgentError::InvalidConfig("min > max".into()),
        &["configuration", "min > max"]
    )]
    fn test_agent_error_display(#[case] error: AgentError, #[case] expected: &[&str]) {
        let display = error.to_string();
        for s in expected {
            assert!(display.contains(s), "Expected '{}' in '{}'", s, display);
        }
    }

    #[test]
    fn test_llm_error_timeout() {
        let err = LlmError::Timeout(5000);
        assert!(err.to_string().contains("5000"));
        assert!(err.to_string().contains("timed out"));
    }

    #[test]
    fn test_error_conversion() {
        let llm_err = LlmError::NoContent;
        let agent_err: AgentError = llm_err.into();
        assert!(matches!(agent_err, AgentError::Llm(_)));

        let gemicro_err: GemicroError = agent_err.into();
        assert!(matches!(gemicro_err, GemicroError::Agent(_)));
    }

    #[test]
    fn test_genai_timeout_maps_to_llm_timeout() {
        use std::time::Duration;

        // GenaiError::Timeout should be mapped to LlmError::Timeout, not GenAi
        let genai_err = genai_rs::GenaiError::Timeout(Duration::from_secs(5));
        let llm_err: LlmError = genai_err.into();

        assert!(
            matches!(llm_err, LlmError::Timeout(ms) if ms == 5000),
            "Expected LlmError::Timeout(5000), got {:?}",
            llm_err
        );
    }

    #[test]
    fn test_genai_other_error_maps_to_genai() {
        // Other GenaiError variants should map to LlmError::GenAi
        let genai_err = genai_rs::GenaiError::Internal("test".to_string());
        let llm_err: LlmError = genai_err.into();

        assert!(
            matches!(llm_err, LlmError::GenAi(_)),
            "Expected LlmError::GenAi, got {:?}",
            llm_err
        );
    }

    // Tests for AgentError query methods
    #[rstest]
    #[case::timeout(AgentError::Timeout { elapsed_ms: 5000, timeout_ms: 3000, phase: "test".into() }, true)]
    #[case::llm_rate_limit(AgentError::Llm(LlmError::RateLimit("quota exceeded".into())), true)]
    #[case::llm_timeout(AgentError::Llm(LlmError::Timeout(5000)), true)]
    #[case::cancelled(AgentError::Cancelled, false)]
    #[case::parse_failed(AgentError::ParseFailed("bad format".into()), false)]
    #[case::other(AgentError::Other("some error".into()), false)]
    fn test_is_retriable(#[case] error: AgentError, #[case] expected: bool) {
        assert_eq!(error.is_retriable(), expected);
    }

    #[rstest]
    #[case::timeout(AgentError::Timeout { elapsed_ms: 5000, timeout_ms: 3000, phase: "test".into() }, true)]
    #[case::cancelled(AgentError::Cancelled, false)]
    #[case::parse_failed(AgentError::ParseFailed("bad format".into()), false)]
    fn test_is_timeout(#[case] error: AgentError, #[case] expected: bool) {
        assert_eq!(error.is_timeout(), expected);
    }

    #[rstest]
    #[case::cancelled(AgentError::Cancelled, true)]
    #[case::timeout(AgentError::Timeout { elapsed_ms: 5000, timeout_ms: 3000, phase: "test".into() }, false)]
    #[case::parse_failed(AgentError::ParseFailed("bad format".into()), false)]
    fn test_is_cancelled(#[case] error: AgentError, #[case] expected: bool) {
        assert_eq!(error.is_cancelled(), expected);
    }
}
