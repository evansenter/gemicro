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

/// Errors that can occur in the LLM client
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LlmError {
    /// Error from the underlying rust-genai library
    #[error("GenAI error: {0}")]
    GenAi(#[from] rust_genai::GenaiError),

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_error_display() {
        let err = AgentError::DecompositionFailed("Invalid JSON".to_string());
        assert!(err.to_string().contains("decompose"));
        assert!(err.to_string().contains("Invalid JSON"));
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
    fn test_all_sub_queries_failed() {
        let err = AgentError::AllSubQueriesFailed;
        assert!(err.to_string().contains("All sub-queries failed"));
    }

    #[test]
    fn test_parse_failed_display() {
        let err = AgentError::ParseFailed("Invalid JSON".to_string());
        let display = err.to_string();
        assert!(display.contains("parse"));
        assert!(display.contains("Invalid JSON"));
    }

    #[test]
    fn test_synthesis_failed_display() {
        let err = AgentError::SynthesisFailed("Empty response".to_string());
        let display = err.to_string();
        assert!(display.contains("synthesize"));
        assert!(display.contains("Empty response"));
    }

    #[test]
    fn test_cancelled_display() {
        let err = AgentError::Cancelled;
        let display = err.to_string();
        assert!(display.contains("cancelled"));
    }

    #[test]
    fn test_timeout_display() {
        let err = AgentError::Timeout {
            elapsed_ms: 5000,
            timeout_ms: 3000,
            phase: "decomposition".to_string(),
        };
        let display = err.to_string();
        assert!(display.contains("5000"));
        assert!(display.contains("3000"));
        assert!(display.contains("decomposition"));
    }

    #[test]
    fn test_invalid_config_display() {
        let err = AgentError::InvalidConfig("min > max".to_string());
        let display = err.to_string();
        assert!(display.contains("Invalid configuration"));
        assert!(display.contains("min > max"));
    }
}
