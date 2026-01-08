//! LLM request and response types.

use rust_genai::Turn;

/// Request to the LLM
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LlmRequest {
    /// User prompt
    pub prompt: String,

    /// Optional conversation history for multi-turn conversations
    ///
    /// When provided, the model receives this structured conversation history
    /// followed by the current `prompt` as the final user turn.
    ///
    /// Use this for:
    /// - Client-managed conversation history (sliding window, summarization)
    /// - Importing conversation history from external sources
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_core::LlmRequest;
    /// use rust_genai::Turn;
    ///
    /// let history = vec![
    ///     Turn::user("What is 2+2?"),
    ///     Turn::model("2+2 equals 4."),
    /// ];
    ///
    /// let request = LlmRequest::new("And what's that times 3?")
    ///     .with_turns(history);
    /// ```
    pub turns: Option<Vec<Turn>>,

    /// Optional system instruction
    pub system_instruction: Option<String>,

    /// Enable Google Search grounding for real-time web data
    ///
    /// When enabled, the model can search the web and ground its responses
    /// in real-time information. This is useful for queries about current
    /// events, recent releases, or live data.
    ///
    /// Note: Grounded requests may have different pricing.
    pub use_google_search: bool,

    /// Optional JSON schema for structured output
    ///
    /// When provided, the model will generate a response that conforms to
    /// this JSON schema. Use this for reliable structured data extraction.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_core::LlmRequest;
    /// use serde_json::json;
    ///
    /// let schema = json!({
    ///     "type": "object",
    ///     "properties": {
    ///         "correct": {"type": "boolean"},
    ///         "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    ///     },
    ///     "required": ["correct", "confidence"]
    /// });
    ///
    /// let request = LlmRequest::new("Is Paris the capital of France?")
    ///     .with_response_format(schema);
    /// ```
    pub response_format: Option<serde_json::Value>,
}

/// Chunk from streaming LLM response
///
/// The stream ends naturally when no more chunks are available (yields `None`).
/// There is no artificial "final" marker - simply iterate until the stream ends.
#[derive(Debug, Clone)]
pub struct LlmStreamChunk {
    /// Text content of this chunk (may be empty for some chunks)
    pub text: String,
}

impl LlmRequest {
    /// Create a new LLM request with just a prompt
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            turns: None,
            system_instruction: None,
            use_google_search: false,
            response_format: None,
        }
    }

    /// Create a new LLM request with prompt and system instruction
    pub fn with_system(prompt: impl Into<String>, system: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            turns: None,
            system_instruction: Some(system.into()),
            use_google_search: false,
            response_format: None,
        }
    }

    /// Set conversation history for multi-turn conversations
    ///
    /// The provided turns become the conversation history. The current `prompt`
    /// is automatically appended as the final user turn when the request is sent.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_core::LlmRequest;
    /// use rust_genai::Turn;
    ///
    /// let history = vec![
    ///     Turn::user("What is 2+2?"),
    ///     Turn::model("2+2 equals 4."),
    /// ];
    ///
    /// let request = LlmRequest::new("And what's that times 3?")
    ///     .with_turns(history);
    /// ```
    pub fn with_turns(mut self, turns: Vec<Turn>) -> Self {
        self.turns = Some(turns);
        self
    }

    /// Enable Google Search grounding for this request
    ///
    /// When enabled, the model can search the web to ground its responses
    /// in real-time information.
    pub fn with_google_search(mut self) -> Self {
        self.use_google_search = true;
        self
    }

    /// Set a JSON schema for structured output
    ///
    /// When provided, the model will generate a response that conforms to
    /// this JSON schema. The response text will be valid JSON matching the schema.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_core::LlmRequest;
    /// use serde_json::json;
    ///
    /// let schema = json!({
    ///     "type": "object",
    ///     "properties": {
    ///         "answer": {"type": "string"},
    ///         "confidence": {"type": "number"}
    ///     },
    ///     "required": ["answer", "confidence"]
    /// });
    ///
    /// let request = LlmRequest::new("What is the capital of France?")
    ///     .with_response_format(schema);
    /// ```
    pub fn with_response_format(mut self, schema: serde_json::Value) -> Self {
        self.response_format = Some(schema);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_request_new() {
        let req = LlmRequest::new("Test prompt");
        assert_eq!(req.prompt, "Test prompt");
        assert!(req.turns.is_none());
        assert!(req.system_instruction.is_none());
        assert!(!req.use_google_search);
        assert!(req.response_format.is_none());
    }

    #[test]
    fn test_llm_request_with_system() {
        let req = LlmRequest::with_system("User prompt", "System instruction");
        assert_eq!(req.prompt, "User prompt");
        assert!(req.turns.is_none());
        assert_eq!(
            req.system_instruction,
            Some("System instruction".to_string())
        );
        assert!(!req.use_google_search);
        assert!(req.response_format.is_none());
    }

    #[test]
    fn test_llm_request_with_google_search() {
        let req = LlmRequest::new("Test prompt").with_google_search();
        assert_eq!(req.prompt, "Test prompt");
        assert!(req.turns.is_none());
        assert!(req.use_google_search);
        assert!(req.response_format.is_none());
    }

    #[test]
    fn test_llm_request_with_system_and_google_search() {
        let req = LlmRequest::with_system("User prompt", "System instruction").with_google_search();
        assert_eq!(req.prompt, "User prompt");
        assert!(req.turns.is_none());
        assert_eq!(
            req.system_instruction,
            Some("System instruction".to_string())
        );
        assert!(req.use_google_search);
        assert!(req.response_format.is_none());
    }

    #[test]
    fn test_llm_request_with_response_format() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "correct": {"type": "boolean"},
                "confidence": {"type": "number"}
            },
            "required": ["correct", "confidence"]
        });

        let req = LlmRequest::new("Test prompt").with_response_format(schema.clone());
        assert_eq!(req.prompt, "Test prompt");
        assert!(req.turns.is_none());
        assert!(req.system_instruction.is_none());
        assert!(!req.use_google_search);
        assert_eq!(req.response_format, Some(schema));
    }

    #[test]
    fn test_llm_request_with_all_options() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"result": {"type": "string"}}
        });

        let req = LlmRequest::with_system("User prompt", "System instruction")
            .with_google_search()
            .with_response_format(schema.clone());

        assert_eq!(req.prompt, "User prompt");
        assert!(req.turns.is_none());
        assert_eq!(
            req.system_instruction,
            Some("System instruction".to_string())
        );
        assert!(req.use_google_search);
        assert_eq!(req.response_format, Some(schema));
    }

    #[test]
    fn test_llm_request_with_all_options_including_turns() {
        let turns = vec![Turn::user("Hello"), Turn::model("Hi!")];
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"result": {"type": "string"}}
        });

        let req = LlmRequest::with_system("User prompt", "System instruction")
            .with_turns(turns)
            .with_google_search()
            .with_response_format(schema.clone());

        assert_eq!(req.prompt, "User prompt");
        assert!(req.turns.is_some());
        assert_eq!(req.turns.as_ref().unwrap().len(), 2);
        assert_eq!(
            req.system_instruction,
            Some("System instruction".to_string())
        );
        assert!(req.use_google_search);
        assert_eq!(req.response_format, Some(schema));
    }

    #[test]
    fn test_llm_request_with_turns() {
        let turns = vec![Turn::user("What is 2+2?"), Turn::model("2+2 equals 4.")];

        let req = LlmRequest::new("And what's that times 3?").with_turns(turns.clone());

        assert_eq!(req.prompt, "And what's that times 3?");
        assert!(req.turns.is_some());
        let stored_turns = req.turns.unwrap();
        assert_eq!(stored_turns.len(), 2);
        assert!(stored_turns[0].is_user());
        assert!(stored_turns[1].is_model());
    }

    #[test]
    fn test_llm_request_with_turns_and_system() {
        let turns = vec![Turn::user("Hello"), Turn::model("Hi!")];

        let req = LlmRequest::with_system("New question", "Be helpful").with_turns(turns.clone());

        assert_eq!(req.prompt, "New question");
        assert!(req.turns.is_some());
        assert_eq!(req.system_instruction, Some("Be helpful".to_string()));
    }

    #[test]
    fn test_llm_request_with_empty_turns() {
        let turns: Vec<Turn> = vec![];
        let req = LlmRequest::new("Test prompt").with_turns(turns);

        assert_eq!(req.prompt, "Test prompt");
        assert!(req.turns.is_some());
        assert!(req.turns.unwrap().is_empty());
    }

    #[test]
    fn test_llm_stream_chunk_creation() {
        let chunk = LlmStreamChunk {
            text: "Hello".to_string(),
        };
        assert_eq!(chunk.text, "Hello");

        let empty_chunk = LlmStreamChunk {
            text: String::new(),
        };
        assert!(empty_chunk.text.is_empty());
    }
}
