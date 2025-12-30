//! Utilities for the gemicro platform.
//!
//! These are pure functions with no external dependencies, providing
//! common text manipulation and response handling operations used across crates.

use rust_genai::InteractionResponse;

/// Extract total token count from an LLM response.
///
/// Safely converts from `i32` to `u32`, returning `None` on negative values
/// or if usage metadata is unavailable. This centralizes the token extraction
/// pattern used across all agents.
///
/// # Example
///
/// ```no_run
/// use gemicro_core::{LlmClient, LlmConfig, LlmRequest, extract_total_tokens};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = LlmClient::new(
///     rust_genai::Client::builder("api-key".to_string()).build(),
///     LlmConfig::default(),
/// );
/// let response = client.generate(LlmRequest::new("Hello")).await?;
/// let tokens = extract_total_tokens(&response);
/// println!("Tokens used: {:?}", tokens);
/// # Ok(())
/// # }
/// ```
pub fn extract_total_tokens(response: &InteractionResponse) -> Option<u32> {
    response
        .usage
        .as_ref()
        .and_then(|u| u.total_tokens)
        .and_then(|t| u32::try_from(t).ok())
}

/// Truncate text to a maximum character count, adding ellipsis if needed.
///
/// Uses Unicode-aware character counting to handle multi-byte characters correctly.
/// Trims whitespace from input and from truncated output before adding ellipsis.
///
/// # Examples
///
/// ```
/// use gemicro_core::truncate;
///
/// assert_eq!(truncate("hello world", 8), "hello...");
/// assert_eq!(truncate("short", 10), "short");
/// ```
pub fn truncate(s: &str, max_chars: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars.saturating_sub(3)).collect();
        format!("{}...", truncated.trim_end())
    }
}

/// Truncate text with total character count appended.
///
/// Similar to [`truncate`], but appends the total character count for context.
/// Useful for error messages where knowing the full size helps debugging.
///
/// # Examples
///
/// ```
/// use gemicro_core::truncate_with_count;
///
/// let long_text = "a".repeat(500);
/// let result = truncate_with_count(&long_text, 50);
/// assert!(result.ends_with("(500 chars total)"));
/// ```
pub fn truncate_with_count(s: &str, max_chars: usize) -> String {
    let s = s.trim();
    let char_count = s.chars().count();
    if char_count <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars).collect();
        format!("{}... ({} chars total)", truncated.trim_end(), char_count)
    }
}

/// Extract the first sentence or line from text for preview purposes.
///
/// Looks for the first period or newline as a sentence boundary.
/// Falls back to truncation if the first sentence is too short (< 10 chars).
///
/// # Examples
///
/// ```
/// use gemicro_core::first_sentence;
///
/// assert_eq!(first_sentence("Hello world. More text."), "Hello world.");
/// assert_eq!(first_sentence("First line\nSecond line"), "First line");
/// ```
pub fn first_sentence(s: &str) -> String {
    let s = s.trim();
    // Try to find end of first sentence
    if let Some(pos) = s.find(['.', '\n']) {
        let sentence = s[..=pos].trim();
        if sentence.chars().count() >= 10 {
            return truncate(sentence, 100);
        }
    }
    truncate(s, 100)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_response(usage: Option<rust_genai::UsageMetadata>) -> InteractionResponse {
        InteractionResponse {
            id: Some("test".to_string()),
            model: None,
            agent: None,
            input: vec![],
            outputs: vec![],
            status: rust_genai::InteractionStatus::Completed,
            usage,
            tools: None,
            grounding_metadata: None,
            url_context_metadata: None,
            previous_interaction_id: None,
        }
    }

    #[test]
    fn test_extract_total_tokens_none_usage() {
        let response = test_response(None);
        assert_eq!(extract_total_tokens(&response), None);
    }

    #[test]
    fn test_extract_total_tokens_none_total() {
        let response = test_response(Some(rust_genai::UsageMetadata {
            total_tokens: None,
            total_input_tokens: Some(10),
            total_output_tokens: Some(20),
            ..Default::default()
        }));
        assert_eq!(extract_total_tokens(&response), None);
    }

    #[test]
    fn test_extract_total_tokens_valid() {
        let response = test_response(Some(rust_genai::UsageMetadata {
            total_tokens: Some(100),
            ..Default::default()
        }));
        assert_eq!(extract_total_tokens(&response), Some(100));
    }

    #[test]
    fn test_extract_total_tokens_negative() {
        let response = test_response(Some(rust_genai::UsageMetadata {
            total_tokens: Some(-1),
            ..Default::default()
        }));
        // Should return None on negative values
        assert_eq!(extract_total_tokens(&response), None);
    }

    #[test]
    fn test_truncate_short_string() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_exact_length() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_long_string() {
        assert_eq!(truncate("hello world", 8), "hello...");
    }

    #[test]
    fn test_truncate_unicode() {
        // 5 emoji characters
        let emoji = "😀😁😂🤣😃";
        assert_eq!(truncate(emoji, 5), emoji);
        assert_eq!(truncate(emoji, 4), "😀...");
    }

    #[test]
    fn test_truncate_trims_whitespace() {
        assert_eq!(truncate("  hello  ", 10), "hello");
    }

    #[test]
    fn test_truncate_with_count_short_string() {
        assert_eq!(truncate_with_count("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_with_count_exact_length() {
        assert_eq!(truncate_with_count("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_with_count_empty_string() {
        assert_eq!(truncate_with_count("", 10), "");
    }

    #[test]
    fn test_truncate_with_count_zero_max() {
        // Edge case: max_chars = 0 should still work
        let result = truncate_with_count("hello", 0);
        assert!(result.contains("..."));
        assert!(result.contains("5 chars total"));
    }

    #[test]
    fn test_truncate_with_count_long_string() {
        let long = "a".repeat(100);
        let result = truncate_with_count(&long, 20);
        assert!(result.contains("..."));
        assert!(result.ends_with("(100 chars total)"));
    }

    #[test]
    fn test_truncate_with_count_unicode() {
        let emoji = "😀".repeat(50);
        let result = truncate_with_count(&emoji, 10);
        assert!(result.ends_with("(50 chars total)"));
    }

    #[test]
    fn test_first_sentence_with_period() {
        assert_eq!(
            first_sentence("Hello world. More text here."),
            "Hello world."
        );
    }

    #[test]
    fn test_first_sentence_with_newline() {
        assert_eq!(first_sentence("First line\nSecond line"), "First line");
    }

    #[test]
    fn test_first_sentence_short_sentence() {
        // Short sentences (<=10 chars) fall back to truncation
        assert_eq!(first_sentence("Hi. More."), "Hi. More.");
    }

    #[test]
    fn test_first_sentence_no_delimiter() {
        assert_eq!(first_sentence("No delimiter here"), "No delimiter here");
    }
}
