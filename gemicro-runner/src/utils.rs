//! Text utilities for execution tracking.
//!
//! These are pure functions with no terminal dependencies, used for
//! generating previews and formatting durations.

use std::time::Duration;

/// Truncate text to a maximum character count, adding ellipsis if needed.
///
/// Uses Unicode-aware character counting to handle multi-byte characters correctly.
pub fn truncate(s: &str, max_chars: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars.saturating_sub(3)).collect();
        format!("{}...", truncated.trim_end())
    }
}

/// Extract the first sentence or line from text for preview purposes.
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

/// Format a duration as a human-readable string.
pub fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs_f64();
    if secs < 1.0 {
        format!("{}ms", duration.as_millis())
    } else if secs < 10.0 {
        // Show 2 decimal places for times under 10s to differentiate similar times
        format!("{:.2}s", secs)
    } else {
        format!("{:.1}s", secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_format_duration_milliseconds() {
        assert_eq!(format_duration(Duration::from_millis(500)), "500ms");
    }

    #[test]
    fn test_format_duration_seconds_under_10() {
        assert_eq!(format_duration(Duration::from_secs_f64(2.5)), "2.50s");
    }

    #[test]
    fn test_format_duration_seconds_over_10() {
        assert_eq!(format_duration(Duration::from_secs_f64(12.5)), "12.5s");
    }
}
