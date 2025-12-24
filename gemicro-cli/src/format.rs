//! Text formatting utilities shared across renderers.

use std::io::IsTerminal;
use std::panic;
use std::time::Duration;
use termimad::MadSkin;

/// Render markdown text for terminal display.
///
/// Uses termimad to render markdown with appropriate styling for the terminal.
/// Falls back to plain text in these cases:
/// - Not running in a terminal (e.g., output piped to a file)
/// - Markdown rendering fails for any reason
///
/// # Supported Markdown Features
///
/// The following markdown elements are rendered with terminal styling:
/// - **Headers** (`#`, `##`, etc.) - displayed with emphasis
/// - **Bold** (`**text**`) and *italic* (`*text*`)
/// - **Code blocks** (fenced with ```) - syntax highlighted when possible
/// - **Inline code** (backticks)
/// - **Lists** (ordered and unordered)
/// - **Links** - URL displayed alongside text
///
/// # Graceful Degradation
///
/// If termimad encounters any issues during rendering, this function
/// gracefully falls back to returning the original plain text rather
/// than panicking or returning an error.
pub fn render_markdown(text: &str) -> String {
    if !std::io::stdout().is_terminal() {
        return text.to_string();
    }

    // Use catch_unwind to gracefully handle any panics from termimad
    let result = panic::catch_unwind(|| {
        let skin = MadSkin::default();
        skin.term_text(text).to_string()
    });

    match result {
        Ok(rendered) => rendered,
        Err(_) => {
            // If rendering fails, fall back to plain text
            log::warn!("Markdown rendering failed, falling back to plain text");
            text.to_string()
        }
    }
}

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

/// Information for printing the final research result.
pub struct FinalResultInfo<'a> {
    /// The synthesized answer
    pub answer: &'a str,
    /// Total duration of the research
    pub duration: Duration,
    /// Estimated sequential execution time (for parallel speedup calculation)
    pub sequential_time: Option<Duration>,
    /// Number of sub-queries that succeeded
    pub sub_queries_succeeded: usize,
    /// Number of sub-queries that failed
    pub sub_queries_failed: usize,
    /// Total tokens used
    pub total_tokens: u32,
    /// Whether token data was unavailable for some requests
    pub tokens_unavailable: bool,
    /// Whether to use plain text output (no markdown)
    pub plain: bool,
}

/// Print the final research result with formatting.
///
/// If `plain` is false and stdout is a terminal, the answer will be rendered
/// as markdown with syntax highlighting and formatting.
pub fn print_final_result(info: &FinalResultInfo<'_>) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    SYNTHESIZED ANSWER                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Render as markdown unless plain mode is requested
    if info.plain {
        println!("{}", info.answer);
    } else {
        println!("{}", render_markdown(info.answer));
    }
    println!();
    println!("══════════════════════════════════════════════════════════════");

    let total = info.sub_queries_succeeded + info.sub_queries_failed;

    println!("📊 Performance:");
    println!("   Total time: {}", format_duration(info.duration));

    // Show parallel speedup if we have sequential timing data
    if let Some(seq_time) = info.sequential_time {
        if seq_time > info.duration {
            let saved = seq_time - info.duration;
            let speedup = seq_time.as_secs_f64() / info.duration.as_secs_f64();
            println!(
                "   Parallel speedup: {:.1}x (saved {})",
                speedup,
                format_duration(saved)
            );
        }
    }

    println!(
        "   Sub-queries: {}/{} succeeded",
        info.sub_queries_succeeded, total
    );

    if !info.tokens_unavailable && info.total_tokens > 0 {
        println!("   Tokens used: {}", info.total_tokens);
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

    #[test]
    fn test_render_markdown_preserves_content() {
        // In test environment (not a terminal), render_markdown returns plain text
        let input = "# Hello\n\nSome **bold** text.";
        let result = render_markdown(input);
        // Should contain the original content (may have formatting removed in non-tty)
        assert!(result.contains("Hello"));
        assert!(result.contains("bold"));
    }

    #[test]
    fn test_render_markdown_empty_string() {
        // Empty string should return empty string without panicking
        let result = render_markdown("");
        assert_eq!(result, "");
    }

    #[test]
    fn test_render_markdown_large_input() {
        // Large input should be handled without panicking
        let large = "# Header\n\nParagraph with **bold** and *italic* text.\n\n".repeat(1000);
        let result = render_markdown(&large);
        // Should not panic and should contain content
        assert!(!result.is_empty());
        assert!(result.contains("Header"));
    }

    #[test]
    fn test_render_markdown_special_characters() {
        // Special characters and edge cases should be handled gracefully
        let input = "Special chars: <>&\"' and unicode: 日本語 émojis: 🎉🚀";
        let result = render_markdown(input);
        assert!(result.contains("Special chars"));
        assert!(result.contains("日本語"));
    }
}
