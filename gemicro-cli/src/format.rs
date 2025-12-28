//! Text formatting utilities shared across renderers.

use gemicro_core::FinalResultData;
use std::io::IsTerminal;
use std::panic;
use std::time::Duration;
use termimad::MadSkin;

// Re-export text utilities from gemicro-runner
pub use gemicro_runner::{format_duration, truncate};

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

/// Print the final result with formatting.
///
/// Uses `FinalResultData` from gemicro-core which contains the answer,
/// token counts, and agent-specific metadata in the `extra` field.
///
/// If `plain` is false and stdout is a terminal, the answer will be rendered
/// as markdown with syntax highlighting and formatting.
pub fn print_final_result(result: &FinalResultData, elapsed: Duration, plain: bool) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    SYNTHESIZED ANSWER                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Render as markdown unless plain mode is requested
    if plain {
        println!("{}", result.answer);
    } else {
        println!("{}", render_markdown(&result.answer));
    }
    println!();
    println!("══════════════════════════════════════════════════════════════");

    // Extract step counts from extra field (agent-specific data)
    let steps_succeeded = result.extra["steps_succeeded"].as_u64().unwrap_or(0) as usize;
    let steps_failed = result.extra["steps_failed"].as_u64().unwrap_or(0) as usize;
    let total = steps_succeeded + steps_failed;

    println!("📊 Performance:");
    println!("   Total time: {}", format_duration(elapsed));

    // Show step counts if available
    if total > 0 {
        println!("   Steps: {}/{} succeeded", steps_succeeded, total);
    }

    // Show tokens if available and non-zero
    if result.tokens_unavailable_count == 0 && result.total_tokens > 0 {
        println!("   Tokens used: {}", result.total_tokens);
    }
}

/// Print a message when execution is interrupted.
///
/// Shows what was in progress and provides a tip about timeout.
pub fn print_interrupted(status: Option<&str>) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                      INTERRUPTED                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    if let Some(msg) = status {
        println!("Status when interrupted: {}", msg);
    } else {
        println!("Execution was interrupted.");
    }

    println!();
    println!("💡 Tip: Run again with a higher --timeout to allow more time.");
    println!();
    println!("✓ Cancellation complete");
}

#[cfg(test)]
mod tests {
    use super::*;

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
