//! Text utilities for execution tracking.
//!
//! Re-exports text utilities from gemicro-core and provides
//! runner-specific formatting functions.

use std::time::Duration;

// Re-export text utilities from core
pub use gemicro_core::{first_sentence, truncate};

/// Format a duration as a human-readable string.
///
/// This is runner-specific as it's used for metrics display.
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

    // Verify re-exports work
    #[test]
    fn test_truncate_reexport() {
        assert_eq!(truncate("hello world", 8), "hello...");
    }

    #[test]
    fn test_first_sentence_reexport() {
        assert_eq!(first_sentence("Hello world. More."), "Hello world.");
    }
}
