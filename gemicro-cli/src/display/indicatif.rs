//! Indicatif-based renderer for CLI progress display.
//!
//! Provides a simple single-spinner display that shows status messages
//! from the agent's execution tracker. This design is agent-agnostic -
//! any agent can provide its own tracker with appropriate status messages.

use super::renderer::Renderer;
use crate::format::{print_final_result, print_interrupted};
use anyhow::Result;
use gemicro_core::ExecutionTracking;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::{Duration, Instant};

/// Spinner animation tick interval in milliseconds.
const SPINNER_TICK_MS: u64 = 120;

/// Renderer using indicatif for progress bar display.
///
/// Uses a single spinner that displays status messages from the agent's tracker.
/// This is simpler and more flexible than tracking individual steps, as each
/// agent can provide its own appropriate status messages.
pub struct IndicatifRenderer {
    spinner: ProgressBar,
    start_time: Instant,
    /// Whether to use plain text output (no markdown rendering).
    plain: bool,
    /// The last status message, for interrupted display.
    last_status: Option<String>,
}

impl IndicatifRenderer {
    /// Create a new IndicatifRenderer.
    ///
    /// If `plain` is true, markdown rendering will be disabled for final output.
    pub fn new(plain: bool) -> Self {
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .expect("Invalid template"),
        );
        spinner.enable_steady_tick(Duration::from_millis(SPINNER_TICK_MS));
        spinner.set_message("Starting...");

        Self {
            spinner,
            start_time: Instant::now(),
            plain,
            last_status: None,
        }
    }
}

impl Default for IndicatifRenderer {
    fn default() -> Self {
        Self::new(false)
    }
}

impl Renderer for IndicatifRenderer {
    fn on_status(&mut self, tracker: &dyn ExecutionTracking) -> Result<()> {
        if let Some(msg) = tracker.status_message() {
            self.spinner.set_message(msg.to_string());
            self.last_status = Some(msg.to_string());
        }
        Ok(())
    }

    fn on_complete(&mut self, tracker: &dyn ExecutionTracking) -> Result<()> {
        self.spinner.finish_and_clear();

        if let Some(result) = tracker.final_result() {
            print_final_result(result, self.start_time.elapsed(), self.plain);
        } else {
            log::warn!("on_complete called but tracker.final_result() returned None");
        }

        Ok(())
    }

    fn on_interrupted(&mut self, tracker: &dyn ExecutionTracking) -> Result<()> {
        self.spinner
            .finish_with_message("⚠️  Execution interrupted by user");

        // Use the tracker's status message if available, otherwise fall back to last known
        let status = tracker
            .status_message()
            .map(|s| s.to_string())
            .or_else(|| self.last_status.clone());

        print_interrupted(status.as_deref());

        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if !self.spinner.is_finished() {
            self.spinner.finish_and_clear();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time validation of constants
    const _: () = assert!(SPINNER_TICK_MS > 0);
    const _: () = assert!(SPINNER_TICK_MS <= 500);

    #[test]
    fn test_new_renderer() {
        let renderer = IndicatifRenderer::new(false);
        assert!(!renderer.plain);
        assert!(renderer.last_status.is_none());
    }

    #[test]
    fn test_default_renderer() {
        let renderer = IndicatifRenderer::default();
        assert!(!renderer.plain);
    }

    #[test]
    fn test_plain_mode() {
        let renderer = IndicatifRenderer::new(true);
        assert!(renderer.plain);
    }
}
