//! Indicatif-based renderer for CLI progress display.

use super::renderer::Renderer;
use crate::format::{format_duration, print_final_result, truncate, FinalResultInfo};
use anyhow::Result;
use gemicro_runner::{phases, ExecutionState, StepStatus};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::time::Duration;

/// Spinner animation tick interval in milliseconds.
const SPINNER_TICK_MS: u64 = 120;

/// Maximum characters for preview strings displayed during execution.
///
/// Used for truncating:
/// - Step text in spinner progress bars
/// - Result previews when steps complete (e.g., `[1] ✅ 2.5s → "The answer..."`)
/// - Error messages when steps fail
///
/// This does NOT affect the final synthesized answer, which is always shown in full.
const PREVIEW_CHARS: usize = 256;

/// Multiplier for showing extra context when execution is interrupted.
///
/// When the user presses Ctrl+C, we show partial results with 50% more context
/// than normal previews, since these may be the only results the user sees.
const INTERRUPTED_CONTEXT_MULTIPLIER: f32 = 1.5;

/// Renderer using indicatif for progress bar display.
pub struct IndicatifRenderer {
    multi: MultiProgress,
    phase_bar: ProgressBar,
    step_bars: HashMap<String, ProgressBar>,
    /// Whether to use plain text output (no markdown rendering).
    plain: bool,
}

impl IndicatifRenderer {
    /// Create a new IndicatifRenderer.
    ///
    /// If `plain` is true, markdown rendering will be disabled for final output.
    pub fn new(plain: bool) -> Self {
        let multi = MultiProgress::new();
        let phase_bar = multi.add(ProgressBar::new_spinner());

        phase_bar.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .expect("Invalid template"),
        );
        phase_bar.enable_steady_tick(Duration::from_millis(SPINNER_TICK_MS));

        Self {
            multi,
            phase_bar,
            step_bars: HashMap::new(),
            plain,
        }
    }

    /// Create progress bars for all steps when entering execution phase.
    fn create_step_bars(&mut self, state: &ExecutionState) {
        for (idx, step) in state.steps().iter().enumerate() {
            let pb = self.multi.add(ProgressBar::new_spinner());
            pb.set_style(
                ProgressStyle::default_spinner()
                    .template("   {spinner:.cyan} [{prefix}] {msg}")
                    .expect("Invalid template"),
            );
            let prefix = format!("{}", idx + 1);
            log::debug!("Creating step bar: id={}, prefix={}", step.id, prefix);
            pb.set_prefix(prefix);
            pb.set_message(truncate(&step.label, PREVIEW_CHARS));
            self.step_bars.insert(step.id.clone(), pb);
        }
        log::debug!("Created {} step bars", self.step_bars.len());
    }
}

impl Default for IndicatifRenderer {
    fn default() -> Self {
        Self::new(false)
    }
}

impl Renderer for IndicatifRenderer {
    fn on_phase_change(&mut self, state: &ExecutionState) -> Result<()> {
        match state.phase() {
            phases::NOT_STARTED => {
                // Nothing to do
            }

            phases::DECOMPOSING => {
                self.phase_bar
                    .set_message("🔍 Analyzing query and generating research plan...");
            }

            phases::EXECUTING => {
                let count = state.steps().len();
                self.phase_bar
                    .finish_with_message(format!("✓ Decomposed into {} sub-queries", count));

                // Create a new phase bar for execution.
                // Defensive check: finish_with_message above should mark it finished,
                // but we guard against unexpected state to prevent orphaned spinners.
                if !self.phase_bar.is_finished() {
                    self.phase_bar.finish_and_clear();
                }
                self.phase_bar = self.multi.add(ProgressBar::new_spinner());
                self.phase_bar.set_style(
                    ProgressStyle::default_spinner()
                        .template("{spinner:.cyan} {msg}")
                        .expect("Invalid template"),
                );
                self.phase_bar
                    .enable_steady_tick(Duration::from_millis(SPINNER_TICK_MS));
                self.phase_bar
                    .set_message(format!("⚡ Executing {} queries in parallel...", count));

                // Create bars for all steps
                self.create_step_bars(state);
            }

            phases::SYNTHESIZING => {
                // Finish all remaining step bars
                for pb in self.step_bars.values() {
                    if !pb.is_finished() {
                        pb.finish_and_clear();
                    }
                }

                self.phase_bar
                    .finish_with_message("✓ All sub-queries complete");

                // Create a new phase bar for synthesis.
                // Defensive check: finish_with_message above should mark it finished,
                // but we guard against unexpected state to prevent orphaned spinners.
                if !self.phase_bar.is_finished() {
                    self.phase_bar.finish_and_clear();
                }
                self.phase_bar = self.multi.add(ProgressBar::new_spinner());
                self.phase_bar.set_style(
                    ProgressStyle::default_spinner()
                        .template("{spinner:.cyan} {msg}")
                        .expect("Invalid template"),
                );
                self.phase_bar
                    .enable_steady_tick(Duration::from_millis(SPINNER_TICK_MS));
                self.phase_bar.set_message("🧠 Synthesizing results...");
            }

            phases::COMPLETE => {
                self.phase_bar.finish_with_message("✓ Synthesis complete");
            }

            // Handle other phases (ReAct, etc.) or unknown phases
            _ => {
                log::debug!("Unhandled phase: {}", state.phase());
            }
        }

        Ok(())
    }

    fn on_step_update(&mut self, state: &ExecutionState, id: &str) -> Result<()> {
        let step = match state.step(id) {
            Some(step) => step,
            None => return Ok(()),
        };

        let pb = match self.step_bars.get(id) {
            Some(pb) => pb,
            None => return Ok(()),
        };

        // Parse numeric ID for display (1-based)
        let display_id = id.parse::<usize>().map(|n| n + 1).unwrap_or(0);

        match &step.status {
            StepStatus::Pending => {
                // Already showing step text
            }

            StepStatus::InProgress => {
                pb.enable_steady_tick(Duration::from_millis(SPINNER_TICK_MS));
            }

            StepStatus::Completed {
                result_preview,
                tokens,
            } => {
                let duration_str = step
                    .duration
                    .map(format_duration)
                    .unwrap_or_else(|| "?".to_string());

                // Only show token count if available
                let token_info = if let Some(t) = tokens {
                    if *t > 0 {
                        format!(" ({} tokens)", t)
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };

                // Suspend multi-progress, print, then resume to avoid display issues
                self.multi.suspend(|| {
                    println!(
                        "   [{}] ✅ {} → \"{}\"{}",
                        display_id,
                        duration_str,
                        truncate(result_preview, PREVIEW_CHARS),
                        token_info
                    );
                });
                pb.finish_and_clear();
            }

            StepStatus::Failed { error } => {
                let duration_str = step
                    .duration
                    .map(format_duration)
                    .unwrap_or_else(|| "?".to_string());

                self.multi.suspend(|| {
                    println!(
                        "   [{}] ❌ {} → Failed: {}",
                        display_id,
                        duration_str,
                        truncate(error, PREVIEW_CHARS)
                    );
                });
                pb.finish_and_clear();
            }

            // Handle future status variants gracefully
            _ => {
                log::warn!("Unknown step status encountered for id {}", id);
            }
        }

        Ok(())
    }

    fn on_final_result(&mut self, state: &ExecutionState) -> Result<()> {
        if let Some(result) = state.final_result() {
            print_final_result(&FinalResultInfo {
                answer: &result.answer,
                duration: state.elapsed(),
                sequential_time: state.sequential_time(),
                steps_succeeded: result.steps_succeeded,
                steps_failed: result.steps_failed,
                total_tokens: result.total_tokens,
                tokens_unavailable: result.tokens_unavailable_count > 0,
                plain: self.plain,
            });
        }

        Ok(())
    }

    fn on_interrupted(&mut self, state: &ExecutionState) -> Result<()> {
        // Finish all progress bars
        self.phase_bar
            .finish_with_message("⚠️  Research interrupted by user");

        for pb in self.step_bars.values() {
            if !pb.is_finished() {
                pb.finish_and_clear();
            }
        }

        // Show partial results header
        println!();
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║                      Partial Results                         ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!();

        // Show phase information
        println!("Research was interrupted during: {}", state.phase());
        println!();

        // Show completed steps if any
        let completed: Vec<_> = state
            .steps()
            .iter()
            .filter(|step| matches!(step.status, StepStatus::Completed { .. }))
            .collect();

        if !completed.is_empty() {
            println!(
                "Completed sub-queries ({}/{}):",
                completed.len(),
                state.steps().len()
            );
            for step in completed {
                if let StepStatus::Completed { result_preview, .. } = &step.status {
                    let chars = (PREVIEW_CHARS as f32 * INTERRUPTED_CONTEXT_MULTIPLIER) as usize;
                    // Parse numeric ID for display (1-based)
                    let display_id = step.id.parse::<usize>().map(|n| n + 1).unwrap_or(0);
                    println!("  [{}] {}", display_id, truncate(result_preview, chars));
                }
            }
        } else {
            println!("No sub-queries completed before interruption.");
        }

        println!();
        println!("💡 Tip: Run again with a higher --timeout to allow more time.");
        println!();
        println!("✓ Cancellation complete");

        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        // Clean up any remaining progress bars
        for (_, pb) in self.step_bars.drain() {
            if !pb.is_finished() {
                pb.finish_and_clear();
            }
        }

        if !self.phase_bar.is_finished() {
            self.phase_bar.finish_and_clear();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time validation of constants
    const _: () = assert!(PREVIEW_CHARS >= 100, "PREVIEW_CHARS too small");
    const _: () = assert!(PREVIEW_CHARS <= 500, "PREVIEW_CHARS too large");
    const _: () = assert!(INTERRUPTED_CONTEXT_MULTIPLIER > 1.0);
    const _: () = assert!(INTERRUPTED_CONTEXT_MULTIPLIER <= 2.0);

    #[test]
    fn test_interrupted_context_calculation() {
        // Verify the calculated value for interrupted results
        let chars = (PREVIEW_CHARS as f32 * INTERRUPTED_CONTEXT_MULTIPLIER) as usize;
        assert!(chars > PREVIEW_CHARS);
        assert_eq!(chars, 384); // 256 * 1.5
    }

    #[test]
    fn test_truncate_at_preview_chars() {
        let long_text = "a".repeat(500);
        let truncated = truncate(&long_text, PREVIEW_CHARS);

        // Should be truncated to PREVIEW_CHARS (including "...")
        assert!(truncated.chars().count() <= PREVIEW_CHARS);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_short_text_not_truncated() {
        let short_text = "This is a short result";
        let result = truncate(short_text, PREVIEW_CHARS);

        // Short text should pass through unchanged
        assert_eq!(result, short_text);
    }
}
