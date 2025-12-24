//! Indicatif-based renderer for CLI progress display.

use super::renderer::Renderer;
use super::state::{DisplayState, Phase, SubQueryStatus};
use crate::format::{format_duration, print_final_result, truncate};
use anyhow::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::time::Duration;

/// Spinner animation tick interval in milliseconds.
const SPINNER_TICK_MS: u64 = 120;

/// Maximum characters for preview strings displayed during execution.
///
/// Used for truncating:
/// - Sub-query text in spinner progress bars
/// - Result previews when sub-queries complete (e.g., `[1] ✅ 2.5s → "The answer..."`)
/// - Error messages when sub-queries fail
/// - Partial results shown on interrupt
///
/// This does NOT affect the final synthesized answer, which is always shown in full.
const PREVIEW_CHARS: usize = 256;

/// Renderer using indicatif for progress bar display.
pub struct IndicatifRenderer {
    multi: MultiProgress,
    phase_bar: ProgressBar,
    sub_query_bars: HashMap<usize, ProgressBar>,
}

impl IndicatifRenderer {
    /// Create a new IndicatifRenderer.
    pub fn new() -> Self {
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
            sub_query_bars: HashMap::new(),
        }
    }

    /// Create progress bars for all sub-queries when entering execution phase.
    fn create_sub_query_bars(&mut self, state: &DisplayState) {
        for sq in state.sub_queries() {
            let pb = self.multi.add(ProgressBar::new_spinner());
            pb.set_style(
                ProgressStyle::default_spinner()
                    .template("   {spinner:.cyan} [{prefix}] {msg}")
                    .expect("Invalid template"),
            );
            let prefix = format!("{}", sq.id + 1);
            log::debug!("Creating sub-query bar: id={}, prefix={}", sq.id, prefix);
            pb.set_prefix(prefix);
            pb.set_message(truncate(&sq.query, PREVIEW_CHARS));
            self.sub_query_bars.insert(sq.id, pb);
        }
        log::debug!("Created {} sub-query bars", self.sub_query_bars.len());
    }
}

impl Default for IndicatifRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer for IndicatifRenderer {
    fn on_phase_change(&mut self, state: &DisplayState) -> Result<()> {
        match state.phase() {
            Phase::NotStarted => {
                // Nothing to do
            }

            Phase::Decomposing => {
                self.phase_bar
                    .set_message("🔍 Analyzing query and generating research plan...");
            }

            Phase::Executing => {
                let count = state.sub_queries().len();
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

                // Create bars for all sub-queries
                self.create_sub_query_bars(state);
            }

            Phase::Synthesizing => {
                // Finish all remaining sub-query bars
                for pb in self.sub_query_bars.values() {
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

            Phase::Complete => {
                self.phase_bar.finish_with_message("✓ Synthesis complete");
            }
        }

        Ok(())
    }

    fn on_sub_query_update(&mut self, state: &DisplayState, id: usize) -> Result<()> {
        let sq = match state.sub_query(id) {
            Some(sq) => sq,
            None => return Ok(()),
        };

        let pb = match self.sub_query_bars.get(&id) {
            Some(pb) => pb,
            None => return Ok(()),
        };

        match &sq.status {
            SubQueryStatus::Pending => {
                // Already showing query text
            }

            SubQueryStatus::InProgress => {
                pb.enable_steady_tick(Duration::from_millis(SPINNER_TICK_MS));
            }

            SubQueryStatus::Completed {
                result_preview,
                tokens,
            } => {
                let duration_str = sq
                    .duration
                    .map(format_duration)
                    .unwrap_or_else(|| "?".to_string());

                // Only show token count if available (non-zero)
                let token_info = if *tokens > 0 {
                    format!(" ({} tokens)", tokens)
                } else {
                    String::new()
                };

                // Suspend multi-progress, print, then resume to avoid display issues
                self.multi.suspend(|| {
                    println!(
                        "   [{}] ✅ {} → \"{}\"{}",
                        id + 1,
                        duration_str,
                        truncate(result_preview, PREVIEW_CHARS),
                        token_info
                    );
                });
                pb.finish_and_clear();
            }

            SubQueryStatus::Failed { error } => {
                let duration_str = sq
                    .duration
                    .map(format_duration)
                    .unwrap_or_else(|| "?".to_string());

                self.multi.suspend(|| {
                    println!(
                        "   [{}] ❌ {} → Failed: {}",
                        id + 1,
                        duration_str,
                        truncate(error, PREVIEW_CHARS)
                    );
                });
                pb.finish_and_clear();
            }
        }

        Ok(())
    }

    fn on_final_result(&mut self, state: &DisplayState) -> Result<()> {
        if let Some(result) = state.final_result() {
            print_final_result(
                &result.answer,
                state.elapsed(),
                state.sequential_time(),
                result.sub_queries_succeeded,
                result.sub_queries_failed,
                result.total_tokens,
                result.tokens_unavailable_count > 0,
            );
        }

        Ok(())
    }

    fn on_interrupted(&mut self, state: &DisplayState) -> Result<()> {
        // Finish all progress bars
        self.phase_bar
            .finish_with_message("⚠️  Research interrupted by user");

        for pb in self.sub_query_bars.values() {
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
        println!("Research was interrupted during: {:?}", state.phase());
        println!();

        // Show completed sub-queries if any
        let completed: Vec<_> = state
            .sub_queries()
            .iter()
            .filter(|sq| matches!(sq.status, SubQueryStatus::Completed { .. }))
            .collect();

        if !completed.is_empty() {
            println!(
                "Completed sub-queries ({}/{}):",
                completed.len(),
                state.sub_queries().len()
            );
            for sq in completed {
                if let SubQueryStatus::Completed { result_preview, .. } = &sq.status {
                    // Use PREVIEW_CHARS * 1.5 for interrupted results to show more context
                    let chars = (PREVIEW_CHARS * 3) / 2;
                    println!("  [{}] {}", sq.id + 1, truncate(result_preview, chars));
                }
            }
        } else {
            println!("No sub-queries completed before interruption.");
        }

        println!();
        println!("💡 Tip: Run again with a higher --timeout to allow more time.");

        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        // Clean up any remaining progress bars
        for (_, pb) in self.sub_query_bars.drain() {
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
