//! Indicatif-based renderer for CLI progress display.

use super::renderer::Renderer;
use super::state::{DisplayState, Phase, SubQueryStatus};
use crate::format::{format_duration, print_final_result, truncate};
use anyhow::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::time::Duration;

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
        phase_bar.enable_steady_tick(Duration::from_millis(120));

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
            pb.set_prefix(format!("{}", sq.id + 1));
            pb.set_message(truncate(&sq.query, 55));
            self.sub_query_bars.insert(sq.id, pb);
        }
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

                // Create a new phase bar for execution
                self.phase_bar = self.multi.add(ProgressBar::new_spinner());
                self.phase_bar.set_style(
                    ProgressStyle::default_spinner()
                        .template("{spinner:.cyan} {msg}")
                        .expect("Invalid template"),
                );
                self.phase_bar
                    .enable_steady_tick(Duration::from_millis(120));
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

                // Create a new phase bar for synthesis
                self.phase_bar = self.multi.add(ProgressBar::new_spinner());
                self.phase_bar.set_style(
                    ProgressStyle::default_spinner()
                        .template("{spinner:.cyan} {msg}")
                        .expect("Invalid template"),
                );
                self.phase_bar
                    .enable_steady_tick(Duration::from_millis(120));
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
                pb.enable_steady_tick(Duration::from_millis(120));
            }

            SubQueryStatus::Completed {
                result_preview,
                tokens,
            } => {
                let duration_str = sq
                    .duration
                    .map(format_duration)
                    .unwrap_or_else(|| "?".to_string());

                pb.finish_with_message(format!(
                    "✅ {} → \"{}\" ({} tokens)",
                    duration_str,
                    truncate(result_preview, 40),
                    tokens
                ));
            }

            SubQueryStatus::Failed { error } => {
                let duration_str = sq
                    .duration
                    .map(format_duration)
                    .unwrap_or_else(|| "?".to_string());

                pb.finish_with_message(format!(
                    "❌ {} → Failed: {}",
                    duration_str,
                    truncate(error, 40)
                ));
            }
        }

        Ok(())
    }

    fn on_final_result(&mut self, state: &DisplayState) -> Result<()> {
        if let Some(result) = state.final_result() {
            print_final_result(
                &result.answer,
                state.elapsed(),
                result.sub_queries_succeeded,
                result.sub_queries_failed,
                result.total_tokens,
                result.tokens_unavailable_count > 0,
            );
        }

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
