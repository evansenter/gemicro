//! Metrics collection hook for gemicro tool execution.
//!
//! Tracks tool usage statistics in memory, providing observability
//! into tool invocation patterns and success rates.
//!
//! # Example
//!
//! ```no_run
//! use gemicro_core::tool::HookRegistry;
//! use gemicro_metrics::Metrics;
//!
//! let metrics = Metrics::new();
//!
//! // Clone to share with hook registry (shallow clone, shares data)
//! let hooks = HookRegistry::new()
//!     .with_hook(metrics.clone());
//!
//! // Later, extract metrics from the original:
//! let snapshot = metrics.snapshot();
//! println!("Total invocations: {}", snapshot.total_invocations());
//! ```

use async_trait::async_trait;
use gemicro_core::tool::{HookDecision, HookError, ToolHook, ToolResult};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Metrics collection hook for tracking tool usage.
///
/// Collects per-tool invocation counts and success/failure rates in memory.
/// Thread-safe and can be shared across multiple hook registries.
///
/// # Usage
///
/// ```no_run
/// use gemicro_metrics::Metrics;
///
/// let metrics = Metrics::new();
///
/// // Clone and use in hook registry...
/// // Later:
/// let snapshot = metrics.snapshot();
/// for (tool, stats) in snapshot.by_tool() {
///     println!("{}: {} calls ({} success, {} failed)",
///         tool, stats.invocations, stats.successes, stats.failures);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct Metrics {
    tools: Arc<RwLock<HashMap<String, ToolStats>>>,
}

/// Statistics for a single tool.
#[derive(Debug, Clone, Default)]
struct ToolStats {
    invocations: Arc<AtomicU64>,
    successes: Arc<AtomicU64>,
    failures: Arc<AtomicU64>,
}

/// Snapshot of metrics at a point in time.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    tools: HashMap<String, ToolStatsSnapshot>,
}

/// Point-in-time statistics for a tool.
#[derive(Debug, Clone)]
pub struct ToolStatsSnapshot {
    pub invocations: u64,
    pub successes: u64,
    pub failures: u64,
}

impl Metrics {
    /// Create a new metrics collector.
    pub fn new() -> Self {
        Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get or create stats for a tool.
    fn get_or_create_stats(&self, tool_name: &str) -> ToolStats {
        // Fast path: read lock
        {
            let tools = self.tools.read().unwrap();
            if let Some(stats) = tools.get(tool_name) {
                return stats.clone();
            }
        }

        // Slow path: write lock
        let mut tools = self.tools.write().unwrap();
        tools.entry(tool_name.to_string()).or_default().clone()
    }

    /// Record a tool invocation.
    fn record_invocation(&self, tool_name: &str) {
        let stats = self.get_or_create_stats(tool_name);
        stats.invocations.fetch_add(1, Ordering::Relaxed);
        log::debug!("Metric: tool.{}.invocations += 1", tool_name);
    }

    /// Record a tool success.
    fn record_success(&self, tool_name: &str) {
        let stats = self.get_or_create_stats(tool_name);
        stats.successes.fetch_add(1, Ordering::Relaxed);
        log::debug!("Metric: tool.{}.success += 1", tool_name);
    }

    /// Record a tool failure.
    fn record_failure(&self, tool_name: &str) {
        let stats = self.get_or_create_stats(tool_name);
        stats.failures.fetch_add(1, Ordering::Relaxed);
        log::debug!("Metric: tool.{}.failure += 1", tool_name);
    }

    /// Get a snapshot of current metrics.
    ///
    /// Returns a point-in-time copy of all collected metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let tools = self.tools.read().unwrap();
        let snapshot_tools = tools
            .iter()
            .map(|(name, stats)| {
                (
                    name.clone(),
                    ToolStatsSnapshot {
                        invocations: stats.invocations.load(Ordering::Relaxed),
                        successes: stats.successes.load(Ordering::Relaxed),
                        failures: stats.failures.load(Ordering::Relaxed),
                    },
                )
            })
            .collect();

        MetricsSnapshot {
            tools: snapshot_tools,
        }
    }

    /// Reset all metrics to zero.
    pub fn reset(&self) {
        let mut tools = self.tools.write().unwrap();
        tools.clear();
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsSnapshot {
    /// Get total invocations across all tools.
    pub fn total_invocations(&self) -> u64 {
        self.tools.values().map(|s| s.invocations).sum()
    }

    /// Get total successes across all tools.
    pub fn total_successes(&self) -> u64 {
        self.tools.values().map(|s| s.successes).sum()
    }

    /// Get total failures across all tools.
    pub fn total_failures(&self) -> u64 {
        self.tools.values().map(|s| s.failures).sum()
    }

    /// Get per-tool statistics.
    pub fn by_tool(&self) -> &HashMap<String, ToolStatsSnapshot> {
        &self.tools
    }

    /// Get statistics for a specific tool.
    pub fn get(&self, tool_name: &str) -> Option<&ToolStatsSnapshot> {
        self.tools.get(tool_name)
    }
}

#[async_trait]
impl ToolHook for Metrics {
    async fn pre_tool_use(
        &self,
        tool_name: &str,
        _input: &Value,
    ) -> Result<HookDecision, HookError> {
        self.record_invocation(tool_name);
        Ok(HookDecision::Allow)
    }

    async fn post_tool_use(
        &self,
        tool_name: &str,
        _input: &Value,
        output: &ToolResult,
    ) -> Result<(), HookError> {
        // Consider it a failure if there's an "error" key in metadata
        let is_error = output.metadata.get("error").is_some();

        if is_error {
            self.record_failure(tool_name);
        } else {
            self.record_success(tool_name);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_metrics_records_invocation() {
        let metrics = Metrics::new();
        let _ = metrics.pre_tool_use("test", &json!({})).await.unwrap();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_invocations(), 1);
    }

    #[tokio::test]
    async fn test_metrics_records_success() {
        let metrics = Metrics::new();
        let result = ToolResult::text("success");

        metrics.pre_tool_use("test", &json!({})).await.unwrap();
        metrics
            .post_tool_use("test", &json!({}), &result)
            .await
            .unwrap();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_successes(), 1);
        assert_eq!(snapshot.total_failures(), 0);
    }

    #[tokio::test]
    async fn test_metrics_records_failure() {
        let metrics = Metrics::new();
        let result =
            ToolResult::text("error").with_metadata(json!({"error": "something went wrong"}));

        metrics.pre_tool_use("test", &json!({})).await.unwrap();
        metrics
            .post_tool_use("test", &json!({}), &result)
            .await
            .unwrap();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_successes(), 0);
        assert_eq!(snapshot.total_failures(), 1);
    }

    #[tokio::test]
    async fn test_metrics_per_tool() {
        let metrics = Arc::new(Metrics::new());

        // Tool A: 2 calls, both success
        metrics.pre_tool_use("tool_a", &json!({})).await.unwrap();
        metrics
            .post_tool_use("tool_a", &json!({}), &ToolResult::text("ok"))
            .await
            .unwrap();

        metrics.pre_tool_use("tool_a", &json!({})).await.unwrap();
        metrics
            .post_tool_use("tool_a", &json!({}), &ToolResult::text("ok"))
            .await
            .unwrap();

        // Tool B: 1 call, failed
        metrics.pre_tool_use("tool_b", &json!({})).await.unwrap();
        metrics
            .post_tool_use(
                "tool_b",
                &json!({}),
                &ToolResult::text("err").with_metadata(json!({"error": "fail"})),
            )
            .await
            .unwrap();

        let snapshot = metrics.snapshot();
        let tool_a_stats = snapshot.get("tool_a").unwrap();
        assert_eq!(tool_a_stats.invocations, 2);
        assert_eq!(tool_a_stats.successes, 2);
        assert_eq!(tool_a_stats.failures, 0);

        let tool_b_stats = snapshot.get("tool_b").unwrap();
        assert_eq!(tool_b_stats.invocations, 1);
        assert_eq!(tool_b_stats.successes, 0);
        assert_eq!(tool_b_stats.failures, 1);
    }

    #[tokio::test]
    async fn test_metrics_reset() {
        let metrics = Metrics::new();
        metrics.pre_tool_use("test", &json!({})).await.unwrap();

        assert_eq!(metrics.snapshot().total_invocations(), 1);

        metrics.reset();
        assert_eq!(metrics.snapshot().total_invocations(), 0);
    }

    #[test]
    fn test_snapshot_totals() {
        let mut tools = HashMap::new();
        tools.insert(
            "tool1".to_string(),
            ToolStatsSnapshot {
                invocations: 10,
                successes: 8,
                failures: 2,
            },
        );
        tools.insert(
            "tool2".to_string(),
            ToolStatsSnapshot {
                invocations: 5,
                successes: 5,
                failures: 0,
            },
        );

        let snapshot = MetricsSnapshot { tools };

        assert_eq!(snapshot.total_invocations(), 15);
        assert_eq!(snapshot.total_successes(), 13);
        assert_eq!(snapshot.total_failures(), 2);
    }
}
