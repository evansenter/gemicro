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
#[non_exhaustive]
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
#[non_exhaustive]
pub struct MetricsSnapshot {
    tools: HashMap<String, ToolStatsSnapshot>,
}

/// Point-in-time statistics for a tool.
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    ///
    /// Handles RwLock poisoning gracefully by recovering the data and continuing.
    /// This is appropriate for best-effort metrics collection.
    fn get_or_create_stats(&self, tool_name: &str) -> ToolStats {
        // Fast path: read lock
        {
            let tools = match self.tools.read() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    log::warn!("Metrics lock was poisoned (read), recovering: {}", poisoned);
                    poisoned.into_inner()
                }
            };
            if let Some(stats) = tools.get(tool_name) {
                return stats.clone();
            }
        }

        // Slow path: write lock
        let mut tools = match self.tools.write() {
            Ok(guard) => guard,
            Err(poisoned) => {
                log::warn!(
                    "Metrics lock was poisoned (write), recovering: {}",
                    poisoned
                );
                poisoned.into_inner()
            }
        };
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
    /// Handles lock poisoning gracefully by recovering the data.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let tools = match self.tools.read() {
            Ok(guard) => guard,
            Err(poisoned) => {
                log::warn!(
                    "Metrics lock was poisoned (snapshot), recovering: {}",
                    poisoned
                );
                poisoned.into_inner()
            }
        };
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
    ///
    /// # Concurrency Note
    ///
    /// This method clears the internal HashMap, removing all tool entries.
    /// If metrics are being recorded concurrently (via `record_invocation` etc.),
    /// there may be brief inconsistencies where a tool has non-zero stats but
    /// zero invocations, due to the sequence:
    /// 1. Thread A: Creates tool entry via `get_or_create_stats`
    /// 2. Thread B: Calls `reset()`, clearing the HashMap
    /// 3. Thread A: Increments stats on the now-orphaned ToolStats
    ///
    /// This is acceptable for best-effort metrics. For strict consistency,
    /// avoid calling `reset()` during active tool execution.
    pub fn reset(&self) {
        let mut tools = match self.tools.write() {
            Ok(guard) => guard,
            Err(poisoned) => {
                log::warn!(
                    "Metrics lock was poisoned (reset), recovering: {}",
                    poisoned
                );
                poisoned.into_inner()
            }
        };
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

    #[tokio::test]
    async fn test_concurrent_invocations() {
        use std::sync::Arc;

        let metrics = Arc::new(Metrics::new());

        // Spawn 10 tasks, each doing 100 iterations
        let mut handles = vec![];
        for _ in 0..10 {
            let m = Arc::clone(&metrics);
            handles.push(tokio::spawn(async move {
                for _ in 0..100 {
                    m.pre_tool_use("concurrent_tool", &json!({})).await.unwrap();
                    m.post_tool_use("concurrent_tool", &json!({}), &ToolResult::text("ok"))
                        .await
                        .unwrap();
                }
            }));
        }

        // Wait for all tasks
        for h in handles {
            h.await.unwrap();
        }

        // Verify totals
        let snapshot = metrics.snapshot();
        let stats = snapshot.get("concurrent_tool").unwrap();

        // 10 tasks * 100 iterations = 1000 total
        assert_eq!(
            stats.invocations, 1000,
            "Concurrent invocations not counted correctly"
        );
        assert_eq!(
            stats.successes, 1000,
            "Concurrent successes not counted correctly"
        );
        assert_eq!(stats.failures, 0);
    }

    #[tokio::test]
    async fn test_snapshot_consistency_during_updates() {
        use std::sync::Arc;

        let metrics = Arc::new(Metrics::new());

        // Start background task doing rapid updates
        let m = Arc::clone(&metrics);
        let updater = tokio::spawn(async move {
            for _ in 0..1000 {
                m.pre_tool_use("tool", &json!({})).await.unwrap();
                m.post_tool_use("tool", &json!({}), &ToolResult::text("ok"))
                    .await
                    .unwrap();
            }
        });

        // Take snapshots concurrently
        for _ in 0..20 {
            let snapshot = metrics.snapshot();
            if let Some(stats) = snapshot.get("tool") {
                // Invariant: invocations should equal successes + failures
                assert_eq!(
                    stats.invocations,
                    stats.successes + stats.failures,
                    "Snapshot inconsistency detected"
                );
            }
            tokio::task::yield_now().await;
        }

        updater.await.unwrap();
    }

    #[tokio::test]
    async fn test_reset_during_active_recording() {
        let metrics = Metrics::new();

        // Start recording
        metrics.pre_tool_use("tool", &json!({})).await.unwrap();

        // Reset while in "pre" state (before post)
        metrics.reset();

        // Now call post for a tool that was reset
        metrics
            .post_tool_use("tool", &json!({}), &ToolResult::text("ok"))
            .await
            .unwrap();

        let snapshot = metrics.snapshot();

        // After reset and post, tool should exist with 0 invocations, 1 success
        // This is acceptable behavior - metrics are best-effort
        if let Some(stats) = snapshot.get("tool") {
            assert_eq!(stats.invocations, 0, "Invocations should be 0 after reset");
            assert_eq!(stats.successes, 1, "Success recorded after reset");
        }
    }

    #[tokio::test]
    async fn test_incomplete_tool_execution() {
        // Documents behavior when pre_tool_use is called but post_tool_use never is
        // (e.g., tool execution crashes between hooks)
        let metrics = Metrics::new();

        // Call pre_tool_use but never call post_tool_use
        metrics.pre_tool_use("tool", &json!({})).await.unwrap();

        let snapshot = metrics.snapshot();
        let stats = snapshot.get("tool").unwrap();

        // Document the expected invariant violation:
        // invocations is incremented but successes + failures is not
        assert_eq!(stats.invocations, 1);
        assert_eq!(
            stats.successes + stats.failures,
            0,
            "Expected invariant gap when post_tool_use never called"
        );
    }

    #[tokio::test]
    async fn test_error_metadata_edge_cases() {
        let metrics = Metrics::new();

        // Test 1: null value in error field - should NOT count as failure
        metrics.pre_tool_use("tool1", &json!({})).await.unwrap();
        metrics
            .post_tool_use(
                "tool1",
                &json!({}),
                &ToolResult::text("ok").with_metadata(json!({"error": null})),
            )
            .await
            .unwrap();

        // Test 2: empty object in error field - should count as failure (key exists)
        metrics.pre_tool_use("tool2", &json!({})).await.unwrap();
        metrics
            .post_tool_use(
                "tool2",
                &json!({}),
                &ToolResult::text("ok").with_metadata(json!({"error": {}})),
            )
            .await
            .unwrap();

        // Test 3: false in error field - should count as failure (key exists)
        metrics.pre_tool_use("tool3", &json!({})).await.unwrap();
        metrics
            .post_tool_use(
                "tool3",
                &json!({}),
                &ToolResult::text("ok").with_metadata(json!({"error": false})),
            )
            .await
            .unwrap();

        let snapshot = metrics.snapshot();

        // null counts as "exists" in JSON, so is_some() returns true
        let tool1_stats = snapshot.get("tool1").unwrap();
        assert_eq!(
            tool1_stats.failures, 1,
            "null error value should count as failure (key exists)"
        );

        let tool2_stats = snapshot.get("tool2").unwrap();
        assert_eq!(
            tool2_stats.failures, 1,
            "empty object error should count as failure"
        );

        let tool3_stats = snapshot.get("tool3").unwrap();
        assert_eq!(
            tool3_stats.failures, 1,
            "false error value should count as failure"
        );
    }

    // Property-based tests

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// Property: invocations = successes + failures (after complete call pairs)
            #[test]
            fn invocations_equals_successes_plus_failures(
                num_success in 0usize..100,
                num_failure in 0usize..100,
            ) {
                let metrics = Metrics::new();
                let rt = tokio::runtime::Runtime::new().unwrap();

                rt.block_on(async {
                    // Record successes
                    for _ in 0..num_success {
                        metrics.pre_tool_use("tool", &json!({})).await.unwrap();
                        metrics
                            .post_tool_use("tool", &json!({}), &ToolResult::text("ok"))
                            .await
                            .unwrap();
                    }

                    // Record failures
                    for _ in 0..num_failure {
                        metrics.pre_tool_use("tool", &json!({})).await.unwrap();
                        metrics
                            .post_tool_use(
                                "tool",
                                &json!({}),
                                &ToolResult::text("err").with_metadata(json!({"error": "fail"})),
                            )
                            .await
                            .unwrap();
                    }

                    let snapshot = metrics.snapshot();
                    if let Some(stats) = snapshot.get("tool") {
                        prop_assert_eq!(
                            stats.invocations,
                            (num_success + num_failure) as u64,
                            "Invocations should equal sum"
                        );
                        prop_assert_eq!(stats.successes, num_success as u64);
                        prop_assert_eq!(stats.failures, num_failure as u64);
                    } else if num_success + num_failure > 0 {
                        return Err(TestCaseError::fail("Expected stats but none found"));
                    }

                    Ok(())
                })?;
            }

            /// Property: Reset clears all metrics
            #[test]
            fn reset_clears_all_metrics(num_invocations in 1usize..50) {
                let metrics = Metrics::new();
                let rt = tokio::runtime::Runtime::new().unwrap();

                rt.block_on(async {
                    // Record some invocations
                    for _ in 0..num_invocations {
                        metrics.pre_tool_use("tool", &json!({})).await.unwrap();
                        metrics
                            .post_tool_use("tool", &json!({}), &ToolResult::text("ok"))
                            .await
                            .unwrap();
                    }

                    // Verify non-zero
                    prop_assert!(metrics.snapshot().total_invocations() > 0);

                    // Reset
                    metrics.reset();

                    // Verify zero
                    prop_assert_eq!(metrics.snapshot().total_invocations(), 0);

                    Ok(())
                })?;
            }

            /// Property: Multiple tools tracked independently
            #[test]
            fn multiple_tools_independent(
                tool1_calls in 0usize..20,
                tool2_calls in 0usize..20,
            ) {
                let metrics = Metrics::new();
                let rt = tokio::runtime::Runtime::new().unwrap();

                rt.block_on(async {
                    // Record tool1 calls
                    for _ in 0..tool1_calls {
                        metrics.pre_tool_use("tool1", &json!({})).await.unwrap();
                        metrics
                            .post_tool_use("tool1", &json!({}), &ToolResult::text("ok"))
                            .await
                            .unwrap();
                    }

                    // Record tool2 calls
                    for _ in 0..tool2_calls {
                        metrics.pre_tool_use("tool2", &json!({})).await.unwrap();
                        metrics
                            .post_tool_use("tool2", &json!({}), &ToolResult::text("ok"))
                            .await
                            .unwrap();
                    }

                    let snapshot = metrics.snapshot();

                    // Verify tool1
                    if tool1_calls > 0 {
                        let tool1_stats = snapshot.get("tool1").unwrap();
                        prop_assert_eq!(tool1_stats.invocations, tool1_calls as u64);
                    }

                    // Verify tool2
                    if tool2_calls > 0 {
                        let tool2_stats = snapshot.get("tool2").unwrap();
                        prop_assert_eq!(tool2_stats.invocations, tool2_calls as u64);
                    }

                    // Verify total
                    prop_assert_eq!(
                        snapshot.total_invocations(),
                        (tool1_calls + tool2_calls) as u64
                    );

                    Ok(())
                })?;
            }
        }
    }
}
