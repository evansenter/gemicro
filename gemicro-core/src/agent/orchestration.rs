//! Orchestration infrastructure for subagent concurrency control.
//!
//! This module provides configuration and runtime state for managing
//! concurrent subagent execution, including:
//!
//! - **Global limits**: Maximum concurrent subagents across all parents
//! - **Per-parent limits**: Maximum concurrent subagents per parent agent
//! - **Depth limits**: Maximum nesting depth to prevent infinite recursion
//! - **Total timeout**: Budget for entire execution tree (root to leaves)
//!
//! # Example
//!
//! ```
//! use gemicro_core::agent::{OrchestrationConfig, OrchestrationState};
//! use std::time::Duration;
//!
//! // Create config with custom limits
//! let config = OrchestrationConfig::default()
//!     .with_global_max_concurrent(5)
//!     .with_max_depth(2)
//!     .with_total_timeout(Duration::from_secs(120));
//!
//! // Validate before use
//! config.validate().expect("config should be valid");
//!
//! // Create runtime state (shared across execution tree)
//! let state = OrchestrationState::new(config);
//! ```

use crate::agent::ExecutionId;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::{OwnedSemaphorePermit, RwLock, Semaphore};

/// Default values for orchestration configuration.
pub const DEFAULT_GLOBAL_MAX_CONCURRENT: usize = 10;
pub const DEFAULT_PER_PARENT_MAX_CONCURRENT: usize = 5;
pub const DEFAULT_MAX_DEPTH: usize = 3;
pub const DEFAULT_TOTAL_TIMEOUT_SECS: u64 = 300; // 5 minutes

/// Configuration for subagent orchestration.
///
/// Controls concurrency limits, nesting depth, and timeout budgets
/// for subagent execution trees.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct OrchestrationConfig {
    /// Maximum concurrent subagents across ALL parent agents (global limit).
    pub global_max_concurrent: usize,

    /// Maximum concurrent subagents PER parent agent.
    pub per_parent_max_concurrent: usize,

    /// Maximum nesting depth (prevents infinite recursion).
    ///
    /// A value of 3 means: root → child → grandchild → great-grandchild (depth 3).
    pub max_depth: usize,

    /// Total timeout for entire execution tree (root to all leaves).
    ///
    /// This budget is shared across all subagents in the tree.
    pub total_timeout: Duration,
}

impl Default for OrchestrationConfig {
    fn default() -> Self {
        Self {
            global_max_concurrent: DEFAULT_GLOBAL_MAX_CONCURRENT,
            per_parent_max_concurrent: DEFAULT_PER_PARENT_MAX_CONCURRENT,
            max_depth: DEFAULT_MAX_DEPTH,
            total_timeout: Duration::from_secs(DEFAULT_TOTAL_TIMEOUT_SECS),
        }
    }
}

impl OrchestrationConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if any value is invalid (zero or otherwise inappropriate).
    pub fn validate(&self) -> Result<(), OrchestrationError> {
        if self.global_max_concurrent == 0 {
            return Err(OrchestrationError::InvalidConfig(
                "global_max_concurrent must be > 0".into(),
            ));
        }
        if self.per_parent_max_concurrent == 0 {
            return Err(OrchestrationError::InvalidConfig(
                "per_parent_max_concurrent must be > 0".into(),
            ));
        }
        if self.max_depth == 0 {
            return Err(OrchestrationError::InvalidConfig(
                "max_depth must be > 0".into(),
            ));
        }
        if self.total_timeout.is_zero() {
            return Err(OrchestrationError::InvalidConfig(
                "total_timeout must be > 0".into(),
            ));
        }
        Ok(())
    }

    /// Set the global maximum concurrent subagents.
    pub fn with_global_max_concurrent(mut self, limit: usize) -> Self {
        self.global_max_concurrent = limit;
        self
    }

    /// Set the per-parent maximum concurrent subagents.
    pub fn with_per_parent_max_concurrent(mut self, limit: usize) -> Self {
        self.per_parent_max_concurrent = limit;
        self
    }

    /// Set the maximum nesting depth.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set the total timeout for the execution tree.
    pub fn with_total_timeout(mut self, timeout: Duration) -> Self {
        self.total_timeout = timeout;
        self
    }
}

/// Errors that can occur during orchestration.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OrchestrationError {
    /// Configuration is invalid.
    #[error("invalid orchestration config: {0}")]
    InvalidConfig(String),

    /// Global concurrent subagent limit exceeded.
    #[error("global concurrent subagent limit exceeded")]
    GlobalLimitExceeded,

    /// Per-parent concurrent subagent limit exceeded.
    #[error("per-parent concurrent subagent limit exceeded")]
    PerParentLimitExceeded,

    /// Maximum nesting depth exceeded.
    #[error("max depth {depth} exceeds limit {limit}")]
    MaxDepthExceeded {
        /// Current depth that triggered the error.
        depth: usize,
        /// Configured maximum depth.
        limit: usize,
    },

    /// Total timeout for execution tree exceeded.
    #[error("total timeout exceeded: {elapsed:?} >= {limit:?}")]
    TotalTimeoutExceeded {
        /// Time elapsed since execution started.
        elapsed: Duration,
        /// Configured total timeout.
        limit: Duration,
    },

    /// Orchestration system is shutting down.
    #[error("orchestration shutdown")]
    Shutdown,
}

/// Runtime state for orchestration (shared across execution tree).
///
/// This struct manages semaphores for enforcing concurrency limits
/// and tracks the start time for total timeout calculations.
///
/// # Thread Safety
///
/// This type is designed to be shared across tasks via `Arc<OrchestrationState>`.
/// All internal state uses appropriate synchronization primitives.
#[derive(Clone)]
pub struct OrchestrationState {
    config: OrchestrationConfig,

    /// Global semaphore for all subagents.
    global_semaphore: Arc<Semaphore>,

    /// Per-parent semaphores, keyed by parent execution_id.
    parent_semaphores: Arc<RwLock<HashMap<String, Arc<Semaphore>>>>,

    /// Start time for total timeout calculation.
    start_time: Instant,
}

impl OrchestrationState {
    /// Create new orchestration state from configuration.
    ///
    /// The start time is set to `Instant::now()`, beginning the
    /// total timeout countdown.
    pub fn new(config: OrchestrationConfig) -> Self {
        Self {
            global_semaphore: Arc::new(Semaphore::new(config.global_max_concurrent)),
            parent_semaphores: Arc::new(RwLock::new(HashMap::new())),
            start_time: Instant::now(),
            config,
        }
    }

    /// Get the orchestration configuration.
    pub fn config(&self) -> &OrchestrationConfig {
        &self.config
    }

    /// Get the start time of this orchestration.
    pub fn start_time(&self) -> Instant {
        self.start_time
    }

    /// Check if the maximum depth would be exceeded.
    ///
    /// Call this before spawning a subagent to prevent exceeding depth limits.
    ///
    /// # Arguments
    ///
    /// * `current_depth` - The depth of the current execution context
    ///
    /// # Errors
    ///
    /// Returns `MaxDepthExceeded` if `current_depth >= max_depth`.
    pub fn check_depth(&self, current_depth: usize) -> Result<(), OrchestrationError> {
        if current_depth >= self.config.max_depth {
            return Err(OrchestrationError::MaxDepthExceeded {
                depth: current_depth,
                limit: self.config.max_depth,
            });
        }
        Ok(())
    }

    /// Get remaining time from the total timeout budget.
    ///
    /// Returns `None` if the timeout has already been exceeded.
    pub fn remaining_time(&self) -> Option<Duration> {
        let elapsed = self.start_time.elapsed();
        if elapsed >= self.config.total_timeout {
            None
        } else {
            Some(self.config.total_timeout - elapsed)
        }
    }

    /// Check if the total timeout has been exceeded.
    ///
    /// # Errors
    ///
    /// Returns `TotalTimeoutExceeded` if elapsed time >= total_timeout.
    pub fn check_timeout(&self) -> Result<(), OrchestrationError> {
        let elapsed = self.start_time.elapsed();
        if elapsed >= self.config.total_timeout {
            return Err(OrchestrationError::TotalTimeoutExceeded {
                elapsed,
                limit: self.config.total_timeout,
            });
        }
        Ok(())
    }

    /// Acquire permits for spawning a subagent.
    ///
    /// This method:
    /// 1. Checks total timeout hasn't been exceeded
    /// 2. Acquires a global permit (blocks if at limit)
    /// 3. Acquires a per-parent permit (blocks if at limit)
    ///
    /// Returns a guard that releases permits when dropped.
    ///
    /// # Arguments
    ///
    /// * `parent_id` - The execution ID of the parent agent
    ///
    /// # Errors
    ///
    /// - `TotalTimeoutExceeded`: Timeout was exceeded while waiting
    /// - `GlobalLimitExceeded`: Timed out waiting for global permit
    /// - `PerParentLimitExceeded`: Timed out waiting for per-parent permit
    /// - `Shutdown`: Semaphore was closed
    pub async fn acquire_permits(
        &self,
        parent_id: &ExecutionId,
    ) -> Result<OrchestrationGuard, OrchestrationError> {
        // Check total timeout first
        self.check_timeout()?;

        // Calculate remaining time for acquire timeout
        let remaining =
            self.remaining_time()
                .ok_or_else(|| OrchestrationError::TotalTimeoutExceeded {
                    elapsed: self.start_time.elapsed(),
                    limit: self.config.total_timeout,
                })?;

        // Acquire global permit with timeout
        let global_permit =
            tokio::time::timeout(remaining, self.global_semaphore.clone().acquire_owned())
                .await
                .map_err(|_| OrchestrationError::GlobalLimitExceeded)?
                .map_err(|_| OrchestrationError::Shutdown)?;

        // Get or create per-parent semaphore
        let parent_key = parent_id.as_str().to_string();
        let parent_semaphore = {
            // First try read lock (common case: semaphore exists)
            let read_guard = self.parent_semaphores.read().await;
            if let Some(sem) = read_guard.get(&parent_key) {
                sem.clone()
            } else {
                // Need to create - drop read lock and acquire write lock
                drop(read_guard);
                let mut write_guard = self.parent_semaphores.write().await;
                // Double-check in case another task created it
                write_guard
                    .entry(parent_key)
                    .or_insert_with(|| {
                        Arc::new(Semaphore::new(self.config.per_parent_max_concurrent))
                    })
                    .clone()
            }
        };

        // Calculate remaining time for per-parent acquire
        // Note: global_permit is dropped automatically if we return Err
        let remaining = self
            .remaining_time()
            .ok_or(OrchestrationError::TotalTimeoutExceeded {
                elapsed: self.start_time.elapsed(),
                limit: self.config.total_timeout,
            })?;

        let parent_permit =
            match tokio::time::timeout(remaining, parent_semaphore.acquire_owned()).await {
                Ok(Ok(permit)) => permit,
                Ok(Err(_)) => return Err(OrchestrationError::Shutdown),
                Err(_) => return Err(OrchestrationError::PerParentLimitExceeded),
            };

        Ok(OrchestrationGuard {
            _global_permit: global_permit,
            _parent_permit: parent_permit,
        })
    }
}

impl std::fmt::Debug for OrchestrationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrchestrationState")
            .field("config", &self.config)
            .field("start_time", &self.start_time)
            .field(
                "global_available",
                &self.global_semaphore.available_permits(),
            )
            .finish_non_exhaustive()
    }
}

/// RAII guard that releases orchestration permits when dropped.
///
/// This guard holds both a global permit and a per-parent permit.
/// When dropped, both permits are automatically released, allowing
/// other subagents to acquire them.
#[must_use = "permit is released when guard is dropped"]
pub struct OrchestrationGuard {
    _global_permit: OwnedSemaphorePermit,
    _parent_permit: OwnedSemaphorePermit,
}

impl std::fmt::Debug for OrchestrationGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrchestrationGuard").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // OrchestrationConfig tests

    #[test]
    fn test_config_default() {
        let config = OrchestrationConfig::default();
        assert_eq!(config.global_max_concurrent, DEFAULT_GLOBAL_MAX_CONCURRENT);
        assert_eq!(
            config.per_parent_max_concurrent,
            DEFAULT_PER_PARENT_MAX_CONCURRENT
        );
        assert_eq!(config.max_depth, DEFAULT_MAX_DEPTH);
        assert_eq!(
            config.total_timeout,
            Duration::from_secs(DEFAULT_TOTAL_TIMEOUT_SECS)
        );
    }

    #[test]
    fn test_config_validate_ok() {
        let config = OrchestrationConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_global() {
        let config = OrchestrationConfig::default().with_global_max_concurrent(0);
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("global_max_concurrent"));
    }

    #[test]
    fn test_config_validate_zero_per_parent() {
        let config = OrchestrationConfig::default().with_per_parent_max_concurrent(0);
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("per_parent_max_concurrent"));
    }

    #[test]
    fn test_config_validate_zero_depth() {
        let config = OrchestrationConfig::default().with_max_depth(0);
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max_depth"));
    }

    #[test]
    fn test_config_validate_zero_timeout() {
        let config = OrchestrationConfig::default().with_total_timeout(Duration::ZERO);
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("total_timeout"));
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = OrchestrationConfig::default()
            .with_global_max_concurrent(20)
            .with_per_parent_max_concurrent(8)
            .with_max_depth(5)
            .with_total_timeout(Duration::from_secs(600));

        assert_eq!(config.global_max_concurrent, 20);
        assert_eq!(config.per_parent_max_concurrent, 8);
        assert_eq!(config.max_depth, 5);
        assert_eq!(config.total_timeout, Duration::from_secs(600));
    }

    // OrchestrationState tests

    #[test]
    fn test_state_new() {
        let config = OrchestrationConfig::default();
        let state = OrchestrationState::new(config.clone());

        assert_eq!(state.config().max_depth, config.max_depth);
        assert!(state.remaining_time().is_some());
    }

    #[test]
    fn test_state_check_depth_ok() {
        let config = OrchestrationConfig::default().with_max_depth(3);
        let state = OrchestrationState::new(config);

        assert!(state.check_depth(0).is_ok()); // Root
        assert!(state.check_depth(1).is_ok()); // First child
        assert!(state.check_depth(2).is_ok()); // Grandchild
    }

    #[test]
    fn test_state_check_depth_exceeded() {
        let config = OrchestrationConfig::default().with_max_depth(3);
        let state = OrchestrationState::new(config);

        let result = state.check_depth(3);
        assert!(result.is_err());

        match result.unwrap_err() {
            OrchestrationError::MaxDepthExceeded { depth, limit } => {
                assert_eq!(depth, 3);
                assert_eq!(limit, 3);
            }
            _ => panic!("Expected MaxDepthExceeded"),
        }
    }

    #[test]
    fn test_state_remaining_time() {
        let config = OrchestrationConfig::default().with_total_timeout(Duration::from_secs(10));
        let state = OrchestrationState::new(config);

        let remaining = state.remaining_time().expect("should have remaining time");
        assert!(remaining <= Duration::from_secs(10));
        assert!(remaining > Duration::from_secs(9)); // Should be close to 10s
    }

    #[test]
    fn test_state_check_timeout_ok() {
        let config = OrchestrationConfig::default().with_total_timeout(Duration::from_secs(10));
        let state = OrchestrationState::new(config);

        assert!(state.check_timeout().is_ok());
    }

    #[tokio::test]
    async fn test_state_acquire_permits_basic() {
        let config = OrchestrationConfig::default()
            .with_global_max_concurrent(2)
            .with_per_parent_max_concurrent(2);
        let state = OrchestrationState::new(config);

        let parent_id = ExecutionId::new();
        let guard = state.acquire_permits(&parent_id).await;
        assert!(guard.is_ok());

        // Guard is dropped here, releasing permits
    }

    #[tokio::test]
    async fn test_state_acquire_permits_respects_global_limit() {
        let config = OrchestrationConfig::default()
            .with_global_max_concurrent(2)
            .with_per_parent_max_concurrent(10)
            .with_total_timeout(Duration::from_millis(100));
        let state = Arc::new(OrchestrationState::new(config));

        let parent_id = ExecutionId::new();

        // Acquire 2 permits (at limit)
        let _guard1 = state.acquire_permits(&parent_id).await.unwrap();
        let _guard2 = state.acquire_permits(&parent_id).await.unwrap();

        // Third should timeout
        let result = state.acquire_permits(&parent_id).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_state_acquire_permits_respects_per_parent_limit() {
        let config = OrchestrationConfig::default()
            .with_global_max_concurrent(10)
            .with_per_parent_max_concurrent(2)
            .with_total_timeout(Duration::from_millis(100));
        let state = Arc::new(OrchestrationState::new(config));

        let parent_id = ExecutionId::new();

        // Acquire 2 permits (at per-parent limit)
        let _guard1 = state.acquire_permits(&parent_id).await.unwrap();
        let _guard2 = state.acquire_permits(&parent_id).await.unwrap();

        // Third from same parent should timeout
        let result = state.acquire_permits(&parent_id).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_state_different_parents_independent() {
        let config = OrchestrationConfig::default()
            .with_global_max_concurrent(10)
            .with_per_parent_max_concurrent(2);
        let state = Arc::new(OrchestrationState::new(config));

        let parent1 = ExecutionId::new();
        let parent2 = ExecutionId::new();

        // Each parent can have 2 concurrent subagents
        let _guard1a = state.acquire_permits(&parent1).await.unwrap();
        let _guard1b = state.acquire_permits(&parent1).await.unwrap();
        let _guard2a = state.acquire_permits(&parent2).await.unwrap();
        let _guard2b = state.acquire_permits(&parent2).await.unwrap();

        // All 4 acquired successfully (different parents)
    }

    #[tokio::test]
    async fn test_state_permits_released_on_drop() {
        let config = OrchestrationConfig::default()
            .with_global_max_concurrent(1)
            .with_per_parent_max_concurrent(1);
        let state = Arc::new(OrchestrationState::new(config));

        let parent_id = ExecutionId::new();

        // Acquire and drop
        {
            let _guard = state.acquire_permits(&parent_id).await.unwrap();
        } // guard dropped here

        // Should be able to acquire again
        let guard2 = state.acquire_permits(&parent_id).await;
        assert!(guard2.is_ok());
    }

    // Error display tests

    #[test]
    fn test_error_display_invalid_config() {
        let err = OrchestrationError::InvalidConfig("test error".into());
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_error_display_max_depth() {
        let err = OrchestrationError::MaxDepthExceeded { depth: 5, limit: 3 };
        let msg = err.to_string();
        assert!(msg.contains("5"));
        assert!(msg.contains("3"));
    }

    #[test]
    fn test_error_display_timeout() {
        let err = OrchestrationError::TotalTimeoutExceeded {
            elapsed: Duration::from_secs(65),
            limit: Duration::from_secs(60),
        };
        let msg = err.to_string();
        assert!(msg.contains("65"));
        assert!(msg.contains("60"));
    }

    // Debug trait tests

    #[test]
    fn test_config_debug() {
        let config = OrchestrationConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("OrchestrationConfig"));
        assert!(debug.contains("global_max_concurrent"));
    }

    #[test]
    fn test_state_debug() {
        let state = OrchestrationState::new(OrchestrationConfig::default());
        let debug = format!("{:?}", state);
        assert!(debug.contains("OrchestrationState"));
        assert!(debug.contains("config"));
    }

    #[tokio::test]
    async fn test_guard_debug() {
        let state = OrchestrationState::new(OrchestrationConfig::default());
        let guard = state.acquire_permits(&ExecutionId::new()).await.unwrap();
        let debug = format!("{:?}", guard);
        assert!(debug.contains("OrchestrationGuard"));
    }
}
