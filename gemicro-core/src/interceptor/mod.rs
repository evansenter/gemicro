//! Generic interceptor pattern for unified lifecycle hooks.
//!
//! This module provides a generic [`Interceptor<In, Out>`] trait that can intercept
//! any async operation: tool calls, user messages, LLM requests/responses, and
//! external events.
//!
//! # Design Philosophy
//!
//! The Interceptor pattern unifies interception *semantics*, not the things being
//! intercepted. User messages, tool calls, LLM requests, and events are different
//! concepts with different roles—but they all benefit from the same allow/transform/
//! deny/observe pattern.
//!
//! # Type Aliases
//!
//! Common interceptor specializations:
//! - [`ToolInterceptor`] - Intercepts tool calls (replaces the old `ToolHook` trait)
//! - [`MessageInterceptor`] - Intercepts user messages
//! - [`EventInterceptor`] - Intercepts external events (for cross-agent coordination)
//!
//! # Example
//!
//! ```
//! use gemicro_core::interceptor::{
//!     Interceptor, InterceptDecision, InterceptError, InterceptorChain, ToolCall,
//! };
//! use gemicro_core::tool::ToolResult;
//! use async_trait::async_trait;
//!
//! #[derive(Debug)]
//! struct AuditLog;
//!
//! #[async_trait]
//! impl Interceptor<ToolCall, ToolResult> for AuditLog {
//!     async fn intercept(&self, input: &ToolCall) -> Result<InterceptDecision<ToolCall>, InterceptError> {
//!         log::info!("Tool invoked: {} with input: {}", input.name, input.arguments);
//!         Ok(InterceptDecision::Allow)
//!     }
//!
//!     async fn observe(&self, input: &ToolCall, output: &ToolResult) -> Result<(), InterceptError> {
//!         log::info!("Tool completed: {}", input.name);
//!         Ok(())
//!     }
//! }
//!
//! // Create a chain of interceptors
//! let chain = InterceptorChain::new()
//!     .with(AuditLog);
//! ```

mod types;

pub use types::{ExternalEvent, ToolCall, UserContent, UserMessage};

use async_trait::async_trait;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

// ============================================================================
// Intercept Error
// ============================================================================

/// Errors that can occur during interceptor execution.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum InterceptError {
    /// Interceptor logic failed
    #[error("Interceptor execution failed: {0}")]
    ExecutionFailed(String),

    /// Interceptor attempted to modify input in an invalid way
    #[error("Invalid input modification: {0}")]
    InvalidModification(String),

    /// Generic error from interceptor implementation
    #[error("Interceptor error: {0}")]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

// ============================================================================
// Intercept Decision
// ============================================================================

/// Decision from an interceptor's `intercept` method.
///
/// Interceptors run in order and the first deny, confirm, or error stops
/// the chain. Transform decisions are chained (each interceptor sees the
/// transformed input).
#[derive(Debug, Clone, PartialEq)]
pub enum InterceptDecision<T> {
    /// Allow execution with original input
    Allow,

    /// Allow execution with modified input
    ///
    /// The modified input will be used for:
    /// - Actual execution
    /// - Subsequent interceptors in the chain
    Transform(T),

    /// Request confirmation before proceeding
    ///
    /// Aligns with the tool confirmation system. The adapter will call
    /// the confirmation handler if one is configured. If approved, execution
    /// continues; if denied, returns an error.
    Confirm {
        /// Message shown to user (e.g., "Allow write to /etc?")
        message: String,
    },

    /// Block execution entirely
    Deny {
        /// Reason for denial (shown to user/LLM)
        reason: String,
    },
}

// ============================================================================
// Interceptor Trait
// ============================================================================

/// Generic interceptor for any async operation.
///
/// Interceptors can:
/// - Validate/transform input before execution
/// - Block execution entirely
/// - Observe output after execution
///
/// # Type Parameters
///
/// - `In`: The input type being intercepted (e.g., `ToolCall`, `UserMessage`)
/// - `Out`: The output type after execution (e.g., `ToolResult`, `()`)
///
/// # Execution Model
///
/// - [`intercept`](Self::intercept): Called before execution. Can allow, modify, or deny.
/// - [`observe`](Self::observe): Called after execution. Observability only.
///
/// # Example
///
/// ```
/// use gemicro_core::interceptor::{Interceptor, InterceptDecision, InterceptError, ToolCall};
/// use gemicro_core::tool::ToolResult;
/// use async_trait::async_trait;
///
/// #[derive(Debug)]
/// struct SecurityHook;
///
/// #[async_trait]
/// impl Interceptor<ToolCall, ToolResult> for SecurityHook {
///     async fn intercept(&self, input: &ToolCall) -> Result<InterceptDecision<ToolCall>, InterceptError> {
///         if input.name == "bash" && input.arguments["command"] == "rm -rf /" {
///             return Ok(InterceptDecision::Deny {
///                 reason: "Dangerous command blocked".into(),
///             });
///         }
///         Ok(InterceptDecision::Allow)
///     }
/// }
/// ```
#[async_trait]
pub trait Interceptor<In, Out>: Send + Sync + fmt::Debug
where
    In: Send + Sync,
    Out: Send + Sync,
{
    /// Called before execution. Can allow, modify, or deny.
    ///
    /// # Arguments
    ///
    /// * `input` - The input about to be processed (possibly modified by previous interceptors)
    ///
    /// # Returns
    ///
    /// * `Ok(InterceptDecision)` - Decision to allow/modify/deny
    /// * `Err(InterceptError)` - Interceptor logic failed (treated as deny)
    async fn intercept(&self, input: &In) -> Result<InterceptDecision<In>, InterceptError>;

    /// Called after execution. Observability only.
    ///
    /// For logging, metrics collection, auditing, etc. Cannot modify output
    /// or block execution.
    ///
    /// # Arguments
    ///
    /// * `input` - The input that was used (after pre-interceptor modifications)
    /// * `output` - The result from execution
    ///
    /// # Errors
    ///
    /// Errors are logged but do not affect the result returned to the caller.
    async fn observe(&self, input: &In, output: &Out) -> Result<(), InterceptError> {
        let _ = (input, output);
        Ok(())
    }
}

// ============================================================================
// Interceptor Chain
// ============================================================================

/// Chain of interceptors executed in order.
///
/// Provides a unified registry for any interceptor type. The same chain
/// implementation works for tools, messages, LLM calls, and events.
///
/// # Execution Semantics
///
/// For [`intercept`](Self::intercept):
/// - Interceptors run in registration order
/// - First `Deny` or `Confirm` stops the chain
/// - `Transform` decisions are chained (each sees the transformed input)
/// - If all return `Allow`, returns `Allow`
///
/// For [`observe`](Self::observe):
/// - All interceptors run, errors are logged but don't propagate
/// - Observation failures don't affect the result
///
/// # Example
///
/// ```
/// use gemicro_core::interceptor::{
///     Interceptor, InterceptDecision, InterceptError, InterceptorChain, ToolCall,
/// };
/// use gemicro_core::tool::ToolResult;
/// use async_trait::async_trait;
///
/// #[derive(Debug)]
/// struct AllowHook;
///
/// #[async_trait]
/// impl Interceptor<ToolCall, ToolResult> for AllowHook {
///     async fn intercept(&self, _input: &ToolCall) -> Result<InterceptDecision<ToolCall>, InterceptError> {
///         Ok(InterceptDecision::Allow)
///     }
/// }
///
/// let chain = InterceptorChain::new()
///     .with(AllowHook);
///
/// assert_eq!(chain.len(), 1);
/// ```
#[derive(Clone)]
pub struct InterceptorChain<In, Out>
where
    In: Send + Sync,
    Out: Send + Sync,
{
    interceptors: Vec<Arc<dyn Interceptor<In, Out>>>,
}

impl<In, Out> fmt::Debug for InterceptorChain<In, Out>
where
    In: Send + Sync,
    Out: Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InterceptorChain")
            .field("count", &self.interceptors.len())
            .finish()
    }
}

impl<In, Out> InterceptorChain<In, Out>
where
    In: Clone + Send + Sync,
    Out: Send + Sync,
{
    /// Create a new empty interceptor chain.
    pub fn new() -> Self {
        Self {
            interceptors: Vec::new(),
        }
    }

    /// Add an interceptor to the chain.
    ///
    /// Interceptors are executed in the order they are registered.
    pub fn with(mut self, interceptor: impl Interceptor<In, Out> + 'static) -> Self {
        self.interceptors.push(Arc::new(interceptor));
        self
    }

    /// Run all interceptors, returning final decision.
    ///
    /// Execution stops at the first Deny or Confirm. Transform decisions
    /// are chained (each interceptor sees the transformed input).
    ///
    /// # Returns
    ///
    /// - `Ok(Allow)` if all interceptors allow with original input
    /// - `Ok(Transform(input))` if any interceptor transformed the input
    /// - `Ok(Confirm { message })` if any interceptor requests confirmation
    /// - `Ok(Deny { reason })` if any interceptor denies
    /// - `Err(e)` if any interceptor fails
    pub async fn intercept(&self, input: &In) -> Result<InterceptDecision<In>, InterceptError> {
        let mut current_input = input.clone();
        let mut was_transformed = false;

        for interceptor in &self.interceptors {
            match interceptor.intercept(&current_input).await? {
                InterceptDecision::Allow => {
                    // Continue to next interceptor
                    continue;
                }
                InterceptDecision::Transform(new_input) => {
                    current_input = new_input;
                    was_transformed = true;
                    // Continue checking remaining interceptors with modified input
                    continue;
                }
                decision @ InterceptDecision::Confirm { .. } => {
                    // Confirm stops chain
                    return Ok(decision);
                }
                decision @ InterceptDecision::Deny { .. } => {
                    // Deny stops chain
                    return Ok(decision);
                }
            }
        }

        // All interceptors passed - check if input was transformed
        if was_transformed {
            Ok(InterceptDecision::Transform(current_input))
        } else {
            Ok(InterceptDecision::Allow)
        }
    }

    /// Run all observers (errors logged, not propagated).
    ///
    /// All interceptors run even if some fail. Failures are logged as warnings.
    pub async fn observe(&self, input: &In, output: &Out) {
        for interceptor in &self.interceptors {
            if let Err(e) = interceptor.observe(input, output).await {
                log::warn!("Interceptor observation failed: {}", e);
            }
        }
    }

    /// Check if the chain has any interceptors.
    pub fn is_empty(&self) -> bool {
        self.interceptors.is_empty()
    }

    /// Get the number of registered interceptors.
    pub fn len(&self) -> usize {
        self.interceptors.len()
    }
}

impl<In, Out> Default for InterceptorChain<In, Out>
where
    In: Clone + Send + Sync,
    Out: Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Type Aliases
// ============================================================================

use crate::tool::ToolResult;

/// Tool interceptor - intercepts tool calls.
///
/// Replaces the old `ToolHook` trait.
pub type ToolInterceptor = dyn Interceptor<ToolCall, ToolResult>;

/// Message interceptor - intercepts user messages.
pub type MessageInterceptor = dyn Interceptor<UserMessage, ()>;

/// Event interceptor - intercepts external events (for cross-agent coordination).
pub type EventInterceptor = dyn Interceptor<ExternalEvent, crate::AgentUpdate>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolResult;
    use serde_json::json;

    #[derive(Debug)]
    struct AllowInterceptor;

    #[async_trait]
    impl Interceptor<ToolCall, ToolResult> for AllowInterceptor {
        async fn intercept(
            &self,
            _input: &ToolCall,
        ) -> Result<InterceptDecision<ToolCall>, InterceptError> {
            Ok(InterceptDecision::Allow)
        }
    }

    #[derive(Debug)]
    struct DenyInterceptor {
        reason: String,
    }

    #[async_trait]
    impl Interceptor<ToolCall, ToolResult> for DenyInterceptor {
        async fn intercept(
            &self,
            _input: &ToolCall,
        ) -> Result<InterceptDecision<ToolCall>, InterceptError> {
            Ok(InterceptDecision::Deny {
                reason: self.reason.clone(),
            })
        }
    }

    #[derive(Debug)]
    struct TransformInterceptor;

    #[async_trait]
    impl Interceptor<ToolCall, ToolResult> for TransformInterceptor {
        async fn intercept(
            &self,
            input: &ToolCall,
        ) -> Result<InterceptDecision<ToolCall>, InterceptError> {
            let mut modified = input.clone();
            modified.arguments["modified"] = json!(true);
            Ok(InterceptDecision::Transform(modified))
        }
    }

    #[tokio::test]
    async fn test_empty_chain() {
        let chain: InterceptorChain<ToolCall, ToolResult> = InterceptorChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);

        let input = ToolCall::new("test", json!({}));
        let decision = chain.intercept(&input).await.unwrap();
        assert_eq!(decision, InterceptDecision::Allow);
    }

    #[tokio::test]
    async fn test_single_allow() {
        let chain = InterceptorChain::new().with(AllowInterceptor);

        let input = ToolCall::new("test", json!({}));
        let decision = chain.intercept(&input).await.unwrap();
        assert_eq!(decision, InterceptDecision::Allow);
    }

    #[tokio::test]
    async fn test_single_deny() {
        let chain = InterceptorChain::new().with(DenyInterceptor {
            reason: "Blocked".into(),
        });

        let input = ToolCall::new("test", json!({}));
        let decision = chain.intercept(&input).await.unwrap();
        assert_eq!(
            decision,
            InterceptDecision::Deny {
                reason: "Blocked".into()
            }
        );
    }

    #[tokio::test]
    async fn test_single_transform() {
        let chain = InterceptorChain::new().with(TransformInterceptor);

        let input = ToolCall::new("test", json!({"original": true}));
        let decision = chain.intercept(&input).await.unwrap();

        match decision {
            InterceptDecision::Transform(modified) => {
                assert_eq!(modified.arguments["modified"], json!(true));
                assert_eq!(modified.arguments["original"], json!(true));
            }
            _ => panic!("Expected Transform"),
        }
    }

    #[tokio::test]
    async fn test_deny_stops_chain() {
        let chain = InterceptorChain::new()
            .with(DenyInterceptor {
                reason: "First deny".into(),
            })
            .with(AllowInterceptor);

        let input = ToolCall::new("test", json!({}));
        let decision = chain.intercept(&input).await.unwrap();
        assert_eq!(
            decision,
            InterceptDecision::Deny {
                reason: "First deny".into()
            }
        );
    }

    #[tokio::test]
    async fn test_transform_then_deny() {
        let chain = InterceptorChain::new()
            .with(TransformInterceptor)
            .with(DenyInterceptor {
                reason: "Denied after transform".into(),
            });

        let input = ToolCall::new("test", json!({}));
        let decision = chain.intercept(&input).await.unwrap();
        assert_eq!(
            decision,
            InterceptDecision::Deny {
                reason: "Denied after transform".into()
            }
        );
    }

    #[tokio::test]
    async fn test_multiple_allows() {
        let chain = InterceptorChain::new()
            .with(AllowInterceptor)
            .with(AllowInterceptor);

        let input = ToolCall::new("test", json!({}));
        let decision = chain.intercept(&input).await.unwrap();
        assert_eq!(decision, InterceptDecision::Allow);
    }

    #[tokio::test]
    async fn test_observe_errors_logged() {
        #[derive(Debug)]
        struct FailingObserver;

        #[async_trait]
        impl Interceptor<ToolCall, ToolResult> for FailingObserver {
            async fn intercept(
                &self,
                _input: &ToolCall,
            ) -> Result<InterceptDecision<ToolCall>, InterceptError> {
                Ok(InterceptDecision::Allow)
            }

            async fn observe(
                &self,
                _input: &ToolCall,
                _output: &ToolResult,
            ) -> Result<(), InterceptError> {
                Err(InterceptError::ExecutionFailed("Observer failed".into()))
            }
        }

        let chain = InterceptorChain::new().with(FailingObserver);
        let input = ToolCall::new("test", json!({}));
        let output = ToolResult::text("test");

        // Should not panic even though observer fails
        chain.observe(&input, &output).await;
    }

    #[tokio::test]
    async fn test_chain_clone() {
        let chain = InterceptorChain::new().with(AllowInterceptor);
        let cloned = chain.clone();

        assert_eq!(cloned.len(), 1);
        let input = ToolCall::new("test", json!({}));
        let decision = cloned.intercept(&input).await.unwrap();
        assert_eq!(decision, InterceptDecision::Allow);
    }

    #[test]
    fn test_chain_debug() {
        let chain: InterceptorChain<ToolCall, ToolResult> =
            InterceptorChain::new().with(AllowInterceptor);
        let debug_str = format!("{:?}", chain);
        assert!(debug_str.contains("InterceptorChain"));
        assert!(debug_str.contains("count"));
    }
}
