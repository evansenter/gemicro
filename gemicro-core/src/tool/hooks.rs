//! Hook system for intercepting tool execution.
//!
//! Hooks enable validation, logging, security controls, and custom logic
//! to be injected before and after tool execution without modifying the
//! tools themselves.
//!
//! # Design
//!
//! Following the confirmation handler pattern:
//! - Hooks are enforced in [`ToolCallableAdapter`](super::ToolCallableAdapter)
//! - [`HookRegistry`] chains multiple hooks together
//! - Pre-hooks can allow, modify, or deny execution
//! - Post-hooks are for observability only
//!
//! # Why Hooks Are Enforced in the Adapter
//!
//! Hooks live in [`ToolCallableAdapter`](super::ToolCallableAdapter) because
//! it's the **only interception point** when using rust-genai's
//! `create_with_auto_functions()`. The LLM calls `CallableFunction::call()`
//! directly, bypassing `Tool` and `ToolRegistry` abstractions.
//!
//! See [`adapter`](super::adapter) module docs for detailed architecture rationale.
//!
//! # Example
//!
//! ```
//! use gemicro_core::tool::{HookRegistry, ToolHook, HookDecision, HookError, ToolResult};
//! use async_trait::async_trait;
//! use serde_json::Value;
//!
//! #[derive(Debug)]
//! struct AuditHook;
//!
//! #[async_trait]
//! impl ToolHook for AuditHook {
//!     async fn pre_tool_use(
//!         &self,
//!         tool_name: &str,
//!         input: &Value,
//!     ) -> Result<HookDecision, HookError> {
//!         println!("Tool called: {}", tool_name);
//!         Ok(HookDecision::Allow)
//!     }
//!
//!     async fn post_tool_use(
//!         &self,
//!         tool_name: &str,
//!         input: &Value,
//!         output: &ToolResult,
//!     ) -> Result<(), HookError> {
//!         println!("Tool completed: {}", tool_name);
//!         Ok(())
//!     }
//! }
//!
//! let registry = HookRegistry::new()
//!     .with_hook(AuditHook);
//! ```

use super::ToolResult;
use async_trait::async_trait;
use serde_json::Value;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

// ============================================================================
// Hook Error
// ============================================================================

/// Errors that can occur during hook execution.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum HookError {
    /// Hook logic failed
    #[error("Hook execution failed: {0}")]
    ExecutionFailed(String),

    /// Hook attempted to modify input in an invalid way
    #[error("Invalid input modification: {0}")]
    InvalidModification(String),

    /// Generic error from hook implementation
    #[error("Hook error: {0}")]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

// ============================================================================
// Hook Decision
// ============================================================================

/// Decision from a pre-execution hook.
///
/// Hooks run in order and the first deny, permission request, or modification
/// stops the chain (except Allow which continues to the next hook).
#[derive(Debug, Clone, PartialEq)]
pub enum HookDecision {
    /// Allow execution with original input
    Allow,

    /// Allow execution but modify the input first
    ///
    /// The modified input will be used for both:
    /// - Tool execution
    /// - Subsequent hooks in the chain
    AllowWithModifiedInput(Value),

    /// Request user permission before proceeding
    ///
    /// Aligns with the tool confirmation system. The adapter will call
    /// the confirmation handler if one is configured. If approved, execution
    /// continues; if denied, returns `ToolError::ConfirmationDenied`.
    RequestPermission {
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
// Tool Hook Trait
// ============================================================================

/// Hook for intercepting tool execution.
///
/// Hooks are called before and after tool execution, allowing:
/// - **Pre-execution**: Validation, input modification, or denial
/// - **Post-execution**: Logging, metrics, auditing
///
/// # Execution Order
///
/// Multiple hooks run in registration order:
/// ```text
/// pre_hook_1 → pre_hook_2 → ... → EXECUTE → ... → post_hook_2 → post_hook_1
/// ```
///
/// For pre-hooks:
/// - First [`HookDecision::Deny`] stops the chain and prevents execution
/// - First [`HookDecision::AllowWithModifiedInput`] modifies input for subsequent hooks
/// - If all return [`HookDecision::Allow`], execution proceeds
///
/// # Example
///
/// ```
/// use gemicro_core::tool::{ToolHook, HookDecision, HookError, ToolResult};
/// use async_trait::async_trait;
/// use serde_json::Value;
///
/// #[derive(Debug)]
/// struct SecurityHook;
///
/// #[async_trait]
/// impl ToolHook for SecurityHook {
///     async fn pre_tool_use(
///         &self,
///         tool_name: &str,
///         input: &Value,
///     ) -> Result<HookDecision, HookError> {
///         // Block dangerous operations
///         if tool_name == "bash" && input["command"].as_str() == Some("rm -rf /") {
///             return Ok(HookDecision::Deny {
///                 reason: "Dangerous command blocked".into(),
///             });
///         }
///         Ok(HookDecision::Allow)
///     }
///
///     async fn post_tool_use(
///         &self,
///         _tool_name: &str,
///         _input: &Value,
///         _output: &ToolResult,
///     ) -> Result<(), HookError> {
///         // Logging, metrics, etc.
///         Ok(())
///     }
/// }
/// ```
#[async_trait]
pub trait ToolHook: Send + Sync + fmt::Debug {
    /// Called before tool execution.
    ///
    /// Return [`HookDecision`] to control whether execution proceeds:
    /// - [`HookDecision::Allow`]: Continue normally
    /// - [`HookDecision::AllowWithModifiedInput`]: Execute with modified input
    /// - [`HookDecision::Deny`]: Block execution entirely
    ///
    /// # Arguments
    ///
    /// * `tool_name` - The name of the tool about to execute
    /// * `input` - The input arguments (possibly modified by previous hooks)
    ///
    /// # Returns
    ///
    /// * `Ok(HookDecision)` - Decision to allow/modify/deny
    /// * `Err(HookError)` - Hook logic failed (treated as deny)
    async fn pre_tool_use(&self, tool_name: &str, input: &Value)
        -> Result<HookDecision, HookError>;

    /// Called after tool execution.
    ///
    /// For observability only - cannot modify output or block execution.
    /// Use for logging, metrics collection, auditing, etc.
    ///
    /// # Arguments
    ///
    /// * `tool_name` - The tool that was executed
    /// * `input` - The input that was used (after pre-hook modifications)
    /// * `output` - The result from execution
    ///
    /// # Errors
    ///
    /// Errors are logged but do not affect the tool result returned to the LLM.
    async fn post_tool_use(
        &self,
        tool_name: &str,
        input: &Value,
        output: &ToolResult,
    ) -> Result<(), HookError>;
}

// ============================================================================
// Hook Registry
// ============================================================================

/// Registry for managing multiple hooks.
///
/// Chains hooks together and provides a unified interface for execution.
/// Hooks run in registration order.
///
/// # Example
///
/// ```
/// use gemicro_core::tool::{HookRegistry, ToolHook, HookDecision, HookError};
/// use async_trait::async_trait;
/// use serde_json::Value;
///
/// # #[derive(Debug)]
/// # struct AuditHook;
/// # #[async_trait]
/// # impl ToolHook for AuditHook {
/// #     async fn pre_tool_use(&self, _: &str, _: &Value) -> Result<HookDecision, HookError> {
/// #         Ok(HookDecision::Allow)
/// #     }
/// #     async fn post_tool_use(&self, _: &str, _: &Value, _: &gemicro_core::tool::ToolResult) -> Result<(), HookError> {
/// #         Ok(())
/// #     }
/// # }
/// # #[derive(Debug)]
/// # struct SecurityHook;
/// # #[async_trait]
/// # impl ToolHook for SecurityHook {
/// #     async fn pre_tool_use(&self, _: &str, _: &Value) -> Result<HookDecision, HookError> {
/// #         Ok(HookDecision::Allow)
/// #     }
/// #     async fn post_tool_use(&self, _: &str, _: &Value, _: &gemicro_core::tool::ToolResult) -> Result<(), HookError> {
/// #         Ok(())
/// #     }
/// # }
/// let registry = HookRegistry::new()
///     .with_hook(AuditHook)
///     .with_hook(SecurityHook);
/// ```
#[derive(Debug, Clone)]
pub struct HookRegistry {
    hooks: Vec<Arc<dyn ToolHook>>,
}

impl HookRegistry {
    /// Create a new empty hook registry.
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// Add a hook to the registry.
    ///
    /// Hooks are executed in the order they are registered.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_core::tool::HookRegistry;
    /// # use gemicro_core::tool::{ToolHook, HookDecision, HookError};
    /// # use async_trait::async_trait;
    /// # use serde_json::Value;
    /// # #[derive(Debug)]
    /// # struct MyHook;
    /// # #[async_trait]
    /// # impl ToolHook for MyHook {
    /// #     async fn pre_tool_use(&self, _: &str, _: &Value) -> Result<HookDecision, HookError> {
    /// #         Ok(HookDecision::Allow)
    /// #     }
    /// #     async fn post_tool_use(&self, _: &str, _: &Value, _: &gemicro_core::tool::ToolResult) -> Result<(), HookError> {
    /// #         Ok(())
    /// #     }
    /// # }
    ///
    /// let registry = HookRegistry::new()
    ///     .with_hook(MyHook);
    /// ```
    pub fn with_hook(mut self, hook: impl ToolHook + 'static) -> Self {
        self.hooks.push(Arc::new(hook));
        self
    }

    /// Run all pre-execution hooks in order.
    ///
    /// Stops at the first deny, permission request, or modification.
    /// If all hooks return [`HookDecision::Allow`], returns `Allow`.
    ///
    /// # Arguments
    ///
    /// * `tool_name` - The tool about to execute
    /// * `input` - The original input arguments
    ///
    /// # Returns
    ///
    /// * `Ok(HookDecision)` - The final decision
    /// * `Err(HookError)` - A hook failed (treated as deny)
    pub async fn pre_tool_use(
        &self,
        tool_name: &str,
        input: &Value,
    ) -> Result<HookDecision, HookError> {
        let mut current_input = input.clone();

        for hook in &self.hooks {
            match hook.pre_tool_use(tool_name, &current_input).await? {
                HookDecision::Allow => {
                    // Continue to next hook
                    continue;
                }
                HookDecision::AllowWithModifiedInput(modified) => {
                    // Use modified input for remaining hooks
                    current_input = modified.clone();
                    // Continue checking remaining hooks with modified input
                    continue;
                }
                HookDecision::RequestPermission { message } => {
                    // Permission request stops chain - adapter will handle
                    return Ok(HookDecision::RequestPermission { message });
                }
                HookDecision::Deny { reason } => {
                    // Deny stops chain
                    return Ok(HookDecision::Deny { reason });
                }
            }
        }

        // All hooks passed - check if input was modified
        if &current_input != input {
            Ok(HookDecision::AllowWithModifiedInput(current_input))
        } else {
            Ok(HookDecision::Allow)
        }
    }

    /// Run all post-execution hooks in order.
    ///
    /// Errors are logged but do not affect the result returned to the LLM.
    ///
    /// # Arguments
    ///
    /// * `tool_name` - The tool that executed
    /// * `input` - The input that was used (after pre-hook modifications)
    /// * `output` - The execution result
    pub async fn post_tool_use(
        &self,
        tool_name: &str,
        input: &Value,
        output: &ToolResult,
    ) -> Result<(), HookError> {
        for hook in &self.hooks {
            if let Err(e) = hook.post_tool_use(tool_name, input, output).await {
                // Log but don't fail - post-hooks are observability only
                log::warn!("Post-hook failed for tool '{}': {}", tool_name, e);
            }
        }
        Ok(())
    }

    /// Check if the registry has any hooks.
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }

    /// Get the number of registered hooks.
    pub fn len(&self) -> usize {
        self.hooks.len()
    }
}

impl Default for HookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[derive(Debug)]
    struct AllowHook;

    #[async_trait]
    impl ToolHook for AllowHook {
        async fn pre_tool_use(&self, _: &str, _: &Value) -> Result<HookDecision, HookError> {
            Ok(HookDecision::Allow)
        }

        async fn post_tool_use(&self, _: &str, _: &Value, _: &ToolResult) -> Result<(), HookError> {
            Ok(())
        }
    }

    #[derive(Debug)]
    struct DenyHook {
        reason: String,
    }

    #[async_trait]
    impl ToolHook for DenyHook {
        async fn pre_tool_use(&self, _: &str, _: &Value) -> Result<HookDecision, HookError> {
            Ok(HookDecision::Deny {
                reason: self.reason.clone(),
            })
        }

        async fn post_tool_use(&self, _: &str, _: &Value, _: &ToolResult) -> Result<(), HookError> {
            Ok(())
        }
    }

    #[derive(Debug)]
    struct ModifyHook;

    #[async_trait]
    impl ToolHook for ModifyHook {
        async fn pre_tool_use(&self, _: &str, input: &Value) -> Result<HookDecision, HookError> {
            let mut modified = input.clone();
            modified["modified"] = json!(true);
            Ok(HookDecision::AllowWithModifiedInput(modified))
        }

        async fn post_tool_use(&self, _: &str, _: &Value, _: &ToolResult) -> Result<(), HookError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_empty_registry() {
        let registry = HookRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);

        let decision = registry.pre_tool_use("test", &json!({})).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);
    }

    #[tokio::test]
    async fn test_single_allow_hook() {
        let registry = HookRegistry::new().with_hook(AllowHook);

        let decision = registry.pre_tool_use("test", &json!({})).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);
    }

    #[tokio::test]
    async fn test_single_deny_hook() {
        let registry = HookRegistry::new().with_hook(DenyHook {
            reason: "Blocked".into(),
        });

        let decision = registry.pre_tool_use("test", &json!({})).await.unwrap();
        assert_eq!(
            decision,
            HookDecision::Deny {
                reason: "Blocked".into()
            }
        );
    }

    #[tokio::test]
    async fn test_single_modify_hook() {
        let registry = HookRegistry::new().with_hook(ModifyHook);

        let input = json!({"original": true});
        let decision = registry.pre_tool_use("test", &input).await.unwrap();

        match decision {
            HookDecision::AllowWithModifiedInput(modified) => {
                assert_eq!(modified["modified"], json!(true));
                assert_eq!(modified["original"], json!(true));
            }
            _ => panic!("Expected AllowWithModifiedInput"),
        }
    }

    #[tokio::test]
    async fn test_deny_stops_chain() {
        // Even if allow comes after deny, deny should win
        let registry = HookRegistry::new()
            .with_hook(DenyHook {
                reason: "First deny".into(),
            })
            .with_hook(AllowHook);

        let decision = registry.pre_tool_use("test", &json!({})).await.unwrap();
        assert_eq!(
            decision,
            HookDecision::Deny {
                reason: "First deny".into()
            }
        );
    }

    #[tokio::test]
    async fn test_modify_then_deny() {
        // Modify first, then deny - deny should win
        let registry = HookRegistry::new()
            .with_hook(ModifyHook)
            .with_hook(DenyHook {
                reason: "Denied after modify".into(),
            });

        let decision = registry.pre_tool_use("test", &json!({})).await.unwrap();
        assert_eq!(
            decision,
            HookDecision::Deny {
                reason: "Denied after modify".into()
            }
        );
    }

    #[tokio::test]
    async fn test_multiple_allows() {
        let registry = HookRegistry::new()
            .with_hook(AllowHook)
            .with_hook(AllowHook);

        let decision = registry.pre_tool_use("test", &json!({})).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);
    }

    #[tokio::test]
    async fn test_post_hook_errors_logged() {
        #[derive(Debug)]
        struct FailingPostHook;

        #[async_trait]
        impl ToolHook for FailingPostHook {
            async fn pre_tool_use(&self, _: &str, _: &Value) -> Result<HookDecision, HookError> {
                Ok(HookDecision::Allow)
            }

            async fn post_tool_use(
                &self,
                _: &str,
                _: &Value,
                _: &ToolResult,
            ) -> Result<(), HookError> {
                Err(HookError::ExecutionFailed("Post hook failed".into()))
            }
        }

        let registry = HookRegistry::new().with_hook(FailingPostHook);
        let result = ToolResult::text("test");

        // Should not error even though post-hook fails
        let res = registry.post_tool_use("test", &json!({}), &result).await;
        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_registry_clone() {
        let registry = HookRegistry::new().with_hook(AllowHook);
        let cloned = registry.clone();

        assert_eq!(cloned.len(), 1);
        let decision = cloned.pre_tool_use("test", &json!({})).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);
    }
}
