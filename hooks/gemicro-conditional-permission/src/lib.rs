//! Conditional permission hook for gemicro tool execution.
//!
//! Dynamically requests user permission based on detecting dangerous
//! patterns in tool inputs, providing a flexible safety layer.
//!
//! # Example
//!
//! ```no_run
//! use gemicro_core::tool::HookRegistry;
//! use gemicro_conditional_permission::ConditionalPermission;
//!
//! // Request permission for shell commands containing "rm" or "delete"
//! let hook = ConditionalPermission::new(vec![
//!     "rm".to_string(),
//!     "delete".to_string(),
//!     "DROP TABLE".to_string(),
//! ]);
//!
//! let hooks = HookRegistry::new()
//!     .with_hook(hook);
//! ```

use async_trait::async_trait;
use gemicro_core::tool::{HookDecision, HookError, ToolHook, ToolResult};
use serde_json::Value;

/// Conditional permission hook that requests permission for specific operations.
///
/// Unlike tools that always or never require confirmation, this hook dynamically
/// requests permission based on detecting dangerous patterns in the input.
///
/// # Pattern Matching Strategy
///
/// The hook searches for patterns in two ways:
/// 1. **Field values**: Checks string values in specific JSON fields (configurable)
/// 2. **Full serialization**: Falls back to checking the entire JSON string
///
/// Matching is case-insensitive for better coverage.
///
/// # Limitations
///
/// This is a heuristic-based approach with trade-offs:
/// - **False positives**: May trigger on benign inputs containing keywords
/// - **False negatives**: Can miss dangerous operations phrased differently
/// - **Field-based**: Only checks string values, not structure
///
/// For critical security, combine with other hooks or tool-level confirmation.
///
/// # Example
///
/// ```
/// use gemicro_conditional_permission::ConditionalPermission;
///
/// let hook = ConditionalPermission::builder()
///     .add_pattern("rm")
///     .add_pattern("delete")
///     .check_field("command")  // Check these specific fields
///     .check_field("query")
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct ConditionalPermission {
    dangerous_patterns: Vec<String>,
    fields_to_check: Vec<String>,
}

impl ConditionalPermission {
    /// Create a new conditional permission hook with default configuration.
    ///
    /// By default, checks common fields: "command", "query", "code", "script".
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_conditional_permission::ConditionalPermission;
    ///
    /// let hook = ConditionalPermission::new(vec![
    ///     "rm".to_string(),
    ///     "delete".to_string(),
    /// ]);
    /// ```
    pub fn new(dangerous_patterns: Vec<String>) -> Self {
        Self {
            dangerous_patterns,
            fields_to_check: vec![
                "command".to_string(),
                "query".to_string(),
                "code".to_string(),
                "script".to_string(),
            ],
        }
    }

    /// Create a builder for more control over configuration.
    pub fn builder() -> ConditionalPermissionBuilder {
        ConditionalPermissionBuilder::default()
    }

    /// Check if input contains dangerous patterns.
    ///
    /// Returns the matched pattern if found.
    fn is_dangerous(&self, input: &Value) -> Option<String> {
        // Strategy 1: Check specific fields
        for field_name in &self.fields_to_check {
            if let Some(field_value) = input.get(field_name).and_then(|v| v.as_str()) {
                let field_lower = field_value.to_lowercase();
                for pattern in &self.dangerous_patterns {
                    if field_lower.contains(&pattern.to_lowercase()) {
                        return Some(pattern.clone());
                    }
                }
            }
        }

        // Strategy 2: Check full JSON string (fallback for nested/unknown structure)
        let input_str = input.to_string().to_lowercase();
        for pattern in &self.dangerous_patterns {
            if input_str.contains(&pattern.to_lowercase()) {
                return Some(pattern.clone());
            }
        }

        None
    }
}

/// Builder for configuring ConditionalPermission.
#[derive(Debug, Default)]
pub struct ConditionalPermissionBuilder {
    patterns: Vec<String>,
    fields: Vec<String>,
}

impl ConditionalPermissionBuilder {
    /// Add a dangerous pattern to check for.
    pub fn add_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.patterns.push(pattern.into());
        self
    }

    /// Add a field name to check for patterns.
    pub fn check_field(mut self, field: impl Into<String>) -> Self {
        self.fields.push(field.into());
        self
    }

    /// Build the ConditionalPermission hook.
    ///
    /// If no fields specified, uses default fields:
    /// "command", "query", "code", "script"
    pub fn build(self) -> ConditionalPermission {
        let fields = if self.fields.is_empty() {
            vec![
                "command".to_string(),
                "query".to_string(),
                "code".to_string(),
                "script".to_string(),
            ]
        } else {
            self.fields
        };

        ConditionalPermission {
            dangerous_patterns: self.patterns,
            fields_to_check: fields,
        }
    }
}

#[async_trait]
impl ToolHook for ConditionalPermission {
    async fn pre_tool_use(
        &self,
        tool_name: &str,
        input: &Value,
    ) -> Result<HookDecision, HookError> {
        if let Some(pattern) = self.is_dangerous(input) {
            return Ok(HookDecision::RequestPermission {
                message: format!(
                    "Tool '{}' wants to perform operation containing '{}'. Allow?",
                    tool_name, pattern
                ),
            });
        }
        Ok(HookDecision::Allow)
    }

    async fn post_tool_use(
        &self,
        _tool_name: &str,
        _input: &Value,
        _output: &ToolResult,
    ) -> Result<(), HookError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_allows_safe_command() {
        let hook = ConditionalPermission::new(vec!["rm".to_string(), "delete".to_string()]);
        let input = json!({"command": "ls -la"});
        let decision = hook.pre_tool_use("bash", &input).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);
    }

    #[tokio::test]
    async fn test_requests_permission_for_dangerous() {
        let hook = ConditionalPermission::new(vec!["rm".to_string(), "delete".to_string()]);
        let input = json!({"command": "rm -rf /"});
        let decision = hook.pre_tool_use("bash", &input).await.unwrap();
        match decision {
            HookDecision::RequestPermission { message } => {
                assert!(message.contains("rm"));
            }
            _ => panic!("Expected RequestPermission"),
        }
    }

    #[tokio::test]
    async fn test_case_insensitive() {
        let hook = ConditionalPermission::new(vec!["DELETE".to_string()]);
        let input = json!({"query": "delete from users"});
        let decision = hook.pre_tool_use("sql", &input).await.unwrap();
        match decision {
            HookDecision::RequestPermission { .. } => {}
            _ => panic!("Expected RequestPermission"),
        }
    }

    #[tokio::test]
    async fn test_builder() {
        let hook = ConditionalPermission::builder()
            .add_pattern("rm")
            .add_pattern("delete")
            .check_field("cmd")
            .build();

        let input = json!({"cmd": "rm file.txt"});
        let decision = hook.pre_tool_use("bash", &input).await.unwrap();
        match decision {
            HookDecision::RequestPermission { .. } => {}
            _ => panic!("Expected RequestPermission"),
        }
    }

    #[tokio::test]
    async fn test_checks_multiple_fields() {
        let hook = ConditionalPermission::builder()
            .add_pattern("danger")
            .check_field("field1")
            .check_field("field2")
            .build();

        let input = json!({"field1": "safe", "field2": "dangerous"});
        let decision = hook.pre_tool_use("test", &input).await.unwrap();
        match decision {
            HookDecision::RequestPermission { .. } => {}
            _ => panic!("Expected RequestPermission"),
        }
    }

    #[tokio::test]
    async fn test_fallback_to_full_json() {
        let hook = ConditionalPermission::new(vec!["hidden".to_string()]);
        let input = json!({"nested": {"deep": {"value": "hidden danger"}}});
        let decision = hook.pre_tool_use("test", &input).await.unwrap();
        match decision {
            HookDecision::RequestPermission { .. } => {}
            _ => panic!("Expected RequestPermission"),
        }
    }
}
