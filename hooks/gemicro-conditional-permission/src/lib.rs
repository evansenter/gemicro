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
///
/// ## False Positives
///
/// The fallback JSON string matching can trigger on:
/// - **JSON field names**: Pattern "delete" matches `{"delete_flag": false}`
/// - **Numeric patterns**: Pattern "1" matches any JSON with numbers or timestamps
/// - **String metadata**: Patterns appearing in escaped strings or nested keys
///
/// Prefer field-based matching over full JSON fallback for critical operations.
///
/// ## False Negatives (Bypass Methods)
///
/// Pattern matching can be evaded by:
/// - **Whitespace obfuscation**: `r m -rf` (spaces) won't match pattern "rm"
/// - **Character substitution**: Unicode lookalikes like `rм` (Cyrillic 'м')
/// - **Encoding**: Base64 or hex-encoded dangerous commands in nested JSON
///
/// For critical security, combine with other hooks, semantic analysis, or
/// tool-level confirmation mechanisms.
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
    /// # Panics
    ///
    /// Panics if `dangerous_patterns` is empty or contains only whitespace.
    /// Use at least one non-empty pattern to create a meaningful permission policy.
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
        let validated: Vec<String> = dangerous_patterns
            .into_iter()
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect();

        if validated.is_empty() {
            panic!("ConditionalPermission requires at least one non-empty pattern");
        }

        Self {
            dangerous_patterns: validated,
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
    #[must_use]
    pub fn add_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.patterns.push(pattern.into());
        self
    }

    /// Add a field name to check for patterns.
    #[must_use]
    pub fn check_field(mut self, field: impl Into<String>) -> Self {
        self.fields.push(field.into());
        self
    }

    /// Build the ConditionalPermission hook.
    ///
    /// If no fields specified, uses default fields:
    /// "command", "query", "code", "script"
    ///
    /// # Panics
    ///
    /// Panics if no patterns were added or all patterns are empty/whitespace.
    pub fn build(self) -> ConditionalPermission {
        let validated: Vec<String> = self
            .patterns
            .into_iter()
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect();

        if validated.is_empty() {
            panic!("ConditionalPermission requires at least one non-empty pattern. Use add_pattern() to add patterns.");
        }

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
            dangerous_patterns: validated,
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

    #[test]
    #[should_panic(expected = "ConditionalPermission requires at least one non-empty pattern")]
    fn test_panics_on_empty_patterns() {
        ConditionalPermission::new(vec![]);
    }

    #[test]
    #[should_panic(expected = "ConditionalPermission requires at least one non-empty pattern")]
    fn test_panics_on_whitespace_only_patterns() {
        ConditionalPermission::new(vec!["  ".to_string(), "\t".to_string()]);
    }

    // Tests documenting known limitations (bypasses and false positives)

    #[tokio::test]
    async fn test_whitespace_obfuscation_bypass() {
        // KNOWN BYPASS: Whitespace can split patterns
        let hook = ConditionalPermission::new(vec!["rm".to_string()]);
        let input = json!({"command": "r m -rf /"});
        let decision = hook.pre_tool_use("bash", &input).await.unwrap();

        // Currently ALLOWS (no "rm" substring due to space)
        assert_eq!(
            decision,
            HookDecision::Allow,
            "KNOWN BYPASS: Whitespace obfuscation not detected"
        );
    }

    #[tokio::test]
    async fn test_false_positive_json_field_name() {
        // KNOWN FALSE POSITIVE: Fallback JSON matching triggers on field names
        let hook = ConditionalPermission::new(vec!["delete".to_string()]);
        // Field name contains "delete" but value is benign
        let input = json!({"delete_flag": false, "action": "read"});
        let decision = hook.pre_tool_use("test", &input).await.unwrap();

        // Currently triggers permission request (field name matched in full JSON)
        assert!(
            matches!(decision, HookDecision::RequestPermission { .. }),
            "KNOWN FALSE POSITIVE: Field names trigger fallback matching"
        );
    }

    #[tokio::test]
    async fn test_false_positive_numeric_pattern() {
        // KNOWN FALSE POSITIVE: Numeric patterns match any number
        let hook = ConditionalPermission::new(vec!["1".to_string()]);
        let input = json!({"timestamp": 1234567890});
        let decision = hook.pre_tool_use("test", &input).await.unwrap();

        // Currently triggers (fallback JSON contains "1")
        assert!(
            matches!(decision, HookDecision::RequestPermission { .. }),
            "KNOWN FALSE POSITIVE: Numeric patterns match all numbers"
        );
    }

    #[tokio::test]
    async fn test_empty_pattern_after_trimming_bypasses_builder() {
        // Patterns are trimmed and filtered - empty after trim should be removed
        let hook = ConditionalPermission::builder()
            .add_pattern("  ")
            .add_pattern("valid")
            .build();

        let input = json!({"command": "safe"});
        let decision = hook.pre_tool_use("test", &input).await.unwrap();
        assert_eq!(decision, HookDecision::Allow);
    }
}
