//! Interactive confirmation handler for CLI tool execution.
//!
//! Provides [`InteractiveConfirmation`] which prompts users via the terminal
//! before executing potentially dangerous tool operations.

use async_trait::async_trait;
use dialoguer::Confirm;
use gemicro_core::tool::ConfirmationHandler;
use serde_json::Value;

/// Interactive CLI confirmation handler using dialoguer.
///
/// When a tool requires confirmation (e.g., bash commands, file writes),
/// this handler prompts the user in the terminal before proceeding.
///
/// # Example
///
/// ```no_run
/// use gemicro_cli::confirmation::InteractiveConfirmation;
/// use gemicro_core::AgentContext;
/// use std::sync::Arc;
///
/// let handler = Arc::new(InteractiveConfirmation::default());
/// let context = AgentContext::new(llm_client)
///     .with_confirmation_handler(handler);
/// ```
#[derive(Debug, Clone, Default)]
pub struct InteractiveConfirmation {
    /// Whether to show tool arguments in the confirmation prompt.
    pub verbose: bool,
}

impl InteractiveConfirmation {
    /// Create a new interactive confirmation handler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a verbose handler that shows tool arguments in prompts.
    pub fn verbose() -> Self {
        Self { verbose: true }
    }
}

#[async_trait]
impl ConfirmationHandler for InteractiveConfirmation {
    async fn confirm(&self, tool_name: &str, message: &str, args: &Value) -> bool {
        // Build the prompt
        let prompt = if self.verbose {
            format!(
                "[{}] {}\nArgs: {}",
                tool_name,
                message,
                serde_json::to_string_pretty(args).unwrap_or_else(|_| args.to_string())
            )
        } else {
            format!("[{}] {}", tool_name, message)
        };

        // Use dialoguer for terminal confirmation
        match Confirm::new()
            .with_prompt(prompt)
            .default(false) // Safe default: deny if user just presses Enter
            .interact()
        {
            Ok(confirmed) => confirmed,
            Err(e) => {
                // Log the error so users know why confirmation failed
                // (e.g., terminal not available, stdin closed, signal interrupt)
                log::warn!(
                    "Confirmation prompt failed for tool '{}': {}. Denying by default.",
                    tool_name,
                    e
                );
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_not_verbose() {
        let handler = InteractiveConfirmation::default();
        assert!(!handler.verbose);
    }

    #[test]
    fn test_verbose_constructor() {
        let handler = InteractiveConfirmation::verbose();
        assert!(handler.verbose);
    }

    #[test]
    fn test_debug_implementation() {
        let handler = InteractiveConfirmation::default();
        let debug = format!("{:?}", handler);
        assert!(debug.contains("InteractiveConfirmation"));
    }
}
