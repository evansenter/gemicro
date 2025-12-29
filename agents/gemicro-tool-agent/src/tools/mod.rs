//! Tool implementations for the ToolAgent.
//!
//! This module provides concrete tool implementations that can be used
//! with the ToolAgent or any agent that uses the Tool trait.
//!
//! # Available Tools
//!
//! - [`Calculator`]: Evaluates mathematical expressions
//! - [`CurrentDatetime`]: Gets the current date and time (UTC)
//!
//! # Example
//!
//! ```no_run
//! use gemicro_tool_agent::tools::{Calculator, CurrentDatetime};
//! use gemicro_core::tool::ToolRegistry;
//!
//! let mut registry = ToolRegistry::new();
//! registry.register(Calculator);
//! registry.register(CurrentDatetime);
//!
//! // Use with an agent that accepts a ToolRegistry
//! ```

mod calculator;
mod datetime;

pub use calculator::Calculator;
pub use datetime::CurrentDatetime;

/// Create a default tool registry with all built-in tools.
///
/// This is a convenience function for creating a registry with
/// all tools pre-registered.
pub fn default_registry() -> gemicro_core::tool::ToolRegistry {
    let mut registry = gemicro_core::tool::ToolRegistry::new();
    registry.register(Calculator);
    registry.register(CurrentDatetime);
    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use gemicro_core::tool::ToolSet;

    #[test]
    fn test_default_registry() {
        let registry = default_registry();
        assert_eq!(registry.len(), 2);
        assert!(registry.contains("calculator"));
        assert!(registry.contains("current_datetime"));
    }

    #[test]
    fn test_default_registry_filter() {
        let registry = default_registry();

        // All tools
        let all = registry.filter(&ToolSet::All);
        assert_eq!(all.len(), 2);

        // Specific tool
        let specific = registry.filter(&ToolSet::Specific(vec!["calculator".into()]));
        assert_eq!(specific.len(), 1);
        assert_eq!(specific[0].name(), "calculator");

        // Except tool
        let except = registry.filter(&ToolSet::Except(vec!["calculator".into()]));
        assert_eq!(except.len(), 1);
        assert_eq!(except[0].name(), "current_datetime");
    }
}
