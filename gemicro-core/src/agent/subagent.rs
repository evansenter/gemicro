//! Subagent configuration for resource isolation.
//!
//! This module provides [`SubagentConfig`] for controlling what resources
//! subagents can access when spawned by the Task tool or other orchestrators.
//!
//! ## Design Philosophy
//!
//! Following [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) principles:
//!
//! - **Sensible defaults**: Default config inherits everything (backward compatible)
//! - **Explicit isolation**: Users can explicitly restrict tools, hooks, and nesting
//! - **Timeout enforcement**: Prevents runaway subagent executions
//!
//! ## Example
//!
//! ```
//! use gemicro_core::agent::SubagentConfig;
//! use gemicro_core::ToolSet;
//! use std::time::Duration;
//!
//! // Default config: inherit everything
//! let default_config = SubagentConfig::default();
//! assert!(default_config.allow_nested);
//!
//! // Restricted config: no tools, no nesting
//! let restricted = SubagentConfig::default()
//!     .with_tools(ToolSet::None)
//!     .with_allow_nested(false);
//!
//! // Specific tools only
//! let tool_limited = SubagentConfig::default()
//!     .with_tools(ToolSet::Specific(vec!["file_read".into(), "grep".into()]));
//!
//! // Custom timeout
//! let quick = SubagentConfig::default()
//!     .with_timeout(Duration::from_secs(30));
//! ```

use crate::ToolSet;
use std::time::Duration;

/// Default timeout for subagent execution (60 seconds).
pub const DEFAULT_SUBAGENT_TIMEOUT_SECS: u64 = 60;

/// Configuration for controlling subagent resource access.
///
/// Used by the Task tool and other orchestrators to specify what resources
/// a subagent can access. Enables isolation and security controls.
///
/// # Defaults
///
/// The default configuration is permissive for backward compatibility:
/// - `tools`: [`ToolSet::Inherit`] - inherit parent's tools
/// - `inherit_hooks`: `true` - inherit parent's hooks
/// - `allow_nested`: `true` - subagent can spawn its own subagents
/// - `timeout`: 60 seconds
///
/// # Example
///
/// ```
/// use gemicro_core::agent::SubagentConfig;
/// use gemicro_core::ToolSet;
/// use std::time::Duration;
///
/// // Secure subagent: limited tools, no nesting
/// let config = SubagentConfig::default()
///     .with_tools(ToolSet::Except(vec!["bash".into(), "file_write".into()]))
///     .with_allow_nested(false)
///     .with_timeout(Duration::from_secs(30));
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SubagentConfig {
    /// Which tools the subagent can access.
    ///
    /// Defaults to [`ToolSet::Inherit`] which uses the parent's tool set.
    /// Use [`ToolSet::None`] for pure LLM agents with no tool access.
    pub tools: ToolSet,

    /// Whether the subagent inherits the parent's hooks.
    ///
    /// When `true` (default), hooks like audit logging and security checks
    /// apply to the subagent. Set to `false` to run the subagent without
    /// parent hooks (e.g., for trusted internal operations).
    pub inherit_hooks: bool,

    /// Whether the subagent can spawn its own subagents.
    ///
    /// When `true` (default), the subagent can use the Task tool to spawn
    /// further subagents. Set to `false` to prevent deep nesting and
    /// potential infinite recursion.
    pub allow_nested: bool,

    /// Maximum execution time for the subagent.
    ///
    /// The subagent will be cancelled if it exceeds this timeout.
    /// Defaults to 60 seconds.
    pub timeout: Duration,
}

impl Default for SubagentConfig {
    fn default() -> Self {
        Self {
            tools: ToolSet::Inherit,
            inherit_hooks: true,
            allow_nested: true,
            timeout: Duration::from_secs(DEFAULT_SUBAGENT_TIMEOUT_SECS),
        }
    }
}

impl SubagentConfig {
    /// Create a new subagent config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set which tools the subagent can access.
    #[must_use]
    pub fn with_tools(mut self, tools: ToolSet) -> Self {
        self.tools = tools;
        self
    }

    /// Set whether the subagent inherits parent hooks.
    #[must_use]
    pub fn with_inherit_hooks(mut self, inherit: bool) -> Self {
        self.inherit_hooks = inherit;
        self
    }

    /// Set whether the subagent can spawn its own subagents.
    #[must_use]
    pub fn with_allow_nested(mut self, allow: bool) -> Self {
        self.allow_nested = allow;
        self
    }

    /// Set the execution timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Check if nesting is allowed based on config.
    ///
    /// Returns `false` if `allow_nested` is `false`.
    pub fn can_spawn_subagents(&self) -> bool {
        self.allow_nested
    }

    /// Create a restrictive config suitable for untrusted subagents.
    ///
    /// - No tools
    /// - No nesting
    /// - 30 second timeout
    /// - Inherits hooks (for security monitoring)
    pub fn restrictive() -> Self {
        Self {
            tools: ToolSet::None,
            inherit_hooks: true,
            allow_nested: false,
            timeout: Duration::from_secs(30),
        }
    }

    /// Create a permissive config for trusted subagents.
    ///
    /// - All tools
    /// - Nesting allowed
    /// - 5 minute timeout
    /// - Inherits hooks
    pub fn permissive() -> Self {
        Self {
            tools: ToolSet::All,
            inherit_hooks: true,
            allow_nested: true,
            timeout: Duration::from_secs(300),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SubagentConfig::default();
        assert!(matches!(config.tools, ToolSet::Inherit));
        assert!(config.inherit_hooks);
        assert!(config.allow_nested);
        assert_eq!(config.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_builder_pattern() {
        let config = SubagentConfig::new()
            .with_tools(ToolSet::None)
            .with_inherit_hooks(false)
            .with_allow_nested(false)
            .with_timeout(Duration::from_secs(10));

        assert!(matches!(config.tools, ToolSet::None));
        assert!(!config.inherit_hooks);
        assert!(!config.allow_nested);
        assert_eq!(config.timeout, Duration::from_secs(10));
    }

    #[test]
    fn test_can_spawn_subagents() {
        let allowed = SubagentConfig::default().with_allow_nested(true);
        assert!(allowed.can_spawn_subagents());

        let disallowed = SubagentConfig::default().with_allow_nested(false);
        assert!(!disallowed.can_spawn_subagents());
    }

    #[test]
    fn test_restrictive_preset() {
        let config = SubagentConfig::restrictive();
        assert!(matches!(config.tools, ToolSet::None));
        assert!(config.inherit_hooks); // Still monitors
        assert!(!config.allow_nested);
        assert_eq!(config.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_permissive_preset() {
        let config = SubagentConfig::permissive();
        assert!(matches!(config.tools, ToolSet::All));
        assert!(config.inherit_hooks);
        assert!(config.allow_nested);
        assert_eq!(config.timeout, Duration::from_secs(300));
    }

    #[test]
    fn test_with_specific_tools() {
        let config = SubagentConfig::default()
            .with_tools(ToolSet::Specific(vec!["file_read".into(), "grep".into()]));

        if let ToolSet::Specific(tools) = config.tools {
            assert_eq!(tools.len(), 2);
            assert!(tools.contains(&"file_read".to_string()));
            assert!(tools.contains(&"grep".to_string()));
        } else {
            panic!("Expected ToolSet::Specific");
        }
    }

    #[test]
    fn test_with_except_tools() {
        let config = SubagentConfig::default().with_tools(ToolSet::Except(vec!["bash".into()]));

        if let ToolSet::Except(tools) = config.tools {
            assert_eq!(tools.len(), 1);
            assert!(tools.contains(&"bash".to_string()));
        } else {
            panic!("Expected ToolSet::Except");
        }
    }

    #[test]
    fn test_clone() {
        let config = SubagentConfig::default().with_allow_nested(false);
        let cloned = config.clone();

        assert!(!cloned.allow_nested);
        assert_eq!(config.timeout, cloned.timeout);
    }

    #[test]
    fn test_debug() {
        let config = SubagentConfig::default();
        let debug = format!("{:?}", config);

        assert!(debug.contains("SubagentConfig"));
        assert!(debug.contains("Inherit")); // From ToolSet::Inherit
    }
}
