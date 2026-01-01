//! Ephemeral agent definitions for inline subagent creation.
//!
//! This module provides types for defining agents inline without pre-registration,
//! enabling the Task tool to spawn custom prompt-based agents on demand.
//!
//! ## Design Philosophy
//!
//! Following [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) principles:
//!
//! - **Flexible agent definition**: Define agents via prompts without code changes
//! - **Named or inline**: Reference registered agents or define inline
//! - **Tool isolation**: Each ephemeral agent gets its own tool set
//!
//! ## Example
//!
//! ```
//! use gemicro_core::agent::{AgentSpec, PromptAgentDef};
//! use gemicro_core::ToolSet;
//!
//! // Reference a pre-registered agent
//! let named = AgentSpec::Named("simple_qa".into());
//!
//! // Define an ephemeral agent inline using the builder pattern
//! let ephemeral = AgentSpec::Prompt(
//!     PromptAgentDef::new("Python security reviewer")
//!         .with_system_prompt("You are an expert Python security auditor...")
//!         .with_tools(ToolSet::Specific(vec!["file_read".into(), "grep".into()]))
//!         .with_model("gemini-1.5-flash")
//! );
//!
//! // Or use the convenience builder
//! let ephemeral = AgentSpec::prompt("Code reviewer")
//!     .with_system_prompt("Review code for bugs and security issues...")
//!     .with_tools(ToolSet::Except(vec!["bash".into()]));
//! ```

use crate::ToolSet;

/// Definition for an ephemeral prompt-based agent.
///
/// Ephemeral agents are defined inline and don't require pre-registration.
/// They use a system prompt to define behavior and can optionally override
/// the model and tool set.
///
/// # Example
///
/// ```
/// use gemicro_core::agent::PromptAgentDef;
/// use gemicro_core::ToolSet;
///
/// let agent = PromptAgentDef::new("Math tutor")
///     .with_system_prompt("You are a helpful math tutor...")
///     .with_tools(ToolSet::Specific(vec!["calculator".into()]));
/// // model defaults to None (use default model)
/// ```
/// # Construction
///
/// Due to `#[non_exhaustive]`, struct literal syntax is not available.
/// Use [`Self::new()`] with builder methods:
///
/// ```
/// use gemicro_core::agent::PromptAgentDef;
///
/// let def = PromptAgentDef::new("Code reviewer")
///     .with_system_prompt("You review code for quality issues.");
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PromptAgentDef {
    /// Human-readable description of what this agent does.
    ///
    /// Used for logging and observability.
    pub description: String,

    /// System prompt that defines the agent's behavior.
    ///
    /// This is passed to the LLM as the system message.
    pub system_prompt: String,

    /// Which tools this agent can access.
    ///
    /// Defaults to [`ToolSet::Inherit`] which inherits from the parent.
    pub tools: ToolSet,

    /// Optional model override.
    ///
    /// If `None`, uses the default model from configuration.
    /// Example: `Some("gemini-1.5-flash".into())` for cheaper operations.
    pub model: Option<String>,
}

impl Default for PromptAgentDef {
    fn default() -> Self {
        Self {
            description: String::new(),
            system_prompt: String::new(),
            tools: ToolSet::Inherit,
            model: None,
        }
    }
}

impl PromptAgentDef {
    /// Create a new prompt agent definition with a description.
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            ..Default::default()
        }
    }

    /// Set the system prompt.
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Set the tool set.
    #[must_use]
    pub fn with_tools(mut self, tools: ToolSet) -> Self {
        self.tools = tools;
        self
    }

    /// Set the model override.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Validate the definition.
    ///
    /// Returns an error if required fields are missing or whitespace-only.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.description.trim().is_empty() {
            return Err("PromptAgentDef requires a description");
        }
        if self.system_prompt.trim().is_empty() {
            return Err("PromptAgentDef requires a system_prompt");
        }
        Ok(())
    }
}

/// Specification for which agent to use.
///
/// Can reference a pre-registered agent by name or define an ephemeral
/// prompt-based agent inline.
///
/// # Example
///
/// ```
/// use gemicro_core::agent::AgentSpec;
///
/// // Reference a registered agent
/// let named = AgentSpec::Named("deep_research".into());
///
/// // Define inline using builder
/// let inline = AgentSpec::prompt("Security reviewer")
///     .with_system_prompt("You audit code for security vulnerabilities...");
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum AgentSpec {
    /// Reference a pre-registered agent by name.
    Named(String),

    /// Define an ephemeral prompt-based agent inline.
    Prompt(PromptAgentDef),
}

impl AgentSpec {
    /// Create a named agent reference.
    pub fn named(name: impl Into<String>) -> Self {
        Self::Named(name.into())
    }

    /// Start building an ephemeral prompt agent.
    ///
    /// Returns a [`PromptAgentDef`] builder that can be wrapped with
    /// [`AgentSpec::Prompt`].
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_core::agent::AgentSpec;
    /// use gemicro_core::ToolSet;
    ///
    /// let spec = AgentSpec::prompt("Code reviewer")
    ///     .with_system_prompt("Review code for quality...")
    ///     .with_tools(ToolSet::None);
    /// ```
    pub fn prompt(description: impl Into<String>) -> PromptAgentDef {
        PromptAgentDef::new(description)
    }

    /// Get the name of this agent.
    ///
    /// For named agents, returns the registered name.
    /// For prompt agents, returns the description.
    pub fn name(&self) -> &str {
        match self {
            Self::Named(name) => name,
            Self::Prompt(def) => &def.description,
        }
    }

    /// Check if this is a named (pre-registered) agent.
    pub fn is_named(&self) -> bool {
        matches!(self, Self::Named(_))
    }

    /// Check if this is an ephemeral prompt agent.
    pub fn is_prompt(&self) -> bool {
        matches!(self, Self::Prompt(_))
    }
}

impl From<PromptAgentDef> for AgentSpec {
    fn from(def: PromptAgentDef) -> Self {
        Self::Prompt(def)
    }
}

impl From<String> for AgentSpec {
    fn from(name: String) -> Self {
        Self::Named(name)
    }
}

impl From<&str> for AgentSpec {
    fn from(name: &str) -> Self {
        Self::Named(name.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_agent_def_new() {
        let def = PromptAgentDef::new("Test agent");
        assert_eq!(def.description, "Test agent");
        assert!(def.system_prompt.is_empty());
        assert!(matches!(def.tools, ToolSet::Inherit));
        assert!(def.model.is_none());
    }

    #[test]
    fn test_prompt_agent_def_builder() {
        let def = PromptAgentDef::new("Code reviewer")
            .with_system_prompt("Review code for quality")
            .with_tools(ToolSet::Specific(vec!["file_read".into()]))
            .with_model("gemini-1.5-flash");

        assert_eq!(def.description, "Code reviewer");
        assert_eq!(def.system_prompt, "Review code for quality");
        assert!(matches!(def.tools, ToolSet::Specific(_)));
        assert_eq!(def.model, Some("gemini-1.5-flash".to_string()));
    }

    #[test]
    fn test_prompt_agent_def_validate() {
        // Missing description
        let def = PromptAgentDef::default();
        assert!(def.validate().is_err());

        // Missing system prompt
        let def = PromptAgentDef::new("Test");
        assert!(def.validate().is_err());

        // Valid
        let def = PromptAgentDef::new("Test").with_system_prompt("Do things");
        assert!(def.validate().is_ok());
    }

    #[test]
    fn test_prompt_agent_def_validate_whitespace() {
        // Whitespace-only description should fail
        let def = PromptAgentDef::new("   ").with_system_prompt("Valid prompt");
        assert!(def.validate().is_err());

        // Whitespace-only system prompt should fail
        let def = PromptAgentDef::new("Valid").with_system_prompt("   \t\n   ");
        assert!(def.validate().is_err());
    }

    #[test]
    fn test_agent_spec_named() {
        let spec = AgentSpec::Named("deep_research".into());
        assert!(spec.is_named());
        assert!(!spec.is_prompt());
        assert_eq!(spec.name(), "deep_research");
    }

    #[test]
    fn test_agent_spec_prompt() {
        let spec = AgentSpec::Prompt(PromptAgentDef::new("Custom agent"));
        assert!(!spec.is_named());
        assert!(spec.is_prompt());
        assert_eq!(spec.name(), "Custom agent");
    }

    #[test]
    fn test_agent_spec_builder() {
        let def = AgentSpec::prompt("Reviewer")
            .with_system_prompt("Review code")
            .with_tools(ToolSet::None);

        assert_eq!(def.description, "Reviewer");
        assert_eq!(def.system_prompt, "Review code");
        assert!(matches!(def.tools, ToolSet::None));
    }

    #[test]
    fn test_agent_spec_from_string() {
        let spec: AgentSpec = "simple_qa".into();
        assert!(spec.is_named());
        assert_eq!(spec.name(), "simple_qa");
    }

    #[test]
    fn test_agent_spec_from_prompt_def() {
        let def = PromptAgentDef::new("Test").with_system_prompt("Do things");
        let spec: AgentSpec = def.into();
        assert!(spec.is_prompt());
    }

    #[test]
    fn test_clone() {
        let def = PromptAgentDef::new("Test").with_system_prompt("Prompt");
        let cloned = def.clone();
        assert_eq!(cloned.description, def.description);

        let spec = AgentSpec::Named("test".into());
        let cloned = spec.clone();
        assert_eq!(cloned.name(), spec.name());
    }

    #[test]
    fn test_debug() {
        let def = PromptAgentDef::new("Test");
        let debug = format!("{:?}", def);
        assert!(debug.contains("PromptAgentDef"));

        let spec = AgentSpec::Named("test".into());
        let debug = format!("{:?}", spec);
        assert!(debug.contains("Named"));
    }
}
