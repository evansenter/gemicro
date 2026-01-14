//! Agent registry construction and configuration.
//!
//! This module provides functions for creating pre-configured agent registries
//! with all bundled agents. It can be used by both the CLI binary and by users
//! building custom binaries that extend the agent set.
//!
//! # Example
//!
//! ```no_run
//! use gemicro_cli::{default_registry, RegistryOptions};
//!
//! // Create registry with model override
//! let options = RegistryOptions::default().with_model("gemini-3.0-flash-preview");
//! let mut registry = default_registry(Some(options));
//!
//! // Registry contains all bundled agents
//! assert!(registry.contains("deep_research"));
//! assert!(registry.contains("prompt_agent"));
//! assert!(registry.contains("developer"));
//! ```

use gemicro_critique_agent::CritiqueAgent;
use gemicro_deep_research_agent::{DeepResearchAgent, DeepResearchAgentConfig};
use gemicro_developer_agent::{DeveloperAgent, DeveloperAgentConfig};
use gemicro_echo_agent::EchoAgent;
use gemicro_loader::load_markdown_agents_from_dir;
use gemicro_prompt_agent::{PromptAgent, PromptAgentConfig};
use gemicro_react_agent::{ReactAgent, ReactAgentConfig};
use gemicro_runner::AgentRegistry;
use std::path::Path;

/// Options for configuring agent registration.
///
/// Used with [`default_registry`] to customize how agents are created.
#[derive(Debug, Clone, Default)]
pub struct RegistryOptions {
    /// Model to use for all agents (overrides agent defaults).
    ///
    /// When set, this model will be passed to all agent configs via their
    /// `with_model()` builder method.
    pub model: Option<String>,
}

impl RegistryOptions {
    /// Create options with a model override.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_cli::RegistryOptions;
    ///
    /// let options = RegistryOptions::default()
    ///     .with_model("gemini-3.0-flash-preview");
    /// ```
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

/// Register all built-in agents with the given options.
///
/// This registers the core bundled agents:
/// - `deep_research` - Multi-step research agent with parallel sub-queries
/// - `prompt_agent` - General-purpose prompt-based agent
/// - `developer` - Tool-using development agent
/// - `react` - ReAct (Reasoning + Acting) agent
/// - `echo` - Simple echo agent for testing
/// - `critique` - Self-validation agent for Task tool
///
/// # Arguments
///
/// * `registry` - The registry to populate
/// * `options` - Configuration options (model override, etc.)
pub fn register_builtin_agents(registry: &mut AgentRegistry, options: &RegistryOptions) {
    let model = options.model.clone();

    // Deep Research Agent
    {
        let model = model.clone();
        registry.register("deep_research", move || {
            let mut config = DeepResearchAgentConfig::default();
            if let Some(ref m) = model {
                config = config.with_model(m);
            }
            Box::new(DeepResearchAgent::new(config).expect("default config should not fail"))
        });
    }

    // Prompt Agent
    {
        let model = model.clone();
        registry.register("prompt_agent", move || {
            let mut config = PromptAgentConfig::default();
            if let Some(ref m) = model {
                config = config.with_model(m);
            }
            Box::new(PromptAgent::new(config).expect("default config should not fail"))
        });
    }

    // Developer Agent
    {
        let model = model.clone();
        registry.register("developer", move || {
            let mut config = DeveloperAgentConfig::default();
            if let Some(ref m) = model {
                config = config.with_model(m);
            }
            Box::new(DeveloperAgent::new(config).expect("default config should not fail"))
        });
    }

    // React Agent
    {
        let model = model.clone();
        registry.register("react", move || {
            let mut config = ReactAgentConfig::default();
            if let Some(ref m) = model {
                config = config.with_model(m);
            }
            Box::new(ReactAgent::new(config).expect("default config should not fail"))
        });
    }

    // Echo Agent (no config needed - for testing)
    registry.register("echo", || Box::new(EchoAgent));

    // Critique Agent (for self-validation via Task tool)
    registry.register("critique", || Box::new(CritiqueAgent::default_agent()));
}

/// Register markdown-defined agents from a directory.
///
/// Loads agent definitions from `.md` files in the specified directory.
/// Each markdown file with YAML frontmatter becomes a registered agent
/// backed by `PromptAgent`.
///
/// Parse errors are logged as warnings but don't cause failure (graceful degradation).
///
/// # Arguments
///
/// * `registry` - The registry to populate
/// * `dir` - Directory containing markdown agent definitions
pub fn register_markdown_agents(registry: &mut AgentRegistry, dir: &Path) {
    let (agents, errors) = load_markdown_agents_from_dir(dir);

    // Log parse errors but continue (graceful degradation)
    for (path, err) in errors {
        log::warn!("Failed to parse markdown agent {:?}: {}", path, err);
    }

    for agent in agents {
        let def = agent.definition.clone();
        let name = agent.name.clone();

        // Log tool requests for debugging
        if let gemicro_core::ToolSet::Specific(ref tools) = def.tools {
            for tool in tools {
                log::debug!("Markdown agent '{}' requests tool: {}", name, tool);
            }
        }

        registry.register(&name, move || {
            Box::new(
                PromptAgent::with_definition(&def)
                    .expect("pre-validated markdown agent definition"),
            )
        });

        log::info!(
            "Registered markdown agent: {} - {}",
            agent.name,
            agent.definition.description
        );
    }
}

/// Create an agent registry with all bundled agents.
///
/// This is the main entry point for creating a pre-configured registry.
/// The registry will contain all built-in agents (deep_research, prompt_agent,
/// developer, react, echo, critique) configured with the provided options.
///
/// # Arguments
///
/// * `options` - Optional configuration (model override, etc.). Pass `None` for defaults.
///
/// # Example
///
/// ```no_run
/// use gemicro_cli::{default_registry, RegistryOptions};
///
/// // With defaults
/// let registry = default_registry(None);
///
/// // With model override
/// let registry = default_registry(Some(
///     RegistryOptions::default().with_model("gemini-3.0-flash-preview")
/// ));
///
/// // Create an agent
/// let agent = registry.get("deep_research").expect("agent exists");
/// ```
pub fn default_registry(options: Option<RegistryOptions>) -> AgentRegistry {
    let options = options.unwrap_or_default();
    let mut registry = AgentRegistry::new();
    register_builtin_agents(&mut registry, &options);
    registry
}

/// Create an agent registry with bundled agents plus markdown agents.
///
/// Extends [`default_registry`] by also loading markdown agent definitions
/// from the specified directory.
///
/// # Arguments
///
/// * `options` - Optional configuration (model override, etc.)
/// * `markdown_dir` - Directory containing markdown agent definitions
///
/// # Example
///
/// ```no_run
/// use gemicro_cli::{default_registry_with_markdown, RegistryOptions};
/// use std::path::Path;
///
/// let registry = default_registry_with_markdown(
///     Some(RegistryOptions::default().with_model("gemini-3.0-flash-preview")),
///     Path::new("agents/runtime-agents"),
/// );
/// ```
#[allow(dead_code)] // Part of public library API, not used by binary
pub fn default_registry_with_markdown(
    options: Option<RegistryOptions>,
    markdown_dir: &Path,
) -> AgentRegistry {
    let mut registry = default_registry(options);
    register_markdown_agents(&mut registry, markdown_dir);
    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_options_default() {
        let options = RegistryOptions::default();
        assert!(options.model.is_none());
    }

    #[test]
    fn test_registry_options_with_model() {
        let options = RegistryOptions::default().with_model("test-model");
        assert_eq!(options.model, Some("test-model".to_string()));
    }

    #[test]
    fn test_default_registry_contains_builtin_agents() {
        let registry = default_registry(None);

        assert!(registry.contains("deep_research"));
        assert!(registry.contains("prompt_agent"));
        assert!(registry.contains("developer"));
        assert!(registry.contains("react"));
        assert!(registry.contains("echo"));
        assert!(registry.contains("critique"));
    }

    #[test]
    fn test_default_registry_agent_count() {
        let registry = default_registry(None);
        // 6 built-in agents
        assert_eq!(registry.len(), 6);
    }

    #[test]
    fn test_register_builtin_agents_with_model() {
        let options = RegistryOptions::default().with_model("test-model");
        let registry = default_registry(Some(options));

        // Agents should be registered (we can't easily verify model was applied
        // without inspecting agent internals, but registration should succeed)
        assert!(registry.contains("deep_research"));
        assert!(registry.contains("prompt_agent"));
    }
}
