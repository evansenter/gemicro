//! TOML-serializable configuration types.
//!
//! These types mirror the agent config structs but use serde-friendly types
//! (e.g., `u64` for seconds instead of `Duration`).

use gemicro_deep_research::{ResearchConfig, ResearchPrompts};
use gemicro_tool_agent::ToolAgentConfig;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Root configuration structure for the gemicro.toml file.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
#[non_exhaustive]
pub struct GemicroConfig {
    /// Deep Research agent configuration
    pub deep_research: Option<DeepResearchToml>,

    /// Tool Agent configuration
    pub tool_agent: Option<ToolAgentToml>,
}

impl GemicroConfig {
    /// Merge another config into this one (other takes precedence).
    pub fn merge(&mut self, other: GemicroConfig) {
        if let Some(dr) = other.deep_research {
            match &mut self.deep_research {
                Some(existing) => existing.merge(dr),
                None => self.deep_research = Some(dr),
            }
        }
        if let Some(ta) = other.tool_agent {
            match &mut self.tool_agent {
                Some(existing) => existing.merge(ta),
                None => self.tool_agent = Some(ta),
            }
        }
    }

    /// Check if this config is empty (all sections are None or default).
    pub fn is_empty(&self) -> bool {
        self.deep_research.is_none() && self.tool_agent.is_none()
    }
}

/// TOML-serializable Deep Research configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
#[non_exhaustive]
pub struct DeepResearchToml {
    /// Minimum number of sub-queries to generate
    pub min_sub_queries: Option<usize>,

    /// Maximum number of sub-queries to generate
    pub max_sub_queries: Option<usize>,

    /// Maximum concurrent sub-queries
    pub max_concurrent_sub_queries: Option<usize>,

    /// Whether to continue if some sub-queries fail
    pub continue_on_partial_failure: Option<bool>,

    /// Total timeout in seconds
    pub timeout_secs: Option<u64>,

    /// Enable Google Search grounding
    pub use_google_search: Option<bool>,

    /// Custom prompts
    pub prompts: Option<PromptsToml>,
}

impl DeepResearchToml {
    /// Merge another config into this one (other takes precedence for Some values).
    pub fn merge(&mut self, other: DeepResearchToml) {
        if other.min_sub_queries.is_some() {
            self.min_sub_queries = other.min_sub_queries;
        }
        if other.max_sub_queries.is_some() {
            self.max_sub_queries = other.max_sub_queries;
        }
        if other.max_concurrent_sub_queries.is_some() {
            self.max_concurrent_sub_queries = other.max_concurrent_sub_queries;
        }
        if other.continue_on_partial_failure.is_some() {
            self.continue_on_partial_failure = other.continue_on_partial_failure;
        }
        if other.timeout_secs.is_some() {
            self.timeout_secs = other.timeout_secs;
        }
        if other.use_google_search.is_some() {
            self.use_google_search = other.use_google_search;
        }
        if let Some(other_prompts) = other.prompts {
            match &mut self.prompts {
                Some(existing) => existing.merge(other_prompts),
                None => self.prompts = Some(other_prompts),
            }
        }
    }

    /// Convert to ResearchConfig, applying overrides to defaults.
    pub fn to_research_config(&self) -> ResearchConfig {
        let mut config = ResearchConfig::default();

        if let Some(v) = self.min_sub_queries {
            config = config.with_min_sub_queries(v);
        }
        if let Some(v) = self.max_sub_queries {
            config = config.with_max_sub_queries(v);
        }
        if let Some(v) = self.max_concurrent_sub_queries {
            config = config.with_max_concurrent_sub_queries(v);
        }
        if let Some(v) = self.continue_on_partial_failure {
            config = config.with_continue_on_partial_failure(v);
        }
        if let Some(v) = self.timeout_secs {
            config = config.with_total_timeout(Duration::from_secs(v));
        }
        if let Some(v) = self.use_google_search {
            config = config.with_google_search(v);
        }
        if let Some(prompts) = &self.prompts {
            config = config.with_prompts(prompts.to_research_prompts());
        }

        config
    }
}

/// TOML-serializable prompts configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
#[non_exhaustive]
pub struct PromptsToml {
    /// System instruction for decomposition phase
    pub decomposition_system: Option<String>,

    /// User template for decomposition phase
    pub decomposition_template: Option<String>,

    /// System instruction for sub-query execution
    pub sub_query_system: Option<String>,

    /// System instruction for synthesis phase
    pub synthesis_system: Option<String>,

    /// User template for synthesis phase
    pub synthesis_template: Option<String>,
}

impl PromptsToml {
    /// Merge another prompts config into this one.
    pub fn merge(&mut self, other: PromptsToml) {
        if other.decomposition_system.is_some() {
            self.decomposition_system = other.decomposition_system;
        }
        if other.decomposition_template.is_some() {
            self.decomposition_template = other.decomposition_template;
        }
        if other.sub_query_system.is_some() {
            self.sub_query_system = other.sub_query_system;
        }
        if other.synthesis_system.is_some() {
            self.synthesis_system = other.synthesis_system;
        }
        if other.synthesis_template.is_some() {
            self.synthesis_template = other.synthesis_template;
        }
    }

    /// Convert to ResearchPrompts, applying overrides to defaults.
    pub fn to_research_prompts(&self) -> ResearchPrompts {
        let mut prompts = ResearchPrompts::default();

        if let Some(v) = &self.decomposition_system {
            prompts.decomposition_system = v.clone();
        }
        if let Some(v) = &self.decomposition_template {
            prompts.decomposition_template = v.clone();
        }
        if let Some(v) = &self.sub_query_system {
            prompts.sub_query_system = v.clone();
        }
        if let Some(v) = &self.synthesis_system {
            prompts.synthesis_system = v.clone();
        }
        if let Some(v) = &self.synthesis_template {
            prompts.synthesis_template = v.clone();
        }

        prompts
    }
}

/// TOML-serializable Tool Agent configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
#[non_exhaustive]
pub struct ToolAgentToml {
    /// Total timeout in seconds
    pub timeout_secs: Option<u64>,

    /// System prompt for the agent
    pub system_prompt: Option<String>,
}

impl ToolAgentToml {
    /// Merge another config into this one.
    pub fn merge(&mut self, other: ToolAgentToml) {
        if other.timeout_secs.is_some() {
            self.timeout_secs = other.timeout_secs;
        }
        if other.system_prompt.is_some() {
            self.system_prompt = other.system_prompt;
        }
    }

    /// Convert to ToolAgentConfig, applying overrides to defaults.
    pub fn to_tool_agent_config(&self) -> ToolAgentConfig {
        let mut config = ToolAgentConfig::default();

        if let Some(v) = self.timeout_secs {
            config.timeout = Duration::from_secs(v);
        }
        if let Some(v) = &self.system_prompt {
            config.system_prompt = v.clone();
        }

        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_empty_config() {
        let config: GemicroConfig = toml::from_str("").unwrap();
        assert!(config.is_empty());
    }

    #[test]
    fn test_parse_deep_research_config() {
        let toml = r#"
            [deep_research]
            min_sub_queries = 2
            max_sub_queries = 8
            timeout_secs = 120
        "#;

        let config: GemicroConfig = toml::from_str(toml).unwrap();
        let dr = config.deep_research.unwrap();

        assert_eq!(dr.min_sub_queries, Some(2));
        assert_eq!(dr.max_sub_queries, Some(8));
        assert_eq!(dr.timeout_secs, Some(120));
    }

    #[test]
    fn test_parse_prompts() {
        let toml = r#"
            [deep_research.prompts]
            decomposition_system = "Custom decomposition system"
            sub_query_system = "Custom sub-query system"
        "#;

        let config: GemicroConfig = toml::from_str(toml).unwrap();
        let prompts = config.deep_research.unwrap().prompts.unwrap();

        assert_eq!(
            prompts.decomposition_system,
            Some("Custom decomposition system".to_string())
        );
        assert_eq!(
            prompts.sub_query_system,
            Some("Custom sub-query system".to_string())
        );
    }

    #[test]
    fn test_parse_tool_agent_config() {
        let toml = r#"
            [tool_agent]
            timeout_secs = 90
            system_prompt = "You are a specialized assistant."
        "#;

        let config: GemicroConfig = toml::from_str(toml).unwrap();
        let ta = config.tool_agent.unwrap();

        assert_eq!(ta.timeout_secs, Some(90));
        assert_eq!(
            ta.system_prompt,
            Some("You are a specialized assistant.".to_string())
        );
    }

    #[test]
    fn test_merge_configs() {
        let mut base = GemicroConfig {
            deep_research: Some(DeepResearchToml {
                min_sub_queries: Some(2),
                max_sub_queries: Some(5),
                ..Default::default()
            }),
            tool_agent: None,
        };

        let overlay = GemicroConfig {
            deep_research: Some(DeepResearchToml {
                max_sub_queries: Some(10),
                timeout_secs: Some(120),
                ..Default::default()
            }),
            tool_agent: Some(ToolAgentToml {
                timeout_secs: Some(60),
                ..Default::default()
            }),
        };

        base.merge(overlay);

        let dr = base.deep_research.unwrap();
        assert_eq!(dr.min_sub_queries, Some(2)); // From base
        assert_eq!(dr.max_sub_queries, Some(10)); // Overridden by overlay
        assert_eq!(dr.timeout_secs, Some(120)); // From overlay

        let ta = base.tool_agent.unwrap();
        assert_eq!(ta.timeout_secs, Some(60)); // From overlay
    }

    #[test]
    fn test_to_research_config() {
        let toml_config = DeepResearchToml {
            min_sub_queries: Some(2),
            max_sub_queries: Some(8),
            timeout_secs: Some(120),
            use_google_search: Some(true),
            ..Default::default()
        };

        let config = toml_config.to_research_config();

        assert_eq!(config.min_sub_queries, 2);
        assert_eq!(config.max_sub_queries, 8);
        assert_eq!(config.total_timeout, Duration::from_secs(120));
        assert!(config.use_google_search);
    }

    #[test]
    fn test_to_tool_agent_config() {
        let toml_config = ToolAgentToml {
            timeout_secs: Some(90),
            system_prompt: Some("Custom prompt".to_string()),
        };

        let config = toml_config.to_tool_agent_config();

        assert_eq!(config.timeout, Duration::from_secs(90));
        assert_eq!(config.system_prompt, "Custom prompt");
    }

    #[test]
    fn test_serialize_config() {
        let config = GemicroConfig {
            deep_research: Some(DeepResearchToml {
                min_sub_queries: Some(3),
                max_sub_queries: Some(7),
                ..Default::default()
            }),
            tool_agent: None,
        };

        let toml_str = toml::to_string_pretty(&config).unwrap();
        assert!(toml_str.contains("min_sub_queries = 3"));
        assert!(toml_str.contains("max_sub_queries = 7"));
    }
}
