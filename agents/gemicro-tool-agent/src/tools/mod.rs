//! Tool implementations for the ToolAgent.
//!
//! This module provides concrete tool implementations that can be used
//! with the ToolAgent or any agent that uses the Tool trait.
//!
//! # Available Tools
//!
//! **Built-in tools (no external dependencies):**
//! - [`Calculator`]: Evaluates mathematical expressions
//! - [`CurrentDatetime`]: Gets the current date and time (UTC)
//! - [`FileRead`]: Read file contents (1MB size limit)
//! - [`WebFetch`]: Fetch content from URLs
//!
//! **Tools requiring external resources:**
//! - [`Task`]: Spawn subagents (requires AgentRegistry + LlmClient)
//! - [`WebSearch`]: Web search via Gemini grounding (requires LlmClient)
//!
//! # Example
//!
//! ```no_run
//! use gemicro_tool_agent::tools::{Calculator, CurrentDatetime, FileRead, WebFetch};
//! use gemicro_core::tool::ToolRegistry;
//!
//! // Create registry with default tools
//! let registry = gemicro_tool_agent::tools::default_registry();
//!
//! // Or manually register specific tools
//! let mut registry = ToolRegistry::new();
//! registry.register(Calculator);
//! registry.register(CurrentDatetime);
//! registry.register(FileRead);
//! registry.register(WebFetch::new());
//! ```

mod calculator;
mod datetime;

pub use calculator::Calculator;
pub use datetime::CurrentDatetime;

// Re-export tools from external crates
pub use gemicro_file_read::FileRead;
pub use gemicro_task::Task;
pub use gemicro_web_fetch::WebFetch;
pub use gemicro_web_search::WebSearch;

use gemicro_core::tool::ToolRegistry;
use gemicro_core::LlmClient;
use gemicro_runner::AgentRegistry;
use std::sync::Arc;

/// Create a default tool registry with stateless built-in tools.
///
/// This includes tools that don't require external resources:
/// - Calculator
/// - CurrentDatetime
/// - FileRead
/// - WebFetch
///
/// For tools that require external resources (Task, WebSearch),
/// use [`register_task_tool`] and [`register_web_search_tool`].
pub fn default_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Calculator);
    registry.register(CurrentDatetime);
    registry.register(FileRead);
    registry.register(WebFetch::new());
    registry
}

/// Register the Task tool in a registry.
///
/// The Task tool allows spawning subagents to handle subtasks.
/// Requires an AgentRegistry and LlmClient.
///
/// # Example
///
/// ```no_run
/// use gemicro_tool_agent::tools::{default_registry, register_task_tool};
/// use gemicro_runner::AgentRegistry;
/// use gemicro_core::{LlmClient, LlmConfig};
/// use std::sync::Arc;
///
/// let mut registry = default_registry();
/// let agent_registry = Arc::new(AgentRegistry::new());
/// let genai_client = rust_genai::Client::builder("key".to_string()).build();
/// let llm = Arc::new(LlmClient::new(genai_client, LlmConfig::default()));
///
/// register_task_tool(&mut registry, agent_registry, llm);
/// ```
pub fn register_task_tool(
    registry: &mut ToolRegistry,
    agent_registry: Arc<AgentRegistry>,
    llm: Arc<LlmClient>,
) {
    registry.register(Task::new(agent_registry, llm));
}

/// Register the WebSearch tool in a registry.
///
/// The WebSearch tool uses Gemini's Google Search grounding
/// to search the web for real-time information.
///
/// # Example
///
/// ```no_run
/// use gemicro_tool_agent::tools::{default_registry, register_web_search_tool};
/// use gemicro_core::{LlmClient, LlmConfig};
/// use std::sync::Arc;
///
/// let mut registry = default_registry();
/// let genai_client = rust_genai::Client::builder("key".to_string()).build();
/// let llm = Arc::new(LlmClient::new(genai_client, LlmConfig::default()));
///
/// register_web_search_tool(&mut registry, llm);
/// ```
pub fn register_web_search_tool(registry: &mut ToolRegistry, llm: Arc<LlmClient>) {
    registry.register(WebSearch::new(llm));
}

/// Create a fully-featured tool registry with all tools.
///
/// This is a convenience function that creates a registry with all
/// available tools, including those that require external resources.
///
/// # Example
///
/// ```no_run
/// use gemicro_tool_agent::tools::full_registry;
/// use gemicro_runner::AgentRegistry;
/// use gemicro_core::{LlmClient, LlmConfig};
/// use std::sync::Arc;
///
/// let agent_registry = Arc::new(AgentRegistry::new());
/// let genai_client = rust_genai::Client::builder("key".to_string()).build();
/// let llm = Arc::new(LlmClient::new(genai_client, LlmConfig::default()));
///
/// let registry = full_registry(agent_registry, llm);
/// assert_eq!(registry.len(), 6); // All 6 tools
/// ```
pub fn full_registry(agent_registry: Arc<AgentRegistry>, llm: Arc<LlmClient>) -> ToolRegistry {
    let mut registry = default_registry();
    register_task_tool(&mut registry, agent_registry, Arc::clone(&llm));
    register_web_search_tool(&mut registry, llm);
    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use gemicro_core::tool::ToolSet;
    use gemicro_core::LlmConfig;

    #[test]
    fn test_default_registry() {
        let registry = default_registry();
        assert_eq!(registry.len(), 4);
        assert!(registry.contains("calculator"));
        assert!(registry.contains("current_datetime"));
        assert!(registry.contains("file_read"));
        assert!(registry.contains("web_fetch"));
    }

    #[test]
    fn test_default_registry_filter() {
        let registry = default_registry();

        // All tools
        let all = registry.filter(&ToolSet::All);
        assert_eq!(all.len(), 4);

        // Specific tool
        let specific = registry.filter(&ToolSet::Specific(vec!["calculator".into()]));
        assert_eq!(specific.len(), 1);
        assert_eq!(specific[0].name(), "calculator");

        // Except tool
        let except = registry.filter(&ToolSet::Except(vec!["calculator".into()]));
        assert_eq!(except.len(), 3);
    }

    #[test]
    fn test_register_task_tool() {
        let mut registry = ToolRegistry::new();
        let agent_registry = Arc::new(AgentRegistry::new());
        let genai_client = rust_genai::Client::builder("test-key".to_string()).build();
        let llm = Arc::new(LlmClient::new(genai_client, LlmConfig::default()));

        register_task_tool(&mut registry, agent_registry, llm);

        assert_eq!(registry.len(), 1);
        assert!(registry.contains("task"));
    }

    #[test]
    fn test_register_web_search_tool() {
        let mut registry = ToolRegistry::new();
        let genai_client = rust_genai::Client::builder("test-key".to_string()).build();
        let llm = Arc::new(LlmClient::new(genai_client, LlmConfig::default()));

        register_web_search_tool(&mut registry, llm);

        assert_eq!(registry.len(), 1);
        assert!(registry.contains("web_search"));
    }

    #[test]
    fn test_full_registry() {
        let agent_registry = Arc::new(AgentRegistry::new());
        let genai_client = rust_genai::Client::builder("test-key".to_string()).build();
        let llm = Arc::new(LlmClient::new(genai_client, LlmConfig::default()));

        let registry = full_registry(agent_registry, llm);

        assert_eq!(registry.len(), 6);
        assert!(registry.contains("calculator"));
        assert!(registry.contains("current_datetime"));
        assert!(registry.contains("file_read"));
        assert!(registry.contains("web_fetch"));
        assert!(registry.contains("task"));
        assert!(registry.contains("web_search"));
    }
}
