//! Agent registry for managing available agents in the REPL
//!
//! The registry stores agent factories (closures) rather than agent instances,
//! allowing agents to be created fresh when needed or with updated configs.

use gemicro_core::Agent;
use std::collections::HashMap;
use std::sync::Arc;

/// Factory function type for creating agents
///
/// Using `Arc<dyn Fn() -> ...>` allows the factory to be cloned and shared.
pub type AgentFactory = Arc<dyn Fn() -> Box<dyn Agent> + Send + Sync>;

/// Registry of available agents
///
/// Manages agent factories and tracks the currently selected agent.
/// Agents are created on-demand from their factories.
pub struct AgentRegistry {
    /// Agent factories keyed by name
    factories: HashMap<String, AgentFactory>,

    /// Currently selected agent name
    current: String,
}

impl AgentRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
            current: String::new(),
        }
    }

    /// Register an agent factory
    ///
    /// # Arguments
    ///
    /// * `name` - Machine-readable agent name (e.g., "deep_research")
    /// * `factory` - Closure that creates the agent
    ///
    /// # Example
    ///
    /// ```ignore
    /// registry.register("deep_research", || {
    ///     Box::new(DeepResearchAgent::new(ResearchConfig::default()).unwrap())
    /// });
    /// ```
    pub fn register<F>(&mut self, name: impl Into<String>, factory: F)
    where
        F: Fn() -> Box<dyn Agent> + Send + Sync + 'static,
    {
        let name = name.into();
        self.factories.insert(name.clone(), Arc::new(factory));

        // Set as current if this is the first agent
        if self.current.is_empty() {
            self.current = name;
        }
    }

    /// Switch to a different agent
    ///
    /// Returns `Ok(())` if the agent exists, `Err` with message otherwise.
    pub fn switch(&mut self, name: &str) -> Result<(), String> {
        if self.factories.contains_key(name) {
            self.current = name.to_string();
            Ok(())
        } else {
            Err(format!(
                "Unknown agent '{}'. Available: {}",
                name,
                self.list().join(", ")
            ))
        }
    }

    /// Get the current agent name
    pub fn current_name(&self) -> &str {
        &self.current
    }

    /// Create a new instance of the current agent
    ///
    /// Returns `None` if no agent is registered.
    pub fn current_agent(&self) -> Option<Box<dyn Agent>> {
        self.factories.get(&self.current).map(|f| f())
    }

    /// Create a new instance of a specific agent by name
    pub fn get(&self, name: &str) -> Option<Box<dyn Agent>> {
        self.factories.get(name).map(|f| f())
    }

    /// List all registered agent names (sorted)
    pub fn list(&self) -> Vec<&str> {
        let mut names: Vec<_> = self.factories.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// Check if an agent with the given name exists
    #[allow(dead_code)]
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }

    /// Get the number of registered agents
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.factories.len()
    }

    /// Check if the registry is empty
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.factories.is_empty()
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gemicro_core::{Agent, AgentContext, AgentStream, AgentUpdate};

    // Mock agent for testing
    struct MockAgent {
        name: &'static str,
    }

    impl Agent for MockAgent {
        fn name(&self) -> &str {
            self.name
        }

        fn description(&self) -> &str {
            "A mock agent for testing"
        }

        fn execute(&self, _query: &str, _context: AgentContext) -> AgentStream<'_> {
            Box::pin(async_stream::try_stream! {
                yield AgentUpdate::decomposition_started();
            })
        }
    }

    #[test]
    fn test_register_and_get() {
        let mut registry = AgentRegistry::new();
        registry.register("test", || Box::new(MockAgent { name: "test" }));

        let agent = registry.get("test").unwrap();
        assert_eq!(agent.name(), "test");
    }

    #[test]
    fn test_first_registered_becomes_current() {
        let mut registry = AgentRegistry::new();
        assert!(registry.current_name().is_empty());

        registry.register("first", || Box::new(MockAgent { name: "first" }));
        assert_eq!(registry.current_name(), "first");

        registry.register("second", || Box::new(MockAgent { name: "second" }));
        assert_eq!(registry.current_name(), "first"); // Still first
    }

    #[test]
    fn test_switch() {
        let mut registry = AgentRegistry::new();
        registry.register("a", || Box::new(MockAgent { name: "a" }));
        registry.register("b", || Box::new(MockAgent { name: "b" }));

        assert_eq!(registry.current_name(), "a");
        registry.switch("b").unwrap();
        assert_eq!(registry.current_name(), "b");
    }

    #[test]
    fn test_switch_nonexistent() {
        let mut registry = AgentRegistry::new();
        registry.register("a", || Box::new(MockAgent { name: "a" }));

        let result = registry.switch("nonexistent");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown agent"));
    }

    #[test]
    fn test_list() {
        let mut registry = AgentRegistry::new();
        registry.register("zebra", || Box::new(MockAgent { name: "zebra" }));
        registry.register("alpha", || Box::new(MockAgent { name: "alpha" }));

        let list = registry.list();
        assert_eq!(list, vec!["alpha", "zebra"]); // Sorted
    }

    #[test]
    fn test_current_agent() {
        let mut registry = AgentRegistry::new();
        registry.register("test", || Box::new(MockAgent { name: "test" }));

        let agent = registry.current_agent().unwrap();
        assert_eq!(agent.name(), "test");
    }

    #[test]
    fn test_contains() {
        let mut registry = AgentRegistry::new();
        registry.register("exists", || Box::new(MockAgent { name: "exists" }));

        assert!(registry.contains("exists"));
        assert!(!registry.contains("missing"));
    }
}
