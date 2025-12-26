//! Agent registry for managing available agents.
//!
//! The registry stores agent factories (closures) rather than agent instances,
//! allowing agents to be created fresh when needed or with updated configs.
//!
//! # Example
//!
//! ```ignore
//! use gemicro_runner::AgentRegistry;
//! use gemicro_core::DeepResearchAgent;
//!
//! let mut registry = AgentRegistry::new();
//! registry.register("deep_research", || {
//!     Box::new(DeepResearchAgent::new(Default::default()).unwrap())
//! });
//!
//! // Create an agent instance
//! let agent = registry.get("deep_research").unwrap();
//! ```

use gemicro_core::Agent;
use std::collections::HashMap;
use std::sync::Arc;

/// Factory function type for creating agents.
///
/// Using `Arc<dyn Fn() -> ...>` allows the factory to be cloned and shared.
pub type AgentFactory = Arc<dyn Fn() -> Box<dyn Agent> + Send + Sync>;

/// Registry of available agents.
///
/// A pure collection of agent factories, keyed by name. This is a stateless
/// catalog - current selection state should be managed by the consumer
/// (e.g., a REPL session or evaluation harness).
///
/// # Example
///
/// ```ignore
/// let mut registry = AgentRegistry::new();
/// registry.register("agent_a", || Box::new(AgentA::new()));
/// registry.register("agent_b", || Box::new(AgentB::new()));
///
/// for name in registry.list() {
///     let agent = registry.get(name).unwrap();
///     println!("{}: {}", agent.name(), agent.description());
/// }
/// ```
pub struct AgentRegistry {
    /// Agent factories keyed by name
    factories: HashMap<String, AgentFactory>,
}

impl AgentRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Register an agent factory.
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
        self.factories.insert(name.into(), Arc::new(factory));
    }

    /// Create a new instance of an agent by name.
    ///
    /// Returns `None` if no agent with the given name is registered.
    pub fn get(&self, name: &str) -> Option<Box<dyn Agent>> {
        self.factories.get(name).map(|f| f())
    }

    /// List all registered agent names (sorted alphabetically).
    pub fn list(&self) -> Vec<&str> {
        let mut names: Vec<_> = self.factories.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// Check if an agent with the given name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }

    /// Get the number of registered agents.
    pub fn len(&self) -> usize {
        self.factories.len()
    }

    /// Check if the registry is empty.
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
    use gemicro_core::{AgentContext, AgentStream, AgentUpdate};

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
    fn test_get_nonexistent() {
        let registry = AgentRegistry::new();
        assert!(registry.get("nonexistent").is_none());
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
    fn test_contains() {
        let mut registry = AgentRegistry::new();
        registry.register("exists", || Box::new(MockAgent { name: "exists" }));

        assert!(registry.contains("exists"));
        assert!(!registry.contains("missing"));
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut registry = AgentRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);

        registry.register("test", || Box::new(MockAgent { name: "test" }));
        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_default() {
        let registry = AgentRegistry::default();
        assert!(registry.is_empty());
    }
}
