//! Tool registry for managing available tools.

use super::{Tool, ToolSet};
use genai_rs::FunctionDeclaration;
use std::collections::HashMap;
use std::sync::Arc;

/// Registry of available tools.
///
/// The registry stores tool instances (not factories) since tools are
/// inherently stateless. Tools can be registered, retrieved by name,
/// listed, and filtered based on a [`ToolSet`] specification.
///
/// # Thread Safety
///
/// The registry stores tools as `Arc<dyn Tool>`, making them safely
/// shareable across threads. The registry itself is `Send + Sync`.
///
/// # Example
///
/// ```no_run
/// use gemicro_core::tool::{ToolRegistry, ToolSet};
///
/// let mut registry = ToolRegistry::new();
/// // registry.register(Calculator);
/// // registry.register(WebFetch::new());
///
/// // Get a specific tool
/// if let Some(calc) = registry.get("calculator") {
///     println!("Found: {}", calc.description());
/// }
///
/// // List all tools
/// for name in registry.list() {
///     println!("Available: {}", name);
/// }
///
/// // Filter for specific agent
/// let tools = registry.filter(&ToolSet::Specific(vec!["calculator".into()]));
/// ```
#[derive(Debug, Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool in the registry.
    ///
    /// If a tool with the same name already exists, it will be replaced.
    /// Returns `&mut Self` for chaining.
    pub fn register<T: Tool + 'static>(&mut self, tool: T) -> &mut Self {
        self.tools.insert(tool.name().to_string(), Arc::new(tool));
        self
    }

    /// Register a tool that's already wrapped in Arc.
    ///
    /// Useful when you have an `Arc<dyn Tool>` from elsewhere.
    pub fn register_arc(&mut self, tool: Arc<dyn Tool>) -> &mut Self {
        self.tools.insert(tool.name().to_string(), tool);
        self
    }

    /// Get a tool by name.
    ///
    /// Returns `None` if no tool with the given name is registered.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// List all registered tool names, sorted alphabetically.
    pub fn list(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tools.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// Check if a tool is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Get the number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Filter tools based on a [`ToolSet`] specification.
    ///
    /// Returns a vector of tools matching the filter.
    pub fn filter(&self, tool_set: &ToolSet) -> Vec<Arc<dyn Tool>> {
        self.tools
            .iter()
            .filter(|(name, _)| tool_set.matches(name))
            .map(|(_, tool)| Arc::clone(tool))
            .collect()
    }

    /// Get FunctionDeclarations for tools matching a filter.
    ///
    /// Useful for genai-rs integration where you need to pass
    /// function declarations to the LLM.
    pub fn to_function_declarations(&self, tool_set: &ToolSet) -> Vec<FunctionDeclaration> {
        self.filter(tool_set)
            .iter()
            .map(|tool| tool.to_function_declaration())
            .collect()
    }

    /// Get an iterator over all tools.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Arc<dyn Tool>)> {
        self.tools.iter().map(|(k, v)| (k.as_str(), v))
    }
}

impl Clone for ToolRegistry {
    fn clone(&self) -> Self {
        Self {
            tools: self.tools.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{ToolError, ToolResult};
    use async_trait::async_trait;
    use serde_json::{json, Value};

    #[derive(Debug)]
    struct MockTool {
        name: String,
    }

    impl MockTool {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
    }

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "A mock tool for testing"
        }

        fn parameters_schema(&self) -> Value {
            json!({
                "type": "object",
                "properties": {}
            })
        }

        async fn execute(&self, _input: Value) -> Result<ToolResult, ToolError> {
            Ok(ToolResult::text("mock result"))
        }
    }

    #[test]
    fn test_registry_new() {
        let registry = ToolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("test"));

        assert_eq!(registry.len(), 1);
        assert!(registry.contains("test"));

        let tool = registry.get("test").unwrap();
        assert_eq!(tool.name(), "test");
    }

    #[test]
    fn test_registry_get_nonexistent() {
        let registry = ToolRegistry::new();
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_list_sorted() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("zebra"));
        registry.register(MockTool::new("alpha"));
        registry.register(MockTool::new("middle"));

        let list = registry.list();
        assert_eq!(list, vec!["alpha", "middle", "zebra"]);
    }

    #[test]
    fn test_registry_filter_all() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("a"));
        registry.register(MockTool::new("b"));

        let filtered = registry.filter(&ToolSet::All);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_registry_filter_none() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("a"));
        registry.register(MockTool::new("b"));

        let filtered = registry.filter(&ToolSet::None);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_registry_filter_specific() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("a"));
        registry.register(MockTool::new("b"));
        registry.register(MockTool::new("c"));

        let filtered = registry.filter(&ToolSet::Specific(vec!["a".into(), "c".into()]));
        assert_eq!(filtered.len(), 2);

        let names: Vec<&str> = filtered.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"a"));
        assert!(names.contains(&"c"));
        assert!(!names.contains(&"b"));
    }

    #[test]
    fn test_registry_filter_except() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("a"));
        registry.register(MockTool::new("b"));
        registry.register(MockTool::new("c"));

        let filtered = registry.filter(&ToolSet::Except(vec!["b".into()]));
        assert_eq!(filtered.len(), 2);

        let names: Vec<&str> = filtered.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"a"));
        assert!(names.contains(&"c"));
        assert!(!names.contains(&"b"));
    }

    #[test]
    fn test_registry_replace_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("test"));
        registry.register(MockTool::new("test")); // Same name

        assert_eq!(registry.len(), 1); // Only one tool
    }

    #[test]
    fn test_registry_chaining() {
        let mut registry = ToolRegistry::new();
        registry
            .register(MockTool::new("a"))
            .register(MockTool::new("b"))
            .register(MockTool::new("c"));

        assert_eq!(registry.len(), 3);
    }

    #[test]
    fn test_registry_clone() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("test"));

        let cloned = registry.clone();
        assert_eq!(cloned.len(), 1);
        assert!(cloned.contains("test"));
    }

    #[test]
    fn test_to_function_declarations() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("a"));
        registry.register(MockTool::new("b"));

        let declarations = registry.to_function_declarations(&ToolSet::All);
        assert_eq!(declarations.len(), 2);
    }
}
