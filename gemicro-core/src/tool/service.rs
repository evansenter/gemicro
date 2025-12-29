//! GemicroToolService for rust-genai integration.
//!
//! Provides [`GemicroToolService`] which implements rust-genai's `ToolService` trait,
//! enabling gemicro tools to work with `create_with_auto_functions()`.

use super::{ConfirmationHandler, ToolCallableAdapter, ToolRegistry, ToolSet};
use rust_genai::{CallableFunction, ToolService};
use std::sync::Arc;

/// Gemicro's implementation of rust-genai's [`ToolService`].
///
/// Combines [`ToolRegistry`] with [`ToolSet`] filtering and optional
/// [`ConfirmationHandler`] for dangerous tools.
///
/// The registry starts **empty** - consumers register their own tools.
/// This keeps gemicro-core generic and avoids auto-importing tools.
///
/// # Example
///
/// ```no_run
/// use gemicro_core::tool::{
///     GemicroToolService, ToolRegistry, ToolSet, AutoApprove,
/// };
/// use std::sync::Arc;
///
/// // Create registry and register tools
/// let mut registry = ToolRegistry::new();
/// // registry.register(MyCalculator);
/// // registry.register(MyDatetime);
///
/// // Create service with confirmation handler
/// let service = GemicroToolService::new(Arc::new(registry))
///     .with_confirmation_handler(Arc::new(AutoApprove));
///
/// // Use with rust-genai:
/// // client.interaction()
/// //     .with_tool_service(Arc::new(service))
/// //     .create_with_auto_functions()
/// //     .await?;
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GemicroToolService {
    registry: Arc<ToolRegistry>,
    filter: ToolSet,
    confirmation_handler: Option<Arc<dyn ConfirmationHandler>>,
}

impl GemicroToolService {
    /// Create a new tool service with the given registry.
    ///
    /// The registry should be populated by the consumer with
    /// whatever tools are appropriate for their use case.
    ///
    /// By default:
    /// - All tools in the registry are available (filter = `ToolSet::All`)
    /// - No confirmation handler is set (dangerous tools will be denied)
    pub fn new(registry: Arc<ToolRegistry>) -> Self {
        Self {
            registry,
            filter: ToolSet::All,
            confirmation_handler: None,
        }
    }

    /// Create a tool service from a reference to a registry.
    ///
    /// Convenience constructor when you have `&ToolRegistry` instead of `Arc`.
    pub fn from_registry(registry: &ToolRegistry) -> Self {
        Self::new(Arc::new(registry.clone()))
    }

    /// Filter which tools are available to the LLM.
    ///
    /// Use this to restrict the tool set for specific agents or contexts.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_core::tool::{GemicroToolService, ToolRegistry, ToolSet};
    /// use std::sync::Arc;
    ///
    /// let registry = ToolRegistry::new();
    /// let service = GemicroToolService::new(Arc::new(registry))
    ///     .with_filter(ToolSet::Specific(vec!["calculator".into()]));
    /// ```
    pub fn with_filter(mut self, filter: ToolSet) -> Self {
        self.filter = filter;
        self
    }

    /// Set the confirmation handler for tools that require user approval.
    ///
    /// If not set, tools requiring confirmation will be denied by default.
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_core::tool::{GemicroToolService, ToolRegistry, AutoApprove};
    /// use std::sync::Arc;
    ///
    /// let registry = ToolRegistry::new();
    /// let service = GemicroToolService::new(Arc::new(registry))
    ///     .with_confirmation_handler(Arc::new(AutoApprove));
    /// ```
    pub fn with_confirmation_handler(mut self, handler: Arc<dyn ConfirmationHandler>) -> Self {
        self.confirmation_handler = Some(handler);
        self
    }

    /// Get the underlying registry.
    pub fn registry(&self) -> &Arc<ToolRegistry> {
        &self.registry
    }

    /// Get the current filter.
    pub fn filter(&self) -> &ToolSet {
        &self.filter
    }

    /// Check if a confirmation handler is configured.
    pub fn has_confirmation_handler(&self) -> bool {
        self.confirmation_handler.is_some()
    }
}

impl ToolService for GemicroToolService {
    fn tools(&self) -> Vec<Arc<dyn CallableFunction>> {
        self.registry
            .filter(&self.filter)
            .into_iter()
            .map(|tool| {
                let mut adapter = ToolCallableAdapter::new(tool);
                if let Some(h) = &self.confirmation_handler {
                    adapter = adapter.with_confirmation_handler(Arc::clone(h));
                }
                Arc::new(adapter) as Arc<dyn CallableFunction>
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{AutoApprove, AutoDeny, Tool, ToolError, ToolResult};
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
            "A mock tool"
        }

        fn parameters_schema(&self) -> Value {
            json!({"type": "object", "properties": {}})
        }

        async fn execute(&self, _input: Value) -> Result<ToolResult, ToolError> {
            Ok(ToolResult::text("mock result"))
        }
    }

    #[test]
    fn test_service_new_empty_registry() {
        let registry = Arc::new(ToolRegistry::new());
        let service = GemicroToolService::new(registry);

        assert!(service.tools().is_empty());
        assert!(!service.has_confirmation_handler());
    }

    #[test]
    fn test_service_with_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("tool_a"));
        registry.register(MockTool::new("tool_b"));

        let service = GemicroToolService::new(Arc::new(registry));
        let tools = service.tools();

        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn test_service_with_filter() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("tool_a"));
        registry.register(MockTool::new("tool_b"));
        registry.register(MockTool::new("tool_c"));

        let service = GemicroToolService::new(Arc::new(registry))
            .with_filter(ToolSet::Specific(vec!["tool_a".into(), "tool_c".into()]));

        let tools = service.tools();
        assert_eq!(tools.len(), 2);

        let names: Vec<String> = tools
            .iter()
            .map(|t| t.declaration().name().to_string())
            .collect();
        assert!(names.contains(&"tool_a".to_string()));
        assert!(names.contains(&"tool_c".to_string()));
        assert!(!names.contains(&"tool_b".to_string()));
    }

    #[test]
    fn test_service_with_confirmation_handler() {
        let registry = Arc::new(ToolRegistry::new());
        let handler: Arc<dyn ConfirmationHandler> = Arc::new(AutoApprove);

        let service = GemicroToolService::new(registry).with_confirmation_handler(handler);

        assert!(service.has_confirmation_handler());
    }

    #[test]
    fn test_service_from_registry() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("test"));

        let service = GemicroToolService::from_registry(&registry);
        assert_eq!(service.tools().len(), 1);
    }

    #[test]
    fn test_service_filter_none() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("tool_a"));

        let service = GemicroToolService::new(Arc::new(registry)).with_filter(ToolSet::None);

        assert!(service.tools().is_empty());
    }

    #[test]
    fn test_service_clone() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("tool"));

        let service = GemicroToolService::new(Arc::new(registry))
            .with_filter(ToolSet::All)
            .with_confirmation_handler(Arc::new(AutoDeny));

        let cloned = service.clone();
        assert_eq!(cloned.tools().len(), 1);
        assert!(cloned.has_confirmation_handler());
    }

    #[test]
    fn test_service_debug() {
        let registry = Arc::new(ToolRegistry::new());
        let service = GemicroToolService::new(registry);

        let debug = format!("{:?}", service);
        assert!(debug.contains("GemicroToolService"));
    }
}
