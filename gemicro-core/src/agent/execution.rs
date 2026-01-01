//! Execution context tracking for subagent orchestration.
//!
//! This module provides [`ExecutionContext`] for tracking parent-child relationships
//! between agents, enabling observability, debugging, and depth limiting.
//!
//! ## Design Philosophy
//!
//! Following [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) principles:
//!
//! - **Opaque IDs**: [`ExecutionId`] values are opaque; never encode semantics in them
//! - **Soft-typed data**: Execution IDs are included in [`AgentUpdate`] data fields, not structure
//! - **Backward compatible**: Root context is the default; existing code works unchanged
//!
//! ## Example
//!
//! ```
//! use gemicro_core::agent::ExecutionContext;
//!
//! // Root agent creates its own context (happens automatically in AgentContext::new())
//! let root = ExecutionContext::root();
//! assert_eq!(root.depth, 0);
//! assert!(root.parent_id.is_none());
//!
//! // When spawning a subagent, create a child context
//! let child = root.child("simple_qa");
//! assert_eq!(child.depth, 1);
//! assert_eq!(child.parent_id, Some(root.execution_id.clone()));
//! assert_eq!(child.path, vec!["simple_qa".to_string()]);
//!
//! // Grandchild context
//! let grandchild = child.child("llm_judge");
//! assert_eq!(grandchild.depth, 2);
//! assert_eq!(grandchild.path, vec!["simple_qa".to_string(), "llm_judge".to_string()]);
//! ```

use std::fmt;
use uuid::Uuid;

/// Unique identifier for an agent execution.
///
/// Opaque identifier - do not encode semantics in the value.
/// Use for tracking, logging, and correlation only.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ExecutionId(String);

impl ExecutionId {
    /// Create a new unique execution ID.
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create an ExecutionId from an existing string.
    ///
    /// Use this for deserialization or testing. For new executions, use [`new()`].
    pub fn from_string(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the string representation of this ID.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for ExecutionId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ExecutionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for ExecutionId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Context for tracking agent execution hierarchy.
///
/// Tracks parent-child relationships between agents for observability,
/// debugging, and depth limiting. Each agent execution gets a unique ID
/// and knows its position in the execution tree.
///
/// # Usage
///
/// Root agents create their context automatically via [`AgentContext::new()`].
/// When spawning subagents, use [`child()`] to create a derived context:
///
/// ```
/// use gemicro_core::agent::ExecutionContext;
///
/// let root = ExecutionContext::root();
/// let child = root.child("simple_qa");
///
/// // Include execution_id in AgentUpdate data for observability
/// // data: json!({ "execution_id": child.execution_id.as_str(), ... })
/// ```
///
/// # Path Tracking
///
/// The [`path`] field tracks the agent hierarchy from root to current:
///
/// ```
/// use gemicro_core::agent::ExecutionContext;
///
/// let root = ExecutionContext::root();
/// assert!(root.path.is_empty());
///
/// let child = root.child("deep_research");
/// assert_eq!(child.path, vec!["deep_research"]);
///
/// let grandchild = child.child("simple_qa");
/// assert_eq!(grandchild.path, vec!["deep_research", "simple_qa"]);
/// ```
///
/// This is useful for logging (e.g., `[deep_research > simple_qa]`) and
/// debugging complex multi-agent workflows.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ExecutionContext {
    /// This execution's unique ID.
    pub execution_id: ExecutionId,

    /// Parent execution ID (None for root agents).
    pub parent_id: Option<ExecutionId>,

    /// Depth in execution tree (0 for root).
    ///
    /// Used for enforcing max depth limits to prevent infinite recursion.
    pub depth: usize,

    /// Path from root (for logging/debugging).
    ///
    /// Contains agent names from root to current execution.
    /// Example: `["deep_research", "simple_qa"]` for a simple_qa called by deep_research.
    pub path: Vec<String>,
}

impl ExecutionContext {
    /// Create a root execution context.
    ///
    /// Use this for top-level agent executions that aren't subagents.
    /// This is called automatically by [`AgentContext::new()`].
    pub fn root() -> Self {
        Self {
            execution_id: ExecutionId::new(),
            parent_id: None,
            depth: 0,
            path: vec![],
        }
    }

    /// Create a child execution context for a subagent.
    ///
    /// The child context:
    /// - Gets a new unique execution ID
    /// - References this context as parent
    /// - Increments depth by 1
    /// - Appends the agent name to the path
    ///
    /// # Example
    ///
    /// ```
    /// use gemicro_core::agent::ExecutionContext;
    ///
    /// let root = ExecutionContext::root();
    /// let child = root.child("simple_qa");
    ///
    /// assert_eq!(child.depth, 1);
    /// assert_eq!(child.parent_id, Some(root.execution_id.clone()));
    /// assert_eq!(child.path, vec!["simple_qa"]);
    /// ```
    pub fn child(&self, agent_name: &str) -> Self {
        let mut path = self.path.clone();
        path.push(agent_name.to_string());

        Self {
            execution_id: ExecutionId::new(),
            parent_id: Some(self.execution_id.clone()),
            depth: self.depth + 1,
            path,
        }
    }

    /// Check if this is a root execution (no parent).
    pub fn is_root(&self) -> bool {
        self.parent_id.is_none()
    }

    /// Get a formatted path string for logging.
    ///
    /// Returns something like `"deep_research > simple_qa"` for nested agents,
    /// or `"root"` for root executions.
    pub fn path_string(&self) -> String {
        if self.path.is_empty() {
            "root".to_string()
        } else {
            self.path.join(" > ")
        }
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::root()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_id_new_is_unique() {
        let id1 = ExecutionId::new();
        let id2 = ExecutionId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_execution_id_from_string() {
        let id = ExecutionId::from_string("test-id-123");
        assert_eq!(id.as_str(), "test-id-123");
        assert_eq!(id.to_string(), "test-id-123");
    }

    #[test]
    fn test_execution_id_default() {
        let id1: ExecutionId = Default::default();
        let id2: ExecutionId = Default::default();
        assert_ne!(id1, id2); // Each default creates a new ID
    }

    #[test]
    fn test_execution_id_display() {
        let id = ExecutionId::from_string("display-test");
        assert_eq!(format!("{}", id), "display-test");
    }

    #[test]
    fn test_execution_id_as_ref() {
        let id = ExecutionId::from_string("as-ref-test");
        let s: &str = id.as_ref();
        assert_eq!(s, "as-ref-test");
    }

    #[test]
    fn test_execution_context_root() {
        let ctx = ExecutionContext::root();
        assert_eq!(ctx.depth, 0);
        assert!(ctx.parent_id.is_none());
        assert!(ctx.path.is_empty());
        assert!(ctx.is_root());
    }

    #[test]
    fn test_execution_context_child() {
        let root = ExecutionContext::root();
        let child = root.child("simple_qa");

        assert_eq!(child.depth, 1);
        assert_eq!(child.parent_id, Some(root.execution_id.clone()));
        assert_eq!(child.path, vec!["simple_qa".to_string()]);
        assert!(!child.is_root());
    }

    #[test]
    fn test_execution_context_grandchild() {
        let root = ExecutionContext::root();
        let child = root.child("deep_research");
        let grandchild = child.child("simple_qa");

        assert_eq!(grandchild.depth, 2);
        assert_eq!(grandchild.parent_id, Some(child.execution_id.clone()));
        assert_eq!(
            grandchild.path,
            vec!["deep_research".to_string(), "simple_qa".to_string()]
        );
    }

    #[test]
    fn test_execution_context_path_string() {
        let root = ExecutionContext::root();
        assert_eq!(root.path_string(), "root");

        let child = root.child("deep_research");
        assert_eq!(child.path_string(), "deep_research");

        let grandchild = child.child("simple_qa");
        assert_eq!(grandchild.path_string(), "deep_research > simple_qa");
    }

    #[test]
    fn test_execution_context_default() {
        let ctx: ExecutionContext = Default::default();
        assert!(ctx.is_root());
        assert_eq!(ctx.depth, 0);
    }

    #[test]
    fn test_execution_context_child_gets_unique_id() {
        let root = ExecutionContext::root();
        let child1 = root.child("agent_a");
        let child2 = root.child("agent_b");

        assert_ne!(child1.execution_id, child2.execution_id);
        // Both have same parent
        assert_eq!(child1.parent_id, child2.parent_id);
    }

    #[test]
    fn test_deep_nesting() {
        let mut ctx = ExecutionContext::root();
        for i in 0..10 {
            ctx = ctx.child(&format!("agent_{}", i));
        }

        assert_eq!(ctx.depth, 10);
        assert_eq!(ctx.path.len(), 10);
        assert_eq!(ctx.path[0], "agent_0");
        assert_eq!(ctx.path[9], "agent_9");
    }
}
