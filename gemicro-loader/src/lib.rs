//! Agent definition loaders for gemicro.
//!
//! This crate provides loaders for agent definitions from various sources:
//!
//! - **Markdown**: YAML frontmatter + markdown body ([`markdown`] module)
//!
//! # Example
//!
//! ```
//! use gemicro_loader::markdown::{parse_markdown_agent_str, MarkdownAgent};
//!
//! let markdown = r#"---
//! name: my-agent
//! description: A helpful assistant
//! ---
//!
//! You are a helpful assistant.
//! "#;
//!
//! let agent = parse_markdown_agent_str(markdown).unwrap();
//! assert_eq!(agent.name, "my-agent");
//! ```

pub mod markdown;

// Re-export commonly used types at crate root
pub use markdown::{
    parse_markdown_agent, parse_markdown_agent_str, MarkdownAgent, MarkdownAgentError,
};
