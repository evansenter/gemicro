//! Markdown agent loader for defining agents via markdown files with YAML frontmatter.
//!
//! This module provides parsing for markdown-defined agents. The format uses YAML
//! frontmatter for configuration and markdown body for the system prompt.
//!
//! # Format
//!
//! ```markdown
//! ---
//! name: my-agent
//! description: What this agent does
//! model: gemini-2.0-flash  # optional
//! tools:                   # optional, defaults to inherit
//!   - file_read
//!   - grep
//! ---
//!
//! You are an expert assistant...
//! (rest of markdown becomes the system prompt)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use gemicro_cli::config::markdown_agents::parse_markdown_agent_str;
//!
//! let markdown = r#"---
//! name: code-reviewer
//! description: Reviews code for quality issues
//! ---
//!
//! You are an expert code reviewer.
//! "#;
//!
//! let agent = parse_markdown_agent_str(markdown).unwrap();
//! assert_eq!(agent.name, "code-reviewer");
//! ```

use gemicro_core::agent::PromptAgentDef;
use gemicro_core::ToolSet;
use serde::Deserialize;
use std::fmt;
use std::io;
use std::path::Path;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur when parsing markdown agent definitions.
#[derive(Debug)]
#[non_exhaustive]
pub enum MarkdownAgentError {
    /// File I/O error.
    Io(io::Error),
    /// Missing or malformed YAML frontmatter.
    MissingFrontmatter,
    /// YAML parsing error.
    Yaml(serde_yaml::Error),
    /// Validation error (e.g., missing required fields).
    Validation(String),
}

impl fmt::Display for MarkdownAgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::MissingFrontmatter => write!(
                f,
                "Missing or malformed YAML frontmatter (expected ---\\n...\\n---)"
            ),
            Self::Yaml(e) => write!(f, "YAML parse error: {}", e),
            Self::Validation(msg) => write!(f, "Validation error: {}", msg),
        }
    }
}

impl std::error::Error for MarkdownAgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Yaml(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for MarkdownAgentError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<serde_yaml::Error> for MarkdownAgentError {
    fn from(err: serde_yaml::Error) -> Self {
        Self::Yaml(err)
    }
}

// ============================================================================
// Frontmatter Schema
// ============================================================================

/// YAML frontmatter schema for markdown agents.
#[derive(Deserialize)]
struct MarkdownFrontmatter {
    /// Unique name for the agent (used for registration).
    name: String,

    /// Human-readable description.
    #[serde(default)]
    description: Option<String>,

    /// Model override (e.g., "gemini-2.0-flash").
    #[serde(default)]
    model: Option<String>,

    /// Tool names to make available. Empty = inherit from parent.
    #[serde(default)]
    tools: Vec<String>,
}

// ============================================================================
// Public Types
// ============================================================================

/// A parsed markdown agent definition.
#[derive(Debug, Clone)]
pub struct MarkdownAgent {
    /// Unique name for registration.
    pub name: String,

    /// The agent definition (system prompt, tools, model).
    pub definition: PromptAgentDef,
}

// ============================================================================
// Parsing Functions
// ============================================================================

/// Parse a markdown agent from a file path.
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be read
/// - The frontmatter is missing or malformed
/// - The YAML cannot be parsed
/// - Required fields are missing
pub fn parse_markdown_agent(path: &Path) -> Result<MarkdownAgent, MarkdownAgentError> {
    let content = std::fs::read_to_string(path)?;
    parse_markdown_agent_str(&content)
}

/// Parse a markdown agent from a string.
///
/// # Errors
///
/// Returns an error if:
/// - The frontmatter is missing or malformed
/// - The YAML cannot be parsed
/// - Required fields are missing
pub fn parse_markdown_agent_str(content: &str) -> Result<MarkdownAgent, MarkdownAgentError> {
    // Split frontmatter from body
    let (frontmatter, body) = split_frontmatter(content)?;

    // Parse YAML frontmatter
    let fm: MarkdownFrontmatter = serde_yaml::from_str(frontmatter)?;

    // Validate name
    if fm.name.trim().is_empty() {
        return Err(MarkdownAgentError::Validation(
            "name is required and cannot be empty".into(),
        ));
    }

    // Convert tools list to ToolSet
    let tools = if fm.tools.is_empty() {
        ToolSet::Inherit
    } else {
        ToolSet::Specific(fm.tools)
    };

    // Build the definition
    let description = fm.description.unwrap_or_else(|| fm.name.clone());
    let mut def = PromptAgentDef::new(description)
        .with_system_prompt(body.trim())
        .with_tools(tools);

    if let Some(model) = fm.model {
        def = def.with_model(model);
    }

    // Validate the definition
    def.validate()
        .map_err(|e| MarkdownAgentError::Validation(e.to_string()))?;

    Ok(MarkdownAgent {
        name: fm.name,
        definition: def,
    })
}

/// Split markdown content into frontmatter and body.
///
/// Expects format:
/// ```text
/// ---
/// yaml content
/// ---
/// body content
/// ```
fn split_frontmatter(content: &str) -> Result<(&str, &str), MarkdownAgentError> {
    // Must start with ---
    if !content.starts_with("---") {
        return Err(MarkdownAgentError::MissingFrontmatter);
    }

    // Find the closing ---
    let rest = &content[3..];
    let end = rest
        .find("\n---")
        .ok_or(MarkdownAgentError::MissingFrontmatter)?;

    let frontmatter = rest[..end].trim();
    let body = &rest[end + 4..]; // Skip "\n---"

    Ok((frontmatter, body))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_agent() {
        let markdown = r#"---
name: test-agent
---

You are a helpful assistant."#;

        let agent = parse_markdown_agent_str(markdown).unwrap();
        assert_eq!(agent.name, "test-agent");
        assert_eq!(agent.definition.description, "test-agent"); // Uses name as default
        assert_eq!(
            agent.definition.system_prompt,
            "You are a helpful assistant."
        );
        assert!(matches!(agent.definition.tools, ToolSet::Inherit));
        assert!(agent.definition.model.is_none());
    }

    #[test]
    fn test_parse_full_agent() {
        let markdown = r#"---
name: code-reviewer
description: Reviews code for quality issues
model: gemini-2.0-flash
tools:
  - file_read
  - grep
---

You are an expert code reviewer.

## Instructions

Review the code carefully."#;

        let agent = parse_markdown_agent_str(markdown).unwrap();
        assert_eq!(agent.name, "code-reviewer");
        assert_eq!(
            agent.definition.description,
            "Reviews code for quality issues"
        );
        assert!(agent
            .definition
            .system_prompt
            .contains("expert code reviewer"));
        assert!(agent.definition.system_prompt.contains("## Instructions"));
        assert_eq!(agent.definition.model, Some("gemini-2.0-flash".to_string()));

        if let ToolSet::Specific(tools) = &agent.definition.tools {
            assert_eq!(tools.len(), 2);
            assert!(tools.contains(&"file_read".to_string()));
            assert!(tools.contains(&"grep".to_string()));
        } else {
            panic!("Expected ToolSet::Specific");
        }
    }

    #[test]
    fn test_missing_frontmatter() {
        let markdown = "# No frontmatter\n\nJust content.";
        let result = parse_markdown_agent_str(markdown);
        assert!(matches!(
            result,
            Err(MarkdownAgentError::MissingFrontmatter)
        ));
    }

    #[test]
    fn test_unclosed_frontmatter() {
        let markdown = "---\nname: test\n\nNo closing delimiter";
        let result = parse_markdown_agent_str(markdown);
        assert!(matches!(
            result,
            Err(MarkdownAgentError::MissingFrontmatter)
        ));
    }

    #[test]
    fn test_empty_name() {
        let markdown = r#"---
name: ""
---

Content"#;
        let result = parse_markdown_agent_str(markdown);
        assert!(matches!(result, Err(MarkdownAgentError::Validation(_))));
    }

    #[test]
    fn test_whitespace_name() {
        let markdown = r#"---
name: "   "
---

Content"#;
        let result = parse_markdown_agent_str(markdown);
        assert!(matches!(result, Err(MarkdownAgentError::Validation(_))));
    }

    #[test]
    fn test_missing_name() {
        let markdown = r#"---
description: Test agent
---

Content"#;
        let result = parse_markdown_agent_str(markdown);
        // serde_yaml will error on missing required field
        assert!(matches!(result, Err(MarkdownAgentError::Yaml(_))));
    }

    #[test]
    fn test_empty_body() {
        let markdown = r#"---
name: test
---
"#;
        let result = parse_markdown_agent_str(markdown);
        // Empty system prompt should fail validation
        assert!(matches!(result, Err(MarkdownAgentError::Validation(_))));
    }

    #[test]
    fn test_body_whitespace_trimmed() {
        let markdown = r#"---
name: test
---


  Trimmed content.

"#;

        let agent = parse_markdown_agent_str(markdown).unwrap();
        assert_eq!(agent.definition.system_prompt, "Trimmed content.");
    }

    #[test]
    fn test_error_display() {
        let err = MarkdownAgentError::MissingFrontmatter;
        let msg = format!("{}", err);
        assert!(msg.contains("frontmatter"));

        let err = MarkdownAgentError::Validation("test error".into());
        let msg = format!("{}", err);
        assert!(msg.contains("test error"));
    }
}
