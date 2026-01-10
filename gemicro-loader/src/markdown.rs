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
//! model: gemini-3.0-flash-preview  # optional
//! tools:                           # optional, see below for behavior
//!   - file_read
//!   - grep
//! ---
//!
//! You are an expert assistant...
//! (rest of markdown becomes the system prompt)
//! ```
//!
//! # Tool Configuration
//!
//! The `tools` field controls which tools the agent can use:
//!
//! - **Omitted**: Inherit tools from parent context (`ToolSet::Inherit`)
//! - **Empty array `[]`**: No tools allowed (`ToolSet::None`)
//! - **List of names**: Only those specific tools (`ToolSet::Specific`)
//!
//! # Example
//!
//! ```
//! use gemicro_loader::markdown::parse_markdown_agent_str;
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

    /// Model override (e.g., "gemini-3.0-flash-preview").
    #[serde(default)]
    model: Option<String>,

    /// Tool names to make available.
    ///
    /// - `None` (omitted): Inherit tools from parent context
    /// - `Some([])` (empty array): No tools allowed
    /// - `Some([...])` (list): Only those specific tools
    #[serde(default)]
    tools: Option<Vec<String>>,
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
    // Split frontmatter from body (handles CRLF normalization)
    let (frontmatter, body) = split_frontmatter(content)?;

    // Parse YAML frontmatter
    let fm: MarkdownFrontmatter = serde_yaml::from_str(&frontmatter)?;

    // Validate name
    if fm.name.trim().is_empty() {
        return Err(MarkdownAgentError::Validation(
            "name is required and cannot be empty".into(),
        ));
    }

    // Convert tools list to ToolSet:
    // - None (omitted): Inherit from parent
    // - Some([]) (empty array): No tools
    // - Some([...]) (list): Specific tools
    let tools = match fm.tools {
        None => ToolSet::Inherit,
        Some(list) if list.is_empty() => ToolSet::None,
        Some(list) => ToolSet::Specific(list),
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
///
/// Handles both Unix (`\n`) and Windows (`\r\n`) line endings.
fn split_frontmatter(content: &str) -> Result<(String, String), MarkdownAgentError> {
    // Normalize line endings for cross-platform compatibility
    let content = content.replace("\r\n", "\n");

    // Must start with ---
    if !content.starts_with("---") {
        return Err(MarkdownAgentError::MissingFrontmatter);
    }

    // Find the closing ---
    let rest = &content[3..];
    let end = rest
        .find("\n---")
        .ok_or(MarkdownAgentError::MissingFrontmatter)?;

    let frontmatter = rest[..end].trim().to_string();
    let body = rest[end + 4..].to_string(); // Skip "\n---"

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
model: gemini-3.0-flash-preview
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
        assert_eq!(
            agent.definition.model,
            Some("gemini-3.0-flash-preview".to_string())
        );

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

    #[test]
    fn test_windows_line_endings() {
        // CRLF line endings should be handled
        let markdown = "---\r\nname: test\r\n---\r\n\r\nYou are a helper.";
        let agent = parse_markdown_agent_str(markdown).unwrap();
        assert_eq!(agent.name, "test");
        assert_eq!(agent.definition.system_prompt, "You are a helper.");
    }

    #[test]
    fn test_empty_tools_means_none() {
        // Explicit empty tools array means no tools (ToolSet::None)
        let markdown = r#"---
name: no-tools-agent
tools: []
---

You have no tools."#;

        let agent = parse_markdown_agent_str(markdown).unwrap();
        assert!(matches!(agent.definition.tools, ToolSet::None));
    }

    #[test]
    fn test_omitted_tools_means_inherit() {
        // Omitted tools field means inherit from parent (ToolSet::Inherit)
        let markdown = r#"---
name: inherit-tools-agent
---

You inherit tools."#;

        let agent = parse_markdown_agent_str(markdown).unwrap();
        assert!(matches!(agent.definition.tools, ToolSet::Inherit));
    }
}
