//! gemicro-cli library for building custom agent runners.
//!
//! This crate provides pre-configured agent registries that can be used to build
//! custom CLI tools or integrate gemicro agents into other applications.
//!
//! # Quick Start
//!
//! Use [`default_registry`] to get a registry with all bundled agents:
//!
//! ```no_run
//! use gemicro_cli::{default_registry, RegistryOptions};
//!
//! // Create registry with all bundled agents
//! let registry = default_registry(None);
//!
//! // Or with a model override
//! let registry = default_registry(Some(
//!     RegistryOptions::default().with_model("gemini-3.0-flash-preview")
//! ));
//!
//! // Get an agent by name
//! let agent = registry.get("deep_research").expect("agent should exist");
//! ```
//!
//! # Building a Custom Binary
//!
//! You can create your own binary that extends the bundled agents:
//!
//! ```no_run
//! use gemicro_cli::{default_registry, RegistryOptions};
//!
//! // Start with bundled agents
//! let mut registry = default_registry(Some(
//!     RegistryOptions::default().with_model("gemini-3.0-flash-preview")
//! ));
//!
//! // Add your custom agent
//! // registry.register("my_agent", || Box::new(MyCustomAgent::new()));
//!
//! // Now use the registry in your application...
//! ```
//!
//! # Bundled Agents
//!
//! The default registry includes:
//!
//! | Agent | Description |
//! |-------|-------------|
//! | `deep_research` | Multi-step research with parallel sub-queries |
//! | `prompt_agent` | General-purpose prompt-based agent |
//! | `developer` | Tool-using development agent |
//! | `react` | ReAct (Reasoning + Acting) agent |
//! | `echo` | Simple echo agent for testing |
//! | `critique` | Self-validation agent |
//!
//! # Markdown Agents
//!
//! Use [`default_registry_with_markdown`] to also load agents defined in
//! markdown files:
//!
//! ```no_run
//! use gemicro_cli::default_registry_with_markdown;
//! use std::path::Path;
//!
//! let registry = default_registry_with_markdown(
//!     None,
//!     Path::new("agents/runtime-agents"),
//! );
//! ```

mod registry;

pub mod config;
pub mod confirmation;

// Re-export public API from registry module
pub use registry::{
    default_registry, default_registry_with_markdown, register_builtin_agents,
    register_markdown_agents, RegistryOptions,
};

// Re-export commonly used types from dependencies
pub use gemicro_runner::AgentRegistry;
