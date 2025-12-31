//! Configuration file loading and hot-reload support.
//!
//! This module provides file-based configuration for gemicro agents,
//! supporting both project-local and user-global config files.
//!
//! # Config File Locations
//!
//! Config files are loaded in priority order (later overrides earlier):
//! 1. `~/.gemicro/config.toml` - User-global defaults
//! 2. `./gemicro.toml` - Project-local overrides
//!
//! # Example Config File
//!
//! ```toml
//! [deep_research]
//! min_sub_queries = 3
//! max_sub_queries = 5
//! max_concurrent_sub_queries = 5
//! continue_on_partial_failure = true
//! timeout_secs = 60
//! use_google_search = false
//!
//! [deep_research.prompts]
//! decomposition_system = "You are a research query decomposition expert..."
//! sub_query_system = "You are a research assistant..."
//! synthesis_system = "You are a research synthesis expert..."
//!
//! [tool_agent]
//! timeout_secs = 60
//! system_prompt = "You are a helpful assistant..."
//! ```

pub mod loader;
mod types;

pub use loader::{ConfigChange, ConfigLoader, ConfigSource};
pub use types::{DeepResearchToml, GemicroConfig, PromptsToml, ToolAgentToml};
