//! Optional utility tools for PromptAgent.
//!
//! These tools are included for convenience. Use [`default_registry()`] to get
//! a pre-configured registry with all bundled tools, or register individually.
//!
//! # Example (Quick Start)
//!
//! ```no_run
//! use gemicro_prompt_agent::tools;
//! use gemicro_core::AgentContext;
//!
//! // Get registry with Calculator + CurrentDatetime pre-registered
//! let registry = tools::default_registry();
//! // let context = AgentContext::new(llm).with_tools(registry);
//! ```
//!
//! # Example (Selective Registration)
//!
//! ```no_run
//! use gemicro_prompt_agent::tools::Calculator;
//! use gemicro_core::tool::ToolRegistry;
//!
//! let mut registry = ToolRegistry::new();
//! registry.register(Calculator);
//! ```

mod calculator;
mod datetime;

pub use calculator::Calculator;
pub use datetime::CurrentDatetime;

use gemicro_core::tool::ToolRegistry;

/// Create a ToolRegistry with all bundled tools pre-registered.
///
/// Includes: Calculator, CurrentDatetime
///
/// # Example
///
/// ```no_run
/// use gemicro_prompt_agent::tools;
/// use gemicro_core::AgentContext;
///
/// let registry = tools::default_registry();
/// // let context = AgentContext::new(llm).with_tools(registry);
/// ```
pub fn default_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Calculator);
    registry.register(CurrentDatetime);
    registry
}
