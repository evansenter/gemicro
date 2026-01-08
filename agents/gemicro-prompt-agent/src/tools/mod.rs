//! Optional utility tools for PromptAgent.
//!
//! These tools are included for convenience but follow the "Explicit Over Implicit"
//! principle - they are NOT automatically registered. Callers must explicitly
//! add them to a [`ToolRegistry`] if needed.
//!
//! # Example
//!
//! ```no_run
//! use gemicro_prompt_agent::tools::{Calculator, CurrentDatetime};
//! use gemicro_core::tool::ToolRegistry;
//!
//! let mut registry = ToolRegistry::new();
//! registry.register(Calculator);
//! registry.register(CurrentDatetime);
//! ```

mod calculator;
mod datetime;

pub use calculator::Calculator;
pub use datetime::CurrentDatetime;
