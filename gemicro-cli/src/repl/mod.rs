//! REPL (Read-Eval-Print-Loop) for interactive agent exploration
//!
//! This module provides the interactive terminal interface for running
//! and testing agents with real-time streaming output.

mod commands;
mod registry;
mod session;

// Re-exports for internal use
pub(crate) use session::Session;
