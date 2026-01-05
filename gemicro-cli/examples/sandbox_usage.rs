//! Example: Using --sandbox-path for file access restriction.
//!
//! This example demonstrates how the `--sandbox-path` CLI flag restricts
//! file operations to whitelisted directories, providing defense-in-depth
//! for agent execution.
//!
//! # Usage
//!
//! ## Restricting to a single directory
//!
//! ```bash
//! gemicro "Read the config file" \
//!     --agent developer \
//!     --sandbox-path /workspace
//! ```
//!
//! With this configuration:
//! - File reads/writes within `/workspace` are allowed
//! - File reads/writes to `/etc`, `/home`, etc. are blocked
//! - Path traversal attempts (`/workspace/../etc`) are blocked
//!
//! ## Restricting to multiple directories
//!
//! ```bash
//! gemicro "Set up the project" \
//!     --agent developer \
//!     --sandbox-path /workspace \
//!     --sandbox-path /tmp
//! ```
//!
//! ## Combining with interactive mode
//!
//! ```bash
//! gemicro --interactive \
//!     --agent developer \
//!     --sandbox-path /workspace \
//!     --sandbox-path /tmp
//! ```
//!
//! # How It Works
//!
//! 1. CLI parses `--sandbox-path` arguments and validates they exist and are directories
//! 2. Session builds a `PathSandbox` interceptor with the allowed paths
//! 3. The interceptor chain is attached to `AgentContext.interceptors`
//! 4. Agents (ToolAgent, DeveloperAgent) attach interceptors to their tool service
//! 5. Every file operation passes through the interceptor before execution
//!
//! # Security Model
//!
//! - **Whitelist-based**: Only explicitly allowed paths are accessible
//! - **Path canonicalization**: Symlinks and `..` are resolved before checking
//! - **Defense-in-depth**: Works alongside agent tool restrictions
//!
//! # Programmatic Example
//!
//! For programmatic usage (not via CLI), see the `sandboxed_tools` example
//! in `gemicro-path-sandbox`:
//!
//! ```bash
//! cargo run -p gemicro-path-sandbox --example sandboxed_tools
//! ```

fn main() {
    println!("This is a documentation-only example.");
    println!("Run the CLI directly with --sandbox-path to use this feature.");
    println!();
    println!("Example:");
    println!("  gemicro \"List files\" --agent developer --sandbox-path /workspace");
}
