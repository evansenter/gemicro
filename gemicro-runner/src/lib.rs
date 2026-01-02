//! # Gemicro Runner
//!
//! Headless execution runtime for gemicro agents.
//!
//! This crate provides terminal-agnostic state tracking and orchestration,
//! enabling programmatic agent execution without display dependencies.
//!
//! ## Use Cases
//!
//! - **Headless execution**: Run agents in servers, batch jobs, or tests
//! - **Custom rendering**: Build alternative display backends (GUI, web, etc.)
//! - **Evaluation frameworks**: Benchmark prompts and agent configurations
//! - **Metrics collection**: Capture structured execution data for analysis
//!
//! ## Architecture
//!
//! ```text
//! gemicro-core (agents, events)
//!     ↓
//! gemicro-runner (execution state, metrics, runner)  ← this crate
//!     ↓
//! gemicro-cli (terminal rendering)
//! ```
//!
//! ## Example
//!
//! ```text
//! // Requires an agent crate like gemicro-deep-research
//! use gemicro_runner::AgentRunner;
//! use gemicro_core::{AgentContext, LlmClient, LlmConfig};
//! use gemicro_deep_research::{DeepResearchAgent, ResearchConfig};
//!
//! async fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create agent
//!     let agent = DeepResearchAgent::new(ResearchConfig::default())?;
//!
//!     // Create context
//!     let genai_client = rust_genai::Client::builder("api-key".to_string()).build();
//!     let llm = LlmClient::new(genai_client, LlmConfig::default());
//!     let context = AgentContext::new(llm);
//!
//!     // Execute headlessly
//!     let runner = AgentRunner::new();
//!     let metrics = runner.execute(&agent, "What is Rust?", context).await?;
//!
//!     println!("Duration: {:?}", metrics.total_duration);
//!     println!("Tokens: {}", metrics.total_tokens);
//!     println!("Answer: {:?}", metrics.final_answer);
//!     Ok(())
//! }
//! ```

pub mod metrics;
pub mod registry;
pub mod runner;
pub mod state;
pub mod utils;

// Re-export public API
pub use metrics::{ExecutionMetrics, StepTiming};
pub use registry::{AgentFactory, AgentRegistry};
pub use runner::{AgentRunner, EVENT_EXTERNAL};
pub use state::{phases, ExecutionState, ExecutionStep, StepStatus};
pub use utils::{first_sentence, format_duration, truncate};
