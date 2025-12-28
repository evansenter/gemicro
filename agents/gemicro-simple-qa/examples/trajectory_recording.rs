//! Trajectory Recording Example
//!
//! Demonstrates how to record agent executions for offline replay and evaluation.
//!
//! Run with:
//!   GEMINI_API_KEY=your_key cargo run -p gemicro-simple-qa --example trajectory_recording
//!
//! This example:
//! 1. Records a trajectory during agent execution
//! 2. Saves it to a JSON file
//! 3. Loads and inspects the trajectory
//! 4. Demonstrates replay with MockLlmClient

use gemicro_core::{LlmConfig, LlmRequest, MockLlmClient, Trajectory};
use gemicro_runner::AgentRunner;
use gemicro_simple_qa::{SimpleQaAgent, SimpleQaConfig};
use serde_json::json;
use std::env;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key = env::var("GEMINI_API_KEY").expect(
        "GEMINI_API_KEY environment variable not set.\n\
         Set it with: export GEMINI_API_KEY=your_key",
    );

    println!("======================================");
    println!("  Trajectory Recording Example");
    println!("======================================\n");

    // Create agent
    let config = SimpleQaConfig {
        timeout: Duration::from_secs(30),
        ..Default::default()
    };
    let agent = SimpleQaAgent::new(config)?;

    // Create LLM client with recording enabled
    let genai_client = rust_genai::Client::builder(api_key).build();
    let llm_config = LlmConfig {
        timeout: Duration::from_secs(60),
        max_tokens: 1024,
        temperature: 0.7,
        ..Default::default()
    };

    // Execute agent and capture trajectory
    println!("Step 1: Recording agent execution...\n");
    let runner = AgentRunner::new();
    let query = "What is the capital of France?";

    let (metrics, trajectory) = runner
        .execute_with_trajectory(
            &agent,
            query,
            json!({
                "timeout_secs": 30,
                "system_prompt": "default"
            }),
            genai_client,
            llm_config,
        )
        .await?;

    println!("  Query: {}", query);
    println!("  Answer: {:?}", metrics.final_answer);
    println!("  Duration: {:?}", metrics.total_duration);
    println!("  Tokens: {}", metrics.total_tokens);
    println!();

    // Save trajectory
    let temp_dir = std::env::temp_dir();
    let trajectory_path = temp_dir.join(format!("trajectory_{}.json", trajectory.id));
    println!("Step 2: Saving trajectory...\n");
    trajectory.save(&trajectory_path)?;
    println!("  Saved to: {}", trajectory_path.display());
    println!();

    // Load and inspect trajectory
    println!("Step 3: Loading and inspecting trajectory...\n");
    let loaded = Trajectory::load(&trajectory_path)?;

    println!("  Trajectory ID: {}", loaded.id);
    println!("  Agent: {}", loaded.agent_name);
    println!("  Query: {}", loaded.query);
    println!("  LLM Steps: {}", loaded.step_count());
    println!("  Events: {}", loaded.events.len());
    println!("  Schema Version: {}", loaded.metadata.schema_version);
    println!();

    // Show step details
    println!("  Step Details:");
    for (i, step) in loaded.steps.iter().enumerate() {
        println!("    [{}] Phase: {}", i + 1, step.phase);
        println!("        Duration: {}ms", step.duration_ms);
        println!(
            "        Prompt: {}...",
            step.request
                .prompt
                .chars()
                .take(50)
                .collect::<String>()
                .replace('\n', " ")
        );
    }
    println!();

    // Demonstrate replay with MockLlmClient
    println!("Step 4: Replaying with MockLlmClient...\n");
    let mock = MockLlmClient::from_trajectory(&loaded);

    // Replay returns the same responses in order
    let request = LlmRequest::new("Any prompt - the recorded response is returned");
    let response = mock.generate(request).await?;

    // Extract text from rust-genai's InteractionResponse structure
    // The response has outputs: [{type: "thought"}, {type: "text", text: "..."}]
    let text = response
        .get("outputs")
        .and_then(|outputs| outputs.as_array())
        .and_then(|arr| {
            arr.iter().find_map(|output| {
                if output.get("type").and_then(|t| t.as_str()) == Some("text") {
                    output.get("text").and_then(|t| t.as_str())
                } else {
                    None
                }
            })
        });

    if let Some(text) = text {
        println!(
            "  Replayed response: {}...",
            text.chars().take(100).collect::<String>()
        );
    }

    println!("  Mock exhausted: {}", mock.is_exhausted());
    println!();

    // Cleanup
    std::fs::remove_file(&trajectory_path)?;
    println!("  Cleaned up temporary file");
    println!();

    println!("======================================");
    println!("  Example Complete!");
    println!("======================================");
    println!("\nTrajectory recording enables:");
    println!("  - Offline testing without API calls");
    println!("  - Building evaluation datasets from production");
    println!("  - Debugging with exact request/response inspection");

    Ok(())
}
