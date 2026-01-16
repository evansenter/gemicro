//! Example: Multi-Turn Conversations with Server-Side Chaining
//!
//! This example demonstrates how gemicro uses the Gemini Interactions API
//! for efficient multi-turn conversations. Instead of re-sending conversation
//! history as text each turn (wasteful), gemicro uses server-side interaction
//! chaining via `with_previous_interaction()`.
//!
//! # How It Works
//!
//! 1. Each LLM response includes an `interaction_id`
//! 2. The Session tracks the last `interaction_id`
//! 3. Subsequent turns pass this ID via `AgentContext.previous_interaction_id`
//! 4. Agents use `with_previous_interaction(id)` in their request builder
//! 5. The Gemini API preserves conversation context server-side
//!
//! # CLI Usage (Interactive Mode)
//!
//! ```bash
//! # Start interactive session
//! gemicro --interactive --agent prompt_agent
//!
//! # First turn establishes context
//! > My favorite color is blue.
//! # (interaction_id captured internally)
//!
//! # Second turn references previous context (no re-sending needed)
//! > What's my favorite color?
//! # Model recalls "blue" via server-side chaining
//!
//! # Clear resets the chain
//! > /clear
//! > What's my favorite color?
//! # Model no longer knows (chain was reset)
//! ```
//!
//! # Benefits
//!
//! - **Token efficiency**: No re-sending of conversation history
//! - **Structured context**: Server preserves typed messages, not flattened text
//! - **Reliability**: API handles context truncation/summarization if needed
//!
//! # Programmatic Usage
//!
//! For programmatic multi-turn conversations, use `AgentContext.previous_interaction_id`:
//!
//! ```rust,ignore
//! use gemicro_core::{AgentContext, Agent, LlmClient};
//! use gemicro_prompt_agent::PromptAgent;
//! use futures_util::StreamExt;
//!
//! async fn multi_turn_example(agent: &PromptAgent, llm: LlmClient) {
//!     // First turn - no previous interaction
//!     let context1 = AgentContext::new(llm.clone());
//!     let mut stream1 = agent.execute("Remember: my name is Alice", context1);
//!
//!     // Collect events and extract interaction_id from final_result
//!     let mut interaction_id = None;
//!     while let Some(Ok(event)) = stream1.next().await {
//!         if event.event_type == "final_result" {
//!             if let Some(id) = event.data.get("interaction_id") {
//!                 interaction_id = id.as_str().map(String::from);
//!             }
//!         }
//!     }
//!
//!     // Second turn - chain from previous interaction
//!     let context2 = AgentContext::new(llm)
//!         .with_previous_interaction(interaction_id.unwrap());
//!     let mut stream2 = agent.execute("What's my name?", context2);
//!     // Model recalls "Alice" via server-side chaining
//! }
//! ```
//!
//! # API Details
//!
//! Server-side chaining uses genai-rs 0.7.0's interaction builder:
//!
//! ```rust,ignore
//! // First turn (no chaining)
//! let request = client.interaction()
//!     .with_model("gemini-3.0-flash-preview")
//!     .with_system_instruction(system_prompt)
//!     .with_text(query)
//!     .with_store_enabled()  // Enable for continuation
//!     .build()?;
//!
//! // Subsequent turns (with chaining)
//! let request = client.interaction()
//!     .with_model("gemini-3.0-flash-preview")
//!     .with_system_instruction(system_prompt)  // Re-sent each turn
//!     .with_text(query)
//!     .with_previous_interaction(&interaction_id)  // Chain to previous
//!     .with_store_enabled()
//!     .build()?;
//! ```
//!
//! Note: System instructions, tools, and model must be re-sent each turn.
//! Only conversation history is preserved server-side.
//!
//! # Run the CLI
//!
//! ```bash
//! # Start interactive mode
//! GEMINI_API_KEY=your_key gemicro --interactive --agent prompt_agent
//! ```

fn main() {
    println!("Multi-Turn Conversation Example");
    println!("================================\n");

    println!("This is a documentation example. Run the CLI in interactive mode:\n");
    println!("  GEMINI_API_KEY=your_key gemicro --interactive --agent prompt_agent\n");

    println!("In interactive mode:");
    println!("  1. Enter a message establishing context (e.g., 'My name is Alice')");
    println!("  2. Ask a follow-up question (e.g., 'What's my name?')");
    println!("  3. The model recalls context via server-side chaining");
    println!("  4. Use /clear to reset the conversation chain\n");

    println!("The interaction_id is automatically tracked in the Session struct.");
    println!("Each AgentContext receives previous_interaction_id from the Session.");
}
