//! Streaming Tool Agent Example
//!
//! Demonstrates using create_stream_with_auto_functions() with:
//! - Real-time incremental text updates via AutoFunctionStreamChunk
//! - Automatic function execution (via ToolService)
//! - Hooks for logging/validation
//! - Confirmation handlers for dangerous operations
//!
//! Run with:
//! ```bash
//! GEMINI_API_KEY=your-key cargo run -p gemicro-tool-agent --example streaming_tool_agent
//! ```

use futures_util::StreamExt;
use gemicro_audit_log::AuditLog;
use gemicro_core::tool::{AutoApprove, GemicroToolService, HookRegistry};
use gemicro_core::{ToolSet, MODEL};
use gemicro_tool_agent::tools::default_registry;
use rust_genai::AutoFunctionStreamChunk;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let api_key =
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY environment variable must be set");

    println!("╭─────────────────────────────────────────────╮");
    println!("│     Streaming Tool Agent with Hooks        │");
    println!("╰─────────────────────────────────────────────╯\n");

    // Set up hooks for audit logging
    let hooks = Arc::new(HookRegistry::new().with_hook(AuditLog));

    // Create tool registry and service
    let registry = Arc::new(default_registry());
    let service = GemicroToolService::new(Arc::clone(&registry))
        .with_filter(ToolSet::Specific(vec!["calculator".into()]))
        .with_hooks(hooks)
        .with_confirmation_handler(Arc::new(AutoApprove)); // Auto-approve for demo

    println!("🔧 Tools available: calculator");
    println!("🔐 Hooks enabled: AuditLog");
    println!("✓ Confirmation: AutoApprove\n");

    // Create streaming interaction using rust-genai's streaming auto-functions
    let query = "What is the square root of 256, plus 17 squared?";
    println!("❓ Query: {}\n", query);
    println!("📡 Streaming response:\n");
    println!("─────────────────────────────────────────────");

    let genai_client = rust_genai::Client::builder(api_key).build();
    let mut stream = genai_client
        .interaction()
        .with_model(MODEL)
        .with_system_instruction(
            "You are a math assistant. Use the calculator tool to solve problems. \
            Show your work and provide clear explanations.",
        )
        .with_text(query)
        .with_tool_service(Arc::new(service))
        .create_stream_with_auto_functions();

    let mut text_parts = Vec::new();
    let mut chunk_count = 0;
    let mut function_execution_count = 0;

    // Consume the stream and display incremental updates
    while let Some(result) = stream.next().await {
        match result {
            Ok(chunk) => match chunk {
                AutoFunctionStreamChunk::Delta(content) => {
                    if let Some(text) = content.text() {
                        print!("{}", text);
                        std::io::Write::flush(&mut std::io::stdout())?;
                        text_parts.push(text.to_string());
                        chunk_count += 1;
                    }
                }
                AutoFunctionStreamChunk::ExecutingFunctions(_) => {
                    println!("\n🔄 [Executing functions with hooks...]");
                    function_execution_count += 1;
                }
                AutoFunctionStreamChunk::FunctionResults(_) => {
                    println!("✅ [Function execution complete]");
                }
                AutoFunctionStreamChunk::Complete(response) => {
                    println!("\n─────────────────────────────────────────────");
                    println!("\n📊 Statistics:");
                    if let Some(usage) = response.usage {
                        println!("   Usage: {:?}", usage);
                    }
                    println!("   Text chunks received: {}", chunk_count);
                    println!("   Function executions: {}", function_execution_count);
                }
                _ => {
                    println!("[Unknown AutoFunctionStreamChunk variant]");
                }
            },
            Err(e) => {
                eprintln!("\n❌ Streaming error: {:?}", e);
                return Err(e.into());
            }
        }
    }

    let full_text: String = text_parts.join("");
    println!("\n✅ Stream completed");
    println!("\n📝 Full response ({} bytes):", full_text.len());
    println!("{}", full_text);

    println!("\n💡 Key Features Demonstrated:");
    println!("   • Real-time incremental text updates (streaming)");
    println!("   • Automatic function execution via ToolService");
    println!(
        "   • AuditLog hook intercepted {} tool call(s)",
        function_execution_count
    );
    println!("   • Confirmation handler approved operations");
    println!("   • Tools were filtered to calculator only");
    println!("   • ExecutingFunctions and FunctionResults events");

    Ok(())
}
