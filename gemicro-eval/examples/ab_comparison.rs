//! A/B Comparison Example
//!
//! Demonstrates using AgentRegistry to compare different agent configurations
//! on the same dataset, with CritiqueAgent for semantic evaluation.
//!
//! Run with:
//!   GEMINI_API_KEY=your_key cargo run -p gemicro-eval --example ab_comparison
//!
//! This example:
//! 1. Registers research agents with different configurations
//! 2. Registers a CritiqueAgent for semantic evaluation
//! 3. Runs the same evaluation dataset against each research agent
//! 4. Uses the critique agent to score semantic correctness
//! 5. Compares performance metrics

use gemicro_core::{AgentContext, LlmClient, LlmConfig};
use gemicro_critique::{CritiqueAgent, CritiqueConfig, CritiqueCriteria, CritiqueInput};
use gemicro_deep_research::{DeepResearchAgent, ResearchConfig};
use gemicro_eval::{EvalConfig, EvalHarness, EvalProgress, JsonFileDataset, Scorers};
use gemicro_runner::AgentRegistry;
use std::env;
use std::io::Write;
use std::time::Duration;
use tempfile::NamedTempFile;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    // Get API key
    let api_key = env::var("GEMINI_API_KEY").expect(
        "GEMINI_API_KEY environment variable not set.\n\
         Set it with: export GEMINI_API_KEY=your_key",
    );

    println!("=====================================================");
    println!("              A/B Agent Comparison Demo              ");
    println!("=====================================================");
    println!();

    // Create a small test dataset
    let dataset = create_test_dataset()?;
    println!("Dataset: {} questions", 3);
    println!();

    // Register agents with different configurations
    let mut registry = AgentRegistry::new();

    // Conservative: fewer sub-queries, faster but potentially less thorough
    registry.register("conservative", || {
        Box::new(
            DeepResearchAgent::new(
                ResearchConfig::default()
                    .with_min_sub_queries(2)
                    .with_max_sub_queries(2)
                    .with_total_timeout(Duration::from_secs(120)),
            )
            .expect("Invalid config"),
        )
    });

    // Aggressive: more sub-queries, slower but potentially more thorough
    registry.register("aggressive", || {
        Box::new(
            DeepResearchAgent::new(
                ResearchConfig::default()
                    .with_min_sub_queries(4)
                    .with_max_sub_queries(5)
                    .with_total_timeout(Duration::from_secs(180)),
            )
            .expect("Invalid config"),
        )
    });

    // Register the Critique agent for evaluation (demonstrates agents are composable)
    registry.register("critique", || {
        Box::new(CritiqueAgent::new(CritiqueConfig::default()).expect("Invalid config"))
    });

    println!("Agents registered:");
    println!("   - conservative: 2 sub-queries (faster)");
    println!("   - aggressive: 4-5 sub-queries (more thorough)");
    println!("   - critique: semantic correctness evaluator");
    println!();

    // Create LLM client configuration
    let genai_client = rust_genai::Client::builder(api_key).build()?;
    let llm_config = LlmConfig::default()
        .with_timeout(Duration::from_secs(60))
        .with_max_tokens(1024)
        .with_temperature(0.7)
        .with_max_retries(2)
        .with_retry_base_delay_ms(1000);

    // Evaluation config: small concurrency, no retries for cleaner comparison
    let eval_config = EvalConfig::new().with_concurrency(1).with_max_retries(0);
    let harness = EvalHarness::new(eval_config);

    // Get research agent names (exclude the critique agent)
    let research_agents: Vec<_> = registry
        .list()
        .into_iter()
        .filter(|name| *name != "critique")
        .collect();

    // Run A/B comparison
    println!("===================================================================");
    println!("                    Running A/B Comparison");
    println!("===================================================================");
    println!();

    let mut results = Vec::new();

    for name in research_agents {
        println!("-------------------------------------------------------------");
        println!(" Testing: {}", name);
        println!("-------------------------------------------------------------");

        let agent = registry.get(name).unwrap();
        let llm = LlmClient::new(genai_client.clone(), llm_config.clone());

        let mut summary = harness
            .evaluate_with_progress(
                agent.as_ref(),
                &dataset,
                None,
                Scorers::default(),
                llm,
                |progress| match progress {
                    EvalProgress::Started { total } => {
                        println!("   Starting {} questions...", total);
                    }
                    EvalProgress::QuestionCompleted {
                        completed,
                        total,
                        success,
                    } => {
                        let icon = if success { "[OK]" } else { "[FAIL]" };
                        println!("   [{}/{}] {}", completed, total, icon);
                    }
                    _ => {}
                },
            )
            .await?;

        // Run critique agent on results
        // Note: We create CritiqueAgent directly here to use the typed critique() helper.
        // The registry registration above demonstrates agents are composable/registrable.
        println!("   Running critique agent...");
        let critique_agent =
            CritiqueAgent::new(CritiqueConfig::default()).expect("default config is valid");

        for result in &mut summary.results {
            if let Some(predicted) = &result.predicted {
                // Create critique input with ground truth criteria
                let input =
                    CritiqueInput::new(predicted).with_criteria(CritiqueCriteria::GroundTruth {
                        expected: result.ground_truth.clone(),
                    });
                let critique_llm = LlmClient::new(genai_client.clone(), llm_config.clone());
                let context = AgentContext::new(critique_llm);

                // Use the critique() helper for cleaner typed output
                match critique_agent.critique(&input, context).await {
                    Ok(output) => {
                        result
                            .scores
                            .insert("critique".to_string(), output.to_score());
                    }
                    Err(e) => {
                        eprintln!("   Warning: Critique error: {}", e);
                    }
                }
            }
        }

        // Recalculate averages after adding critique scores
        summary.recalculate_averages();

        println!();
        println!("   Results for '{}':", name);
        println!(
            "   - Succeeded: {}/{}",
            summary.succeeded, summary.total_questions
        );
        println!(
            "   - Critique: {:.3}",
            summary.avg_score("critique").unwrap_or(0.0)
        );
        println!(
            "   - Contains:  {:.3}",
            summary.avg_score("contains").unwrap_or(0.0)
        );
        println!(
            "   - Duration: {:.1}s",
            summary.total_duration.as_secs_f64()
        );
        println!();

        results.push((name.to_string(), summary));
    }

    // Print comparison summary
    println!("===================================================================");
    println!("                      Comparison Summary");
    println!("===================================================================");
    println!();
    println!(
        "{:<15} {:>10} {:>10} {:>10} {:>10}",
        "Agent", "Success", "Critique", "Contains", "Time(s)"
    );
    println!("{}", "-".repeat(55));

    for (name, summary) in &results {
        println!(
            "{:<15} {:>10} {:>10.3} {:>10.3} {:>10.1}",
            name,
            format!("{}/{}", summary.succeeded, summary.total_questions),
            summary.avg_score("critique").unwrap_or(0.0),
            summary.avg_score("contains").unwrap_or(0.0),
            summary.total_duration.as_secs_f64()
        );
    }
    println!();

    // Determine winner based on critique score
    if results.len() == 2 {
        let (name_a, summary_a) = &results[0];
        let (name_b, summary_b) = &results[1];
        let score_a = summary_a.avg_score("critique").unwrap_or(0.0);
        let score_b = summary_b.avg_score("critique").unwrap_or(0.0);

        if (score_a - score_b).abs() < 0.01 {
            println!("Result: Tie (Critique scores within 0.01)");
        } else if score_a > score_b {
            println!(
                "Winner: {} (Critique: {:.3} vs {:.3})",
                name_a, score_a, score_b
            );
        } else {
            println!(
                "Winner: {} (Critique: {:.3} vs {:.3})",
                name_b, score_b, score_a
            );
        }
    }

    Ok(())
}

/// Create a small test dataset for the demo
fn create_test_dataset() -> Result<JsonFileDataset, Box<dyn std::error::Error>> {
    let questions = r#"[
        {
            "id": "q1",
            "question": "What is the capital of France?",
            "ground_truth": "Paris"
        },
        {
            "id": "q2",
            "question": "Who wrote Romeo and Juliet?",
            "ground_truth": "William Shakespeare"
        },
        {
            "id": "q3",
            "question": "What is the largest planet in our solar system?",
            "ground_truth": "Jupiter"
        }
    ]"#;

    let mut file = NamedTempFile::new()?;
    file.write_all(questions.as_bytes())?;

    // Keep the file alive by leaking it (it will be cleaned up on process exit)
    let path = file.into_temp_path().keep()?;

    Ok(JsonFileDataset::new(path))
}
