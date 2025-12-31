//! A/B Comparison Example
//!
//! Demonstrates using AgentRegistry to compare different agent configurations
//! on the same dataset, with LlmJudgeAgent for semantic evaluation.
//!
//! Run with:
//!   GEMINI_API_KEY=your_key cargo run -p gemicro-eval --example ab_comparison
//!
//! This example:
//! 1. Registers research agents with different configurations
//! 2. Registers an LlmJudgeAgent for semantic evaluation
//! 3. Runs the same evaluation dataset against each research agent
//! 4. Uses the judge agent to score semantic correctness
//! 5. Compares performance metrics

use futures_util::StreamExt;
use gemicro_core::{AgentContext, LlmClient, LlmConfig};
use gemicro_deep_research::{DeepResearchAgent, ResearchConfig};
use gemicro_eval::{EvalConfig, EvalHarness, EvalProgress, JsonFileDataset, Scorers};
use gemicro_judge::{JudgeConfig, JudgeInput, LlmJudgeAgent};
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

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              A/B Agent Comparison Demo                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Create a small test dataset
    let dataset = create_test_dataset()?;
    println!("📊 Dataset: {} questions", 3);
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

    // Register the LLM Judge as an agent (demonstrates judges are just agents)
    registry.register("llm_judge", || {
        Box::new(LlmJudgeAgent::new(JudgeConfig::default()))
    });

    println!("🤖 Agents registered:");
    println!("   - conservative: 2 sub-queries (faster)");
    println!("   - aggressive: 4-5 sub-queries (more thorough)");
    println!("   - llm_judge: semantic correctness evaluator");
    println!();

    // Create LLM client configuration
    let genai_client = rust_genai::Client::builder(api_key).build();
    let llm_config = LlmConfig::default()
        .with_timeout(Duration::from_secs(60))
        .with_max_tokens(1024)
        .with_temperature(0.7)
        .with_max_retries(2)
        .with_retry_base_delay_ms(1000);

    // Evaluation config: small concurrency, no retries for cleaner comparison
    let eval_config = EvalConfig::new().with_concurrency(1).with_max_retries(0);
    let harness = EvalHarness::new(eval_config);

    // Get research agent names (exclude the judge)
    let research_agents: Vec<_> = registry
        .list()
        .into_iter()
        .filter(|name| *name != "llm_judge")
        .collect();

    // Run A/B comparison
    println!("═══════════════════════════════════════════════════════════════");
    println!("                    Running A/B Comparison");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut results = Vec::new();

    for name in research_agents {
        println!("┌─────────────────────────────────────────────────────────────┐");
        println!("│ Testing: {:<51}│", name);
        println!("└─────────────────────────────────────────────────────────────┘");

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
                        let icon = if success { "✓" } else { "✗" };
                        println!("   [{}/{}] {}", completed, total, icon);
                    }
                    _ => {}
                },
            )
            .await?;

        // Run LLM judge on results using the registered agent
        println!("   Running LLM judge (via agent)...");
        let judge = registry.get("llm_judge").unwrap();

        for result in &mut summary.results {
            if let Some(predicted) = &result.predicted {
                // Create judge input and run the agent
                let input = JudgeInput::new(predicted, &result.ground_truth);
                let judge_llm = LlmClient::new(genai_client.clone(), llm_config.clone());
                let context = AgentContext::new(judge_llm);

                let stream = judge.execute(&input.to_query(), context);
                futures_util::pin_mut!(stream);

                // Collect the judge result
                while let Some(update_result) = stream.next().await {
                    match update_result {
                        Ok(update) => {
                            if update.event_type == "judge_result" {
                                let correct = update.data["correct"].as_bool().unwrap_or(false);
                                result.scores.insert(
                                    "llm_judge".to_string(),
                                    if correct { 1.0 } else { 0.0 },
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!("   ⚠ Judge error: {}", e);
                        }
                    }
                }
            }
        }

        // Recalculate averages after adding judge scores
        summary.recalculate_averages();

        println!();
        println!("   Results for '{}':", name);
        println!(
            "   ├── Succeeded: {}/{}",
            summary.succeeded, summary.total_questions
        );
        println!(
            "   ├── LLM Judge: {:.3}",
            summary.avg_score("llm_judge").unwrap_or(0.0)
        );
        println!(
            "   ├── Contains:  {:.3}",
            summary.avg_score("contains").unwrap_or(0.0)
        );
        println!(
            "   └── Duration: {:.1}s",
            summary.total_duration.as_secs_f64()
        );
        println!();

        results.push((name.to_string(), summary));
    }

    // Print comparison summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("                      Comparison Summary");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!(
        "{:<15} {:>10} {:>10} {:>10} {:>10}",
        "Agent", "Success", "LLM Judge", "Contains", "Time(s)"
    );
    println!("{}", "-".repeat(55));

    for (name, summary) in &results {
        println!(
            "{:<15} {:>10} {:>10.3} {:>10.3} {:>10.1}",
            name,
            format!("{}/{}", summary.succeeded, summary.total_questions),
            summary.avg_score("llm_judge").unwrap_or(0.0),
            summary.avg_score("contains").unwrap_or(0.0),
            summary.total_duration.as_secs_f64()
        );
    }
    println!();

    // Determine winner based on LLM judge
    if results.len() == 2 {
        let (name_a, summary_a) = &results[0];
        let (name_b, summary_b) = &results[1];
        let judge_a = summary_a.avg_score("llm_judge").unwrap_or(0.0);
        let judge_b = summary_b.avg_score("llm_judge").unwrap_or(0.0);

        if (judge_a - judge_b).abs() < 0.01 {
            println!("🤝 Result: Tie (LLM Judge scores within 0.01)");
        } else if judge_a > judge_b {
            println!(
                "🏆 Winner: {} (LLM Judge: {:.3} vs {:.3})",
                name_a, judge_a, judge_b
            );
        } else {
            println!(
                "🏆 Winner: {} (LLM Judge: {:.3} vs {:.3})",
                name_b, judge_b, judge_a
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
