//! CritiqueAgent Demo
//!
//! Demonstrates all CritiqueAgent capabilities:
//! - Ground truth validation (semantic comparison of answers)
//! - Specification checking
//! - Checklist verification
//! - Code conventions review
//! - Custom criteria evaluation
//!
//! Run with:
//!   GEMINI_API_KEY=your_key cargo run -p gemicro-critique --example critique_demo

use futures_util::StreamExt;
use gemicro_core::{AgentContext, LlmClient, LlmConfig};
use gemicro_critique::{
    CritiqueAgent, CritiqueConfig, CritiqueContext, CritiqueCriteria, CritiqueInput,
    CritiqueVerdict,
};
use std::env;
use std::time::Duration;

fn create_llm_client(api_key: &str) -> LlmClient {
    let genai_client = rust_genai::Client::builder(api_key.to_string()).build();
    LlmClient::new(
        genai_client,
        LlmConfig::default()
            .with_timeout(Duration::from_secs(30))
            .with_temperature(0.0),
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let api_key = env::var("GEMINI_API_KEY").expect(
        "GEMINI_API_KEY environment variable not set.\n\
         Set it with: export GEMINI_API_KEY=your_key",
    );

    println!("============================================================");
    println!("              CritiqueAgent Demo                            ");
    println!("============================================================");
    println!();

    // Create the critique agent
    let agent = CritiqueAgent::new(CritiqueConfig::default())?;

    // ============================================================
    // Demo 1: Ground Truth Validation
    // ============================================================
    println!("------------------------------------------------------------");
    println!("1. GROUND TRUTH VALIDATION");
    println!("   (Compare predicted answers to expected answers)");
    println!("------------------------------------------------------------");
    println!();

    let ground_truth_tests = vec![
        ("Paris", "Paris is the capital of France", "Exact match"),
        (
            "The answer is forty-two",
            "42",
            "Semantic equivalence (number)",
        ),
        ("London", "Paris", "Incorrect answer"),
    ];

    for (predicted, expected, description) in ground_truth_tests {
        let context = AgentContext::new(create_llm_client(&api_key));
        let input =
            CritiqueInput::new(predicted).with_criteria(CritiqueCriteria::GroundTruth {
                expected: expected.to_string(),
            });

        print!("  {} ... ", description);
        let result = run_critique(&agent, &input, context).await?;
        println!(
            "{} (confidence: {:.0}%)",
            verdict_icon(result.verdict),
            result.confidence * 100.0
        );
    }
    println!();

    // ============================================================
    // Demo 2: Specification Checking
    // ============================================================
    println!("------------------------------------------------------------");
    println!("2. SPECIFICATION CHECKING");
    println!("   (Verify output meets requirements)");
    println!("------------------------------------------------------------");
    println!();

    let good_code = r#"
fn hash_password(password: &str) -> String {
    bcrypt::hash(password, bcrypt::DEFAULT_COST).unwrap()
}
"#;

    let bad_code = r#"
fn hash_password(password: &str) -> String {
    format!("{:x}", md5::compute(password))
}
"#;

    let spec = "Password hashing MUST use bcrypt with a cost factor of at least 10. \
                Never use MD5 or SHA1 for password storage.";

    for (code, description) in [(good_code, "bcrypt implementation"), (bad_code, "MD5 implementation")] {
        let context = AgentContext::new(create_llm_client(&api_key));
        let input = CritiqueInput::new(code)
            .with_context(
                CritiqueContext::new()
                    .with_query("Implement password hashing")
                    .with_agent("tool_agent"),
            )
            .with_criteria(CritiqueCriteria::Specification { spec: spec.into() });

        print!("  {} ... ", description);
        let result = run_critique(&agent, &input, context).await?;
        println!(
            "{} (retry: {})",
            verdict_icon(result.verdict),
            if result.should_retry { "yes" } else { "no" }
        );
        if !result.suggestions.is_empty() {
            println!("    Suggestions: {}", result.suggestions.join("; "));
        }
    }
    println!();

    // ============================================================
    // Demo 3: Checklist Verification
    // ============================================================
    println!("------------------------------------------------------------");
    println!("3. CHECKLIST VERIFICATION");
    println!("   (Verify multiple criteria are met)");
    println!("------------------------------------------------------------");
    println!();

    let api_response = r#"
{
    "id": "user_123",
    "email": "user@example.com",
    "created_at": "2024-01-15T10:30:00Z"
}
"#;

    let checklist = vec![
        "Response is valid JSON".to_string(),
        "Contains user ID field".to_string(),
        "Contains email field".to_string(),
        "Contains timestamp field".to_string(),
        "No sensitive data exposed (password, SSN)".to_string(),
    ];

    let context = AgentContext::new(create_llm_client(&api_key));
    let input = CritiqueInput::new(api_response)
        .with_criteria(CritiqueCriteria::Checklist { items: checklist });

    print!("  API response validation ... ");
    let result = run_critique(&agent, &input, context).await?;
    println!(
        "{} ({} findings)",
        verdict_icon(result.verdict),
        result.findings.len()
    );
    for finding in &result.findings {
        println!("    - [{}] {}", finding.severity, finding.issue);
    }
    println!();

    // ============================================================
    // Demo 4: Code Conventions
    // ============================================================
    println!("------------------------------------------------------------");
    println!("4. CODE CONVENTIONS");
    println!("   (Validate against style guide / CLAUDE.md)");
    println!("------------------------------------------------------------");
    println!();

    let code_with_issues = r#"
// Helper function to do stuff
pub fn DoSomething(x: i32) -> i32 {
    let MAGIC_NUMBER = 42;
    x + MAGIC_NUMBER
}
"#;

    let conventions = r#"
Rust Code Conventions:
- Function names use snake_case (not PascalCase)
- Constants use SCREAMING_SNAKE_CASE and must be defined as const, not let
- Avoid vague names like "do_something" - be specific
- Comments should explain "why", not "what"
"#;

    let context = AgentContext::new(create_llm_client(&api_key));
    let input = CritiqueInput::new(code_with_issues)
        .with_criteria(CritiqueCriteria::CodeConventions {
            conventions: conventions.into(),
        });

    print!("  Code style review ... ");
    let result = run_critique(&agent, &input, context).await?;
    println!("{}", verdict_icon(result.verdict));
    for finding in &result.findings {
        println!("    - [{}] {}", finding.severity, finding.issue);
        if let Some(suggestion) = &finding.suggestion {
            println!("      Fix: {}", suggestion);
        }
    }
    println!();

    // ============================================================
    // Demo 5: Custom Criteria
    // ============================================================
    println!("------------------------------------------------------------");
    println!("5. CUSTOM CRITERIA");
    println!("   (Flexible natural language evaluation)");
    println!("------------------------------------------------------------");
    println!();

    let essay = "Climate change is a pressing global issue that requires \
                 immediate action. Scientists agree that human activities \
                 are the primary cause of rising temperatures.";

    let context = AgentContext::new(create_llm_client(&api_key));
    let input = CritiqueInput::new(essay).with_criteria(CritiqueCriteria::Custom {
        description: "Check if the text is well-structured, makes a clear argument, \
                      and uses appropriate academic language. Flag any unsupported claims."
            .into(),
    });

    print!("  Essay evaluation ... ");
    let result = run_critique(&agent, &input, context).await?;
    println!("{}", verdict_icon(result.verdict));
    if !result.suggestions.is_empty() {
        for suggestion in &result.suggestions {
            println!("    - {}", suggestion);
        }
    }
    println!();

    println!("============================================================");
    println!("Demo complete!");
    println!("============================================================");

    Ok(())
}

/// Run critique and extract result
async fn run_critique(
    agent: &CritiqueAgent,
    input: &CritiqueInput,
    context: AgentContext,
) -> Result<CritiqueResult, Box<dyn std::error::Error>> {
    use gemicro_core::Agent;

    let stream = agent.execute(&input.to_query(), context);
    futures_util::pin_mut!(stream);

    let mut result = CritiqueResult::default();

    while let Some(update) = stream.next().await {
        let update = update?;
        if update.event_type == "critique_result" {
            result.verdict = match update.data["verdict"].as_str() {
                Some("Pass") => CritiqueVerdict::Pass,
                Some("PassWithWarnings") => CritiqueVerdict::PassWithWarnings,
                Some("NeedsRevision") => CritiqueVerdict::NeedsRevision,
                Some("Reject") => CritiqueVerdict::Reject,
                _ => CritiqueVerdict::Reject,
            };
            result.confidence = update.data["confidence"].as_f64().unwrap_or(0.0) as f32;
            result.should_retry = update.data["should_retry"].as_bool().unwrap_or(false);
        } else if update.event_type == "final_result" {
            // Parse full output from final_result
            if let Some(answer) = update.data.get("answer") {
                if let Some(findings) = answer.get("findings").and_then(|f| f.as_array()) {
                    for f in findings {
                        result.findings.push(Finding {
                            issue: f["issue"].as_str().unwrap_or("").to_string(),
                            severity: f["severity"].as_str().unwrap_or("Info").to_string(),
                            suggestion: f["suggestion"].as_str().map(String::from),
                        });
                    }
                }
                if let Some(suggestions) = answer.get("suggestions").and_then(|s| s.as_array()) {
                    for s in suggestions {
                        if let Some(text) = s.as_str() {
                            result.suggestions.push(text.to_string());
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}

struct CritiqueResult {
    verdict: CritiqueVerdict,
    confidence: f32,
    should_retry: bool,
    findings: Vec<Finding>,
    suggestions: Vec<String>,
}

impl Default for CritiqueResult {
    fn default() -> Self {
        Self {
            verdict: CritiqueVerdict::Reject, // Default to most conservative
            confidence: 0.0,
            should_retry: false,
            findings: Vec::new(),
            suggestions: Vec::new(),
        }
    }
}

struct Finding {
    issue: String,
    severity: String,
    suggestion: Option<String>,
}

fn verdict_icon(verdict: CritiqueVerdict) -> &'static str {
    match verdict {
        CritiqueVerdict::Pass => "PASS",
        CritiqueVerdict::PassWithWarnings => "PASS (warnings)",
        CritiqueVerdict::NeedsRevision => "NEEDS REVISION",
        CritiqueVerdict::Reject => "REJECT",
    }
}
