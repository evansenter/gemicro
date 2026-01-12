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
//!   GEMINI_API_KEY=your_key cargo run -p gemicro-critique-agent --example critique_demo

use gemicro_core::{AgentContext, LlmClient, LlmConfig};
use gemicro_critique_agent::{
    CritiqueAgent, CritiqueAgentConfig, CritiqueContext, CritiqueCriteria, CritiqueFinding,
    CritiqueInput, CritiqueVerdict,
};
use std::env;
use std::time::Duration;

fn create_llm_client(api_key: &str) -> LlmClient {
    let genai_client = genai_rs::Client::builder(api_key.to_string())
        .build()
        .unwrap();
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
    let agent = CritiqueAgent::new(CritiqueAgentConfig::default())?;

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
        let input = CritiqueInput::new(predicted).with_criteria(CritiqueCriteria::GroundTruth {
            expected: expected.to_string(),
        });

        print!("  {} ... ", description);
        let output = agent.critique(&input, context).await?;
        println!(
            "{} (confidence: {:.0}%)",
            verdict_display(output.verdict),
            output.confidence * 100.0
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

    for (code, description) in [
        (good_code, "bcrypt implementation"),
        (bad_code, "MD5 implementation"),
    ] {
        let context = AgentContext::new(create_llm_client(&api_key));
        let input = CritiqueInput::new(code)
            .with_context(
                CritiqueContext::new()
                    .with_query("Implement password hashing")
                    .with_agent("prompt_agent"),
            )
            .with_criteria(CritiqueCriteria::Specification { spec: spec.into() });

        print!("  {} ... ", description);
        let output = agent.critique(&input, context).await?;
        println!(
            "{} (retry: {})",
            verdict_display(output.verdict),
            if output.should_retry { "yes" } else { "no" }
        );
        if !output.suggestions.is_empty() {
            println!("    Suggestions: {}", output.suggestions.join("; "));
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
    let output = agent.critique(&input, context).await?;
    println!(
        "{} ({} findings)",
        verdict_display(output.verdict),
        output.findings.len()
    );
    print_findings(&output.findings);
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
    let input =
        CritiqueInput::new(code_with_issues).with_criteria(CritiqueCriteria::CodeConventions {
            conventions: conventions.into(),
        });

    print!("  Code style review ... ");
    let output = agent.critique(&input, context).await?;
    println!("{}", verdict_display(output.verdict));
    print_findings(&output.findings);
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
    let output = agent.critique(&input, context).await?;
    println!("{}", verdict_display(output.verdict));
    if !output.suggestions.is_empty() {
        for suggestion in &output.suggestions {
            println!("    - {}", suggestion);
        }
    }
    println!();

    // ============================================================
    // Demo 6: Using Convenience Methods
    // ============================================================
    println!("------------------------------------------------------------");
    println!("6. CONVENIENCE METHODS");
    println!("   (Using is_passing() and to_score())");
    println!("------------------------------------------------------------");
    println!();

    let context = AgentContext::new(create_llm_client(&api_key));
    let input = CritiqueInput::new("42").with_criteria(CritiqueCriteria::GroundTruth {
        expected: "The answer is 42".into(),
    });

    let output = agent.critique(&input, context).await?;
    println!("  Verdict: {}", output.verdict);
    println!("  is_passing(): {}", output.verdict.is_passing());
    println!("  to_score(): {:.2}", output.verdict.to_score());
    println!("  output.to_score(): {:.2}", output.to_score());
    println!();

    println!("============================================================");
    println!("Demo complete!");
    println!("============================================================");

    Ok(())
}

fn print_findings(findings: &[CritiqueFinding]) {
    for finding in findings {
        println!("    - [{}] {}", finding.severity, finding.issue);
        if let Some(suggestion) = &finding.suggestion {
            println!("      Fix: {}", suggestion);
        }
    }
}

fn verdict_display(verdict: CritiqueVerdict) -> &'static str {
    match verdict {
        CritiqueVerdict::Pass => "PASS",
        CritiqueVerdict::PassWithWarnings => "PASS (warnings)",
        CritiqueVerdict::NeedsRevision => "NEEDS REVISION",
        CritiqueVerdict::Reject => "REJECT",
    }
}
