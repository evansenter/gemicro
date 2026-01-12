//! Critique Agent - Generalized Output Validation and Feedback
//!
//! A versatile agent that validates any content against configurable criteria
//! and provides actionable feedback. Enables inference-time compute scaling
//! through validation loops.
//!
//! # Design Philosophy
//!
//! Following [Evergreen spec](https://github.com/google-deepmind/evergreen-spec) principles:
//! - Flexible criteria types for different validation needs
//! - Rich output with verdicts, findings, and suggestions
//! - Supports retry loops with actionable feedback
//!
//! # Usage Patterns
//!
//! ## Ground Truth Validation (semantic comparison of answers)
//!
//! ```no_run
//! use gemicro_critique_agent::{CritiqueAgent, CritiqueAgentConfig, CritiqueInput, CritiqueCriteria};
//!
//! let input = CritiqueInput::new("Paris")
//!     .with_criteria(CritiqueCriteria::GroundTruth {
//!         expected: "Paris is the capital of France".into()
//!     });
//! ```
//!
//! ## Specification Checking
//!
//! ```no_run
//! use gemicro_critique_agent::{CritiqueInput, CritiqueCriteria, CritiqueContext};
//! use serde_json::json;
//!
//! # let generated_code = "fn hash_password(pwd: &str) { bcrypt::hash(pwd); }";
//! let input = CritiqueInput::new(generated_code)
//!     .with_context(CritiqueContext::new()
//!         .with_query("Implement user authentication")
//!         .with_agent("prompt_agent"))
//!     .with_criteria(CritiqueCriteria::Specification {
//!         spec: "Must use bcrypt for password hashing".into()
//!     });
//! ```
//!
//! ## Code Conventions (for validating against CLAUDE.md)
//!
//! ```no_run
//! use gemicro_critique_agent::{CritiqueInput, CritiqueCriteria};
//!
//! # let proposed_changes = "fn my_function() -> i32 { 42 }";
//! let input = CritiqueInput::new(proposed_changes)
//!     .with_criteria(CritiqueCriteria::CodeConventions {
//!         conventions: "Follow Rust naming conventions...".into()
//!     });
//! ```

use gemicro_core::{
    remaining_time, timeout_error, truncate, with_timeout_and_cancellation, Agent, AgentContext,
    AgentError, AgentStream, AgentUpdate, ResultMetadata,
};

use async_stream::try_stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

// ============================================================================
// Event Type Constants
// ============================================================================

/// Event emitted when critique evaluation starts.
pub const EVENT_CRITIQUE_STARTED: &str = "critique_started";

/// Event emitted when critique produces its result.
pub const EVENT_CRITIQUE_RESULT: &str = "critique_result";

// ============================================================================
// Input Types
// ============================================================================

/// What to critique.
///
/// Contains the content to validate along with optional context and criteria.
///
/// # Example
///
/// ```
/// use gemicro_critique_agent::{CritiqueInput, CritiqueCriteria};
///
/// let input = CritiqueInput::new("The answer is 42")
///     .with_criteria(CritiqueCriteria::GroundTruth {
///         expected: "42".into()
///     });
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CritiqueInput {
    /// The content to critique.
    pub content: String,

    /// Context about what produced this content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<CritiqueContext>,

    /// What to evaluate against.
    pub criteria: CritiqueCriteria,
}

impl CritiqueInput {
    /// Create a new critique input with the content to evaluate.
    ///
    /// Uses default criteria (Custom with "Evaluate for correctness and quality").
    /// Call `.with_criteria()` to specify what to evaluate against.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            context: None,
            criteria: CritiqueCriteria::Custom {
                description: "Evaluate for correctness and quality".into(),
            },
        }
    }

    /// Set the context for the content being critiqued.
    #[must_use]
    pub fn with_context(mut self, context: CritiqueContext) -> Self {
        self.context = Some(context);
        self
    }

    /// Set the criteria to evaluate against.
    #[must_use]
    pub fn with_criteria(mut self, criteria: CritiqueCriteria) -> Self {
        self.criteria = criteria;
        self
    }

    /// Validate the input before critique.
    ///
    /// # Errors
    ///
    /// Returns error if content is empty or whitespace-only.
    pub fn validate(&self) -> Result<(), AgentError> {
        if self.content.trim().is_empty() {
            return Err(AgentError::InvalidConfig(
                "Critique content must not be empty".into(),
            ));
        }
        Ok(())
    }

    /// Serialize to JSON string for use as query.
    pub fn to_query(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|e| {
            log::error!("Failed to serialize CritiqueInput: {}. This is a bug.", e);
            String::new()
        })
    }
}

/// Context about the content's origin.
///
/// Provides optional metadata about what produced the content being critiqued.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CritiqueContext {
    /// The original query/task that led to this content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,

    /// Name of the agent that produced the content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,

    /// Any additional context as JSON.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub metadata: Value,
}

impl CritiqueContext {
    /// Create an empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the original query.
    #[must_use]
    pub fn with_query(mut self, query: impl Into<String>) -> Self {
        self.query = Some(query.into());
        self
    }

    /// Set the agent name.
    #[must_use]
    pub fn with_agent(mut self, agent: impl Into<String>) -> Self {
        self.agent = Some(agent.into());
        self
    }

    /// Set additional metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// What to evaluate content against.
///
/// Different criteria types enable different validation modes:
/// - `GroundTruth`: Compare to expected answer (semantic match)
/// - `Specification`: Check adherence to requirements
/// - `Checklist`: Verify specific items are addressed
/// - `Custom`: Natural language criteria
/// - `CodeConventions`: Check code against conventions
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum CritiqueCriteria {
    /// Compare to ground truth (semantic match).
    ///
    /// Used for evaluation scoring where expected answer is known.
    GroundTruth {
        /// The expected/correct answer.
        expected: String,
    },

    /// Evaluate against a specification or requirements.
    ///
    /// Used to check if output meets functional requirements.
    Specification {
        /// The specification to check against.
        spec: String,
    },

    /// Check for specific items in a checklist.
    ///
    /// Used to verify multiple criteria are met.
    Checklist {
        /// Items to check for.
        items: Vec<String>,
    },

    /// Custom criteria described in natural language.
    ///
    /// Most flexible option for ad-hoc validation.
    Custom {
        /// Natural language description of what to check.
        description: String,
    },

    /// Code-specific: check against conventions.
    ///
    /// Used to validate code against style guides like CLAUDE.md.
    CodeConventions {
        /// The conventions to check against.
        conventions: String,
    },
}

// ============================================================================
// Output Types
// ============================================================================

/// Result of critique evaluation.
///
/// Contains the overall verdict, detailed findings, and actionable suggestions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CritiqueOutput {
    /// Overall assessment of the content.
    pub verdict: CritiqueVerdict,

    /// Confidence in the verdict (0.0-1.0).
    pub confidence: f32,

    /// Detailed findings from the evaluation.
    #[serde(default)]
    pub findings: Vec<CritiqueFinding>,

    /// Actionable suggestions for improvement.
    #[serde(default)]
    pub suggestions: Vec<String>,

    /// Whether this should trigger a retry with the suggestions.
    #[serde(default)]
    pub should_retry: bool,
}

impl CritiqueOutput {
    /// Get the JSON schema for structured output.
    pub fn schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["Pass", "PassWithWarnings", "NeedsRevision", "Reject"],
                    "description": "Overall assessment: Pass (meets criteria), PassWithWarnings (minor issues), NeedsRevision (significant issues), Reject (fundamental problems)"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in the verdict (0.0-1.0)"
                },
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "issue": {
                                "type": "string",
                                "description": "What was found"
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["Info", "Warning", "Error", "Critical"],
                                "description": "Severity of the finding"
                            },
                            "location": {
                                "type": "string",
                                "description": "Where in the content (if applicable)"
                            },
                            "suggestion": {
                                "type": "string",
                                "description": "How to fix"
                            }
                        },
                        "required": ["issue", "severity"]
                    },
                    "description": "Detailed findings from the evaluation"
                },
                "suggestions": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Actionable suggestions for improvement"
                },
                "should_retry": {
                    "type": "boolean",
                    "description": "Whether the content should be revised and retried"
                }
            },
            "required": ["verdict", "confidence", "findings", "suggestions", "should_retry"]
        })
    }

    /// Check if the verdict indicates the content passed.
    pub fn passed(&self) -> bool {
        self.verdict.is_passing()
    }

    /// Check if the verdict indicates failure.
    pub fn failed(&self) -> bool {
        !self.verdict.is_passing()
    }

    /// Convert verdict to a numeric score (0.0 = Reject, 1.0 = Pass).
    pub fn to_score(&self) -> f64 {
        self.verdict.to_score()
    }
}

/// Overall assessment verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CritiqueVerdict {
    /// Passes all criteria.
    Pass,
    /// Minor issues, but acceptable.
    PassWithWarnings,
    /// Significant issues, needs revision.
    NeedsRevision,
    /// Fundamental problems, reject.
    Reject,
}

impl std::fmt::Display for CritiqueVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CritiqueVerdict::Pass => write!(f, "Pass"),
            CritiqueVerdict::PassWithWarnings => write!(f, "PassWithWarnings"),
            CritiqueVerdict::NeedsRevision => write!(f, "NeedsRevision"),
            CritiqueVerdict::Reject => write!(f, "Reject"),
        }
    }
}

impl Default for CritiqueVerdict {
    /// Defaults to `Reject` (conservative fail-safe).
    fn default() -> Self {
        CritiqueVerdict::Reject
    }
}

impl std::str::FromStr for CritiqueVerdict {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Case-insensitive matching for robustness with LLM responses
        match s.to_lowercase().as_str() {
            "pass" => Ok(CritiqueVerdict::Pass),
            "passwithwarnings" => Ok(CritiqueVerdict::PassWithWarnings),
            "needsrevision" => Ok(CritiqueVerdict::NeedsRevision),
            "reject" => Ok(CritiqueVerdict::Reject),
            _ => Err(format!("Unknown verdict: {}", s)),
        }
    }
}

impl CritiqueVerdict {
    /// Returns true if this verdict indicates passing (Pass or PassWithWarnings).
    #[must_use]
    pub fn is_passing(&self) -> bool {
        matches!(
            self,
            CritiqueVerdict::Pass | CritiqueVerdict::PassWithWarnings
        )
    }

    /// Convert verdict to a numeric score (0.0 = Reject, 1.0 = Pass).
    #[must_use]
    pub fn to_score(&self) -> f64 {
        match self {
            CritiqueVerdict::Pass => 1.0,
            CritiqueVerdict::PassWithWarnings => 0.75,
            CritiqueVerdict::NeedsRevision => 0.25,
            CritiqueVerdict::Reject => 0.0,
        }
    }
}

/// A specific finding from the critique.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CritiqueFinding {
    /// What was found.
    pub issue: String,

    /// Severity of the finding.
    pub severity: Severity,

    /// Where in the content (if applicable).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,

    /// How to fix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
}

/// Severity level for findings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    /// Informational note.
    Info,
    /// Minor issue, worth noting.
    Warning,
    /// Significant issue that should be fixed.
    Error,
    /// Critical issue that must be addressed.
    Critical,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Info => write!(f, "Info"),
            Severity::Warning => write!(f, "Warning"),
            Severity::Error => write!(f, "Error"),
            Severity::Critical => write!(f, "Critical"),
        }
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the Critique agent.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CritiqueAgentConfig {
    /// Base system instruction (criteria-specific instructions are added).
    pub system_instruction: String,

    /// Timeout for the LLM call.
    pub timeout: Duration,

    /// Minimum verdict to suggest retry.
    ///
    /// If the verdict is this threshold or worse, `should_retry` will be true.
    /// For example, with threshold `NeedsRevision`, both `NeedsRevision` and
    /// `Reject` verdicts will set `should_retry = true`.
    pub retry_threshold: CritiqueVerdict,
}

impl Default for CritiqueAgentConfig {
    fn default() -> Self {
        Self {
            system_instruction: DEFAULT_SYSTEM.to_string(),
            timeout: Duration::from_secs(60),
            retry_threshold: CritiqueVerdict::NeedsRevision,
        }
    }
}

impl CritiqueAgentConfig {
    /// Set the base system instruction.
    #[must_use]
    pub fn with_system_instruction(mut self, instruction: impl Into<String>) -> Self {
        self.system_instruction = instruction.into();
        self
    }

    /// Set the timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the retry threshold.
    #[must_use]
    pub fn with_retry_threshold(mut self, threshold: CritiqueVerdict) -> Self {
        self.retry_threshold = threshold;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), AgentError> {
        let mut errors = Vec::new();

        if self.timeout.is_zero() {
            errors.push("timeout must be greater than zero");
        }

        if self.system_instruction.trim().is_empty() {
            errors.push("system_instruction must not be empty");
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(AgentError::InvalidConfig(errors.join("; ")))
        }
    }
}

/// Default base system instruction.
const DEFAULT_SYSTEM: &str = r#"You are an expert evaluator that critiques content against specified criteria.

Your task is to:
1. Carefully analyze the content provided
2. Evaluate it against the specified criteria
3. Identify any issues, problems, or areas for improvement
4. Provide actionable suggestions

Be thorough but fair. Focus on substantive issues rather than nitpicking.
When in doubt about severity, err on the side of providing helpful feedback.

Respond with your evaluation in the specified JSON format."#;

// ============================================================================
// Criteria-Specific System Instructions
// ============================================================================

fn build_system_instruction(base: &str, criteria: &CritiqueCriteria) -> String {
    let criteria_instruction = match criteria {
        CritiqueCriteria::GroundTruth { .. } => {
            r#"
EVALUATION MODE: Ground Truth Comparison

You are comparing a predicted answer to an expected ground truth answer.

Guidelines:
- Focus on semantic meaning, not exact wording
- A correct answer may be more or less detailed than the ground truth
- Case and punctuation differences don't matter
- Partial answers that contain the key information are correct
- Answers that are factually equivalent are correct

Verdict mapping:
- Pass: Semantically equivalent to ground truth
- PassWithWarnings: Mostly correct with minor discrepancies
- NeedsRevision: Contains the right idea but significant issues
- Reject: Incorrect or fundamentally different from ground truth"#
        }

        CritiqueCriteria::Specification { .. } => {
            r#"
EVALUATION MODE: Specification Checking

You are checking if the content meets the specified requirements.

Guidelines:
- Check each requirement systematically
- Note which requirements are met and which are not
- Consider both explicit and implicit requirements
- Be strict about security and correctness requirements
- Be lenient about stylistic preferences

Verdict mapping:
- Pass: All requirements met
- PassWithWarnings: All critical requirements met, minor issues
- NeedsRevision: Some important requirements not met
- Reject: Critical requirements not met"#
        }

        CritiqueCriteria::Checklist { .. } => {
            r#"
EVALUATION MODE: Checklist Verification

You are checking if the content addresses each item in the checklist.

Guidelines:
- Go through each checklist item systematically
- Mark each item as addressed or not addressed
- Note partial coverage
- Be specific about what is missing

Verdict mapping:
- Pass: All items addressed
- PassWithWarnings: All items addressed with minor gaps
- NeedsRevision: Some items missing
- Reject: Most items not addressed"#
        }

        CritiqueCriteria::Custom { .. } => {
            r#"
EVALUATION MODE: Custom Criteria

You are evaluating the content against a custom description of what to check.

Guidelines:
- Interpret the criteria charitably but thoroughly
- Consider both the letter and spirit of the criteria
- Provide specific examples from the content
- Be constructive in your feedback"#
        }

        CritiqueCriteria::CodeConventions { .. } => {
            r#"
EVALUATION MODE: Code Conventions

You are checking if code follows the specified conventions.

Guidelines:
- Check naming conventions, patterns, and style
- Verify architectural consistency
- Note any anti-patterns
- Focus on maintainability and readability
- Be pragmatic - conventions exist to help, not restrict

Verdict mapping:
- Pass: Follows conventions well
- PassWithWarnings: Minor deviations that don't hurt maintainability
- NeedsRevision: Significant convention violations
- Reject: Code doesn't follow the conventions at all"#
        }
    };

    format!("{}\n{}", base, criteria_instruction)
}

fn build_prompt(input: &CritiqueInput) -> String {
    let mut prompt = String::new();

    // Add context if present
    if let Some(ctx) = &input.context {
        prompt.push_str("## Context\n\n");
        if let Some(query) = &ctx.query {
            prompt.push_str(&format!("Original query: {}\n", query));
        }
        if let Some(agent) = &ctx.agent {
            prompt.push_str(&format!("Produced by: {}\n", agent));
        }
        if !ctx.metadata.is_null() {
            prompt.push_str(&format!("Metadata: {}\n", ctx.metadata));
        }
        prompt.push('\n');
    }

    // Add criteria
    prompt.push_str("## Criteria\n\n");
    match &input.criteria {
        CritiqueCriteria::GroundTruth { expected } => {
            prompt.push_str(&format!("Expected answer:\n{}\n\n", expected));
        }
        CritiqueCriteria::Specification { spec } => {
            prompt.push_str(&format!("Specification:\n{}\n\n", spec));
        }
        CritiqueCriteria::Checklist { items } => {
            prompt.push_str("Checklist items:\n");
            for (i, item) in items.iter().enumerate() {
                prompt.push_str(&format!("{}. {}\n", i + 1, item));
            }
            prompt.push('\n');
        }
        CritiqueCriteria::Custom { description } => {
            prompt.push_str(&format!("Evaluation criteria:\n{}\n\n", description));
        }
        CritiqueCriteria::CodeConventions { conventions } => {
            prompt.push_str(&format!("Conventions:\n{}\n\n", conventions));
        }
    }

    // Add content to evaluate
    prompt.push_str("## Content to Evaluate\n\n");
    prompt.push_str(&input.content);

    prompt
}

// ============================================================================
// Agent Implementation
// ============================================================================

/// Critique Agent - validates content against configurable criteria.
///
/// A versatile validation agent that can:
/// - Compare answers to ground truth (semantic comparison)
/// - Check outputs against specifications
/// - Verify checklist items
/// - Apply custom criteria
/// - Validate code conventions
///
/// # Events
///
/// - `critique_started`: Evaluation beginning
/// - `critique_result`: Final result with verdict, findings, and suggestions
/// - `final_result`: Standard completion event
pub struct CritiqueAgent {
    config: CritiqueAgentConfig,
}

impl CritiqueAgent {
    /// Create a new Critique agent with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `AgentError::InvalidConfig` if the configuration is invalid.
    pub fn new(config: CritiqueAgentConfig) -> Result<Self, AgentError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Create with default configuration.
    pub fn default_agent() -> Self {
        Self::new(CritiqueAgentConfig::default()).expect("default config is valid")
    }

    /// Execute critique and return typed output directly.
    ///
    /// This is a convenience method that consumes the event stream internally
    /// and returns the structured `CritiqueOutput`. Use `execute()` directly
    /// if you need streaming observability or access to individual events.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gemicro_core::{AgentContext, LlmClient, LlmConfig};
    /// use gemicro_critique_agent::{CritiqueAgent, CritiqueAgentConfig, CritiqueInput, CritiqueCriteria};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let agent = CritiqueAgent::new(CritiqueAgentConfig::default())?;
    /// let input = CritiqueInput::new("Paris")
    ///     .with_criteria(CritiqueCriteria::GroundTruth {
    ///         expected: "Paris".into()
    ///     });
    ///
    /// # let genai_client = genai_rs::Client::builder("key".to_string()).build().unwrap();
    /// # let llm = LlmClient::new(genai_client, LlmConfig::default());
    /// let context = AgentContext::new(llm);
    /// let output = agent.critique(&input, context).await?;
    ///
    /// if output.verdict.is_passing() {
    ///     println!("Passed with confidence: {:.0}%", output.confidence * 100.0);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn critique(
        &self,
        input: &CritiqueInput,
        context: AgentContext,
    ) -> Result<CritiqueOutput, AgentError> {
        use futures_util::StreamExt;

        let stream = <Self as Agent>::execute(self, &input.to_query(), context);
        futures_util::pin_mut!(stream);

        while let Some(result) = stream.next().await {
            let update = result?;
            // Skip intermediate events; wait for final_result which contains full CritiqueOutput
            // Use as_final_result() for type-safe access to the result field
            if let Some(final_result) = update.as_final_result() {
                // Use result_as<T>() for ergonomic typed deserialization
                let output = final_result.result_as::<CritiqueOutput>().map_err(|e| {
                    AgentError::ParseFailed(format!(
                        "Failed to parse CritiqueOutput from final_result: {}",
                        e
                    ))
                })?;
                return Ok(output);
            }
        }

        Err(AgentError::ParseFailed(
            "Critique stream ended without producing a result".to_string(),
        ))
    }
}

impl Agent for CritiqueAgent {
    fn name(&self) -> &str {
        "critique"
    }

    fn description(&self) -> &str {
        "Validates content against criteria and provides actionable feedback"
    }

    fn execute(&self, query: &str, context: AgentContext) -> AgentStream<'_> {
        let query = query.to_string();
        let config = self.config.clone();

        Box::pin(try_stream! {
            let start = Instant::now();

            // Parse input
            let input: CritiqueInput = serde_json::from_str(&query)
                .map_err(|e| AgentError::InvalidConfig(format!(
                    "Invalid critique input format. Expected JSON with 'content' and 'criteria' fields. Error: {}",
                    e
                )))?;

            // Validate input
            input.validate()?;

            // Emit started event
            yield AgentUpdate::custom(
                EVENT_CRITIQUE_STARTED,
                format!("Starting critique: {}", truncate(&input.content, 50)),
                json!({
                    "content_length": input.content.len(),
                    "criteria_type": format!("{:?}", std::mem::discriminant(&input.criteria))
                })
            );

            // Build system instruction and prompt
            let system = build_system_instruction(&config.system_instruction, &input.criteria);
            let prompt = build_prompt(&input);

            // Calculate remaining time and execute LLM call
            let timeout = remaining_time(start, config.timeout, "critique")?;

            let request = context.llm.client().interaction()
                .with_system_instruction(&system)
                .with_text(&prompt)
                .with_response_format(CritiqueOutput::schema())
                .build().map_err(|e| AgentError::Other(e.to_string()))?;

            let generate_future = async {
                context
                    .llm
                    .generate(request)
                    .await
                    .map_err(AgentError::Llm)
            };

            let response = with_timeout_and_cancellation(
                generate_future,
                timeout,
                &context.cancellation_token,
                || timeout_error(start, config.timeout, "critique"),
            )
            .await?;

            // Parse structured output
            let response_text = response.text().ok_or_else(|| {
                AgentError::ParseFailed(
                    "LLM returned no text content. The model may have refused or failed to respond.".to_string()
                )
            })?;
            if response_text.trim().is_empty() {
                Err(AgentError::ParseFailed(
                    "LLM returned empty text content. The model may have refused or failed to respond.".to_string()
                ))?;
            }
            let mut output: CritiqueOutput = serde_json::from_str(response_text)
                .map_err(|e| AgentError::ParseFailed(format!(
                    "Failed to parse critique output: {}. Response: {}",
                    e,
                    gemicro_core::truncate_with_count(response_text, 200)
                )))?;

            // Set should_retry based on threshold
            output.should_retry = should_suggest_retry(output.verdict, config.retry_threshold);

            let tokens_used = gemicro_core::extract_total_tokens(&response);
            let duration_ms = start.elapsed().as_millis() as u64;

            // Emit result event
            yield AgentUpdate::custom(
                EVENT_CRITIQUE_RESULT,
                format!("Verdict: {}", output.verdict),
                json!({
                    "verdict": output.verdict,
                    "confidence": output.confidence,
                    "findings_count": output.findings.len(),
                    "suggestions_count": output.suggestions.len(),
                    "should_retry": output.should_retry,
                    "tokens_used": tokens_used,
                    "duration_ms": duration_ms
                })
            );

            // Emit standard final_result
            let metadata = ResultMetadata::new(
                tokens_used.unwrap_or(0),
                if tokens_used.is_none() { 1 } else { 0 },
                duration_ms,
            );

            // The answer is the structured output as JSON
            let answer = serde_json::to_value(&output)
                .unwrap_or_else(|e| {
                    log::error!(
                        "Failed to serialize CritiqueOutput: {}. Falling back to minimal response.",
                        e
                    );
                    json!({ "verdict": output.verdict.to_string() })
                });

            yield AgentUpdate::final_result(answer, metadata);
        })
    }

    fn create_tracker(&self) -> Box<dyn gemicro_core::ExecutionTracking> {
        Box::new(gemicro_core::DefaultTracker::default())
    }
}

/// Determine if retry should be suggested based on verdict and threshold.
fn should_suggest_retry(verdict: CritiqueVerdict, threshold: CritiqueVerdict) -> bool {
    let verdict_rank = verdict_to_rank(verdict);
    let threshold_rank = verdict_to_rank(threshold);
    verdict_rank >= threshold_rank
}

fn verdict_to_rank(verdict: CritiqueVerdict) -> u8 {
    match verdict {
        CritiqueVerdict::Pass => 0,
        CritiqueVerdict::PassWithWarnings => 1,
        CritiqueVerdict::NeedsRevision => 2,
        CritiqueVerdict::Reject => 3,
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Input tests
    #[test]
    fn test_critique_input_creation() {
        let input = CritiqueInput::new("test content");
        assert_eq!(input.content, "test content");
        assert!(input.context.is_none());
    }

    #[test]
    fn test_critique_input_with_context() {
        let input = CritiqueInput::new("test")
            .with_context(CritiqueContext::new().with_query("original query"));
        assert!(input.context.is_some());
        assert_eq!(input.context.unwrap().query, Some("original query".into()));
    }

    #[test]
    fn test_critique_input_with_criteria() {
        let input = CritiqueInput::new("test").with_criteria(CritiqueCriteria::GroundTruth {
            expected: "expected".into(),
        });
        assert!(matches!(
            input.criteria,
            CritiqueCriteria::GroundTruth { .. }
        ));
    }

    #[test]
    fn test_critique_input_serialization() {
        let input = CritiqueInput::new("content").with_criteria(CritiqueCriteria::GroundTruth {
            expected: "truth".into(),
        });
        let json = input.to_query();
        let parsed: CritiqueInput = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.content, "content");
    }

    #[test]
    fn test_critique_input_validate_empty() {
        let empty = CritiqueInput::new("");
        assert!(empty.validate().is_err());

        let whitespace = CritiqueInput::new("   ");
        assert!(whitespace.validate().is_err());

        let valid = CritiqueInput::new("content");
        assert!(valid.validate().is_ok());
    }

    // Context tests
    #[test]
    fn test_critique_context_builder() {
        let ctx = CritiqueContext::new()
            .with_query("the query")
            .with_agent("simple_qa")
            .with_metadata(json!({"key": "value"}));
        assert_eq!(ctx.query, Some("the query".into()));
        assert_eq!(ctx.agent, Some("simple_qa".into()));
        assert_eq!(ctx.metadata["key"], "value");
    }

    // Criteria serialization tests
    #[test]
    fn test_criteria_ground_truth_serialization() {
        let criteria = CritiqueCriteria::GroundTruth {
            expected: "42".into(),
        };
        let json = serde_json::to_string(&criteria).unwrap();
        assert!(json.contains("GroundTruth"));
        assert!(json.contains("42"));
    }

    #[test]
    fn test_criteria_checklist_serialization() {
        let criteria = CritiqueCriteria::Checklist {
            items: vec!["item1".into(), "item2".into()],
        };
        let json = serde_json::to_string(&criteria).unwrap();
        assert!(json.contains("Checklist"));
        assert!(json.contains("item1"));
    }

    // Output tests
    #[test]
    fn test_critique_output_schema() {
        let schema = CritiqueOutput::schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["verdict"].is_object());
        assert!(schema["properties"]["confidence"].is_object());
        assert!(schema["properties"]["findings"].is_object());
    }

    #[test]
    fn test_critique_output_deserialization() {
        let json = r#"{
            "verdict": "Pass",
            "confidence": 0.95,
            "findings": [],
            "suggestions": ["Consider adding tests"],
            "should_retry": false
        }"#;
        let output: CritiqueOutput = serde_json::from_str(json).unwrap();
        assert_eq!(output.verdict, CritiqueVerdict::Pass);
        assert!((output.confidence - 0.95).abs() < f32::EPSILON);
        assert!(!output.should_retry);
    }

    #[test]
    fn test_critique_output_passed() {
        let pass = CritiqueOutput {
            verdict: CritiqueVerdict::Pass,
            confidence: 1.0,
            findings: vec![],
            suggestions: vec![],
            should_retry: false,
        };
        assert!(pass.passed());
        assert!(!pass.failed());

        let warn = CritiqueOutput {
            verdict: CritiqueVerdict::PassWithWarnings,
            confidence: 0.9,
            findings: vec![],
            suggestions: vec![],
            should_retry: false,
        };
        assert!(warn.passed());
        assert!(!warn.failed());
    }

    #[test]
    fn test_critique_output_failed() {
        let revision = CritiqueOutput {
            verdict: CritiqueVerdict::NeedsRevision,
            confidence: 0.8,
            findings: vec![],
            suggestions: vec![],
            should_retry: true,
        };
        assert!(revision.failed());
        assert!(!revision.passed());
    }

    #[test]
    fn test_critique_output_to_score() {
        assert!(
            (CritiqueOutput {
                verdict: CritiqueVerdict::Pass,
                confidence: 1.0,
                findings: vec![],
                suggestions: vec![],
                should_retry: false,
            }
            .to_score()
                - 1.0)
                .abs()
                < f64::EPSILON
        );

        assert!(
            (CritiqueOutput {
                verdict: CritiqueVerdict::Reject,
                confidence: 1.0,
                findings: vec![],
                suggestions: vec![],
                should_retry: false,
            }
            .to_score()
                - 0.0)
                .abs()
                < f64::EPSILON
        );

        // Test intermediate values
        assert!(
            (CritiqueOutput {
                verdict: CritiqueVerdict::PassWithWarnings,
                confidence: 1.0,
                findings: vec![],
                suggestions: vec![],
                should_retry: false,
            }
            .to_score()
                - 0.75)
                .abs()
                < f64::EPSILON
        );

        assert!(
            (CritiqueOutput {
                verdict: CritiqueVerdict::NeedsRevision,
                confidence: 1.0,
                findings: vec![],
                suggestions: vec![],
                should_retry: true,
            }
            .to_score()
                - 0.25)
                .abs()
                < f64::EPSILON
        );
    }

    // Verdict tests
    #[test]
    fn test_verdict_display() {
        assert_eq!(format!("{}", CritiqueVerdict::Pass), "Pass");
        assert_eq!(
            format!("{}", CritiqueVerdict::NeedsRevision),
            "NeedsRevision"
        );
    }

    // Severity tests
    #[test]
    fn test_severity_display() {
        assert_eq!(format!("{}", Severity::Critical), "Critical");
        assert_eq!(format!("{}", Severity::Info), "Info");
    }

    // Config tests
    #[test]
    fn test_config_default_is_valid() {
        let config = CritiqueAgentConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_rejects_zero_timeout() {
        let config = CritiqueAgentConfig::default().with_timeout(Duration::ZERO);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_rejects_empty_system() {
        let config = CritiqueAgentConfig::default().with_system_instruction("   ");
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = CritiqueAgentConfig::default()
            .with_timeout(Duration::from_secs(120))
            .with_retry_threshold(CritiqueVerdict::PassWithWarnings);
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.retry_threshold, CritiqueVerdict::PassWithWarnings);
    }

    // Agent tests
    #[test]
    fn test_agent_name_and_description() {
        let agent = CritiqueAgent::default_agent();
        assert_eq!(agent.name(), "critique");
        assert!(!agent.description().is_empty());
    }

    #[test]
    fn test_agent_creation_validates_config() {
        let invalid = CritiqueAgentConfig::default().with_timeout(Duration::ZERO);
        assert!(CritiqueAgent::new(invalid).is_err());
    }

    // Retry logic tests
    #[test]
    fn test_should_suggest_retry() {
        // With NeedsRevision threshold (default)
        let threshold = CritiqueVerdict::NeedsRevision;
        assert!(!should_suggest_retry(CritiqueVerdict::Pass, threshold));
        assert!(!should_suggest_retry(
            CritiqueVerdict::PassWithWarnings,
            threshold
        ));
        assert!(should_suggest_retry(
            CritiqueVerdict::NeedsRevision,
            threshold
        ));
        assert!(should_suggest_retry(CritiqueVerdict::Reject, threshold));
    }

    #[test]
    fn test_should_suggest_retry_with_strict_threshold() {
        let threshold = CritiqueVerdict::PassWithWarnings;
        assert!(!should_suggest_retry(CritiqueVerdict::Pass, threshold));
        assert!(should_suggest_retry(
            CritiqueVerdict::PassWithWarnings,
            threshold
        ));
        assert!(should_suggest_retry(
            CritiqueVerdict::NeedsRevision,
            threshold
        ));
    }

    // Prompt building tests
    #[test]
    fn test_build_prompt_ground_truth() {
        let input =
            CritiqueInput::new("predicted answer").with_criteria(CritiqueCriteria::GroundTruth {
                expected: "expected answer".into(),
            });
        let prompt = build_prompt(&input);
        assert!(prompt.contains("Expected answer:"));
        assert!(prompt.contains("expected answer"));
        assert!(prompt.contains("predicted answer"));
    }

    #[test]
    fn test_build_prompt_with_context() {
        let input = CritiqueInput::new("content")
            .with_context(
                CritiqueContext::new()
                    .with_query("what is 2+2?")
                    .with_agent("simple_qa"),
            )
            .with_criteria(CritiqueCriteria::Custom {
                description: "check math".into(),
            });
        let prompt = build_prompt(&input);
        assert!(prompt.contains("Original query:"));
        assert!(prompt.contains("what is 2+2?"));
        assert!(prompt.contains("simple_qa"));
    }

    #[test]
    fn test_build_prompt_checklist() {
        let input = CritiqueInput::new("content").with_criteria(CritiqueCriteria::Checklist {
            items: vec!["item one".into(), "item two".into()],
        });
        let prompt = build_prompt(&input);
        assert!(prompt.contains("1. item one"));
        assert!(prompt.contains("2. item two"));
    }

    // FromStr tests
    #[test]
    fn test_verdict_from_str_case_insensitive() {
        use std::str::FromStr;

        // Exact case
        assert_eq!(
            CritiqueVerdict::from_str("Pass").unwrap(),
            CritiqueVerdict::Pass
        );

        // Lowercase
        assert_eq!(
            CritiqueVerdict::from_str("pass").unwrap(),
            CritiqueVerdict::Pass
        );

        // Uppercase
        assert_eq!(
            CritiqueVerdict::from_str("PASS").unwrap(),
            CritiqueVerdict::Pass
        );

        // Mixed case
        assert_eq!(
            CritiqueVerdict::from_str("pAsS").unwrap(),
            CritiqueVerdict::Pass
        );
        assert_eq!(
            CritiqueVerdict::from_str("PASSWITHWARNINGS").unwrap(),
            CritiqueVerdict::PassWithWarnings
        );
        assert_eq!(
            CritiqueVerdict::from_str("needsrevision").unwrap(),
            CritiqueVerdict::NeedsRevision
        );
        assert_eq!(
            CritiqueVerdict::from_str("REJECT").unwrap(),
            CritiqueVerdict::Reject
        );
    }

    #[test]
    fn test_verdict_from_str_invalid() {
        use std::str::FromStr;

        assert!(CritiqueVerdict::from_str("invalid").is_err());
        assert!(CritiqueVerdict::from_str("").is_err());
        assert!(CritiqueVerdict::from_str("pass ").is_err()); // trailing space
    }
}
