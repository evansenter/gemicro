//! Calculator tool for evaluating mathematical expressions.

use async_trait::async_trait;
use gemicro_core::tool::{Tool, ToolError, ToolResult};
use serde_json::{json, Value};

/// Maximum allowed length for calculator expressions to prevent abuse.
const MAX_EXPRESSION_LENGTH: usize = 1000;

/// Calculator tool for evaluating mathematical expressions.
///
/// Supports basic arithmetic (+, -, *, /), exponents (^), parentheses,
/// and common functions (sqrt, sin, cos, tan, log, ln, abs).
///
/// # Example
///
/// ```no_run
/// use gemicro_prompt_agent::tools::Calculator;
/// use gemicro_core::tool::Tool;
/// use serde_json::json;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let calc = Calculator;
/// let result = calc.execute(json!({"expression": "2 + 2"})).await?;
/// // result.content is now Value::String("4")
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Calculator;

#[async_trait]
impl Tool for Calculator {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Evaluate mathematical expressions. Supports basic arithmetic (+, -, *, /), \
         exponents (^), parentheses, and common functions (sqrt, sin, cos, tan, log, ln, abs)."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate, e.g., '2 + 2', 'sqrt(16)', '3.14 * 2^3'"
                }
            },
            "required": ["expression"]
        })
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let expression = input
            .get("expression")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'expression' field".into()))?;

        // Validate input length to prevent abuse
        if expression.len() > MAX_EXPRESSION_LENGTH {
            return Err(ToolError::InvalidInput(format!(
                "Expression too long ({} chars, max {})",
                expression.len(),
                MAX_EXPRESSION_LENGTH
            )));
        }

        match meval::eval_str(expression) {
            Ok(result) => {
                if result.is_nan() {
                    return Err(ToolError::ExecutionFailed(
                        "Result is not a number (NaN)".into(),
                    ));
                }
                if result.is_infinite() {
                    return Err(ToolError::ExecutionFailed(
                        "Result is infinite (division by zero or overflow)".into(),
                    ));
                }

                let formatted = if result.fract() == 0.0 && result.abs() < 1e15 {
                    format!("{:.0}", result)
                } else {
                    format!("{}", result)
                };
                Ok(ToolResult::text(formatted))
            }
            Err(e) => Err(ToolError::ExecutionFailed(format!(
                "Failed to evaluate expression: {}",
                e
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_calculator_basic_arithmetic() {
        let calc = Calculator;
        let result = calc.execute(json!({"expression": "2 + 2"})).await.unwrap();
        assert_eq!(result.content.as_str().unwrap(), "4");
    }

    #[tokio::test]
    async fn test_calculator_division() {
        let calc = Calculator;
        let result = calc.execute(json!({"expression": "10 / 4"})).await.unwrap();
        assert_eq!(result.content.as_str().unwrap(), "2.5");
    }

    #[tokio::test]
    async fn test_calculator_sqrt() {
        let calc = Calculator;
        let result = calc
            .execute(json!({"expression": "sqrt(16)"}))
            .await
            .unwrap();
        assert_eq!(result.content.as_str().unwrap(), "4");
    }

    #[tokio::test]
    async fn test_calculator_missing_expression() {
        let calc = Calculator;
        let result = calc.execute(json!({})).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_calculator_invalid_expression() {
        let calc = Calculator;
        let result = calc.execute(json!({"expression": "invalid"})).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::ExecutionFailed(_)));
    }

    #[tokio::test]
    async fn test_calculator_long_expression() {
        let calc = Calculator;
        let long_expr = "1+".repeat(600); // 1200 chars
        let result = calc.execute(json!({"expression": long_expr})).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_calculator_division_by_zero_returns_error() {
        let calc = Calculator;
        let result = calc.execute(json!({"expression": "1/0"})).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed(_)));
        assert!(err.to_string().contains("infinite"));
    }

    #[tokio::test]
    async fn test_calculator_sqrt_negative_returns_error() {
        let calc = Calculator;
        let result = calc.execute(json!({"expression": "sqrt(-1)"})).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed(_)));
        assert!(err.to_string().contains("NaN"));
    }

    #[test]
    fn test_calculator_name_and_description() {
        let calc = Calculator;
        assert_eq!(calc.name(), "calculator");
        assert!(!calc.description().is_empty());
    }

    #[test]
    fn test_calculator_parameters_schema() {
        let calc = Calculator;
        let schema = calc.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["expression"].is_object());
        assert!(schema["required"]
            .as_array()
            .unwrap()
            .contains(&json!("expression")));
    }
}
