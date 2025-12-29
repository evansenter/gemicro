//! Current datetime tool.

use async_trait::async_trait;
use gemicro_core::tool::{Tool, ToolError, ToolResult};
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};

/// Gets the current date and time.
///
/// Currently only supports UTC timezone.
///
/// # Example
///
/// ```no_run
/// use gemicro_tool_agent::tools::CurrentDatetime;
/// use gemicro_core::tool::Tool;
/// use serde_json::json;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let tool = CurrentDatetime;
/// let result = tool.execute(json!({"timezone": "UTC"})).await?;
/// // result.content will be JSON like:
/// // {"timezone": "UTC", "date": "2024-12-29", "time": "15:30:45"}
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CurrentDatetime;

#[async_trait]
impl Tool for CurrentDatetime {
    fn name(&self) -> &str {
        "current_datetime"
    }

    fn description(&self) -> &str {
        "Get the current date and time. Currently only UTC timezone is supported."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The timezone to get the time for. Currently only 'UTC' is supported."
                }
            },
            "required": ["timezone"]
        })
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let timezone = input
            .get("timezone")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing 'timezone' field".into()))?;

        // Only UTC is supported
        if !timezone.eq_ignore_ascii_case("utc") {
            return Err(ToolError::InvalidInput(format!(
                "Only UTC timezone is currently supported, got '{}'",
                timezone
            )));
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();

        // Calculate time components
        let total_secs = now.as_secs();
        let days_since_epoch = total_secs / 86400;
        let secs_today = total_secs % 86400;

        let hours = secs_today / 3600;
        let minutes = (secs_today % 3600) / 60;
        let seconds = secs_today % 60;

        // Calculate date from days since epoch
        let (year, month, day) = days_to_ymd(days_since_epoch);

        // Return structured JSON directly - the LLM receives this as-is
        Ok(ToolResult::json(json!({
            "timezone": "UTC",
            "date": format!("{:04}-{:02}-{:02}", year, month, day),
            "time": format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
        })))
    }
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(days_since_epoch: u64) -> (i64, u32, u32) {
    let mut remaining_days = days_since_epoch as i64;
    let mut year: i64 = 1970;

    // Find the year
    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    // Find the month
    let days_in_months: [i64; 12] = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month: u32 = 1;
    for &days_in_month in &days_in_months {
        if remaining_days < days_in_month {
            break;
        }
        remaining_days -= days_in_month;
        month += 1;
    }

    // remaining_days is now days within the month (0-indexed), add 1 for day of month
    let day = (remaining_days + 1) as u32;

    (year, month, day)
}

fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_current_datetime_utc() {
        let tool = CurrentDatetime;
        let result = tool.execute(json!({"timezone": "UTC"})).await.unwrap();

        // content is now structured JSON directly (not a string containing JSON)
        assert_eq!(result.content["timezone"], "UTC");
        assert!(result.content["date"].is_string());
        assert!(result.content["time"].is_string());
    }

    #[tokio::test]
    async fn test_current_datetime_utc_case_insensitive() {
        let tool = CurrentDatetime;

        for tz in &["UTC", "utc", "Utc", "uTc"] {
            let result = tool.execute(json!({"timezone": tz})).await;
            assert!(result.is_ok(), "Should accept {} as UTC", tz);
        }
    }

    #[tokio::test]
    async fn test_current_datetime_non_utc() {
        let tool = CurrentDatetime;
        let result = tool.execute(json!({"timezone": "EST"})).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_current_datetime_missing_timezone() {
        let tool = CurrentDatetime;
        let result = tool.execute(json!({})).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidInput(_)));
    }

    #[test]
    fn test_days_to_ymd_known_dates() {
        // Unix epoch: Jan 1, 1970
        assert_eq!(days_to_ymd(0), (1970, 1, 1));

        // Jan 2, 1970
        assert_eq!(days_to_ymd(1), (1970, 1, 2));

        // Feb 1, 1970 (31 days after epoch)
        assert_eq!(days_to_ymd(31), (1970, 2, 1));

        // Jan 1, 1971 (365 days after epoch)
        assert_eq!(days_to_ymd(365), (1971, 1, 1));

        // 2000-03-01 (known date for validation)
        assert_eq!(days_to_ymd(11017), (2000, 3, 1));
    }

    #[test]
    fn test_current_datetime_name_and_description() {
        let tool = CurrentDatetime;
        assert_eq!(tool.name(), "current_datetime");
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn test_current_datetime_parameters_schema() {
        let tool = CurrentDatetime;
        let schema = tool.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["timezone"].is_object());
        assert!(schema["required"]
            .as_array()
            .unwrap()
            .contains(&json!("timezone")));
    }
}
