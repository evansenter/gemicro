//! REPL command parsing
//!
//! Parses user input into structured commands for the REPL to execute.

/// Parsed REPL command
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Switch to a different agent by name
    Agent(String),

    /// List all available agents
    ListAgents,

    /// Show help text
    Help,

    /// Show conversation history
    History,

    /// Clear conversation history
    Clear,

    /// Hot-reload agents (rebuild and re-exec)
    Reload,

    /// Exit the REPL
    Quit,

    /// Execute a query with the current agent
    Query(String),

    /// Empty input (just pressed enter)
    Empty,

    /// Unknown command
    Unknown(String),
}

impl Command {
    /// Parse user input into a command
    ///
    /// Commands start with `/`. Everything else is treated as a query.
    pub fn parse(input: &str) -> Self {
        let trimmed = input.trim();

        if trimmed.is_empty() {
            return Command::Empty;
        }

        // Check if it's a command (starts with /)
        if let Some(cmd) = trimmed.strip_prefix('/') {
            let parts: Vec<&str> = cmd.splitn(2, char::is_whitespace).collect();
            let cmd_name = parts[0].to_lowercase();
            let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

            match cmd_name.as_str() {
                "agent" | "a" => {
                    if arg.is_empty() {
                        Command::ListAgents
                    } else {
                        Command::Agent(arg.to_string())
                    }
                }
                "help" | "?" => Command::Help,
                "history" | "h" => Command::History,
                "clear" => Command::Clear,
                "reload" | "r" => Command::Reload,
                "quit" | "exit" | "q" => Command::Quit,
                _ => Command::Unknown(cmd_name),
            }
        } else {
            Command::Query(trimmed.to_string())
        }
    }

    /// Get help text for all commands
    pub fn help_text() -> &'static str {
        r#"Available commands:
  /help          Show this help message (alias: /?)
  /agent [name]  Switch to agent or list available agents (alias: /a)
  /history       Show conversation history (alias: /h)
  /clear         Clear conversation history
  /reload        Hot-reload agents from source (alias: /r)
  /quit          Exit the REPL (aliases: /exit, /q)

Or just type a query to send it to the current agent."#
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_query() {
        assert_eq!(
            Command::parse("What is Rust?"),
            Command::Query("What is Rust?".to_string())
        );
    }

    #[test]
    fn test_parse_query_with_whitespace() {
        assert_eq!(
            Command::parse("  What is Rust?  "),
            Command::Query("What is Rust?".to_string())
        );
    }

    #[test]
    fn test_parse_empty() {
        assert_eq!(Command::parse(""), Command::Empty);
        assert_eq!(Command::parse("   "), Command::Empty);
    }

    #[test]
    fn test_parse_agent_switch() {
        assert_eq!(
            Command::parse("/agent deep_research"),
            Command::Agent("deep_research".to_string())
        );
    }

    #[test]
    fn test_parse_agent_list() {
        assert_eq!(Command::parse("/agent"), Command::ListAgents);
    }

    #[test]
    fn test_parse_agent_alias() {
        assert_eq!(Command::parse("/a"), Command::ListAgents);
        assert_eq!(
            Command::parse("/a simple_qa"),
            Command::Agent("simple_qa".to_string())
        );
    }

    #[test]
    fn test_parse_help() {
        assert_eq!(Command::parse("/help"), Command::Help);
        assert_eq!(Command::parse("/?"), Command::Help);
    }

    #[test]
    fn test_parse_history() {
        assert_eq!(Command::parse("/history"), Command::History);
        assert_eq!(Command::parse("/h"), Command::History);
    }

    #[test]
    fn test_parse_clear() {
        assert_eq!(Command::parse("/clear"), Command::Clear);
    }

    #[test]
    fn test_parse_reload() {
        assert_eq!(Command::parse("/reload"), Command::Reload);
        assert_eq!(Command::parse("/r"), Command::Reload);
    }

    #[test]
    fn test_parse_quit() {
        assert_eq!(Command::parse("/quit"), Command::Quit);
        assert_eq!(Command::parse("/exit"), Command::Quit);
        assert_eq!(Command::parse("/q"), Command::Quit);
    }

    #[test]
    fn test_parse_unknown_command() {
        assert_eq!(
            Command::parse("/foobar"),
            Command::Unknown("foobar".to_string())
        );
    }

    #[test]
    fn test_parse_case_insensitive() {
        assert_eq!(Command::parse("/AGENT"), Command::ListAgents);
        assert_eq!(
            Command::parse("/Agent foo"),
            Command::Agent("foo".to_string())
        );
        assert_eq!(Command::parse("/QUIT"), Command::Quit);
    }

    #[test]
    fn test_parse_command_with_extra_spaces() {
        assert_eq!(
            Command::parse("/agent   deep_research  "),
            Command::Agent("deep_research".to_string())
        );
    }

    #[test]
    fn test_help_text_not_empty() {
        assert!(!Command::help_text().is_empty());
        assert!(Command::help_text().contains("/agent"));
    }
}
