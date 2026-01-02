//! Context usage tracking for token limit awareness.
//!
//! Tracks cumulative token usage during agent execution and provides
//! warnings when approaching model context limits.

/// Default context window size for Gemini models.
pub const DEFAULT_CONTEXT_WINDOW: u32 = 1_000_000;

/// Default warning threshold (80% of context).
pub const DEFAULT_WARNING_THRESHOLD: f32 = 0.80;

/// Tracks token usage throughout an agent session.
#[derive(Debug, Clone)]
pub struct ContextUsage {
    /// Cumulative tokens used so far.
    pub tokens_used: u32,
    /// Maximum context window size.
    pub context_window: u32,
    /// Threshold percentage at which to warn (0.0 to 1.0).
    pub warning_threshold: f32,
}

impl Default for ContextUsage {
    fn default() -> Self {
        Self {
            tokens_used: 0,
            context_window: DEFAULT_CONTEXT_WINDOW,
            warning_threshold: DEFAULT_WARNING_THRESHOLD,
        }
    }
}

impl ContextUsage {
    /// Create a new context usage tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a custom context window size.
    pub fn with_context_window(mut self, size: u32) -> Self {
        self.context_window = size;
        self
    }

    /// Create with a custom warning threshold (0.0 to 1.0).
    pub fn with_warning_threshold(mut self, threshold: f32) -> Self {
        self.warning_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Add tokens to the usage counter.
    pub fn add_tokens(&mut self, tokens: u32) {
        self.tokens_used = self.tokens_used.saturating_add(tokens);
    }

    /// Get the usage ratio (0.0 to 1.0+).
    pub fn usage_ratio(&self) -> f32 {
        if self.context_window == 0 {
            return 0.0;
        }
        self.tokens_used as f32 / self.context_window as f32
    }

    /// Get usage as a percentage (0 to 100+).
    pub fn usage_percent(&self) -> f32 {
        self.usage_ratio() * 100.0
    }

    /// Check if usage is at or above the warning threshold.
    pub fn is_warning(&self) -> bool {
        self.usage_ratio() >= self.warning_threshold
    }

    /// Check if usage is at or above 100%.
    pub fn is_critical(&self) -> bool {
        self.usage_ratio() >= 1.0
    }

    /// Get remaining tokens.
    pub fn remaining(&self) -> u32 {
        self.context_window.saturating_sub(self.tokens_used)
    }

    /// Get the warning level.
    pub fn level(&self) -> ContextLevel {
        let ratio = self.usage_ratio();
        if ratio >= 1.0 {
            ContextLevel::Critical
        } else if ratio >= self.warning_threshold {
            ContextLevel::Warning
        } else {
            ContextLevel::Normal
        }
    }
}

/// Context usage level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextLevel {
    /// Usage is within normal limits.
    Normal,
    /// Usage is above warning threshold but below limit.
    Warning,
    /// Usage is at or above the context limit.
    Critical,
}

impl std::fmt::Display for ContextLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Normal => write!(f, "normal"),
            Self::Warning => write!(f, "warning"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_context_usage() {
        let usage = ContextUsage::default();
        assert_eq!(usage.tokens_used, 0);
        assert_eq!(usage.context_window, DEFAULT_CONTEXT_WINDOW);
        assert_eq!(usage.usage_ratio(), 0.0);
        assert!(!usage.is_warning());
    }

    #[test]
    fn test_add_tokens() {
        let mut usage = ContextUsage::default();
        usage.add_tokens(100);
        assert_eq!(usage.tokens_used, 100);
        usage.add_tokens(200);
        assert_eq!(usage.tokens_used, 300);
    }

    #[test]
    fn test_usage_ratio() {
        let mut usage = ContextUsage::new().with_context_window(1000);
        usage.add_tokens(500);
        assert!((usage.usage_ratio() - 0.5).abs() < 0.001);
        assert!((usage.usage_percent() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_warning_threshold() {
        let mut usage = ContextUsage::new()
            .with_context_window(1000)
            .with_warning_threshold(0.8);

        usage.add_tokens(799);
        assert!(!usage.is_warning());
        assert_eq!(usage.level(), ContextLevel::Normal);

        usage.add_tokens(1);
        assert!(usage.is_warning());
        assert_eq!(usage.level(), ContextLevel::Warning);
    }

    #[test]
    fn test_critical_level() {
        let mut usage = ContextUsage::new().with_context_window(1000);
        usage.add_tokens(1000);
        assert!(usage.is_critical());
        assert_eq!(usage.level(), ContextLevel::Critical);
    }

    #[test]
    fn test_remaining() {
        let mut usage = ContextUsage::new().with_context_window(1000);
        usage.add_tokens(300);
        assert_eq!(usage.remaining(), 700);
    }

    #[test]
    fn test_saturating_add() {
        let mut usage = ContextUsage::new().with_context_window(100);
        usage.add_tokens(u32::MAX);
        assert_eq!(usage.tokens_used, u32::MAX);
        usage.add_tokens(1); // Should not overflow
        assert_eq!(usage.tokens_used, u32::MAX);
    }
}
