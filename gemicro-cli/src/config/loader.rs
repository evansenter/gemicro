//! Config file discovery and loading.

use super::types::GemicroConfig;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Represents a config file source with its path and mtime.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ConfigSource {
    /// Path to the config file
    pub path: PathBuf,

    /// Last modified time (for staleness detection)
    pub mtime: Option<SystemTime>,

    /// Whether this file actually exists
    pub exists: bool,
}

impl ConfigSource {
    /// Create a new config source from a path.
    pub fn new(path: PathBuf) -> Self {
        let (exists, mtime) = match std::fs::metadata(&path) {
            Ok(meta) => (true, meta.modified().ok()),
            Err(_) => (false, None),
        };

        Self {
            path,
            mtime,
            exists,
        }
    }

    /// Check if the file has been modified since we last loaded it.
    pub fn is_stale(&self) -> bool {
        if !self.exists {
            // If it didn't exist before, check if it exists now
            return self.path.exists();
        }

        match (self.mtime, std::fs::metadata(&self.path).ok()) {
            (Some(old_mtime), Some(meta)) => meta
                .modified()
                .ok()
                .is_some_and(|new_mtime| new_mtime > old_mtime),
            (None, Some(_)) => true, // File now has mtime
            (Some(_), None) => true, // File was deleted
            (None, None) => false,
        }
    }

    /// Refresh the mtime from disk.
    pub fn refresh(&mut self) {
        match std::fs::metadata(&self.path) {
            Ok(meta) => {
                self.exists = true;
                self.mtime = meta.modified().ok();
            }
            Err(_) => {
                self.exists = false;
                self.mtime = None;
            }
        }
    }
}

/// Handles config file discovery and loading.
#[derive(Debug, Clone)]
pub struct ConfigLoader {
    /// User-global config (~/.gemicro/config.toml)
    global_source: ConfigSource,

    /// Project-local config (./gemicro.toml)
    local_source: ConfigSource,

    /// Last loaded config
    last_config: GemicroConfig,
}

impl ConfigLoader {
    /// Create a new config loader.
    ///
    /// Discovers config files in standard locations:
    /// - `~/.gemicro/config.toml` (user-global)
    /// - `./gemicro.toml` (project-local)
    pub fn new() -> Self {
        let global_path = Self::global_config_path();
        let local_path = Self::local_config_path();

        Self {
            global_source: ConfigSource::new(global_path),
            local_source: ConfigSource::new(local_path),
            last_config: GemicroConfig::default(),
        }
    }

    /// Get the path to the user-global config file.
    pub fn global_config_path() -> PathBuf {
        dirs::home_dir()
            .map(|h| h.join(".gemicro").join("config.toml"))
            .unwrap_or_else(|| PathBuf::from(".gemicro/config.toml"))
    }

    /// Get the path to the project-local config file.
    pub fn local_config_path() -> PathBuf {
        PathBuf::from("gemicro.toml")
    }

    /// Load config from all sources, merging them together.
    ///
    /// Returns the merged config and a list of files that were loaded.
    pub fn load(&mut self) -> Result<(GemicroConfig, Vec<PathBuf>), ConfigError> {
        let mut config = GemicroConfig::default();
        let mut loaded_files = Vec::new();

        // Load global config first (lower priority)
        if self.global_source.exists {
            match self.load_file(&self.global_source.path) {
                Ok(global_config) => {
                    config.merge(global_config);
                    loaded_files.push(self.global_source.path.clone());
                }
                Err(e) => {
                    log::warn!(
                        "Failed to load global config {}: {}",
                        self.global_source.path.display(),
                        e
                    );
                }
            }
        }

        // Load local config (higher priority, overrides global)
        if self.local_source.exists {
            match self.load_file(&self.local_source.path) {
                Ok(local_config) => {
                    config.merge(local_config);
                    loaded_files.push(self.local_source.path.clone());
                }
                Err(e) => {
                    log::warn!(
                        "Failed to load local config {}: {}",
                        self.local_source.path.display(),
                        e
                    );
                }
            }
        }

        // Refresh mtimes after loading
        self.global_source.refresh();
        self.local_source.refresh();

        self.last_config = config.clone();
        Ok((config, loaded_files))
    }

    /// Load a single config file.
    fn load_file(&self, path: &Path) -> Result<GemicroConfig, ConfigError> {
        let contents = std::fs::read_to_string(path).map_err(|e| ConfigError::Io {
            path: path.to_path_buf(),
            error: e.to_string(),
        })?;

        toml::from_str(&contents).map_err(|e| ConfigError::Parse {
            path: path.to_path_buf(),
            error: e.to_string(),
        })
    }

    /// Check if any config file has been modified since last load.
    pub fn is_stale(&self) -> bool {
        self.global_source.is_stale() || self.local_source.is_stale()
    }

    /// Get the paths of config files that exist.
    pub fn existing_paths(&self) -> Vec<&Path> {
        let mut paths = Vec::new();
        if self.global_source.exists {
            paths.push(self.global_source.path.as_path());
        }
        if self.local_source.exists {
            paths.push(self.local_source.path.as_path());
        }
        paths
    }

    /// Get the last loaded config.
    pub fn last_config(&self) -> &GemicroConfig {
        &self.last_config
    }

    /// Compute the diff between the last config and a new config.
    pub fn diff(&self, new_config: &GemicroConfig) -> Vec<ConfigChange> {
        let mut changes = Vec::new();

        // Compare deep_research configs
        match (&self.last_config.deep_research, &new_config.deep_research) {
            (Some(old), Some(new)) => {
                Self::diff_option(
                    "deep_research.min_sub_queries",
                    &old.min_sub_queries,
                    &new.min_sub_queries,
                    &mut changes,
                );
                Self::diff_option(
                    "deep_research.max_sub_queries",
                    &old.max_sub_queries,
                    &new.max_sub_queries,
                    &mut changes,
                );
                Self::diff_option(
                    "deep_research.max_concurrent_sub_queries",
                    &old.max_concurrent_sub_queries,
                    &new.max_concurrent_sub_queries,
                    &mut changes,
                );
                Self::diff_option(
                    "deep_research.continue_on_partial_failure",
                    &old.continue_on_partial_failure,
                    &new.continue_on_partial_failure,
                    &mut changes,
                );
                Self::diff_option(
                    "deep_research.timeout_secs",
                    &old.timeout_secs,
                    &new.timeout_secs,
                    &mut changes,
                );
                Self::diff_option(
                    "deep_research.use_google_search",
                    &old.use_google_search,
                    &new.use_google_search,
                    &mut changes,
                );

                // Compare prompts
                if let (Some(old_prompts), Some(new_prompts)) = (&old.prompts, &new.prompts) {
                    Self::diff_option(
                        "deep_research.prompts.decomposition_system",
                        &old_prompts.decomposition_system,
                        &new_prompts.decomposition_system,
                        &mut changes,
                    );
                    Self::diff_option(
                        "deep_research.prompts.decomposition_template",
                        &old_prompts.decomposition_template,
                        &new_prompts.decomposition_template,
                        &mut changes,
                    );
                    Self::diff_option(
                        "deep_research.prompts.sub_query_system",
                        &old_prompts.sub_query_system,
                        &new_prompts.sub_query_system,
                        &mut changes,
                    );
                    Self::diff_option(
                        "deep_research.prompts.synthesis_system",
                        &old_prompts.synthesis_system,
                        &new_prompts.synthesis_system,
                        &mut changes,
                    );
                    Self::diff_option(
                        "deep_research.prompts.synthesis_template",
                        &old_prompts.synthesis_template,
                        &new_prompts.synthesis_template,
                        &mut changes,
                    );
                } else if old.prompts.is_some() != new.prompts.is_some() {
                    changes.push(ConfigChange {
                        field: "deep_research.prompts".to_string(),
                        old_value: format!("{:?}", old.prompts.is_some()),
                        new_value: format!("{:?}", new.prompts.is_some()),
                    });
                }
            }
            (None, Some(_)) => {
                changes.push(ConfigChange {
                    field: "deep_research".to_string(),
                    old_value: "(none)".to_string(),
                    new_value: "(added)".to_string(),
                });
            }
            (Some(_), None) => {
                changes.push(ConfigChange {
                    field: "deep_research".to_string(),
                    old_value: "(present)".to_string(),
                    new_value: "(removed)".to_string(),
                });
            }
            (None, None) => {}
        }

        // Compare tool_agent configs
        match (&self.last_config.tool_agent, &new_config.tool_agent) {
            (Some(old), Some(new)) => {
                Self::diff_option(
                    "tool_agent.timeout_secs",
                    &old.timeout_secs,
                    &new.timeout_secs,
                    &mut changes,
                );
                Self::diff_option(
                    "tool_agent.system_prompt",
                    &old.system_prompt,
                    &new.system_prompt,
                    &mut changes,
                );
            }
            (None, Some(_)) => {
                changes.push(ConfigChange {
                    field: "tool_agent".to_string(),
                    old_value: "(none)".to_string(),
                    new_value: "(added)".to_string(),
                });
            }
            (Some(_), None) => {
                changes.push(ConfigChange {
                    field: "tool_agent".to_string(),
                    old_value: "(present)".to_string(),
                    new_value: "(removed)".to_string(),
                });
            }
            (None, None) => {}
        }

        changes
    }

    fn diff_option<T: std::fmt::Debug + PartialEq>(
        field: &str,
        old: &Option<T>,
        new: &Option<T>,
        changes: &mut Vec<ConfigChange>,
    ) {
        if old != new {
            changes.push(ConfigChange {
                field: field.to_string(),
                old_value: format!("{:?}", old),
                new_value: format!("{:?}", new),
            });
        }
    }
}

impl Default for ConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a config value change.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ConfigChange {
    /// The field that changed
    pub field: String,
    /// Old value (as debug string)
    pub old_value: String,
    /// New value (as debug string)
    pub new_value: String,
}

impl std::fmt::Display for ConfigChange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {} -> {}",
            self.field, self.old_value, self.new_value
        )
    }
}

/// Errors that can occur during config loading.
#[derive(Debug)]
#[non_exhaustive]
pub enum ConfigError {
    /// IO error reading config file
    Io { path: PathBuf, error: String },
    /// Parse error in config file
    Parse { path: PathBuf, error: String },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io { path, error } => {
                write!(f, "Failed to read {}: {}", path.display(), error)
            }
            ConfigError::Parse { path, error } => {
                write!(f, "Failed to parse {}: {}", path.display(), error)
            }
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_config_source_new_nonexistent() {
        let source = ConfigSource::new(PathBuf::from("/nonexistent/path/config.toml"));
        assert!(!source.exists);
        assert!(source.mtime.is_none());
    }

    #[test]
    fn test_config_source_staleness() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");

        // File doesn't exist yet
        let source = ConfigSource::new(path.clone());
        assert!(!source.exists);

        // Create the file
        std::fs::write(&path, "").unwrap();

        // Should detect as stale (file now exists)
        assert!(source.is_stale());
    }

    #[test]
    #[serial]
    fn test_load_empty_config() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("gemicro.toml");

        std::fs::write(&path, "").unwrap();

        // Change to temp dir to test local config loading
        let original_dir = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir.path()).unwrap();

        let mut loader = ConfigLoader::new();
        let (config, _) = loader.load().unwrap();

        std::env::set_current_dir(original_dir).unwrap();

        assert!(config.is_empty());
    }

    #[test]
    #[serial]
    fn test_load_config_with_values() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("gemicro.toml");

        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(
            file,
            r#"
            [deep_research]
            min_sub_queries = 2
            max_sub_queries = 8
        "#
        )
        .unwrap();

        let original_dir = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir.path()).unwrap();

        let mut loader = ConfigLoader::new();
        let (config, loaded_files) = loader.load().unwrap();

        std::env::set_current_dir(original_dir).unwrap();

        assert!(!loaded_files.is_empty());
        let dr = config.deep_research.unwrap();
        assert_eq!(dr.min_sub_queries, Some(2));
        assert_eq!(dr.max_sub_queries, Some(8));
    }

    #[test]
    fn test_diff_configs() {
        let old = GemicroConfig {
            deep_research: Some(super::super::types::DeepResearchToml {
                min_sub_queries: Some(3),
                max_sub_queries: Some(5),
                ..Default::default()
            }),
            tool_agent: None,
        };

        let new = GemicroConfig {
            deep_research: Some(super::super::types::DeepResearchToml {
                min_sub_queries: Some(3),
                max_sub_queries: Some(10), // Changed
                timeout_secs: Some(120),   // Added
                ..Default::default()
            }),
            tool_agent: None,
        };

        let mut loader = ConfigLoader::new();
        loader.last_config = old;

        let changes = loader.diff(&new);

        assert_eq!(changes.len(), 2);
        assert!(changes
            .iter()
            .any(|c| c.field == "deep_research.max_sub_queries"));
        assert!(changes
            .iter()
            .any(|c| c.field == "deep_research.timeout_secs"));
    }
}
