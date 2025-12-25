//! Dataset loading for evaluation.
//!
//! Provides the [`Dataset`] trait and built-in loaders for common benchmarks.

use crate::results::EvalQuestion;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that can occur when loading datasets.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DatasetError {
    /// Failed to download dataset
    #[error("Failed to download dataset: {0}")]
    Download(String),

    /// Failed to read dataset file
    #[error("Failed to read dataset: {0}")]
    Io(#[from] std::io::Error),

    /// Failed to parse dataset
    #[error("Failed to parse dataset: {0}")]
    Parse(String),

    /// Cache directory could not be created
    #[error("Failed to create cache directory: {0}")]
    CacheDir(String),
}

/// Trait for evaluation datasets.
///
/// Implement this trait to add support for custom datasets.
///
/// # Example
///
/// ```ignore
/// struct MyDataset {
///     path: PathBuf,
/// }
///
/// impl Dataset for MyDataset {
///     fn name(&self) -> &str {
///         "my_dataset"
///     }
///
///     async fn load(&self, sample_size: Option<usize>) -> Result<Vec<EvalQuestion>, DatasetError> {
///         // Load questions from self.path
///         // ...
///     }
/// }
/// ```
pub trait Dataset: Send + Sync {
    /// The name of this dataset (used in reports).
    fn name(&self) -> &str;

    /// Load questions from the dataset.
    ///
    /// If `sample_size` is specified, return at most that many questions.
    /// The implementation may sample randomly or take the first N questions.
    fn load(
        &self,
        sample_size: Option<usize>,
    ) -> impl std::future::Future<Output = Result<Vec<EvalQuestion>, DatasetError>> + Send;
}

/// HotpotQA dataset loader.
///
/// Automatically downloads and caches the HotpotQA dev set from the official source.
/// Uses the distractor setting by default.
///
/// # Example
///
/// ```no_run
/// use gemicro_eval::HotpotQA;
/// use gemicro_eval::Dataset;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = HotpotQA::new()?;
/// let questions = dataset.load(Some(100)).await?;
/// println!("Loaded {} questions", questions.len());
/// # Ok(())
/// # }
/// ```
pub struct HotpotQA {
    cache_dir: PathBuf,
    /// URL to download the dataset from
    url: String,
}

impl HotpotQA {
    /// Default URL for HotpotQA dev set (distractor setting).
    const DEFAULT_URL: &'static str =
        "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json";

    /// Cache filename
    const CACHE_FILE: &'static str = "hotpot_dev_distractor_v1.json";

    /// Create a new HotpotQA loader with default cache directory.
    ///
    /// The cache directory is `~/.cache/gemicro-eval/hotpotqa/`.
    pub fn new() -> Result<Self, DatasetError> {
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| DatasetError::CacheDir("Could not find cache directory".to_string()))?
            .join("gemicro-eval")
            .join("hotpotqa");

        Ok(Self {
            cache_dir,
            url: Self::DEFAULT_URL.to_string(),
        })
    }

    /// Create a loader with a custom cache directory.
    pub fn with_cache_dir(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            url: Self::DEFAULT_URL.to_string(),
        }
    }

    /// Create a loader from a local file (skip download).
    pub fn from_file(path: PathBuf) -> Self {
        let cache_dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        Self {
            cache_dir,
            url: String::new(), // Empty URL means use local file only
        }
    }

    fn cache_path(&self) -> PathBuf {
        self.cache_dir.join(Self::CACHE_FILE)
    }

    async fn ensure_downloaded(&self) -> Result<PathBuf, DatasetError> {
        let cache_path = self.cache_path();

        // Check if already cached
        if cache_path.exists() {
            log::debug!("Using cached HotpotQA from {:?}", cache_path);
            return Ok(cache_path);
        }

        // Create cache directory
        std::fs::create_dir_all(&self.cache_dir).map_err(|e| {
            DatasetError::CacheDir(format!("Failed to create {:?}: {}", self.cache_dir, e))
        })?;

        // Download
        log::info!("Downloading HotpotQA dataset...");
        let response = reqwest::get(&self.url)
            .await
            .map_err(|e| DatasetError::Download(e.to_string()))?;

        if !response.status().is_success() {
            return Err(DatasetError::Download(format!(
                "HTTP {}: {}",
                response.status(),
                response.status().canonical_reason().unwrap_or("Unknown")
            )));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| DatasetError::Download(e.to_string()))?;

        std::fs::write(&cache_path, &bytes)?;
        log::info!("Cached HotpotQA to {:?}", cache_path);

        Ok(cache_path)
    }
}

impl Dataset for HotpotQA {
    fn name(&self) -> &str {
        "hotpotqa"
    }

    async fn load(&self, sample_size: Option<usize>) -> Result<Vec<EvalQuestion>, DatasetError> {
        let path = self.ensure_downloaded().await?;

        let content = std::fs::read_to_string(&path)?;
        let data: Vec<HotpotQAEntry> =
            serde_json::from_str(&content).map_err(|e| DatasetError::Parse(e.to_string()))?;

        let mut questions: Vec<EvalQuestion> = data
            .into_iter()
            .map(|entry| EvalQuestion {
                id: entry.id,
                question: entry.question,
                ground_truth: entry.answer,
            })
            .collect();

        // Apply sample size limit
        if let Some(size) = sample_size {
            questions.truncate(size);
        }

        Ok(questions)
    }
}

/// Internal structure for parsing HotpotQA JSON.
#[derive(serde::Deserialize)]
struct HotpotQAEntry {
    #[serde(rename = "_id")]
    id: String,
    question: String,
    answer: String,
    // We ignore context, supporting_facts, type, level for evaluation
}

/// A dataset loaded from a JSON file.
///
/// Expects a JSON array of objects with `id`, `question`, and `ground_truth` (or `answer`) fields.
///
/// # Example JSON format
///
/// ```json
/// [
///   {"id": "q1", "question": "What is 2+2?", "ground_truth": "4"},
///   {"id": "q2", "question": "Capital of France?", "answer": "Paris"}
/// ]
/// ```
pub struct JsonFileDataset {
    path: PathBuf,
    name: String,
}

impl JsonFileDataset {
    /// Create a dataset from a JSON file.
    pub fn new(path: PathBuf) -> Self {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("json_dataset")
            .to_string();

        Self { path, name }
    }

    /// Create a dataset with a custom name.
    pub fn with_name(path: PathBuf, name: String) -> Self {
        Self { path, name }
    }
}

impl Dataset for JsonFileDataset {
    fn name(&self) -> &str {
        &self.name
    }

    async fn load(&self, sample_size: Option<usize>) -> Result<Vec<EvalQuestion>, DatasetError> {
        let content = std::fs::read_to_string(&self.path)?;
        let data: Vec<JsonEntry> =
            serde_json::from_str(&content).map_err(|e| DatasetError::Parse(e.to_string()))?;

        let mut questions: Vec<EvalQuestion> = data
            .into_iter()
            .map(|entry| EvalQuestion {
                id: entry.id,
                question: entry.question,
                ground_truth: entry.ground_truth.or(entry.answer).unwrap_or_default(),
            })
            .collect();

        if let Some(size) = sample_size {
            questions.truncate(size);
        }

        Ok(questions)
    }
}

/// Internal structure for parsing generic JSON datasets.
#[derive(serde::Deserialize)]
struct JsonEntry {
    id: String,
    question: String,
    #[serde(default)]
    ground_truth: Option<String>,
    #[serde(default)]
    answer: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_hotpotqa_cache_path() {
        let loader = HotpotQA::with_cache_dir(PathBuf::from("/tmp/test-cache"));
        assert_eq!(
            loader.cache_path(),
            PathBuf::from("/tmp/test-cache/hotpot_dev_distractor_v1.json")
        );
    }

    #[tokio::test]
    async fn test_json_file_dataset() {
        let json = r#"[
            {"id": "1", "question": "Q1?", "ground_truth": "A1"},
            {"id": "2", "question": "Q2?", "answer": "A2"}
        ]"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(json.as_bytes()).unwrap();

        let dataset = JsonFileDataset::new(file.path().to_path_buf());
        let questions = dataset.load(None).await.unwrap();

        assert_eq!(questions.len(), 2);
        assert_eq!(questions[0].ground_truth, "A1");
        assert_eq!(questions[1].ground_truth, "A2");
    }

    #[tokio::test]
    async fn test_json_file_dataset_sample_size() {
        let json = r#"[
            {"id": "1", "question": "Q1?", "ground_truth": "A1"},
            {"id": "2", "question": "Q2?", "ground_truth": "A2"},
            {"id": "3", "question": "Q3?", "ground_truth": "A3"}
        ]"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(json.as_bytes()).unwrap();

        let dataset = JsonFileDataset::new(file.path().to_path_buf());
        let questions = dataset.load(Some(2)).await.unwrap();

        assert_eq!(questions.len(), 2);
    }

    #[test]
    fn test_json_file_dataset_name() {
        let dataset = JsonFileDataset::new(PathBuf::from("/path/to/my_questions.json"));
        assert_eq!(dataset.name(), "my_questions");

        let dataset =
            JsonFileDataset::with_name(PathBuf::from("/path/to/file.json"), "custom".to_string());
        assert_eq!(dataset.name(), "custom");
    }
}
