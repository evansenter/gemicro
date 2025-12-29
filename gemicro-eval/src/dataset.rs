//! Dataset loading for evaluation.
//!
//! Provides the [`Dataset`] trait and built-in loaders for common benchmarks.

use crate::results::EvalQuestion;
use gemicro_core::Trajectory;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tokio::fs;

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
/// ```text
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
/// # Security Note
///
/// The official HotpotQA dataset is only available over HTTP (not HTTPS).
/// For production use, consider using [`HotpotQA::from_file`] with a
/// pre-downloaded and verified dataset file.
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

    /// Create a loader from a local file (skips download).
    ///
    /// Use this when you have a pre-downloaded and verified HotpotQA dataset,
    /// or for offline/air-gapped environments.
    ///
    /// The file must be named `hotpot_dev_distractor_v1.json` and be in the
    /// HotpotQA JSON format.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gemicro_eval::HotpotQA;
    /// use std::path::PathBuf;
    ///
    /// // Use a pre-verified local copy
    /// let dataset = HotpotQA::from_file(PathBuf::from("/data/hotpotqa/"));
    /// ```
    pub fn from_file(path: PathBuf) -> Self {
        let cache_dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        Self {
            cache_dir,
            url: String::new(), // Empty URL skips download
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
        fs::create_dir_all(&self.cache_dir).await.map_err(|e| {
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

        fs::write(&cache_path, &bytes).await?;
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

        let content = fs::read_to_string(&path).await?;
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
        let content = fs::read_to_string(&self.path).await?;
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

/// GSM8K (Grade School Math 8K) dataset loader.
///
/// Loads the OpenAI GSM8K math word problem dataset. Each problem requires
/// 2-8 steps of basic arithmetic to solve, making it ideal for testing
/// calculator tool use.
///
/// The dataset is available from GitHub:
/// <https://github.com/openai/grade-school-math>
///
/// # Format
///
/// GSM8K uses JSONL format (one JSON object per line):
/// ```json
/// {"question": "Janet's ducks lay 16 eggs...", "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9...\n#### 18"}
/// ```
///
/// The final numeric answer is extracted from after the `####` marker.
///
/// # Example
///
/// ```no_run
/// use gemicro_eval::{GSM8K, Dataset};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = GSM8K::new()?;
/// let questions = dataset.load(Some(50)).await?;
/// println!("Loaded {} math problems", questions.len());
/// # Ok(())
/// # }
/// ```
pub struct GSM8K {
    /// Either a cache directory (when downloading) or the explicit file path (when loading from file)
    path: PathBuf,
    /// URL to download the dataset from (empty if loading from local file)
    url: String,
    /// Which split to use (test or train)
    split: GSM8KSplit,
    /// Whether path is a direct file path (true) or cache directory (false)
    is_direct_path: bool,
}

/// GSM8K dataset split.
#[derive(Debug, Clone, Copy, Default)]
pub enum GSM8KSplit {
    /// Test set (1,319 problems)
    #[default]
    Test,
    /// Training set (7,473 problems)
    Train,
}

impl GSM8K {
    /// Base URL for GSM8K raw data
    const BASE_URL: &'static str =
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data";

    /// Create a new GSM8K loader with default cache directory.
    ///
    /// Uses the test split by default. The cache directory is `~/.cache/gemicro-eval/gsm8k/`.
    pub fn new() -> Result<Self, DatasetError> {
        Self::with_split(GSM8KSplit::Test)
    }

    /// Create a GSM8K loader for a specific split.
    pub fn with_split(split: GSM8KSplit) -> Result<Self, DatasetError> {
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| DatasetError::CacheDir("Could not find cache directory".to_string()))?
            .join("gemicro-eval")
            .join("gsm8k");

        let url = match split {
            GSM8KSplit::Test => format!("{}/test.jsonl", Self::BASE_URL),
            GSM8KSplit::Train => format!("{}/train.jsonl", Self::BASE_URL),
        };

        Ok(Self {
            path: cache_dir,
            url,
            split,
            is_direct_path: false,
        })
    }

    /// Create a loader with a custom cache directory.
    pub fn with_cache_dir(cache_dir: PathBuf, split: GSM8KSplit) -> Self {
        let url = match split {
            GSM8KSplit::Test => format!("{}/test.jsonl", Self::BASE_URL),
            GSM8KSplit::Train => format!("{}/train.jsonl", Self::BASE_URL),
        };

        Self {
            path: cache_dir,
            url,
            split,
            is_direct_path: false,
        }
    }

    /// Create a loader from a local file (skips download).
    ///
    /// Use this when you have a pre-downloaded GSM8K dataset file.
    pub fn from_file(path: PathBuf) -> Self {
        Self {
            path,
            url: String::new(), // Empty URL skips download
            split: GSM8KSplit::Test,
            is_direct_path: true,
        }
    }

    fn cache_filename(&self) -> &'static str {
        match self.split {
            GSM8KSplit::Test => "test.jsonl",
            GSM8KSplit::Train => "train.jsonl",
        }
    }

    fn cache_path(&self) -> PathBuf {
        if self.is_direct_path {
            self.path.clone()
        } else {
            self.path.join(self.cache_filename())
        }
    }

    async fn ensure_downloaded(&self) -> Result<PathBuf, DatasetError> {
        let cache_path = self.cache_path();

        // Check if already cached or is a direct file path
        if cache_path.exists() {
            log::debug!("Using GSM8K from {:?}", cache_path);
            return Ok(cache_path);
        }

        // Skip download if no URL (local file mode)
        if self.url.is_empty() {
            return Err(DatasetError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("GSM8K file not found: {:?}", cache_path),
            )));
        }

        // Create cache directory
        fs::create_dir_all(&self.path).await.map_err(|e| {
            DatasetError::CacheDir(format!("Failed to create {:?}: {}", self.path, e))
        })?;

        // Download
        log::info!("Downloading GSM8K {} dataset...", self.cache_filename());
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

        fs::write(&cache_path, &bytes).await?;
        log::info!("Cached GSM8K to {:?}", cache_path);

        Ok(cache_path)
    }

    /// Extract the final numeric answer from a GSM8K answer string.
    ///
    /// The answer format is: `step-by-step reasoning...\n#### final_answer`
    fn extract_answer(answer: &str) -> String {
        if let Some(idx) = answer.rfind("####") {
            let after_marker = &answer[idx + 4..];
            // Clean up the answer: remove whitespace and commas
            after_marker
                .trim()
                .replace(',', "")
                .split_whitespace()
                .next()
                .unwrap_or("")
                .to_string()
        } else {
            // No marker found, return as-is
            answer.trim().to_string()
        }
    }
}

impl Dataset for GSM8K {
    fn name(&self) -> &str {
        "gsm8k"
    }

    async fn load(&self, sample_size: Option<usize>) -> Result<Vec<EvalQuestion>, DatasetError> {
        let path = self.ensure_downloaded().await?;

        let content = fs::read_to_string(&path).await?;

        let mut questions = Vec::new();
        for (idx, line) in content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }

            let entry: GSM8KEntry =
                serde_json::from_str(line).map_err(|e| DatasetError::Parse(e.to_string()))?;

            questions.push(EvalQuestion {
                id: format!("gsm8k_{}", idx),
                question: entry.question,
                ground_truth: Self::extract_answer(&entry.answer),
            });
        }

        // Apply sample size limit
        if let Some(size) = sample_size {
            questions.truncate(size);
        }

        Ok(questions)
    }
}

/// Internal structure for parsing GSM8K JSONL entries.
#[derive(serde::Deserialize)]
struct GSM8KEntry {
    question: String,
    answer: String,
}

/// A dataset created from saved trajectory files.
///
/// This allows evaluating agents on previously recorded runs, enabling:
/// - Regression testing against known-good trajectories
/// - Offline evaluation without API calls (using MockLlmClient for replay)
/// - Building evaluation sets from production data
///
/// # Directory Structure
///
/// The dataset expects a directory containing JSON trajectory files:
/// ```text
/// trajectories/
/// ├── run_001.json
/// ├── run_002.json
/// └── run_003.json
/// ```
///
/// Each file should be a valid `Trajectory` JSON (as created by `Trajectory::save()`).
/// Files without `.json` extension are ignored.
///
/// # Example
///
/// ```no_run
/// use gemicro_eval::{TrajectoryDataset, Dataset};
/// use std::path::PathBuf;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = TrajectoryDataset::new(PathBuf::from("trajectories/"));
/// let questions = dataset.load(Some(10)).await?;
///
/// // Each question comes from a trajectory:
/// // - id: trajectory.id
/// // - question: trajectory.query
/// // - ground_truth: trajectory.metadata.final_answer
/// println!("Loaded {} trajectory-based questions", questions.len());
/// # Ok(())
/// # }
/// ```
pub struct TrajectoryDataset {
    /// Directory containing trajectory JSON files
    dir: PathBuf,
    /// Custom name for the dataset
    name: String,
}

impl TrajectoryDataset {
    /// Create a dataset from a directory of trajectory files.
    pub fn new(dir: PathBuf) -> Self {
        let name = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("trajectories")
            .to_string();

        Self { dir, name }
    }

    /// Create a dataset with a custom name.
    pub fn with_name(dir: PathBuf, name: impl Into<String>) -> Self {
        Self {
            dir,
            name: name.into(),
        }
    }

    /// Get the directory path.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Load all trajectories from the directory.
    ///
    /// Returns the full `Trajectory` objects for cases where you need
    /// more than just the question/answer pairs (e.g., for replay).
    pub async fn load_trajectories(
        &self,
        sample_size: Option<usize>,
    ) -> Result<Vec<Trajectory>, DatasetError> {
        let mut trajectories = Vec::new();

        if !self.dir.exists() {
            return Err(DatasetError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Trajectory directory not found: {:?}", self.dir),
            )));
        }

        let mut entries = fs::read_dir(&self.dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            // Only process .json files
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }

            match Trajectory::load(&path) {
                Ok(trajectory) => trajectories.push(trajectory),
                Err(e) => {
                    log::warn!("Failed to load trajectory {:?}: {}", path, e);
                }
            }
        }

        // Sort by trajectory ID for deterministic ordering
        trajectories.sort_by(|a, b| a.id.cmp(&b.id));

        // Apply sample size limit
        if let Some(size) = sample_size {
            trajectories.truncate(size);
        }

        Ok(trajectories)
    }
}

impl Dataset for TrajectoryDataset {
    fn name(&self) -> &str {
        &self.name
    }

    async fn load(&self, sample_size: Option<usize>) -> Result<Vec<EvalQuestion>, DatasetError> {
        let trajectories = self.load_trajectories(sample_size).await?;

        let questions = trajectories
            .into_iter()
            .filter_map(|t| {
                // Only include trajectories that have a final answer
                t.metadata.final_answer.map(|answer| EvalQuestion {
                    id: t.id,
                    question: t.query,
                    ground_truth: answer,
                })
            })
            .collect();

        Ok(questions)
    }
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

    // GSM8K tests

    #[test]
    fn test_gsm8k_extract_answer_simple() {
        let answer = "Janet sells 9 duck eggs a day.\n#### 18";
        assert_eq!(GSM8K::extract_answer(answer), "18");
    }

    #[test]
    fn test_gsm8k_extract_answer_with_commas() {
        let answer = "Total is $1,234.\n#### 1,234";
        assert_eq!(GSM8K::extract_answer(answer), "1234");
    }

    #[test]
    fn test_gsm8k_extract_answer_multiline() {
        let answer = "Step 1: 16 - 3 = 13\nStep 2: 13 - 4 = 9\nStep 3: 9 * 2 = 18\n#### 18";
        assert_eq!(GSM8K::extract_answer(answer), "18");
    }

    #[test]
    fn test_gsm8k_extract_answer_no_marker() {
        let answer = "The answer is 42";
        assert_eq!(GSM8K::extract_answer(answer), "The answer is 42");
    }

    #[test]
    fn test_gsm8k_cache_path() {
        let loader = GSM8K::with_cache_dir(PathBuf::from("/tmp/test-cache"), GSM8KSplit::Test);
        assert_eq!(
            loader.cache_path(),
            PathBuf::from("/tmp/test-cache/test.jsonl")
        );

        let loader = GSM8K::with_cache_dir(PathBuf::from("/tmp/test-cache"), GSM8KSplit::Train);
        assert_eq!(
            loader.cache_path(),
            PathBuf::from("/tmp/test-cache/train.jsonl")
        );
    }

    #[tokio::test]
    async fn test_gsm8k_load_from_file() {
        // Create a temporary JSONL file with sample GSM8K data
        let jsonl = r#####"{"question": "What is 2+2?", "answer": "2+2=4\n#### 4"}
{"question": "What is 3*3?", "answer": "3*3=9\n#### 9"}"#####;

        let mut file = NamedTempFile::with_suffix(".jsonl").unwrap();
        file.write_all(jsonl.as_bytes()).unwrap();
        let path = file.path().to_path_buf();
        file.flush().unwrap();

        // Create loader from file (must keep file handle alive for Windows)
        let loader = GSM8K::from_file(path);
        let questions = loader.load(None).await.unwrap();

        assert_eq!(questions.len(), 2);
        assert_eq!(questions[0].id, "gsm8k_0");
        assert_eq!(questions[0].question, "What is 2+2?");
        assert_eq!(questions[0].ground_truth, "4");
        assert_eq!(questions[1].ground_truth, "9");
    }

    #[tokio::test]
    async fn test_gsm8k_load_sample_size() {
        let jsonl = r#####"{"question": "Q1", "answer": "#### 1"}
{"question": "Q2", "answer": "#### 2"}
{"question": "Q3", "answer": "#### 3"}"#####;

        let mut file = NamedTempFile::with_suffix(".jsonl").unwrap();
        file.write_all(jsonl.as_bytes()).unwrap();
        file.flush().unwrap();

        let loader = GSM8K::from_file(file.path().to_path_buf());
        let questions = loader.load(Some(2)).await.unwrap();

        assert_eq!(questions.len(), 2);
    }

    // TrajectoryDataset tests

    #[test]
    fn test_trajectory_dataset_name() {
        let dataset = TrajectoryDataset::new(PathBuf::from("/path/to/trajectories"));
        assert_eq!(dataset.name(), "trajectories");

        let dataset =
            TrajectoryDataset::with_name(PathBuf::from("/path/to/data"), "my_custom_name");
        assert_eq!(dataset.name(), "my_custom_name");
    }

    #[test]
    fn test_trajectory_dataset_dir() {
        let dir = PathBuf::from("/path/to/trajectories");
        let dataset = TrajectoryDataset::new(dir.clone());
        assert_eq!(dataset.dir(), dir);
    }

    #[tokio::test]
    async fn test_trajectory_dataset_load_empty_dir() {
        let temp_dir = tempfile::tempdir().unwrap();
        let dataset = TrajectoryDataset::new(temp_dir.path().to_path_buf());

        let questions = dataset.load(None).await.unwrap();
        assert!(questions.is_empty());
    }

    #[tokio::test]
    async fn test_trajectory_dataset_load_nonexistent_dir() {
        let dataset = TrajectoryDataset::new(PathBuf::from("/nonexistent/path/xyz"));

        let result = dataset.load(None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_trajectory_dataset_load_trajectories() {
        // Create a temporary directory with trajectory files
        let temp_dir = tempfile::tempdir().unwrap();

        // Create a sample trajectory using builder
        let trajectory = Trajectory::builder()
            .query("What is 2+2?")
            .agent_name("test_agent")
            .agent_config(serde_json::json!({}))
            .model("test-model")
            .build(vec![], vec![], 100, Some("4".to_string()));

        // Save trajectory
        trajectory.save(temp_dir.path().join("test.json")).unwrap();

        // Load via TrajectoryDataset
        let dataset = TrajectoryDataset::new(temp_dir.path().to_path_buf());
        let questions = dataset.load(None).await.unwrap();

        assert_eq!(questions.len(), 1);
        // Note: ID is now UUID-based from builder, so we check query/answer instead
        assert_eq!(questions[0].question, "What is 2+2?");
        assert_eq!(questions[0].ground_truth, "4");
    }

    #[tokio::test]
    async fn test_trajectory_dataset_ignores_non_json() {
        let temp_dir = tempfile::tempdir().unwrap();

        // Create a valid trajectory file using builder
        let trajectory = Trajectory::builder()
            .query("Valid query")
            .agent_name("test")
            .agent_config(serde_json::json!({}))
            .model("test")
            .build(vec![], vec![], 100, Some("Valid answer".to_string()));
        trajectory.save(temp_dir.path().join("valid.json")).unwrap();

        // Create a non-JSON file (should be ignored)
        std::fs::write(temp_dir.path().join("notes.txt"), "some notes").unwrap();

        let dataset = TrajectoryDataset::new(temp_dir.path().to_path_buf());
        let questions = dataset.load(None).await.unwrap();

        // Only the valid JSON should be loaded
        assert_eq!(questions.len(), 1);
        assert_eq!(questions[0].question, "Valid query");
    }

    #[tokio::test]
    async fn test_trajectory_dataset_filters_no_answer() {
        let temp_dir = tempfile::tempdir().unwrap();

        // Create trajectory WITH answer using builder
        let with_answer = Trajectory::builder()
            .query("Q1")
            .agent_name("test")
            .agent_config(serde_json::json!({}))
            .model("test")
            .build(vec![], vec![], 100, Some("A1".to_string()));
        with_answer
            .save(temp_dir.path().join("with_answer.json"))
            .unwrap();

        // Create trajectory WITHOUT answer (failed run) using builder
        let without_answer = Trajectory::builder()
            .query("Q2")
            .agent_name("test")
            .agent_config(serde_json::json!({}))
            .model("test")
            .build(vec![], vec![], 100, None); // No answer
        without_answer
            .save(temp_dir.path().join("without_answer.json"))
            .unwrap();

        let dataset = TrajectoryDataset::new(temp_dir.path().to_path_buf());
        let questions = dataset.load(None).await.unwrap();

        // Only trajectories with answers should be included
        assert_eq!(questions.len(), 1);
        assert_eq!(questions[0].question, "Q1");
    }
}
