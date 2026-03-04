use serde::{Deserialize, Serialize};

// =========================================================================
// Request types
// =========================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatTemplateKwargs {
    #[serde(default)]
    pub enable_thinking: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "defaults::max_tokens")]
    pub max_tokens: Option<usize>,
    #[serde(default = "defaults::temperature")]
    pub temperature: Option<f64>,
    #[serde(default = "defaults::top_p")]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<StopSequence>,
    #[serde(default = "defaults::repeat_penalty")]
    pub repeat_penalty: Option<f32>,
    #[serde(default = "defaults::repeat_last_n")]
    pub repeat_last_n: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: StringOrArray,
    #[serde(default = "defaults::max_tokens")]
    pub max_tokens: Option<usize>,
    #[serde(default = "defaults::temperature")]
    pub temperature: Option<f64>,
    #[serde(default = "defaults::top_p")]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<StopSequence>,
    #[serde(default = "defaults::repeat_penalty")]
    pub repeat_penalty: Option<f32>,
    #[serde(default = "defaults::repeat_last_n")]
    pub repeat_last_n: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StringOrArray {
    Single(String),
    Array(Vec<String>),
}

impl StringOrArray {
    pub fn first(&self) -> String {
        match self {
            Self::Single(s) => s.clone(),
            Self::Array(a) => a.first().cloned().unwrap_or_default(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Array(Vec<String>),
}

impl StopSequence {
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s.clone()],
            Self::Array(a) => a.clone(),
        }
    }
}

mod defaults {
    pub fn max_tokens() -> Option<usize> {
        Some(256)
    }
    pub fn temperature() -> Option<f64> {
        Some(0.7)
    }
    pub fn top_p() -> Option<f64> {
        Some(0.9)
    }
    pub fn repeat_penalty() -> Option<f32> {
        Some(1.1)
    }
    pub fn repeat_last_n() -> Option<usize> {
        Some(64)
    }
}

// =========================================================================
// Response types
// =========================================================================

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessageResponse,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct ChatMessageResponse {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

#[derive(Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
}

#[derive(Serialize)]
pub struct CompletionChunkChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Serialize)]
pub struct ModelPermission {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub allow_create_engine: bool,
    pub allow_sampling: bool,
    pub allow_logprobs: bool,
    pub allow_search_indices: bool,
    pub allow_view: bool,
    pub allow_fine_tuning: bool,
    pub organization: String,
    pub group: Option<String>,
    pub is_blocking: bool,
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
    pub root: String,
    pub parent: Option<String>,
    pub max_model_len: usize,
    pub permission: Vec<ModelPermission>,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: Option<String>,
}

// =========================================================================
// GenParams
// =========================================================================

pub struct GenParams {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
    pub stop: Vec<String>,
}

impl GenParams {
    pub fn from_chat(req: &ChatCompletionRequest) -> Self {
        Self::from_common(
            req.max_tokens,
            req.temperature,
            req.top_p,
            req.top_k,
            req.repeat_penalty,
            req.repeat_last_n,
            req.seed,
            req.stop.as_ref(),
        )
    }

    pub fn from_completion(req: &CompletionRequest) -> Self {
        Self::from_common(
            req.max_tokens,
            req.temperature,
            req.top_p,
            req.top_k,
            req.repeat_penalty,
            req.repeat_last_n,
            req.seed,
            req.stop.as_ref(),
        )
    }

    fn from_common(
        max_tokens: Option<usize>,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        repeat_penalty: Option<f32>,
        repeat_last_n: Option<usize>,
        seed: Option<u64>,
        stop: Option<&StopSequence>,
    ) -> Self {
        Self {
            max_tokens: max_tokens.unwrap_or(256),
            temperature: temperature.unwrap_or(0.7),
            top_p: top_p.unwrap_or(0.9),
            top_k: top_k.unwrap_or(40),
            repeat_penalty: repeat_penalty.unwrap_or(1.1),
            repeat_last_n: repeat_last_n.unwrap_or(64),
            seed: seed.unwrap_or_else(rand::random),
            stop: stop.map(|s| s.to_vec()).unwrap_or_default(),
        }
    }
}
