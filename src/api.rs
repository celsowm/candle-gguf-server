use actix_web::rt::time::interval;
use actix_web::{web, HttpRequest, HttpResponse};
use bytes::Bytes;
use futures::stream;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error, info};

use crate::middleware::check_auth;
use crate::tokenizer_utils::{apply_template, detect_template};
use crate::AppState;

// =========================================================================
// Request types
// =========================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
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
}

#[derive(Debug, Deserialize)]
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
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
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
// Shared helpers
// =========================================================================

struct GenParams {
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    top_k: usize,
    repeat_penalty: f32,
    repeat_last_n: usize,
    seed: u64,
    stop: Vec<String>,
}

impl GenParams {
    fn from_chat(req: &ChatCompletionRequest) -> Self {
        Self {
            max_tokens: req.max_tokens.unwrap_or(256),
            temperature: req.temperature.unwrap_or(0.7),
            top_p: req.top_p.unwrap_or(0.9),
            top_k: req.top_k.unwrap_or(40),
            repeat_penalty: req.repeat_penalty.unwrap_or(1.1),
            repeat_last_n: req.repeat_last_n.unwrap_or(64),
            seed: req.seed.unwrap_or_else(rand::random),
            stop: req.stop.as_ref().map(|s| s.to_vec()).unwrap_or_default(),
        }
    }

    fn from_completion(req: &CompletionRequest) -> Self {
        Self {
            max_tokens: req.max_tokens.unwrap_or(256),
            temperature: req.temperature.unwrap_or(0.7),
            top_p: req.top_p.unwrap_or(0.9),
            top_k: req.top_k.unwrap_or(40),
            repeat_penalty: req.repeat_penalty.unwrap_or(1.1),
            repeat_last_n: req.repeat_last_n.unwrap_or(64),
            seed: req.seed.unwrap_or_else(rand::random),
            stop: req.stop.as_ref().map(|s| s.to_vec()).unwrap_or_default(),
        }
    }
}

fn make_error(status: actix_web::http::StatusCode, msg: impl ToString) -> HttpResponse {
    HttpResponse::build(status).json(ErrorResponse {
        error: ErrorDetail {
            message: msg.to_string(),
            r#type: "server_error".to_string(),
            code: None,
        },
    })
}

fn make_validation_error(msg: impl ToString) -> HttpResponse {
    HttpResponse::BadRequest().json(ErrorResponse {
        error: ErrorDetail {
            message: msg.to_string(),
            r#type: "invalid_request_error".to_string(),
            code: Some("invalid_request".to_string()),
        },
    })
}

fn request_id(prefix: &str) -> String {
    format!("{}-{}", prefix, uuid::Uuid::new_v4())
}

fn sse_line<T: Serialize>(payload: &T) -> Bytes {
    let json = serde_json::to_string(payload).unwrap();
    Bytes::from(format!("data: {}\n\n", json))
}

fn sse_done() -> Bytes {
    Bytes::from("data: [DONE]\n\n")
}

// =========================================================================
// Validation
// =========================================================================

fn validate_chat_request(req: &ChatCompletionRequest) -> Result<(), HttpResponse> {
    if req.messages.is_empty() {
        return Err(make_validation_error("'messages' must not be empty"));
    }

    for (i, msg) in req.messages.iter().enumerate() {
        if !["system", "user", "assistant"].contains(&msg.role.as_str()) {
            return Err(make_validation_error(format!(
                "Invalid role '{}' at messages[{}]",
                msg.role, i
            )));
        }
    }

    if let Some(temp) = req.temperature {
        if temp < 0.0 || temp > 2.0 {
            return Err(make_validation_error(
                "temperature must be between 0.0 and 2.0",
            ));
        }
    }

    if let Some(top_p) = req.top_p {
        if top_p < 0.0 || top_p > 1.0 {
            return Err(make_validation_error("top_p must be between 0.0 and 1.0"));
        }
    }

    if let Some(max_tokens) = req.max_tokens {
        if max_tokens == 0 {
            return Err(make_validation_error("max_tokens must be greater than 0"));
        }
    }

    Ok(())
}

fn validate_completion_request(req: &CompletionRequest) -> Result<(), HttpResponse> {
    if req.prompt.first().is_empty() {
        return Err(make_validation_error("'prompt' must not be empty"));
    }

    if let Some(temp) = req.temperature {
        if temp < 0.0 || temp > 2.0 {
            return Err(make_validation_error(
                "temperature must be between 0.0 and 2.0",
            ));
        }
    }

    Ok(())
}

// =========================================================================
// Handlers
// =========================================================================

pub async fn health(state: web::Data<Arc<AppState>>) -> HttpResponse {
    let uptime = state.start_time.elapsed().as_secs();
    let reqs = state.request_count.load(std::sync::atomic::Ordering::Relaxed);
    let toks = state.tokens_generated.load(std::sync::atomic::Ordering::Relaxed);

    HttpResponse::Ok().json(serde_json::json!({
        "status": "ok",
        "model": state.model_name,
        "uptime_seconds": uptime,
        "total_requests": reqs,
        "total_tokens_generated": toks,
    }))
}

pub async fn metrics(state: web::Data<Arc<AppState>>) -> HttpResponse {
    let uptime = state.start_time.elapsed().as_secs_f64();
    let reqs = state.request_count.load(std::sync::atomic::Ordering::Relaxed);
    let toks = state.tokens_generated.load(std::sync::atomic::Ordering::Relaxed);

    // Prometheus-style plain text metrics
    let body = format!(
        "# HELP candle_gguf_uptime_seconds Server uptime in seconds\n\
         # TYPE candle_gguf_uptime_seconds gauge\n\
         candle_gguf_uptime_seconds {:.1}\n\
         # HELP candle_gguf_requests_total Total inference requests\n\
         # TYPE candle_gguf_requests_total counter\n\
         candle_gguf_requests_total {}\n\
         # HELP candle_gguf_tokens_generated_total Total tokens generated\n\
         # TYPE candle_gguf_tokens_generated_total counter\n\
         candle_gguf_tokens_generated_total {}\n",
        uptime, reqs, toks
    );

    HttpResponse::Ok()
        .content_type("text/plain; version=0.0.4")
        .body(body)
}

pub async fn not_found() -> HttpResponse {
    HttpResponse::NotFound().json(ErrorResponse {
        error: ErrorDetail {
            message: "Unknown endpoint. See /v1/models, /v1/chat/completions, /v1/completions"
                .to_string(),
            r#type: "invalid_request_error".to_string(),
            code: Some("not_found".to_string()),
        },
    })
}

pub async fn list_models(state: web::Data<Arc<AppState>>) -> HttpResponse {
    HttpResponse::Ok().json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_name.clone(),
            object: "model".to_string(),
            created: chrono::Utc::now().timestamp(),
            owned_by: "local".to_string(),
        }],
    })
}

pub async fn get_model(
    state: web::Data<Arc<AppState>>,
    path: web::Path<String>,
) -> HttpResponse {
    let id = path.into_inner();
    if id == state.model_name {
        HttpResponse::Ok().json(ModelInfo {
            id: state.model_name.clone(),
            object: "model".to_string(),
            created: chrono::Utc::now().timestamp(),
            owned_by: "local".to_string(),
        })
    } else {
        HttpResponse::NotFound().json(ErrorResponse {
            error: ErrorDetail {
                message: format!("Model '{}' not found", id),
                r#type: "invalid_request_error".to_string(),
                code: Some("model_not_found".to_string()),
            },
        })
    }
}

// =========================================================================
// Chat Completions
// =========================================================================

pub async fn chat_completions(
    state: web::Data<Arc<AppState>>,
    req: web::Json<ChatCompletionRequest>,
    http_req: HttpRequest,
) -> HttpResponse {
    // Auth check
    if let Some(err) = check_auth(&http_req, &state) {
        return err;
    }

    // Validation
    if let Err(err) = validate_chat_request(&req) {
        return err;
    }

    let is_stream = req.stream.unwrap_or(false);
    info!(
        "POST /v1/chat/completions  stream={}  messages={}",
        is_stream,
        req.messages.len()
    );

    state.increment_requests();

    // Acquire concurrency permit
    let _permit = match state.concurrency_semaphore.acquire().await {
        Ok(p) => p,
        Err(_) => {
            return make_error(
                actix_web::http::StatusCode::SERVICE_UNAVAILABLE,
                "Server is shutting down",
            );
        }
    };

    let params = GenParams::from_chat(&req);
    let id = request_id("chatcmpl");
    let model = state.model_name.clone();

    let mut engine = state.engine.lock().await;

    let template = detect_template(engine.tokenizer());
    debug!("Detected chat template: {:?}", template);
    let prompt = apply_template(&req.messages, &template);

    if is_stream {
        return stream_chat_response(engine, prompt, params, id, model).await;
    }

    // Non-streaming
    match engine.generate(
        &prompt,
        params.max_tokens,
        params.temperature,
        params.top_p,
        params.top_k,
        params.repeat_penalty,
        params.repeat_last_n,
        params.seed,
        &params.stop,
    ) {
        Ok(r) => {
            state.add_tokens(r.completion_tokens as u64);
            info!(
                "Generated {} tokens in {}ms ({:.1} tok/s)",
                r.completion_tokens, r.generation_time_ms, r.tokens_per_second
            );

            HttpResponse::Ok().json(ChatCompletionResponse {
                id,
                object: "chat.completion".to_string(),
                created: chrono::Utc::now().timestamp(),
                model,
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessageResponse {
                        role: "assistant".to_string(),
                        content: r.text,
                    },
                    finish_reason: r.finish_reason,
                }],
                usage: Usage {
                    prompt_tokens: r.prompt_tokens,
                    completion_tokens: r.completion_tokens,
                    total_tokens: r.prompt_tokens + r.completion_tokens,
                },
            })
        }
        Err(e) => {
            error!("Generation error: {:#}", e);
            make_error(
                actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Generation failed: {}", e),
            )
        }
    }
}

/// True SSE streaming using a channel.
/// Generation runs in a background task, tokens are sent through an mpsc channel,
/// and we stream them as SSE events to the client in real time.
async fn stream_chat_response(
    mut engine: tokio::sync::MutexGuard<'_, crate::model::ModelEngine>,
    prompt: String,
    params: GenParams,
    id: String,
    model: String,
) -> HttpResponse {
    let (tx, rx) = mpsc::channel::<Bytes>(64);
    let created = chrono::Utc::now().timestamp();

    // Spawn generation in a blocking task to not block the event loop
    let _handle = tokio::task::spawn_blocking(move || {
        // Send initial role chunk
        let role_chunk = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        let _ = tx.try_send(sse_line(&role_chunk));

        let id_clone = id.clone();
        let model_clone = model.clone();
        let tx_clone = tx.clone();

        // Run generation, sending deltas through the channel
        let result = engine.generate_streaming(
            &prompt,
            params.max_tokens,
            params.temperature,
            params.top_p,
            params.top_k,
            params.repeat_penalty,
            params.repeat_last_n,
            params.seed,
            &params.stop,
            |delta: &str| {
                let chunk = ChatCompletionChunk {
                    id: id_clone.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_clone.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: Some(delta.to_string()),
                        },
                        finish_reason: None,
                    }],
                };
                tx_clone.try_send(sse_line(&chunk)).is_ok()
            },
        );

        // Send final chunk
        if let Ok(r) = &result {
            let final_chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some(r.finish_reason.clone()),
                }],
            };
            let _ = tx.try_send(sse_line(&final_chunk));
        }
        let _ = tx.try_send(sse_done());
        
        // Explicitly drop the sender to signal end of stream
        drop(tx);
    });

    // Convert the receiver into a stream
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);

    HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("Connection", "keep-alive"))
        .insert_header(("X-Accel-Buffering", "no"))
        .streaming(stream)
}

// =========================================================================
// Text Completions
// =========================================================================

pub async fn completions(
    state: web::Data<Arc<AppState>>,
    req: web::Json<CompletionRequest>,
    http_req: HttpRequest,
) -> HttpResponse {
    if let Some(err) = check_auth(&http_req, &state) {
        return err;
    }

    if let Err(err) = validate_completion_request(&req) {
        return err;
    }

    let is_stream = req.stream.unwrap_or(false);
    info!("POST /v1/completions  stream={}", is_stream);

    state.increment_requests();

    let _permit = match state.concurrency_semaphore.acquire().await {
        Ok(p) => p,
        Err(_) => {
            return make_error(
                actix_web::http::StatusCode::SERVICE_UNAVAILABLE,
                "Server is shutting down",
            );
        }
    };

    let params = GenParams::from_completion(&req);
    let prompt = req.prompt.first();
    let id = request_id("cmpl");
    let model = state.model_name.clone();
    let mut engine = state.engine.lock().await;

    if is_stream {
        return stream_completion_response(engine, prompt, params, id, model).await;
    }

    match engine.generate(
        &prompt,
        params.max_tokens,
        params.temperature,
        params.top_p,
        params.top_k,
        params.repeat_penalty,
        params.repeat_last_n,
        params.seed,
        &params.stop,
    ) {
        Ok(r) => {
            state.add_tokens(r.completion_tokens as u64);
            info!(
                "Generated {} tokens in {}ms ({:.1} tok/s)",
                r.completion_tokens, r.generation_time_ms, r.tokens_per_second
            );

            HttpResponse::Ok().json(CompletionResponse {
                id,
                object: "text_completion".to_string(),
                created: chrono::Utc::now().timestamp(),
                model,
                choices: vec![CompletionChoice {
                    index: 0,
                    text: r.text,
                    finish_reason: r.finish_reason,
                }],
                usage: Usage {
                    prompt_tokens: r.prompt_tokens,
                    completion_tokens: r.completion_tokens,
                    total_tokens: r.prompt_tokens + r.completion_tokens,
                },
            })
        }
        Err(e) => {
            error!("Generation error: {:#}", e);
            make_error(
                actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Generation failed: {}", e),
            )
        }
    }
}

async fn stream_completion_response(
    mut engine: tokio::sync::MutexGuard<'_, crate::model::ModelEngine>,
    prompt: String,
    params: GenParams,
    id: String,
    model: String,
) -> HttpResponse {
    let (tx, rx) = mpsc::channel::<Bytes>(64);
    let created = chrono::Utc::now().timestamp();

    // Spawn generation in a blocking task to not block the event loop
    let _handle = tokio::task::spawn_blocking(move || {
        let id_clone = id.clone();
        let model_clone = model.clone();
        let tx_clone = tx.clone();

        let result = engine.generate_streaming(
            &prompt,
            params.max_tokens,
            params.temperature,
            params.top_p,
            params.top_k,
            params.repeat_penalty,
            params.repeat_last_n,
            params.seed,
            &params.stop,
            |delta: &str| {
                let chunk = CompletionChunk {
                    id: id_clone.clone(),
                    object: "text_completion".to_string(),
                    created,
                    model: model_clone.clone(),
                    choices: vec![CompletionChunkChoice {
                        index: 0,
                        text: delta.to_string(),
                        finish_reason: None,
                    }],
                };
                tx_clone.try_send(sse_line(&chunk)).is_ok()
            },
        );

        if let Ok(r) = &result {
            let final_chunk = CompletionChunk {
                id: id.clone(),
                object: "text_completion".to_string(),
                created,
                model: model.clone(),
                choices: vec![CompletionChunkChoice {
                    index: 0,
                    text: String::new(),
                    finish_reason: Some(r.finish_reason.clone()),
                }],
            };
            let _ = tx.try_send(sse_line(&final_chunk));
        }
        let _ = tx.try_send(sse_done());
        
        // Explicitly drop the sender to signal end of stream
        drop(tx);
    });

    // Convert the receiver into a stream
    let stream = ReceiverStream::new(rx);

    HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("Connection", "keep-alive"))
        .insert_header(("X-Accel-Buffering", "no"))
        .streaming(stream)
}