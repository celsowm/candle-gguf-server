use actix_web::{web, HttpRequest, HttpResponse};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

use crate::dto::{
    ChatChoice, ChatCompletionResponse, ChatMessageResponse, CompletionChoice,
    CompletionResponse, ErrorDetail, ErrorResponse, GenParams, ModelInfo, ModelList,
    ModelPermission, Usage,
};
use crate::middleware::check_auth;
use crate::sse::{stream_chat_response, stream_completion_response};
use crate::tokenizer_utils::{apply_template, detect_template};
use crate::validation::{validate_chat_request, validate_completion_request};
use crate::AppState;

// =========================================================================
// Shared helpers
// =========================================================================

fn make_error(status: actix_web::http::StatusCode, msg: impl ToString) -> HttpResponse {
    HttpResponse::build(status).json(ErrorResponse {
        error: ErrorDetail {
            message: msg.to_string(),
            r#type: "server_error".to_string(),
            code: None,
        },
    })
}

fn request_id(prefix: &str) -> String {
    format!("{}-{}", prefix, uuid::Uuid::new_v4())
}

// =========================================================================
// Handlers
// =========================================================================

pub async fn health(state: web::Data<Arc<AppState>>) -> HttpResponse {
    let uptime = state.start_time.elapsed().as_secs();
    let reqs = state
        .request_count
        .load(std::sync::atomic::Ordering::Relaxed);
    let toks = state
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);

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
    let reqs = state
        .request_count
        .load(std::sync::atomic::Ordering::Relaxed);
    let toks = state
        .tokens_generated
        .load(std::sync::atomic::Ordering::Relaxed);

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

fn build_model_info(state: &AppState) -> ModelInfo {
    let created = state.created_timestamp;
    ModelInfo {
        id: state.model_name.clone(),
        object: "model".to_string(),
        created,
        owned_by: "vllm".to_string(),
        root: state.model_name.clone(),
        parent: None,
        max_model_len: state.max_model_len,
        permission: vec![ModelPermission {
            id: format!(
                "modelperm-{}",
                uuid::Uuid::new_v4().to_string().replace("-", "")
            ),
            object: "model_permission".to_string(),
            created,
            allow_create_engine: false,
            allow_sampling: true,
            allow_logprobs: true,
            allow_search_indices: false,
            allow_view: true,
            allow_fine_tuning: false,
            organization: "*".to_string(),
            group: None,
            is_blocking: false,
        }],
    }
}

pub async fn list_models(state: web::Data<Arc<AppState>>) -> HttpResponse {
    HttpResponse::Ok().json(ModelList {
        object: "list".to_string(),
        data: vec![build_model_info(&state)],
    })
}

pub async fn get_model(state: web::Data<Arc<AppState>>, path: web::Path<String>) -> HttpResponse {
    let id = path.into_inner();
    if id == state.model_name {
        HttpResponse::Ok().json(build_model_info(&state))
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
    req: web::Json<crate::dto::ChatCompletionRequest>,
    http_req: HttpRequest,
) -> HttpResponse {
    if let Some(err) = check_auth(&http_req, &state) {
        return err;
    }

    if let Err(err) = validate_chat_request(&req) {
        return err;
    }

    let is_stream = req.stream.unwrap_or(false);
    let model = state.model_name.clone();
    info!(
        "POST /v1/chat/completions  stream={}  messages={}  model={}  chat_template_kwargs={}",
        is_stream,
        req.messages.len(),
        model,
        req.chat_template_kwargs.is_some()
    );
    if let Some(kwargs) = req.chat_template_kwargs.as_ref() {
        info!(
            "chat_template_kwargs received: enable_thinking={:?}",
            kwargs.enable_thinking
        );
    }

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

    let params = GenParams::from_chat(&req);
    let id = request_id("chatcmpl");

    let mut engine = state.engine.lock().await;

    let template = detect_template(engine.tokenizer(), engine.architecture());
    let template_name = template.to_string();
    let architecture = engine.architecture().to_string();
    debug!("Detected chat template: {:?}", template);
    let prompt = apply_template(&req.messages, &template);
    match engine.prompt_token_count(&prompt) {
        Ok(prompt_tokens) => info!(
            "Prepared chat generation  template={}  architecture={}  prompt_tokens={}  max_tokens={}",
            template_name, architecture, prompt_tokens, params.max_tokens
        ),
        Err(e) => warn!(
            "Failed to count chat prompt tokens before generation  template={}  architecture={}  error={}",
            template_name, architecture, e
        ),
    }

    if is_stream {
        drop(engine);
        let state_arc: Arc<AppState> = state.as_ref().clone();
        return stream_chat_response(
            state_arc,
            prompt,
            params,
            id,
            model,
            template_name,
            architecture,
        )
        .await;
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
            if r.aborted_hidden_only {
                warn!(
                    "Hidden-only chat generation aborted early  request_id={}  model={}  template={}  architecture={}",
                    id, model, template_name, architecture
                );
            }
            info!(
                "Generated {} tokens in {}ms ({:.1} tok/s)  had_visible_output={}  aborted_hidden_only={}",
                r.completion_tokens,
                r.generation_time_ms,
                r.tokens_per_second,
                r.had_visible_output,
                r.aborted_hidden_only
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

// =========================================================================
// Text Completions
// =========================================================================

pub async fn completions(
    state: web::Data<Arc<AppState>>,
    req: web::Json<crate::dto::CompletionRequest>,
    http_req: HttpRequest,
) -> HttpResponse {
    if let Some(err) = check_auth(&http_req, &state) {
        return err;
    }

    if let Err(err) = validate_completion_request(&req) {
        return err;
    }

    let is_stream = req.stream.unwrap_or(false);
    let model = state.model_name.clone();
    info!(
        "POST /v1/completions  stream={}  model={}",
        is_stream, model
    );

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
    let mut engine = state.engine.lock().await;
    let architecture = engine.architecture().to_string();
    match engine.prompt_token_count(&prompt) {
        Ok(prompt_tokens) => info!(
            "Prepared completion generation  architecture={}  prompt_tokens={}  max_tokens={}",
            architecture, prompt_tokens, params.max_tokens
        ),
        Err(e) => warn!(
            "Failed to count completion prompt tokens before generation  architecture={}  error={}",
            architecture, e
        ),
    }

    if is_stream {
        drop(engine);
        let state_arc: Arc<AppState> = state.as_ref().clone();
        return stream_completion_response(state_arc, prompt, params, id, model, architecture)
            .await;
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
            if r.aborted_hidden_only {
                warn!(
                    "Hidden-only completion generation aborted early  request_id={}  model={}  architecture={}",
                    id, model, architecture
                );
            }
            info!(
                "Generated {} tokens in {}ms ({:.1} tok/s)  had_visible_output={}  aborted_hidden_only={}",
                r.completion_tokens,
                r.generation_time_ms,
                r.tokens_per_second,
                r.had_visible_output,
                r.aborted_hidden_only
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

#[cfg(test)]
mod tests {
    use crate::dto::ChatCompletionRequest;

    #[test]
    fn test_chat_request_accepts_template_kwargs() {
        let parsed: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "gguf-model",
            "messages": [{"role": "user", "content": "ola"}],
            "chat_template_kwargs": {"enable_thinking": false}
        }))
        .unwrap();

        let kwargs = parsed
            .chat_template_kwargs
            .expect("missing chat_template_kwargs");
        assert_eq!(kwargs.enable_thinking, Some(false));
    }

    #[test]
    fn test_chat_request_ignores_unknown_fields() {
        let parsed: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "gguf-model",
            "messages": [{"role": "user", "content": "ola"}],
            "unknown_field": "ignored"
        }))
        .unwrap();

        assert_eq!(parsed.messages.len(), 1);
        assert!(parsed.chat_template_kwargs.is_none());
    }
}
