use actix_web::HttpResponse;
use bytes::Bytes;
use serde::Serialize;
use std::sync::Arc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;
use tracing::{error, info, warn};

use crate::dto::{
    ChatCompletionChunk, ChatChunkChoice, ChatDelta,
    CompletionChunk, CompletionChunkChoice,
    GenParams,
};
use crate::AppState;

pub fn sse_line<T: Serialize>(payload: &T) -> Bytes {
    let json = serde_json::to_string(payload).unwrap();
    Bytes::from(format!("data: {}\n\n", json))
}

pub fn sse_done() -> Bytes {
    Bytes::from("data: [DONE]\n\n")
}

/// Build an SSE streaming HttpResponse from the given channel receiver.
fn build_sse_response(rx: tokio::sync::mpsc::UnboundedReceiver<Result<Bytes, String>>) -> HttpResponse {
    let stream = UnboundedReceiverStream::new(rx)
        .map(|result| result.map_err(|err| actix_web::error::ErrorInternalServerError(err)));

    HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("Connection", "keep-alive"))
        .insert_header(("X-Accel-Buffering", "no"))
        .streaming(stream)
}

pub async fn stream_chat_response(
    state: Arc<AppState>,
    prompt: String,
    params: GenParams,
    id: String,
    model: String,
    template_name: String,
    architecture: String,
) -> HttpResponse {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Bytes, String>>();
    let created = chrono::Utc::now().timestamp();

    let _handle = tokio::task::spawn_blocking(move || {
        let mut engine = state.engine.blocking_lock();
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
        let _ = tx.send(Ok(sse_line(&role_chunk)));

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
                tx_clone.send(Ok(sse_line(&chunk))).is_ok()
            },
        );

        match &result {
            Ok(r) => {
                state.add_tokens(r.completion_tokens as u64);
                if r.aborted_hidden_only {
                    warn!(
                        "Hidden-only chat stream aborted early  request_id={}  model={}  template={}  architecture={}",
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
                let _ = tx.send(Ok(sse_line(&final_chunk)));
            }
            Err(e) => {
                error!("Streaming generation error after SSE start: {:#}", e);
            }
        }
        let _ = tx.send(Ok(sse_done()));
        drop(tx);
    });

    build_sse_response(rx)
}

pub async fn stream_completion_response(
    state: Arc<AppState>,
    prompt: String,
    params: GenParams,
    id: String,
    model: String,
    architecture: String,
) -> HttpResponse {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Bytes, String>>();
    let created = chrono::Utc::now().timestamp();

    let _handle = tokio::task::spawn_blocking(move || {
        let mut engine = state.engine.blocking_lock();
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
                tx_clone.send(Ok(sse_line(&chunk))).is_ok()
            },
        );

        match &result {
            Ok(r) => {
                state.add_tokens(r.completion_tokens as u64);
                if r.aborted_hidden_only {
                    warn!(
                        "Hidden-only completion stream aborted early  request_id={}  model={}  architecture={}",
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
                let _ = tx.send(Ok(sse_line(&final_chunk)));
            }
            Err(e) => {
                error!("Streaming completion error after SSE start: {:#}", e);
            }
        }
        let _ = tx.send(Ok(sse_done()));
        drop(tx);
    });

    build_sse_response(rx)
}
