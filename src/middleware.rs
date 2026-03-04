use actix_web::{HttpRequest, HttpResponse};
use std::sync::Arc;

use crate::dto::{ErrorDetail, ErrorResponse};
use crate::AppState;

pub fn check_auth(req: &HttpRequest, state: &Arc<AppState>) -> Option<HttpResponse> {
    let expected = match &state.api_key {
        Some(key) => key,
        None => return None,
    };

    let auth_header = req
        .headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(header) if header.starts_with("Bearer ") => {
            let token = &header[7..];
            if token == expected {
                None
            } else {
                Some(HttpResponse::Unauthorized().json(ErrorResponse {
                    error: ErrorDetail {
                        message: "Invalid API key".to_string(),
                        r#type: "authentication_error".to_string(),
                        code: Some("invalid_api_key".to_string()),
                    },
                }))
            }
        }
        Some(_) => Some(HttpResponse::Unauthorized().json(ErrorResponse {
            error: ErrorDetail {
                message: "Invalid Authorization header format. Expected: Bearer <key>".to_string(),
                r#type: "authentication_error".to_string(),
                code: Some("invalid_auth_header".to_string()),
            },
        })),
        None => Some(HttpResponse::Unauthorized().json(ErrorResponse {
            error: ErrorDetail {
                message: "Missing Authorization header".to_string(),
                r#type: "authentication_error".to_string(),
                code: Some("missing_auth".to_string()),
            },
        })),
    }
}
