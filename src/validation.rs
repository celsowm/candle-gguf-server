use actix_web::HttpResponse;

use crate::dto::{ChatCompletionRequest, CompletionRequest};

fn make_validation_error(msg: impl ToString) -> HttpResponse {
    use crate::dto::{ErrorDetail, ErrorResponse};
    HttpResponse::BadRequest().json(ErrorResponse {
        error: ErrorDetail {
            message: msg.to_string(),
            r#type: "invalid_request_error".to_string(),
            code: Some("invalid_request".to_string()),
        },
    })
}

pub fn validate_chat_request(req: &ChatCompletionRequest) -> Result<(), HttpResponse> {
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

pub fn validate_completion_request(req: &CompletionRequest) -> Result<(), HttpResponse> {
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
