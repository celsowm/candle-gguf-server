use tokenizers::Tokenizer;

use crate::dto::ChatMessage;

#[derive(Debug, Clone, Copy)]
pub enum ChatTemplate {
    ChatML,
    Llama2,
    Llama3,
    Gemma,
    Phi2Base,
    Phi3,
    Zephyr,
    Mistral,
    Generic,
}

impl std::fmt::Display for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Auto-detect chat template from the tokenizer vocabulary and model architecture.
pub fn detect_template(tokenizer: &Tokenizer, architecture: &str) -> ChatTemplate {
    let vocab = tokenizer.get_vocab(true);

    if vocab.contains_key("<|im_start|>") && vocab.contains_key("<|im_end|>") {
        // Could be ChatML or Phi-3; Phi-3 also uses <|im_start|>
        if vocab.contains_key("<|assistant|>") || vocab.contains_key("<|end|>") {
            return ChatTemplate::Phi3;
        }
        return ChatTemplate::ChatML;
    }

    if vocab.contains_key("<|begin_of_text|>") || vocab.contains_key("<|start_header_id|>") {
        return ChatTemplate::Llama3;
    }

    if vocab.contains_key("<start_of_turn>") && vocab.contains_key("<end_of_turn>") {
        return ChatTemplate::Gemma;
    }

    if vocab.contains_key("[INST]") {
        // Differentiate Llama-2 vs Mistral: Mistral often has <s>[INST]
        // Both use [INST] so we check for Mistral-specific tokens
        if vocab.contains_key("▁[INST]") || vocab.contains_key("<s>") {
            // Could be either, but Mistral wraps differently
            // For safety, check if we also have <<SYS>>
            if vocab.contains_key("<<SYS>>") {
                return ChatTemplate::Llama2;
            }
            return ChatTemplate::Mistral;
        }
        return ChatTemplate::Llama2;
    }

    if vocab.contains_key("<|user|>") && vocab.contains_key("<|assistant|>") {
        return ChatTemplate::Zephyr;
    }

    if architecture == "phi2" {
        return ChatTemplate::Phi2Base;
    }

    ChatTemplate::Generic
}

/// Apply the detected template to format messages into a prompt string.
pub fn apply_template(messages: &[ChatMessage], template: &ChatTemplate) -> String {
    match template {
        ChatTemplate::ChatML => format_chatml(messages),
        ChatTemplate::Llama2 => format_llama2(messages),
        ChatTemplate::Llama3 => format_llama3(messages),
        ChatTemplate::Gemma => format_gemma(messages),
        ChatTemplate::Phi2Base => format_phi2_base(messages),
        ChatTemplate::Phi3 => format_phi3(messages),
        ChatTemplate::Zephyr => format_zephyr(messages),
        ChatTemplate::Mistral => format_mistral(messages),
        ChatTemplate::Generic => format_generic(messages),
    }
}

// ========================================================================
// Template formatters
// ========================================================================

fn format_chatml(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            m.role, m.content
        ));
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

fn format_llama2(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    let mut system_msg: Option<&str> = None;
    let mut first_user = true;

    for m in messages {
        match m.role.as_str() {
            "system" => {
                system_msg = Some(&m.content);
            }
            "user" => {
                if first_user {
                    if let Some(sys) = system_msg.take() {
                        out.push_str(&format!(
                            "<s>[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]",
                            sys, m.content
                        ));
                    } else {
                        out.push_str(&format!("<s>[INST] {} [/INST]", m.content));
                    }
                    first_user = false;
                } else {
                    out.push_str(&format!("<s>[INST] {} [/INST]", m.content));
                }
            }
            "assistant" => {
                out.push_str(&format!(" {} </s>", m.content));
            }
            _ => {}
        }
    }
    out
}

fn format_llama3(messages: &[ChatMessage]) -> String {
    let mut out = String::from("<|begin_of_text|>");
    for m in messages {
        out.push_str(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
            m.role, m.content
        ));
    }
    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    out
}

fn format_gemma(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        let role = match m.role.as_str() {
            "system" | "user" => "user",
            "assistant" => "model",
            other => other,
        };
        out.push_str(&format!(
            "<start_of_turn>{}\n{}<end_of_turn>\n",
            role, m.content
        ));
    }
    out.push_str("<start_of_turn>model\n");
    out
}

fn format_phi2_base(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        match m.role.as_str() {
            "system" => out.push_str(&format!("System: {}\n", m.content)),
            "user" => out.push_str(&format!("Instruct: {}\n", m.content)),
            "assistant" => out.push_str(&format!("Output: {}\n", m.content)),
            _ => {}
        }
    }
    out.push_str("Output:");
    out
}

fn format_phi3(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str(&format!("<|{}|>\n{}<|end|>\n", m.role, m.content));
    }
    out.push_str("<|assistant|>\n");
    out
}

fn format_zephyr(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str(&format!("<|{}|>\n{}</s>\n", m.role, m.content));
    }
    out.push_str("<|assistant|>\n");
    out
}

fn format_mistral(messages: &[ChatMessage]) -> String {
    let mut out = String::from("<s>");
    let mut has_system = false;

    for m in messages {
        match m.role.as_str() {
            "system" => {
                // Mistral doesn't have explicit system; prepend to first user message
                out.push_str(&format!("[INST] {}\n\n", m.content));
                has_system = true;
            }
            "user" => {
                if has_system {
                    out.push_str(&format!("{} [/INST]", m.content));
                    has_system = false;
                } else {
                    out.push_str(&format!("[INST] {} [/INST]", m.content));
                }
            }
            "assistant" => {
                out.push_str(&format!(" {}</s>", m.content));
            }
            _ => {}
        }
    }
    out
}

fn format_generic(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        let label = match m.role.as_str() {
            "system" => "System",
            "user" => "User",
            "assistant" => "Assistant",
            other => other,
        };
        out.push_str(&format!("### {}:\n{}\n\n", label, m.content));
    }
    out.push_str("### Assistant:\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dto::ChatMessage;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
        }
    }

    #[test]
    fn test_chatml_format() {
        let messages = vec![msg("system", "You are helpful."), msg("user", "Hello")];
        let result = format_chatml(&messages);
        assert!(result.contains("<|im_start|>system"));
        assert!(result.contains("<|im_start|>user"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_llama3_format() {
        let messages = vec![msg("user", "Hi")];
        let result = format_llama3(&messages);
        assert!(result.starts_with("<|begin_of_text|>"));
        assert!(result.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_gemma_format() {
        let messages = vec![msg("system", "Be concise."), msg("user", "Hello")];
        let result = format_gemma(&messages);
        // System messages map to "user" role in Gemma
        assert!(result.contains("<start_of_turn>user\nBe concise."));
        assert!(result.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn test_generic_format() {
        let messages = vec![msg("user", "test")];
        let result = format_generic(&messages);
        assert!(result.contains("### User:"));
        assert!(result.ends_with("### Assistant:\n"));
    }

    #[test]
    fn test_phi2_base_format() {
        let messages = vec![msg("system", "be concise"), msg("user", "ola")];
        let result = format_phi2_base(&messages);
        assert!(result.contains("System: be concise\n"));
        assert!(result.contains("Instruct: ola\n"));
        assert!(result.ends_with("Output:"));
    }

    #[test]
    fn test_detect_phi2_base_template() {
        let tokenizer = Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default());
        let template = detect_template(&tokenizer, "phi2");
        assert!(matches!(template, ChatTemplate::Phi2Base));
    }
}
