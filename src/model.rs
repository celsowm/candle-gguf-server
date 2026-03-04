use candle_core::quantized::gguf_file;
use candle_core::quantized::tokenizer::TokenizerFromGguf;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights as LlamaWeights;
use candle_transformers::models::quantized_phi::ModelWeights as PhiWeights;
use candle_transformers::models::quantized_phi3::ModelWeights as Phi3Weights;
use std::collections::{HashMap, HashSet};
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

enum ModelArch {
    Llama(LlamaWeights),
    Phi2(PhiWeights),
    Phi3(Phi3Weights),
}

impl ModelArch {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Llama(m) => m.forward(x, index_pos),
            Self::Phi2(m) => m.forward(x, index_pos),
            Self::Phi3(m) => m.forward(x, index_pos),
        }
    }
}

use crate::sampling::LogitsProcessor;

/// Result of a single generation.
#[derive(Debug)]
pub struct GenerationResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
    pub generation_time_ms: u128,
    pub tokens_per_second: f64,
    pub had_visible_output: bool,
    pub aborted_hidden_only: bool,
}

/// Core model engine holding weights, tokenizer, and device.
pub struct ModelEngine {
    model: ModelArch,
    tokenizer: Tokenizer,
    device: Device,
    ctx_len: usize,
    architecture: String,
    eos_tokens: Vec<u32>,
}

impl ModelEngine {
    pub fn new(
        model_path: &str,
        tokenizer_path: Option<&str>,
        device: &Device,
        ctx_len_override: usize,
    ) -> anyhow::Result<Self> {
        let mut file = std::fs::File::open(model_path)
            .map_err(|e| anyhow::anyhow!("Cannot open model file '{}': {}", model_path, e))?;

        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("Failed to parse GGUF: {}", e))?;

        info!(
            "GGUF loaded: {} tensors, {} metadata keys",
            content.tensor_infos.len(),
            content.metadata.len()
        );

        // Try to extract context length from metadata
        let ctx_len = if ctx_len_override > 0 {
            ctx_len_override
        } else {
            Self::read_ctx_len_from_metadata(&content).unwrap_or(4096)
        };
        info!("Context length: {}", ctx_len);

        // Log some metadata if available
        Self::log_metadata(&content);

        // Try to build tokenizer: explicit file > embedded in GGUF
        let tokenizer = if let Some(path) = tokenizer_path {
            info!("Loading tokenizer from file: {}", path);
            Tokenizer::from_file(path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer '{}': {}", path, e))?
        } else {
            info!("Extracting tokenizer from GGUF metadata");
            Tokenizer::from_gguf(&content)
                .map_err(|e| anyhow::anyhow!("Failed to extract tokenizer from GGUF: {}", e))?
        };

        let architecture = content
            .metadata
            .get("general.architecture")
            .and_then(|v| match v {
                gguf_file::Value::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_default();
        info!("Detected architecture: {}", architecture);

        let model = match architecture.as_str() {
            "phi2" => {
                let weights = PhiWeights::from_gguf(content, &mut file, device)
                    .map_err(|e| anyhow::anyhow!("Failed to load phi2 weights: {}", e))?;
                ModelArch::Phi2(weights)
            }
            "phi3" => {
                let weights = Phi3Weights::from_gguf(false, content, &mut file, device)
                    .map_err(|e| anyhow::anyhow!("Failed to load phi3 weights: {}", e))?;
                ModelArch::Phi3(weights)
            }
            _ => {
                let weights = LlamaWeights::from_gguf(content, &mut file, device)
                    .map_err(|e| anyhow::anyhow!("Failed to load model weights: {}", e))?;
                ModelArch::Llama(weights)
            }
        };

        // Pre-compute all possible EOS token IDs
        let vocab = tokenizer.get_vocab(true);
        let (eos_tokens, matched_eos_tokens) =
            Self::resolve_eos_tokens(&vocab, &architecture, tokenizer.get_vocab_size(true));

        if eos_tokens.is_empty() {
            warn!("No EOS token found in tokenizer vocabulary!");
        } else {
            info!("Matched EOS tokens: {:?}", matched_eos_tokens);
            info!("EOS token IDs: {:?}", eos_tokens);
        }

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            ctx_len,
            architecture,
            eos_tokens,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn ctx_len(&self) -> usize {
        self.ctx_len
    }

    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn prompt_token_count(&self, prompt: &str) -> anyhow::Result<usize> {
        self.tokenize(prompt).map(|tokens| tokens.len())
    }

    /// Non-streaming generation.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
        top_k: usize,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
        stop_sequences: &[String],
    ) -> anyhow::Result<GenerationResult> {
        let start = std::time::Instant::now();

        let prompt_tokens = self.tokenize(prompt)?;
        let prompt_len = prompt_tokens.len();

        self.validate_prompt_length(prompt_len, max_tokens)?;

        let effective_max = self.effective_max_tokens(prompt_len, max_tokens);

        let mut all_tokens = prompt_tokens.clone();
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(effective_max);
        let mut logits_proc = LogitsProcessor::new(seed, temperature, top_p, top_k);
        let mut visibility = VisibilityState::default();
        let mut aborted_hidden_only = false;
        let hidden_only_abort_limit = self.hidden_only_abort_limit();

        // Reset KV cache for fresh generation
        // Note: Clearing KV cache might require different approach in newer candle version
        // For now, we'll comment this out until we determine the correct approach
        // self.model.reset_kv_cache();

        // === Prefill ===
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?;
        let logits = self.extract_last_logits(&logits)?;
        let logits =
            Self::apply_repeat_penalty(&logits, repeat_penalty, &all_tokens, repeat_last_n)?;

        let mut next = logits_proc.sample(&logits)?;
        generated_tokens.push(next);
        all_tokens.push(next);

        if !self.is_eos(next) {
            let delta = self.next_text_delta(&generated_tokens, &mut visibility.prev_text_len)?;
            visibility.record(delta.is_some());
            if visibility.should_abort_hidden_only(hidden_only_abort_limit) {
                aborted_hidden_only = true;
            }
        }

        // === Decode loop ===
        if !aborted_hidden_only {
            for i in 1..effective_max {
                if self.is_eos(next) {
                    break;
                }

                if self.check_stop_sequences(&generated_tokens, stop_sequences)? {
                    break;
                }

                let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
                let logits = self.model.forward(&input, prompt_len + i)?;
                let logits = self.extract_last_logits(&logits)?;
                let logits = Self::apply_repeat_penalty(
                    &logits,
                    repeat_penalty,
                    &all_tokens,
                    repeat_last_n,
                )?;

                next = logits_proc.sample(&logits)?;
                generated_tokens.push(next);
                all_tokens.push(next);

                if self.is_eos(next) {
                    break;
                }

                let delta =
                    self.next_text_delta(&generated_tokens, &mut visibility.prev_text_len)?;
                visibility.record(delta.is_some());
                if visibility.should_abort_hidden_only(hidden_only_abort_limit) {
                    aborted_hidden_only = true;
                    warn!(
                        "Aborting hidden-only generation early: architecture={} prompt_tokens={} sampled_tokens={} hidden_token_limit={}",
                        self.architecture,
                        prompt_len,
                        generated_tokens.len(),
                        hidden_only_abort_limit
                    );
                    break;
                }
            }
        }

        let elapsed = start.elapsed();
        let text = if aborted_hidden_only {
            String::new()
        } else {
            self.decode_tokens(&generated_tokens)?
        };
        let (final_text, finish_reason) = if aborted_hidden_only {
            (String::new(), "stop".to_string())
        } else {
            Self::trim_stop_sequences(text, generated_tokens.len(), effective_max, stop_sequences)
        };

        let tps = if elapsed.as_secs_f64() > 0.0 {
            generated_tokens.len() as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        debug!(
            "Generated {} tokens in {:.1}ms ({:.1} tok/s)",
            generated_tokens.len(),
            elapsed.as_millis(),
            tps
        );

        Ok(GenerationResult {
            text: final_text,
            prompt_tokens: prompt_len,
            completion_tokens: generated_tokens.len(),
            finish_reason,
            generation_time_ms: elapsed.as_millis(),
            tokens_per_second: tps,
            had_visible_output: visibility.had_visible_output(),
            aborted_hidden_only,
        })
    }

    /// Streaming generation — calls `on_token` with each incremental text delta.
    /// Return `false` from `on_token` to cancel early.
    pub fn generate_streaming<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
        top_k: usize,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
        stop_sequences: &[String],
        mut on_token: F,
    ) -> anyhow::Result<GenerationResult>
    where
        F: FnMut(&str) -> bool,
    {
        let start = std::time::Instant::now();

        let prompt_tokens = self.tokenize(prompt)?;
        let prompt_len = prompt_tokens.len();

        self.validate_prompt_length(prompt_len, max_tokens)?;

        let effective_max = self.effective_max_tokens(prompt_len, max_tokens);

        let mut all_tokens = prompt_tokens.clone();
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(effective_max);
        let mut logits_proc = LogitsProcessor::new(seed, temperature, top_p, top_k);
        let mut visibility = VisibilityState::default();
        let mut cancelled = false;
        let mut aborted_hidden_only = false;
        let hidden_only_abort_limit = self.hidden_only_abort_limit();

        // self.model.clear_kv_cache(); // Commented out due to API change in newer candle version

        // === Prefill ===
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?;
        let logits = self.extract_last_logits(&logits)?;
        let logits =
            Self::apply_repeat_penalty(&logits, repeat_penalty, &all_tokens, repeat_last_n)?;

        let mut next = logits_proc.sample(&logits)?;
        generated_tokens.push(next);
        all_tokens.push(next);

        if !self.is_eos(next) {
            let (emitted, keep_going) = self.emit_incremental_delta(
                &generated_tokens,
                &mut visibility.prev_text_len,
                &mut on_token,
            )?;
            visibility.record(emitted);
            if visibility.should_abort_hidden_only(hidden_only_abort_limit) {
                aborted_hidden_only = true;
                warn!(
                    "Aborting hidden-only streaming generation early: architecture={} prompt_tokens={} sampled_tokens={} hidden_token_limit={}",
                    self.architecture,
                    prompt_len,
                    generated_tokens.len(),
                    hidden_only_abort_limit
                );
            } else {
                cancelled = !keep_going;
            }
        }

        // === Decode loop ===
        if !cancelled && !aborted_hidden_only && !self.is_eos(next) {
            for i in 1..effective_max {
                if self.check_stop_sequences(&generated_tokens, stop_sequences)? {
                    break;
                }

                let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
                let logits = self.model.forward(&input, prompt_len + i)?;
                let logits = self.extract_last_logits(&logits)?;
                let logits = Self::apply_repeat_penalty(
                    &logits,
                    repeat_penalty,
                    &all_tokens,
                    repeat_last_n,
                )?;

                next = logits_proc.sample(&logits)?;
                generated_tokens.push(next);
                all_tokens.push(next);

                if self.is_eos(next) {
                    break;
                }

                let (emitted, keep_going) = self.emit_incremental_delta(
                    &generated_tokens,
                    &mut visibility.prev_text_len,
                    &mut on_token,
                )?;
                visibility.record(emitted);

                if visibility.should_abort_hidden_only(hidden_only_abort_limit) {
                    aborted_hidden_only = true;
                    warn!(
                        "Aborting hidden-only streaming generation early: architecture={} prompt_tokens={} sampled_tokens={} hidden_token_limit={}",
                        self.architecture,
                        prompt_len,
                        generated_tokens.len(),
                        hidden_only_abort_limit
                    );
                    break;
                }

                if !keep_going {
                    cancelled = true;
                    break;
                }
            }
        }

        let elapsed = start.elapsed();
        let full_text = if aborted_hidden_only {
            String::new()
        } else {
            self.decode_tokens(&generated_tokens)?
        };
        let (final_text, finish_reason) = if aborted_hidden_only {
            (String::new(), "stop".to_string())
        } else if cancelled {
            (full_text, "stop".to_string())
        } else {
            Self::trim_stop_sequences(
                full_text,
                generated_tokens.len(),
                effective_max,
                stop_sequences,
            )
        };

        let tps = if elapsed.as_secs_f64() > 0.0 {
            generated_tokens.len() as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        Ok(GenerationResult {
            text: final_text,
            prompt_tokens: prompt_len,
            completion_tokens: generated_tokens.len(),
            finish_reason,
            generation_time_ms: elapsed.as_millis(),
            tokens_per_second: tps,
            had_visible_output: visibility.had_visible_output(),
            aborted_hidden_only,
        })
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    fn tokenize(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode_tokens(&self, tokens: &[u32]) -> anyhow::Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("Decode error: {}", e))
    }

    fn is_eos(&self, token: u32) -> bool {
        self.eos_tokens.contains(&token)
    }

    fn validate_prompt_length(&self, prompt_len: usize, max_tokens: usize) -> anyhow::Result<()> {
        if prompt_len >= self.ctx_len {
            anyhow::bail!(
                "Prompt length ({}) exceeds context length ({})",
                prompt_len,
                self.ctx_len
            );
        }
        if prompt_len + max_tokens > self.ctx_len {
            debug!(
                "prompt({}) + max_tokens({}) exceeds ctx_len({}); clamping",
                prompt_len, max_tokens, self.ctx_len
            );
        }
        Ok(())
    }

    fn effective_max_tokens(&self, prompt_len: usize, max_tokens: usize) -> usize {
        std::cmp::min(max_tokens, self.ctx_len.saturating_sub(prompt_len))
    }

    fn extract_last_logits(&self, logits: &Tensor) -> anyhow::Result<Tensor> {
        let logits = logits.squeeze(0)?;
        let last_idx = logits.dim(0)? - 1;
        Ok(logits.get(last_idx)?)
    }

    fn resolve_eos_tokens(
        vocab: &HashMap<String, u32>,
        architecture: &str,
        vocab_size: usize,
    ) -> (Vec<u32>, Vec<&'static str>) {
        let eos_candidates = [
            "</s>",
            "<|endoftext|>",
            "<|end|>",
            "<|eot_id|>",
            "<eos>",
            "<|end_of_text|>",
        ];
        let mut seen = HashSet::new();
        let mut ids = Vec::new();
        let mut matched_tokens = Vec::new();

        for token in eos_candidates {
            if let Some(id) = vocab.get(token).copied() {
                if seen.insert(id) {
                    ids.push(id);
                }
                matched_tokens.push(token);
            }
        }

        if ids.is_empty() && architecture == "phi2" {
            let phi2_eos = 50_256_u32;
            if (phi2_eos as usize) < vocab_size && seen.insert(phi2_eos) {
                ids.push(phi2_eos);
                matched_tokens.push("<|endoftext|>#fallback(50256)");
            }
        }

        (ids, matched_tokens)
    }

    fn next_text_delta(
        &self,
        generated: &[u32],
        prev_text_len: &mut usize,
    ) -> anyhow::Result<Option<String>> {
        let current = self.decode_tokens(generated)?;
        let current_len = current.len();
        let delta = if current_len > *prev_text_len {
            let delta = &current[*prev_text_len..];
            if delta.is_empty() {
                None
            } else {
                Some(delta.to_string())
            }
        } else {
            None
        };
        *prev_text_len = current_len;
        Ok(delta)
    }

    fn hidden_only_abort_limit(&self) -> usize {
        if self.architecture == "phi2" {
            PHI2_MAX_HIDDEN_TOKENS_WITHOUT_VISIBLE_OUTPUT
        } else {
            DEFAULT_MAX_HIDDEN_TOKENS_WITHOUT_VISIBLE_OUTPUT
        }
    }

    /// Emit the new characters since `prev_text_len` to the callback.
    /// Returns `(emitted_visible_delta, keep_going)`.
    fn emit_incremental_delta<F>(
        &self,
        generated: &[u32],
        prev_text_len: &mut usize,
        on_token: &mut F,
    ) -> anyhow::Result<(bool, bool)>
    where
        F: FnMut(&str) -> bool,
    {
        if let Some(delta) = self.next_text_delta(generated, prev_text_len)? {
            let keep_going = on_token(&delta);
            return Ok((true, keep_going));
        }
        Ok((false, true))
    }

    fn check_stop_sequences(
        &self,
        generated: &[u32],
        stop_sequences: &[String],
    ) -> anyhow::Result<bool> {
        if stop_sequences.is_empty() {
            return Ok(false);
        }
        let text = self.decode_tokens(generated)?;
        Ok(stop_sequences.iter().any(|s| text.contains(s)))
    }

    fn apply_repeat_penalty(
        logits: &Tensor,
        penalty: f32,
        tokens: &[u32],
        last_n: usize,
    ) -> anyhow::Result<Tensor> {
        if penalty <= 1.0 || tokens.is_empty() {
            return Ok(logits.clone());
        }
        let start = tokens.len().saturating_sub(last_n);
        candle_transformers::utils::apply_repeat_penalty(logits, penalty, &tokens[start..])
            .map_err(|e| anyhow::anyhow!("Repeat penalty error: {}", e))
    }

    fn trim_stop_sequences(
        text: String,
        generated_len: usize,
        max_tokens: usize,
        stop_sequences: &[String],
    ) -> (String, String) {
        let mut output = text;
        let mut reason = if generated_len >= max_tokens {
            "length"
        } else {
            "stop"
        }
        .to_string();

        for seq in stop_sequences {
            if let Some(pos) = output.find(seq) {
                output.truncate(pos);
                reason = "stop".to_string();
                break;
            }
        }

        (output, reason)
    }

    fn read_ctx_len_from_metadata(content: &gguf_file::Content) -> Option<usize> {
        // Common GGUF metadata keys for context length
        let keys = [
            "llama.context_length",
            "general.context_length",
            "mistral.context_length",
            "qwen2.context_length",
            "phi2.context_length",
            "phi3.context_length",
            "gemma.context_length",
        ];
        for key in &keys {
            if let Some(val) = content.metadata.get(*key) {
                if let gguf_file::Value::U32(n) = val {
                    return Some(*n as usize);
                }
                if let gguf_file::Value::U64(n) = val {
                    return Some(*n as usize);
                }
            }
        }
        None
    }

    fn log_metadata(content: &gguf_file::Content) {
        let interesting = [
            "general.name",
            "general.architecture",
            "general.quantization_version",
            "general.file_type",
        ];
        for key in &interesting {
            if let Some(val) = content.metadata.get(*key) {
                info!("GGUF metadata: {} = {:?}", key, val);
            }
        }
    }
}

const DEFAULT_MAX_HIDDEN_TOKENS_WITHOUT_VISIBLE_OUTPUT: usize = 4;
const PHI2_MAX_HIDDEN_TOKENS_WITHOUT_VISIBLE_OUTPUT: usize = 1;

#[derive(Debug, Default)]
struct VisibilityState {
    prev_text_len: usize,
    visible_emissions: usize,
    consecutive_non_emitting_tokens: usize,
}

impl VisibilityState {
    fn record(&mut self, emitted_visible_delta: bool) {
        if emitted_visible_delta {
            self.visible_emissions += 1;
            self.consecutive_non_emitting_tokens = 0;
        } else {
            self.consecutive_non_emitting_tokens += 1;
        }
    }

    fn had_visible_output(&self) -> bool {
        self.visible_emissions > 0
    }

    fn should_abort_hidden_only(&self, limit: usize) -> bool {
        self.visible_emissions == 0 && self.consecutive_non_emitting_tokens >= limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_eos_tokens_deduplicates_and_matches() {
        let vocab = HashMap::from([
            ("</s>".to_string(), 2_u32),
            ("<|end|>".to_string(), 2_u32),
            ("<eos>".to_string(), 5_u32),
        ]);

        let (ids, matched) = ModelEngine::resolve_eos_tokens(&vocab, "llama", 32_000);

        assert_eq!(ids, vec![2, 5]);
        assert_eq!(matched, vec!["</s>", "<|end|>", "<eos>"]);
    }

    #[test]
    fn test_resolve_eos_tokens_uses_phi2_fallback() {
        let vocab = HashMap::new();

        let (ids, matched) = ModelEngine::resolve_eos_tokens(&vocab, "phi2", 60_000);

        assert_eq!(ids, vec![50_256]);
        assert_eq!(matched, vec!["<|endoftext|>#fallback(50256)"]);
    }

    #[test]
    fn test_visibility_state_aborts_hidden_only_after_threshold() {
        let mut visibility = VisibilityState::default();

        for _ in 0..DEFAULT_MAX_HIDDEN_TOKENS_WITHOUT_VISIBLE_OUTPUT {
            visibility.record(false);
        }

        assert!(!visibility.had_visible_output());
        assert!(
            visibility.should_abort_hidden_only(DEFAULT_MAX_HIDDEN_TOKENS_WITHOUT_VISIBLE_OUTPUT)
        );
    }
}
