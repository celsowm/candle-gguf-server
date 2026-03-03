mod api;
mod middleware;
mod model;
mod sampling;
mod tokenizer_utils;

use actix_cors::Cors;
use actix_web::{web, App, HttpServer, middleware as actix_middleware};
use clap::Parser;
use hf_hub::{api::tokio::Api, Repo, RepoType};
use model::ModelEngine;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "candle-gguf-server",
    about = "Serve GGUF models via OpenAI-compatible API",
    version
)]
pub struct Args {
    /// Path to the GGUF model file
    #[arg(short, long, conflicts_with = "hf_model")]
    model: Option<String>,

    /// Hugging Face model ID to download (e.g., microsoft/phi-2)
    #[arg(long, conflicts_with = "model")]
    hf_model: Option<String>,

    /// Path to the tokenizer.json file (optional if downloading from HF)
    #[arg(short, long)]
    tokenizer: Option<String>,

    /// Name of the GGUF file to download from Hugging Face (default: auto-detect by quantization)
    #[arg(long, default_value = "")]
    hf_filename: String,

    /// Preferred quantization when auto-selecting GGUF file (e.g., Q4_K_M, Q5_K_M, Q8_0)
    #[arg(long, default_value = "Q4_K_M")]
    hf_quant: String,

    /// Host to bind
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to bind
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    /// Device: "cpu", "cuda", or "metal"
    #[arg(short, long, default_value = "cpu")]
    device: String,

    /// CUDA device ordinal (when using --device cuda)
    #[arg(long, default_value_t = 0)]
    device_id: usize,

    /// Model name reported in API responses
    #[arg(long, default_value = "gguf-model")]
    model_name: String,

    /// Number of threads for CPU inference (0 = auto)
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Maximum concurrent generation requests
    #[arg(long, default_value_t = 1)]
    max_concurrent: usize,

    /// Context length override (0 = read from GGUF metadata)
    #[arg(long, default_value_t = 0)]
    ctx_len: usize,

    /// API key for authentication (empty = no auth)
    #[arg(long, default_value = "")]
    api_key: String,

    /// Enable CORS for all origins
    #[arg(long, default_value_t = false)]
    cors: bool,

    /// Log level: trace, debug, info, warn, error
    #[arg(long, default_value = "info")]
    log_level: String,
}

pub struct AppState {
    pub engine: tokio::sync::Mutex<ModelEngine>,
    pub model_name: String,
    pub concurrency_semaphore: Semaphore,
    pub api_key: Option<String>,
    pub start_time: std::time::Instant,
    pub request_count: std::sync::atomic::AtomicU64,
    pub tokens_generated: std::sync::atomic::AtomicU64,
}

impl AppState {
    pub fn increment_requests(&self) {
        self.request_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn add_tokens(&self, n: u64) {
        self.tokens_generated
            .fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive(args.log_level.parse().unwrap_or_else(|_| {
                    warn!("Invalid log level '{}', defaulting to 'info'", args.log_level);
                    "info".parse().unwrap()
                })),
        )
        .with_target(false)
        .with_thread_ids(true)
        .init();

    info!("candle-gguf-server v{}", env!("CARGO_PKG_VERSION"));

    let threads = if args.threads == 0 {
        num_cpus::get()
    } else {
        args.threads
    };
    info!("CPU threads : {}", threads);

    // Set rayon / candle thread count
    std::env::set_var("RAYON_NUM_THREADS", threads.to_string());

    let device = build_device(&args);

    // Handle model loading (either local file or from Hugging Face)
    let (model_path, tokenizer_path) = if let Some(hf_model_id) = &args.hf_model {
        // Download model from Hugging Face
        let (model_path, tokenizer_path) = download_hf_model(hf_model_id, &args.hf_filename, &args.hf_quant).await
            .expect("Failed to download model from Hugging Face");
        
        info!("Model path  : {}", model_path.display());
        info!("Tokenizer   : {}", tokenizer_path.display());
        
        (model_path.to_string_lossy().to_string(), tokenizer_path.to_string_lossy().to_string())
    } else {
        // Use local model file
        let model_path = args.model.as_ref().expect("--model or --hf-model must be provided");
        let tokenizer_path = args.tokenizer.as_ref().expect("--tokenizer must be provided when using --model");
        
        info!("Model path  : {}", model_path);
        info!("Tokenizer   : {}", tokenizer_path);
        
        (model_path.clone(), tokenizer_path.clone())
    };

    info!("Device      : {}", args.device);
    info!("Max concurrent: {}", args.max_concurrent);

    let engine = ModelEngine::new(&model_path, &tokenizer_path, &device, args.ctx_len)
        .expect("Failed to load model");

    info!(
        "Model loaded — vocab size: {}, ctx_len: {}",
        engine.vocab_size(),
        engine.ctx_len()
    );

    let api_key = if args.api_key.is_empty() {
        None
    } else {
        info!("API key authentication enabled");
        Some(args.api_key.clone())
    };

    let app_state = Arc::new(AppState {
        engine: tokio::sync::Mutex::new(engine),
        model_name: args.model_name.clone(),
        concurrency_semaphore: Semaphore::new(args.max_concurrent),
        api_key,
        start_time: std::time::Instant::now(),
        request_count: std::sync::atomic::AtomicU64::new(0),
        tokens_generated: std::sync::atomic::AtomicU64::new(0),
    });

    let bind_addr = format!("{}:{}", args.host, args.port);
    info!("Listening on http://{}", bind_addr);

    let enable_cors = args.cors;

    HttpServer::new(move || {
        let cors = if enable_cors {
            Cors::permissive()
        } else {
            Cors::default()
        };

        App::new()
            .wrap(cors)
            .wrap(actix_middleware::Logger::new(
                "%a \"%r\" %s %b %Dms",
            ))
            .app_data(web::Data::from(app_state.clone()))
            .app_data(web::JsonConfig::default().limit(4 * 1024 * 1024))
            // OpenAI-compatible routes
            .route("/v1/chat/completions", web::post().to(api::chat_completions))
            .route("/v1/completions", web::post().to(api::completions))
            .route("/v1/models", web::get().to(api::list_models))
            .route("/v1/models/{model_id}", web::get().to(api::get_model))
            // Utility routes
            .route("/health", web::get().to(api::health))
            .route("/metrics", web::get().to(api::metrics))
            // Catch-all for 404
            .default_service(web::to(api::not_found))
    })
    .workers(num_cpus::get())  // Use number of CPU cores for workers
    .bind(&bind_addr)?
    .run()
    .await
}

async fn download_hf_model(model_id: &str, hf_filename: &str, preferred_quant: &str) -> anyhow::Result<(PathBuf, PathBuf)> {
    let api = Api::new()?;
    let repo = Repo::new(model_id.to_string(), RepoType::Model);
    
    // If no specific filename provided, look for .gguf files in the repository
    let gguf_filename = if hf_filename.is_empty() {
        let repo_info = api.repo(repo.clone()).info().await
            .map_err(|e| anyhow::anyhow!("Failed to get repo info for {}: {}", model_id, e))?;
        
        let gguf_files: Vec<&str> = repo_info.siblings
            .iter()
            .map(|s| s.rfilename.as_str())
            .filter(|name| name.ends_with(".gguf"))
            .collect();
        
        match gguf_files.len() {
            0 => return Err(anyhow::anyhow!(
                "No .gguf files found in repository '{}'. This repo may not contain GGUF-format models. \
                 Try a GGUF-specific repo (e.g. 'TheBloke/phi-2-GGUF') or specify --hf-filename.",
                model_id
            )),
            1 => gguf_files[0].to_string(),
            _ => {
                info!("Found {} GGUF files, looking for '{}' quantization", gguf_files.len(), preferred_quant);
                let quant_lower = preferred_quant.to_lowercase();
                let selected = gguf_files.iter()
                    .find(|f| f.to_lowercase().contains(&quant_lower))
                    .or_else(|| gguf_files.first())
                    .unwrap()
                    .to_string();
                info!("Selected: {}", selected);
                selected
            }
        }
    } else {
        hf_filename.to_string()
    };
    
    info!("Downloading GGUF model file: {}", gguf_filename);
    let model_path = api.repo(repo.clone()).get(&gguf_filename).await
        .map_err(|e| anyhow::anyhow!("Failed to download model {}: {}", gguf_filename, e))?;

    // Try to download the tokenizer as well
    let tokenizer_path = match api.repo(repo.clone()).get("tokenizer.json").await {
        Ok(path) => {
            info!("Found and downloaded tokenizer.json from Hugging Face repo");
            path
        },
        Err(_) => {
            // Try alternative tokenizer files
            let alt_files = ["tokenizer.model", "vocab.json"];
            let mut found_tokenizer = None;
            
            for &filename in &alt_files {
                if let Ok(path) = api.repo(repo.clone()).get(filename).await {
                    info!("Found and downloaded {} from Hugging Face repo", filename);
                    found_tokenizer = Some(path);
                    break;
                }
            }
            
            if let Some(tokenizer_path) = found_tokenizer {
                tokenizer_path
            } else {
                // If no tokenizer found, look for config to possibly generate one or warn
                warn!("No tokenizer found in Hugging Face repo. Looking for config...");
                
                // Attempt to download config.json to see if we can extract tokenizer info
                if let Ok(_config_path) = api.repo(repo.clone()).get("config.json").await {
                    info!("Found config.json in Hugging Face repo");
                    // For now, we still need a tokenizer file, so we'll use the one in the model directory
                    // or prompt the user to provide one
                }
                
                // Create a path for tokenizer in the same directory as the model
                let mut tokenizer_path = model_path.clone();
                tokenizer_path.set_file_name("tokenizer.json");
                
                // Check if tokenizer exists in the same directory as the model file
                if !tokenizer_path.exists() {
                    return Err(anyhow::anyhow!(
                        "No tokenizer found in the Hugging Face repo and no tokenizer provided. \
                         Please ensure tokenizer.json exists or provide one manually."
                    ));
                }
                
                tokenizer_path
            }
        }
    };

    Ok((model_path, tokenizer_path))
}

fn build_device(args: &Args) -> candle_core::Device {
    match args.device.as_str() {
        "cuda" | "gpu" => {
            #[cfg(feature = "cuda")]
            {
                candle_core::Device::new_cuda(args.device_id)
                    .expect("Failed to initialize CUDA device")
            }
            #[cfg(not(feature = "cuda"))]
            {
                warn!("CUDA requested but not compiled with 'cuda' feature. Falling back to CPU.");
                candle_core::Device::Cpu
            }
        }
        "metal" => {
            #[cfg(feature = "metal")]
            {
                candle_core::Device::new_metal(args.device_id)
                    .expect("Failed to initialize Metal device")
            }
            #[cfg(not(feature = "metal"))]
            {
                warn!("Metal requested but not compiled with 'metal' feature. Falling back to CPU.");
                candle_core::Device::Cpu
            }
        }
        _ => candle_core::Device::Cpu,
    }
}