mod api;
mod middleware;
mod model;
mod sampling;
mod tokenizer_utils;

use actix_cors::Cors;
use actix_web::{web, App, HttpServer, middleware as actix_middleware};
use clap::Parser;
use model::ModelEngine;
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
    #[arg(short, long)]
    model: String,

    /// Path to the tokenizer.json file
    #[arg(short, long)]
    tokenizer: String,

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
    info!("Model path  : {}", args.model);
    info!("Tokenizer   : {}", args.tokenizer);
    info!("Device      : {}", args.device);
    info!("Max concurrent: {}", args.max_concurrent);

    let threads = if args.threads == 0 {
        num_cpus::get()
    } else {
        args.threads
    };
    info!("CPU threads : {}", threads);

    // Set rayon / candle thread count
    std::env::set_var("RAYON_NUM_THREADS", threads.to_string());

    let device = build_device(&args);

    let engine = ModelEngine::new(&args.model, &args.tokenizer, &device, args.ctx_len)
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