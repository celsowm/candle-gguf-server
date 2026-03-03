use actix_web::{web, App, HttpResponse, HttpServer, middleware};
use serde::Serialize;

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    message: String,
}

async fn health() -> HttpResponse {
    let response = HealthResponse {
        status: "ok".to_string(),
        message: "Server is running".to_string(),
    };
    HttpResponse::Ok().json(response)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Starting server...");
    
    HttpServer::new(|| {
        App::new()
            .wrap(middleware::Logger::default())
            .route("/health", web::get().to(health))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}