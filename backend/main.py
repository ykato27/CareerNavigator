from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import (
    upload,
    recommendation,
    train,
    weights,
    graph,
    organizational,
    career_dashboard,
    role_based_dashboard,
)
from backend.middleware import error_handler_middleware, logging_middleware
from backend.core.logging import configure_logging

# Configure structured logging
configure_logging(log_level="INFO", json_logs=False)  # Set json_logs=True in production

app = FastAPI(
    title="CareerNavigator API",
    description="AI-powered career recommendation system with causal inference",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Middleware registration order matters!
# Logging middleware should be first to capture all requests
app.middleware("http")(logging_middleware)
# Error handler should be after logging to catch errors in logged requests
app.middleware("http")(error_handler_middleware)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーター登録
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(recommendation.router, prefix="/api", tags=["recommendation"])
app.include_router(train.router, prefix="/api", tags=["train"])
app.include_router(weights.router, prefix="/api", tags=["weights"])
app.include_router(graph.router, prefix="/api", tags=["graph"])
app.include_router(organizational.router, prefix="/api", tags=["organizational"])
app.include_router(career_dashboard.router, prefix="/api/career", tags=["career-dashboard"])
app.include_router(
    role_based_dashboard.router, prefix="/api/career/role", tags=["role-based-career"]
)


@app.get("/")
async def root():
    return {"message": "Welcome to CareerNavigator API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "version": "1.0.0"}
