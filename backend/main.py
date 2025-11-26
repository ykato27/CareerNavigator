from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import upload, recommendation, train, weights, graph

app = FastAPI()

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
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

@app.get("/")
async def root():
    return {"message": "Welcome to CareerNavigator API"}
