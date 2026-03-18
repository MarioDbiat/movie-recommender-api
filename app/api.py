from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.engine import engine
from app.schemas import RecommendRequest, RecommendResponse

import os

app = FastAPI(
    title="Mario Movie Recommender API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    engine.load_metadata()
    engine.load_faiss()

@app.get("/")
def root():
    return {"message": "Movie recommender API is running."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "metadata_loaded": engine.meta is not None,
        "model_loaded": engine.model is not None,
        "faiss_loaded": engine.index is not None,
        "embeddings_loaded": engine.embeddings is not None,
        "llm_rewrite_enabled": os.getenv("ENABLE_LLM_REWRITE"),
        "FULL_PARQUET": os.getenv("FULL_PARQUET"),
        "FULL_INDEX": os.getenv("FULL_INDEX"),
        "FULL_EMB": os.getenv("FULL_EMB"),
        "LOAD_EMB": os.getenv("LOAD_EMB"),
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return engine.recommend(
        query=req.query,
        franchise_only=req.franchise_only,
        safe_mode=req.safe_mode,
        use_genres=req.use_genres,
        use_popularity=req.use_popularity,
        top_k=req.top_k,
        search_mode=req.search_mode,
    )