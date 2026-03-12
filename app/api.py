from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.engine import engine
from app.schemas import RecommendRequest, RecommendResponse

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


@app.get("/")
def root():
    return {"message": "Movie recommender API is running."}


@app.get("/health")
def health():
    return engine.health()


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