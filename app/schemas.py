from typing import List, Optional
from pydantic import BaseModel


class RecommendRequest(BaseModel):
    query: str
    franchise_only: bool = False
    safe_mode: bool = True
    use_genres: bool = False
    use_popularity: bool = True
    top_k: int = 10
    search_mode: str = "Describe a movie vibe / story"


class MovieResult(BaseModel):
    title: str
    genres: Optional[str] = ""
    franchise: Optional[str] = ""
    popularity: Optional[float] = 0.0
    overview: Optional[str] = ""
    poster_url: Optional[str] = ""
    similarity: Optional[float] = 0.0
    score: Optional[float] = 0.0


class RecommendResponse(BaseModel):
    original_query: str
    final_query: str
    rewritten_query: Optional[str] = None
    rewrite_source: Optional[str] = None
    results: List[MovieResult]