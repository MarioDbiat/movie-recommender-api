import os
import difflib
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODE = os.getenv("MODE", "auto").lower().strip()
_load_emb_raw = os.getenv("LOAD_EMB", "1")
LOAD_EMB = str(_load_emb_raw).strip().lower() in {"1", "true", "yes", "on"}

HF_REPO_ID = os.getenv("HF_REPO_ID", "Mariodb/movie-recommender-dataset").strip()
HF_REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset").strip()
HF_REVISION = os.getenv("HF_REVISION", "main").strip()
HF_TOKEN = os.getenv("HF_TOKEN") or None

FULL_PARQUET = os.getenv("FULL_PARQUET", "movies.parquet").strip()
FULL_INDEX = os.getenv("FULL_INDEX", "movie_index.faiss").strip()
FULL_EMB = os.getenv("FULL_EMB", "movie_embeddings.npy").strip()

MODEL_ID = os.getenv("MODEL_ID", "Mariodb/movie-recommender-model").strip()

ENABLE_LLM_REWRITE = os.getenv("ENABLE_LLM_REWRITE", "0").strip().lower() in {"1", "true", "yes", "on"}
REWRITE_MODEL_ID = os.getenv("REWRITE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct").strip()
REWRITE_MAX_NEW_TOKENS = int(os.getenv("REWRITE_MAX_NEW_TOKENS", "80"))
REWRITE_TEMPERATURE = float(os.getenv("REWRITE_TEMPERATURE", "0.0"))
USE_ORIGINAL_PLUS_REWRITE = os.getenv("USE_ORIGINAL_PLUS_REWRITE", "1").strip().lower() in {"1", "true", "yes", "on"}

TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

FAISS_OK = False
try:
    import faiss
    FAISS_OK = True
    try:
        faiss.omp_set_num_threads(1)
    except Exception:
        pass
except Exception:
    pass

REWRITE_STACK_OK = False
REWRITE_IMPORT_ERROR = ""
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    REWRITE_STACK_OK = True
except Exception as e:
    REWRITE_IMPORT_ERROR = str(e)


def _split_genres(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s or s.lower() == "unknown":
        return []
    for sep in ["|", ",", ";", "/"]:
        if sep in s:
            return [g.strip() for g in s.split(sep) if g.strip() and g.lower() != "unknown"]
    return [s] if s.lower() != "unknown" else []


_FRANCHISE_KEYWORDS = {
    "marvel": ["avengers", "iron man", "captain america", "thor", "hulk", "marvel"],
    "dc": ["batman", "superman", "wonder woman", "justice league", "joker", "dc"],
    "harry potter": ["harry potter", "hogwarts", "voldemort", "dumbledore"],
    "lord of the rings": ["lord of the rings", "frodo", "gandalf", "aragorn", "middle earth"],
    "star wars": ["star wars", "skywalker", "darth vader", "yoda", "jedi", "sith"],
    "fast & furious": ["fast and furious", "fast & furious", "dom toretto", "tokyo drift"],
    "transformers": ["transformers", "bumblebee", "optimus prime", "megatron"],
    "john wick": ["john wick", "continental", "high table"],
    "the matrix": ["matrix", "neo", "trinity", "morpheus", "agent smith"],
    "jurassic park": ["jurassic park", "jurassic world", "raptor", "velociraptor"],
}

NSFW_TERMS = ["nsfw", "adult", "porn", "xxx", "sex", "erotic", "babe"]

GENERIC_VIBE_WORDS = {
    "good", "nice", "best", "movie", "film", "something", "fun", "cool", "great",
    "watch", "story", "vibe", "vibes", "feels", "feel", "feeling", "atmosphere"
}

MOOD_WORDS = {
    "rainy", "cozy", "warm", "cold", "lonely", "sad", "dark", "dreamy",
    "nostalgic", "emotional", "beautiful", "quiet", "slow", "calm",
    "bittersweet", "haunting", "melancholic", "tense", "romantic", "tragic",
    "hopeful", "uplifting", "empty", "soft"
}

SETTING_WORDS = {
    "city", "town", "village", "space", "school", "house", "forest", "sea",
    "ocean", "mountains", "road", "train", "war", "prison", "hotel", "island"
}

GENRE_WORDS = {
    "drama", "thriller", "romance", "comedy", "horror", "sci-fi", "scifi",
    "science fiction", "mystery", "crime", "fantasy", "adventure", "action",
    "family", "animation", "animated", "psychological"
}

THEME_WORDS = {
    "grief", "loss", "love", "loneliness", "memory", "healing", "revenge",
    "survival", "friendship", "betrayal", "obsession", "identity", "hope",
    "isolation", "family", "death", "regret", "childhood"
}


def detect_franchise(title: str, overview: str) -> str:
    text = f"{str(title)} {str(overview)}".lower()
    matches = {}
    for franchise, kws in _FRANCHISE_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                matches[franchise] = matches.get(franchise, 0) + 1
    return max(matches, key=matches.get) if matches else ""


def _ensure_franchise(df: pd.DataFrame) -> pd.DataFrame:
    if "franchise" not in df.columns:
        df["franchise"] = ""
    mask = df["franchise"].fillna("").str.strip()
    mask = (mask == "") | (mask.str.lower() == "unknown")
    if mask.any():
        df.loc[mask, "franchise"] = [
            detect_franchise(t, o) for t, o in zip(
                df.loc[mask, "title"].astype(str),
                df.loc[mask, "overview"].astype(str) if "overview" in df.columns else ["" for _ in range(mask.sum())]
            )
        ]
    return df


def _clean_tokens(text: str) -> List[str]:
    return [w.strip(".,!?;:()[]{}\"'").lower() for w in text.split() if w.strip()]


def analyze_query_style(query: str) -> dict:
    tokens = _clean_tokens(query)
    generic_count = sum(t in GENERIC_VIBE_WORDS for t in tokens)
    mood_count = sum(t in MOOD_WORDS for t in tokens)
    setting_count = sum(t in SETTING_WORDS for t in tokens)
    genre_count = sum(t in GENRE_WORDS for t in tokens)
    theme_count = sum(t in THEME_WORDS for t in tokens)

    has_plot_signals = any(
        phrase in query.lower()
        for phrase in [
            "about", "who", "struggles", "trying to", "must", "after", "when",
            "set in", "follows", "discovers", "forced to", "falls in love",
            "investigates", "journey", "survive", "survival"
        ]
    )

    meaningful_count = mood_count + setting_count + genre_count + theme_count

    return {
        "word_count": len(tokens),
        "generic_count": generic_count,
        "mood_count": mood_count,
        "setting_count": setting_count,
        "genre_count": genre_count,
        "theme_count": theme_count,
        "meaningful_count": meaningful_count,
        "has_plot_signals": has_plot_signals,
        "is_vibe_heavy": mood_count >= 1 and not has_plot_signals,
        "is_too_short": len(tokens) < 6,
        "is_too_generic": meaningful_count == 0 or (generic_count >= max(1, len(tokens) // 2)),
    }


def _build_fallback_rewrite(user_query: str) -> str:
    q = user_query.strip()
    if not q:
        return q
    info = analyze_query_style(q)
    if info["is_too_short"] or info["is_vibe_heavy"] or info["is_too_generic"]:
        return f"{q}. A character-driven film focused on emotion, tone, and human conflict."
    return q


def should_rewrite_query(query: str, search_mode: str) -> bool:
    q = query.strip()
    if not q:
        return False
    if search_mode == "Movie title":
        return False
    info = analyze_query_style(q)
    return info["is_too_short"] or info["is_vibe_heavy"] or info["is_too_generic"]


def _build_rewrite_prompt(user_query: str) -> str:
    return (
        "Rewrite the following movie search request into a concise, retrieval-friendly movie overview style description.\n"
        "Keep the original meaning, mood, and intent.\n"
        "Output only the rewritten query.\n\n"
        f"User query: {user_query}"
    )


def _llm_rewrite_query(user_query: str) -> str:
    tokenizer, model, device = load_rewriter()
    prompt = _build_rewrite_prompt(user_query)

    messages = [
        {"role": "system", "content": "You rewrite movie preference queries into retrieval-friendly overview-style descriptions."},
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        )
    else:
        fallback_prompt = "\n\n".join([m["content"] for m in messages])
        inputs = tokenizer(fallback_prompt, return_tensors="pt").input_ids

    inputs = inputs.to(device)
    attention_mask = torch.ones_like(inputs)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=REWRITE_MAX_NEW_TOKENS,
            do_sample=REWRITE_TEMPERATURE > 0,
            temperature=max(REWRITE_TEMPERATURE, 1e-5),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError("Rewrite model returned empty output.")
    return text.splitlines()[0].strip().strip('"')


def rewrite_query_to_imdb_style(user_query: str) -> Tuple[str, str]:
    fallback = _build_fallback_rewrite(user_query)
    if not ENABLE_LLM_REWRITE:
        return fallback, "rule-based fallback (LLM disabled)"
    try:
        rewritten = _llm_rewrite_query(user_query)
        return rewritten, f"LLM rewrite via {REWRITE_MODEL_ID}"
    except Exception as e:
        return fallback, f"rule-based fallback ({e})"


def prepare_query_for_retrieval(original_query: str, rewritten_query: str) -> str:
    original = original_query.strip()
    rewritten = rewritten_query.strip()
    if not original:
        return rewritten
    if not rewritten:
        return original
    if not USE_ORIGINAL_PLUS_REWRITE:
        return rewritten
    if original.lower() == rewritten.lower():
        return rewritten
    return f"{original}. {rewritten}"


class RecommenderEngine:
    def __init__(self):
        self.meta = None
        self.model = None
        self.index = None
        self.embeddings = None
        self.rewriter = None

    def _hf_path(self, filename: str) -> str:
        return hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type=HF_REPO_TYPE,
            revision=HF_REVISION,
            local_dir="data",
            token=HF_TOKEN,
        )

    def load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(MODEL_ID)
        return self.model

    def load_rewriter(self):
        if self.rewriter is not None:
            return self.rewriter
        if not ENABLE_LLM_REWRITE:
            raise RuntimeError("LLM rewriting disabled.")
        if not REWRITE_STACK_OK:
            raise RuntimeError(f"Rewrite stack unavailable: {REWRITE_IMPORT_ERROR}")

        tokenizer = AutoTokenizer.from_pretrained(REWRITE_MODEL_ID, token=HF_TOKEN)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            REWRITE_MODEL_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            token=HF_TOKEN,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        self.rewriter = (tokenizer, model, device)
        return self.rewriter

    def load_metadata(self):
        if self.meta is None:
            path = self._hf_path(FULL_PARQUET)
            df = pd.read_parquet(path)
            for col in ["title", "overview", "genres", "franchise"]:
                if col in df.columns:
                    df[col] = df[col].fillna("").astype(str)
            if "popularity" in df.columns:
                df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
            df = _ensure_franchise(df)
            self.meta = df
        return self.meta

    def load_faiss(self):
        if self.index is None and FAISS_OK:
            path = self._hf_path(FULL_INDEX)
            self.index = faiss.read_index(path)
        return self.index

    def load_embeddings(self):
        if self.embeddings is None and LOAD_EMB:
            path = self._hf_path(FULL_EMB)
            arr = np.load(path, mmap_mode="r")
            self.embeddings = arr.astype("float32") if arr.dtype != np.float32 else arr
        return self.embeddings

    def startup(self):
        self.load_metadata()
        self.load_model()
        self.load_faiss()
        self.load_embeddings()

    def health(self):
        return {
        "status": "ok",
        "metadata_loaded": self.meta is not None,
        "model_loaded": self.model is not None,
        "faiss_loaded": self.index is not None,
        "embeddings_loaded": self.embeddings is not None,
        "llm_rewrite_enabled": ENABLE_LLM_REWRITE,
    }

    def recommend(
        self,
        query: str,
        franchise_only: bool = False,
        safe_mode: bool = True,
        use_genres: bool = False,
        use_popularity: bool = True,
        top_k: int = 10,
        search_mode: str = "Describe a movie vibe / story",
    ):
        meta = self.load_metadata()
        model = self.load_model()
        index = self.load_faiss()
        embeddings = self.load_embeddings()

        raw_query = query.strip()
        final_query = raw_query
        rewritten_query = None
        rewrite_source = None

        if search_mode == "Describe a movie vibe / story" and should_rewrite_query(raw_query, search_mode):
            rewritten_query, rewrite_source = rewrite_query_to_imdb_style(raw_query)
            final_query = prepare_query_for_retrieval(raw_query, rewritten_query)

        query_vec = model.encode([final_query], convert_to_numpy=True)[0].astype("float32")
        query_np = query_vec.reshape(1, -1).astype("float32")

        candidate_idx = None
        if index is not None:
            _, idxs = index.search(query_np, max(top_k * 20, top_k))
            candidate_idx = idxs[0]

        if (candidate_idx is None or len(candidate_idx) == 0) and embeddings is not None:
            q_norm = np.linalg.norm(query_vec) + 1e-12
            emb_norms = np.linalg.norm(embeddings, axis=1) + 1e-12
            sims_all = (embeddings @ query_vec) / (emb_norms * q_norm)
            candidate_idx = np.argsort(-sims_all)[: max(top_k * 20, top_k)]

        if candidate_idx is None or len(candidate_idx) == 0:
            return {
                "original_query": raw_query,
                "final_query": final_query,
                "rewritten_query": rewritten_query,
                "rewrite_source": rewrite_source,
                "results": [],
            }

        candidate_idx = np.asarray(candidate_idx, dtype=int)
        candidate_idx = candidate_idx[(candidate_idx >= 0) & (candidate_idx < len(meta))]
        results = meta.iloc[candidate_idx].copy()

        if embeddings is not None:
            cand_embeddings = embeddings[candidate_idx]
            query_norm = np.linalg.norm(query_vec) + 1e-12
            cand_norms = np.linalg.norm(cand_embeddings, axis=1) + 1e-12
            sims = (cand_embeddings @ query_vec) / (cand_norms * query_norm)
        else:
            texts = results["overview"].fillna("").astype(str).tolist()
            cand_vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
            q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
            cand_norm = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-12)
            sims = cand_norm @ q_norm

        results = results.assign(similarity=sims.astype("float32"), score=sims.astype("float32"))

        if use_popularity and "popularity" in results.columns:
            pop_vals = pd.to_numeric(results["popularity"], errors="coerce").fillna(0).to_numpy("float32")
            pop_adj = np.log1p(np.clip(pop_vals, 0, None)).astype("float32")
            if pop_adj.size > 0 and float(pop_adj.max()) > 0:
                results["score"] = results["similarity"] * pop_adj

        if safe_mode:
            pattern = "(?i)" + "|".join(NSFW_TERMS)
            mask = pd.Series(False, index=results.index)
            for col in ["genres", "title", "overview"]:
                if col in results.columns:
                    mask |= results[col].astype(str).str.contains(pattern, na=False, regex=True)
            results = results.loc[~mask]

        if "poster_path" in results.columns:
            results["poster_url"] = results["poster_path"].astype(str).str.strip()
            mask = results["poster_url"].str.len() > 0
            results.loc[mask, "poster_url"] = TMDB_IMG_BASE + results.loc[mask, "poster_url"]
        else:
            results["poster_url"] = ""

        results = results.sort_values(by="score", ascending=False).head(top_k).reset_index(drop=True)

        payload = []
        for _, row in results.iterrows():
            payload.append({
                "title": str(row.get("title", "")),
                "genres": str(row.get("genres", "")),
                "franchise": str(row.get("franchise", "")),
                "popularity": float(row.get("popularity", 0.0) or 0.0),
                "overview": str(row.get("overview", "")),
                "poster_url": str(row.get("poster_url", "")),
                "similarity": float(row.get("similarity", 0.0) or 0.0),
                "score": float(row.get("score", 0.0) or 0.0),
            })

        return {
            "original_query": raw_query,
            "final_query": final_query,
            "rewritten_query": rewritten_query,
            "rewrite_source": rewrite_source,
            "results": payload,
        }


engine = RecommenderEngine()
load_rewriter = engine.load_rewriter