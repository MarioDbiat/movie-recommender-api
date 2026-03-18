import numpy as np
import faiss

# =========================
# Paths
# =========================
DEMO_EMB_PATH = r"C:\MovieReommenderSystem\movie-recommender-api\data\movie_embeddings_demo.npy"
DEMO_FAISS_PATH = r"C:\MovieReommenderSystem\movie-recommender-api\data\movie_index_demo.faiss"

# =========================
# Load embeddings
# =========================
embeddings = np.load(DEMO_EMB_PATH).astype("float32")
print("Loaded embeddings:", embeddings.shape)

# =========================
# Normalize for cosine similarity via inner product
# =========================
faiss.normalize_L2(embeddings)

d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)   # cosine-like search after normalization
index.add(embeddings)

print("FAISS index total vectors:", index.ntotal)

# =========================
# Save index
# =========================
faiss.write_index(index, DEMO_FAISS_PATH)
print("Saved demo FAISS index to:", DEMO_FAISS_PATH)