import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# =========================
# Paths
# =========================
DEMO_PARQUET_PATH = r"C:\MovieReommenderSystem\movie-recommender-api\data\movies_demo.parquet"
MODEL_PATH = r"C:\MovieReommenderSystem\SBERT_Movie_Recommender_v1"
DEMO_EMB_PATH = r"C:\MovieReommenderSystem\movie-recommender-api\data\movie_embeddings_demo.npy"

# =========================
# Load demo dataset
# =========================
df_demo = pd.read_parquet(DEMO_PARQUET_PATH)
print("Demo movie count:", len(df_demo))

df_demo["overview"] = df_demo["overview"].fillna("").astype(str)

# =========================
# Load model
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_PATH)
model.to(device)

print(f"Model loaded on: {device}")

# =========================
# Encode demo overviews
# =========================
texts = df_demo["overview"].tolist()

demo_embeddings = model.encode(
    texts,
    batch_size=32,                 # can increase if GPU memory allows
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=False
)

demo_embeddings = demo_embeddings.astype("float32")
print("Embeddings shape:", demo_embeddings.shape)

# =========================
# Save
# =========================
np.save(DEMO_EMB_PATH, demo_embeddings)
print("Saved demo embeddings to:", DEMO_EMB_PATH)