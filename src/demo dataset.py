import pandas as pd
import numpy as np

# =========================
# Paths
# =========================
FULL_DATA_PATH = r"C:\MovieReommenderSystem\TMDB_movie_dataset_v12.csv"
DEMO_PARQUET_PATH = r"C:\MovieReommenderSystem\movie-recommender-api\data\movies_demo.parquet"

# =========================
# Config
# =========================
TOP_POPULAR_N = 20000
RANDOM_SAMPLE_N = 10000
MIN_OVERVIEW_LEN = 30
RANDOM_STATE = 42

# =========================
# Load full dataset
# =========================
df = pd.read_csv(FULL_DATA_PATH)
print("Loaded full dataset:", len(df))

# =========================
# Basic cleaning
# =========================
for col in ["title", "overview", "genres"]:
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str)

if "popularity" in df.columns:
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
else:
    df["popularity"] = 0.0

# remove duplicate titles + overviews if present
subset_cols = [c for c in ["title", "overview"] if c in df.columns]
if subset_cols:
    df = df.drop_duplicates(subset=subset_cols).reset_index(drop=True)

# remove rows with empty/weak overview
df = df[df["overview"].str.strip().str.len() >= MIN_OVERVIEW_LEN].copy()

# remove rows with empty title
df = df[df["title"].str.strip() != ""].copy()

print("After cleaning:", len(df))

# =========================
# Better strategy:
# 1) top popular movies
# 2) diverse random sample from the rest
# =========================
df_sorted = df.sort_values("popularity", ascending=False).reset_index(drop=True)

top_popular = df_sorted.head(TOP_POPULAR_N).copy()
remaining = df_sorted.iloc[TOP_POPULAR_N:].copy()

random_n = min(RANDOM_SAMPLE_N, len(remaining))
random_sample = remaining.sample(n=random_n, random_state=RANDOM_STATE)

df_demo = pd.concat([top_popular, random_sample], ignore_index=True)

# final dedup
subset_cols = [c for c in ["title", "overview"] if c in df_demo.columns]
if subset_cols:
    df_demo = df_demo.drop_duplicates(subset=subset_cols).reset_index(drop=True)

# optional: sort again by popularity a bit so demo looks stronger
df_demo = df_demo.sort_values("popularity", ascending=False).reset_index(drop=True)

print("Demo dataset size:", len(df_demo))

# =========================
# Save demo parquet
# =========================
df_demo.to_parquet(DEMO_PARQUET_PATH, index=False)
print("Saved demo parquet to:", DEMO_PARQUET_PATH)