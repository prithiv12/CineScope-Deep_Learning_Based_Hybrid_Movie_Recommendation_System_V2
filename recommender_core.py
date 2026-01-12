import pandas as pd
import numpy as np
import requests
import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collaborative_filtering import predict_rating

# =====================================================
# LOAD DATA
# =====================================================
movies = pd.read_csv("movies_with_description.csv")

movies["title"] = movies["title"].fillna("")
movies["genres"] = movies["genres"].fillna("")
movies["description"] = movies["description"].fillna("")
movies["year"] = movies["year"].fillna(0).astype(int)
movies["rating"] = movies["rating"].fillna(0)
movies["votes"] = movies["votes"].fillna(0)

movies["clean_title"] = movies["title"].str.lower().str.strip()
movies = movies.drop_duplicates(subset="clean_title").reset_index(drop=True)

# =====================================================
# METADATA (FOR SBERT)
# =====================================================
movies["metadata"] = (
    (movies["genres"].str.replace(",", " ").str.lower() + " ") * 3
    + movies["description"].str.lower()
)

# =====================================================
# SBERT EMBEDDINGS (CACHED FOR STREAMLIT)
# =====================================================
@st.cache_resource(show_spinner="Loading SBERT model & embeddings...")
def load_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(
        movies["metadata"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=False
    )

embeddings = load_embeddings()

# =====================================================
# HELPERS
# =====================================================
def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-6)

def normalize_minmax(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin + 1e-6)

# =====================================================
# SEARCH
# =====================================================
def search_movie(query, limit=100):
    if not query:
        return pd.DataFrame()

    df = movies[movies["clean_title"].str.contains(query.lower(), na=False)].copy()
    df["score"] = (
        0.6 * normalize(df["rating"]) +
        0.4 * normalize(np.log1p(df["votes"]))
    )

    return df.sort_values("score", ascending=False).head(limit)[
        ["title", "year", "genres", "rating", "votes"]
    ]

def search_by_genre(genre, limit=100):
    if not genre:
        return pd.DataFrame()

    df = movies[movies["genres"].str.lower().str.contains(genre.lower(), na=False)].copy()
    df["score"] = (
        0.6 * normalize(df["rating"]) +
        0.4 * normalize(np.log1p(df["votes"]))
    )

    return df.sort_values("score", ascending=False).head(limit)[
        ["title", "year", "genres", "rating", "votes"]
    ]

# =====================================================
# SIMILAR MOVIES (SBERT)
# =====================================================
def get_movie_index(query):
    matches = movies[movies["clean_title"].str.contains(query.lower(), na=False)]
    return None if matches.empty else matches.index[0]

def get_similar_movies(query, top_n=50):
    idx = get_movie_index(query)
    if idx is None:
        return pd.DataFrame()

    sim = cosine_similarity(
        embeddings[idx].reshape(1, -1), embeddings
    ).flatten()

    df = movies.copy()
    df["similarity"] = sim
    df = df[df.index != idx]

    df["final_score"] = (
        0.6 * df["similarity"] +
        0.25 * normalize(df["rating"]) +
        0.15 * normalize(np.log1p(df["votes"]))
    )

    return df.sort_values("final_score", ascending=False).head(top_n)[
        ["title", "year", "genres", "rating", "votes"]
    ]

# =====================================================
# HEATMAP SUPPORT
# =====================================================
def compute_similarity_for_index(idx, top_n=10):
    sim = cosine_similarity(
        embeddings[idx].reshape(1, -1), embeddings
    ).flatten()
    indices = sim.argsort()[::-1]
    return indices[indices != idx][:top_n]

def compute_similarity_matrix(indices):
    selected_embeddings = embeddings[indices]
    return cosine_similarity(selected_embeddings)


# =====================================================
# HYBRID RECOMMENDATION
# =====================================================
def get_hybrid_recommendations(base_movie, user_id=1, alpha=0.6, top_n=10):
    idx = get_movie_index(base_movie)
    if idx is None:
        return pd.DataFrame()

    sim = cosine_similarity(
        embeddings[idx].reshape(1, -1), embeddings
    ).flatten()

    df = movies.copy()
    df["similarity"] = sim
    df = df[df.index != idx]

    sim_min, sim_max = df["similarity"].min(), df["similarity"].max()
    df["content_score"] = df["similarity"].apply(
        lambda x: normalize_minmax(x, sim_min, sim_max)
    )

    df["quality"] = (
        0.6 * normalize(df["rating"]) +
        0.4 * normalize(np.log1p(df["votes"]))
    )

    df["content_score"] = 0.7 * df["content_score"] + 0.3 * df["quality"]

    cf_vals = [predict_rating(user_id, i) for i in df.index]
    df["cf_raw"] = cf_vals

    valid_cf = df["cf_raw"].dropna()
    if not valid_cf.empty:
        cf_min, cf_max = valid_cf.min(), valid_cf.max()
        df["cf_score"] = df["cf_raw"].apply(
            lambda x: normalize_minmax(x, cf_min, cf_max)
            if x is not None else None
        )
    else:
        df["cf_score"] = None

    df["hybrid_score"] = df.apply(
        lambda r: r["content_score"]
        if r["cf_score"] is None
        else alpha * r["content_score"] + (1 - alpha) * r["cf_score"],
        axis=1
    )

    return df.sort_values("hybrid_score", ascending=False).head(top_n)[
        ["title", "year", "genres", "rating", "votes", "hybrid_score"]
    ]

# =====================================================
# EVALUATION
# =====================================================
def precision_recall_at_k(k=10, rating_threshold=7.0):
    precisions, recalls = [], []
    sample = movies.sample(min(100, len(movies)), random_state=42)

    relevant = movies[movies["rating"] >= rating_threshold]

    for _, row in sample.iterrows():
        recs = get_similar_movies(row["title"], k)
        if recs.empty:
            continue

        rel_recs = recs[recs["rating"] >= rating_threshold]
        precisions.append(len(rel_recs) / len(recs))
        recalls.append(len(rel_recs) / max(1, len(relevant)))

    return float(np.mean(precisions)), float(np.mean(recalls))

# =====================================================
# POSTER
# =====================================================
TMDB_API_KEY = "1a1c768f0babc9f17bf0ea5ffcc011f2"

def get_poster_url(title):
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": TMDB_API_KEY, "query": title},
            timeout=5
        ).json()
        return f"https://image.tmdb.org/t/p/w500{r['results'][0]['poster_path']}"
    except:
        return "https://dummyimage.com/300x450/000/fff&text=No+Poster"

