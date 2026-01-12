from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from recommender_core import *
from recommender_core import movies          # 🔧 FIX 1: required for heatmap
from collaborative_filtering import train_collaborative_model

RESULTS_PER_PAGE = 10

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="🎬 CineScope – Movie Recommender",
    layout="wide"
)

# ==================================================
# UI CSS (UNCHANGED)
# ==================================================
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0f172a, #020617);
}
.block-container {
    padding-top: 1.5rem;
}
h1, h3 {
    font-family: 'Segoe UI', sans-serif;
    color: #FFFFF0;
}
h2 {
    font-family: 'Segoe UI', sans-serif;
    color: #111213;
}
.card {
    background: rgba(180, 7, 16, 0.8);
    backdrop-filter: blur(12px);
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.45);
    margin-bottom: 20px;
}
.poster {
    border-radius: 14px;
}
.badge {
    display: inline-block;
    padding: 6px 12px;
    background: linear-gradient(90deg, #f97316, #000000);
    color: white;
    border-radius: 999px;
    font-weight: 700;
}
.empty-box {
    text-align: center;
    padding: 35px;
    background: rgba(2,6,23,0.7);
    border-radius: 18px;
    color: #94a3b8;
}
div.stButton > button {
    background: rgba(180, 7, 16, 0.8);
    color: white;
    border-radius: 999px;
    padding: 0.45rem 1.5rem;
    border: none;
    font-weight: 700;
}
.info-card {
    background: rgba(15, 23, 42, 0.85);
    border-left: 5px solid #f59e0b;
    padding: 18px 22px;
    border-radius: 14px;
    margin: 15px 0;
    color: #e5e7eb;
    font-size: 0.95rem;
    line-height: 1.6;
}
.info-card strong {
    color: #fbbf24;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# SESSION STATE
# ==================================================
defaults = {
    "logged_in": False,
    "username": "",
    "search_page": 0,
    "genre_page": 0,
    "similar_page": 0,
    "cf_trained": False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==================================================
# LOGIN
# ==================================================
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center;'>🎬 CineScope</h1>", unsafe_allow_html=True)
    name = st.text_input("Enter your name")
    if st.button("Enter App") and name.strip():
        st.session_state.logged_in = True
        st.session_state.username = name
    st.stop()

# ==================================================
# TRAIN COLLABORATIVE FILTERING (ONCE)
# ==================================================
if not st.session_state.cf_trained:
    with st.spinner("Training collaborative filtering model..."):
        st.session_state.rmse = train_collaborative_model("ratings.csv")
        st.session_state.cf_trained = True

# ==================================================
# HEADER
# ==================================================
st.markdown(f"""
<div style="background:#E50914;padding:25px;border-radius:20px;">
<h1>👋 Welcome, {st.session_state.username}</h1>
<p style="color:#cbd5e1;">Hybrid movie discovery using IMDb, NLP & User Preferences</p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# TABS
# ==================================================
tab1, tab2 = st.tabs(["🔍 Recommendations", "📊 Model Evaluation"])

# ==================================================
# 🔍 TAB 1: RECOMMENDATIONS
# ==================================================
with tab1:

    st.header("🔍 Search Movies")
    query = st.text_input("Movie title", key="search_query")  # 🔧 FIX 3

    if st.button("Search Movie"):
        st.session_state.search_results = search_movie(query)

    if query and "search_results" in st.session_state:
        for _, row in st.session_state.search_results.head(RESULTS_PER_PAGE).iterrows():
            st.markdown(f"""
            <div class="card">
                <h3>{row['title']} ({row['year']})</h3>
                <p>🎭 {row['genres']}</p>
                <span class="badge">⭐ {row['rating']}</span>
            </div>
            """, unsafe_allow_html=True)

    st.header("🎥 Similar Movies")
    base_movie = st.text_input("Base movie", key="base_movie")  # 🔧 FIX 3

    if st.button("Find Similar Movies"):
        st.session_state.similar_results = get_similar_movies(base_movie)

    if base_movie and "similar_results" in st.session_state:
        for _, row in st.session_state.similar_results.head(RESULTS_PER_PAGE).iterrows():
            st.markdown(f"""
            <div class="card">
                <h3>{row['title']}</h3>
                <span class="badge">⭐ {row['rating']}</span>
            </div>
            """, unsafe_allow_html=True)

    st.header("🤝 Hybrid Recommendations")

    user_id = st.number_input("User ID (MovieLens)", min_value=1, value=1)
    alpha = st.slider("Content vs User Preference Weight (α)", 0.0, 1.0, 0.6)

    if st.button("Get Hybrid Recommendations"):
        if base_movie.strip() == "":                     # 🔧 FIX 2
            st.warning("Please enter a base movie title.")
        else:
            hybrid_results = get_hybrid_recommendations(base_movie, user_id, alpha)
            for _, row in hybrid_results.iterrows():
                st.markdown(f"""
                <div class="card">
                    <h3>{row['title']} ({row['year']})</h3>
                    <p>🎭 {row['genres']}</p>
                    <span class="badge">⭐ {row['rating']}</span>
                    <p>Hybrid Score: {row['hybrid_score']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)

    st.header("🔎 Similarity Heatmap")

    if st.button("Show Similarity Heatmap"):
        idx = get_movie_index(base_movie)
        if idx is not None:
            top_idx = compute_similarity_for_index(idx, 10)
            labels = movies.iloc[top_idx]["title"]       # 🔧 FIX 1 used here
            matrix = compute_similarity_matrix(top_idx)

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap="magma", ax=ax)
            plt.xticks(rotation=90)
            st.pyplot(fig)

# ==================================================
# 📊 TAB 2: MODEL EVALUATION
# ==================================================
with tab2:
    st.header("📊 Model Evaluation (Offline Metrics)")

    st.metric("Collaborative Filtering RMSE", f"{st.session_state.rmse:.3f}")

    precision, recall = precision_recall_at_k(k=10, rating_threshold=7.0)

    col1, col2 = st.columns(2)
    col1.metric("Precision@10", f"{precision:.3f}")
    col2.metric("Recall@10", f"{recall:.3f}")

    st.markdown("""
    ---
    **Note:**  
    RMSE, Precision, and Recall are computed offline using a fixed evaluation setup.
    The hybrid α slider affects recommendation fusion only and does not influence
    evaluation metrics.
    """)
