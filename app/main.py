"""
Complete Movie Recommendation System Dashboard
7 pages: Home | Content-Based | Collaborative | Hybrid | EDA | Evaluation | About
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { font-size: 1.1rem; color: #666; margin-bottom: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem; border-radius: 12px; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #667eea; }
    .metric-label { font-size: 0.85rem; color: #555; margin-top: 4px; }
    .movie-card {
        background: white; border-radius: 10px; padding: 1rem;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 0.8rem;
    }
    .score-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; padding: 3px 10px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600;
    }
    .explanation-box {
        background: #f0f4ff; border-left: 4px solid #667eea;
        padding: 1rem; border-radius: 8px; margin-top: 1rem;
    }
    .stSelectbox > div > div { background: #f8f9ff; }
    .sidebar .sidebar-content { background: #1a1a2e; }
</style>
""", unsafe_allow_html=True)


# ── Load models (cached so they only load once per session) ───────────────────
@st.cache_resource(show_spinner="🎬 Loading recommendation models...")
def load_models():
    from src.utils.model_cache import load_cache
    return load_cache()


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Movie Recommender")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home",
         "🔍 Content-Based",
         "👥 Collaborative",
         "🔀 Hybrid Engine",
         "📊 Data Analysis",
         "📈 Evaluation",
         "ℹ️ About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**ML Stack**")
    st.markdown("- TF-IDF + Cosine Similarity")
    st.markdown("- KNN Collaborative Filtering")
    st.markdown("- SVD Matrix Factorization")
    st.markdown("- Weighted Hybrid Engine")
    st.markdown("---")
    st.caption("Built with MovieLens Dataset")

# Load models
cache = load_models()
movies = cache["movies_clean"]
ratings = cache["ratings_clean"]
user_features = cache["user_features"]
content_rec = cache["content_rec"]
item_cf = cache["item_cf"]
user_cf = cache["user_cf"]
mf_model = cache["mf_model"]
hybrid = cache["hybrid"]
soup_df = cache["soup"]

all_titles = sorted(soup_df["title"].dropna().tolist())
all_users = sorted(ratings["userId"].unique().tolist())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<p class="main-header">🎬 Movie Recommendation System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Hybrid engine combining Content-Based · Collaborative · Matrix Factorization</p>', unsafe_allow_html=True)

    # ── Key metrics row ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{movies["movieId"].nunique():,}</div>
            <div class="metric-label">📽️ Total Movies</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{ratings["userId"].nunique():,}</div>
            <div class="metric-label">👤 Total Users</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(ratings):,}</div>
            <div class="metric-label">⭐ Total Ratings</div></div>""", unsafe_allow_html=True)
    with col4:
        sparsity = (1 - len(ratings) / (movies["movieId"].nunique() * ratings["userId"].nunique())) * 100
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{sparsity:.1f}%</div>
            <div class="metric-label">🕳️ Matrix Sparsity</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Top movies ──
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🏆 Top 10 Most Rated Movies")
        top10 = (movies[movies["n_ratings"] > 0]
                 .nlargest(10, "n_ratings")[["title", "n_ratings", "avg_rating"]]
                 .reset_index(drop=True))
        top10.index = top10.index + 1
        top10.columns = ["Title", "# Ratings", "Avg Rating"]
        top10["Avg Rating"] = top10["Avg Rating"].round(2)
        st.dataframe(top10, use_container_width=True)

    with col_right:
        st.subheader("⭐ Highest Rated Movies (min 50 ratings)")
        top_quality = (movies[movies["n_ratings"] >= 50]
                       .nlargest(10, "avg_rating")[["title", "avg_rating", "n_ratings"]]
                       .reset_index(drop=True))
        top_quality.index = top_quality.index + 1
        top_quality.columns = ["Title", "Avg Rating", "# Ratings"]
        top_quality["Avg Rating"] = top_quality["Avg Rating"].round(3)
        st.dataframe(top_quality, use_container_width=True)

    st.markdown("---")

    # ── Architecture diagram ──
    st.subheader("🏗️ System Architecture")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        **📄 Content-Based**
Movie Genres + Tags
          ↓
    TF-IDF Vectorization
          ↓
    Cosine Similarity
          ↓
    Similar Movies
""")
    with col_b:
        st.markdown("""
        **👥 Collaborative**
User-Item Matrix
          ↓
    KNN (k=20 neighbors)
          ↓
    Similarity Scores
          ↓
    Personalized Recs
""")
    with col_c:
        st.markdown("""
        **🔀 Hybrid Engine**
Content (20%)
         +
    Collaborative (30%)
         +
    SVD/MF (50%)
         ↓
    Weighted Final Score
""")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CONTENT-BASED
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Content-Based":
    st.markdown('<p class="main-header">🔍 Content-Based Recommendations</p>', unsafe_allow_html=True)
    st.markdown("Finds movies similar to your choice based on **genres and user tags** using TF-IDF + Cosine Similarity.")

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_title = st.selectbox("🎬 Select a movie:", all_titles, index=0)
    with col2:
        k = st.slider("Number of recommendations:", 5, 20, 10)

    if st.button("🔍 Find Similar Movies", type="primary", use_container_width=True):
        try:
            results = content_rec.get_similar_movies(selected_title, k=k)

            # Show selected movie info
            movie_info = movies[movies["title"] == selected_title]
            if not movie_info.empty:
                m = movie_info.iloc[0]
                st.info(f"**Selected:** {selected_title}  |  **Genres:** {m['genres']}  |  "
                        f"**Avg Rating:** {m['avg_rating']:.2f}⭐  |  **# Ratings:** {m['n_ratings']:,}")

            st.subheader(f"Top {k} Similar Movies")
            for i, row in results.iterrows():
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    movie_data = movies[movies["movieId"] == row["movieId"]]
                    avg_r = f"{movie_data['avg_rating'].values[0]:.2f}⭐" if not movie_data.empty and movie_data['avg_rating'].values[0] > 0 else "N/A"
                    st.markdown(f"""<div class="movie-card">
                        <strong>{i+1}. {row['title']}</strong><br>
                        <small>🎭 {row['genres']} &nbsp;|&nbsp; {avg_r}</small>
                    </div>""", unsafe_allow_html=True)
                with col_b:
                    score_pct = int(row["similarity_score"] * 100)
                    st.metric("Match", f"{score_pct}%")

            # Similarity bar chart
            fig = px.bar(results, x="similarity_score", y="title",
                         orientation="h", title="Content Similarity Scores",
                         color="similarity_score",
                         color_continuous_scale="Viridis",
                         labels={"similarity_score": "Cosine Similarity", "title": "Movie"})
            fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

            # Explanation box
            st.markdown(f"""<div class="explanation-box">
                <strong>💡 Why these movies?</strong><br>
                Movies are ranked by <b>cosine similarity</b> of their TF-IDF content vectors —
                computed from genre tags (double-weighted as more reliable signal) and user-applied
                free-text tags. A score of 1.0 = identical content profile; 0.0 = no overlap.
                The top result shares genre/tag profile most closely with <em>{selected_title}</em>.
            </div>""", unsafe_allow_html=True)

        except ValueError as e:
            st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — COLLABORATIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Collaborative":
    st.markdown('<p class="main-header">👥 Collaborative Filtering</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["👤 User-Based CF", "🎬 Item-Based CF"])

    with tab1:
        st.markdown("Finds users most similar to you, then recommends what they rated highly.")
        col1, col2 = st.columns([2, 1])
        with col1:
            user_id = st.selectbox("Select User ID:", all_users, key="user_cf_uid")
        with col2:
            k_u = st.slider("Recommendations:", 5, 20, 10, key="k_user")

        if st.button("👤 Get User-Based Recommendations", type="primary"):
            recs = user_cf.recommend(user_id, k=k_u)
            if len(recs) == 0:
                st.warning("No recommendations found. This user may be too sparse for CF.")
            else:
                # Show user profile
                u_data = user_features[user_features["userId"] == user_id]
                if not u_data.empty:
                    u = u_data.iloc[0]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Movies Rated", int(u["n_ratings"]))
                    c2.metric("Avg Rating Given", f"{u['avg_rating']:.2f}")
                    c3.metric("Rating Consistency", f"σ={u['rating_std']:.2f}")

                st.subheader(f"Top {k_u} Recommendations for User {user_id}")
                merged = recs.merge(movies[["movieId", "title", "genres", "avg_rating"]], on="movieId", how="left")
                for i, row in merged.iterrows():
                    st.markdown(f"""<div class="movie-card">
                        <strong>{i+1}. {row.get('title','Unknown')}</strong>
                        &nbsp;<span class="score-badge">Score: {row['predicted_score']:.3f}</span><br>
                        <small>🎭 {row.get('genres','N/A')} &nbsp;|&nbsp; ⭐ {row.get('avg_rating',0):.2f} global avg</small>
                    </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("Finds movies with similar rating patterns to a chosen movie.")
        col1, col2 = st.columns([2, 1])
        with col1:
            item_title = st.selectbox("Select a movie:", all_titles, key="item_cf_title")
        with col2:
            k_i = st.slider("Similar movies:", 5, 20, 10, key="k_item")

        if st.button("🎬 Find Behaviorally Similar Movies", type="primary"):
            movie_id_row = movies[movies["title"] == item_title]
            if movie_id_row.empty:
                st.error("Movie not found.")
            else:
                mid = int(movie_id_row.iloc[0]["movieId"])
                sims = item_cf.get_similar_movies(mid, k=k_i)
                if len(sims) == 0:
                    st.warning("No CF signal — this movie may have very few ratings.")
                else:
                    merged = sims.merge(movies[["movieId","title","genres","avg_rating"]], on="movieId", how="left")
                    for i, row in merged.iterrows():
                        st.markdown(f"""<div class="movie-card">
                            <strong>{i+1}. {row.get('title','Unknown')}</strong>
                            &nbsp;<span class="score-badge">{row['similarity_score']:.3f}</span><br>
                            <small>🎭 {row.get('genres','N/A')} &nbsp;|&nbsp; ⭐ {row.get('avg_rating',0):.2f}</small>
                        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HYBRID ENGINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔀 Hybrid Engine":
    st.markdown('<p class="main-header">🔀 Hybrid Recommendation Engine</p>', unsafe_allow_html=True)
    st.markdown("Combines **Content-Based + Collaborative + SVD Matrix Factorization** into one unified score.")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_id_h = st.selectbox("Select User ID:", all_users, key="hybrid_uid")
    with col2:
        k_h = st.slider("Recommendations:", 5, 20, 10, key="k_hybrid")

    st.subheader("⚖️ Adjust Hybrid Weights")
    w_col1, w_col2, w_col3 = st.columns(3)
    with w_col1:
        w_content = st.slider("Content Weight", 0.0, 1.0, 0.2, 0.05)
    with w_col2:
        w_collab = st.slider("Collaborative Weight", 0.0, 1.0, 0.3, 0.05)
    with w_col3:
        w_mf = st.slider("Matrix Factorization Weight", 0.0, 1.0, 0.5, 0.05)

    total_w = w_content + w_collab + w_mf
    if abs(total_w - 1.0) > 0.01:
        st.warning(f"⚠️ Weights sum to {total_w:.2f}. They will be auto-normalized to 1.0.")
        w_content /= total_w
        w_collab /= total_w
        w_mf /= total_w

    if st.button("🔀 Get Hybrid Recommendations", type="primary", use_container_width=True):
        from src.hybrid.hybrid_recommender import HybridWeights
        custom_weights = HybridWeights(content=round(w_content,4),
                                        collaborative=round(w_collab,4),
                                        matrix_factorization=round(w_mf,4))
        hybrid.weights = custom_weights

        with st.spinner("Computing hybrid scores..."):
            recs = hybrid.recommend(user_id_h, k=k_h, pool=300)

        st.subheader(f"🎯 Top {k_h} Hybrid Recommendations for User {user_id_h}")

        # Score breakdown chart
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Content", x=recs["title"].str[:30], y=recs["content_score"],
                              marker_color="#667eea"))
        fig.add_trace(go.Bar(name="Collaborative", x=recs["title"].str[:30], y=recs["collaborative_score"],
                              marker_color="#764ba2"))
        fig.add_trace(go.Bar(name="Matrix Factorization", x=recs["title"].str[:30], y=recs["mf_score"],
                              marker_color="#f093fb"))
        fig.update_layout(barmode="group", title="Score Breakdown per Recommendation",
                          xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Recommendation cards
        for i, row in recs.iterrows():
            col_a, col_b, col_c, col_d, col_e = st.columns([3, 1, 1, 1, 1])
            with col_a:
                st.markdown(f"""<div class="movie-card">
                    <strong>{i+1}. {row['title']}</strong><br>
                    <small>🎭 {row['genres']}</small>
                </div>""", unsafe_allow_html=True)
            with col_b:
                st.metric("Content", f"{row['content_score']:.2f}")
            with col_c:
                st.metric("Collab", f"{row['collaborative_score']:.2f}")
            with col_d:
                st.metric("MF", f"{row['mf_score']:.2f}")
            with col_e:
                st.metric("🏆 Hybrid", f"{row['hybrid_score']:.2f}")

        # Explanation
        top_movie = recs.iloc[0]["title"] if len(recs) > 0 else "N/A"
        top_driver = max(
            [("Content-Based", recs.iloc[0]["content_score"] * w_content),
             ("Collaborative Filtering", recs.iloc[0]["collaborative_score"] * w_collab),
             ("Matrix Factorization", recs.iloc[0]["mf_score"] * w_mf)],
            key=lambda x: x[1]
        )[0] if len(recs) > 0 else "N/A"
        st.markdown(f"""<div class="explanation-box">
            <strong>💡 Recommendation Explanation</strong><br>
            Top pick: <strong>{top_movie}</strong><br>
            Primary signal: <strong>{top_driver}</strong>
            (weights: Content={w_content:.0%}, Collaborative={w_collab:.0%}, MF={w_mf:.0%})<br>
            Each signal is min-max normalized to [0,1] before combining, ensuring weights
            reflect genuine importance rather than raw score magnitude.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DATA ANALYSIS (EDA)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Analysis":
    st.markdown('<p class="main-header">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🎭 Genres", "⭐ Ratings", "👤 Users", "📉 Sparsity"])

    with tab1:
        st.subheader("Genre Distribution")
        exploded = movies.assign(genres=movies["genres"].str.split("|")).explode("genres")
        exploded = exploded[exploded["genres"] != "Unknown"]
        genre_counts = exploded["genres"].value_counts().head(15)

        fig = px.bar(x=genre_counts.values, y=genre_counts.index,
                     orientation="h", title="Movie Count by Genre",
                     color=genre_counts.values, color_continuous_scale="Viridis",
                     labels={"x": "Count", "y": "Genre"})
        fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        # Genre vs avg rating
        genre_ratings = (exploded.merge(ratings, on="movieId")
                         .groupby("genres")["rating"].agg(["mean","count"])
                         .reset_index())
        genre_ratings.columns = ["Genre", "Avg Rating", "Count"]
        genre_ratings = genre_ratings[genre_ratings["Count"] > 50].sort_values("Avg Rating", ascending=False)
        fig2 = px.bar(genre_ratings.head(12), x="Genre", y="Avg Rating",
                      title="Average Rating by Genre (min 50 ratings)",
                      color="Avg Rating", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Rating Distribution")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(ratings, x="rating", nbins=9,
                               title="Rating Distribution (Histogram)",
                               color_discrete_sequence=["#667eea"])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.box(ratings, y="rating", title="Rating Distribution (Boxplot)",
                          color_discrete_sequence=["#764ba2"])
            st.plotly_chart(fig2, use_container_width=True)

        # Rating over time
        ratings_time = ratings.copy()
        ratings_time["date"] = pd.to_datetime(ratings_time["timestamp"], unit="s")
        ratings_time["year"] = ratings_time["date"].dt.year
        yearly = ratings_time.groupby("year").agg(count=("rating","size"), avg=("rating","mean")).reset_index()
        fig3 = px.line(yearly, x="year", y="count", title="Number of Ratings Per Year",
                       markers=True, color_discrete_sequence=["#667eea"])
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.subheader("User Activity Analysis")
        per_user = ratings.groupby("userId").size().reset_index(name="n_ratings")
        fig = px.histogram(per_user, x="n_ratings", nbins=50,
                           title="Distribution of Ratings per User",
                           color_discrete_sequence=["#C44E52"])
        fig.update_xaxes(range=[0, per_user["n_ratings"].quantile(0.95)])
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Median Ratings/User", int(per_user["n_ratings"].median()))
        col2.metric("Top 10% Users (ratings)", int(per_user["n_ratings"].quantile(0.9)))
        col3.metric("Users with <5 ratings",
                    f"{(per_user['n_ratings'] < 5).mean()*100:.1f}%")

        # Scatter: user activity vs avg rating given
        uf_plot = user_features[user_features["n_ratings"] > 0].sample(min(500, len(user_features)))
        fig2 = px.scatter(uf_plot, x="n_ratings", y="avg_rating",
                          title="User Activity vs Average Rating Given",
                          opacity=0.5, color_discrete_sequence=["#667eea"])
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.subheader("Matrix Sparsity & Long Tail Analysis")
        n_u = ratings["userId"].nunique()
        n_m = movies["movieId"].nunique()
        filled = len(ratings)
        possible = n_u * n_m
        density = filled / possible * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Matrix Density", f"{density:.3f}%")
        col2.metric("Cells Filled", f"{filled:,}")
        col3.metric("Cells Possible", f"{possible:,}")

        # Long tail
        counts = ratings.groupby("movieId").size().sort_values(ascending=False).reset_index()
        counts.columns = ["movieId","n_ratings"]
        counts["rank"] = range(1, len(counts)+1)
        counts["cumulative_share"] = counts["n_ratings"].cumsum() / counts["n_ratings"].sum()

        fig = px.area(counts, x="rank", y="cumulative_share",
                      title="Long Tail: Cumulative Share of Ratings",
                      labels={"rank":"Movies Ranked by Popularity","cumulative_share":"Cumulative Share"},
                      color_discrete_sequence=["#667eea"])
        top10_idx = int(len(counts) * 0.10)
        fig.add_vline(x=top10_idx, line_dash="dash", line_color="red",
                      annotation_text=f"Top 10% movies = {counts['cumulative_share'].iloc[top10_idx-1]*100:.1f}% of ratings")
        st.plotly_chart(fig, use_container_width=True)

        # Popularity histogram
        fig2 = px.histogram(counts, x="n_ratings", nbins=50,
                            title="Movie Popularity Distribution (log scale)",
                            color_discrete_sequence=["#764ba2"])
        fig2.update_xaxes(type="log")
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Evaluation":
    st.markdown('<p class="main-header">📈 Model Evaluation</p>', unsafe_allow_html=True)
    st.info("Offline evaluation using held-out ratings. Select a user to compute metrics on their hidden preferences.")

    col1, col2 = st.columns([2, 1])
    with col1:
        eval_user = st.selectbox("Select User for Evaluation:", all_users, key="eval_uid")
    with col2:
        k_eval = st.slider("K (top-K metrics):", 5, 20, 10)

    if st.button("📈 Compute Evaluation Metrics", type="primary", use_container_width=True):
        from src.evaluation.metrics import precision_at_k, recall_at_k, average_precision, ndcg_at_k

        user_ratings = ratings[ratings["userId"] == eval_user].sort_values("rating", ascending=False)

        if len(user_ratings) < 10:
            st.warning("This user has fewer than 10 ratings — metrics may not be meaningful.")
        else:
            split = int(len(user_ratings) * 0.8)
            train_set = set(user_ratings.iloc[:split]["movieId"])
            test_set = set(user_ratings.iloc[split:]["movieId"])

            with st.spinner("Getting recommendations from each model..."):
                # Content
                liked_title = movies[movies["movieId"].isin(train_set)].nlargest(1, "avg_rating")["title"].values
                content_recs = []
                if len(liked_title) > 0 and liked_title[0] in content_rec._title_to_idx.index:
                    cr = content_rec.get_similar_movies(liked_title[0], k=k_eval*2)
                    content_recs = cr["movieId"].tolist()[:k_eval]

                # Collaborative
                collab_df = user_cf.recommend(eval_user, k=k_eval*2)
                collab_recs = collab_df["movieId"].tolist()[:k_eval]

                # MF/SVD
                cand = movies[~movies["movieId"].isin(train_set)]["movieId"].tolist()
                mf_df = mf_model.recommend(eval_user, cand, k=k_eval)
                mf_recs = mf_df["movieId"].tolist()

                # Hybrid
                hybrid_df = hybrid.recommend(eval_user, k=k_eval, pool=300)
                hybrid_recs = hybrid_df["movieId"].tolist()

            models_recs = {
                "Content-Based": content_recs,
                "User CF": collab_recs,
                "SVD (MF)": mf_recs,
                "Hybrid": hybrid_recs,
            }
            rel = test_set

            rows = []
            for model_name, recs_list in models_recs.items():
                rows.append({
                    "Model": model_name,
                    f"Precision@{k_eval}": round(precision_at_k(recs_list, rel, k_eval), 4),
                    f"Recall@{k_eval}": round(recall_at_k(recs_list, rel, k_eval), 4),
                    "MAP": round(average_precision(recs_list, rel), 4),
                    f"NDCG@{k_eval}": round(ndcg_at_k(recs_list, rel, k_eval), 4),
                })
            metrics_df = pd.DataFrame(rows).set_index("Model")
            st.subheader("📊 Evaluation Results")
            st.dataframe(metrics_df.style.highlight_max(axis=0, color="#d4f1c0"), use_container_width=True)

            # Radar chart
            metric_cols = metrics_df.columns.tolist()
            fig = go.Figure()
            for model_name in metrics_df.index:
                vals = metrics_df.loc[model_name].tolist()
                fig.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=metric_cols + [metric_cols[0]],
                    fill="toself", name=model_name
                ))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])),
                              title="Model Comparison — Radar Chart")
            st.plotly_chart(fig, use_container_width=True)

            # Bar chart comparison
            metrics_melted = metrics_df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
            fig2 = px.bar(metrics_melted, x="Metric", y="Score", color="Model",
                          barmode="group", title="Metric Comparison Across Models",
                          color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("""<div class="explanation-box">
                <strong>📖 Metric Explanations</strong><br>
                <b>Precision@K:</b> Of the top-K recommendations, what fraction were actually relevant?<br>
                <b>Recall@K:</b> Of all relevant items, what fraction appeared in the top-K?<br>
                <b>MAP:</b> Mean Average Precision — accounts for <em>where</em> relevant items appear in the ranking.<br>
                <b>NDCG@K:</b> Normalized Discounted Cumulative Gain — rewards hitting relevant items higher in the list.
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)

    st.markdown("""
    ## 🎬 Hybrid Movie Recommendation System

    A **portfolio-grade, production-structured** recommendation system built on the
    [MovieLens dataset](https://grouplens.org/datasets/movielens/) demonstrating
    three complementary recommendation approaches combined into a hybrid engine.

    ---

    ### 🧠 Algorithms Implemented

    | Algorithm | File | Key Technique |
    |---|---|---|
    | Content-Based Filtering | `src/content_based/recommender.py` | TF-IDF + Cosine Similarity |
    | Item-Based CF | `src/collaborative/recommender.py` | KNN on user-item matrix |
    | User-Based CF | `src/collaborative/recommender.py` | KNN on transposed matrix |
    | Matrix Factorization | `src/collaborative/matrix_factorization.py` | SVD (scikit-surprise) |
    | Hybrid Engine | `src/hybrid/hybrid_recommender.py` | Weighted combination |

    ---

    ### 📁 Project Structure
movie-recommendation-system/
├── data/raw/              ← MovieLens CSVs (gitignored)
├── data/processed/        ← Cached trained models
├── src/
│   ├── data/              ← Download + loader
│   ├── preprocessing/     ← Cleaning + feature engineering
│   ├── content_based/     ← TF-IDF recommender
│   ├── collaborative/     ← KNN + SVD
│   ├── hybrid/            ← Weighted hybrid engine
│   ├── evaluation/        ← Precision@K, NDCG, MAP
│   └── utils/             ← Model cache
├── app/main.py            ← This Streamlit dashboard
├── tests/                 ← pytest test suite
├── Dockerfile
└── .github/workflows/     ← CI/CD
---

    ### 📊 Dataset
    - **MovieLens Small:** ~100K ratings, 9,742 movies, 610 users
    - **Source:** GroupLens Research, University of Minnesota
    - **Features used:** ratings, genres, user-applied tags

    ---

    ### 🛠️ Tech Stack
    `Python 3.11` · `pandas` · `scikit-learn` · `scikit-surprise` · `scipy`
    · `Streamlit` · `Plotly` · `Docker` · `GitHub Actions`
    """)

