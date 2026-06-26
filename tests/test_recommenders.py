"""Basic smoke tests for all three recommendation engines."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import pandas as pd
import numpy as np

from src.content_based.recommender import build_content_soup, ContentBasedRecommender
from src.collaborative.recommender import ItemBasedCF, UserBasedCF
from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_movies():
    return pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5],
        "title": ["Action Hero (2000)", "Comedy King (2001)", "Drama Queen (2002)",
                  "Sci-Fi Space (2003)", "Horror Night (2004)"],
        "genres": ["Action|Adventure", "Comedy|Romance", "Drama",
                   "Sci-Fi|Action", "Horror|Thriller"],
        "n_ratings": [100, 80, 60, 40, 20],
        "avg_rating": [4.0, 3.5, 4.2, 3.8, 3.1],
    })


@pytest.fixture
def sample_ratings():
    rows = []
    for user_id in range(1, 11):
        for movie_id in [1, 2, 3, 4, 5]:
            if np.random.random() > 0.3:
                rows.append((user_id, movie_id, np.random.choice([3.0, 3.5, 4.0, 4.5, 5.0])))
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating"])


# ── Content-Based Tests ───────────────────────────────────────────────────────
class TestContentBased:
    def test_build_soup_has_content_column(self, sample_movies):
        soup = build_content_soup(sample_movies)
        assert "content_soup" in soup.columns
        assert soup["content_soup"].str.len().min() > 0

    def test_recommender_returns_correct_count(self, sample_movies):
        soup = build_content_soup(sample_movies)
        rec = ContentBasedRecommender(soup)
        results = rec.get_similar_movies("Action Hero (2000)", k=3)
        assert len(results) == 3

    def test_recommender_excludes_query_movie(self, sample_movies):
        soup = build_content_soup(sample_movies)
        rec = ContentBasedRecommender(soup)
        results = rec.get_similar_movies("Action Hero (2000)", k=4)
        assert "Action Hero (2000)" not in results["title"].values

    def test_similarity_scores_between_0_and_1(self, sample_movies):
        soup = build_content_soup(sample_movies)
        rec = ContentBasedRecommender(soup)
        results = rec.get_similar_movies("Drama Queen (2002)", k=3)
        assert (results["similarity_score"] >= 0).all()
        assert (results["similarity_score"] <= 1).all()

    def test_unknown_title_raises_valueerror(self, sample_movies):
        soup = build_content_soup(sample_movies)
        rec = ContentBasedRecommender(soup)
        with pytest.raises(ValueError):
            rec.get_similar_movies("Nonexistent Movie (9999)")


# ── Collaborative Filtering Tests ─────────────────────────────────────────────
class TestCollaborativeFiltering:
    def test_item_cf_returns_dataframe(self, sample_ratings):
        cf = ItemBasedCF(sample_ratings, k_neighbors=3)
        result = cf.get_similar_movies(1, k=2)
        assert isinstance(result, pd.DataFrame)

    def test_item_cf_similar_movies_not_include_query(self, sample_ratings):
        cf = ItemBasedCF(sample_ratings, k_neighbors=3)
        result = cf.get_similar_movies(1, k=4)
        assert 1 not in result["movieId"].values

    def test_user_cf_recommend_returns_dataframe(self, sample_ratings):
        cf = UserBasedCF(sample_ratings, k_neighbors=3)
        result = cf.recommend(1, k=3)
        assert isinstance(result, pd.DataFrame)

    def test_user_cf_cold_start_returns_empty(self, sample_ratings):
        cf = UserBasedCF(sample_ratings, k_neighbors=3)
        result = cf.recommend(9999, k=5)
        assert len(result) == 0


# ── Evaluation Metric Tests ───────────────────────────────────────────────────
class TestEvaluationMetrics:
    def test_precision_perfect(self):
        assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_precision_none(self):
        assert precision_at_k([4, 5, 6], {1, 2, 3}, k=3) == 0.0

    def test_precision_partial(self):
        assert abs(precision_at_k([1, 4, 2], {1, 2, 3}, k=3) - 2/3) < 1e-9

    def test_recall_perfect(self):
        assert recall_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_recall_empty_relevant(self):
        assert recall_at_k([1, 2], set(), k=2) == 0.0

    def test_ndcg_perfect(self):
        assert ndcg_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_ndcg_no_hits(self):
        assert ndcg_at_k([4, 5, 6], {1, 2, 3}, k=3) == 0.0
