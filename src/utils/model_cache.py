"""
Trains and caches all models to disk so the Streamlit app doesn't retrain
on every page load (which would take 60+ seconds every refresh).
"""
import pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_all, extract_year
from src.preprocessing.clean import run_full_cleaning
from src.content_based.recommender import build_content_soup, ContentBasedRecommender
from src.collaborative.recommender import ItemBasedCF, UserBasedCF
from src.collaborative.matrix_factorization import SurpriseSVD
from src.hybrid.hybrid_recommender import HybridRecommender, HybridWeights

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CACHE_FILE = CACHE_DIR / "models.pkl"


def train_and_save():
    print("Loading data...")
    data = load_all()
    movies = extract_year(data.movies)
    movies_clean, ratings_clean, user_features = run_full_cleaning(movies, data.ratings)

    print("Building content-based model...")
    soup = build_content_soup(movies_clean, data.tags)
    content_rec = ContentBasedRecommender(soup)

    print("Building item-based CF...")
    item_cf = ItemBasedCF(ratings_clean, k_neighbors=20)

    print("Building user-based CF...")
    user_cf = UserBasedCF(ratings_clean, k_neighbors=20)

    print("Training SVD (matrix factorization)...")
    mf_model = SurpriseSVD(n_factors=50, n_epochs=20).fit(ratings_clean)

    print("Building hybrid recommender...")
    hybrid = HybridRecommender(
        content_rec, item_cf, mf_model, movies_clean, ratings_clean,
        weights=HybridWeights(content=0.2, collaborative=0.3, matrix_factorization=0.5)
    )

    cache = {
        "movies_clean": movies_clean,
        "ratings_clean": ratings_clean,
        "user_features": user_features,
        "content_rec": content_rec,
        "item_cf": item_cf,
        "user_cf": user_cf,
        "mf_model": mf_model,
        "hybrid": hybrid,
        "soup": soup,
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    print(f"Models saved to {CACHE_FILE}")
    return cache


def load_cache():
    if not CACHE_FILE.exists():
        print("No cache found — training now...")
        return train_and_save()
    print("Loading cached models...")
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    train_and_save()
