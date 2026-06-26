from typing import Tuple
import pandas as pd


def handle_missing_values(movies: pd.DataFrame) -> pd.DataFrame:
    out = movies.copy()
    out["genres"] = out["genres"].replace("(no genres listed)", "Unknown")
    out["genres"] = out["genres"].fillna("Unknown")
    out["title"] = out["title"].fillna("Unknown Title")
    return out


def remove_duplicate_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    before = len(ratings)
    out = ratings.drop_duplicates(subset=["userId", "movieId"], keep="last").copy()
    if before - len(out):
        print(f"Removed {before - len(out)} duplicate rows.")
    return out


def validate_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    out = ratings.copy()
    out = out.dropna(subset=["userId", "movieId", "rating"])
    out = out[(out["rating"] >= 0.5) & (out["rating"] <= 5.0)]
    return out


def engineer_movie_features(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    agg = ratings.groupby("movieId").agg(
        n_ratings=("rating", "size"),
        avg_rating=("rating", "mean")
    )
    out = movies.merge(agg, on="movieId", how="left")
    out["n_ratings"] = out["n_ratings"].fillna(0).astype(int)
    out["genre_list"] = out["genres"].str.split("|")
    out["n_genres"] = out["genre_list"].apply(len)
    genre_dummies = out["genre_list"].explode().str.get_dummies().groupby(level=0).sum()
    genre_dummies.columns = [f"genre_{c}" for c in genre_dummies.columns]
    out = pd.concat([out.reset_index(drop=True), genre_dummies.reset_index(drop=True)], axis=1)
    return out


def engineer_user_features(ratings: pd.DataFrame) -> pd.DataFrame:
    out = ratings.groupby("userId").agg(
        n_ratings=("rating", "size"),
        avg_rating=("rating", "mean"),
        rating_std=("rating", "std")
    ).reset_index()
    out["rating_std"] = out["rating_std"].fillna(0.0)
    return out


def run_full_cleaning(movies, ratings) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    movies_clean = handle_missing_values(movies)
    ratings_clean = validate_ratings(remove_duplicate_ratings(ratings))
    movies_clean = engineer_movie_features(movies_clean, ratings_clean)
    user_features = engineer_user_features(ratings_clean)
    return movies_clean, ratings_clean, user_features
