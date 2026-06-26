from typing import Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_content_soup(movies: pd.DataFrame, tags: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    out = movies.copy()
    genre_text = out["genres"].str.replace("|", " ", regex=False)
    if tags is not None and len(tags) > 0:
        tag_text = (
            tags.groupby("movieId")["tag"]
            .apply(lambda s: " ".join(s.astype(str)))
            .rename("tag_text")
        )
        out = out.merge(tag_text, on="movieId", how="left")
        out["tag_text"] = out["tag_text"].fillna("")
    else:
        out["tag_text"] = ""
    out["content_soup"] = (genre_text + " ") * 2 + out["tag_text"]
    return out


class ContentBasedRecommender:
    def __init__(self, movies_with_soup: pd.DataFrame, max_features: int = 5000):
        self.movies = movies_with_soup.reset_index(drop=True)
        self._title_to_idx = pd.Series(self.movies.index, index=self.movies["title"])
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies["content_soup"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def get_similar_movies(self, title: str, k: int = 10) -> pd.DataFrame:
        if title not in self._title_to_idx.index:
            raise ValueError(f"Movie '{title}' not found.")
        idx = self._title_to_idx[title]
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:k + 1]
        result = self.movies.iloc[[i for i, _ in scores]][["movieId", "title", "genres"]].copy()
        result["similarity_score"] = [s for _, s in scores]
        return result.reset_index(drop=True)

    def get_score(self, title_a: str, title_b: str) -> float:
        return float(self.similarity_matrix[
            self._title_to_idx[title_a],
            self._title_to_idx[title_b]
        ])
