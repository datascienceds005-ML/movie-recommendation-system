import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def build_user_item_matrix(ratings: pd.DataFrame):
    user_ids = np.sort(ratings["userId"].unique())
    movie_ids = np.sort(ratings["movieId"].unique())
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    movie_to_idx = {m: i for i, m in enumerate(movie_ids)}
    matrix = csr_matrix(
        (ratings["rating"].values,
         (ratings["userId"].map(user_to_idx), ratings["movieId"].map(movie_to_idx))),
        shape=(len(user_ids), len(movie_ids))
    )
    return matrix, user_ids, movie_ids


class ItemBasedCF:
    def __init__(self, ratings: pd.DataFrame, k_neighbors: int = 20):
        self.matrix, self.user_ids, self.movie_ids = build_user_item_matrix(ratings)
        self._movie_to_row = {m: i for i, m in enumerate(self.movie_ids)}
        self.item_matrix = self.matrix.T.tocsr()
        self.model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.model.fit(self.item_matrix)

    def get_similar_movies(self, movie_id: int, k: int = 10) -> pd.DataFrame:
        if movie_id not in self._movie_to_row:
            return pd.DataFrame(columns=["movieId", "similarity_score"])
        row = self._movie_to_row[movie_id]
        distances, indices = self.model.kneighbors(
            self.item_matrix[row], n_neighbors=min(k + 1, self.item_matrix.shape[0])
        )
        return pd.DataFrame({
            "movieId": self.movie_ids[indices[0][1:]],
            "similarity_score": 1 - distances[0][1:]
        })


class UserBasedCF:
    def __init__(self, ratings: pd.DataFrame, k_neighbors: int = 20):
        self.matrix, self.user_ids, self.movie_ids = build_user_item_matrix(ratings)
        self._user_to_row = {u: i for i, u in enumerate(self.user_ids)}
        self.model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.model.fit(self.matrix)
        self.k_neighbors = k_neighbors

    def recommend(self, user_id: int, k: int = 10) -> pd.DataFrame:
        if user_id not in self._user_to_row:
            return pd.DataFrame(columns=["movieId", "predicted_score"])
        row = self._user_to_row[user_id]
        distances, indices = self.model.kneighbors(
            self.matrix[row], n_neighbors=min(self.k_neighbors + 1, self.matrix.shape[0])
        )
        neighbor_rows = indices[0][1:]
        neighbor_sims = 1 - distances[0][1:]
        neighbor_ratings = self.matrix[neighbor_rows].toarray()
        weights = neighbor_sims.reshape(-1, 1)
        weight_sum = weights.sum() or 1e-8
        predicted = (neighbor_ratings * weights).sum(axis=0) / weight_sum
        already = self.matrix[row].toarray().flatten() > 0
        predicted[already] = -1
        top = np.argsort(predicted)[::-1][:k]
        result = pd.DataFrame({"movieId": self.movie_ids[top], "predicted_score": predicted[top]})
        return result[result["predicted_score"] > 0].reset_index(drop=True)
