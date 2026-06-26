from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class HybridWeights:
    content: float = 0.2
    collaborative: float = 0.3
    matrix_factorization: float = 0.5

    def __post_init__(self):
        total = self.content + self.collaborative + self.matrix_factorization
        if not np.isclose(total, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total}")


class HybridRecommender:
    def __init__(self, content_rec, item_cf, mf_model, movies, ratings, weights=None):
        self.content_rec = content_rec
        self.item_cf = item_cf
        self.mf_model = mf_model
        self.movies = movies
        self.ratings = ratings
        self.weights = weights or HybridWeights()
        self._title_by_id = movies.set_index("movieId")["title"]

    def _scale(self, x):
        x = x.reshape(-1, 1)
        if x.max() == x.min():
            return np.zeros(len(x))
        return MinMaxScaler().fit_transform(x).flatten()

    def _content_scores(self, user_id, candidate_ids):
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        if len(user_ratings) == 0:
            return np.zeros(len(candidate_ids))
        top_liked = user_ratings.nlargest(5, "rating")
        scores = np.zeros(len(candidate_ids))
        for _, row in top_liked.iterrows():
            liked_title = self._title_by_id.get(row["movieId"])
            if liked_title is None or liked_title not in self.content_rec._title_to_idx.index:
                continue
            liked_idx = self.content_rec._title_to_idx[liked_title]
            for j, cid in enumerate(candidate_ids):
                cand_title = self._title_by_id.get(cid)
                if cand_title is None or cand_title not in self.content_rec._title_to_idx.index:
                    continue
                cand_idx = self.content_rec._title_to_idx[cand_title]
                scores[j] += self.content_rec.similarity_matrix[liked_idx, cand_idx] * (row["rating"] / 5.0)
        return scores / max(len(top_liked), 1)

    def _collab_scores(self, user_id, candidate_ids):
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        if len(user_ratings) == 0:
            return np.zeros(len(candidate_ids))
        scores = np.zeros(len(candidate_ids))
        rated_ids = user_ratings["movieId"].values
        rated_vals = user_ratings["rating"].values
        for j, cid in enumerate(candidate_ids):
            if cid not in self.item_cf._movie_to_row:
                continue
            ci = self.item_cf._movie_to_row[cid]
            tw, ws = 0.0, 0.0
            for rid, rval in zip(rated_ids, rated_vals):
                if rid not in self.item_cf._movie_to_row:
                    continue
                ri = self.item_cf._movie_to_row[rid]
                sim = float(self.item_cf.item_matrix[ci].multiply(self.item_cf.item_matrix[ri]).sum())
                if sim:
                    ws += sim * (rval / 5.0)
                    tw += abs(sim)
            scores[j] = ws / tw if tw > 0 else 0.0
        return scores

    def _mf_scores(self, user_id, candidate_ids):
        preds = np.array([self.mf_model.predict(user_id, mid) for mid in candidate_ids])
        return (preds - 0.5) / 4.5

    def recommend(self, user_id: int, k: int = 10, pool: int = 200) -> pd.DataFrame:
        already = set(self.ratings.loc[self.ratings["userId"] == user_id, "movieId"])
        candidates = self.movies[~self.movies["movieId"].isin(already)]
        if len(candidates) > pool:
            candidates = candidates.sample(pool, random_state=42)
        cids = candidates["movieId"].values

        cs = self._scale(self._content_scores(user_id, cids))
        co = self._scale(self._collab_scores(user_id, cids))
        mf = self._scale(self._mf_scores(user_id, cids))

        hybrid = (self.weights.content * cs
                  + self.weights.collaborative * co
                  + self.weights.matrix_factorization * mf)

        result = pd.DataFrame({
            "movieId": cids,
            "title": [self._title_by_id.get(m, "Unknown") for m in cids],
            "genres": [self.movies.loc[self.movies["movieId"] == m, "genres"].values[0]
                       if m in self.movies["movieId"].values else "Unknown" for m in cids],
            "content_score": cs,
            "collaborative_score": co,
            "mf_score": mf,
            "hybrid_score": hybrid,
        })
        return result.nlargest(k, "hybrid_score").reset_index(drop=True)
