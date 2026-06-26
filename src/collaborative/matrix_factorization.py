import numpy as np
import pandas as pd


class SurpriseSVD:
    def __init__(self, n_factors: int = 50, n_epochs: int = 20, random_state: int = 42):
        from surprise import SVD
        self.model = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=random_state)

    def fit(self, ratings: pd.DataFrame):
        from surprise import Dataset, Reader
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        return self.model.predict(uid=user_id, iid=movie_id).est

    def recommend(self, user_id: int, candidate_ids, k: int = 10) -> pd.DataFrame:
        preds = [(mid, self.predict(user_id, mid)) for mid in candidate_ids]
        preds.sort(key=lambda x: x[1], reverse=True)
        return pd.DataFrame(preds[:k], columns=["movieId", "predicted_rating"])
