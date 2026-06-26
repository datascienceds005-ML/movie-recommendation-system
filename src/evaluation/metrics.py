import numpy as np
import pandas as pd
from typing import List


def precision_at_k(recommended: List, relevant: set, k: int) -> float:
    rec_k = recommended[:k]
    return len([r for r in rec_k if r in relevant]) / k if k > 0 else 0.0


def recall_at_k(recommended: List, relevant: set, k: int) -> float:
    rec_k = recommended[:k]
    return len([r for r in rec_k if r in relevant]) / len(relevant) if relevant else 0.0


def average_precision(recommended: List, relevant: set) -> float:
    hits, sum_prec = 0, 0.0
    for i, r in enumerate(recommended):
        if r in relevant:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / min(len(relevant), len(recommended)) if relevant else 0.0


def ndcg_at_k(recommended: List, relevant: set, k: int) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, r in enumerate(recommended[:k]) if r in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def coverage(all_recommendations: List[List], catalog_size: int) -> float:
    recommended_items = set(item for recs in all_recommendations for item in recs)
    return len(recommended_items) / catalog_size if catalog_size > 0 else 0.0


def diversity(recommendations: List, similarity_matrix, title_to_idx: dict) -> float:
    valid = [r for r in recommendations if r in title_to_idx]
    if len(valid) < 2:
        return 0.0
    pairs, total_dist = 0, 0.0
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            sim = similarity_matrix[title_to_idx[valid[i]], title_to_idx[valid[j]]]
            total_dist += (1 - sim)
            pairs += 1
    return total_dist / pairs if pairs > 0 else 0.0
