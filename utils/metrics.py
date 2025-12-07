import numpy as np
from typing import List

def weighted_interval_score(y_true: np.ndarray, y_preds: np.ndarray, quantiles: List[float]) -> float:
    """
    Вычисляет WIS (Weighted Interval Score).
    y_true: [N]
    y_preds: [N, n_quantiles] (sorted)
    quantiles: list of floats
    """

    score = 0.0
    for i, q in enumerate(quantiles):
        # Pinball loss formula
        errors = y_true - y_preds[:, i]
        score += np.maximum((q - 1) * errors, q * errors).mean()
    
    return score / len(quantiles)