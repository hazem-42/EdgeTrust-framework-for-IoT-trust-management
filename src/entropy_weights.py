"""
entropy_weights.py -- Eq. (1)-(2): Shannon entropy weighting over the
normalized training partition, with the equal-weight fallback and the
Theorem 1 validity check (w_j >= 0, sum w_j = 1).
"""
import numpy as np
import pandas as pd


def compute_entropy_weights(r_train: pd.DataFrame) -> pd.Series:
    """
    r_train: (n_observations x m_features) DataFrame of ALREADY-normalized
    r_ij values in [0,1] (i.e., the output of data_pipeline.apply_normalization
    on the training partition only).

    Returns a pandas Series of weights w_j indexed by feature name, matching
    Eq. (2) exactly, including the degenerate all-uniform fallback.
    """
    n, m = r_train.shape
    R = r_train.to_numpy()
    col_sums = R.sum(axis=0)

    p = np.divide(R, col_sums, out=np.full_like(R, 1.0 / n), where=col_sums > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > 0, p * np.log(p), 0.0)
    e_j = -1.0 / np.log(n) * plogp.sum(axis=0)
    e_j = np.clip(e_j, 0.0, 1.0)

    one_minus_e = 1.0 - e_j
    D = one_minus_e.sum()
    if D > 0:
        w = one_minus_e / D
    else:
        w = np.full(m, 1.0 / m)

    return pd.Series(w, index=r_train.columns, name="entropy_weight")


def check_theorem1(w: pd.Series, atol: float = 1e-9) -> dict:
    """Theorem 1 (Entropy Weight Validity): w_j >= 0 and sum_j w_j = 1."""
    nonneg = bool((w >= -atol).all())
    sums_to_one = bool(abs(w.sum() - 1.0) < 1e-6)
    return {
        "nonnegative": nonneg,
        "sums_to_one": sums_to_one,
        "sum_value": float(w.sum()),
        "holds": nonneg and sums_to_one,
    }
