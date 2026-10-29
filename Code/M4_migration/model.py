"""M4 core: semi-quantitative destination migration.

AHP + entropy combined feature weighting, TOPSIS policy-priority ranking, and
threshold recalibration. Every input is a declared expert coding, not fitted
microdata; see config.json and the README boundary statements.
"""
import numpy as np

# Saaty random index (mean consistency index) for n = 1..10.
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def ahp(matrix):
    """Return weights, lambda_max, consistency index and consistency ratio."""
    A = np.asarray(matrix, float)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("AHP matrix must be square")
    if not np.allclose(A, 1.0 / A.T, atol=5e-4):
        raise ValueError("AHP matrix must be reciprocal (a_ij = 1/a_ji)")
    col_sum = A.sum(axis=0)
    weights = (A / col_sum).mean(axis=1)
    Aw = A @ weights
    lambda_max = float(np.mean(Aw / weights))
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    cr = ci / RI[n] if n > 2 else 0.0
    return weights, lambda_max, ci, cr


def entropy_weights(decision):
    """Entropy weights from an alternatives x criteria decision matrix.

    Higher discrimination across alternatives yields a larger weight. Entropy is
    direction-neutral; callers pass the criteria they want compared.
    """
    X = np.asarray(decision, float)
    m = X.shape[0]
    if m < 2:
        raise ValueError("Entropy needs at least two alternatives")
    col_sum = X.sum(axis=0)
    if np.any(col_sum <= 0):
        raise ValueError("Each criterion column must have a positive total")
    P = X / col_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(P > 0, -P * np.log(P), 0.0)
    e = terms.sum(axis=0) / np.log(m)
    d = 1.0 - e
    if d.sum() <= 0:
        return np.full(X.shape[1], 1.0 / X.shape[1])
    return d / d.sum()


def combine_weights(ahp_w, entropy_w, theta):
    """Convex combination of AHP and entropy weights; theta in [0,1]."""
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must lie in [0,1]")
    w = theta * np.asarray(ahp_w) + (1 - theta) * np.asarray(entropy_w)
    return w / w.sum()


def topsis(performance, weights, benefit=True):
    """TOPSIS closeness coefficient for an alternatives x criteria matrix.

    benefit=True treats larger values as better; False treats smaller as better.
    Returns closeness C, distance to ideal D+ and distance to anti-ideal D-.
    """
    X = np.asarray(performance, float)
    if benefit is False:
        X = X.max(axis=0) - X
    norm = np.sqrt((X ** 2).sum(axis=0))
    norm[norm == 0] = 1.0
    V = (X / norm) * np.asarray(weights, float)
    ideal = V.max(axis=0)
    anti = V.min(axis=0)
    d_plus = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - anti) ** 2).sum(axis=1))
    denom = d_plus + d_minus
    closeness = np.where(denom > 0, d_minus / denom, 0.0)
    return closeness, d_plus, d_minus


def destination_weights(combined_w, intensity):
    """Effective per-destination criterion weights = structural weight x intensity.

    Policies for a non-acute feature are de-prioritized for that destination.
    """
    w = np.asarray(combined_w, float) * np.asarray(intensity, float)
    if w.sum() <= 0:
        return np.full_like(w, 1.0 / len(w))
    return w / w.sum()


def recalibrate(base, gamma, feature_dev, bounds):
    """theta^V = clip(theta^J [1 + sum gamma_k * dev_k], lower, upper)."""
    factor = 1.0 + float(np.sum(np.asarray(gamma, float) * np.asarray(feature_dev, float)))
    raw = base * factor
    lower, upper = bounds
    value = min(max(raw, lower), upper)
    return value, factor, raw
