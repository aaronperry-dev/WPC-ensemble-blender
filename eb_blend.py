"""Multi-system ensemble statistics.

Weighting
---------
Each system gets a total weight W_s (config).  Every member of system s then
carries W_s / n_s so a 51-member ensemble does not outvote a 21-member one just
by size, and the weights are renormalised over the systems actually present.

Statistics (all weighted, all point-wise unless noted)
-------------------------------------------------------
* mean                         - weighted ensemble mean
* probability matched mean     - Ebert (2001, MWR 129, 2461-2480): the mean's
                                 spatial *pattern* with the pooled member
                                 *amplitude distribution*, which avoids the
                                 smoothing / amplitude loss of a plain mean
* exceedance probability       - weighted fraction of members >= threshold
* weighted percentiles
"""
from __future__ import annotations

import numpy as np


def combine(items: list[tuple[float, np.ndarray]]):
    """items = [(system_weight, (n, ny, nx) array), ...] -> (M, ny, nx) array, (M,) weights summing to 1."""
    arrs, ws = [], []
    for w, a in items:
        if a is None or a.shape[0] == 0 or w <= 0:
            continue
        arrs.append(a)
        ws.append(np.full(a.shape[0], w / a.shape[0], dtype=np.float64))
    if not arrs:
        return None, None
    w = np.concatenate(ws)
    return np.concatenate(arrs, axis=0).astype(np.float32), w / w.sum()


def wmean(arr: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.tensordot(w.astype(np.float32), arr, axes=(0, 0))


def exceed_prob(arr: np.ndarray, w: np.ndarray, thr: float) -> np.ndarray:
    return np.tensordot(w.astype(np.float32), (arr >= thr).astype(np.float32), axes=(0, 0))


def wpercentile(arr: np.ndarray, w: np.ndarray, q: float) -> np.ndarray:
    """Weighted percentile (0-100) along the member axis, linear between cumulative-weight midpoints."""
    m = arr.shape[0]
    flat = arr.reshape(m, -1)
    order = np.argsort(flat, axis=0)
    sv = np.take_along_axis(flat, order, axis=0)
    sw = w[order]
    cw = np.cumsum(sw, axis=0) - 0.5 * sw  # mid-point plotting positions
    target = q / 100.0
    out = np.empty(flat.shape[1], dtype=np.float32)
    # vectorised interpolation: find first index where cw >= target
    idx = (cw < target).sum(axis=0)
    lo = np.clip(idx - 1, 0, m - 1)
    hi = np.clip(idx, 0, m - 1)
    cols = np.arange(flat.shape[1])
    c0, c1 = cw[lo, cols], cw[hi, cols]
    v0, v1 = sv[lo, cols], sv[hi, cols]
    t = np.where(c1 > c0, (target - c0) / np.where(c1 > c0, c1 - c0, 1), 0.0)
    out[:] = v0 + np.clip(t, 0, 1) * (v1 - v0)
    return out.reshape(arr.shape[1:])


def prob_matched_mean(arr: np.ndarray, w: np.ndarray, dry: float = 0.0) -> np.ndarray:
    """Weighted probability-matched mean (Ebert 2001) over the whole domain.

    1. rank every grid point by the weighted ensemble mean;
    2. pool all member values (each member's grid values carry that member's weight);
    3. give the point of rank r the pooled value at the matching exceedance quantile.
    """
    m, ny, nx = arr.shape
    n = ny * nx
    mean = wmean(arr, w).ravel()
    vals = arr.reshape(m, n).ravel()
    wt = np.repeat(w, n)
    order = np.argsort(vals, kind="stable")
    sv = vals[order]
    cw = np.cumsum(wt[order]) / wt.sum()  # cumulative probability mass in (0, 1]
    rank = np.empty(n, dtype=np.int64)
    rank[np.argsort(mean, kind="stable")] = np.arange(n)
    q = (rank + 0.5) / n
    out = np.interp(q, cw, sv)
    out[mean <= dry] = 0.0  # every member dry here -> stays dry
    return out.reshape(ny, nx).astype(np.float32)
