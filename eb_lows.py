"""Surface low detection and ensemble clustering.

Detection (per member)
----------------------
A low is a *regional minimum of mean sea-level pressure with dynamic >= h hPa*
(morphological h-minima transform).  That is the digital analogue of
"has at least one closed isobar at h-hPa spacing", the usual operational
definition of a surface low, and it merges shallow neighbouring minima that
are separated by a saddle of less than h.  The field is lightly Gaussian
smoothed first to suppress grid-scale noise; the reported central pressure is
the *unsmoothed* minimum next to the detected point.  Minima within ``border``
grid cells of the domain edge are discarded (they may be a low's flank).

Ensemble summary
----------------
Member lows are grouped with average-linkage agglomerative clustering on
great-circle distance (mean pairwise distance inside a cluster <= ``eps_km``;
complete linkage was tried first and split one broad cloud of member lows
into two adjacent clusters).  Each member is
counted at most once per cluster, weighted by its ensemble weight, so
``support`` is the weighted fraction of members that have a low there.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import label
from skimage.morphology import h_minima

R_EARTH = 6371.0


@dataclass
class Low:
    lat: float
    lon: float
    p_hpa: float


@dataclass
class Cluster:
    lat: float
    lon: float
    p_hpa: float
    support: float  # weighted fraction of members (0-1)
    n_members: int
    pos_sd_km: float
    p_sd_hpa: float
    p_min: float
    p_max: float


def detect_lows(msl_pa: np.ndarray, lat: np.ndarray, lon: np.ndarray, h: float = 2.0,
                sigma: float = 1.0, border: int = 3) -> list[Low]:
    p = msl_pa / 100.0
    ps = gaussian_filter(p, sigma=sigma, mode="nearest")
    # h_minima works on float images; scale to avoid float-tolerance surprises
    mask = h_minima(ps.astype(np.float64), h)
    lab = label(mask, connectivity=2)
    out = []
    ny, nx = p.shape
    for k in range(1, lab.max() + 1):
        ys, xs = np.where(lab == k)
        i = np.argmin(ps[ys, xs])
        y, x = int(ys[i]), int(xs[i])
        if y < border or x < border or y >= ny - border or x >= nx - border:
            continue
        win = p[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        wy, wx = np.unravel_index(np.argmin(win), win.shape)
        yy, xx = max(0, y - 1) + wy, max(0, x - 1) + wx
        out.append(Low(float(lat[yy]), float(lon[xx]), float(p[yy, xx])))
    return out


def filter_lows(member_lows: list[list[Low]], max_hpa: float | None) -> list[list[Low]]:
    """Drop member lows whose central pressure exceeds ``max_hpa`` (WPC's automated low-cluster product
    excludes lows above 1000 hPa so the map shows only strong, potentially impactful cyclones)."""
    if max_hpa is None:
        return member_lows
    return [[l for l in ml if l.p_hpa <= max_hpa] for ml in member_lows]


def _hav_matrix(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    dla = la[:, None] - la[None, :]
    dlo = lo[:, None] - lo[None, :]
    a = np.sin(dla / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def cluster_lows(member_lows: list[list[Low]], weights: np.ndarray, eps_km: float = 450.0,
                 min_support: float = 0.2) -> list[Cluster]:
    from sklearn.cluster import AgglomerativeClustering

    pts = [(mi, lw) for mi, lows in enumerate(member_lows) for lw in lows]
    if not pts:
        return []
    mem = np.array([m for m, _ in pts])
    lat = np.array([l.lat for _, l in pts])
    lon = np.array([l.lon for _, l in pts])
    prs = np.array([l.p_hpa for _, l in pts])
    if len(pts) == 1:
        labels = np.array([0])
    else:
        d = _hav_matrix(lat, lon)
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=eps_km, metric="precomputed",
                                         linkage="average").fit_predict(d)
    out = []
    total_w = float(weights.sum())
    for c in np.unique(labels):
        sel = np.where(labels == c)[0]
        # one low per member per cluster: keep that member's deepest
        best = {}
        for j in sel:
            m = mem[j]
            if m not in best or prs[j] < prs[best[m]]:
                best[m] = j
        idx = np.array(list(best.values()))
        wj = weights[mem[idx]]
        support = float(wj.sum() / total_w)
        if support < min_support:
            continue
        wn = wj / wj.sum()
        clat, clon, cp = float((wn * lat[idx]).sum()), float((wn * lon[idx]).sum()), float((wn * prs[idx]).sum())
        dist = _hav_matrix(np.append(lat[idx], clat), np.append(lon[idx], clon))[-1, :-1]
        out.append(Cluster(clat, clon, cp, support, len(idx),
                           float(np.sqrt((wn * dist ** 2).sum())),
                           float(np.sqrt((wn * (prs[idx] - cp) ** 2).sum())),
                           float(prs[idx].min()), float(prs[idx].max())))
    return sorted(out, key=lambda c: -c.support)
