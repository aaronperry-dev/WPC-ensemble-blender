"""Snowfall depth from snow water equivalent (SWE) and a snow-to-liquid ratio (SLR).

Two methods are provided; the literature does not crown a winner:
Veals et al. (2025) and the NBM SLR study cited in the README find that the
choice of SLR is as consequential for snowfall error as the model-to-model
differences themselves, so we publish both rather than pretend one is "right".

``climo``      Fixed climatological ratio, default 12.5:1 (``snow_climo_slr`` in the
               config).  Deliberately NOT 10:1: verification of Northeast events
               (Baxter et al. 2005 climatology; local WFO experience) puts the
               typical SLR nearer 12-13:1, so 10:1 runs systematically low there.
               Set it per region if you run another domain.
``maxtaloft``  NBM "MaxTaloft" method, a fifth-degree polynomial of the maximum
               temperature (deg C) between 610 m AGL and 400 hPa (polynomial as
               tabulated in Veals et al. 2025).  We approximate the layer
               maximum from pressure-level temperatures (925/850/700/600/500 hPa
               filtered to levels above 610 m AGL, see systems.tloft_from_levels).
"""
from __future__ import annotations

import numpy as np

# coefficients highest power first: T^5 .. T^0  (T in deg C)
_MAXTALOFT = (0.0000045, 0.0004432, 0.0130903, 0.0585968, -1.8150809, 5.9805722)
SLR_CLIMO = 12.5  # default climatological ratio (Northeast); overridden by config
SLR_MIN, SLR_MAX = 1.0, 25.0  # physical guard rails (NBM caps its SLR at 25)
T_VALID = (-30.0, 5.0)  # polynomial is only meaningful over this range; clamp outside


def slr_maxtaloft(tmax_k: np.ndarray) -> np.ndarray:
    t = np.clip(tmax_k - 273.15, *T_VALID)
    slr = np.polyval(_MAXTALOFT, t)
    return np.clip(slr, SLR_MIN, SLR_MAX).astype(np.float32)


def slr_field(method: str, tloft: np.ndarray | None, n: int, shape, climo: float = SLR_CLIMO) -> np.ndarray:
    """Return SLR broadcastable to (n, ny, nx).

    ``tloft`` is (k, ny, nx).  If k == n each member uses its own profile;
    otherwise (profile subset) members share the mean SLR of the k available.
    """
    if method == "climo" or tloft is None:
        return np.float32(climo)
    slr = slr_maxtaloft(tloft)
    if slr.shape[0] == n:
        return slr
    return slr.mean(axis=0, keepdims=True)


def snow_depth_mm(swe_windows: list[np.ndarray], tloft_windows: list, method: str, climo: float = SLR_CLIMO) -> np.ndarray:
    """Sum over windows of SWE(mm) x SLR  ->  snowfall depth in mm (members x ny x nx)."""
    total = None
    for swe, tl in zip(swe_windows, tloft_windows):
        n = swe.shape[0]
        d = swe * slr_field(method, tl, n, swe.shape[1:], climo)
        total = d if total is None else total + d
    return total


def mm_to_in(a):
    return a / 25.4
