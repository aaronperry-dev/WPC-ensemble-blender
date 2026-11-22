"""In-memory GRIB2 decoding (eccodes) with domain cropping.

Byte-range downloads hand us concatenated, complete GRIB messages; we split
them on the GRIB2 total-length field, filter on cheap metadata *before*
touching the packed data, and crop to the plotting domain immediately so only
a few thousand grid points per field are kept in memory.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import eccodes
except Exception as exc:  # pragma: no cover
    eccodes = None
    _IMPORT_ERR = exc


@dataclass
class Msg:
    short: str
    type_of_level: str
    level: int
    start_step: int
    end_step: int
    number: int | None
    values: np.ndarray  # (ny, nx) float32, lat ascending, lon ascending (-180..180)
    lat: np.ndarray
    lon: np.ndarray


def split_messages(blob: bytes):
    """Yield complete GRIB2 messages from a byte blob (junk between messages is skipped)."""
    pos, n = 0, len(blob)
    while pos < n:
        i = blob.find(b"GRIB", pos)
        if i < 0 or i + 16 > n:
            return
        if blob[i + 7] != 2:  # edition
            pos = i + 4
            continue
        length = int.from_bytes(blob[i + 8:i + 16], "big")
        if length <= 16 or i + length > n:
            return  # truncated tail -> ignore, caller will notice missing field
        yield blob[i:i + length]
        pos = i + length


def _get(gid, key, default=None):
    try:
        return eccodes.codes_get(gid, key)
    except Exception:
        return default


def _grid(gid):
    ni, nj = int(_get(gid, "Ni")), int(_get(gid, "Nj"))
    la1 = float(_get(gid, "latitudeOfFirstGridPointInDegrees"))
    lo1 = float(_get(gid, "longitudeOfFirstGridPointInDegrees"))
    di = float(_get(gid, "iDirectionIncrementInDegrees"))
    dj = float(_get(gid, "jDirectionIncrementInDegrees"))
    j_pos = bool(_get(gid, "jScansPositively", 0))
    i_neg = bool(_get(gid, "iScansNegatively", 0))
    lat = la1 + (dj if j_pos else -dj) * np.arange(nj)
    lon = lo1 + (-di if i_neg else di) * np.arange(ni)
    lon = ((lon + 180.0) % 360.0) - 180.0
    return ni, nj, lat, lon


def decode(blobs, want=None, bbox: dict | None = None) -> list[Msg]:
    """Decode every message accepted by ``want(meta_dict)``.

    ``bbox`` = dict(lon_min, lon_max, lat_min, lat_max) in -180..180 / -90..90.
    """
    if eccodes is None:  # pragma: no cover
        raise RuntimeError(f"eccodes not importable: {_IMPORT_ERR}")
    out: list[Msg] = []
    for blob in blobs:
        for raw in split_messages(blob):
            gid = eccodes.codes_new_from_message(raw)
            try:
                meta = {
                    "short": _get(gid, "shortName", ""),
                    "type_of_level": _get(gid, "typeOfLevel", ""),
                    "level": int(_get(gid, "level", 0) or 0),
                    "start_step": int(_get(gid, "startStep", 0) or 0),
                    "end_step": int(_get(gid, "endStep", 0) or 0),
                    "number": _get(gid, "perturbationNumber", None),
                }
                if want is not None and not want(meta):
                    continue
                ni, nj, lat, lon = _grid(gid)
                eccodes.codes_set(gid, "missingValue", 1e20)
                vals = eccodes.codes_get_values(gid).reshape(nj, ni).astype(np.float32)
                vals[vals > 1e19] = np.nan
            finally:
                eccodes.codes_release(gid)
            # orient: lat ascending, lon ascending
            if lat[0] > lat[-1]:
                lat, vals = lat[::-1], vals[::-1, :]
            order = np.argsort(lon)
            lon, vals = lon[order], vals[:, order]
            if bbox is not None:
                jy = np.where((lat >= bbox["lat_min"] - 1e-6) & (lat <= bbox["lat_max"] + 1e-6))[0]
                ix = np.where((lon >= bbox["lon_min"] - 1e-6) & (lon <= bbox["lon_max"] + 1e-6))[0]
                lat, lon, vals = lat[jy], lon[ix], vals[np.ix_(jy, ix)]
            out.append(Msg(meta["short"], meta["type_of_level"], meta["level"], meta["start_step"],
                           meta["end_step"], meta["number"], np.ascontiguousarray(vals), lat, lon))
    return out
