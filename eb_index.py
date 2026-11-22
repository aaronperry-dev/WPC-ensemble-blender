"""GRIB inventory parsing, message selectors and byte-range coalescing.

Two inventory flavours are supported:

* NCEP/wgrib2 ``.idx``  (``n:offset:d=YYYYMMDDHH:VAR:LEVEL:FCST:EXTRA``)
* ECMWF open-data ``.index`` (JSON lines with ``_offset`` / ``_length``)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class Rec:
    start: int
    end: int | None  # inclusive; None = to end of object
    meta: dict = field(default_factory=dict)


_WIN_HOUR = re.compile(r"^(\d+)-(\d+) hour (acc|ave|max|min)")
_WIN_DAY = re.compile(r"^(\d+)-(\d+) day (acc|ave|max|min)")
_INST = re.compile(r"^(\d+) hour fcst")


def fcst_window(ftxt: str):
    """Return (start_h, end_h, kind) from an NCEP forecast-time string."""
    m = _WIN_HOUR.match(ftxt)
    if m:
        return int(m[1]), int(m[2]), m[3]
    m = _WIN_DAY.match(ftxt)
    if m:
        return int(m[1]) * 24, int(m[2]) * 24, m[3]
    m = _INST.match(ftxt)
    if m:
        return int(m[1]), int(m[1]), "inst"
    return None


def parse_ncep_idx(text: str) -> list[Rec]:
    rows = []
    for line in text.splitlines():
        p = line.split(":", 6)
        if len(p) < 6 or not p[1].isdigit():
            continue
        extra = p[6] if len(p) > 6 else ""
        win = fcst_window(p[5])
        rows.append((int(p[1]), {"var": p[3], "level": p[4], "ftxt": p[5], "extra": extra,
                                 "window": win}))
    recs = []
    for i, (off, meta) in enumerate(rows):
        nxt = rows[i + 1][0] if i + 1 < len(rows) else None
        # records can repeat an offset (multi-field messages); skip zero-length
        end = (nxt - 1) if nxt is not None else None
        if nxt is not None and nxt <= off:
            continue
        recs.append(Rec(off, end, meta))
    return recs


def parse_ecmwf_index(text: str) -> list[Rec]:
    recs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        off, ln = d.get("_offset"), d.get("_length")
        if off is None or ln is None:
            continue
        recs.append(Rec(int(off), int(off) + int(ln) - 1, d))
    return recs


# --------------------------------------------------------------------------
# Selectors
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Sel:
    """NCEP selector: variable names x level strings, optional window length."""
    vars: tuple
    levels: tuple
    window_h: int | None = None  # require an accumulation/avg window of this length

    def match(self, meta: dict) -> bool:
        if meta["var"] not in self.vars or meta["level"] not in self.levels:
            return False
        if self.window_h is not None:
            w = meta.get("window")
            if not w or w[2] == "inst" or (w[1] - w[0]) != self.window_h:
                return False
        return True

    def filter_params(self) -> dict:
        p = {f"var_{v}": "on" for v in self.vars}
        p.update({"lev_" + lv.replace(" ", "_"): "on" for lv in self.levels})
        return p


@dataclass(frozen=True)
class ESel:
    """ECMWF open-data selector."""
    params: tuple
    levtype: str = "sfc"
    levelists: tuple | None = None  # e.g. ("850",)
    numbers: tuple | None = None  # ensemble member numbers as strings

    def match(self, meta: dict) -> bool:
        if meta.get("param") not in self.params or meta.get("levtype") != self.levtype:
            return False
        if self.levelists is not None and str(meta.get("levelist")) not in self.levelists:
            return False
        if self.numbers is not None and str(meta.get("number")) not in self.numbers:
            return False
        return True


def coalesce(ranges: list[tuple[int, int | None]], max_gap: int = 0) -> list[tuple[int, int | None]]:
    """Merge touching / nearly-touching byte ranges (S3 does not do multi-range GETs).

    ``max_gap`` bytes of unwanted data may be pulled to save a request; whole
    GRIB messages fall in the gap so the decoder can skip them cheaply.
    """
    if not ranges:
        return []
    rs = sorted(ranges, key=lambda r: r[0])
    out = [list(rs[0])]
    for s, e in rs[1:]:
        cs, ce = out[-1]
        if ce is None or s <= ce + 1 + max_gap:
            out[-1][1] = None if (ce is None or e is None) else max(ce, e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]
