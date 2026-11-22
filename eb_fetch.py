"""Tiered, byte-range GRIB fetching.

For each file we walk its candidate locations in tier order
(AWS S3 -> GCP -> NOMADS / data.ecmwf.int).  Inside a tier we

  1. read the tiny inventory (.idx / .index),
  2. pick only the messages we need,
  3. coalesce neighbouring messages into as few ranged GETs as possible,
  4. pull the ranges in parallel (bounded by the per-host gate in http.py).

If a tier is missing the object (404) or is unhealthy, we move to the next.
NOMADS additionally offers the GRIB-filter service (server-side variable /
level / sub-region cut) which is used as a last resort for GEFS when ranged
reads from NOMADS fail.
"""
from __future__ import annotations

import logging
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from eb_http import Http, NotFound, RangeNotSupported, Unavailable
from eb_index import coalesce, parse_ecmwf_index, parse_ncep_idx
from eb_sources import Cand

log = logging.getLogger("eb.fetch")

TIER_ORDER = ["aws", "gcp", "nomads", "ecmwf", "msc"]


class AnySel:
    """Union of selectors (several variable groups from one file in one pass)."""

    def __init__(self, *sels):
        self.sels = sels

    def match(self, meta):
        return any(s.match(meta) for s in self.sels)

    def filter_params(self):
        p = {}
        for s in self.sels:
            p.update(s.filter_params())
        return p


class AllTiersFailed(Exception):
    pass


class Fetcher:
    def __init__(self, http: Http, bbox: dict, max_gap: int = 65536, range_workers: int = 12,
                 use_filter_fallback: bool = True, breaker_threshold: int = 6,
                 disabled_tiers: set | None = None):
        self.http, self.bbox, self.max_gap = http, bbox, max_gap
        self.pool = ThreadPoolExecutor(max_workers=range_workers, thread_name_prefix="range")
        self.use_filter_fallback = use_filter_fallback
        self.breaker_threshold = breaker_threshold
        self.disabled_tiers = set(disabled_tiers or ())  # {(system, tier)}
        self._fails: Counter = Counter()
        self._absent: Counter = Counter()  # (system, tier) -> consecutive 404s while the tier never served anything
        self.served: Counter = Counter()  # (system, tier) -> files
        self.missing: Counter = Counter()  # system -> files that no tier could serve
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def ordered(self, cands: list[Cand]) -> list[Cand]:
        return sorted(cands, key=lambda c: TIER_ORDER.index(c.tier) if c.tier in TIER_ORDER else 99)

    def exists(self, cand: Cand) -> bool:
        """Cheap existence probe: 1-byte ranged GET on the data object."""
        try:
            self.http.get_range(cand.url, 0, 0, retries=0)
            return True
        except (NotFound, Unavailable, RangeNotSupported):
            return False

    def _tier_ok(self, system: str, tier: str) -> bool:
        return (system, tier) not in self.disabled_tiers

    def _note_fail(self, system: str, tier: str):
        with self._lock:
            self._fails[(system, tier)] += 1
            if self._fails[(system, tier)] >= self.breaker_threshold:
                if (system, tier) not in self.disabled_tiers:
                    log.warning("circuit open: %s via %s disabled for this run", system, tier)
                self.disabled_tiers.add((system, tier))

    # ------------------------------------------------------------------
    def _ranged(self, cand: Cand, sel) -> list[bytes]:
        if cand.index_kind == "none":
            blob = self.http.get_all(cand.url)
            if blob is None:
                raise NotFound(cand.url)
            return [blob]
        text = self.http.get_text(cand.index_url)
        if text is None:
            raise NotFound(cand.index_url)
        recs = parse_ncep_idx(text) if cand.index_kind == "ncep" else parse_ecmwf_index(text)
        want = [r for r in recs if sel.match(r.meta)]
        if not want:
            return []
        ranges = coalesce([(r.start, r.end) for r in want], self.max_gap)
        futs = [self.pool.submit(self.http.get_range, cand.url, a, b) for a, b in ranges]
        return [f.result() for f in futs]

    def get(self, system: str, cands: list[Cand], sel) -> tuple[list[bytes], str]:
        """Return (blobs, tier).  Raises AllTiersFailed if nothing could serve the file."""
        last: Exception | None = None
        for cand in self.ordered(cands):
            if not self._tier_ok(system, cand.tier):
                continue
            try:
                blobs = self._ranged(cand, sel)
                with self._lock:
                    self.served[(system, cand.tier)] += 1
                    self._absent[(system, cand.tier)] = 0
                return blobs, cand.tier
            except NotFound as e:
                last = e
                with self._lock:
                    key = (system, cand.tier)
                    self._absent[key] += 1
                    # a tier that has 404'd repeatedly and never served a file does not carry this product
                    if self._absent[key] >= 3 and self.served[key] == 0 and key not in self.disabled_tiers:
                        log.info("%s is not published on tier %s; skipping that tier for this run", system, cand.tier)
                        self.disabled_tiers.add(key)
                continue
            except (Unavailable, RangeNotSupported) as e:
                last = e
                self._note_fail(system, cand.tier)
                # NOMADS: try the GRIB filter before giving up on the tier
                if self.use_filter_fallback and cand.filter_script and hasattr(sel, "filter_params"):
                    url = cand.filter_url(sel, self.bbox)
                    try:
                        body = self.http.get_all(url)
                        if body:
                            with self._lock:
                                self.served[(system, "nomads-filter")] += 1
                            return [body], "nomads-filter"
                    except (Unavailable, NotFound) as e2:
                        last = e2
                continue
        with self._lock:
            self.missing[system] += 1
        raise AllTiersFailed(f"{system}: {cands[0].url if cands else '?'} ({last})")

    def close(self):
        self.pool.shutdown(wait=True)
