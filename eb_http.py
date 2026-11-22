"""Polite, fast HTTP client for GRIB byte-range downloads.

Design goals
------------
* Server friendly: per-host minimum request spacing and a per-host concurrency
  cap (tight for NOMADS / data.ecmwf.int, looser for cloud object stores),
  exponential backoff with jitter, ``Retry-After`` honoured, an identifying
  User-Agent, and no retries on permanent errors (403/404).
* Efficient: persistent connection pools, ranged GETs only (never a whole GRIB
  file unless the server hands us one that is small), identity encoding so
  binary ranges are never re-compressed.
* Observable: per-host request/byte counters end up in the run manifest.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter

log = logging.getLogger("eb.http")


class NotFound(Exception):
    """HTTP 404 (or 403 on public buckets: key absent)."""


class Unavailable(Exception):
    """Retries exhausted / server unhealthy."""


class RangeNotSupported(Exception):
    """Server ignored the Range header and the full object is too large."""


@dataclass(frozen=True)
class HostPolicy:
    min_interval: float  # seconds between request *starts* to this host
    max_concurrency: int  # simultaneous in-flight requests to this host
    timeout: tuple = (10, 60)  # (connect, read) seconds
    max_retries: int = 4


# Matched by hostname suffix, first match wins.  Values chosen to stay well
# inside published guidance (NOMADS asks for < ~120 requests/min per IP).
DEFAULT_POLICIES: list[tuple[str, HostPolicy]] = [
    ("nomads.ncep.noaa.gov", HostPolicy(min_interval=0.6, max_concurrency=2)),
    ("data.ecmwf.int", HostPolicy(min_interval=0.5, max_concurrency=2)),
    ("collab.science.gc.ca", HostPolicy(min_interval=0.5, max_concurrency=2)),
    ("storage.googleapis.com", HostPolicy(min_interval=0.02, max_concurrency=8)),
    ("amazonaws.com", HostPolicy(min_interval=0.02, max_concurrency=8)),
    ("", HostPolicy(min_interval=0.5, max_concurrency=2)),  # default
]


@dataclass
class _Gate:
    policy: HostPolicy
    sem: threading.BoundedSemaphore = field(init=False)
    lock: threading.Lock = field(default_factory=threading.Lock)
    next_ok: float = 0.0
    penalty_until: float = 0.0
    requests: int = 0
    bytes: int = 0
    errors: int = 0

    def __post_init__(self):
        self.sem = threading.BoundedSemaphore(self.policy.max_concurrency)

    def wait_turn(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_ok, self.penalty_until)
            self.next_ok = t + self.policy.min_interval
        if t > now:
            time.sleep(t - now)

    def penalize(self, seconds: float):
        with self.lock:
            self.penalty_until = max(self.penalty_until, time.monotonic() + seconds)


class Http:
    def __init__(self, user_agent: str, policies=None, max_full_bytes: int = 64 * 1024 * 1024):
        self.policies = policies or DEFAULT_POLICIES
        self.max_full_bytes = max_full_bytes
        self._gates: dict[str, _Gate] = {}
        self._glock = threading.Lock()
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=32, max_retries=0)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "identity"})

    # -- policy / stats ----------------------------------------------------
    def _gate(self, host: str) -> _Gate:
        with self._glock:
            g = self._gates.get(host)
            if g is None:
                for suffix, pol in self.policies:
                    if host.endswith(suffix):
                        g = _Gate(pol)
                        break
                self._gates[host] = g
            return g

    def stats(self) -> dict:
        return {h: {"requests": g.requests, "bytes": g.bytes, "errors": g.errors}
                for h, g in self._gates.items()}

    # -- core request with retry ------------------------------------------
    def _request(self, url: str, headers: dict | None = None, want_range: bool = False,
                 retries: int | None = None) -> requests.Response:
        host = urlparse(url).hostname or ""
        gate = self._gate(host)
        pol = gate.policy
        last: Exception | None = None
        n_retries = pol.max_retries if retries is None else retries
        for attempt in range(n_retries + 1):
            gate.wait_turn()
            with gate.sem:
                try:
                    r = self.session.get(url, headers=headers, timeout=pol.timeout, stream=True)
                except (requests.ConnectionError, requests.Timeout) as e:
                    last = e
                    gate.errors += 1
                else:
                    gate.requests += 1
                    code = r.status_code
                    if code in (200, 206):
                        if want_range and code == 200:
                            clen = int(r.headers.get("Content-Length") or 0)
                            if clen > self.max_full_bytes:
                                r.close()
                                raise RangeNotSupported(f"{host} ignored Range for {url} ({clen} bytes)")
                        return r
                    r.close()
                    if code in (403, 404):
                        raise NotFound(f"{code} {url}")
                    if code == 416:
                        raise NotFound(f"416 (range beyond object) {url}")
                    gate.errors += 1
                    last = Unavailable(f"HTTP {code} for {url}")
                    if code in (429, 503):
                        ra = r.headers.get("Retry-After")
                        wait = float(ra) if ra and ra.isdigit() else 5 * (2 ** attempt)
                        gate.penalize(min(wait, 120))
                    elif code not in (500, 502, 504):
                        raise last
            sleep = min(60.0, 0.75 * (2 ** attempt)) * (0.5 + random.random())
            log.debug("retry %s in %.1fs (%s)", url, sleep, last)
            time.sleep(sleep)
        raise Unavailable(f"gave up on {url}: {last}")

    # -- public helpers -----------------------------------------------------
    def get_text(self, url: str) -> str | None:
        """Small text object (e.g. .idx).  Returns None if the object is absent."""
        try:
            r = self._request(url)
        except NotFound:
            return None
        body = r.content
        self._gate(urlparse(url).hostname or "").bytes += len(body)
        return body.decode("utf-8", "replace")

    def get_range(self, url: str, start: int, end: int | None, retries: int | None = None) -> bytes:
        """Bytes ``start..end`` inclusive (``end=None`` -> to EOF)."""
        rng = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
        r = self._request(url, {"Range": rng}, want_range=True, retries=retries)
        body = r.content
        if r.status_code == 200:  # small object, server ignored Range -> slice locally
            body = body[start:(end + 1 if end is not None else None)]
        elif end is not None and len(body) != end - start + 1:
            raise Unavailable(f"short read {len(body)} != {end - start + 1} for {url} {rng}")
        self._gate(urlparse(url).hostname or "").bytes += len(body)
        return body

    def get_all(self, url: str) -> bytes | None:
        """Whole (small) object, e.g. a GRIB-filter response."""
        try:
            r = self._request(url)
        except NotFound:
            return None
        body = r.content
        self._gate(urlparse(url).hostname or "").bytes += len(body)
        return body
