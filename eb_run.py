"""Orchestrator:  choose cycle -> fetch -> blend -> render -> manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import yaml

import eb_blend as blend
import eb_lows as lows
import eb_plot as plot
import eb_snow as snow
import eb_sources as S
from eb_fetch import Fetcher
from eb_http import Http
from eb_systems import (FRZR, LOADERS, MIX, NONE, PTYPE_NAMES, RAIN, SLEET, SNOW, SysData, load_sysdata,
                      save_sysdata, target_grid)

log = logging.getLogger("eb")
CUMULATIVE = {"ifs", "aifs"}  # tp/sf are running totals from step 0 (others give 6-h windows)
SNOW_SYSTEMS = {"ifs", "aifs", "gefs"}
PTYPE_SYSTEMS = {"ifs", "gefs"}
MM_IN = 1 / 25.4

PROBE = {
    "gefs": lambda c, s: S.gefs_file(c, "gec00", s, "sp25"),
    "ifs": lambda c, s: S.ifs_ens_file(c, s),
    "aifs": lambda c, s: S.aifs_ens_file(c, s, "pf"),
    "aigefs": lambda c, s: S.aigefs_file(c, "mem030", s, "sfc"),
    "aigfs": lambda c, s: S.aigfs_file(c, s, "sfc"),
}


def utc(dt):
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def floor_cycle(now: datetime) -> datetime:
    return now.replace(hour=now.hour // 6 * 6, minute=0, second=0, microsecond=0, tzinfo=None)


def available(fx: Fetcher, name: str, cycle: datetime, step: int) -> bool:
    return any(fx.exists(c) for c in fx.ordered(PROBE[name](cycle, step)))


def choose_cycles(fx, names, now, lead, max_lag, min_systems, forced=None):
    """-> (T0, {system: (init, offset_h)})"""
    base = forced or floor_cycle(now)
    latest = {}
    for n in names:
        for k in range(0 if not forced else 0, 12):
            c = base - timedelta(hours=6 * k)
            if available(fx, n, c, lead):
                latest[n] = c
                break
    log.info("latest complete cycle per system: %s", {k: v.strftime("%Y%m%d%H") for k, v in latest.items()})
    if not latest:
        return None, {}
    cands = [forced] if forced else sorted(set(latest.values()), reverse=True)
    for need in range(min(min_systems, len(names)), 0, -1):
        for t0 in cands:
            use = {}
            for n in names:
                for lag in range(0, max_lag + 1, 6):
                    c = t0 - timedelta(hours=lag)
                    if n in latest and latest[n] >= c and available(fx, n, c, lag + lead):
                        use[n] = (c, lag)
                        break
            if len(use) >= need:
                return t0, use
    return None, {}


# ---------------------------------------------------------------------------
def _cache_path(cache: Path, name, init, off, params):
    h = hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return cache / f"{name}_{init:%Y%m%d%H}_{off}_{h}.npz"


def load_system(name, fx, cfg, init, off, bbox, lat, lon, want_snow, cache: Path):
    lead, plead, st = cfg["lead_hours"], cfg["precip_lead_hours"], cfg["step_hours"]
    want_precip = cfg["products"]["qpf"] or cfg["products"]["snow"] or cfg["products"]["ptype"]
    steps_msl = [off + L for L in range(0, lead + 1, st)] if cfg["products"]["lows"] else []
    p0 = 0 if name in CUMULATIVE else st
    steps_pcp = [off + L for L in range(p0, plead + 1, st)] if want_precip else []
    snow_here = want_snow and name in SNOW_SYSTEMS
    params = dict(steps_msl=steps_msl, steps_pcp=steps_pcp, snow=snow_here, bbox=bbox,
                  prof=cfg["profile_members"])
    cp = _cache_path(cache, name, init, off, params)
    if cp.exists():
        log.info("%s: using cached decode %s", name, cp.name)
        return load_sysdata(cp)
    kw = {}
    if name in ("ifs", "aifs"):
        pm = cfg["profile_members"]
        kw["profile_members"] = None if pm == "all" else int(pm)
    kw["workers"] = cfg["fetch"]["job_workers"] if name in ("gefs", "ifs", "aifs") else 2
    t = time.time()
    sd = LOADERS[name](fx, init, steps_msl, steps_pcp, snow_here, bbox, lat, lon, off=off, **kw)
    log.info("%s: loaded in %.0fs (%d failed files)", name, time.time() - t, sd.failures)
    if sd.failures == 0 and any(sd.fields.values()):
        cache.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(".tmp.npz")
        save_sysdata(sd, tmp)
        tmp.replace(cp)  # atomic: a killed run never leaves a half-written cache file
    return sd


# ---------------------------------------------------------------------------
class Run:
    def __init__(self, cfg, t0, sysdata: dict[str, SysData], bbox, domain_key, out_dir: Path):
        self.cfg, self.t0, self.sd, self.bbox, self.dk, self.out = cfg, t0, sysdata, bbox, domain_key, out_dir
        self.lat, self.lon = next(iter(sysdata.values())).lat, next(iter(sysdata.values())).lon
        self.weights = {n: cfg["systems"][n]["weight"] for n in sysdata}
        self.labels = {n: cfg["systems"][n]["label"] for n in sysdata}
        self.items: list[dict] = []
        out_dir.mkdir(parents=True, exist_ok=True)

    # -- helpers ----------------------------------------------------------
    def valid(self, lead):
        return self.t0 + timedelta(hours=lead)

    def desc(self, names):
        return " + ".join(f"{self.labels[n]} ({self.sd[n].n_nominal})" for n in names)

    def footer(self, names):
        w = " ".join(f"{n.upper()} {self.weights[n]:.2f}" for n in names)
        return (f"Blend of {self.desc(names)}.  Member weights: system weight / member count ({w}), renormalised.  "
                f"Cycle {fmt_cycle(self.t0)} (systems may be time-lagged <= {self.cfg['max_lag_hours']} h).  "
                f"Basemap: Natural Earth.  Colours: NWS (QPF, snow), ColorBrewer (probability, type).")

    def header(self, what, lead, period=None):
        sub = (f"{what}   |   valid {plot.fmt_t(self.valid(lead))}" if not period else
               f"{what}   |   {period}")
        return sub + f"   |   init {plot.fmt_t(self.t0)}  (f{lead:03d})"

    def collect(self, field, lead, names=None):
        items, used = [], []
        for n in (names or self.sd):
            a = self.sd[n].get(field, lead)
            if a is not None and a.shape[0] > 0:
                items.append((self.weights[n], a))
                used.append(n)
        return items, used

    def acc(self, field, n, lead_from, lead_to, st=6):
        """Sum 6-h windows (lead_from, lead_to] for one system; None if any window is missing/inconsistent."""
        arrs = [self.sd[n].get(field, L) for L in range(lead_from + st, lead_to + 1, st)]
        if any(a is None for a in arrs) or len({a.shape for a in arrs}) != 1:
            return None
        return np.sum(arrs, axis=0)

    def add(self, group, pid, title, lead, fname, period=None, extra=None):
        d = {"group": group, "id": pid, "title": title, "lead": lead, "valid": self.valid(lead).strftime("%Y-%m-%dT%H:00Z"),
             "period": period, "file": fname}
        d.update(extra or {})
        self.items.append(d)

    def path(self, name):
        return self.out / name

    # -- products ---------------------------------------------------------
    def make_lows(self):
        c = self.cfg["lows"]
        track = {}  # lead -> (member_lows, weights, used)  kept only for leads in the 0-72 h composite
        t_hours, t_step = c.get("track_hours", 72), c.get("track_step", 12)
        for lead in range(0, self.cfg["lead_hours"] + 1, self.cfg["step_hours"]):
            items, used = self.collect("msl", lead)
            if len(used) == 0:
                continue
            arr, w = blend.combine(items)
            member_lows = [lows.detect_lows(a, self.lat, self.lon, c["h_hpa"], c["sigma_gridpts"], c["border_gridpts"])
                           for a in arr]
            if lead <= t_hours and lead % t_step == 0:
                track[lead] = (member_lows, w, used)
            clusters = lows.cluster_lows(member_lows, w, c["eps_km"], c["min_support"])
            pts = np.array([[l.lat, l.lon] for ml in member_lows for l in ml]) if any(member_lows) else np.empty((0, 2))
            mean = blend.wmean(arr, w) / 100.0
            fn = f"lows_f{lead:03d}.png"
            plot.plot_lows(self.path(fn), self.bbox, self.lat, self.lon, mean, clusters, pts,
                           "Surface lows - ensemble clusters (L)", self.header("MSLP", lead), self.footer(used))
            self.add("lows", "lows", "Surface lows (L)", lead, fn,
                     extra={"systems": used, "members": int(arr.shape[0]),
                            "lows": [{"lat": round(k.lat, 2), "lon": round(k.lon, 2), "hpa": round(k.p_hpa, 1),
                                      "support": round(k.support, 2), "pos_sd_km": round(k.pos_sd_km), "p_sd_hpa": round(k.p_sd_hpa, 1)}
                                     for k in clusters]})
            log.info("lows f%03d: %d members, %d clusters", lead, arr.shape[0], len(clusters))
        self.make_lows_track(track)

    def make_lows_track(self, track):
        """WPC-style composite of member lows (coloured by valid time) and cluster-mean 'L' stamps."""
        c = self.cfg["lows"]
        if len(track) < 2:
            return
        frames, used_all, n_mem, info = [], set(), 0, []
        for lead in sorted(track):
            member_lows, w, used = track[lead]
            kept = lows.filter_lows(member_lows, c.get("track_max_hpa", 1000))
            clusters = [k for k in lows.cluster_lows(kept, w, c["eps_km"], c["min_support"])
                        if k.p_hpa <= c.get("track_max_hpa", 1000)]
            pts = np.array([[l.lat, l.lon] for ml in kept for l in ml]) if any(kept) else np.empty((0, 2))
            frames.append((lead, self.valid(lead), pts, clusters))
            used_all.update(used)
            n_mem = max(n_mem, len(member_lows))
            info.append({"lead": lead, "valid": self.valid(lead).strftime("%Y-%m-%dT%H:00Z"), "n_member_lows": int(len(pts)),
                         "clusters": [{"lat": round(k.lat, 2), "lon": round(k.lon, 2), "hpa": round(k.p_hpa, 1),
                                       "support": round(k.support, 2)} for k in clusters]})
        last = frames[-1][0]
        issued = ["Ensemble Blender - low clusters", f"Init {self.t0:%HZ %a %b %d, %Y}",
                  f"Valid {self.valid(0):%HZ %a %b %d, %Y}", f"Through {self.valid(last):%HZ %a %b %d, %Y}",
                  "Not an official NWS product"]
        fn = "lows_track.png"
        names = [n for n in self.sd if n in used_all]
        plot.plot_lows_track(self.path(fn), self.bbox, self.lat, self.lon, frames,
                             "Automated low clusters - 0-%d h ensemble guidance" % last,
                             f"Member lows <= {c.get('track_max_hpa', 1000):g} hPa (dots, by valid time) and cluster-mean lows (L)"
                             f"   |   init {plot.fmt_t(self.t0)}",
                             self.footer(names) + "  Display adapted from the WPC automated low-cluster graphic.", issued)
        self.add("lows_track", "lows_track", "Low clusters 0-%d h" % last, 0, fn, extra={"frames": info})
        log.info("lows track: %d valid times", len(frames))

    def _qpf_members(self, lead_from, lead_to):
        items, used = [], []
        for n in self.sd:
            a = self.acc("pcp6", n, lead_from, lead_to)
            if a is not None:
                items.append((self.weights[n], a))
                used.append(n)
        return items, used

    def make_qpf(self):
        plead, st = self.cfg["precip_lead_hours"], self.cfg["step_hours"]
        pe = self.cfg["prob_every_hours"]
        for lead in range(st, plead + 1, st):
            # 6-h
            items, used = self._qpf_members(lead - st, lead)
            if items:
                arr, w = blend.combine(items)
                data = blend.prob_matched_mean(arr, w) * MM_IN
                fn = f"qpf6_f{lead:03d}.png"
                plot.plot_filled(self.path(fn), self.bbox, self.lat, self.lon, data, "qpf6",
                                 "QPF - 6-hour precipitation (probability-matched mean)",
                                 self.header("6-h QPF", lead, f"6 h ending {plot.fmt_t(self.valid(lead))}"),
                                 self.footer(used), "inches of liquid-equivalent precipitation")
                self.add("qpf6", "qpf6", "6-h QPF", lead, fn, period="6 h", extra={"systems": used})
            # 24-h
            if lead % 24 == 0:
                items, used = self._qpf_members(lead - 24, lead)
                if items:
                    arr, w = blend.combine(items)
                    data = blend.prob_matched_mean(arr, w) * MM_IN
                    fn = f"qpf24_f{lead:03d}.png"
                    plot.plot_filled(self.path(fn), self.bbox, self.lat, self.lon, data, "qpf24",
                                     "QPF - 24-hour precipitation (probability-matched mean)",
                                     self.header("24-h QPF", lead, f"24 h ending {plot.fmt_t(self.valid(lead))}"),
                                     self.footer(used), "inches of liquid-equivalent precipitation")
                    self.add("qpf24", "qpf24", "24-h QPF", lead, fn, period="24 h", extra={"systems": used})
            # storm total
            items, used = self._qpf_members(0, lead)
            if items:
                arr, w = blend.combine(items)
                data = blend.prob_matched_mean(arr, w) * MM_IN
                fn = f"qpf_total_f{lead:03d}.png"
                plot.plot_filled(self.path(fn), self.bbox, self.lat, self.lon, data, "qpf_total",
                                 "QPF - total since forecast start (probability-matched mean)",
                                 self.header("Total QPF", lead, f"0-{lead} h, valid {plot.fmt_t(self.valid(lead))}"),
                                 self.footer(used), "inches of liquid-equivalent precipitation")
                self.add("qpf_total", "qpf_total", "Total QPF", lead, fn, period=f"0-{lead} h", extra={"systems": used})
                if lead % pe == 0:
                    for thr in self.cfg["qpf_prob_thresholds_in"]:
                        p = blend.exceed_prob(arr * MM_IN, w, thr) * 100
                        fn = f"qpf_prob_{thr:g}in_f{lead:03d}.png"
                        plot.plot_filled(self.path(fn), self.bbox, self.lat, self.lon, p, "prob",
                                         f"Probability total precipitation >= {thr:g} in",
                                         self.header(f"P(total QPF >= {thr:g} in)", lead, f"0-{lead} h"),
                                         self.footer(used), "% of (weighted) ensemble members")
                        self.add("qpf_prob", f"qpf_prob_{thr:g}", f"P(total QPF >= {thr:g} in)", lead, fn,
                                 period=f"0-{lead} h", extra={"systems": used})
            log.info("qpf f%03d done", lead)

    def _snow_members(self, lead_from, lead_to, method):
        items, used = [], []
        for n in self.sd:
            if n not in SNOW_SYSTEMS:
                continue
            sw = [self.sd[n].get("snow6", L) for L in range(lead_from + 6, lead_to + 1, 6)]
            if any(a is None for a in sw) or len({a.shape for a in sw}) != 1:
                continue
            tl = [self.sd[n].get("tloft", L) for L in range(lead_from + 6, lead_to + 1, 6)]
            if method != "climo" and any(t is None for t in tl):
                log.warning("%s: missing temperature profiles for %s-%s h; skipping for method %s", n, lead_from, lead_to, method)
                continue
            items.append((self.weights[n], snow.snow_depth_mm(sw, tl, method, self.cfg.get("snow_climo_slr", snow.SLR_CLIMO))))
            used.append(n)
        return items, used

    def make_snow(self):
        plead, st, pe = self.cfg["precip_lead_hours"], self.cfg["step_hours"], self.cfg["prob_every_hours"]
        names = {"climo": "climatological %g:1 ratio" % self.cfg.get("snow_climo_slr", snow.SLR_CLIMO), "maxtaloft": "temperature-dependent ratio (NBM MaxTaloft)"}
        for lead in range(st, plead + 1, st):
            for method, mlabel in names.items():
                if lead % 24 == 0:
                    items, used = self._snow_members(lead - 24, lead, method)
                    if items:
                        arr, w = blend.combine(items)
                        data = blend.prob_matched_mean(arr, w) * MM_IN
                        fn = f"snow24_{method}_f{lead:03d}.png"
                        plot.plot_filled(self.path(fn), self.bbox, self.lat, self.lon, data, "snow",
                                         f"Snowfall - 24 hours, {mlabel}",
                                         self.header("24-h snowfall (PMM)", lead, f"24 h ending {plot.fmt_t(self.valid(lead))}"),
                                         self.footer(used), "inches of snow")
                        self.add(f"snow24_{method}", f"snow24_{method}", f"24-h snow, {mlabel}", lead, fn, period="24 h",
                                 extra={"systems": used})
                items, used = self._snow_members(0, lead, method)
                if not items:
                    continue
                arr, w = blend.combine(items)
                data = blend.prob_matched_mean(arr, w) * MM_IN
                fn = f"snow_total_{method}_f{lead:03d}.png"
                plot.plot_filled(self.path(fn), self.bbox, self.lat, self.lon, data, "snow",
                                 f"Snowfall - total since start, {mlabel}",
                                 self.header("Total snowfall (PMM)", lead, f"0-{lead} h"),
                                 self.footer(used), "inches of snow")
                self.add(f"snow_total_{method}", f"snow_total_{method}", f"Total snow, {mlabel}", lead, fn,
                         period=f"0-{lead} h", extra={"systems": used})
                if method == "maxtaloft" and lead % pe == 0:
                    for thr in self.cfg["snow_prob_thresholds_in"]:
                        p = blend.exceed_prob(arr * MM_IN, w, thr) * 100
                        fn = f"snow_prob_{thr:g}in_f{lead:03d}.png"
                        plot.plot_filled(self.path(fn), self.bbox, self.lat, self.lon, p, "prob",
                                         f"Probability total snowfall >= {thr:g} in",
                                         self.header(f"P(total snow >= {thr:g} in)", lead, f"0-{lead} h"),
                                         self.footer(used), "% of (weighted) ensemble members")
                        self.add("snow_prob", f"snow_prob_{thr:g}", f"P(total snow >= {thr:g} in)", lead, fn,
                                 period=f"0-{lead} h", extra={"systems": used})
            log.info("snow f%03d done", lead)

    def make_ptype(self):
        for lead in range(self.cfg["step_hours"], self.cfg["precip_lead_hours"] + 1, self.cfg["step_hours"]):
            items, used = self.collect("ptype", lead, [n for n in self.sd if n in PTYPE_SYSTEMS])
            if not items:
                continue
            arr, w = blend.combine(items)
            probs = {c: blend.wmean((arr == c).astype(np.float32), w) for c in (RAIN, SNOW, FRZR, SLEET, MIX)}
            p_any = blend.wmean((arr != NONE).astype(np.float32), w)
            probs = {c: p for c, p in probs.items() if p.max() > 0}
            if not probs:
                continue
            fn = f"ptype_f{lead:03d}.png"
            plot.plot_ptype(self.path(fn), self.bbox, self.lat, self.lon, probs, p_any,
                            "Precipitation type - most likely (ensemble vote)",
                            self.header("Precip type (shown where >=50% of members have precip)", lead),
                            self.footer(used))
            self.add("ptype", "ptype", "Precip type", lead, fn, extra={"systems": used})
        log.info("ptype done")


def fmt_cycle(t):
    return t.strftime("%Y-%m-%d %HZ")


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Ensemble Blender product builder")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    ap.add_argument("--out", default="site/data")
    ap.add_argument("--cache", default=".cache/eb")
    ap.add_argument("--domain")
    ap.add_argument("--cycle", help="force T0 as YYYYMMDDHH")
    ap.add_argument("--systems", help="comma list, overrides config enabled")
    ap.add_argument("--only", help="comma list of: lows,qpf,snow,ptype")
    ap.add_argument("--lead", type=int, help="override lead_hours")
    ap.add_argument("--precip-lead", type=int, help="override precip_lead_hours")
    ap.add_argument("--force", action="store_true", help="rebuild even if this run already exists")
    ap.add_argument("--dry-run", action="store_true", help="choose cycles, print plan, exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.lead:
        cfg["lead_hours"] = a.lead
    if a.precip_lead:
        cfg["precip_lead_hours"] = a.precip_lead
    if a.only:
        sel = set(a.only.split(","))
        cfg["products"] = {k: (k in sel) for k in cfg["products"]}
    dk = a.domain or cfg["domain"]
    bbox = cfg["domains"][dk]
    names = a.systems.split(",") if a.systems else [n for n, v in cfg["systems"].items() if v["enabled"]]
    lat, lon = target_grid(bbox)
    http = Http(cfg["user_agent"])
    fx = Fetcher(http, bbox, cfg["fetch"]["max_gap_bytes"], cfg["fetch"]["range_workers"],
                 cfg["fetch"]["nomads_filter_fallback"],
                 disabled_tiers={(n, t) for n in LOADERS for t in cfg["fetch"]["disabled_tiers"]})
    now = datetime.now(timezone.utc)
    forced = datetime.strptime(a.cycle, "%Y%m%d%H") if a.cycle else None
    t0, use = choose_cycles(fx, names, now, cfg["lead_hours"], cfg["max_lag_hours"], cfg["min_systems"], forced)
    if t0 is None:
        log.error("no usable cycle found")
        return 2
    log.info("T0 = %s ; systems: %s", fmt_cycle(t0), {n: f"{i:%Y%m%d%H} (lag {o}h)" for n, (i, o) in use.items()})
    run_id = t0.strftime("%Y%m%d%H")
    out_root = Path(a.out) / dk
    run_dir = out_root / run_id
    if (run_dir / "manifest.json").exists() and not a.force:
        log.info("run %s already built; nothing to do (use --force)", run_id)
        return 0
    if a.dry_run:
        return 0
    want_snow = cfg["products"]["snow"] or cfg["products"]["ptype"]
    sysdata = {}
    with ThreadPoolExecutor(max_workers=len(use), thread_name_prefix="sys") as ex:
        futs = {n: ex.submit(load_system, n, fx, cfg, init, off, bbox, lat, lon, want_snow, Path(a.cache))
                for n, (init, off) in use.items()}
        for n, f in futs.items():
            sysdata[n] = f.result()
            sysdata[n].offset = use[n][1]
    sysdata = {n: s for n, s in sysdata.items() if any(s.fields.values())}
    if not sysdata:
        log.error("no data loaded")
        return 3
    run = Run(cfg, t0, sysdata, bbox, dk, run_dir)
    for key, fn in (("lows", run.make_lows), ("qpf", run.make_qpf), ("snow", run.make_snow), ("ptype", run.make_ptype)):
        if cfg["products"].get(key):
            fn()
    manifest = {
        "run": run_id, "t0": t0.strftime("%Y-%m-%dT%H:00Z"), "domain": dk, "domain_label": bbox["label"],
        "bbox": {k: bbox[k] for k in ("lon_min", "lon_max", "lat_min", "lat_max")},
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "systems": {n: {"label": cfg["systems"][n]["label"], "init": s.init.strftime("%Y-%m-%dT%H:00Z"),
                        "lag_h": getattr(s, "offset", 0), "weight": cfg["systems"][n]["weight"],
                        "members": s.n_nominal, "failed_files": s.failures}
                    for n, s in sysdata.items()},
        "tiers_used": {f"{k[0]}:{k[1]}": v for k, v in sorted(fx.served.items())},
        "http": http.stats(),
        "products": run.items,
        "methods": {
            "weights": "system weight / n_members, renormalised over available systems",
            "qpf": "weighted probability-matched mean (Ebert 2001); exceedance = weighted member fraction",
            "snow": "6-h SWE x SLR summed over windows; SWE = model snowfall (IFS, AIFS) or APCP x CSNOW (GEFS); SLR = climatological %g:1 (config snow_climo_slr) and NBM MaxTaloft"
                    % cfg.get("snow_climo_slr", snow.SLR_CLIMO) + "",
            "ptype": "IFS ptype (instantaneous at valid time) and GEFS categorical flags (6-h average); weighted vote",
            "lows": "h-minima (dynamic >= %.1f hPa) per member, average-linkage clusters <= %d km; composite map drops member lows above %g hPa (WPC convention)" % (cfg["lows"]["h_hpa"], cfg["lows"]["eps_km"], cfg["lows"].get("track_max_hpa", 1000)),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out_root / "latest.json").write_text(json.dumps({"run": run_id, "manifest": f"{run_id}/manifest.json"}))
    # prune old runs
    runs = sorted(p for p in out_root.iterdir() if p.is_dir())
    for old in runs[:-cfg["keep_runs"]]:
        shutil.rmtree(old, ignore_errors=True)
    fx.close()
    tot = sum(v["bytes"] for v in http.stats().values())
    log.info("done: %d images, %.0f MB downloaded, requests %s", len(run.items), tot / 1e6,
             {h: v["requests"] for h, v in http.stats().items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
