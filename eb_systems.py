"""Per-system loaders: raw GRIB bytes -> standardised member arrays on one grid.

Every loader returns a :class:`SysData` holding, per *valid-time lead* (hours
after the blend's target cycle T0):

  msl     (n, ny, nx)  Pa      mean sea-level pressure (instantaneous)
  pcp6    (n, ny, nx)  mm      6-h liquid-equivalent precipitation ending at that lead
  snow6   (n, ny, nx)  mm w.e. 6-h *snow* water equivalent (systems with snow info only)
  ptype   (n, ny, nx)  int8    0 none,1 rain,2 snow,3 freezing rain,4 sleet,5 mixed
  tloft   (k, ny, nx)  K       max temp in ~610 m AGL..400 hPa layer (pressure-level approximation)

Units are normalised from the GRIB ``units`` key, never assumed (IFS ENS
reports precipitation in metres, AIFS ENS and NCEP in kg m-2).
"""
from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import eb_sources as S
from eb_fetch import AllTiersFailed, AnySel, Fetcher
from eb_grib import decode
from eb_index import ESel, Sel

log = logging.getLogger("eb.systems")

NONE, RAIN, SNOW, FRZR, SLEET, MIX = 0, 1, 2, 3, 4, 5
PTYPE_NAMES = {RAIN: "Rain", SNOW: "Snow", FRZR: "Freezing rain", SLEET: "Sleet / ice pellets", MIX: "Mixed"}

# ECMWF/NCEP GRIB2 code table 4.201 -> our categories (observed live: 0,1,3,5,6,7,8,12)
_IFS_PTYPE = np.zeros(32, dtype=np.int8)
for _c, _v in {1: RAIN, 11: RAIN, 3: FRZR, 12: FRZR, 5: SNOW, 6: SNOW, 7: MIX, 8: SLEET, 4: SLEET}.items():
    _IFS_PTYPE[_c] = _v

_LOCK = threading.Lock()


def _put(raw, step, key, mi, arr):
    with _LOCK:
        raw[step][key][mi] = arr


PCP_MIN_MM = 0.254  # 0.01 in: below this a gridpoint is "no precipitation" for type diagnosis
T_LEVELS = (925, 850, 700)  # hPa levels scanned for the warm-layer maximum (the warmest level in the
# 610 m AGL..400 hPa layer is essentially always at or below 700 hPa; 600/500 cost ~40% more bytes for no change)
SCALE_H = 8000.0  # m, for 610 m AGL ~ sp*exp(-610/H)


@dataclass
class SysData:
    name: str
    init: datetime
    n_nominal: int
    lat: np.ndarray
    lon: np.ndarray
    fields: dict = field(default_factory=lambda: defaultdict(dict))  # name -> {lead: array}
    tiers: dict = field(default_factory=dict)
    failures: int = 0

    def get(self, name, lead):
        return self.fields.get(name, {}).get(lead)


def save_sysdata(sd: "SysData", path):
    """Plain .npz (no pickle): arrays keyed 'field|lead', plus a small JSON header."""
    arrays = {f"{k}|{L}": a for k, d in sd.fields.items() for L, a in d.items()}
    meta = json.dumps({"name": sd.name, "init": sd.init.isoformat(), "n_nominal": sd.n_nominal,
                       "failures": sd.failures})
    np.savez_compressed(path, _meta=np.array(meta), _lat=sd.lat, _lon=sd.lon, **arrays)


def load_sysdata(path) -> "SysData":
    z = np.load(path, allow_pickle=False)
    m = json.loads(str(z["_meta"]))
    sd = SysData(m["name"], datetime.fromisoformat(m["init"]), m["n_nominal"], z["_lat"], z["_lon"])
    sd.failures = m["failures"]
    for key in z.files:
        if key.startswith("_"):
            continue
        k, L = key.rsplit("|", 1)
        sd.fields[k][int(L)] = z[key]
    return sd


# ---------------------------------------------------------------------------
def target_grid(bbox: dict, res: float = 0.25):
    lat = np.arange(bbox["lat_min"], bbox["lat_max"] + 1e-6, res)
    lon = np.arange(bbox["lon_min"], bbox["lon_max"] + 1e-6, res)
    return lat, lon


def regrid(vals, lat_src, lon_src, lat_dst, lon_dst):
    if vals.shape == (len(lat_dst), len(lon_dst)) and np.allclose(lat_src, lat_dst) and np.allclose(lon_src, lon_dst):
        return vals
    f = RegularGridInterpolator((lat_src, lon_src), vals, method="linear", bounds_error=False, fill_value=None)
    yy, xx = np.meshgrid(lat_dst, lon_dst, indexing="ij")
    return f(np.stack([yy.ravel(), xx.ravel()], -1)).reshape(len(lat_dst), len(lon_dst)).astype(np.float32)


def _to_mm(msg_vals, units: str):
    return msg_vals * 1000.0 if units.strip() == "m" or units.startswith("m of water") else msg_vals


def tloft_from_levels(tl: dict[int, np.ndarray], sp: np.ndarray, t2m: np.ndarray | None = None):
    """Max temperature between ~610 m AGL and 400 hPa from available pressure levels.

    A level is used only where it lies above 610 m AGL (p <= sp*exp(-610/H)); this
    removes below-ground (extrapolated) levels over high terrain.  Where no level
    qualifies we fall back to the warmest available level.
    """
    p_cut = sp / 100.0 * np.exp(-610.0 / SCALE_H)
    stack, valid = [], []
    for lev in sorted(tl):
        stack.append(tl[lev])
        valid.append(np.broadcast_to(lev <= p_cut, tl[lev].shape))
    stack, valid = np.stack(stack), np.stack(valid)
    masked = np.where(valid, stack, -np.inf).max(axis=0)
    fallback = stack.max(axis=0)
    return np.where(np.isfinite(masked), masked, fallback).astype(np.float32)


# ---------------------------------------------------------------------------
# generic job runner
# ---------------------------------------------------------------------------
def _run_jobs(jobs, workers):
    """jobs: list of callables returning a result or None.  Returns list of results."""
    out = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job") as ex:
        for r in ex.map(lambda j: j(), jobs):
            out.append(r)
    return out


def _stack(per_member: dict[int, np.ndarray], n: int):
    """Dict {member_idx: (ny,nx)} -> (m, ny, nx) of members that arrived (sorted)."""
    if not per_member:
        return None
    idx = sorted(per_member)
    return np.stack([per_member[i] for i in idx]).astype(np.float32)


def _subset(n_total: int, k: int | None):
    if not k or k >= n_total:
        return list(range(n_total))
    return sorted({int(round(i)) for i in np.linspace(0, n_total - 1, k)})


# ---------------------------------------------------------------------------
# GEFS
# ---------------------------------------------------------------------------
def load_gefs(fx: Fetcher, init, steps_msl, steps_pcp, want_snow, bbox, lat, lon, workers=8, off=0):
    sd = SysData("gefs", init, len(S.GEFS_MEMBERS), lat, lon)
    sel_msl = Sel(("PRMSL",), ("mean sea level",))
    sel_pcp = Sel(("APCP",), ("surface",), 6)
    sel_typ = Sel(("CSNOW", "CRAIN", "CFRZR", "CICEP"), ("surface",), 6)
    sel_sp = Sel(("PRES",), ("surface",))
    sel_t = Sel(("TMP",), tuple(f"{p} mb" for p in T_LEVELS))
    raw = defaultdict(lambda: defaultdict(dict))  # step -> field -> {mi: arr}

    def job_sp25(mi, m, step):
        def run():
            sels = []
            if step in steps_msl:
                sels.append(sel_msl)
            if step in steps_pcp:
                sels += [sel_pcp] + ([sel_typ] if want_snow else [])
                if mi == 0 and want_snow:
                    sels.append(sel_sp)
            if not sels:
                return
            try:
                blobs, tier = fx.get("gefs", S.gefs_file(init, m, step, "sp25"), AnySel(*sels))
            except AllTiersFailed as e:
                sd.failures += 1
                log.warning("%s", e)
                return
            for g in decode(blobs, bbox=bbox):
                a = regrid(g.values, g.lat, g.lon, lat, lon)
                _put(raw, step, g.short, mi, a)
        return run

    def job_ap5(mi, m, step):
        def run():
            try:
                blobs, tier = fx.get("gefs", S.gefs_file(init, m, step, "ap5"), sel_t)
            except AllTiersFailed as e:
                sd.failures += 1
                log.warning("%s", e)
                return
            for g in decode(blobs, bbox=bbox):
                a = regrid(g.values, g.lat, g.lon, lat, lon)
                _put(raw, step, f"t{g.level}", mi, a)
        return run

    jobs = []
    for step in sorted(set(steps_msl) | set(steps_pcp)):
        for mi, m in enumerate(S.GEFS_MEMBERS):
            jobs.append(job_sp25(mi, m, step))
            if want_snow and step in steps_pcp:
                jobs.append(job_ap5(mi, m, step))
    _run_jobs(jobs, workers)

    for step, f in raw.items():
        lead = step - off
        if "prmsl" in f:
            sd.fields["msl"][lead] = _stack(f["prmsl"], 31)
        if "tp" in f:
            pcp = _stack(f["tp"], 31)
            sd.fields["pcp6"][lead] = np.clip(pcp, 0, None)
            if want_snow and "csnow" in f:
                members = sorted(set(f["tp"]) & set(f["csnow"]) & set(f["crain"]) & set(f["cfrzr"]) & set(f["cicep"]))
                if members:
                    fr = np.stack([np.stack([f[k][i] for i in members]) for k in ("crain", "csnow", "cfrzr", "cicep")])
                    tp = np.stack([f["tp"][i] for i in members])
                    # fractions are 6-h averages of the categorical flags -> weight precip by snow fraction
                    snow_frac = np.clip(fr[1], 0, 1)
                    sd.fields["snow6"][lead] = np.clip(tp * snow_frac, 0, None).astype(np.float32)
                    # dominant type, ties resolved in favour of the hazardous type
                    prio = np.array([RAIN, SNOW, FRZR, SLEET], dtype=np.int8)
                    score = fr + np.array([0, 0.002, 0.004, 0.003])[:, None, None, None]  # tie-break
                    typ = prio[np.argmax(score, axis=0)]
                    typ = np.where((fr.max(axis=0) > 0) & (tp >= PCP_MIN_MM), typ, NONE).astype(np.int8)
                    sd.fields["ptype"][lead] = typ
                lv = [p for p in T_LEVELS if f.get(f"t{p}")]
                if "sp" in f and lv:
                    sp = f["sp"][0]
                    common = sorted(set.intersection(*[set(f[f"t{p}"]) for p in lv]))
                    if common:
                        sd.fields["tloft"][lead] = np.stack([
                            tloft_from_levels({p: f[f"t{p}"][m] for p in lv}, sp) for m in common])
    return sd


# ---------------------------------------------------------------------------
# ECMWF IFS ENS  (50 perturbed members, cumulative tp / sf in metres)
# ---------------------------------------------------------------------------
def load_ifs(fx: Fetcher, init, steps_msl, steps_pcp, want_snow, bbox, lat, lon, workers=6, off=0,
             profile_members=10):
    sd = SysData("ifs", init, 50, lat, lon)
    nums = [str(i) for i in range(1, 51)]
    prof = [nums[i] for i in _subset(50, profile_members)]
    # cumulative sources need the step *before* each window: steps_pcp already includes `off`
    raw = defaultdict(lambda: defaultdict(dict))

    def job(step):
        def run():
            sels = []
            if step in steps_msl:
                sels.append(ESel(("msl",), "sfc", None, tuple(nums)))
            if step in steps_pcp:
                params = ("tp",) + (("sf", "ptype") if want_snow else ())
                sels.append(ESel(params, "sfc", None, tuple(nums)))
                if want_snow:
                    sels.append(ESel(("sp",), "sfc", None, (prof[0],)))
                    sels.append(ESel(("t",), "pl", tuple(str(p) for p in T_LEVELS), tuple(prof)))
            try:
                blobs, tier = fx.get("ifs", S.ifs_ens_file(init, step), AnySel(*sels))
            except AllTiersFailed as e:
                sd.failures += 1
                log.warning("%s", e)
                return
            for g in decode(blobs, bbox=bbox):
                key = g.short if g.type_of_level != "isobaricInhPa" else f"t{g.level}"
                v = g.values
                if key in ("tp", "sf"):
                    v = v * 1000.0  # m (of water equivalent) -> mm
                _put(raw, step, key, int(g.number), v)
        return run

    _run_jobs([job(s) for s in sorted(set(steps_msl) | set(steps_pcp))], workers)
    _finish_ecmwf(sd, raw, off, steps_pcp, want_snow, n_nominal=50, ptype_map=_IFS_PTYPE, prof_n=len(prof))
    return sd


# ---------------------------------------------------------------------------
# ECMWF AIFS ENS (control + 50 members, cumulative tp / sf already in kg m-2)
# ---------------------------------------------------------------------------
def load_aifs(fx: Fetcher, init, steps_msl, steps_pcp, want_snow, bbox, lat, lon, workers=6, off=0,
              profile_members=10):
    sd = SysData("aifs", init, 51, lat, lon)
    nums = [str(i) for i in range(1, 51)]
    prof = [nums[i] for i in _subset(50, profile_members)]
    raw = defaultdict(lambda: defaultdict(dict))

    def job(step, kind):
        def run():
            ns = None if kind == "cf" else tuple(nums)  # control index carries no "number"
            pn = None if kind == "cf" else tuple(prof)
            sels = []
            if step in steps_msl:
                sels.append(ESel(("msl",), "sfc", None, ns))
            if step in steps_pcp:
                params = ("tp",) + (("sf",) if want_snow else ())
                sels.append(ESel(params, "sfc", None, ns))
                if want_snow:
                    sels.append(ESel(("sp",), "sfc", None, None if pn is None else pn[:1]))
                    sels.append(ESel(("t",), "pl", tuple(str(p) for p in T_LEVELS), pn))
            if not sels:
                return
            try:
                blobs, tier = fx.get("aifs", S.aifs_ens_file(init, step, kind), AnySel(*sels))
            except AllTiersFailed as e:
                sd.failures += 1
                log.warning("%s", e)
                return
            for g in decode(blobs, bbox=bbox):
                key = g.short if g.type_of_level != "isobaricInhPa" else f"t{g.level}"
                _put(raw, step, key, int(g.number), g.values)  # AIFS tp/sf are kg m-2 (= mm) already
        return run

    jobs = [job(s, k) for s in sorted(set(steps_msl) | set(steps_pcp)) for k in ("cf", "pf")]
    _run_jobs(jobs, workers)
    _finish_ecmwf(sd, raw, off, steps_pcp, want_snow, n_nominal=51, ptype_map=None, prof_n=len(prof) + 1)
    return sd


def _finish_ecmwf(sd, raw, off, steps_pcp, want_snow, n_nominal, ptype_map, prof_n):
    steps = sorted(raw)
    for step in steps:
        f = raw[step]
        lead = step - off
        if "msl" in f:
            sd.fields["msl"][lead] = _stack(f["msl"], n_nominal)
    pcp_steps = sorted(s for s in steps_pcp if s in raw and "tp" in raw[s])
    for a, b in zip(pcp_steps[:-1], pcp_steps[1:]):
        if b - a != 6:
            continue
        fa, fb = raw[a], raw[b]
        mem = sorted(set(fa.get("tp", {})) & set(fb.get("tp", {})))
        if not mem:
            continue
        lead = b - off
        tp6 = np.stack([fb["tp"][m] - fa["tp"][m] for m in mem])
        sd.fields["pcp6"][lead] = np.clip(tp6, 0, None).astype(np.float32)
        if want_snow and fa.get("sf") and fb.get("sf"):
            sm = sorted(set(mem) & set(fa["sf"]) & set(fb["sf"]))
            if sm:
                sd.fields["snow6"][lead] = np.clip(np.stack([fb["sf"][m] - fa["sf"][m] for m in sm]), 0, None).astype(np.float32)
        if want_snow and ptype_map is not None and fb.get("ptype"):
            pm = sorted(set(mem) & set(fb["ptype"]))
            if pm:
                code = np.stack([fb["ptype"][m] for m in pm])
                code = np.nan_to_num(code, nan=0).astype(int).clip(0, 31)
                typ = ptype_map[code]
                tp_pm = np.stack([fb["tp"][m] - fa["tp"][m] for m in pm])
                sd.fields["ptype"][lead] = np.where(tp_pm >= PCP_MIN_MM, typ, NONE).astype(np.int8)
        if want_snow and fb.get("sp"):
            sp = next(iter(fb["sp"].values()))
            lv = [p for p in T_LEVELS if fb.get(f"t{p}")]
            if lv:
                members = sorted(set.intersection(*[set(fb[f"t{p}"]) for p in lv]))
                if members:
                    sd.fields["tloft"][lead] = np.stack([
                        tloft_from_levels({p: fb[f"t{p}"][m] for p in lv}, sp) for m in members])


# ---------------------------------------------------------------------------
# NOAA AIGEFS (31 members, NOMADS) and AIGFS (deterministic)
# ---------------------------------------------------------------------------
def load_aigefs(fx: Fetcher, init, steps_msl, steps_pcp, want_snow, bbox, lat, lon, workers=2, off=0):
    sd = SysData("aigefs", init, 31, lat, lon)
    raw = defaultdict(lambda: defaultdict(dict))
    sel_msl, sel_pcp = Sel(("PRMSL",), ("mean sea level",)), Sel(("APCP",), ("surface",), 6)

    def job(mi, m, step):
        def run():
            sels = ([sel_msl] if step in steps_msl else []) + ([sel_pcp] if step in steps_pcp and step > off else [])
            if not sels:
                return
            try:
                blobs, tier = fx.get("aigefs", S.aigefs_file(init, m, step, "sfc"), AnySel(*sels))
            except AllTiersFailed as e:
                sd.failures += 1
                log.warning("%s", e)
                return
            for g in decode(blobs, bbox=bbox):
                _put(raw, step, g.short, mi, g.values)
        return run

    jobs = [job(mi, m, s) for s in sorted(set(steps_msl) | set(steps_pcp)) for mi, m in enumerate(S.AIGEFS_MEMBERS)]
    _run_jobs(jobs, workers)
    for step, f in raw.items():
        lead = step - off
        if "prmsl" in f:
            sd.fields["msl"][lead] = _stack(f["prmsl"], 31)
        if "tp" in f:
            sd.fields["pcp6"][lead] = np.clip(_stack(f["tp"], 31), 0, None)
    return sd


def load_aigfs(fx: Fetcher, init, steps_msl, steps_pcp, want_snow, bbox, lat, lon, workers=2, off=0):
    sd = SysData("aigfs", init, 1, lat, lon)
    sel_msl, sel_pcp = Sel(("PRMSL",), ("mean sea level",)), Sel(("APCP",), ("surface",), 6)
    for step in sorted(set(steps_msl) | set(steps_pcp)):
        sels = ([sel_msl] if step in steps_msl else []) + ([sel_pcp] if step in steps_pcp and step > off else [])
        try:
            blobs, _ = fx.get("aigfs", S.aigfs_file(init, step, "sfc"), AnySel(*sels))
        except AllTiersFailed as e:
            sd.failures += 1
            log.warning("%s", e)
            continue
        for g in decode(blobs, bbox=bbox):
            if g.short == "prmsl":
                sd.fields["msl"][step - off] = g.values[None]
            elif g.short == "tp":
                sd.fields["pcp6"][step - off] = np.clip(g.values[None], 0, None)
    return sd


LOADERS = {"gefs": load_gefs, "ifs": load_ifs, "aifs": load_aifs, "aigefs": load_aigefs, "aigfs": load_aigfs}
