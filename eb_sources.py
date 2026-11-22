"""Where every model system lives, per access tier.

Tier order (the user's requirement): cloud object stores first (AWS S3, then
GCP), then the originating agency (NOAA NOMADS incl. GRIB-filter, ECMWF
data.ecmwf.int).  Every tier serves the *same relative path* so one builder
per file produces the full candidate list.

Layouts verified against live listings on 2026-11-21 (AWS S3, NOMADS):
  GEFS   noaa-gefs-pds/gefs.YYYYMMDD/HH/atmos/{pgrb2sp25,pgrb2ap5}/<mem>.tHHz.<..>.fFFF(.idx)
  IFS    ecmwf-forecasts/YYYYMMDD/HHz/ifs/0p25/enfo/<YYYYMMDDHH0000>-<S>h-enfo-ef.{grib2,index}
  AIFS   ecmwf-forecasts/YYYYMMDD/HHz/aifs-ens/0p25/enfo/<..>-<S>h-enfo-{cf,pf}.{grib2,index}
  AIGFS  noaa-nws-graphcastgfs-pds/aigfs.YYYYMMDD/HH/model/atmos/grib2/aigfs.tHHz.{sfc,pres}.fFFF.grib2(.idx)
  AIGEFS nomads .../aigefs/prod/aigefs.YYYYMMDD/HH/mem0NN/model/atmos/grib2/aigefs.tHHz.{sfc,pres}.fFFF.grib2(.idx)
GCP bucket names (gfs-ensemble-forecast-system, ecmwf-open-data) and the
data.ecmwf.int path are taken from the providers' documentation and could not
be reached from the build sandbox; the code treats any tier failure as
"try next tier" so a wrong guess degrades gracefully.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import quote, urlencode

NOMADS = "https://nomads.ncep.noaa.gov"


@dataclass(frozen=True)
class Cand:
    """One candidate location of a file."""
    tier: str
    url: str
    index_kind: str  # 'ncep' | 'ecmwf' | 'none'
    # for NOMADS GRIB-filter fallback
    filter_script: str | None = None
    filter_dir: str | None = None
    filter_file: str | None = None

    @property
    def index_url(self) -> str | None:
        if self.index_kind == "ncep":
            return self.url + ".idx"
        if self.index_kind == "ecmwf":
            return self.url.rsplit(".", 1)[0] + ".index"
        return None

    def filter_url(self, sel, bbox: dict) -> str | None:
        """NOMADS GRIB-filter URL (server-side var/level/sub-region subsetting)."""
        if not self.filter_script:
            return None
        q = {"dir": self.filter_dir, "file": self.filter_file}
        q.update(sel.filter_params())
        q.update({"subregion": "", "toplat": bbox["lat_max"], "bottomlat": bbox["lat_min"],
                  "leftlon": bbox["lon_min"], "rightlon": bbox["lon_max"]})
        return f"{NOMADS}/cgi-bin/{self.filter_script}?" + urlencode(q, quote_via=quote)


def _ymdh(cycle: datetime):
    return cycle.strftime("%Y%m%d"), cycle.strftime("%H")


# ---------------------------------------------------------------------------
# GEFS
# ---------------------------------------------------------------------------
GEFS_MEMBERS = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]


def gefs_file(cycle: datetime, member: str, step: int, product: str) -> list[Cand]:
    """product: 'sp25' (0.25 deg surface set) or 'ap5' (0.5 deg pressure-level set)."""
    ymd, hh = _ymdh(cycle)
    if product == "sp25":
        sub, fn, script = "pgrb2sp25", f"{member}.t{hh}z.pgrb2s.0p25.f{step:03d}", "filter_gefs_atmos_0p25s.pl"
    else:
        sub, fn, script = "pgrb2ap5", f"{member}.t{hh}z.pgrb2a.0p50.f{step:03d}", "filter_gefs_atmos_0p50a.pl"
    rel = f"gefs.{ymd}/{hh}/atmos/{sub}/{fn}"
    return [
        Cand("aws", f"https://noaa-gefs-pds.s3.amazonaws.com/{rel}", "ncep"),
        Cand("gcp", f"https://storage.googleapis.com/gfs-ensemble-forecast-system/{rel}", "ncep"),
        Cand("nomads", f"{NOMADS}/pub/data/nccf/com/gens/prod/{rel}", "ncep",
             filter_script=script, filter_dir=f"/gefs.{ymd}/{hh}/atmos/{sub}", filter_file=fn),
    ]


# ---------------------------------------------------------------------------
# ECMWF open data (IFS ENS, AIFS ENS)
# ---------------------------------------------------------------------------
def _ecmwf_cands(rel: str) -> list[Cand]:
    return [
        Cand("aws", f"https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/{rel}", "ecmwf"),
        Cand("gcp", f"https://storage.googleapis.com/ecmwf-open-data/{rel}", "ecmwf"),
        Cand("ecmwf", f"https://data.ecmwf.int/forecasts/{rel}", "ecmwf"),
    ]


def ifs_ens_file(cycle: datetime, step: int) -> list[Cand]:
    ymd, hh = _ymdh(cycle)
    return _ecmwf_cands(f"{ymd}/{hh}z/ifs/0p25/enfo/{ymd}{hh}0000-{step}h-enfo-ef.grib2")


def aifs_ens_file(cycle: datetime, step: int, kind: str) -> list[Cand]:
    """kind: 'cf' (control) or 'pf' (50 perturbed members)."""
    ymd, hh = _ymdh(cycle)
    return _ecmwf_cands(f"{ymd}/{hh}z/aifs-ens/0p25/enfo/{ymd}{hh}0000-{step}h-enfo-{kind}.grib2")


# ---------------------------------------------------------------------------
# NOAA AI systems
# ---------------------------------------------------------------------------
def aigfs_file(cycle: datetime, step: int, product: str) -> list[Cand]:
    ymd, hh = _ymdh(cycle)
    rel = f"aigfs.{ymd}/{hh}/model/atmos/grib2/aigfs.t{hh}z.{product}.f{step:03d}.grib2"
    return [
        Cand("aws", f"https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/{rel}", "ncep"),
        Cand("nomads", f"{NOMADS}/pub/data/nccf/com/aigfs/prod/{rel}", "ncep"),
    ]


AIGEFS_MEMBERS = [f"mem{i:03d}" for i in range(0, 31)]  # mem000 = control


def aigefs_file(cycle: datetime, member: str, step: int, product: str) -> list[Cand]:
    ymd, hh = _ymdh(cycle)
    rel = f"aigefs.{ymd}/{hh}/{member}/model/atmos/grib2/aigefs.t{hh}z.{product}.f{step:03d}.grib2"
    return [
        # not published on the NODD buckets at time of writing; probed once per run, skipped if absent
        Cand("aws", f"https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/{rel}", "ncep"),
        Cand("nomads", f"{NOMADS}/pub/data/nccf/com/aigefs/prod/{rel}", "ncep"),
    ]
