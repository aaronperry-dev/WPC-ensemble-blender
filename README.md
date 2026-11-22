# Ensemble Blender - QPF, snowfall, precipitation type and surface-low products

Multi-system ensemble products on a basemap, built from open data:

| Group | Product | Method |
|---|---|---|
| `lows` | **L** symbols, cluster pressure and member support, mean MSLP isobars | h-minima per member, clustering |
| `qpf6` `qpf24` `qpf_total` | 6-h, 24-h and storm-total QPF (in) | weighted probability-matched mean |
| `qpf_prob` | P(storm-total QPF >= 0.25 / 1 / 2 in) | weighted member fraction |
| `snow24_*` `snow_total_*` | 24-h and total snowfall, a fixed **climatological 12.5:1** ratio (configurable) and the **NBM MaxTaloft** ratio | SWE x SLR per 6-h window |
| `snow_prob` | P(total snow >= 2 / 6 / 12 in) | weighted member fraction |
| `ptype` | Most likely precipitation type, opacity = agreement | weighted member vote |

Colour ramps: **QPF and snowfall use the NWS forecast-graphic ramps** (colours and class breaks sampled from the NWS Gray/Portland
"Total Precipitation" and "Total Snow" legends); probabilities use ColorBrewer RdPu and precipitation type ColorBrewer Paired. Basemap is Natural Earth.
`products.html` (copied to the Pages site by the workflow) is a phone-friendly viewer (product picker, slider, play).

```
pip install -r requirements.txt
python eb_run.py --dry-run          # pick the cycle, print the plan, download nothing
python eb_run.py                    # build (skips if this cycle is already built; --force to redo)
python eb_run.py --lead 48 --precip-lead 48 --only qpf,lows   # quick test
python -m pytest tests          # unit tests (no network)
```


### Low-cluster composite (WPC style)

`lows_track.png` follows the layout of WPC's "Automated Low Clusters": every member low at 12-hourly valid times out to
72 h is a dot coloured (and shaped) by valid time, the cluster-mean low at each time is a black **L**, the legend sits
lower-left and an init/valid/through block lower-right.  Following WPC, member lows with MSLP above 1000 hPa are
dropped (`lows.track_max_hpa`).  Differences from WPC, on purpose: clusters come from average-linkage distance
clustering (450 km) rather than a fixed-k k-means, so the number of clusters is not preset; support (share of weighted
members) is stored in `manifest.json`; the 1500 m altitude filter is not applied because the configured eastern
domains contain no model terrain that high (add an orography mask before using a western domain); and the graphic is
labelled as not an official NWS product.  The per-lead `lows_fNNN.png` maps with MSLP contours are kept.

## Where the data comes from (tier order)

Every file is tried in this order and the first tier that has it wins; a tier that is
unhealthy (6 consecutive failures) or never carries a product (3 consecutive 404s, e.g. AIGEFS on S3)
is skipped for the rest of the run.

| System | 1. AWS S3 | 2. GCP | 3. Agency |
|---|---|---|---|
| GEFS 0.25 deg (31) | `noaa-gefs-pds` | `gfs-ensemble-forecast-system` | NOMADS (byte range, then **GRIB filter** with server-side sub-region) |
| ECMWF IFS ENS 0.25 deg (50) | `ecmwf-forecasts` | `ecmwf-open-data` | `data.ecmwf.int` |
| ECMWF AIFS ENS 0.25 deg (51) | `ecmwf-forecasts` | `ecmwf-open-data` | `data.ecmwf.int` |
| NOAA AIGEFS (31) | not published there (probed, skipped) | - | NOMADS |
| NOAA AIGFS (1, optional) | `noaa-nws-graphcastgfs-pds` | - | NOMADS |

Verified live on 2026-11-21: all AWS layouts, NOMADS byte ranges for GEFS/AIGEFS, the NOMADS GRIB filter for
GEFS, every `.idx`/`.index` format, units and the GRIB2 precipitation-type codes.
**Not reachable from the build sandbox, so not tested:** the two GCP buckets and `data.ecmwf.int`
(names/paths come from the providers' documentation and use the same relative path as AWS). Any failure there just
falls through to the next tier, but check `manifest.json -> tiers_used` after the first run on GitHub Actions.
CMC GEPS is **not** included: the MSC Datamart hosts were blocked in the sandbox and I could not verify the file layout.

### Politeness and speed

* **Byte ranges only.** The `.idx` (NCEP) or `.index` (ECMWF JSON) is read first; only wanted messages are
  requested, adjacent ones are merged (S3 has no multi-range GET) and up to 64 KB of unwanted neighbours may be
  swallowed to save a request. No whole GRIB file is ever downloaded; a server that ignores `Range` on a large object is refused.
* **Per-host gates** (`eb_http.py`): NOMADS and `data.ecmwf.int` max 2 in flight and request starts >= 0.5-0.6 s apart
  (about 100 req/min, under NOMADS' ~120/min guidance); cloud buckets 8 in flight. Exponential backoff with jitter,
  `Retry-After` honoured, identifying User-Agent (**set your contact in `config.yaml`**), no retry on 403/404.
* **Slow hosts overlap fast ones:** the four systems load concurrently, so the rate-limited NOMADS leg (AIGEFS)
  runs while S3 serves the others.
* **Don't repeat work:** a cycle that is already built exits immediately; decoded fields are cached per
  system/cycle/step set (the workflow persists the cache).

## Methods (and why)

* **Weights.** Each system has a total weight; each member gets weight / member-count, renormalised over what is available.
  Starting values in `config.yaml` are *not skill-fitted*; tune them against verification before trusting fine differences.
* **Cycle choice.** Newest init time at which >= 3 systems are available; others may be time-lagged up to 12 h
  (valid times are aligned, accumulations are differenced from each system's own start). Lagged systems are labelled in the manifest.
* **QPF.** The plain ensemble mean smooths away amplitude, so maps show the **probability-matched mean** (Ebert 2001,
  *Mon. Wea. Rev.* 129, 2461-2480): the mean's spatial pattern with the pooled members' amplitude distribution. Caveat from the literature
  (Clark 2017): PMM computed over a large domain redistributes amounts across it, so read it as a *typical-amplitude* field, not a point forecast;
  use the probability maps for decisions. Exceedance maps are weighted member fractions (uncalibrated).
* **Snowfall = SWE x SLR, per 6-h window, then summed.** SWE is each model's own snowfall (IFS `sf`, AIFS `sf`) or, for GEFS,
  `APCP x CSNOW` (6-h average snow flag). Two ratios are shown on purpose: a fixed climatological **12.5:1** (`snow_climo_slr`; 10:1 is deliberately not used because Northeast climatology runs nearer 12-13:1), and the NBM **MaxTaloft** polynomial of the
  maximum temperature between 610 m AGL and 400 hPa (coefficients from Veals et al. 2025; the layer maximum is approximated from
  925/850/700 hPa, keeping only levels above 610 m AGL so below-ground levels over high terrain are ignored).
  Published verification finds SLR choice matters as much as model differences and no single method wins, so compare the two rather than trusting one. Where temperature profiles are missing the climatological ratio is the fallback.
  IFS/AIFS temperature profiles are downloaded for 10 evenly spaced members and the mean SLR is applied to all members
  (`profile_members: all` for per-member). The Kuchera ratio is not implemented: I could not verify its constants from a primary source.
* **Precipitation type.** IFS `ptype` (code table 4.201; ECMWF tracks type through the model integration) and GEFS categorical flags
  (rain/snow/freezing rain/ice pellets). Per-member type -> weighted vote -> most likely type, drawn where >= 50% of members have >= 0.01 in in the window.
  *Semantics differ:* IFS `ptype` is instantaneous at the valid time, GEFS flags are 6-h averages. AIFS, AIGEFS and AIGFS publish no type, so they are excluded from this product
  (and AIGEFS/AIGFS from snowfall); they still count in QPF and the L maps.
* **Lows.** Per member, a low is a regional MSLP minimum with >= 2 hPa dynamic (h-minima = "at least one closed 2-hPa isobar"),
  after light smoothing; minima within 3 grid cells of the edge are dropped. Member lows are clustered (average linkage, 450 km);
  a cluster is drawn if >= 20% of the weighted members have a low in it. **MSLP over high terrain is a reduction artefact**, so lee-side/mountain "lows" there deserve skepticism.

## Cost

Measured on 2026-11-22 (4 systems, 24 h lead): ~1.3 GB and ~5 min, of which the NOMADS leg (AIGEFS, 31 members, rate-limited) took ~3 min.
The default (120 h MSLP, 96 h precipitation) is roughly 3-4 GB per run - most of it IFS/AIFS member fields, which ECMWF only
publishes global. Levers: `precip_lead_hours`, `lead_hours`, `profile_members`, disabling a system, or running twice a day
instead of four times (edit the cron).

## Not done / next steps

* No verification against observations; weights, the 0.01 in type threshold and the 450 km cluster radius are defensible defaults, not tuned.
* GEPS, calibrated (bias-corrected) QPF, and terrain-aware low masking are the obvious next additions.

References: Ebert 2001 (PMM); Clark 2017 (PMM caveats, *Wea. Forecasting*); Veals et al. 2025, *Predicting snow-to-liquid ratio in the mountains of the western
United States* (NBM MaxTaloft/Cobb/Roebber; SLR error sensitivity); Baxter et al. 2005 (SLR climatology); Brewer & Harrower, ColorBrewer 2.0; NWS Gray/Portland snow and precipitation forecast legends; Natural Earth.


## Files (flat on purpose)

Everything sits in the repo root so it can be uploaded from a phone without creating folders; the only folder is
`.github/workflows/` (GitHub requires that path for Actions).

| File | Role |
|---|---|
| `.github/workflows/blender-products.yml` | scheduled build + Pages deploy |
| `config.yaml` | domains, weights, thresholds, politeness limits, `snow_climo_slr` |
| `eb_run.py` | entry point / orchestrator (`python eb_run.py`) |
| `eb_http.py`, `eb_fetch.py`, `eb_index.py`, `eb_sources.py` | rate-limited, byte-range, tiered data access |
| `eb_grib.py`, `eb_systems.py` | GRIB decode and per-system loaders |
| `eb_blend.py`, `eb_snow.py`, `eb_lows.py` | blending, snowfall ratios, low detection/clustering |
| `eb_plot.py` | maps (NWS / ColorBrewer ramps, Natural Earth basemap) |
| `products.html` | phone viewer |
| `test_core.py` | tests (`pytest`) |
| `requirements.txt`, `README.md`, `.gitignore` | housekeeping |
