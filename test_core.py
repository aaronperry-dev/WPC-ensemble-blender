"""Unit tests for the pieces whose correctness does not depend on live data."""
import http.server
import threading
import time

import numpy as np
import pytest

import eb_blend as blend
import eb_lows as lows
import eb_snow as snow
from eb_http import Http, HostPolicy, NotFound
from eb_index import ESel, Sel, coalesce, fcst_window, parse_ecmwf_index, parse_ncep_idx
from eb_systems import _IFS_PTYPE, FRZR, MIX, RAIN, SLEET, SNOW, tloft_from_levels
import eb_plot as plot


# ---------------------------------------------------------------- index / ranges
IDX = """1:0:d=2026112100:PRMSL:mean sea level:24 hour fcst:ENS=+1
2:1000:d=2026112100:APCP:surface:18-24 hour acc fcst:ENS=+1
3:1300:d=2026112100:CSNOW:surface:18-24 hour ave fcst:ENS=+1
4:1400:d=2026112100:TMP:850 mb:24 hour fcst:ENS=+1
"""


def test_ncep_idx_ranges_and_windows():
    r = parse_ncep_idx(IDX)
    assert [(x.start, x.end) for x in r] == [(0, 999), (1000, 1299), (1300, 1399), (1400, None)]
    assert r[1].meta["window"] == (18, 24, "acc")
    assert fcst_window("0-1 day acc fcst") == (0, 24, "acc")
    assert fcst_window("24 hour fcst") == (24, 24, "inst")


def test_selectors():
    r = parse_ncep_idx(IDX)
    assert [x.meta["var"] for x in r if Sel(("APCP", "CSNOW"), ("surface",), 6).match(x.meta)] == ["APCP", "CSNOW"]
    assert not any(Sel(("APCP",), ("surface",), 3).match(x.meta) for x in r)  # wrong window length
    j = parse_ecmwf_index('{"param":"tp","levtype":"sfc","number":"3","_offset":10,"_length":5}\n'
                          '{"param":"t","levtype":"pl","levelist":"850","number":"3","_offset":15,"_length":5}\n')
    assert (j[0].start, j[0].end) == (10, 14)
    assert ESel(("tp",), "sfc", None, ("3",)).match(j[0].meta)
    assert not ESel(("tp",), "sfc", None, ("4",)).match(j[0].meta)
    assert ESel(("t",), "pl", ("850",)).match(j[1].meta)


def test_coalesce():
    assert coalesce([(0, 9), (10, 19), (40, 49)]) == [(0, 19), (40, 49)]
    assert coalesce([(0, 9), (20, 29)], max_gap=10) == [(0, 29)]
    assert coalesce([(0, 9), (10, None)]) == [(0, None)]


# ---------------------------------------------------------------- statistics
def test_combine_weights_normalise_by_member_count():
    a, b = np.zeros((50, 2, 2), np.float32), np.ones((10, 2, 2), np.float32)
    arr, w = blend.combine([(0.5, a), (0.5, b)])
    assert arr.shape[0] == 60 and np.isclose(w.sum(), 1)
    assert np.isclose(w[:50].sum(), 0.5) and np.isclose(w[50:].sum(), 0.5)  # system totals, not member counts
    assert np.allclose(blend.wmean(arr, w), 0.5)


def test_exceed_prob_weighted():
    arr = np.array([0, 1, 2, 3], np.float32).reshape(4, 1, 1)
    w = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.isclose(blend.exceed_prob(arr, w, 2)[0, 0], 0.7)


def test_weighted_percentile_matches_numpy_for_equal_weights():
    rng = np.random.default_rng(0)
    arr = rng.gamma(2, 1, size=(101, 5, 5)).astype(np.float32)
    w = np.full(101, 1 / 101)
    got = blend.wpercentile(arr, w, 50)
    assert np.allclose(got, np.median(arr, axis=0), atol=1e-4)


def test_pmm_keeps_pattern_and_pooled_distribution():
    rng = np.random.default_rng(1)
    ny, nx, m = 30, 30, 20
    yy, xx = np.mgrid[:ny, :nx]
    members = []
    for _ in range(m):  # same storm, displaced randomly -> smooth mean, sharp members
        cy, cx = 15 + rng.normal(0, 4), 15 + rng.normal(0, 4)
        members.append(40 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 20.0))
    arr = np.stack(members).astype(np.float32)
    w = np.full(m, 1 / m)
    mean, pmm = blend.wmean(arr, w), blend.prob_matched_mean(arr, w)
    assert pmm.max() > mean.max() * 1.5  # amplitude restored
    assert np.isclose(pmm.max(), arr.max(), rtol=0.05)
    # pattern preserved: PMM is a monotone (non-decreasing) function of the ensemble mean
    o = np.argsort(mean.ravel(), kind="stable")
    assert np.all(np.diff(pmm.ravel()[o]) >= -1e-4)
    # domain-mean of PMM ~ pooled mean of all member values (mass conserved)
    assert np.isclose(pmm.mean(), arr.mean(), rtol=0.03)
    # dry points stay dry
    z = np.zeros((4, 3, 3), np.float32)
    assert blend.prob_matched_mean(z, np.full(4, .25)).max() == 0


# ---------------------------------------------------------------- snow
def test_maxtaloft_shape():
    t = np.array([-25, -15, -10, -5, -2, 0, 1, 3]) + 273.15
    s = snow.slr_maxtaloft(t.astype(np.float32))
    assert s.max() <= snow.SLR_MAX and s.min() >= snow.SLR_MIN
    assert 18 < s[1] < 23          # peak near -15 C (dendritic growth zone)
    assert s[3] < s[1]              # warmer aloft -> denser snow
    assert 8 < snow.slr_maxtaloft(np.float32(271.15)) < 11  # about -2 C -> near 10:1
    assert s[6] < s[5] < s[4]       # monotone drop toward melting


def test_snow_depth_sums_windows():
    swe = [np.full((3, 2, 2), 2.0, np.float32)] * 2  # 2 mm per window
    d = snow.snow_depth_mm(swe, [None, None], "climo", 12.5)
    assert np.allclose(d, 50.0)  # 2 windows x 2 mm x 12.5
    assert np.allclose(snow.snow_depth_mm(swe, [None, None], "climo"), 4 * snow.SLR_CLIMO)
    assert snow.SLR_CLIMO > 10  # never fall back to 10:1
    tl = np.full((1, 2, 2), 263.15, np.float32)  # -10 C everywhere, 1 profile for 3 members -> shared mean SLR
    d2 = snow.snow_depth_mm(swe, [tl, tl], "maxtaloft")
    assert d2.shape == (3, 2, 2) and d2[0, 0, 0] > 40  # cold column -> ratio above 10


def test_tloft_ignores_underground_levels():
    # surface pressure 850 hPa (high terrain): 925 hPa is underground and absurdly warm
    sp = np.full((1, 1), 85000.0, np.float32)
    levels = {925: np.full((1, 1), 290.0, np.float32), 850: np.full((1, 1), 270.0, np.float32),
              700: np.full((1, 1), 265.0, np.float32)}
    assert tloft_from_levels(levels, sp)[0, 0] == 265.0  # 850 is within 610 m AGL too -> excluded
    sp2 = np.full((1, 1), 101300.0, np.float32)
    assert tloft_from_levels(levels, sp2)[0, 0] == 290.0  # sea level: 925 hPa is ~760 m AGL -> counts


def test_ptype_table():
    assert _IFS_PTYPE[1] == RAIN and _IFS_PTYPE[3] == FRZR and _IFS_PTYPE[5] == SNOW
    assert _IFS_PTYPE[7] == MIX and _IFS_PTYPE[8] == SLEET and _IFS_PTYPE[0] == 0


# ---------------------------------------------------------------- lows
def _gauss_low(ny, nx, cy, cx, depth, width, base=101500.0):
    yy, xx = np.mgrid[:ny, :nx]
    return base - depth * 100 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * width ** 2))


def test_detect_lows_finds_closed_lows_only():
    lat, lon = np.linspace(33, 50, 69), np.linspace(-84, -58, 105)
    f = _gauss_low(69, 105, 30, 60, 25, 6)  # 25 hPa low
    f -= 100 * 0.4 * np.exp(-((np.mgrid[:69, :105][0] - 50) ** 2 + (np.mgrid[:69, :105][1] - 20) ** 2) / 6.0)  # 0.4 hPa dimple
    found = lows.detect_lows(f.astype(np.float32), lat, lon)
    assert len(found) == 1
    assert abs(found[0].lat - lat[30]) < 0.3 and abs(found[0].lon - lon[60]) < 0.3
    assert 1074 < found[0].p_hpa < 1078 or 988 < found[0].p_hpa < 992  # 1015 - 25 = 990


def test_detect_lows_drops_edge_minima():
    lat, lon = np.linspace(33, 50, 69), np.linspace(-84, -58, 105)
    assert lows.detect_lows(_gauss_low(69, 105, 1, 50, 20, 5).astype(np.float32), lat, lon) == []


def test_cluster_support_and_weights():
    A = [lows.Low(40.0, -70.0, 990.0)]
    B = [lows.Low(40.5, -70.5, 992.0)]
    C = [lows.Low(45.0, -60.0, 1000.0)]  # outlier far away
    members = [A, B, A, C]
    w = np.array([0.3, 0.3, 0.3, 0.1])
    cl = lows.cluster_lows(members, w, eps_km=450, min_support=0.05)
    assert len(cl) == 2
    assert np.isclose(cl[0].support, 0.9) and cl[0].n_members == 3
    assert 990 < cl[0].p_hpa < 992 and np.isclose(cl[1].support, 0.1)
    # min_support filter
    assert len(lows.cluster_lows(members, w, 450, 0.2)) == 1


# ---------------------------------------------------------------- colours
def test_brewer_matches_matplotlib_reference():
    from matplotlib import _cm
    hexes = ["#%02x%02x%02x" % tuple(round(255 * c) for c in rgb) for rgb in _cm._RdPu_data]
    assert hexes == plot.BREWER["RdPu"]
    paired = ["#%02x%02x%02x" % tuple(round(255 * c) for c in rgb) for rgb in _cm._Paired_data]
    assert paired == plot.PAIRED


def test_nws_ramps_are_complete_and_ordered():
    for spec, n_bins in ((plot.NWS_SNOW_IN, 11), (plot.NWS_QPF_IN, 14)):
        lv, co = spec["levels"], spec["colors"]
        assert len(lv) == n_bins + 1 and len(co) == n_bins + 1  # one colour per bin + the "over" colour
        assert lv == sorted(lv) and len(set(co)) == len(co)
    # spot checks against the NWS legend swatches
    assert plot.NWS_SNOW_IN["colors"][0] == "#bdd7e7" and plot.NWS_SNOW_IN["colors"][-1] == "#360000"
    assert plot.NWS_QPF_IN["colors"][4] == "#006d2c" and plot.NWS_QPF_IN["colors"][-1] == "#ffdbff"
    lv, cmap, norm = plot.ramp("snow")
    assert not np.allclose(cmap(norm(0.5)), cmap(norm(1.5)))
    assert np.allclose(cmap(norm(50)), plot.matplotlib.colors.to_rgba("#360000"))  # >= 36 in -> "over" colour
    assert np.allclose(cmap(norm(0.5)), plot.matplotlib.colors.to_rgba("#bdd7e7"))


# ---------------------------------------------------------------- http
class _H(http.server.BaseHTTPRequestHandler):
    DATA = bytes(range(256)) * 40
    hits = 0
    fail_first = 0
    stamps = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        type(self).stamps.append(time.monotonic())
        if self.path == "/missing":
            self.send_response(404); self.end_headers(); return
        if self.path == "/flaky":
            type(self).hits += 1
            if type(self).hits <= type(self).fail_first:
                self.send_response(503); self.send_header("Retry-After", "0"); self.end_headers(); return
        if self.path == "/norange":  # server that ignores Range and has a big body
            self.send_response(200); self.send_header("Content-Length", str(10 ** 8)); self.end_headers(); return
        rng = self.headers.get("Range")
        if rng:
            a, b = rng.split("=")[1].split("-")
            a, b = int(a), int(b) if b else len(self.DATA) - 1
            body = self.DATA[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(self.DATA)}")
        else:
            body = self.DATA
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _H.hits, _H.stamps = 0, []
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_range_exactness_404_and_retry(server):
    c = Http("test", policies=[("", HostPolicy(0.01, 4, (2, 5), 4))])
    assert c.get_range(server + "/x", 100, 199) == _H.DATA[100:200]
    assert c.get_range(server + "/x", 10000, None) == _H.DATA[10000:]
    assert c.get_text(server + "/missing") is None
    _H.fail_first = 2
    assert c.get_range(server + "/flaky", 0, 9) == _H.DATA[:10]  # survived two 503s
    from eb_http import RangeNotSupported
    with pytest.raises(RangeNotSupported):
        c.get_range(server + "/norange", 0, 9)  # refuses to slurp a 100 MB body


def test_rate_limit_spacing_and_concurrency_cap(server):
    c = Http("test", policies=[("", HostPolicy(0.15, 2, (2, 5), 1))])
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda i: c.get_range(server + "/x", 0, 9), range(8)))
    st = sorted(_H.stamps)
    gaps = np.diff(st)
    assert gaps.min() >= 0.13, gaps  # request starts are spaced >= min_interval even with 8 threads


def test_cache_roundtrip_without_pickle(tmp_path):
    from datetime import datetime
    from eb_systems import SysData, load_sysdata, save_sysdata
    sd = SysData("ifs", datetime(2026, 11, 21, 18), 50, np.arange(3.0), np.arange(4.0))
    sd.fields["pcp6"][6] = np.random.rand(5, 3, 4).astype(np.float32)
    sd.fields["msl"][0] = np.random.rand(5, 3, 4).astype(np.float32)
    sd.failures = 2
    save_sysdata(sd, tmp_path / "x.npz")
    r = load_sysdata(tmp_path / "x.npz")
    assert r.name == "ifs" and r.init == sd.init and r.failures == 2 and r.n_nominal == 50
    assert np.array_equal(r.fields["pcp6"][6], sd.fields["pcp6"][6]) and 0 in r.fields["msl"]


def test_filter_lows_pressure():
    from eb_lows import Low, filter_lows
    ml = [[Low(40, -70, 990.0), Low(41, -71, 1004.0)], [Low(42, -72, 1000.0)]]
    out = filter_lows(ml, 1000.0)
    assert [len(x) for x in out] == [1, 1] and out[1][0].p_hpa == 1000.0  # boundary kept
    assert filter_lows(ml, None) is ml


def test_plot_lows_track_renders(tmp_path):
    from datetime import datetime
    from eb_lows import Cluster
    bbox = dict(lon_min=-84, lon_max=-58, lat_min=33, lat_max=50)
    lat, lon = np.linspace(33, 50, 5), np.linspace(-84, -58, 5)
    cl = Cluster(42, -70, 990, 0.6, 30, 80, 3, 985, 996)
    frames = [(0, datetime(2026, 11, 22, 0), np.array([[42.0, -70.5], [41.5, -69.0]]), [cl]),
              (12, datetime(2026, 11, 22, 12), np.array([[43.0, -68.0]]), [cl]),
              (24, datetime(2026, 11, 23, 0), np.empty((0, 2)), [])]
    out = tmp_path / "t.png"
    plot.plot_lows_track(out, bbox, lat, lon, frames, "t", "s", "f", ["a", "b"])
    assert out.stat().st_size > 5000
