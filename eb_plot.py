"""Map rendering: Natural Earth basemap + ColorBrewer ramps + 'L' symbols."""
from __future__ import annotations

import logging
import textwrap
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

import cartopy.crs as ccrs
import cartopy.feature as cfeature

log = logging.getLogger("eb.plot")

# ---- NWS forecast-graphic ramps (sampled pixel-exact from the NWS Gray/Portland "Total Snow Forecast" and
# "Total Precipitation Forecast" legends).  Used for QPF and snowfall so maps read like the NWS products.
# Each list = one colour per bin, plus a final colour for "above the last edge".  "0" is left unfilled (basemap shows).
NWS_SNOW_IN = {  # class edges in inches:  <1 | 1-2 | 2-3 | 3-4 | 4-6 | 6-8 | 8-12 | 12-18 | 18-24 | 24-30 | 30-36 | 36+
    "levels": [0.1, 1, 2, 3, 4, 6, 8, 12, 18, 24, 30, 36],
    "colors": ["#bdd7e7", "#6baed6", "#3182bd", "#08519c", "#082694", "#ffff96", "#ffc400", "#ff8700",
               "#db1400", "#9e0000", "#690000", "#360000"],
}
NWS_QPF_IN = {  # 0.01-0.09 | 0.10-0.24 | 0.25-0.49 | 0.50-0.99 | 1.00-1.49 | 1.50-1.99 | 2-2.99 | 3-3.99 | 4-5.99 | 6-7.99 | 8-9.99 | 10-14.99 | 15-19.99 | 20-29.99 | 30+
    "levels": [0.01, 0.10, 0.25, 0.50, 1.00, 1.50, 2.00, 3.00, 4.00, 6.00, 8.00, 10.00, 15.00, 20.00, 30.00],
    "colors": ["#c7e9c0", "#a1d99b", "#74c476", "#31a353", "#006d2c", "#ffff8a", "#ffcc4f", "#fe8d3c",
               "#fc4e2a", "#d61a1c", "#ad0026", "#700026", "#3b0030", "#4c0073", "#ffdbff"],
}

# ---- ColorBrewer 2.0 (Brewer & Harrower): 9-class RdPu for probabilities, 12-class Paired for precip type ----
BREWER = {
    "RdPu": ["#fff7f3", "#fde0dd", "#fcc5c0", "#fa9fb5", "#f768a1", "#dd3497", "#ae017e", "#7a0177", "#49006a"],
}
PAIRED = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00",
          "#cab2d6", "#6a3d9a", "#ffff99", "#b15928"]
PTYPE_COLORS = {1: PAIRED[3], 2: PAIRED[1], 3: PAIRED[5], 4: PAIRED[9], 5: PAIRED[7]}  # rain,snow,frzr,sleet,mix
L_COLOR = "#b2182b"  # ColorBrewer RdBu dark red; WPC convention is red L

PROB_LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90]


def ramp(kind: str):
    """-> (levels, cmap, norm, extend).  Colour i fills [levels[i], levels[i+1]); the last colour is the 'over' colour."""
    if kind == "snow":
        spec = NWS_SNOW_IN
    elif kind in ("qpf6", "qpf24", "qpf_total"):
        spec = NWS_QPF_IN
    else:  # "prob": drop the palest RdPu class so the lowest bin stays visible on the land fill
        cols = BREWER["RdPu"][1:]
        spec = {"levels": PROB_LEVELS, "colors": cols}
    levels, colors = spec["levels"], spec["colors"]
    cmap = ListedColormap(colors[:-1]).with_extremes(over=colors[-1])
    norm = BoundaryNorm(levels, len(levels) - 1)
    return levels, cmap, norm


LAND, OCEAN = "#ececec", "#d6e4f0"


def _basemap(ax, bbox):
    ax.set_extent([bbox["lon_min"], bbox["lon_max"], bbox["lat_min"], bbox["lat_max"]], crs=ccrs.PlateCarree())
    kw = dict(zorder=0)
    try:
        ax.add_feature(cfeature.NaturalEarthFeature("physical", "ocean", "50m"), facecolor=OCEAN, **kw)
        ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "50m"), facecolor=LAND, **kw)
        ax.add_feature(cfeature.NaturalEarthFeature("physical", "lakes", "50m"), facecolor=OCEAN,
                       edgecolor="none", zorder=1)
    except Exception as e:  # offline / blocked -> degrade to plain axes
        log.warning("Natural Earth unavailable (%s); using plain background", e)
        ax.set_facecolor(OCEAN)
    return ax


def _borders(ax):
    """Boundary lines drawn *above* the data so they stay legible."""
    try:
        ax.add_feature(cfeature.NaturalEarthFeature("physical", "coastline", "50m"), facecolor="none",
                       edgecolor="#404040", linewidth=0.7, zorder=6)
        ax.add_feature(cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", "50m"),
                       facecolor="none", edgecolor="#404040", linewidth=0.8, zorder=6)
        ax.add_feature(cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lines", "50m"),
                       facecolor="none", edgecolor="#707070", linewidth=0.45, zorder=6)
    except Exception:
        pass
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#808080", alpha=0.5, linestyle=":",
                      x_inline=False, y_inline=False)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7}


def _figure(bbox, title, subtitle, footer):
    dlon, dlat = bbox["lon_max"] - bbox["lon_min"], bbox["lat_max"] - bbox["lat_min"]
    mid = 0.5 * (bbox["lat_min"] + bbox["lat_max"])
    aspect = dlat / (dlon * np.cos(np.radians(mid)))
    w = 8.0
    h = min(7.2, max(5.0, w * aspect + 1.6))
    fig = plt.figure(figsize=(w, h), dpi=110)
    ax = fig.add_axes([0.07, 0.205, 0.89, 0.665], projection=ccrs.PlateCarree())
    _basemap(ax, bbox)
    fig.text(0.07, 0.955, title, fontsize=11.5, fontweight="bold", va="top")
    fig.text(0.07, 0.915, subtitle, fontsize=8.5, color="#303030", va="top")
    fig.text(0.07, 0.012, textwrap.fill(footer, 150), fontsize=6.2, color="#505050", va="bottom")
    return fig, ax


def _save(fig, path):
    fig.savefig(path, dpi=110, pil_kwargs={"optimize": True})
    plt.close(fig)


def fmt_t(t: datetime):
    return t.strftime("%a %d %b %Y %HZ")


def plot_filled(path, bbox, lat, lon, data, kind, title, subtitle, footer, label, overlay=None):
    """data in the units implied by ``kind`` (inches or percent)."""
    levels, cmap, norm = ramp(kind)
    fig, ax = _figure(bbox, title, subtitle, footer)
    d = np.ma.masked_less(data, levels[0])
    cs = ax.contourf(lon, lat, d, levels=levels, cmap=cmap, norm=norm, extend="max",
                     transform=ccrs.PlateCarree(), zorder=3, alpha=0.9)
    if overlay:
        overlay(ax)
    _borders(ax)
    cax = fig.add_axes([0.14, 0.15, 0.72, 0.02])
    cb = fig.colorbar(cs, cax=cax, orientation="horizontal", ticks=levels)
    cb.ax.set_xticklabels([f"{v:g}" for v in levels])
    cb.ax.tick_params(labelsize=6.5)
    cb.set_label(label, fontsize=7.5)
    _save(fig, path)


def plot_ptype(path, bbox, lat, lon, probs: dict, p_any, title, subtitle, footer, min_any=0.5):
    """probs: {category: probability array}.  Dominant category, opacity stepped by its probability."""
    cats = sorted(probs)
    stack = np.stack([probs[c] for c in cats])
    dom = np.argmax(stack, axis=0)
    pdom = np.take_along_axis(stack, dom[None], axis=0)[0]
    fig, ax = _figure(bbox, title, subtitle, footer)
    for k, c in enumerate(cats):
        m = (dom == k) & (p_any >= min_any) & (pdom > 0)
        for lo, hi, a in ((0.0, 0.5, 0.35), (0.5, 0.7, 0.6), (0.7, 1.01, 0.9)):
            mm = m & (pdom >= lo) & (pdom < hi)
            if mm.any():
                ax.pcolormesh(lon, lat, np.ma.masked_where(~mm, np.ones_like(pdom)),
                              cmap=ListedColormap([PTYPE_COLORS[c]]), shading="nearest", alpha=a,
                              transform=ccrs.PlateCarree(), zorder=3, rasterized=True)
    _borders(ax)
    from eb_systems import PTYPE_NAMES
    handles = [Patch(facecolor=PTYPE_COLORS[c], label=PTYPE_NAMES[c]) for c in cats]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9, title="Most likely type", title_fontsize=7)
    fig.text(0.96, 0.15, "Opacity: <50% / 50-70% / >70%\nof members agree on type", fontsize=6.5, ha="right", va="center")
    _save(fig, path)


def plot_lows(path, bbox, lat, lon, mean_msl_hpa, clusters, member_pts, title, subtitle, footer):
    fig, ax = _figure(bbox, title, subtitle, footer)
    levels = np.arange(900, 1100, 4)
    cs = ax.contour(lon, lat, mean_msl_hpa, levels=levels, colors="#525252", linewidths=0.6,
                    transform=ccrs.PlateCarree(), zorder=3)
    ax.clabel(cs, fmt="%d", fontsize=6, inline=True)
    if len(member_pts):
        ax.scatter(member_pts[:, 1], member_pts[:, 0], s=7, c=L_COLOR, alpha=0.25, linewidths=0,
                   transform=ccrs.PlateCarree(), zorder=4)
    halo = [pe.withStroke(linewidth=2.5, foreground="white")]
    placed = []  # label anchor points already used, to keep text from stacking
    dlat = bbox["lat_max"] - bbox["lat_min"]
    for c in clusters:
        size = 26 if c.support >= 0.5 else (20 if c.support >= 0.35 else 15)
        ax.text(c.lon, c.lat, "L", color=L_COLOR, fontsize=size, fontweight="bold", ha="center", va="center",
                transform=ccrs.PlateCarree(), zorder=8, path_effects=halo, alpha=1.0 if c.support >= 0.35 else 0.75)
        near = [q for q in placed if abs(q[0] - c.lon) < 0.14 * (bbox["lon_max"] - bbox["lon_min"])
                and abs(q[1] - c.lat) < 0.16 * dlat]
        above = any(q[2] == "below" for q in near)  # a neighbour already took the space below -> go above
        off = (0.045 * dlat + size * 0.012)
        ax.text(c.lon, c.lat + (off if above else -off),
                f"{c.p_hpa:.0f} hPa\n{c.support * 100:.0f}% of members", color="#222222", fontsize=6.6,
                ha="center", va="bottom" if above else "top", transform=ccrs.PlateCarree(), zorder=8,
                path_effects=halo)
        placed.append((c.lon, c.lat, "above" if above else "below"))
    _borders(ax)
    fig.text(0.96, 0.15, "L size/opacity grows with the share of members\nthat have a low in that cluster (>=20% shown).\n"
             "Dots: individual member lows", fontsize=6.5, ha="right", va="center")
    _save(fig, path)


# Valid-time colours and marker shapes follow the WPC automated low-cluster graphic convention (cyan, red, green,
# orange, brown, magenta, blue for successive 12-h valid times).  Marker shape is a redundant second cue so the
# map stays readable for colour-vision deficiency and for cyan/green dots on a pale basemap.
TRACK_COLORS = ["#00e5ff", "#ff0000", "#00dd00", "#ff8c00", "#8b4513", "#ff00ff", "#0000ff", "#555555"]
TRACK_MARKERS = ["o", "s", "^", "D", "P", "d", "X", "v"]


def plot_lows_track(path, bbox, lat, lon, frames, title, subtitle, footer, issued_lines):
    """WPC-style composite.  frames = [(lead_h, valid_datetime, member_pts[N,2](lat,lon), clusters), ...].
    Coloured dots = individual member lows at each valid time; black L = cluster-mean low at that time."""
    fig, ax = _figure(bbox, title, subtitle, footer)
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]
    handles = []
    for k, (lead, valid, pts, clusters) in enumerate(frames):
        col, mk = TRACK_COLORS[k % len(TRACK_COLORS)], TRACK_MARKERS[k % len(TRACK_MARKERS)]
        if len(pts):
            ax.scatter(pts[:, 1], pts[:, 0], s=13, c=col, marker=mk, edgecolors="#202020", linewidths=0.3,
                       alpha=0.9, transform=ccrs.PlateCarree(), zorder=4 + 0.01 * k)
        for c in clusters:
            ax.text(c.lon, c.lat, "L", color="black", fontsize=19, fontweight="bold", ha="center", va="center",
                    transform=ccrs.PlateCarree(), zorder=9, path_effects=halo)
            ax.text(c.lon + 0.012 * (bbox["lon_max"] - bbox["lon_min"]), c.lat - 0.035 * (bbox["lat_max"] - bbox["lat_min"]),
                    f"f{lead:d}", color="black", fontsize=5.8, ha="left", va="top", transform=ccrs.PlateCarree(),
                    zorder=9, path_effects=halo)
        handles.append((col, mk, f"f{lead:03d}  valid {valid:%HZ %m/%d/%Y}"))
    _borders(ax)
    # legend, lower left (as in the WPC graphic): coloured text, marker glyph in front
    y0 = 0.035 + 0.034 * (len(handles) - 1)
    for i, (col, mk, txt) in enumerate(handles):
        y = y0 - 0.034 * i
        ax.plot([0.022], [y], marker=mk, color=col, markeredgecolor="#202020", markeredgewidth=0.3, markersize=5,
                transform=ax.transAxes, zorder=10, clip_on=False, linestyle="none")
        ax.text(0.04, y, txt, color=col if col != "#00e5ff" and col != "#00dd00" else "#007c8a" if col == "#00e5ff" else "#007a00",
                fontsize=6.6, fontweight="bold", va="center", transform=ax.transAxes, zorder=10,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    ax.text(0.985, 0.03, "\n".join(issued_lines), fontsize=6.8, fontweight="bold", ha="right", va="bottom",
            transform=ax.transAxes, zorder=10, path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    _save(fig, path)
