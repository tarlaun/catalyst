#!/usr/bin/env python3
"""Text-free poster figures: AID*-Figure-4-style tile pyramid + density grid example.

Palette matches STARLET_POSTER.pptx (UCR navy #003DA5, gold #FFB81C).
Outputs 300-dpi PNGs with transparent canvas (cells are explicitly filled).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap

OUT = __file__.rsplit("/", 1)[0]

NAVY = "#003DA5"
GOLD = "#FFB81C"
SHALLOW = "#D6E6F8"
EMPTY = "#FFFFFF"
GRIDLINE = "#9FBEE2"
BORDER = "#4A76B8"
POINT = "#1A5CC0"

BLUES = LinearSegmentedColormap.from_list(
    "starlet_blues", ["#FCFDFF", "#D6E6F8", "#8FB9EE", "#3E7CD0", NAVY]
)

rng = np.random.default_rng(7)


def make_points(n_hot1, n_hot2, n_uniform):
    """Two hotspots + uniform sprinkle in [0,1]^2."""
    c1 = rng.normal([0.62, 0.58], 0.10, size=(n_hot1, 2))
    c2 = rng.normal([0.30, 0.33], 0.07, size=(n_hot2, 2))
    u = rng.uniform(0.02, 0.98, size=(n_uniform, 2))
    pts = np.vstack([c1, c2, u])
    return pts[(pts[:, 0] > 0.02) & (pts[:, 0] < 0.98) & (pts[:, 1] > 0.02) & (pts[:, 1] < 0.98)]


C = 1 / np.sqrt(2)
SQUASH = 0.55


def iso(x, y):
    """Isometric projection: rotate 45deg, squash vertically. Y grows downward."""
    return (x - y) * C, (x + y) * C * SQUASH


def cell_color(count, th_hi, th_lo):
    if count == 0:
        return EMPTY
    if count > th_hi:
        return NAVY
    if count > th_lo:
        return GOLD
    return SHALLOW


def draw_land(ax, rings, n, y_off, lw, zorder):
    """Land polygons (nan-separated Web Mercator rings) scaled to an n-unit plane."""
    breaks = np.where(~np.isfinite(rings[:, 0]))[0]
    start = 0
    for b in list(breaks) + [len(rings)]:
        ring = rings[start:b]
        start = b + 1
        if len(ring) < 3:
            continue
        RX, RY = iso(ring[:, 0] * n, ring[:, 1] * n)
        ax.add_patch(Polygon(np.column_stack([RX, RY + y_off]), closed=True,
                             facecolor="#FDFEFF", edgecolor="#6B87AC",
                             linewidth=lw, zorder=zorder))


def draw_level(ax, pts, z, y_off, th_hi, th_lo, rings=None):
    """One pyramid level: 2^z x 2^z grid, unit drawn cell size, top point at y_off.

    With `rings`, the world map is drawn under the grid and the tile fills go
    semi-transparent so the map shows through (paper Figure 4 style).
    """
    n = 2 ** z
    H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=n, range=[[0, 1], [0, 1]])
    outer = [(0, 0), (n, 0), (n, n), (0, n)]
    oxy = [(X, Y + y_off) for X, Y in (iso(x, y) for x, y in outer)]
    if rings is not None:
        ax.add_patch(Polygon(oxy, closed=True, facecolor="#E8F1FA", edgecolor="none", zorder=1))
        draw_land(ax, rings, n, y_off, lw=max(0.3, 0.06 * n), zorder=2)
    alpha = {NAVY: 0.78, GOLD: 0.78, SHALLOW: 0.45, EMPTY: 0.0}
    for i in range(n):
        for j in range(n):
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            xy = [(X, Y + y_off) for X, Y in (iso(x, y) for x, y in corners)]
            col = cell_color(H[i, j], th_hi, th_lo)
            a = alpha[col] if rings is not None else 1.0
            ax.add_patch(Polygon(xy, closed=True, facecolor=col, alpha=a,
                                 edgecolor="none", zorder=3))
            ax.add_patch(Polygon(xy, closed=True, facecolor="none",
                                 edgecolor=GRIDLINE, linewidth=0.8,
                                 joinstyle="round", zorder=4))
    ax.add_patch(Polygon(oxy, closed=True, facecolor="none", edgecolor=BORDER,
                         linewidth=1.6, zorder=5))
    return n * 2 * C * SQUASH  # diamond height


def draw_dataset_plane(ax, pts, n, y_off):
    outer = [(0, 0), (n, 0), (n, n), (0, n)]
    xy = [(X, Y + y_off) for X, Y in (iso(x, y) for x, y in outer)]
    ax.add_patch(Polygon(xy, closed=True, facecolor="#FBFDFF", edgecolor=BORDER, linewidth=1.6))
    PX, PY = iso(pts[:, 0] * n, pts[:, 1] * n)
    ax.scatter(PX, PY + y_off, s=2.2, c=POINT, alpha=0.8, linewidths=0)
    return n * 2 * C * SQUASH


def draw_world_map(ax, rings, points, n, y_off):
    """Base plane as a world map: ocean diamond, land polygons, building dots (Web Mercator)."""
    outer = [(0, 0), (n, 0), (n, n), (0, n)]
    xy = [(X, Y + y_off) for X, Y in (iso(x, y) for x, y in outer)]
    ax.add_patch(Polygon(xy, closed=True, facecolor="#E8F1FA", edgecolor=BORDER,
                         linewidth=1.6, zorder=1))
    breaks = np.where(~np.isfinite(rings[:, 0]))[0]
    start = 0
    for b in list(breaks) + [len(rings)]:
        ring = rings[start:b]
        start = b + 1
        if len(ring) < 3:
            continue
        RX, RY = iso(ring[:, 0] * n, ring[:, 1] * n)
        ax.add_patch(Polygon(np.column_stack([RX, RY + y_off]), closed=True,
                             facecolor="#FDFEFF", edgecolor="#6B87AC", linewidth=0.5, zorder=2))
    PX, PY = iso(points[:, 0] * n, points[:, 1] * n)
    ax.scatter(PX, PY + y_off, s=0.5, c=POINT, alpha=0.55, linewidths=0, zorder=3)
    return n * 2 * C * SQUASH


def draw_dataset_map(ax, segments, points, n, y_off):
    """Base plane drawn as a real map: road polylines + intersection dots in iso view."""
    outer = [(0, 0), (n, 0), (n, n), (0, n)]
    xy = [(X, Y + y_off) for X, Y in (iso(x, y) for x, y in outer)]
    ax.add_patch(Polygon(xy, closed=True, facecolor="#FBFDFF", edgecolor=BORDER,
                         linewidth=1.6, zorder=1))
    PX, PY = iso(points[:, 0] * n, points[:, 1] * n)
    ax.scatter(PX, PY + y_off, s=0.35, c="#3E6FB8", alpha=0.45, linewidths=0, zorder=2)
    SX, SY = iso(segments[:, 0] * n, segments[:, 1] * n)
    ax.plot(SX, SY + y_off, color="#14335F", lw=0.6, alpha=0.95,
            solid_capstyle="round", zorder=3)
    return n * 2 * C * SQUASH


def corners_lr(n, y_off):
    """Left and right corner points of a level's diamond."""
    lx, ly = iso(0, n)
    rx, ry = iso(n, 0)
    return (lx, ly + y_off), (rx, ry + y_off)


def draw_level_binary(ax, pts, n, S, y_off, th, rings, dots=None):
    """Fixed-size plane (S units) with an n x n grid, Figure-4 style.

    Exactly two tile classes: count > th -> navy (pre-generated MVT),
    count <= th (empty included) -> gold (generated on the fly).
    """
    H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=n, range=[[0, 1], [0, 1]])
    outer = [(0, 0), (S, 0), (S, S), (0, S)]
    oxy = [(X, Y + y_off) for X, Y in (iso(x, y) for x, y in outer)]
    ax.add_patch(Polygon(oxy, closed=True, facecolor="#E8F1FA", edgecolor="none", zorder=1))
    draw_land(ax, rings, S, y_off, lw=0.5, zorder=2)
    if dots is not None:
        PX, PY = iso(dots[:, 0] * S, dots[:, 1] * S)
        ax.scatter(PX, PY + y_off, s=0.5, c=POINT, alpha=0.55, linewidths=0, zorder=3)
    c = S / n
    lw_grid = max(0.22, 0.8 * (8.0 / n) ** 0.7)
    for i in range(n):
        for j in range(n):
            corners = [(i * c, j * c), ((i + 1) * c, j * c),
                       ((i + 1) * c, (j + 1) * c), (i * c, (j + 1) * c)]
            xy = [(X, Y + y_off) for X, Y in (iso(x, y) for x, y in corners)]
            col, a = (NAVY, 0.5) if H[i, j] > th else (GOLD, 0.33)
            ax.add_patch(Polygon(xy, closed=True, facecolor=col, alpha=a,
                                 edgecolor="none", zorder=4))
    for k in range(1, n):
        for (x0, y0), (x1, y1) in [((k * c, 0), (k * c, S)), ((0, k * c), (S, k * c))]:
            (X0, Y0), (X1, Y1) = iso(x0, y0), iso(x1, y1)
            ax.plot([X0, X1], [Y0 + y_off, Y1 + y_off], color=GRIDLINE,
                    lw=lw_grid, zorder=5)
    ax.add_patch(Polygon(oxy, closed=True, facecolor="none", edgecolor=BORDER,
                         linewidth=1.6, zorder=6))
    return S * 2 * C * SQUASH


def pyramid(fname, connectors, grids=(1, 2, 4, 8, 16)):
    S = 8.0                     # every plane the same drawn size, Figure-4 style
    grids = list(grids)         # coarse zoom at top -> fine zoom at bottom
    gap = 0.95
    total_h = (len(grids) + 1) * S * 2 * C * SQUASH + len(grids) * gap
    fig, ax = plt.subplots(figsize=(6.8, 6.8 * (total_h + 0.8) / 12.6))

    scratch = ("/private/tmp/claude-501/-Users-tarlan-Downloads-catalyst-git/"
               "d13c616f-39e2-4143-932e-03fb341b872e/scratchpad")
    data = np.load(f"{scratch}/world_map.npz")
    rings, pts = data["rings"], data["points"]

    # One threshold, from the finest level's nonzero counts: tiles above it are
    # pre-generated as MVT (navy), the rest are rendered on the fly (gold).
    # Parents aggregate children, so coarse zooms come out navy - as deployed.
    # Fixed 32-bin basis so shared levels color identically across variants.
    Hf, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=32, range=[[0, 1], [0, 1]])
    th = np.quantile(Hf[Hf > 0], 0.80)

    y = 0.0
    lr_pts = []
    for n in grids:
        h = draw_level_binary(ax, pts, n, S, y, th, rings)
        lr_pts.append(corners_lr(S, y))
        y += h + gap
    h = draw_world_map(ax, rings, pts, S, y)
    lr_pts.append(corners_lr(S, y))
    y += h

    if connectors:
        for (l1, r1), (l2, r2) in zip(lr_pts, lr_pts[1:]):
            ax.plot([l1[0], l2[0]], [l1[1], l2[1]], ls=(0, (4, 3)), lw=1.0, color=GRIDLINE, zorder=0)
            ax.plot([r1[0], r2[0]], [r1[1], r2[1]], ls=(0, (4, 3)), lw=1.0, color=GRIDLINE, zorder=0)

    ax.set_aspect("equal")
    ax.set_xlim(-6.3, 6.3)
    ax.set_ylim(y + 0.4, -0.4)  # inverted: top of pyramid at top
    ax.axis("off")
    fig.savefig(f"{OUT}/{fname}", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def density_figs():
    n = 8
    pts = make_points(150, 80, 70)
    H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=n, range=[[0, 1], [0, 1]])
    vmax = H.max()

    # (a) input scatter with grid overlay, points shaded by local density
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#FCFDFF", edgecolor=BORDER, linewidth=2))
    cell_ids = np.clip((pts * n).astype(int), 0, n - 1)
    dens = H[cell_ids[:, 0], cell_ids[:, 1]]
    ax.scatter(pts[:, 0], pts[:, 1], s=26, c=dens, cmap=LinearSegmentedColormap.from_list(
        "pts", ["#7FB3E8", NAVY]), linewidths=0, alpha=0.9)
    for k in range(1, n):
        ax.plot([k / n, k / n], [0, 1], color=GRIDLINE, lw=0.9, alpha=0.6)
        ax.plot([0, 1], [k / n, k / n], color=GRIDLINE, lw=0.9, alpha=0.6)
    ax.set_aspect("equal"); ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01); ax.axis("off")
    fig.savefig(f"{OUT}/density_points.png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # (b, c) heatmap grid, without and with counts
    for fname, numbers in [("density_grid.png", False), ("density_grid_counts.png", True)]:
        fig, ax = plt.subplots(figsize=(6, 6))
        for i in range(n):
            for j in range(n):
                ax.add_patch(plt.Rectangle((i / n, j / n), 1 / n, 1 / n,
                                           facecolor=BLUES(H[i, j] / vmax),
                                           edgecolor="#FFFFFF", linewidth=1.4))
                if numbers:
                    dark = H[i, j] / vmax > 0.45
                    ax.text(i / n + 0.5 / n, j / n + 0.5 / n, str(int(H[i, j])),
                            ha="center", va="center", fontsize=13, fontweight="bold",
                            color="#FFFFFF" if dark else "#14335F", family="DejaVu Sans")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=BORDER, linewidth=2))
        ax.set_aspect("equal"); ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01); ax.axis("off")
        fig.savefig(f"{OUT}/{fname}", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)


if __name__ == "__main__":
    pyramid("pyramid.png", connectors=False)                       # z = 0..4
    pyramid("pyramid_connectors.png", connectors=True)             # z = 0..4
    pyramid("pyramid_z0_3.png", connectors=False, grids=(1, 2, 4, 8))  # z = 0..3
    density_figs()
    print("done")
