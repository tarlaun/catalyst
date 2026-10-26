#!/usr/bin/env python3
"""Tile-pruning statistics for the poster: Starlet/Catalyst vs no pruning.

Uses the real Starlet artifact (datasets/osm_buildings/histograms/global.npy,
a 4096x4096 vertex-count histogram over the full Web Mercator plane) to
classify every candidate tile for z = 0..10:

  empty         count == 0            -> pruned outright, never touched
  low density   0 < est. size < thr   -> not pre-generated; rendered on the fly
  high density  est. size >= thr      -> pre-generated as MVT

Estimated tile size comes from a linear fit (bytes ~ a*count + b) calibrated
on the dataset's actual pre-generated .mvt files. Baseline "no pruning" =
materialize every tile of the pyramid (HadoopViz-style).

Outputs: tile_pruning_stats.md (tables) and tile_pruning_stats.csv (raw rows).
"""
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "datasets", "osm_buildings")
ZMAX = 10
THRESHOLDS = {"8 KB": 8 * 1024, "16 KB": 16 * 1024, "32 KB": 32 * 1024,
              "64 KB": 64 * 1024, "128 KB": 128 * 1024, "256 KB": 256 * 1024}

# Transposed so C[x, y] matches XYZ tile addressing (verified: all real .mvt
# files land on nonzero cells in this orientation, 77-80% miss otherwise).
H = np.load(os.path.join(DS, "histograms", "global.npy")).T
N = H.shape[0]  # 4096 = zoom-12 tile grid over EPSG:3857 world


def counts_at(z):
    k = N // (2 ** z)
    return H.reshape(2 ** z, k, 2 ** z, k).sum(axis=(1, 3))


# --- calibrate bytes per histogram count on the existing mvt tree ------------
# Fit only the sub-cap regime (lowest 90% of counts): pre-generated tiles are
# reservoir-capped, so the largest tiles' file sizes understate the true
# (uncapped) rendering cost the threshold decision is about.
files = glob.glob(os.path.join(DS, "mvt", "*", "*", "*.mvt"))
rows = []
for f in files:
    z, x, yname = f.split(os.sep)[-3:]
    z, x, y = int(z), int(x), int(yname[:-4])
    size = os.path.getsize(f)
    if size > 100:  # drop near-empty stub tiles
        rows.append((z, x, y, size))

per_z = {z: counts_at(z) for z in sorted({r[0] for r in rows})}
cs = np.array([per_z[z][x, y] for z, x, y, _ in rows])
ss = np.array([s for *_, s in rows])
sub = cs < np.quantile(cs, 0.90)
a, b = np.polyfit(cs[sub], ss[sub], 1)
corr = np.corrcoef(cs[sub], ss[sub])[0, 1]
print(f"calibration: {sub.sum()}/{len(rows)} tiles (sub-cap regime), "
      f"r={corr:.3f}, bytes ~ {a:.2f}*count + {b:.0f}")

# --- classify the full pyramid ----------------------------------------------
md = []
md.append("# Tile pruning statistics — osm_buildings, z = 0..%d\n" % ZMAX)
md.append("Source: `datasets/osm_buildings/histograms/global.npy` "
          "(4096 x 4096 vertex-count histogram, EPSG:3857 world; the same "
          "artifact Starlet's O(1) pruning reads).\n")
md.append(f"Tile size model: bytes = {a:.2f} x count + {b:.0f}, calibrated on "
          f"the pre-generated `.mvt` files of this dataset ({int(sub.sum())} "
          f"tiles in the sub-cap regime, Pearson r = {corr:.3f}; the largest "
          "pre-generated tiles are feature-capped, so they are excluded from "
          "the fit).\n")
md.append("Classes per tile: **empty** (count = 0, pruned outright), "
          "**low density** (below threshold, rendered on the fly), "
          "**high density** (at/above threshold, pre-generated as MVT). "
          "Baseline without pruning materializes every tile.\n")

csv_lines = ["threshold,zoom,total_tiles,empty,low_density,high_density,"
             "pregenerated_pct,tiles_saved_vs_no_pruning,saved_pct"]

for tname, tbytes in THRESHOLDS.items():
    md.append(f"\n## Threshold {tname}\n")
    md.append("| zoom | total tiles | empty | low density (on the fly) | "
              "high density (pre-generated) | pre-generated % | "
              "tiles saved vs no pruning | saved % |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    T = dict.fromkeys(("tot", "emp", "low", "high"), 0)
    for z in range(ZMAX + 1):
        C = counts_at(z)
        total = C.size
        empty = int((C == 0).sum())
        est = a * C + b
        high = int(((C > 0) & (est >= tbytes)).sum())
        low = total - empty - high
        saved = total - high
        md.append(f"| {z} | {total:,} | {empty:,} | {low:,} | {high:,} | "
                  f"{100*high/total:.2f}% | {saved:,} | {100*saved/total:.2f}% |")
        csv_lines.append(f"{tname},{z},{total},{empty},{low},{high},"
                         f"{100*high/total:.4f},{saved},{100*saved/total:.4f}")
        for k, v in zip(("tot", "emp", "low", "high"), (total, empty, low, high)):
            T[k] += v
    saved = T["tot"] - T["high"]
    md.append(f"| **all** | **{T['tot']:,}** | **{T['emp']:,}** | "
              f"**{T['low']:,}** | **{T['high']:,}** | "
              f"**{100*T['high']/T['tot']:.2f}%** | **{saved:,}** | "
              f"**{100*saved/T['tot']:.2f}%** |")
    csv_lines.append(f"{tname},all,{T['tot']},{T['emp']},{T['low']},{T['high']},"
                     f"{100*T['high']/T['tot']:.4f},{saved},"
                     f"{100*saved/T['tot']:.4f}")

with open(os.path.join(HERE, "tile_pruning_stats.md"), "w") as f:
    f.write("\n".join(md) + "\n")
with open(os.path.join(HERE, "tile_pruning_stats.csv"), "w") as f:
    f.write("\n".join(csv_lines) + "\n")
print("wrote tile_pruning_stats.md / .csv")
