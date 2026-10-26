#!/usr/bin/env python3
"""Build a ``parquet_tiles_pd`` pushdown sidecar for legacy datasets.

Legacy partitions (``<n>.parquet``: no filename bbox, no ``_bbox_*`` columns,
one giant row group) defeat every pruning tier of the on-the-fly tile
generator. This script rewrites each partition with:

  * per-row bbox covering columns (``_bbox_xmin`` .. ``_bbox_ymax``),
  * rows sorted by the Morton code of their bbox centre (spatially coherent
    row groups whose Parquet min/max statistics prune tightly),
  * small row groups (default 8192 rows),
  * the partition bbox encoded in the filename
    (``tile_XXXXXX__minx_miny_maxx_maxy.parquet``).

The serving tiler (``VectorTiler``) automatically prefers ``parquet_tiles_pd``
over ``parquet_tiles`` when the directory exists. Original data is untouched.

Usage:
    python scripts/build_pushdown_sidecar.py <datasets_root> [dataset ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shapely

ROW_GROUP_SIZE = 8192
BBOX_COLS = ("_bbox_xmin", "_bbox_ymin", "_bbox_xmax", "_bbox_ymax")


def geometry_column(schema: pa.Schema) -> str:
    names = list(schema.names)
    if "geometry" in names:
        return "geometry"
    for name, typ in zip(names, schema.types):
        if pa.types.is_binary(typ) or pa.types.is_large_binary(typ):
            return name
    return names[-1]


def morton(cx: np.ndarray, cy: np.ndarray, bbox) -> np.ndarray:
    """16-bit interleaved Morton code of normalized centres."""
    minx, miny, maxx, maxy = bbox
    sx = (maxx - minx) or 1.0
    sy = (maxy - miny) or 1.0
    xi = np.clip(((cx - minx) / sx * 65535), 0, 65535).astype(np.uint64)
    yi = np.clip(((cy - miny) / sy * 65535), 0, 65535).astype(np.uint64)

    def spread(v: np.ndarray) -> np.ndarray:
        v = (v | (v << 8)) & np.uint64(0x00FF00FF)
        v = (v | (v << 4)) & np.uint64(0x0F0F0F0F)
        v = (v | (v << 2)) & np.uint64(0x33333333)
        v = (v | (v << 1)) & np.uint64(0x55555555)
        return v

    return spread(xi) | (spread(yi) << np.uint64(1))


def fmt_coord(v: float) -> str:
    return f"{v:.6f}".replace(".", "_")


def build_sidecar(ds_dir: Path) -> None:
    src = ds_dir / "parquet_tiles"
    dst = ds_dir / "parquet_tiles_pd"
    if not src.is_dir():
        print(f"  {ds_dir.name}: no parquet_tiles, skipped")
        return
    tmp = ds_dir / "parquet_tiles_pd.tmp"
    if tmp.exists():
        for f in tmp.glob("*"):
            f.unlink()
    tmp.mkdir(exist_ok=True)

    n_files = 0
    n_rows = 0
    for i, pf in enumerate(sorted(src.glob("*.parquet"))):
        table = pq.read_table(pf)
        geom_col = geometry_column(table.schema)
        drop = [c for c in BBOX_COLS if c in table.column_names]
        if drop:
            table = table.drop(drop)
        wkb = table[geom_col].to_pylist()
        geoms = shapely.from_wkb(wkb, on_invalid="ignore")
        valid = ~shapely.is_missing(geoms)
        if not valid.all():
            table = table.take(pa.array(np.flatnonzero(valid)))
            geoms = geoms[valid]
        if len(table) == 0:
            continue
        b = shapely.bounds(geoms)  # minx, miny, maxx, maxy per row
        # pad outward so the 6-decimal filename bbox always covers every row
        part_bbox = (float(np.nanmin(b[:, 0])) - 1e-5, float(np.nanmin(b[:, 1])) - 1e-5,
                     float(np.nanmax(b[:, 2])) + 1e-5, float(np.nanmax(b[:, 3])) + 1e-5)
        order = np.argsort(morton((b[:, 0] + b[:, 2]) * 0.5, (b[:, 1] + b[:, 3]) * 0.5, part_bbox))
        table = table.take(pa.array(order))
        b = b[order]
        for col, vals in zip(BBOX_COLS, (b[:, 0], b[:, 1], b[:, 2], b[:, 3])):
            table = table.append_column(col, pa.array(vals, type=pa.float64()))
        name = (f"tile_{i:06d}__{fmt_coord(part_bbox[0])}_{fmt_coord(part_bbox[1])}"
                f"_{fmt_coord(part_bbox[2])}_{fmt_coord(part_bbox[3])}.parquet")
        pq.write_table(table, tmp / name, row_group_size=ROW_GROUP_SIZE, compression="zstd")
        n_files += 1
        n_rows += len(table)

    if dst.exists():
        import shutil
        shutil.rmtree(dst)
    tmp.rename(dst)
    print(f"  {ds_dir.name}: {n_files} partitions, {n_rows:,} rows -> {dst}")


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "datasets")
    names = sys.argv[2:]
    dirs = ([root / n for n in names] if names else
            [d for d in sorted(root.iterdir())
             if d.is_dir() and not d.name.startswith("_") and (d / "parquet_tiles").is_dir()])
    for d in dirs:
        build_sidecar(d)


if __name__ == "__main__":
    main()
