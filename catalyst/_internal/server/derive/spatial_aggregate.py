"""Cross-dataset spatial aggregation: derive a new attribute on a target dataset
by spatially joining a source dataset and aggregating one of its attributes.

Example: color parks by their most prominent vegetation type
=> target=parks (polygons), source=vegetation (polygons),
   aggregate="dominant", value_attribute="VEG_TYPE", weight="area".

The result is the target GeoDataFrame with one extra column (`output_attribute`),
which the rest of catalyst tiles, serves, and styles like any other dataset.
"""
from __future__ import annotations

import glob
import os
from typing import Optional

import geopandas as gpd

_SUPPORTED = {"dominant", "count", "mean", "sum", "max", "min"}

# Internal/system columns written by the tiler that should not surface as data.
_INTERNAL_COLS = ("_tile_id", "_bbox_xmin", "_bbox_ymin", "_bbox_xmax", "_bbox_ymax")


def spatial_aggregate(
    target_gdf: gpd.GeoDataFrame,
    source_gdf: gpd.GeoDataFrame,
    *,
    predicate: str = "intersects",
    aggregate: str = "dominant",
    value_attribute: Optional[str] = None,
    weight: Optional[str] = "area",
    output_attribute: str = "derived_value",
) -> gpd.GeoDataFrame:
    """Return ``target_gdf`` with an added ``output_attribute`` column.

    Every target feature is preserved (one output row per input row); targets
    with no spatial match get a null (dominant/mean/...) or 0 (count).
    """
    if aggregate not in _SUPPORTED:
        raise ValueError(f"Unsupported aggregate {aggregate!r}; expected one of {sorted(_SUPPORTED)}")
    if aggregate != "count" and not value_attribute:
        raise ValueError(f"aggregate={aggregate!r} requires a value_attribute")

    target = target_gdf.copy()
    target["__tid"] = range(len(target))

    # For area-weighted aggregation over geographic coords, project to a planar
    # CRS so areas are comparable. (For projected / CRS-less data, use as-is.)
    t = target
    s = source_gdf
    if target.crs is not None and getattr(target.crs, "is_geographic", False):
        t = target.to_crs(epsg=3857)
        s = source_gdf.to_crs(epsg=3857)

    if aggregate == "dominant":
        inter = gpd.overlay(
            t[["__tid", "geometry"]],
            s[[value_attribute, "geometry"]],
            how="intersection",
            keep_geom_type=False,
        )
        if len(inter):
            measure = inter.geometry.area if weight == "area" else inter.geometry.length
            inter = inter.assign(__w=measure)
            agg = inter.groupby(["__tid", value_attribute])["__w"].sum().reset_index()
            winners = agg.loc[agg.groupby("__tid")["__w"].idxmax(), ["__tid", value_attribute]]
            winners = winners.rename(columns={value_attribute: output_attribute})
            target = target.merge(winners, on="__tid", how="left")
        else:
            target[output_attribute] = None
    else:
        cols = ["__tid", "geometry"]
        src = s[["geometry"] + ([value_attribute] if value_attribute else [])]
        joined = gpd.sjoin(src, t[cols], predicate=predicate, how="inner")
        grouped = joined.groupby("__tid")
        if aggregate == "count":
            series = grouped.size()
            target[output_attribute] = target["__tid"].map(series).fillna(0).astype(int)
        else:
            series = grouped[value_attribute].agg(aggregate)
            target[output_attribute] = target["__tid"].map(series)

    return target.drop(columns="__tid")


def load_dataset_gdf(dataset_dir: str) -> gpd.GeoDataFrame:
    """Load a built catalyst dataset's full geometry+attributes as a GeoDataFrame.

    Concatenates all spatially-partitioned parquet tiles (each row appears once),
    decodes the canonical WKB 'geometry' column, and drops tiler-internal columns.
    Source data is EPSG:4326.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from shapely import from_wkb

    tiles = sorted(glob.glob(os.path.join(str(dataset_dir), "parquet_tiles", "*.parquet")))
    if not tiles:
        raise FileNotFoundError(f"No parquet_tiles found under {dataset_dir}")

    table = pa.concat_tables([pq.read_table(t) for t in tiles], promote_options="default")
    df = table.to_pandas()
    if "geometry" not in df.columns:
        raise ValueError(f"Dataset {dataset_dir} has no 'geometry' column")

    geom = from_wkb(df["geometry"].values)
    drop = [c for c in (("geometry",) + _INTERNAL_COLS) if c in df.columns]
    return gpd.GeoDataFrame(df.drop(columns=drop), geometry=geom, crs="EPSG:4326")


def derive_dataset(
    *,
    data_root: str,
    target_dataset: str,
    source_dataset: str,
    predicate: str,
    aggregate: str,
    value_attribute: Optional[str],
    weight: Optional[str],
    output_attribute: str,
    out_path: str,
) -> str:
    """Spatially aggregate ``source_dataset`` onto ``target_dataset`` and write a
    GeoParquet (geometry as WKB) ready to be tiled by ``catalyst.build``.
    Returns the output path."""
    target = load_dataset_gdf(os.path.join(str(data_root), target_dataset))
    source = load_dataset_gdf(os.path.join(str(data_root), source_dataset))

    result = spatial_aggregate(
        target, source,
        predicate=predicate, aggregate=aggregate,
        value_attribute=value_attribute, weight=weight,
        output_attribute=output_attribute,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    result.to_parquet(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Tile-aligned derive: reuse the target's existing spatial partitions instead of
# loading everything + re-partitioning + rebuilding the MVT pyramid.
# ---------------------------------------------------------------------------

def _read_tile_gdf(path: str) -> gpd.GeoDataFrame:
    """Read one target partition into a 4326 GeoDataFrame (no internal columns)."""
    import pyarrow.parquet as pq
    from shapely import from_wkb

    table = pq.read_table(path)
    drop = [c for c in _INTERNAL_COLS if c in table.column_names]
    if drop:
        table = table.drop(drop)
    df = table.to_pandas()
    geom_col = "geometry" if "geometry" in df.columns else next(
        (c for c in df.columns if "geom" in c.lower()), df.columns[-1]
    )
    geom = from_wkb(df[geom_col].to_numpy())
    return gpd.GeoDataFrame(df.drop(columns=[geom_col]), geometry=geom, crs="EPSG:4326")


def _encode_coord(v: float) -> str:
    """Encode a coordinate as ``int_frac3`` to match ``parse_parquet_bbox``."""
    neg = v < 0
    av = abs(float(v))
    ip = int(av)
    fp = int(round((av - ip) * 1000))
    if fp >= 1000:
        ip += 1
        fp -= 1000
    return f"{'-' if neg else ''}{ip}_{fp:03d}"


def _write_derived_tile(gdf: gpd.GeoDataFrame, out_dir: str, idx: int, add_pushdown: bool) -> int:
    """Write one augmented partition with a filename bbox (and per-row ``_bbox_*``
    covering columns) so the derived dataset serves on-the-fly with pruning."""
    if len(gdf) == 0:
        return 0
    minx, miny, maxx, maxy = (float(v) for v in gdf.total_bounds)
    fname = (
        f"tile_{idx:06d}__{_encode_coord(minx)}_{_encode_coord(miny)}_"
        f"{_encode_coord(maxx)}_{_encode_coord(maxy)}.parquet"
    )
    g = gdf
    if add_pushdown:
        b = gdf.geometry.bounds
        g = gdf.assign(
            _bbox_xmin=b["minx"].to_numpy(), _bbox_ymin=b["miny"].to_numpy(),
            _bbox_xmax=b["maxx"].to_numpy(), _bbox_ymax=b["maxy"].to_numpy(),
        )
    g.to_parquet(os.path.join(out_dir, fname))
    return len(gdf)


def derive_dataset_tiled(
    *,
    data_root: str,
    target_dataset: str,
    source_dataset: str,
    predicate: str,
    aggregate: str,
    value_attribute: Optional[str],
    weight: Optional[str],
    output_attribute: str,
    out_dir: str,
    add_pushdown: bool = True,
) -> dict:
    """Tile-aligned cross-dataset derive.

    Reuses the *target's existing spatial partitions*: the source (assumed small)
    is loaded once and each target partition is joined against only the source
    features overlapping that partition, then written straight back as an
    augmented tile. No global re-partition and no MVT pre-generation — the result
    serves on-the-fly (with ``_bbox_`` pushdown). Each target row appears in
    exactly one partition, so per-partition aggregation is exact.

    Returns timing + shape metadata.
    """
    import shutil
    from time import perf_counter

    t0 = perf_counter()
    target_dir = os.path.join(str(data_root), target_dataset)
    source = load_dataset_gdf(os.path.join(str(data_root), source_dataset))
    src_load_s = perf_counter() - t0

    tiles = sorted(glob.glob(os.path.join(target_dir, "parquet_tiles", "*.parquet")))
    if not tiles:
        raise FileNotFoundError(f"No parquet_tiles under {target_dir}")
    out_tiles_dir = os.path.join(str(out_dir), "parquet_tiles")
    os.makedirs(out_tiles_dir, exist_ok=True)

    t_join = perf_counter()
    total_rows = 0
    for i, tile_path in enumerate(tiles):
        tgdf = _read_tile_gdf(tile_path)
        if len(tgdf) == 0:
            continue
        minx, miny, maxx, maxy = tgdf.total_bounds
        ssub = source.cx[minx:maxx, miny:maxy]  # only source overlapping this tile
        if len(ssub) == 0:
            tgdf = tgdf.copy()
            tgdf[output_attribute] = 0 if aggregate == "count" else None
        else:
            tgdf = spatial_aggregate(
                tgdf, ssub, predicate=predicate, aggregate=aggregate,
                value_attribute=value_attribute, weight=weight,
                output_attribute=output_attribute,
            )
        total_rows += _write_derived_tile(tgdf, out_tiles_dir, i, add_pushdown)
    join_write_s = perf_counter() - t_join

    # Geometry is unchanged, so the density histograms and geometry MBR stats are
    # still valid — copy them instead of recomputing.
    for aux in ("histograms", "stats"):
        src_aux = os.path.join(target_dir, aux)
        if os.path.isdir(src_aux):
            shutil.copytree(src_aux, os.path.join(str(out_dir), aux), dirs_exist_ok=True)

    return {
        "dataset": os.path.basename(str(out_dir)),
        "output_attribute": output_attribute,
        "target_dataset": target_dataset,
        "source_dataset": source_dataset,
        "aggregate": aggregate,
        "value_attribute": value_attribute,
        "predicate": predicate,
        "rows": total_rows,
        "tiles": len(tiles),
        "source_load_s": round(src_load_s, 2),
        "join_write_s": round(join_write_s, 2),
        "derive_seconds": round(perf_counter() - t0, 2),
    }
