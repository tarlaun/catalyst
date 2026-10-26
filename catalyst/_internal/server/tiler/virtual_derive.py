"""Lazy cross-dataset derive: join a small source layer into each requested
target tile *at serve time* — no materialization, no re-tiling.

The join runs *inside* the fast tile generator (``fast_tile.generate_tile``'s
``annotate`` hook), not on a finished MVT: selected target features are joined
by bbox-centre point-in-polygon against an STRtree of the source (one batched
shapely query per tile), and sub-pixel packed dots are grouped per joined
class so dense city-zoom targets (3.9M buildings) stay colorable. Tiles whose
bbox misses the source entirely short-circuit to the target's own (cached /
pre-generated) tile. Decoding + re-encoding the MVT with the pure-Python codec
— the old approach, seconds per dense tile and capped at 2500 sampled
features — remains only as a fallback for datasets the fast path can't read.
"""
from __future__ import annotations

import logging
import os

import geopandas as gpd
import mapbox_vector_tile
import numpy as np
import shapely
from shapely.geometry import box

from catalyst._internal.mvt.fast_tile import TileFrame, generate_tile

from .tile_cache import TileCache
from .tiler import VectorTiler
from .tiler_bounds import EXTENT, TileBounds

logger = logging.getLogger(__name__)

# Features re-encoded per tile. The pure-Python MVT encoder is ~0.4-0.5ms/feature,
# so this directly bounds per-tile cost. At city zoom the individual features are
# sub-pixel, so an evenly-spaced sample keeps the join interactive. Tunable.
_DEFAULT_MAX_FEATURES = int(os.environ.get("CATALYST_DERIVE_MAX_FEATURES", "2500") or 2500)


def _first_xy(coords):
    """First (x, y) pair of any nested GeoJSON coordinate list (no shapely)."""
    c = coords
    while isinstance(c, (list, tuple)) and len(c) and isinstance(c[0], (list, tuple)):
        c = c[0]
    return float(c[0]), float(c[1])


class VirtualDeriveTiler:
    """Serve a target layer with one extra attribute joined from a small source,
    computed lazily per tile (the result is never materialized to disk)."""

    def __init__(
        self,
        target_root: str,
        source_gdf,
        value_attribute: str,
        output_attribute: str = None,
        memory_cache_size: int = 256,
        max_features: int = None,
    ) -> None:
        self.target = VectorTiler(target_root)
        self.value_attribute = value_attribute
        self.output_attribute = output_attribute or value_attribute
        self.max_features = _DEFAULT_MAX_FEATURES if max_features is None else max_features

        src = source_gdf
        if src.crs is None:
            src = src.set_crs(4326)
        if value_attribute not in src.columns:
            raise ValueError(f"source has no attribute {value_attribute!r}")

        # Fast join structures (EPSG:4326, same CRS as the parquet partitions):
        # geometries with a non-null value, an STRtree over them, and the values.
        src4326 = src[[value_attribute, "geometry"]].to_crs(4326).reset_index(drop=True)
        keep = src4326[value_attribute].notna().to_numpy()
        self._src_geoms = np.asarray(src4326.geometry.values[keep], dtype=object)
        self._src_vals = np.asarray(src4326[value_attribute].to_numpy(dtype=object)[keep])
        self._tree = shapely.STRtree(self._src_geoms)

        # Legacy fallback (MVT decode/re-encode) joins in 3857.
        self.source = src4326.to_crs(3857)
        self.source.sindex  # build the spatial index eagerly
        self.cache = TileCache(memory_cache_size)

    def _join_values(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Joined source value per (lon, lat) point, None where nothing matches
        — one batched STRtree query, no per-feature Python work."""
        pts = shapely.points(lon, lat)
        out = np.full(len(pts), None, dtype=object)
        qi, si = self._tree.query(pts, predicate="intersects")
        if len(qi):
            # reversed so, on overlap, the first matching source feature wins
            out[qi[::-1]] = self._src_vals[si[::-1]]
        return out

    def get_tile(self, z: int, x: int, y: int) -> bytes:
        key = (z, x, y)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        out = None
        pdir = getattr(self.target, "parquet_dir", None)
        if pdir is not None and pdir.is_dir():
            try:
                qx0, qy0, qx1, qy1 = TileFrame(z, x, y, EXTENT, 256).query_lonlat
                if len(self._tree.query(box(qx0, qy0, qx1, qy1))) == 0:
                    # no source anywhere near: the target's own tile is the answer
                    # (served from its cache / pre-generated pyramid, usually free)
                    return self.target.get_tile(z, x, y)
                out = generate_tile(
                    pdir,
                    (z, x, y),
                    index=getattr(self.target, "index", None),
                    annotate=self._join_values,
                    annotate_attribute=self.output_attribute,
                )
            except Exception:
                logger.exception("[VirtualDerive] fast join tile failed for %s/%s/%s", z, x, y)
                out = None
        if out is None:
            base = self.target.get_tile(z, x, y)
            out = self._annotate(base, z, x, y)
        self.cache.put(key, out)
        return out

    def _annotate(self, mvt_bytes: bytes, z: int, x: int, y: int) -> bytes:
        minx, miny, maxx, maxy = TileBounds(z, x, y).bbox_3857

        # Fast path: if no source polygon is even near this tile, there is nothing to
        # join — return the base tile untouched (no decode/re-encode). Over a large
        # target (a whole city) most tiles fall outside a small source (hazard zones in
        # the hills / ocean / flatland), so the vast majority of tiles stay ~free.
        try:
            if len(self.source.sindex.query(box(minx, miny, maxx, maxy))) == 0:
                return mvt_bytes
        except Exception:
            pass

        try:
            dec = mapbox_vector_tile.decode(mvt_bytes)
        except Exception:
            return mvt_bytes
        if not dec:
            return mvt_bytes

        out_layers = []
        for name, layer in dec.items():
            extent = layer.get("extent", EXTENT)
            ex = (maxx - minx) / extent
            ey = (maxy - miny) / extent
            feats = layer.get("features") or []

            # Cap BEFORE the join + re-encode (not after): we only ever emit
            # max_features, so sampling first avoids O(all-features) point-building,
            # sjoin and encode on dense tiles (~24k buildings at city zoom → ~2.5k).
            if len(feats) > self.max_features:
                step = len(feats) / self.max_features
                feats = [feats[int(k * step)] for k in range(self.max_features)]

            xs, ys, fis = [], [], []
            for i, f in enumerate(feats):
                try:
                    gx, gy = _first_xy(f["geometry"]["coordinates"])
                    xs.append(minx + gx * ex)
                    ys.append(miny + gy * ey)
                    fis.append(i)
                except Exception:
                    pass

            if xs:
                try:
                    qpts = gpd.GeoDataFrame(
                        {"_fi": fis},
                        geometry=gpd.points_from_xy(xs, ys),
                        crs=3857,
                    )
                    j = gpd.sjoin(qpts, self.source, how="inner", predicate="intersects")
                    j = j.dropna(subset=[self.value_attribute]).drop_duplicates("_fi")
                    out_attr = self.output_attribute
                    for fi, val in zip(j["_fi"].tolist(), j[self.value_attribute].tolist()):
                        feats[fi]["properties"][out_attr] = val
                except Exception:
                    logger.exception("[VirtualDerive] join failed")

            out_feats = [
                {"geometry": f["geometry"], "properties": f.get("properties") or {}}
                for f in feats
            ]
            out_layers.append({"name": name, "extent": extent, "features": out_feats})

        try:
            return mapbox_vector_tile.encode(out_layers)
        except Exception:
            logger.exception("[VirtualDerive] re-encode failed")
            return mvt_bytes
