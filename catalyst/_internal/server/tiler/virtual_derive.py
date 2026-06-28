"""Lazy cross-dataset derive: join a small source layer into each requested
target tile *at serve time* — no materialization, no re-tiling.

For each requested (z,x,y) we fetch the target's existing tile (from its pyramid
or on-the-fly), decode it, and annotate every feature with the value of the
source polygon that contains a representative point of the feature, then
re-encode. The source (assumed small, e.g. hazard zones) is loaded once,
reprojected to 3857, and spatially indexed. Results are cached per tile so
panning stays interactive. The per-feature join is vectorised (one batched
spatial join), not a Python loop, so even dense city-wide tiles stay fast.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import mapbox_vector_tile

from .tile_cache import TileCache
from .tiler import VectorTiler
from .tiler_bounds import EXTENT, TileBounds

logger = logging.getLogger(__name__)


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
        max_features: int = 6000,
    ) -> None:
        self.target = VectorTiler(target_root)
        self.value_attribute = value_attribute
        self.output_attribute = output_attribute or value_attribute
        # Cap features re-encoded per tile: the python MVT encoder is ~0.4ms/feature,
        # so a dense city-wide tile (~24k features) would cost ~10s. At low zoom the
        # individual features are sub-pixel anyway, so an evenly-spaced sample keeps
        # the join interactive without changing what the eye sees.
        self.max_features = max_features

        src = source_gdf
        if src.crs is None:
            src = src.set_crs(4326)
        if value_attribute not in src.columns:
            raise ValueError(f"source has no attribute {value_attribute!r}")
        self.source = (
            src[[value_attribute, "geometry"]].to_crs(3857).reset_index(drop=True)
        )
        self.source.sindex  # build the spatial index eagerly
        self.cache = TileCache(memory_cache_size)

    def get_tile(self, z: int, x: int, y: int) -> bytes:
        key = (z, x, y)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        base = self.target.get_tile(z, x, y)
        out = self._annotate(base, z, x, y)
        self.cache.put(key, out)
        return out

    def _annotate(self, mvt_bytes: bytes, z: int, x: int, y: int) -> bytes:
        try:
            dec = mapbox_vector_tile.decode(mvt_bytes)
        except Exception:
            return mvt_bytes
        if not dec:
            return mvt_bytes

        minx, miny, maxx, maxy = TileBounds(z, x, y).bbox_3857
        out_layers = []
        for name, layer in dec.items():
            extent = layer.get("extent", EXTENT)
            ex = (maxx - minx) / extent
            ey = (maxy - miny) / extent
            feats = layer.get("features") or []

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
            if len(out_feats) > self.max_features:
                step = len(out_feats) / self.max_features
                out_feats = [out_feats[int(k * step)] for k in range(self.max_features)]
            out_layers.append({"name": name, "extent": extent, "features": out_feats})

        try:
            return mapbox_vector_tile.encode(out_layers)
        except Exception:
            logger.exception("[VirtualDerive] re-encode failed")
            return mvt_bytes
