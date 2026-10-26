import logging
import os
from pathlib import Path
from time import perf_counter

from .tiler_bounds import TileBounds
from .parquet_index import ParquetIndex
from .mvt_encoder import MVTEncoder
from .tile_cache import TileCache
from shapely.geometry import mapping

logger = logging.getLogger(__name__)

# Pre-generated tiles larger than this are regenerated with the fast path
# instead of served from disk: legacy build pipelines wrote multi-MB tiles at
# low zooms (no raster-consistent capping), and shipping those dominates
# user-perceived latency. The regenerated tile is memory-cached, so the cost
# is paid once per tile per process. Set to 0 to disable the guard.
PREGEN_MAX_BYTES = int(os.environ.get("CATALYST_PREGEN_MAX_BYTES", str(400 * 1024)))

from shapely.geometry import (
    Point,
    LineString,
    Polygon,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
    GeometryCollection
)

def explode_collections(geom):
    if geom is None or geom.is_empty:
        return []

    if isinstance(geom, GeometryCollection):
        out = []
        for g in geom.geoms:
            out.extend(explode_collections(g))
        return out

    if isinstance(geom, (Polygon, MultiPolygon, LineString, MultiLineString, Point, MultiPoint)):
        return [geom]

    return []

class VectorTiler:
    """On-demand MVT tile server with a three-tier lookup.

    For each tile request the lookup order is:
      1. **Memory** — LRU cache (``TileCache``, default 256 entries)
      2. **Disk** — pre-generated ``.mvt`` files under ``<dataset>/mvt/z/x/y.mvt``
      3. **Generate** — reads intersecting GeoParquet tiles, clips/transforms
         geometries, and encodes a fresh MVT on the fly

    Generated tiles are persisted to disk and promoted into the memory cache
    so that subsequent requests are served from cache.
    """

    def __init__(self, dataset_root: str, memory_cache_size: int = 256) -> None:
        self.dataset_root = Path(dataset_root)
        # Prefer a pushdown sidecar (parquet_tiles_pd: filename-bbox + per-row
        # _bbox_ columns) when present, so legacy datasets serve on-the-fly fast.
        pd_dir = self.dataset_root / "parquet_tiles_pd"
        self.parquet_dir = pd_dir if pd_dir.is_dir() else self.dataset_root / "parquet_tiles"
        self.mvt_dir = self.dataset_root / "mvt"
        self.index = ParquetIndex(self.parquet_dir)

        self.cache = TileCache(memory_cache_size)

    def tile_path(self, z: int, x: int, y: int) -> Path:
        return self.mvt_dir / str(z) / str(x) / f"{y}.mvt"

    def generate(self, z: int, x: int, y: int) -> bytes:
        """Generate a tile on the fly.

        Prefers the vectorised single-tile pipeline (ported from starlet:
        bbox/row-group pruning, raster-consistent selection, numpy MVT
        encoding); falls back to the legacy per-row GeoPandas path on error.
        """
        t0 = perf_counter()
        try:
            from catalyst._internal.mvt.fast_tile import generate_tile

            tile_bytes = generate_tile(self.parquet_dir, (z, x, y), index=self.index)
            logger.info("[TileGen] z=%d x=%d y=%d fast path %d bytes elapsed=%.1fms",
                        z, x, y, len(tile_bytes), (perf_counter() - t0) * 1000)
            return tile_bytes
        except Exception:
            logger.exception("[TileGen] z=%d x=%d y=%d fast path failed; using legacy path", z, x, y)
            return self._generate_legacy(z, x, y)

    def _generate_legacy(self, z: int, x: int, y: int) -> bytes:
        t0 = perf_counter()
        bounds = TileBounds(z, x, y)
        encoder = MVTEncoder(bounds.bbox_3857, bounds.tile_poly_3857)

        try:
            intersecting = self.index.find_intersecting_files(bounds.bbox_4326)
        except Exception as e:
            logger.error("[TileGen] z=%d x=%d y=%d index error: %s", z, x, y, e)
            return encoder.empty_tile()

        if not intersecting:
            logger.debug("[TileGen] z=%d x=%d y=%d no intersecting files", z, x, y)
            return encoder.empty_tile()

        features = []

        for pf in intersecting:
            try:
                gdf = self.index.load_and_reproject(pf, bounds.bbox_4326)
            except Exception as e:
                logger.error("[TileGen] z=%d x=%d y=%d load failed %s: %s", z, x, y, pf, e)
                continue

            try:
                clipped = encoder.clip_to_tile(gdf)
            except Exception as e:
                logger.error("[TileGen] z=%d x=%d y=%d clip failed %s: %s", z, x, y, pf, e)
                continue

            if clipped.empty:
                continue

            for idx, row in clipped.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue

                # extract attributes (all non geometry columns)
                attrs = {k: v for k, v in row.items() if k != "geometry" and v is not None}

                parts = explode_collections(geom)
                if not parts:
                    continue

                for part in parts:
                    if part.is_empty:
                        continue

                    try:
                        scaled = encoder.transform_geom(
                            part,
                            lambda xx, yy, zz=None:
                                TileBounds.scale_to_tile_coords(xx, yy, bounds.bbox_3857)
                        )
                    except Exception as e:
                        logger.error("[TileGen] z=%d x=%d y=%d transform failed: %s", z, x, y, e)
                        continue

                    if scaled.is_empty:
                        continue

                    features.append({
                        "geometry": mapping(scaled),
                        "properties": attrs
                    })

        if not features:
            return encoder.empty_tile()

        try:
            elapsed_ms = (perf_counter() - t0) * 1000
            logger.info("[TileGen] z=%d x=%d y=%d features=%d elapsed=%.1fms",
                        z, x, y, len(features), elapsed_ms)
            return encoder.encode(features)
        except Exception as e:
            logger.error("[TileGen] z=%d x=%d y=%d encode failed: %s", z, x, y, e)
            return encoder.empty_tile()

    def get_tile(self, z: int, x: int, y: int) -> bytes:
        key = (z, x, y)

        cached = self.cache.get(key)
        if cached is not None:
            logger.debug("[Cache] HIT memory z=%d x=%d y=%d", z, x, y)
            return cached

        path = self.tile_path(z, x, y)

        if path.exists():
            size = path.stat().st_size
            if PREGEN_MAX_BYTES and size > PREGEN_MAX_BYTES and self.parquet_dir.is_dir():
                # Bloated legacy pre-generated tile: regenerate compactly instead.
                logger.info("[Cache] SKIP oversized pregen z=%d x=%d y=%d (%d bytes) — regenerating", z, x, y, size)
            else:
                t0 = perf_counter()
                data = path.read_bytes()
                elapsed_ms = (perf_counter() - t0) * 1000
                logger.debug("[Cache] HIT disk z=%d x=%d y=%d elapsed=%.1fms", z, x, y, elapsed_ms)
                self.cache.put(key, data)
                return data

        logger.info("[Cache] MISS z=%d x=%d y=%d — generating (memory cache only)", z, x, y)

        tile_bytes = self.generate(z, x, y)

        # On-demand tiles are cached in memory only; we deliberately do NOT
        # persist them to disk. Writing every generated tile would incrementally
        # materialise the full pyramid on disk over time — exactly what lazy
        # serving exists to avoid. Callers who want a durable on-disk pyramid
        # should pre-generate it explicitly (`catalyst mvt` / `generate_mvt`,
        # e.g. with threshold=0 to materialise every non-empty tile).
        self.cache.put(key, tile_bytes)
        return tile_bytes
