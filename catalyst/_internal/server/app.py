"""Flask application factory for the Starlet tile server."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from threading import Lock, Thread
from uuid import uuid4
import json
import logging
import os
import re
import shutil

from flask import Flask, Response, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS

from .catalog.embedder import get_embedder
from .catalog.index import CATALOG_FILENAME, build_catalog_index
from .catalog.pgvector_store import PgVectorConfig, PgVectorStore
from .catalog.router import CatalogRouter, SearchBackend
from .download_service import DatasetFeatureService
from .llm import (
    continue_style_conversation,
    generate_map_code,
    start_multilayer_conversation,
    start_style_conversation,
)
from .llm.provider import LLMProviderError
from .tiler.tiler import VectorTiler

# Env-driven so the same code serves the dev box (root) and the public
# deployment (mounted under a path prefix, e.g. /catalyst, behind a proxy).
# LLM_PROVIDER=fallback uses the gemini->ollama chain (see llm/factory.py).
_LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").strip().lower() or "gemini"
_BASE_PATH = os.environ.get("CATALYST_BASE_PATH", "").rstrip("/")
# Cross-dataset derive uses the fast tile-aligned path when the source has at most
# this many rows (it is loaded once into memory); larger sources use the full build.
_TILED_DERIVE_MAX_SOURCE = int(os.environ.get("CATALYST_TILED_DERIVE_MAX_SOURCE", "2000000") or 2000000)

try:
    from ... import build as starlet_build
except Exception:  # pragma: no cover
    import catalyst as starlet_api

    def starlet_build(*args, **kwargs):
        return starlet_api.build(*args, **kwargs)

logger = logging.getLogger(__name__)


def _normalize_unicode_text(value: Any) -> str:
    text = str(value or "")
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
    )


def _json_response(payload: Any, status: int = 200) -> Tuple[str, int, Dict[str, str]]:
    return (
        json.dumps(payload, indent=2, ensure_ascii=False),
        status,
        {"Content-Type": "application/json; charset=utf-8"},
    )


def create_app(
    data_dir: str,
    cache_size: int = 256,
    log_level: Optional[str] = None,
) -> Flask:
    level = log_level or os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    server_dir = Path(__file__).parent.resolve()
    template_dir = str(server_dir / "templates")

    app = Flask(__name__, template_folder=template_dir)
    app.config["JSON_AS_ASCII"] = False
    CORS(app, resources={r"/*": {"origins": "*"}})

    data_root = Path(data_dir).resolve()
    logger.info("Resolved data root: %s", data_root)
    print("DATA DIR =", data_root)

    tiler_cache: Dict[str, VectorTiler] = {}
    feature_service = DatasetFeatureService(data_root)

    _catalog_runtime: Dict[str, Any] = {
        "router": None,
        "mtime": None,
    }

    _build_jobs: Dict[str, Dict[str, Any]] = {}
    _build_jobs_lock = Lock()

    _tile_metrics: Dict[str, Dict[str, Any]] = {}
    _tile_metrics_lock = Lock()

    uploads_root = data_root / "_uploads"
    uploads_root.mkdir(parents=True, exist_ok=True)

    def _set_build_job(job_id: str, **updates: Any) -> None:
        with _build_jobs_lock:
            current = _build_jobs.get(job_id, {}).copy()
            current.update(updates)
            current["updated_at"] = datetime.now(timezone.utc).isoformat()
            _build_jobs[job_id] = current

    def _get_build_job(job_id: str) -> Optional[Dict[str, Any]]:
        with _build_jobs_lock:
            job = _build_jobs.get(job_id)
            return dict(job) if job else None

    def _set_tile_metric(dataset: str, metric: Dict[str, Any]) -> None:
        with _tile_metrics_lock:
            _tile_metrics[dataset] = dict(metric)

    def _get_tile_metric(dataset: str) -> Optional[Dict[str, Any]]:
        with _tile_metrics_lock:
            metric = _tile_metrics.get(dataset)
            return dict(metric) if metric else None

    def _safe_int(value: Any, fallback: int) -> int:
        try:
            return int(value)
        except Exception:
            return fallback

    def _safe_float(value: Any, fallback: float) -> float:
        try:
            return float(value)
        except Exception:
            return fallback

    def _slugify_dataset_name(name: str) -> str:
        cleaned = _normalize_unicode_text(name).strip()
        cleaned = re.sub(r"\.[A-Za-z0-9]+$", "", cleaned)
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", cleaned)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return cleaned or f"dataset_{uuid4().hex[:8]}"

    def _human_size(num_bytes: int) -> str:
        size = float(max(0, int(num_bytes)))
        units = ["B", "KB", "MB", "GB", "TB"]
        index = 0
        while size >= 1024.0 and index < len(units) - 1:
            size /= 1024.0
            index += 1
        if index == 0:
            return f"{int(size)} {units[index]}"
        return f"{size:.1f} {units[index]}"

    def _apply_pgvector_env_from_request(payload: Dict[str, Any]) -> bool:
        sync_pgvector = str(payload.get("sync_pgvector", "")).strip().lower() in {"1", "true", "yes", "on"}

        if not sync_pgvector:
            os.environ["CATALOG_PGVECTOR_ENABLED"] = "false"
            return False

        os.environ["CATALOG_PGVECTOR_ENABLED"] = "true"
        os.environ["PGVECTOR_HOST"] = _normalize_unicode_text(payload.get("pgvector_host", "localhost")).strip() or "localhost"
        os.environ["PGVECTOR_PORT"] = str(_safe_int(payload.get("pgvector_port", 5432), 5432))
        os.environ["PGVECTOR_DB"] = _normalize_unicode_text(payload.get("pgvector_db", "postgres")).strip() or "postgres"
        os.environ["PGVECTOR_USER"] = _normalize_unicode_text(payload.get("pgvector_user", "")).strip()
        os.environ["PGVECTOR_PASSWORD"] = _normalize_unicode_text(payload.get("pgvector_password", "")).strip()
        os.environ["PGVECTOR_TABLE"] = _normalize_unicode_text(payload.get("pgvector_table", "dataset_catalog_embeddings")).strip() or "dataset_catalog_embeddings"
        return True

    def _run_dataset_build_job(
        *,
        job_id: str,
        uploaded_file_path: Path,
        dataset_name: str,
        num_tiles: int,
        zoom: int,
        threshold: int,
        sync_pgvector: bool,
    ) -> None:
        try:
            _set_build_job(
                job_id,
                status="running",
                step="building_tiles",
                message="Starting Starlet build...",
            )

            outdir = data_root / dataset_name
            outdir.parent.mkdir(parents=True, exist_ok=True)

            build_t0 = perf_counter()
            tile_result, mvt_result, _pmtiles_path = starlet_build(
                input=str(uploaded_file_path),
                outdir=str(outdir),
                num_tiles=num_tiles,
                zoom=zoom,
                threshold=threshold,
            )
            build_seconds = perf_counter() - build_t0

            tiler_cache.pop(dataset_name, None)
            _catalog_runtime["router"] = None
            _catalog_runtime["mtime"] = None

            _set_build_job(
                job_id,
                step="building_index",
                message=f"Tiles generated in {build_seconds:.1f}s. Rebuilding catalogue index...",
                build_seconds=round(build_seconds, 2),
                tile_result={
                    "outdir": str(getattr(tile_result, "outdir", outdir)),
                    "num_files": getattr(tile_result, "num_files", None),
                    "total_rows": getattr(tile_result, "total_rows", None),
                    "bbox": getattr(tile_result, "bbox", None),
                },
                mvt_result={
                    "outdir": str(getattr(mvt_result, "outdir", outdir / "mvt")),
                    "zoom_levels": getattr(mvt_result, "zoom_levels", None),
                    "tile_count": getattr(mvt_result, "tile_count", None),
                },
            )

            index_t0 = perf_counter()
            catalog = build_catalog_index(
                data_root=data_root,
                out_dir=data_root / "_catalog",
                sync_pgvector=sync_pgvector,
            )
            index_seconds = perf_counter() - index_t0

            _catalog_runtime["router"] = None
            _catalog_runtime["mtime"] = None

            # Persist preprocessing timings so they can be reported later
            # (reviewers asked for preprocessing time, not just query-time latency).
            try:
                with open(outdir / "preprocessing.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "build_seconds": round(build_seconds, 2),
                        "index_seconds": round(index_seconds, 2),
                        "num_tiles": getattr(tile_result, "num_files", None),
                        "total_rows": getattr(tile_result, "total_rows", None),
                        "mvt_tiles": getattr(mvt_result, "tile_count", None),
                    }, f, indent=2)
            except Exception:
                logger.exception("Failed to persist preprocessing timings for %s", dataset_name)

            _set_build_job(
                job_id,
                status="completed",
                step="done",
                message=(
                    f"Dataset uploaded, tiled ({build_seconds:.1f}s), and indexed "
                    f"({index_seconds:.1f}s) successfully."
                ),
                dataset=dataset_name,
                output_dir=str(outdir),
                build_seconds=round(build_seconds, 2),
                index_seconds=round(index_seconds, 2),
                catalog_entry_count=catalog.get("entry_count", 0),
            )

        except Exception as e:
            logger.exception("[DatasetBuildJob] Failed for dataset=%s", dataset_name)
            _set_build_job(
                job_id,
                status="failed",
                step="error",
                message=str(e),
            )

    def _default_output_attr(aggregate: str, value_attribute: Optional[str]) -> str:
        if aggregate == "count":
            return "feature_count"
        base = (value_attribute or "value").lower()
        prefix = {"dominant": "dominant", "mean": "avg", "sum": "total"}.get(aggregate, aggregate)
        return f"{prefix}_{base}"

    def _run_derive(spec: Dict[str, Any]) -> Dict[str, Any]:
        """Spatially aggregate one dataset onto another, materialize the result as
        a new built dataset, and (re)index the catalogue. Returns metadata about
        the derived dataset (name + the new attribute to style)."""
        from .derive.spatial_aggregate import derive_dataset

        target = _normalize_unicode_text(spec.get("target_dataset", "")).strip()
        source = _normalize_unicode_text(spec.get("source_dataset", "")).strip()
        aggregate = (_normalize_unicode_text(spec.get("aggregate", "dominant")).strip() or "dominant").lower()
        predicate = (_normalize_unicode_text(spec.get("predicate", "intersects")).strip() or "intersects").lower()
        weight = _normalize_unicode_text(spec.get("weight", "area")).strip() or "area"
        value_attribute = _normalize_unicode_text(spec.get("value_attribute", "")).strip() or None
        output_attribute = _normalize_unicode_text(spec.get("output_attribute", "")).strip() or \
            _default_output_attr(aggregate, value_attribute)

        if not target or not _dataset_exists(target):
            raise LookupError(f"Target dataset not found: {target!r}")
        if not source or not _dataset_exists(source):
            raise LookupError(f"Source dataset not found: {source!r}")
        if aggregate != "count" and not value_attribute:
            raise ValueError(f"aggregate={aggregate!r} requires a value_attribute")

        name = _slugify_dataset_name(f"{target}__{output_attribute}")
        out_parquet = uploads_root / f"{name}.parquet"
        uploads_root.mkdir(parents=True, exist_ok=True)

        spec_key = {
            "target_dataset": target, "source_dataset": source, "aggregate": aggregate,
            "predicate": predicate, "weight": weight, "value_attribute": value_attribute,
            "output_attribute": output_attribute,
        }
        marker_path = data_root / name / "derived.json"

        # Idempotent cache: if this exact derivation was already materialized,
        # reuse it instead of recomputing the join + rebuild.
        if marker_path.exists():
            try:
                cached = json.loads(marker_path.read_text())
                if cached.get("spec") == spec_key:
                    return {**cached.get("info", {}), "dataset": name,
                            "output_attribute": output_attribute, "cached": True,
                            "derive_seconds": 0.0}
            except Exception:
                pass

        outdir = data_root / name
        derive_t0 = perf_counter()

        # Fast path: when the source is small enough to load once, reuse the
        # target's existing spatial partitions (tile-aligned join) instead of a
        # global re-partition + full MVT rebuild. ~40x faster on large targets
        # (e.g. 3.9M buildings: ~1 min vs ~40 min) and the result serves
        # on-the-fly via _bbox_ pushdown. Falls back to the full build on any
        # error or when the source is large.
        used_tiled = False
        source_rows = _dataset_row_count(source)
        if 0 < source_rows <= _TILED_DERIVE_MAX_SOURCE:
            try:
                from .derive.spatial_aggregate import derive_dataset_tiled
                shutil.rmtree(outdir, ignore_errors=True)
                derive_dataset_tiled(
                    data_root=str(data_root),
                    target_dataset=target,
                    source_dataset=source,
                    predicate=predicate,
                    aggregate=aggregate,
                    value_attribute=value_attribute,
                    weight=weight,
                    output_attribute=output_attribute,
                    out_dir=str(outdir),
                )
                used_tiled = True
            except Exception:
                logger.exception("[Derive] tile-aligned path failed for %s; using full build", name)
                shutil.rmtree(outdir, ignore_errors=True)

        if not used_tiled:
            derive_dataset(
                data_root=str(data_root),
                target_dataset=target,
                source_dataset=source,
                predicate=predicate,
                aggregate=aggregate,
                value_attribute=value_attribute,
                weight=weight,
                output_attribute=output_attribute,
                out_path=str(out_parquet),
            )
            starlet_build(input=str(out_parquet), outdir=str(outdir), num_tiles=8, zoom=7, threshold=0)

        derive_seconds = perf_counter() - derive_t0

        info = {
            "dataset": name,
            "output_attribute": output_attribute,
            "target_dataset": target,
            "source_dataset": source,
            "aggregate": aggregate,
            "value_attribute": value_attribute,
            "predicate": predicate,
            "derive_seconds": round(derive_seconds, 2),
        }
        # Mark this as a derived dataset (excluded from search candidates) and
        # record the spec for the idempotent cache.
        try:
            marker_path.write_text(json.dumps({"spec": spec_key, "info": info}, indent=2))
        except Exception:
            logger.exception("[Derive] could not write marker for %s", name)

        tiler_cache.pop(name, None)
        _catalog_runtime["router"] = None
        _catalog_runtime["mtime"] = None
        try:
            build_catalog_index(data_root=data_root, out_dir=data_root / "_catalog", sync_pgvector=False)
        except Exception:
            logger.exception("[Derive] catalogue reindex failed for %s", name)
        _catalog_runtime["router"] = None
        _catalog_runtime["mtime"] = None

        return info

    def _run_derive_job(*, job_id: str, spec: Dict[str, Any]) -> None:
        try:
            _set_build_job(job_id, status="running", step="joining",
                           message="Spatially joining datasets...")
            info = _run_derive(spec)
            _set_build_job(job_id, status="completed", step="done",
                           message=f"Derived dataset '{info['dataset']}' is ready.",
                           **info)
        except Exception as e:
            logger.exception("[DeriveJob] Failed for spec=%s", spec)
            _set_build_job(job_id, status="failed", step="error", message=str(e))

    def _build_derive_response(
        info: Dict[str, Any], query: str, interaction_id: str, assistant_response: str
    ) -> Dict[str, Any]:
        """Build the chat-style 'derive complete' payload from a finished derive.

        Identical shape to a normal initial-turn response (selected_dataset =
        the derived dataset, styled by the derived attribute), so the client
        renders it through the same path.
        """
        derived = info["dataset"]
        attr = info["output_attribute"]
        d_summary = _dataset_summary_for_llm(derived)
        d_geom = _infer_geometry_kind_from_summary(d_summary)
        raw_style = {
            "target_attribute": attr,
            "style_type": f"{_GEOM_PREFIX.get(d_geom, 'fill')}-categorical",
            "color_theme": {"name": "category", "colors": _categorical_palette()},
            "opacity": 0.85, "stroke_width": 1.0, "radius": 4.0,
            "legend_title": attr.replace("_", " ").title(),
            "notes": [f"{info['aggregate']} of {info.get('value_attribute') or 'features'} from {info['source_dataset']}"],
        }
        d_style = _normalize_style_for_client(dataset=derived, dataset_summary=d_summary, style=raw_style)
        d_layer = {
            "dataset": derived, "score": 1.0,
            "reason": f"Spatially joined {info['target_dataset']} x {info['source_dataset']} ({info['aggregate']}).",
            "geometry_kind": d_geom, "selected_attributes": [attr],
            "style_intent": f"{info['aggregate']} {attr}", "style": d_style,
        }
        render_ms = _warm_tile_render_ms([derived])
        fallback_msg = f"Joined {info['target_dataset']} with {info['source_dataset']} and colored by {attr.replace('_', ' ')}."
        return {
            "mode": "initial", "query": query,
            "interaction_id": interaction_id,
            "assistant_response": _normalize_unicode_text(assistant_response or fallback_msg),
            "selection_mode": "derive",
            "selected_dataset": derived, "selected_dataset_score": 1.0,
            "selected_attributes": [attr], "style_intent": f"{info['aggregate']} {attr}",
            "style": d_style, "layers": [d_layer], "derive": info,
            "timings": {
                "derive_ms": round(info.get("derive_seconds", 0.0) * 1000.0, 1),
                "mvt_render_ms": render_ms,
            },
        }

    def _run_chat_derive_job(
        *, job_id: str, spec: Dict[str, Any], query: str,
        interaction_id: str, assistant_response: str,
    ) -> None:
        """Background derive for a chat-style turn: run the (slow) join+tile, then
        store the ready-to-render style payload on the job for the client to poll."""
        try:
            _set_build_job(job_id, status="running", step="joining",
                           message="Combining datasets (spatial join + tiling)...")
            info = _run_derive(spec)
            payload = _build_derive_response(info, query, interaction_id, assistant_response)
            _set_build_job(job_id, status="completed", step="done",
                           message=f"Derived dataset '{info['dataset']}' is ready.",
                           dataset=info["dataset"], result=payload)
        except Exception as e:  # noqa: BLE001
            logger.exception("[ChatDeriveJob] Failed for spec=%s", spec)
            _set_build_job(job_id, status="failed", step="error", message=str(e))

    def get_tiler(dataset: str) -> VectorTiler:
        if dataset not in tiler_cache:
            tiler_cache[dataset] = VectorTiler(
                str(data_root / dataset),
                memory_cache_size=cache_size,
            )
        return tiler_cache[dataset]

    _virtual_derive_cache: Dict[Tuple[str, str, str], Any] = {}

    def _get_virtual_tiler(target: str, source: str, value_attribute: str):
        """Cached lazy-derive tiler (joins `source.value_attribute` into each
        requested tile of `target` at serve time — nothing is materialized)."""
        key = (target, source, value_attribute)
        vt = _virtual_derive_cache.get(key)
        if vt is None:
            from .derive.spatial_aggregate import load_dataset_gdf
            from .tiler.virtual_derive import VirtualDeriveTiler
            src_gdf = load_dataset_gdf(str(data_root / source))
            vt = VirtualDeriveTiler(
                str(data_root / target), src_gdf, value_attribute,
                memory_cache_size=cache_size,
            )
            _virtual_derive_cache[key] = vt
        return vt

    def _catalog_index_path() -> Path:
        return data_root / "_catalog" / CATALOG_FILENAME

    def _catalog_backend_from_env() -> SearchBackend:
        value = os.environ.get("CATALOG_SEARCH_BACKEND", "auto").strip().lower()
        if value == "pgvector":
            return SearchBackend.PGVECTOR
        if value == "npy":
            return SearchBackend.NPY
        return SearchBackend.AUTO

    def _build_catalog_router() -> CatalogRouter:
        index_path = _catalog_index_path()
        if not index_path.exists():
            raise FileNotFoundError(
                f"Catalogue index not found at {index_path}. "
                "Build it first with catalog/index.py."
            )

        embedder = get_embedder()

        # Self-heal: the on-disk embeddings must share the active embedder's
        # vector space, otherwise cosine similarity is meaningless (or crashes
        # on a dimension mismatch). This happens whenever the key state flips
        # between runs (e.g. Gemini index served with the local fallback, or
        # vice-versa). Detect a mismatch and rebuild the index in place.
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                stored = json.load(f)
            stored_id = stored.get("embedder_id")
            stored_dim = stored.get("embedding_dim")
            active_id = getattr(embedder, "embedder_id", None)
            active_dim = getattr(embedder, "embedding_dim", None)
            mismatch = (
                (stored_id is not None and active_id is not None and stored_id != active_id)
                or (stored_dim is not None and active_dim is not None and int(stored_dim) != int(active_dim))
            )
            if mismatch:
                logger.warning(
                    "[Catalog] Stored embedder (%s, dim=%s) differs from active "
                    "embedder (%s, dim=%s); rebuilding catalogue index.",
                    stored_id, stored_dim, active_id, active_dim,
                )
                build_catalog_index(
                    data_root=data_root,
                    out_dir=data_root / "_catalog",
                    embedder=embedder,
                    sync_pgvector=False,
                )
        except Exception:
            logger.exception("[Catalog] Could not verify/repair embedder compatibility")

        backend = _catalog_backend_from_env()

        pg_store = None
        if backend in (SearchBackend.AUTO, SearchBackend.PGVECTOR):
            pg_store = PgVectorStore(PgVectorConfig())

        return CatalogRouter(
            index_dir_or_file=str(index_path.parent),
            embedder=embedder,
            backend=backend,
            pgvector_store=pg_store,
        )

    def _get_catalog_router() -> CatalogRouter:
        index_path = _catalog_index_path()
        if not index_path.exists():
            raise FileNotFoundError(
                f"Catalogue index not found at {index_path}. "
                "Build it first with catalog/index.py."
            )

        mtime = index_path.stat().st_mtime
        if _catalog_runtime["router"] is None or _catalog_runtime["mtime"] != mtime:
            _catalog_runtime["router"] = _build_catalog_router()
            _catalog_runtime["mtime"] = mtime
        return _catalog_runtime["router"]

    def _load_catalog_index_json() -> Dict[str, Any]:
        index_path = _catalog_index_path()
        if not index_path.exists():
            return {"entries": [], "entry_count": 0}
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_catalog_entry_map() -> Dict[str, Dict[str, Any]]:
        catalog = _load_catalog_index_json()
        entries = catalog.get("entries") or []
        return {str(entry.get("dataset")): entry for entry in entries if isinstance(entry, dict)}

    def _load_stats_for_dataset(dataset: str) -> Dict[str, Any]:
        dataset_path = data_root / dataset
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset not found: {dataset}")

        stats_path = dataset_path / "stats" / "attributes.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats not found for dataset: {dataset}")

        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _dataset_exists(dataset: str) -> bool:
        path = data_root / dataset
        return path.exists() and path.is_dir()

    def _dataset_row_count(dataset: str) -> int:
        """Total feature count from parquet metadata (no geometry decode)."""
        import pyarrow.parquet as pq
        total = 0
        for p in (data_root / dataset / "parquet_tiles").glob("*.parquet"):
            try:
                total += pq.read_metadata(str(p)).num_rows
            except Exception:
                pass
        return total

    def _summarize_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
        attributes = stats.get("attributes") or []
        geometry_types: List[str] = []
        geometry_attribute_count = 0
        non_geometry_attribute_count = 0
        sample_attributes: List[str] = []

        for attr in attributes:
            if not isinstance(attr, dict):
                continue

            attr_name = _normalize_unicode_text(attr.get("name"))
            stats_obj = attr.get("stats") or {}

            if "geom_types" in stats_obj:
                geometry_attribute_count += 1
                for geom_type in (stats_obj.get("geom_types") or {}).keys():
                    geom_type_lc = str(geom_type).lower()
                    if geom_type_lc not in geometry_types:
                        geometry_types.append(geom_type_lc)
            else:
                non_geometry_attribute_count += 1
                if attr_name:
                    sample_attributes.append(attr_name)

        return {
            "attribute_count": non_geometry_attribute_count,
            "geometry_attribute_count": geometry_attribute_count,
            "geometry_types": geometry_types,
            "sample_attributes": sample_attributes[:12],
        }

    def _dataset_metadata(dataset: str) -> Dict[str, Any]:
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset not found: {dataset}")

        size = sum(f.stat().st_size for f in dataset_path.rglob("*") if f.is_file())
        file_count = sum(1 for f in dataset_path.rglob("*") if f.is_file())

        bbox = None
        geometry_kind = "unknown"
        try:
            from catalyst._types import Dataset as _Dataset
            bbox = _Dataset(str(dataset_path)).bbox
        except Exception:
            bbox = None
        try:
            geometry_kind = _infer_geometry_kind_from_summary(_dataset_summary_for_llm(dataset))
        except Exception:
            geometry_kind = "unknown"

        return {
            "id": dataset,
            "name": dataset.replace("_", " ").title(),
            "size": size,
            "size_human": _human_size(size),
            "file_count": file_count,
            "path": str(dataset_path),
            "bbox": list(bbox) if bbox else None,
            "geometry_kind": geometry_kind,
        }

    def _list_dataset_metadata(query: Optional[str] = None) -> List[Dict[str, Any]]:
        datasets: List[Dict[str, Any]] = []
        if not data_root.exists():
            return datasets

        query_lc = _normalize_unicode_text(query).strip().lower()
        catalog_map = _load_catalog_entry_map()

        for d in sorted(data_root.iterdir()):
            if not d.is_dir():
                continue
            if d.name.startswith("."):
                continue
            if d.name in {"_catalog", "_uploads"}:
                continue

            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            entry = catalog_map.get(d.name) or {}
            summary = entry.get("summary") or {}

            item = {
                "id": d.name,
                "name": d.name.replace("_", " ").title(),
                "size": size,
                "size_human": _human_size(size),
                "file_count": sum(1 for f in d.rglob("*") if f.is_file()),
                "attribute_count": summary.get("attribute_count"),
                "geometry_attribute_count": summary.get("geometry_attribute_count"),
                "geometry_types": [
                    geom_type
                    for geom in (summary.get("geometry") or [])
                    for geom_type in (geom.get("geom_types") or {}).keys()
                ],
            }

            if not query_lc or query_lc in item["id"].lower() or query_lc in item["name"].lower():
                datasets.append(item)

        return datasets

    def _infer_geometry_kind_from_summary(summary: Dict[str, Any]) -> str:
        geometry = summary.get("geometry") or []
        geom_types: List[str] = []

        for geom_attr in geometry:
            stats = geom_attr.get("geom_types") or {}
            geom_types.extend(str(k).lower() for k in stats.keys())

        joined = " ".join(geom_types)
        if any(x in joined for x in ("line", "multiline")):
            return "line"
        if any(x in joined for x in ("polygon", "multipolygon")):
            return "polygon"
        if any(x in joined for x in ("point", "multipoint")):
            return "point"

        dataset_name = str(summary.get("dataset", "")).lower()
        if "road" in dataset_name or "rail" in dataset_name:
            return "line"
        if any(x in dataset_name for x in ("county", "state", "tract")):
            return "polygon"
        if "point" in dataset_name:
            return "point"
        return "unknown"

    def _attribute_role_from_summary(attr: Dict[str, Any]) -> str:
        role = str(attr.get("role", "")).strip().lower()
        if role:
            return role
        if attr.get("min") is not None or attr.get("max") is not None:
            return "numeric"
        if attr.get("top_k"):
            return "categorical"
        return "unknown"

    def _find_attribute_summary(summary: Dict[str, Any], attr_name: str) -> Optional[Dict[str, Any]]:
        for attr in summary.get("attributes") or []:
            if str(attr.get("name")) == attr_name:
                return attr
        return None

    def _normalize_hex_color(color: str, fallback: str) -> str:
        color = str(color or "").strip()
        if re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            return color
        if re.fullmatch(r"#[0-9a-fA-F]{3}", color):  # #f00 -> #ff0000
            return "#" + "".join(ch * 2 for ch in color[1:])
        return fallback

    # Common colour words -> hex, so a request like "make it red" survives even
    # when the model puts the colour only in the theme name / prose and leaves the
    # structured ``colors`` list empty or invalid.
    _COLOR_WORDS = {
        "bright red": "#ff0000", "red": "#e6194b", "crimson": "#dc143c", "scarlet": "#ff2400",
        "dark green": "#006400", "lime": "#bfff00", "green": "#2ca02c",
        "navy": "#001f7f", "sky blue": "#87ceeb", "light blue": "#87ceeb", "blue": "#1f78b4",
        "yellow": "#ffd700", "gold": "#ffd700", "amber": "#ffbf00",
        "orange": "#ff7f0e", "purple": "#9467bd", "violet": "#8a2be2", "indigo": "#4b0082",
        "magenta": "#ff00ff", "pink": "#ff69b4", "brown": "#8c564b", "teal": "#17becf",
        "cyan": "#00bcd4", "turquoise": "#40e0d0", "black": "#222222", "white": "#ffffff",
        "gray": "#7f7f7f", "grey": "#7f7f7f", "silver": "#c0c0c0",
    }

    def _color_word_to_hex(text: str) -> str:
        t = str(text or "").strip().lower().replace("_", " ").replace("-", " ")
        if not t:
            return ""
        if t in _COLOR_WORDS:
            return _COLOR_WORDS[t]
        # longest phrase first so "bright red"/"sky blue" beat "red"/"blue"
        for word in sorted(_COLOR_WORDS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(word)}\b", t):
                return _COLOR_WORDS[word]
        return ""

    def _categorical_palette() -> List[str]:
        return [
            "#1f78b4",
            "#33a02c",
            "#e31a1c",
            "#ff7f00",
            "#6a3d9a",
            "#b15928",
            "#a6cee3",
            "#b2df8a",
        ]

    def _gradient_palette(default_name: str, colors: List[str]) -> Dict[str, Any]:
        sane = [_normalize_hex_color(c, "#4682B4") for c in (colors or [])]
        sane = [c for c in sane if c]
        if not sane:
            sane = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
        return {
            "name": default_name,
            "colors": sane,
        }

    def _dataset_summary_for_llm(dataset: str) -> Dict[str, Any]:
        stats = _load_stats_for_dataset(dataset)

        selected_summary: Dict[str, Any] = {
            "dataset": dataset,
            "geometry": [],
            "attributes": [],
        }

        for attr in stats.get("attributes") or []:
            if not isinstance(attr, dict):
                continue

            stats_obj = attr.get("stats") or {}
            name = _normalize_unicode_text(attr.get("name"))
            entry: Dict[str, Any] = {"name": name}

            if "geom_types" in stats_obj:
                entry["geom_types"] = stats_obj.get("geom_types") or {}
                selected_summary["geometry"].append(entry)
                continue

            if any(k in stats_obj for k in ("min", "max", "mean", "stddev")):
                entry["role"] = "numeric"
                entry["min"] = stats_obj.get("min")
                entry["max"] = stats_obj.get("max")
                entry["top_k"] = stats_obj.get("top_k") or []
            elif stats_obj.get("top_k") is not None:
                entry["role"] = "categorical"
                entry["top_k"] = stats_obj.get("top_k") or []
            else:
                entry["role"] = "unknown"

            selected_summary["attributes"].append(entry)

        return selected_summary

    def _normalize_style_for_client(
        dataset: str,
        dataset_summary: Dict[str, Any],
        style: Dict[str, Any],
    ) -> Dict[str, Any]:
        geometry_kind = _infer_geometry_kind_from_summary(dataset_summary)
        style_type = str(style.get("style_type", "")).strip() or (
            "line-single-color"
            if geometry_kind == "line"
            else "fill-single-color"
            if geometry_kind == "polygon"
            else "circle-single-color"
            if geometry_kind == "point"
            else "line-single-color"
        )

        target_attribute = _normalize_unicode_text(style.get("target_attribute", "")).strip()
        attr_summary = _find_attribute_summary(dataset_summary, target_attribute) if target_attribute else None
        attr_role = _attribute_role_from_summary(attr_summary or {})
        style_type_l = style_type.lower()
        # An explicit single/uniform/solid intent wins over the attribute's role:
        # "make every zone red" must not be re-expanded into a per-category palette
        # just because the (still-set) target attribute happens to be categorical.
        wants_single = any(w in style_type_l for w in ("single", "uniform", "solid", "simple", "flat"))
        is_gradient = ("gradient" in style_type_l) and not wants_single
        is_categorical = (
            not wants_single and not is_gradient
            and ("categorical" in style_type_l or attr_role in {"categorical", "categorical_text"})
        )

        color_theme = style.get("color_theme") or {}
        theme_name = _normalize_unicode_text(color_theme.get("name", "")).strip() or "custom"
        theme_colors = color_theme.get("colors") or []

        opacity = _safe_float(style.get("opacity", 1.0), 1.0)
        stroke_width = _safe_float(style.get("stroke_width", 2.0), 2.0)
        radius = _safe_float(style.get("radius", 4.0), 4.0)
        legend_title = _normalize_unicode_text(style.get("legend_title", "")).strip() or target_attribute or dataset
        notes = style.get("notes") or []
        if not isinstance(notes, list):
            notes = [str(notes)]

        if is_categorical:
            categorical_values: List[str] = []
            if attr_summary:
                for item in attr_summary.get("top_k") or []:
                    value = item.get("value") if isinstance(item, dict) else item
                    if value is None:
                        continue
                    categorical_values.append(_normalize_unicode_text(value))

            # Keep only the valid, DISTINCT hex colors the model provided; invalid
            # colors are dropped (not collapsed to one fallback, which would make
            # every category identical). Supplement from the qualitative palette so
            # each category gets a distinct color.
            palette: List[str] = []
            for c in theme_colors:
                hexc = _normalize_hex_color(c, "")
                if hexc and hexc not in palette:
                    palette.append(hexc)
            needed = max(len(categorical_values), 1)
            if len(palette) < needed:
                for c in _categorical_palette():
                    if c not in palette:
                        palette.append(c)
                    if len(palette) >= needed:
                        break
            if not palette:
                palette = _categorical_palette()

            stops = [
                {"value": value, "color": palette[i % len(palette)]}
                for i, value in enumerate(categorical_values)
            ]

            return {
                "dataset": dataset,
                "geometry_kind": geometry_kind,
                "style_type": style_type,
                "target_attribute": target_attribute,
                "legend_title": legend_title,
                "opacity": opacity,
                "stroke_width": stroke_width,
                "radius": radius,
                "color_theme": {
                    "name": theme_name,
                    "colors": palette,
                },
                "renderer": {
                    "mode": "categorical",
                    "attribute": target_attribute,
                    "fallback_color": palette[0],
                    "stops": stops,
                },
                "notes": [_normalize_unicode_text(n) for n in notes],
            }

        if is_gradient:
            palette = _gradient_palette(theme_name or "gradient", list(theme_colors))
            min_value = attr_summary.get("min") if attr_summary else None
            max_value = attr_summary.get("max") if attr_summary else None

            min_value = _safe_float(min_value, 0.0)
            max_value = _safe_float(max_value, 1.0)
            if max_value == min_value:
                max_value = min_value + 1.0

            return {
                "dataset": dataset,
                "geometry_kind": geometry_kind,
                "style_type": style_type,
                "target_attribute": target_attribute,
                "legend_title": legend_title,
                "opacity": opacity,
                "stroke_width": stroke_width,
                "radius": radius,
                "color_theme": palette,
                "renderer": {
                    "mode": "gradient",
                    "attribute": target_attribute,
                    "min": min_value,
                    "max": max_value,
                    "colors": palette["colors"],
                },
                "notes": [_normalize_unicode_text(n) for n in notes],
            }

        # Resolve the single colour, recovering one named only in the theme
        # name / legend / notes (e.g. model returns name="bright-red" with an
        # empty/invalid colours list) rather than falling back to the default blue.
        single_color = ""
        for cand in list(theme_colors) + [theme_name, legend_title] + [str(n) for n in notes]:
            single_color = _normalize_hex_color(cand, "") or _color_word_to_hex(cand)
            if single_color:
                break
        fallback_color = single_color or "#4682B4"

        return {
            "dataset": dataset,
            "geometry_kind": geometry_kind,
            "style_type": style_type,
            "target_attribute": target_attribute,
            "legend_title": legend_title,
            "opacity": opacity,
            "stroke_width": stroke_width,
            "radius": radius,
            "color_theme": {
                "name": theme_name,
                "colors": [fallback_color],
            },
            "renderer": {
                "mode": "single",
                "attribute": target_attribute,
                "color": fallback_color,
            },
            "notes": [_normalize_unicode_text(n) for n in notes],
        }

    def _candidate_payload_for_llm(candidates) -> List[Dict[str, Any]]:
        payload = []
        for c in candidates:
            payload.append(
                {
                    "dataset": c.dataset,
                    "score": round(float(c.score), 6),
                    "summary": c.summary,
                }
            )
        return payload

    # ----- Cross-dataset overlay (novelty) -----------------------------------
    # Distinct per-layer color ramps so overlaid datasets stay visually separable.
    _LAYER_THEMES = [
        {"name": "blues", "colors": ["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"]},
        {"name": "oranges", "colors": ["#feedde", "#fdbe85", "#fd8d3c", "#e6550d", "#a63603"]},
        {"name": "greens", "colors": ["#edf8e9", "#bae4b3", "#74c476", "#31a354", "#006d2c"]},
    ]
    _LAYER_SOLID = ["#3182bd", "#e6550d", "#31a354"]
    _GEOM_PREFIX = {"polygon": "fill", "line": "line", "point": "circle", "unknown": "fill"}
    _GEOM_ORDER = {"polygon": 0, "line": 1, "point": 2, "unknown": 1}

    def _safe_geometry_kind(dataset: str) -> str:
        try:
            return _infer_geometry_kind_from_summary(_dataset_summary_for_llm(dataset))
        except Exception:
            return "unknown"

    def _heuristic_pick_attribute(query: str, summary: Dict[str, Any]) -> Tuple[str, str]:
        """Pick (target_attribute, render_mode) from query intent + attribute roles."""
        q = (query or "").lower()
        attrs = summary.get("attributes") or []
        numeric = [a for a in attrs if _attribute_role_from_summary(a) == "numeric"]
        categorical = [
            a for a in attrs
            if _attribute_role_from_summary(a) in {"categorical", "categorical_text"}
        ]
        wants_cat = any(w in q for w in (
            "type", "types", "categor", "class", "kind", "group", "species", "land use", "landuse",
        ))
        wants_grad = any(w in q for w in (
            "density", "magnitude", "gradient", "heat", "amount", "count", "population",
            "by area", "size", "level", "value", "intensity", "elevation",
        ))
        if wants_cat and categorical:
            return categorical[0]["name"], "categorical"
        if wants_grad and numeric:
            return numeric[0]["name"], "gradient"
        if categorical:
            return categorical[0]["name"], "categorical"
        if numeric:
            return numeric[0]["name"], "gradient"
        return "", "single"

    def _normalized_layer_for(
        dataset: str,
        *,
        query: str = "",
        layer_index: int = 0,
        total_layers: int = 1,
        score: float = 0.0,
        reason: str = "",
        raw_style: Optional[Dict[str, Any]] = None,
        selected_attributes: Optional[List[str]] = None,
        style_intent: str = "",
    ) -> Dict[str, Any]:
        """Build one response layer (normalized style) for the overlay.

        If ``raw_style`` is provided (LLM path) it is normalized as-is; otherwise a
        style is synthesized heuristically from the query and attribute roles.
        """
        summary = _dataset_summary_for_llm(dataset)
        geom = _infer_geometry_kind_from_summary(summary)
        prefix = _GEOM_PREFIX.get(geom, "fill")

        if raw_style is None:
            target, mode = _heuristic_pick_attribute(query, summary)
            style_type = f"{prefix}-{mode}"
            if mode == "categorical":
                theme = {"name": "category", "colors": _categorical_palette()}
            elif mode == "gradient":
                theme = _LAYER_THEMES[layer_index % len(_LAYER_THEMES)]
            else:
                theme = {"name": "solid", "colors": [_LAYER_SOLID[layer_index % len(_LAYER_SOLID)]]}
            # Bottom polygon layers get lower opacity so upper layers show through.
            opacity = 0.55 if (geom == "polygon" and total_layers > 1) else 0.85
            raw_style = {
                "target_attribute": target,
                "style_type": style_type,
                "color_theme": theme,
                "opacity": opacity,
                "stroke_width": 2.5 if geom == "line" else 1.5,
                "radius": 3.5,
                "legend_title": target or dataset.replace("_", " ").title(),
                "notes": [reason] if reason else [],
            }
            if not selected_attributes:
                selected_attributes = [target] if target else []
            if not style_intent:
                style_intent = f"heuristic {style_type} for {dataset}"

        normalized = _normalize_style_for_client(
            dataset=dataset,
            dataset_summary=summary,
            style=raw_style,
        )
        return {
            "dataset": dataset,
            "score": round(float(score), 6),
            "reason": _normalize_unicode_text(reason),
            "geometry_kind": geom,
            "selected_attributes": [_normalize_unicode_text(x) for x in (selected_attributes or [])],
            "style_intent": _normalize_unicode_text(style_intent),
            "style": normalized,
        }

    def _select_overlay_candidates(candidates, max_layers: int = 3) -> List[Any]:
        """Pick the jointly-relevant subset, ordered bottom (areas) -> top (points).

        A candidate joins the overlay when its similarity is a meaningful fraction
        of the best match (relative threshold) and clears a small absolute floor
        (to avoid pulling in clearly-irrelevant datasets). The top match is always
        included. Lexical/embedding scores are noisy, so the relative threshold is
        deliberately permissive — the goal is to surface jointly-relevant datasets
        for overlay, which is exactly the cross-dataset behaviour reviewers asked
        Catalyst to demonstrate.
        """
        if not candidates:
            return []
        top = float(candidates[0].score)
        rel_floor = 0.4 * top
        abs_floor = 0.04
        chosen = []
        for i, c in enumerate(candidates):
            score = float(c.score)
            if i == 0 or (score >= rel_floor and score >= abs_floor):
                chosen.append(c)
            if len(chosen) >= max_layers:
                break
        return sorted(chosen, key=lambda c: _GEOM_ORDER.get(_safe_geometry_kind(c.dataset), 1))

    def _heuristic_overlay_layers(query: str, candidates, max_layers: int = 3) -> List[Dict[str, Any]]:
        chosen = _select_overlay_candidates(candidates, max_layers=max_layers)
        total = len(chosen)
        layers: List[Dict[str, Any]] = []
        for idx, c in enumerate(chosen):
            reason = (
                f"Top semantic match for the request."
                if idx == 0 and c is candidates[0]
                else f"Also relevant (similarity {float(c.score):.2f}); overlaid for context."
            )
            layers.append(
                _normalized_layer_for(
                    c.dataset,
                    query=query,
                    layer_index=idx,
                    total_layers=total,
                    score=c.score,
                    reason=reason,
                )
            )
        return layers

    def _llm_overlay_layers(multi_result, candidate_by_name) -> List[Dict[str, Any]]:
        layers: List[Dict[str, Any]] = []
        for idx, spec in enumerate(multi_result.layers):
            if spec.dataset not in candidate_by_name:
                continue
            score = float(getattr(candidate_by_name[spec.dataset], "score", 0.0))
            layers.append(
                _normalized_layer_for(
                    spec.dataset,
                    layer_index=idx,
                    total_layers=len(multi_result.layers),
                    score=score,
                    reason=spec.reason,
                    raw_style=spec.style,
                    selected_attributes=spec.selected_attributes,
                    style_intent=spec.style_intent,
                )
            )
        return layers

    def _warm_tile_render_ms(datasets: List[str]) -> Optional[float]:
        """Render one representative tile per dataset and return the max latency.

        Gives an immediate, server-measured MVT render time for the timing panel
        (reviewers asked specifically for the MVT rendering cost), and warms the
        on-disk/cache tiers so the first browser fetch is fast.
        """
        import math as _math

        def _lonlat_to_tile(lon: float, lat: float, z: int) -> Tuple[int, int]:
            lat = max(min(lat, 85.05112878), -85.05112878)
            n = 2 ** z
            xt = int((lon + 180.0) / 360.0 * n)
            lat_rad = _math.radians(lat)
            yt = int((1.0 - _math.asinh(_math.tan(lat_rad)) / _math.pi) / 2.0 * n)
            return min(max(xt, 0), n - 1), min(max(yt, 0), n - 1)

        worst: Optional[float] = None
        for dataset in datasets:
            try:
                from catalyst._types import Dataset as _Dataset
                bbox = _Dataset(str(data_root / dataset)).bbox
                if not bbox:
                    continue
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                z = 7
                x, y = _lonlat_to_tile(cx, cy, z)
                tiler = get_tiler(dataset)
                t0 = perf_counter()
                tiler.get_tile(z, x, y)
                elapsed = (perf_counter() - t0) * 1000.0
                worst = elapsed if worst is None else max(worst, elapsed)
            except Exception:
                logger.exception("[Overlay] tile warm failed for %s", dataset)
        return round(worst, 3) if worst is not None else None

    def _looks_like_new_dataset_request(
        user_query: str,
        current_dataset: Optional[str],
        current_style: Optional[Dict[str, Any]],
    ) -> bool:
        q = _normalize_unicode_text(user_query).strip().lower()
        if not q:
            return False

        if any(
            phrase in q
            for phrase in [
                "use a different dataset",
                "switch dataset",
                "another dataset",
                "different dataset",
                "new dataset",
            ]
        ):
            return True

        if current_dataset and current_dataset.lower() in q:
            return False

        domain_triggers = [
            "county", "counties", "state", "states", "tract", "tracts",
            "road", "roads", "rail", "rails", "building", "buildings",
            "point", "points", "landmark", "landmarks",
        ]
        if any(word in q for word in domain_triggers):
            target_attr = ""
            if current_style:
                target_attr = str(current_style.get("target_attribute", "")).lower()
            current_dataset_lc = (current_dataset or "").lower()
            if current_dataset_lc and current_dataset_lc not in q and target_attr and target_attr not in q:
                return True

        return False

    def _run_initial_chat_turn(user_query: str, k: int = 5) -> Tuple[Dict[str, Any], int]:
        total_t0 = perf_counter()
        normalized_query = _normalize_unicode_text(user_query)

        router = _get_catalog_router()

        search_t0 = perf_counter()
        candidates = router.search(normalized_query, k=k)
        semantic_search_ms = (perf_counter() - search_t0) * 1000.0

        if not candidates:
            raise LookupError("No indexed datasets available")

        candidate_payload = _candidate_payload_for_llm(candidates)
        candidate_by_name = {c.dataset: c for c in candidates}

        # --- Cross-dataset overlay selection + styling --------------------------
        # Try the LLM to compose a multi-dataset overlay; gracefully fall back to
        # a deterministic heuristic if no provider/key is available. This keeps the
        # whole demo runnable offline and robust to LLM outages.
        llm_t0 = perf_counter()
        selection_mode = "llm"
        assistant_response = ""
        interaction_id = ""
        layers: List[Dict[str, Any]] = []
        try:
            multi = start_multilayer_conversation(
                candidates_summary=candidate_payload,
                user_query=normalized_query,
                max_layers=3,
                provider_name=_LLM_PROVIDER,
            )
            # Spatial-join / derived-attribute request: building+tiling a new
            # dataset can take minutes for large targets, so run it as a
            # background job (the client polls GET /api/upload-dataset/<job_id>)
            # instead of blocking the request (which would exceed proxy/browser
            # timeouts and return a non-JSON 504).
            if getattr(multi, "derive", None):
                spec = dict(multi.derive)
                interaction_id = _normalize_unicode_text(multi.interaction_id)
                assistant_response = _normalize_unicode_text(multi.assistant_response or "")
                derive_target = _normalize_unicode_text(str(spec.get("target_dataset", ""))).strip()
                derive_source = _normalize_unicode_text(str(spec.get("source_dataset", ""))).strip()
                derive_value = _normalize_unicode_text(str(spec.get("value_attribute", ""))).strip()

                # LIVE lazy derive: when a value/categorical source is small, color
                # the target by the source value at each feature's location, joined
                # per tile at serve time (no build, no wait). Instant + interactive.
                if (
                    derive_target and derive_source and derive_value
                    and _dataset_exists(derive_target) and _dataset_exists(derive_source)
                    and 0 < _dataset_row_count(derive_source) <= _TILED_DERIVE_MAX_SOURCE
                ):
                    try:
                        tgt_summary = _dataset_summary_for_llm(derive_target)
                        src_summary = _dataset_summary_for_llm(derive_source)
                        src_attr = _find_attribute_summary(src_summary, derive_value)
                        syn = {
                            "dataset": derive_target,
                            "geometry": tgt_summary.get("geometry", []),
                            "attributes": list(tgt_summary.get("attributes", [])),
                        }
                        if src_attr:
                            syn["attributes"].append(src_attr)
                        geom_kind = _infer_geometry_kind_from_summary(syn)
                        raw_style = {
                            "target_attribute": derive_value,
                            "style_type": f"{_GEOM_PREFIX.get(geom_kind, 'fill')}-categorical",
                            "color_theme": {"name": "category", "colors": _categorical_palette()},
                            "opacity": 0.9, "stroke_width": 0.6, "radius": 3.0,
                            "legend_title": derive_value.replace("_", " ").title(),
                        }
                        live_style = _normalize_style_for_client(
                            dataset=derive_target, dataset_summary=syn, style=raw_style,
                        )
                        msg = assistant_response or (
                            f"Coloring {derive_target} by {derive_value.replace('_', ' ')} "
                            f"from {derive_source} — joined live as you pan/zoom."
                        )
                        return {
                            "mode": "derive-live", "query": normalized_query,
                            "interaction_id": interaction_id,
                            "assistant_response": msg,
                            "selection_mode": "derive-live",
                            "selected_dataset": derive_target,
                            "selected_dataset_score": 1.0,
                            "selected_attributes": [derive_value],
                            "style_intent": f"color {derive_target} by {derive_value}",
                            "style": live_style,
                            "derive_live": {
                                "target": derive_target,
                                "source": derive_source,
                                "value_attribute": derive_value,
                            },
                            "top_k": candidate_payload,
                        }, 200
                    except Exception:
                        logger.exception("[ChatStyle] live-derive setup failed; using materialized build")

                job_id = uuid4().hex
                _set_build_job(
                    job_id, status="queued", step="queued",
                    message="Combining datasets...", spec=spec,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                Thread(
                    target=_run_chat_derive_job,
                    kwargs={
                        "job_id": job_id, "spec": spec, "query": normalized_query,
                        "interaction_id": interaction_id,
                        "assistant_response": assistant_response,
                    },
                    daemon=True,
                ).start()
                pending_msg = assistant_response or (
                    f"Combining {derive_target or 'the target'} with "
                    f"{derive_source or 'the source'} — this can take a minute "
                    "for large datasets…"
                )
                return {
                    "mode": "deriving", "query": normalized_query,
                    "interaction_id": interaction_id,
                    "assistant_response": pending_msg,
                    "selection_mode": "deriving",
                    "derive_job_id": job_id, "derive": spec,
                    "top_k": candidate_payload,
                }, 202
            layers = _llm_overlay_layers(multi, candidate_by_name)
            if not layers:
                raise ValueError("LLM returned no usable layers")
            assistant_response = multi.assistant_response
            interaction_id = _normalize_unicode_text(multi.interaction_id)
        except (LLMProviderError, Exception) as exc:  # noqa: BLE001 - intentional broad fallback
            logger.warning(
                "[ChatStyle] Multilayer LLM unavailable (%s); using heuristic overlay.",
                exc,
            )
            selection_mode = "heuristic"
            layers = _heuristic_overlay_layers(normalized_query, candidates, max_layers=3)
            names = ", ".join(l["dataset"] for l in layers)
            if len(layers) > 1:
                assistant_response = (
                    f"Combined {len(layers)} relevant datasets into one map ({names}). "
                    "Semantic search ranked these as jointly relevant to your request."
                )
            elif layers:
                assistant_response = (
                    f"Showing the most relevant dataset ({names}) for your request."
                )
            else:
                assistant_response = "No relevant datasets were found for your request."
        llm_ms = (perf_counter() - llm_t0) * 1000.0

        if not layers:
            raise LookupError("No relevant datasets for the request")

        # Primary layer = highest-scoring dataset (drives the metadata panel and
        # the backward-compatible single-dataset fields).
        primary = max(layers, key=lambda l: l.get("score", 0.0))
        primary_dataset = primary["dataset"]

        overlay_render_ms = _warm_tile_render_ms([l["dataset"] for l in layers])
        tile_metric = _get_tile_metric(primary_dataset)
        total_ms = (perf_counter() - total_t0) * 1000.0

        response = {
            "mode": "initial",
            "query": normalized_query,
            "interaction_id": interaction_id,
            "assistant_response": _normalize_unicode_text(assistant_response),
            "selection_mode": selection_mode,
            "selected_dataset": primary_dataset,
            "selected_dataset_score": float(primary.get("score", 0.0)),
            "selected_attributes": primary.get("selected_attributes", []),
            "style_intent": primary.get("style_intent", ""),
            "style": primary["style"],
            "layers": layers,
            "top_k": candidate_payload,
            "timings": {
                "semantic_search_ms": round(semantic_search_ms, 3),
                "llm_ms": round(llm_ms, 3),
                "mvt_render_ms": overlay_render_ms,
                "total_ms": round(total_ms, 3),
                "last_tile_request_ms": round(float(tile_metric.get("elapsed_ms")), 3) if tile_metric and tile_metric.get("elapsed_ms") is not None else None,
            },
        }
        return response, 200

    def _run_followup_chat_turn(
        *,
        user_query: str,
        interaction_id: str,
        current_dataset: str,
        current_attributes: Optional[List[str]] = None,
        current_style: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], int]:
        if not current_dataset:
            raise ValueError("current_dataset is required for follow-up turns")
        if not interaction_id:
            raise ValueError("interaction_id is required for follow-up turns")

        total_t0 = perf_counter()
        normalized_query = _normalize_unicode_text(user_query)
        selected_summary = _dataset_summary_for_llm(current_dataset)

        llm_t0 = perf_counter()
        try:
            turn = continue_style_conversation(
                dataset=current_dataset,
                user_query=normalized_query,
                previous_interaction_id=_normalize_unicode_text(interaction_id),
                selected_attributes_hint=[_normalize_unicode_text(x) for x in (current_attributes or [])],
                current_style_hint=current_style or {},
                provider_name=_LLM_PROVIDER,
                temperature=0.2,
            )
        except (LLMProviderError, Exception) as exc:  # noqa: BLE001 - intentional broad fallback
            # Provider unavailable mid-session (e.g. key expired): re-run the
            # query as a fresh overlay rather than failing the turn.
            logger.warning(
                "[ChatStyle] Follow-up LLM unavailable (%s); restarting as overlay.",
                exc,
            )
            return _run_initial_chat_turn(user_query=normalized_query, k=5)
        llm_ms = (perf_counter() - llm_t0) * 1000.0

        returned_dataset = _normalize_unicode_text(turn.selected_dataset).strip() or current_dataset
        if returned_dataset != current_dataset:
            logger.info(
                "[ChatStyle] Follow-up requested dataset switch from '%s' to '%s'; restarting retrieval.",
                current_dataset,
                returned_dataset,
            )
            return _run_initial_chat_turn(user_query=normalized_query, k=5)

        normalized_style = _normalize_style_for_client(
            dataset=current_dataset,
            dataset_summary=selected_summary,
            style=turn.style,
        )

        tile_metric = _get_tile_metric(current_dataset)
        total_ms = (perf_counter() - total_t0) * 1000.0

        followup_layer = {
            "dataset": current_dataset,
            "score": 1.0,
            "reason": _normalize_unicode_text(turn.style_intent),
            "geometry_kind": _infer_geometry_kind_from_summary(selected_summary),
            "selected_attributes": [_normalize_unicode_text(x) for x in (turn.selected_attributes or [])],
            "style_intent": _normalize_unicode_text(turn.style_intent),
            "style": normalized_style,
        }

        response = {
            "mode": "followup",
            "query": normalized_query,
            "interaction_id": _normalize_unicode_text(turn.interaction_id),
            "assistant_response": _normalize_unicode_text(turn.assistant_response),
            "selection_mode": "llm",
            "selected_dataset": current_dataset,
            "selected_attributes": [_normalize_unicode_text(x) for x in (turn.selected_attributes or [])],
            "style_intent": _normalize_unicode_text(turn.style_intent),
            "style": normalized_style,
            # Single refined layer; the client merges it into the existing overlay
            # (updating this dataset's layer, preserving the others).
            "layers": [followup_layer],
            "merge_layers": True,
            "top_k": [],
            "timings": {
                "semantic_search_ms": 0.0,
                "llm_ms": round(llm_ms, 3),
                "mvt_render_ms": _warm_tile_render_ms([current_dataset]),
                "total_ms": round(total_ms, 3),
                "last_tile_request_ms": round(float(tile_metric.get("elapsed_ms")), 3) if tile_metric and tile_metric.get("elapsed_ms") is not None else None,
            },
        }
        return response, 200

    @app.get("/<dataset>/<int:z>/<int:x>/<int:y>.mvt")
    def serve_tile(dataset, z, x, y):
        t0 = perf_counter()
        tiler = get_tiler(dataset)
        data = tiler.get_tile(z, x, y)
        elapsed_ms = (perf_counter() - t0) * 1000.0

        metric = {
            "dataset": dataset,
            "z": z,
            "x": x,
            "y": y,
            "bytes": len(data),
            "elapsed_ms": elapsed_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _set_tile_metric(dataset, metric)

        logger.info(
            "[TileRequest] dataset=%s z=%d x=%d y=%d bytes=%d elapsed=%.1fms",
            dataset,
            z,
            x,
            y,
            len(data),
            elapsed_ms,
        )
        return Response(data, mimetype="application/vnd.mapbox-vector-tile")

    @app.get("/api/derive-live/<target>/<source>/<value_attribute>/<int:z>/<int:x>/<int:y>.mvt")
    def serve_derive_live_tile(target, source, value_attribute, z, x, y):
        """Lazy cross-dataset derive tile: target features annotated with the
        source value at their location, joined on demand (no materialization)."""
        if not _dataset_exists(target) or not _dataset_exists(source):
            return _json_response({"error": "target or source dataset not found"}, 404)
        try:
            tiler = _get_virtual_tiler(target, source, value_attribute)
            data = tiler.get_tile(z, x, y)
            return Response(data, mimetype="application/vnd.mapbox-vector-tile")
        except Exception as e:  # noqa: BLE001
            logger.exception("[DeriveLive] tile failed target=%s source=%s", target, source)
            return _json_response({"error": f"derive-live tile failed: {e}"}, 500)

    @app.get("/datasets/<path:filepath>")
    def serve_dataset_file(filepath):
        full_path = (data_root / filepath).resolve()

        try:
            full_path.relative_to(data_root)
        except ValueError:
            logger.warning("[DatasetFile] Blocked path traversal: %s", full_path)
            return "File not found", 404

        logger.info(
            "[DatasetFile] request=%s resolved=%s exists=%s",
            filepath,
            full_path,
            full_path.exists(),
        )

        if not full_path.exists() or not full_path.is_file():
            return "File not found", 404

        return send_from_directory(str(data_root), filepath)

    @app.get("/api/datasets")
    def list_datasets():
        datasets = sorted(
            [d.name for d in data_root.iterdir() if d.is_dir()]
        ) if data_root.exists() else []
        return app.response_class(
            response=json.dumps({"datasets": datasets}, ensure_ascii=False),
            mimetype="application/json",
        )

    @app.get("/api/catalog/entries")
    def api_catalog_entries():
        query = request.args.get("q", default=None)
        datasets = _list_dataset_metadata(query=query)
        return app.response_class(
            response=json.dumps({"datasets": datasets}, indent=2, ensure_ascii=False),
            mimetype="application/json",
        )

    @app.get("/datasets.json")
    def search_datasets():
        query = request.args.get("q", default=None)
        datasets = _list_dataset_metadata(query=query)
        return app.response_class(
            response=json.dumps({"datasets": datasets}, indent=2, ensure_ascii=False),
            mimetype="application/json",
        )

    @app.get("/datasets/<dataset>.json")
    def get_dataset_metadata(dataset):
        try:
            metadata = _dataset_metadata(dataset)
            return app.response_class(
                response=json.dumps(metadata, indent=2, ensure_ascii=False),
                mimetype="application/json",
            )
        except FileNotFoundError:
            return {"error": "Dataset not found"}, 404
        except Exception as e:
            return {"error": f"Failed to retrieve metadata: {e}"}, 500

    @app.get("/api/datasets/<dataset>/stats")
    def get_dataset_stats(dataset):
        stats_path = data_root / dataset / "stats" / "attributes.json"
        if not stats_path.exists():
            return {"error": "Stats not found for dataset"}, 404
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Failed to load stats: {e}"}, 500

    @app.get("/api/datasets/<dataset>/inspect")
    def inspect_dataset(dataset: str):
        try:
            metadata = _dataset_metadata(dataset)

            stats = None
            stats_summary = {}
            try:
                stats = _load_stats_for_dataset(dataset)
                stats_summary = _summarize_stats(stats)
            except Exception:
                stats = None
                stats_summary = {}

            catalog_entry = _load_catalog_entry_map().get(dataset) or {}
            catalog_summary = catalog_entry.get("summary") or {}

            recent_tile_request = _get_tile_metric(dataset)

            payload = {
                "dataset": dataset,
                "metadata": metadata,
                "catalog_summary": {
                    "dataset": catalog_summary.get("dataset"),
                    "description": catalog_summary.get("description"),
                    "attribute_count": catalog_summary.get("attribute_count"),
                    "geometry_attribute_count": catalog_summary.get("geometry_attribute_count"),
                    "geometry": catalog_summary.get("geometry") or [],
                    "attributes": catalog_summary.get("attributes") or [],
                },
                "stats_summary": stats_summary,
                "recent_tile_request": recent_tile_request,
                "stats_preview": stats,
            }
            return _json_response(payload, 200)
        except FileNotFoundError as e:
            return _json_response({"error": str(e)}, 404)
        except Exception as e:
            logger.exception("[InspectDataset] Failed for dataset=%s", dataset)
            return _json_response({"error": f"Failed to inspect dataset: {e}"}, 500)

    @app.get("/datasets/<dataset>.html")
    def visualize_dataset(dataset):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return "<h1>Dataset not found</h1>", 404
        try:
            return render_template(
                "view_dataset.html",
                dataset_id=dataset,
                dataset_name=dataset.replace("_", " ").title(),
            )
        except Exception as e:
            return f"<h1>Failed to render visualization: {e}</h1>", 500

    @app.get("/datasets/<dataset>/features.<format>")
    def download_features(dataset, format):
        try:
            mbr_string = request.args.get("mbr", default=None)
            feature_stream = feature_service.get_features_stream(dataset, format, mbr_string)
            mime_type = feature_service.get_mime_type(format)
            if mbr_string:
                filename = f"{dataset}_{mbr_string.replace(',', '_')}.{format}"
            else:
                filename = f"{dataset}_full.{format}"
            return Response(
                feature_stream,
                mimetype=mime_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )
        except ValueError as e:
            return {"error": str(e)}, 400
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Internal error: {e}"}, 500

    @app.post("/datasets/<dataset>/features.<format>")
    def download_features_with_geometry(dataset, format):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return {"error": "Dataset not found"}, 404
        try:
            geojson_payload = request.get_json(silent=True)
            mbr_string = request.args.get("mbr", default=None)
            if geojson_payload:
                geometry = geojson_payload.get("geometry")
                if not geometry:
                    return {"error": "Invalid GeoJSON payload: 'geometry' field is required"}, 400
                feature_stream = feature_service.get_features_stream(
                    dataset,
                    format,
                    geometry=geometry,
                )
            else:
                feature_stream = feature_service.get_features_stream(dataset, format, mbr_string)
            mime_type = feature_service.get_mime_type(format)
            filename = f"{dataset}_filtered.{format}" if geojson_payload else f"{dataset}_mbr.{format}"
            return Response(
                feature_stream,
                mimetype=mime_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )
        except ValueError as e:
            return {"error": str(e)}, 400
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Internal error: {e}"}, 500

    @app.get("/datasets/<dataset>/features/sample.json")
    def get_sample_non_geometry_attributes(dataset):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return {"error": "Dataset not found"}, 404
        try:
            mbr_string = request.args.get("mbr", default=None)
            if not mbr_string:
                return {"error": "MBR query parameter is required"}, 400
            sample_record = feature_service.get_sample_record(dataset, mbr_string, include_geometry=False)
            if not sample_record:
                return {"error": "No matching record found"}, 404
            return app.response_class(
                response=json.dumps(sample_record, indent=2, ensure_ascii=False),
                mimetype="application/json",
            )
        except ValueError as e:
            return {"error": str(e)}, 400
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Internal error: {e}"}, 500

    @app.get("/datasets/<dataset>/features/sample.geojson")
    def get_sample_with_geometry(dataset):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return {"error": "Dataset not found"}, 404
        try:
            mbr_string = request.args.get("mbr", default=None)
            if not mbr_string:
                return {"error": "MBR query parameter is required"}, 400
            sample_record = feature_service.get_sample_record(dataset, mbr_string, include_geometry=True)
            if not sample_record:
                return {"error": "No matching record found"}, 404
            return app.response_class(
                response=json.dumps(sample_record, indent=2, ensure_ascii=False),
                mimetype="application/json",
            )
        except ValueError as e:
            return {"error": str(e)}, 400
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Internal error: {e}"}, 500

    @app.get("/")
    def index():
        logger.info("Serving index page")
        return render_template("index.html", base_path=_BASE_PATH)

    @app.get("/map.html")
    def map_page():
        logger.info("Serving map runtime page")
        return render_template("map.html", base_path=_BASE_PATH)

    @app.route("/<path:filename>")
    def serve_file(filename):
        normalized = filename.strip("/")

        if normalized.startswith("datasets/") or normalized.startswith("api/"):
            return "File not found", 404

        file_path = (server_dir / normalized).resolve()

        try:
            file_path.relative_to(server_dir)
        except ValueError:
            return "File not found", 404

        if file_path.exists() and file_path.is_file():
            return send_from_directory(str(server_dir), normalized)

        return "File not found", 404

    @app.post("/api/chat-style")
    def chat_style():
        body = request.get_json(silent=True) or {}

        user_query = _normalize_unicode_text(body.get("query", "")).strip()
        if not user_query:
            return _json_response({"error": "Request body must include a non-empty 'query'"}, 400)

        interaction_id = _normalize_unicode_text(body.get("interaction_id", "")).strip()
        current_dataset = _normalize_unicode_text(body.get("current_dataset", "")).strip()

        current_attributes_raw = body.get("current_attributes")
        if isinstance(current_attributes_raw, list):
            current_attributes = [
                _normalize_unicode_text(x)
                for x in current_attributes_raw
                if isinstance(x, (str, int, float))
            ]
        else:
            current_attributes = []

        current_style = body.get("current_style")
        if not isinstance(current_style, dict):
            current_style = {}

        requested_k = body.get("k", 5)
        try:
            k = max(1, min(int(requested_k), 10))
        except Exception:
            k = 5

        try:
            if not interaction_id or not current_dataset:
                response, status = _run_initial_chat_turn(user_query=user_query, k=k)
                return _json_response(response, status)

            if _looks_like_new_dataset_request(
                user_query=user_query,
                current_dataset=current_dataset,
                current_style=current_style,
            ):
                response, status = _run_initial_chat_turn(user_query=user_query, k=k)
                return _json_response(response, status)

            if not _dataset_exists(current_dataset):
                return _json_response({"error": f"Current dataset not found: {current_dataset}"}, 404)

            response, status = _run_followup_chat_turn(
                user_query=user_query,
                interaction_id=interaction_id,
                current_dataset=current_dataset,
                current_attributes=current_attributes,
                current_style=current_style,
            )
            return _json_response(response, status)

        except FileNotFoundError as e:
            return _json_response({"error": str(e)}, 503)
        except LookupError as e:
            return _json_response({"error": str(e)}, 404)
        except Exception as e:
            logger.exception("[ChatStyle] Failed for query=%r", user_query)
            return _json_response({"error": f"Chat styling failed: {e}"}, 500)

    @app.post("/api/upload-dataset")
    def upload_dataset_and_build():
        try:
            uploaded = request.files.get("file")
            if uploaded is None or not uploaded.filename:
                return _json_response({"error": "A dataset file is required under form field 'file'."}, 400)

            dataset_name_raw = _normalize_unicode_text(request.form.get("dataset_name", uploaded.filename)).strip()
            dataset_name = _slugify_dataset_name(dataset_name_raw)

            num_tiles = max(1, _safe_int(request.form.get("num_tiles", 40), 40))
            zoom = max(0, _safe_int(request.form.get("zoom", 7), 7))
            threshold = max(0, _safe_int(request.form.get("threshold", 0), 0))

            sync_pgvector = _apply_pgvector_env_from_request(request.form)

            dataset_upload_dir = uploads_root / dataset_name
            dataset_upload_dir.mkdir(parents=True, exist_ok=True)

            original_name = secure_filename(uploaded.filename) or f"{dataset_name}.data"
            uploaded_file_path = dataset_upload_dir / original_name
            uploaded.save(str(uploaded_file_path))

            job_id = uuid4().hex
            _set_build_job(
                job_id,
                status="queued",
                step="queued",
                message="Upload received. Waiting to start build...",
                dataset=dataset_name,
                uploaded_file=str(uploaded_file_path),
                num_tiles=num_tiles,
                zoom=zoom,
                threshold=threshold,
                sync_pgvector=sync_pgvector,
                created_at=datetime.now(timezone.utc).isoformat(),
            )

            worker = Thread(
                target=_run_dataset_build_job,
                kwargs={
                    "job_id": job_id,
                    "uploaded_file_path": uploaded_file_path,
                    "dataset_name": dataset_name,
                    "num_tiles": num_tiles,
                    "zoom": zoom,
                    "threshold": threshold,
                    "sync_pgvector": sync_pgvector,
                },
                daemon=True,
            )
            worker.start()

            return _json_response(
                {
                    "ok": True,
                    "job_id": job_id,
                    "dataset": dataset_name,
                    "message": "Upload accepted. Build started.",
                },
                202,
            )

        except Exception as e:
            logger.exception("[UploadDataset] Failed")
            return _json_response({"error": f"Upload/build request failed: {e}"}, 500)

    @app.get("/api/upload-dataset/<job_id>")
    def get_upload_dataset_job(job_id: str):
        job = _get_build_job(job_id)
        if not job:
            return _json_response({"error": f"Job not found: {job_id}"}, 404)
        return _json_response(job, 200)

    @app.post("/api/derive-dataset")
    def derive_dataset_route():
        """Cross-dataset spatial join -> derived dataset (async job).

        Body: {target_dataset, source_dataset, aggregate, value_attribute?,
               predicate?, weight?, output_attribute?}. Poll via
        GET /api/upload-dataset/<job_id> (same job store)."""
        try:
            body = request.get_json(silent=True) or {}
            spec = {k: body.get(k) for k in (
                "target_dataset", "source_dataset", "aggregate", "value_attribute",
                "predicate", "weight", "output_attribute",
            )}
            if not spec.get("target_dataset") or not spec.get("source_dataset"):
                return _json_response({"error": "target_dataset and source_dataset are required"}, 400)

            job_id = uuid4().hex
            _set_build_job(job_id, status="queued", step="queued",
                           message="Derive request received.", spec=spec,
                           created_at=datetime.now(timezone.utc).isoformat())
            Thread(target=_run_derive_job, kwargs={"job_id": job_id, "spec": spec}, daemon=True).start()
            return _json_response(
                {"ok": True, "job_id": job_id, "message": "Spatial join started."}, 202
            )
        except Exception as e:
            logger.exception("[DeriveDataset] request failed")
            return _json_response({"error": f"Derive request failed: {e}"}, 500)

    @app.post("/api/query-styles")
    def query_styles():
        body = request.get_json(silent=True) or {}
        with app.test_request_context(
            "/api/chat-style",
            method="POST",
            json=body,
        ):
            return chat_style()

    @app.post("/api/generate-map-code")
    def generate_map_code_route():
        body = request.get_json(silent=True) or {}

        user_query = _normalize_unicode_text(body.get("query", "")).strip()
        if not user_query:
            return _json_response({"error": "Request body must include a non-empty 'query'"}, 400)

        interaction_id = _normalize_unicode_text(body.get("interaction_id", "")).strip()
        current_dataset = _normalize_unicode_text(body.get("current_dataset", "")).strip()

        current_attributes_raw = body.get("current_attributes")
        if isinstance(current_attributes_raw, list):
            current_attributes = [
                _normalize_unicode_text(x)
                for x in current_attributes_raw
                if isinstance(x, (str, int, float))
            ]
        else:
            current_attributes = []

        current_style = body.get("current_style")
        if not isinstance(current_style, dict):
            current_style = {}

        requested_k = body.get("k", 5)
        try:
            k = max(1, min(int(requested_k), 10))
        except Exception:
            k = 5

        try:
            total_t0 = perf_counter()
            top_k_payload = []
            semantic_search_ms = 0.0

            if not current_dataset:
                initial_t0 = perf_counter()
                initial_response, _ = _run_initial_chat_turn(user_query=user_query, k=k)
                bootstrap_ms = (perf_counter() - initial_t0) * 1000.0

                current_dataset = _normalize_unicode_text(initial_response.get("selected_dataset", "")).strip()
                interaction_id = _normalize_unicode_text(initial_response.get("interaction_id", "")).strip()
                current_style = initial_response.get("style") or {}
                current_attributes = initial_response.get("selected_attributes") or []
                top_k_payload = initial_response.get("top_k") or []
                semantic_search_ms = float((initial_response.get("timings") or {}).get("semantic_search_ms") or 0.0)

                if not current_dataset:
                    return _json_response({"error": "Could not determine a dataset for map-code generation."}, 500)

                if not _dataset_exists(current_dataset):
                    return _json_response({"error": f"Dataset not found: {current_dataset}"}, 404)

                dataset_summary = _dataset_summary_for_llm(current_dataset)

                if not current_style:
                    current_style = _normalize_style_for_client(
                        dataset=current_dataset,
                        dataset_summary=dataset_summary,
                        style={},
                    )

                llm_t0 = perf_counter()
                code_turn = generate_map_code(
                    dataset=current_dataset,
                    dataset_summary=dataset_summary,
                    user_query=user_query,
                    current_style=current_style,
                    previous_interaction_id=None,
                    provider_name=_LLM_PROVIDER,
                    temperature=0.2,
                )
                llm_ms = (perf_counter() - llm_t0) * 1000.0
                total_ms = (perf_counter() - total_t0) * 1000.0
                tile_metric = _get_tile_metric(current_dataset)

                response = {
                    "mode": "generated_code",
                    "query": user_query,
                    "interaction_id": _normalize_unicode_text(code_turn.interaction_id or interaction_id),
                    "assistant_response": _normalize_unicode_text(code_turn.assistant_response),
                    "selected_dataset": current_dataset,
                    "selected_attributes": [_normalize_unicode_text(x) for x in (current_attributes or [])],
                    "style": current_style,
                    "generated_code": _normalize_unicode_text(code_turn.code),
                    "top_k": top_k_payload,
                    "timings": {
                        "semantic_search_ms": round(semantic_search_ms, 3),
                        "llm_ms": round(llm_ms, 3),
                        "total_ms": round(total_ms, 3),
                        "bootstrap_request_ms": round(bootstrap_ms, 3),
                        "last_tile_request_ms": round(float(tile_metric.get("elapsed_ms")), 3) if tile_metric and tile_metric.get("elapsed_ms") is not None else None,
                    },
                }
                return _json_response(response, 200)

            if not _dataset_exists(current_dataset):
                return _json_response({"error": f"Dataset not found: {current_dataset}"}, 404)

            dataset_summary = _dataset_summary_for_llm(current_dataset)

            if not current_style:
                current_style = _normalize_style_for_client(
                    dataset=current_dataset,
                    dataset_summary=dataset_summary,
                    style={},
                )

            llm_t0 = perf_counter()
            code_turn = generate_map_code(
                dataset=current_dataset,
                dataset_summary=dataset_summary,
                user_query=user_query,
                current_style=current_style,
                previous_interaction_id=None,
                provider_name=_LLM_PROVIDER,
                temperature=0.2,
            )
            llm_ms = (perf_counter() - llm_t0) * 1000.0
            total_ms = (perf_counter() - total_t0) * 1000.0
            tile_metric = _get_tile_metric(current_dataset)

            response = {
                "mode": "generated_code",
                "query": user_query,
                "interaction_id": _normalize_unicode_text(code_turn.interaction_id or interaction_id),
                "assistant_response": _normalize_unicode_text(code_turn.assistant_response),
                "selected_dataset": current_dataset,
                "selected_attributes": [_normalize_unicode_text(x) for x in (current_attributes or [])],
                "style": current_style,
                "generated_code": _normalize_unicode_text(code_turn.code),
                "top_k": top_k_payload,
                "timings": {
                    "semantic_search_ms": round(semantic_search_ms, 3),
                    "llm_ms": round(llm_ms, 3),
                    "total_ms": round(total_ms, 3),
                    "last_tile_request_ms": round(float(tile_metric.get("elapsed_ms")), 3) if tile_metric and tile_metric.get("elapsed_ms") is not None else None,
                },
            }
            return _json_response(response, 200)

        except FileNotFoundError as e:
            return _json_response({"error": str(e)}, 503)
        except LookupError as e:
            return _json_response({"error": str(e)}, 404)
        except Exception as e:
            logger.exception("[GenerateMapCode] Failed for query=%r", user_query)
            return _json_response({"error": f"Map code generation failed: {e}"}, 500)

    # ------------------------------------------------------------------ ephemeral
    # Optionally avoid permanently saving join/aggregate-derived datasets (e.g. on
    # a public demo): remove them after CATALYST_DERIVED_TTL_MIN minutes. Default
    # 0 = disabled (derived datasets persist + are reused, as before).
    _derived_ttl_min = float(os.environ.get("CATALYST_DERIVED_TTL_MIN", "0") or 0)

    def _cleanup_derived_datasets(force_all: bool = False) -> int:
        """Remove materialized derived datasets (dirs carrying a ``derived.json``
        marker) plus their ``_uploads`` parquet. ``force_all`` ignores the TTL."""
        if _derived_ttl_min <= 0 and not force_all:
            return 0
        now_ts = datetime.now(timezone.utc).timestamp()
        removed: List[str] = []
        for d in list(data_root.glob("*")):
            marker = d / "derived.json"
            if not (d.is_dir() and marker.exists()):
                continue
            try:
                age_min = (now_ts - marker.stat().st_mtime) / 60.0
            except OSError:
                continue
            if not force_all and age_min < _derived_ttl_min:
                continue
            try:
                shutil.rmtree(d, ignore_errors=True)
                up = uploads_root / f"{d.name}.parquet"
                if up.exists():
                    up.unlink()
                tiler_cache.pop(d.name, None)
                removed.append(d.name)
            except Exception:
                logger.exception("[Derive] cleanup failed for %s", d.name)
        if removed:
            logger.info("[Derive] removed %d ephemeral derived dataset(s): %s",
                        len(removed), ", ".join(removed))
            try:
                build_catalog_index(data_root=data_root, out_dir=data_root / "_catalog",
                                    sync_pgvector=False)
            except Exception:
                logger.exception("[Derive] catalogue reindex after cleanup failed")
            _catalog_runtime["router"] = None
            _catalog_runtime["mtime"] = None
        return len(removed)

    if _derived_ttl_min > 0:
        def _derived_sweeper() -> None:
            import time as _time
            interval = max(30.0, min(_derived_ttl_min * 60.0, 300.0))
            while True:
                _time.sleep(interval)
                try:
                    _cleanup_derived_datasets()
                except Exception:
                    logger.exception("[Derive] sweeper iteration failed")

        _cleanup_derived_datasets(force_all=True)   # clear leftovers from prior runs
        Thread(target=_derived_sweeper, daemon=True).start()
        logger.info("[Derive] ephemeral mode ON: derived datasets removed after %.0f min",
                    _derived_ttl_min)

    return app
