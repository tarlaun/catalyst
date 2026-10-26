# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`catalyst` is a Python package for spatial tiling, Mapbox Vector Tile (MVT) generation, and on-demand tile serving of large geospatial datasets (GeoParquet / GeoJSON). The base pipeline is: **partition a dataset into spatial tiles → build histograms → (optionally) pre-generate MVTs → serve them over HTTP**.

This repo is the **"Catalyst" demo** (VLDB 2026 demo paper; the paper PDFs, reviews, and `PAPER_REVISION_PLAN.md` in the repo root track the resubmission). On top of the base tiling/serving package it adds an **LLM-driven catalog + styling layer**: a chat UI where a user types a natural-language request, the server semantically routes it to the best dataset(s), an LLM proposes structured map styles (up to 3 overlay layers), can **combine two datasets via a cross-dataset spatial join** ("derive"), and can generate JavaScript that runs in a **sandboxed map runtime**. The demo-specific work lives in `_internal/server/` (`app.py`, `llm/`, `catalog/`, `derive/`, `tiler/`) and the two templates `server/templates/index.html` (app shell) and `server/templates/map.html` (iframe map runtime).

> History: this repo was split out from the `starlet` project's `catalog` branch; the importable package and CLI were renamed `starlet` → `catalyst`. The internal postMessage protocol between `index.html` and the map iframe still uses the `starlet:` namespace — that's an opaque wire string, not the package name; keep both ends in sync if you ever change it.

## Setup & Common Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .            # editable install; exposes the `catalyst` console script
```

The CLI (`catalyst._cli:main`, Click-based) is the primary entry point. Each subcommand maps directly to a public function in `catalyst/__init__.py`:

```bash
catalyst tile  --input data.parquet --outdir datasets/mydata --num-tiles 40   # partition only
catalyst mvt   --dir datasets/mydata --zoom 7 --threshold 100000              # MVT only
catalyst build --input data.parquet --outdir datasets/mydata                  # tile + mvt
catalyst build ... --pmtiles                                                  # also export a PMTiles archive
catalyst serve --dir datasets --port 8765                                     # Flask server (demo UI at /)
catalyst info  --dir datasets/mydata                                          # inspect a dataset
```

The README's CLI flag tables are accurate and worth consulting before adding/changing flags. The rest of the README is stale — it's still titled "Starlet", references `make` targets that don't exist (there is **no Makefile**), and its "API Endpoints" / "LLM Styling Suggestions" sections describe an older `POST /datasets/<dataset>/styles.json` endpoint superseded by the `/api/chat-style` flow documented below. Trust this file over the README for the server/LLM surface.

### Running the demo

`catalyst serve --dir datasets` serves the demo UI at `/`. Semantic routing works out of the box (the offline embedder is the no-key fallback); the chat/styling LLM features require `GEMINI_API_KEY` (or a running Ollama). Sample datasets (`OSM2015_33`, `osm_buildings`, several `riverside_*`) and the catalog index (`datasets/_catalog/`) are checked in. After adding datasets under `datasets/`, rebuild the catalog index (`build_catalog_index`) or semantic routing won't find them.

### Tests

Pytest tests exist in `tests/` (pytest is not a declared dependency — `pip install pytest`):

```bash
pytest tests/                          # all tests
pytest tests/test_spatial_aggregate.py # pure unit tests, no data needed
```

`tests/test_derive_dataset.py` is an integration test that auto-skips unless the `riverside_parks` / `riverside_vegetation_types` datasets are built. There is no CI test job. Server/UI changes are still best verified by running `catalyst serve` and exercising endpoints/UI directly (Playwright MCP tools work well). `experiments/scalability_bench.py` is the benchmark harness for the paper's numbers — not a test.

## Architecture

The public API surface is deliberately small — everything user-facing lives in `catalyst/__init__.py` (`tile`, `generate_mvt`, `build`, `create_app`) and `catalyst/_types.py` (`TileResult`, `MVTResult`, `Dataset`). All implementation is under `catalyst/_internal/` and is treated as private. Imports inside the public functions are **lazy** (done inside the function body) to keep CLI startup fast.

### Dataset on-disk layout

A "dataset" is a directory. Subsystems communicate through this directory structure rather than in-memory objects, so understanding it is key:

```
datasets/<name>/
  parquet_tiles/      # spatially-partitioned GeoParquet files (one per tile)
  histograms/         # 2D prefix-sum density histograms (.npy), e.g. global.npy, global_prefix.npy
  mvt/<z>/<x>/<y>.mvt # pre-generated vector tiles (optional)
  stats/attributes.json  # per-attribute statistics incl. geometry MBR (used for bbox)
  derived.json        # only on derived datasets: the derive spec (idempotency marker; excludes it from catalog indexing)
  preprocessing.json  # only on uploaded datasets: build/index timings
datasets/_catalog/    # semantic search index (embeddings + catalog_index.json incl. embedder_id)
datasets/_uploads/    # uploaded source files + intermediate parquet for full derives
```

The `Dataset` class in `_types.py` is a read-only introspection wrapper that derives all metadata by inspecting these paths.

### Tiling pipeline (`_internal/tiling/`)

Driven by `catalyst.tile()`. Flow: a `DataSource` (`GeoParquetSource` / `GeoJSONSource`) streams Arrow tables → an **assigner** maps each row to a spatial partition (`RSGroveAssigner`, the default, builds partitions via the RSGrove algorithm by reservoir-sampling centroids; `TileAssignerFromCSV` uses a legacy precomputed index) → `RoundOrchestrator` buffers rows per-partition in a `WriterPool` and writes GeoParquet files. Rows that can't fit in the current round's open file handles **spill to an overflow Parquet file and are re-processed in subsequent rounds** (the "round-based" design). Within each tile, rows are optionally sorted by a space-filling curve (Z-order / Hilbert) per the `--sort` flag. Attribute statistics are collected in a single pass and written to `stats/`. Finally `histogram/hist_pyramid.py` builds the density histograms.

### MVT generation (`_internal/mvt/`)

Driven by `catalyst.generate_mvt()` → `BucketMVTGenerator`. Four streaming stages: `HistogramLoader` (loads prefix-sum histogram) → `TileAssigner` (computes nonempty z/x/y tiles from the histogram + threshold, reservoir-samples to cap features/tile) → `GeometryStreamer` (decodes WKB, reprojects EPSG:4326 → EPSG:3857) → `TileRenderer` (clip, simplify, encode `.mvt` protobuf). The renderer is process-parallel; `CATALYST_MVT_WORKERS` caps worker count to bound peak memory. `_internal/pmtiles/exporter.py` can bundle the `mvt/` tree into a single PMTiles archive (`--pmtiles` on `build`/`mvt`).

### Tile server (`_internal/server/app.py`)

`create_app()` builds the Flask app — the largest, most active file in the repo, holding both tile-serving and all demo/LLM routes. Almost everything (40+ helpers, all route handlers, caches, tunable constants) is a **closure inside `create_app`**; only `_LLM_PROVIDER`, `_BASE_PATH`, `_TILED_DERIVE_MAX_SOURCE`, and a couple of text helpers are module-level. Consequence: most env vars are read at app-construction or per-request time, not import time.

Tile-serving core: `tiler/tiler.py`'s `VectorTiler` — a **three-tier tile lookup**: (1) in-memory LRU `TileCache` → (2) pre-generated `.mvt` on disk → (3) on-the-fly generation from intersecting GeoParquet tiles (`ParquetIndex` spatial lookup + `MVTEncoder`), then written to disk and promoted into the cache. One `VectorTiler` cached per dataset. `download_service.py` serves feature downloads (csv/geojson, with optional geometry filter / MBR pushdown).

Route groups:
- **Tiles**: `GET /<dataset>/<z>/<x>/<y>.mvt`; `GET /api/derive-live/<target>/<source>/<value_attr>/<z>/<x>/<y>.mvt` (lazy cross-dataset join at serve time, nothing materialized — `tiler/virtual_derive.py`).
- **Catalog / metadata**: `GET /api/datasets`, `GET /api/catalog/entries?q=` (alias `GET /datasets.json?q=`), `GET /datasets/<dataset>.json`, `GET /api/datasets/<dataset>/stats|inspect`.
- **Features**: `GET|POST /datasets/<dataset>/features.<format>`, sample endpoints `features/sample.json|.geojson?mbr=`.
- **LLM demo**: `POST /api/chat-style` (main loop; `POST /api/query-styles` is a thin alias), `POST /api/generate-map-code`, `POST /api/derive-dataset` (async, 202 + job id), `POST /api/upload-dataset` — all async jobs (uploads, explicit derives, chat-initiated derives) are polled via the shared `GET /api/upload-dataset/<job_id>`.
- **UI**: `GET /` (`index.html`), `GET /map.html` (runtime iframe). `GET /datasets/<dataset>.html` renders `view_dataset.html`, **which does not exist in the repo** → `TemplateNotFound` (see Gotchas).

### Chat orchestration (`chat_style` + closures)

1. **Initial turn** (no `interaction_id`/`current_dataset`): semantic-search the catalog (top-k), then `start_multilayer_conversation(..., max_layers=3)` asks the LLM to pick dataset(s), attributes, and structured styles for up to 3 overlay layers. Any LLM failure falls back to `selection_mode="heuristic"` (`_heuristic_overlay_layers`: score-thresholded candidates, sorted polygon→line→point). If the LLM returns a `derive` spec, the derive branch below runs instead.
2. **Follow-up turn**: `_looks_like_new_dataset_request(...)` decides refine vs pivot. Pivot logic: explicit phrases → pivot; current dataset name or target attribute in the query → refine; otherwise a **semantic pivot check** — pivot iff the top search hit ≥ `CATALYST_PIVOT_CHALLENGER_MIN` (0.62) and beats the current dataset's score by ≥ `CATALYST_PIVOT_MARGIN` (0.03). Refinements go through `continue_style_conversation(...)` (threading `previous_interaction_id`) and return a single layer with `merge_layers: True` so the client preserves other overlay layers. An LLM error mid-follow-up, or the LLM itself returning a different dataset, restarts as a fresh initial turn.
3. `POST /api/generate-map-code` emits the JS body for the map runtime; it bootstraps via an initial turn when no dataset is selected and applies the same pivot rule.

### Cross-dataset derive (`_internal/server/derive/`)

`spatial_aggregate.py` joins a target dataset with a source: ops `dominant | count | mean | sum | max | min` (all but `count` need `value_attribute`). `dominant` uses representative-point point-in-polygon (fast); exact area-weighted overlay is opt-in via the `weight="__exact_overlay__"` sentinel. Three execution modes, chosen by source size (`CATALYST_TILED_DERIVE_MAX_SOURCE`, default 2M rows):

1. **derive-live** — lazy viewport join, chosen first in the chat path when eligible: nothing is built; the client points its tile URL at `/api/derive-live/...` and a cached `VirtualDeriveTiler` annotates target features per tile (feature cap `CATALYST_DERIVE_MAX_FEATURES`).
2. **tiled derive** (`derive_dataset_tiled`) — materializes a real dataset by reusing the *target's existing partitions* (no re-tile, no MVT rebuild; histograms/stats copied from the target and patched). ~40× faster than a full build.
3. **full derive** (`derive_dataset` + `catalyst.build`) — fallback for large sources or tiled failure.

Materialized derives become full dataset dirs named `<target>__<output_attribute>` with a `derived.json` spec marker: identical specs short-circuit as cached, catalog indexing skips them, and `CATALYST_DERIVED_TTL_MIN` (> 0) enables a sweeper thread that deletes expired ones. After any derive/upload build: tiler cache popped, catalog router invalidated, index rebuilt.

### Map runtime (`server/templates/map.html`) & UI (`index.html`)

`index.html` is a single-prompt sidebar app (not a chat log) around the map iframe; `map.html` is the runtime that executes LLM-generated JS against **MapLibre GL JS** (loaded from unpkg; raster OSM basemap). Each dataset renders as three layers (fill/line/circle, filtered by `$type`) over one vector source whose source-layer is always `layer0`; structured styles (`renderer.mode`: `single | categorical | gradient`) compile to MapLibre paint expressions.

- **postMessage protocol** (`starlet:` namespace, both directions): parent sends `ping | reset | set-dataset | apply-style | set-layers | run-script`; runtime sends `runtime-ready | pong | runtime-log | runtime-error`. A version handshake compares `RUNTIME_API_LEVEL` (map.html) against `REQUIRED_RUNTIME_API_LEVEL` (index.html) — **bump both together** when changing the runtime API; on mismatch the parent cache-busts and reloads the iframe once.
- **Sandboxing**: the iframe has `sandbox="allow-scripts allow-same-origin"` but is loaded from a **sibling host** (localhost ↔ 127.0.0.1 swap) so it gets a real-but-different origin — MapLibre's tile worker needs a real origin, while the different host still denies access to the parent's DOM/cookies/storage. Generated scripts pass a regex denylist (`document`, `fetch`, `eval`, `Function`, timers, …), then run via `new Function("api","map","maplibregl", ...)` in strict mode — those three bindings are the *only* scope.
- **`api.*` primitives** exposed to generated code: `setDataset`, `applyStyle`, `setLayers`, `addLayer`, `fitToDataset`/`fitToLayers`, `layerIds`, `features` (viewport-scoped `querySourceFeatures`), `paint`, `pulse` (rAF-animated overlay), `hoverPopup`, `reset`, `log`, `getState`. `map` is a **Proxy** that records script-registered listeners so `clearInteractivity()` can tear everything down before each re-run (otherwise stacked click-toggles cancel out). A `runGen` generation counter invalidates deferred callbacks from superseded runs.
- Generated code lands in an editable code box and only executes on "Run in Map". The upload UI section and endpoint are gated by `CATALYST_ALLOW_UPLOADS`.

### LLM styling & providers (`_internal/server/llm/`)

Conversation orchestration lives in `llm/suggestions.py`; HTTP glue in `app.py`. The LLM returns **strict JSON** (or fenced code); `suggestions.py` is defensive — `_extract_json_object` / `_extract_json_array` / `_strip_code_fences` / `_coerce_style_object` recover from chatty or fence-wrapped responses. Never `json.loads` raw provider output. Results are frozen-ish dataclasses; the structured `style` is normalized for the client by `app.py`'s `_normalize_style_for_client`.

**Provider-agnostic**: everything goes through `LLMFactory` (`factory.py`) — never import a concrete provider directly. Providers (`gemini_provider.py`, `ollama_provider.py`) implement the `LLMProvider` ABC using raw REST (`urllib`), no SDKs. `LLM_PROVIDER` selects: `gemini` (default), `ollama`, or `fallback` — a chain (`LLM_FALLBACK_ORDER`, default `gemini,ollama`) that catches any provider exception per-call and tries the next; note a failover drops server-side conversational state (only the first provider gets `previous_interaction_id`). Gemini uses the stateful **Interactions API** (`previous_interaction_id` threads state server-side; no local history), default model `gemini-flash-latest` via `GEMINI_MODEL`. `prompt.md` is package data. To add a provider: implement `LLMProvider`, register a lazy builder in `factory.py`'s `_PROVIDERS`, re-export from `__init__.py`. See `_internal/server/llm/README.md`.

### Catalog / semantic search (`_internal/server/catalog/`)

`CatalogRouter` (`router.py`) does semantic dataset search over embeddings of dataset descriptors, with two `SearchBackend`s: `.npy` cosine similarity (default) or `pgvector` (`CATALOG_SEARCH_BACKEND`: `auto|npy|pgvector`). `index.py`'s `build_catalog_index` builds the index (skipping derived datasets); `embedder.py`'s `get_embedder` picks the embedder: `CATALOG_EMBEDDER=auto` (default) uses `GeminiTextEmbedder` when `GEMINI_API_KEY` works, else the dependency-free **`LocalTextEmbedder`** (deterministic signed feature hashing, `local-hash:512`) so the demo runs fully offline. The index records its `embedder_id`/`embedding_dim`; on startup `app.py` compares them against the active embedder and **auto-rebuilds the index on mismatch** (e.g. a Gemini-built index served with the local fallback). The router is also rebuilt whenever the index file's mtime changes.

### Environment variables

| Var | Default | Effect |
|---|---|---|
| `GEMINI_API_KEY` | — | Required for Gemini LLM + embeddings; without it, embedding falls back to local-hash and LLM calls fail (or fall back if `LLM_PROVIDER=fallback`) |
| `LLM_PROVIDER` / `LLM_FALLBACK_ORDER` | `gemini` / `gemini,ollama` | Provider selection / fallback chain |
| `GEMINI_MODEL` / `GEMINI_EMBEDDING_MODEL` / `OLLAMA_MODEL` | `gemini-flash-latest` / `gemini-embedding-001` / `llama3` | Model choices |
| `CATALOG_EMBEDDER` / `CATALOG_SEARCH_BACKEND` | `auto` / `auto` | Embedder + search backend selection |
| `CATALYST_ALLOW_UPLOADS` | `1` | Gates the upload endpoint (403) and UI section (unauthenticated build jobs) |
| `CATALYST_TILED_DERIVE_MAX_SOURCE` | 2000000 | Source-row ceiling for tiled derive & derive-live eligibility |
| `CATALYST_DERIVED_TTL_MIN` | `0` (off) | Ephemeral cleanup of derived datasets |
| `CATALYST_DERIVE_MAX_FEATURES` | 2500 | Per-tile feature cap in derive-live tiles |
| `CATALYST_PIVOT_CHALLENGER_MIN` / `CATALYST_PIVOT_MARGIN` | 0.62 / 0.03 | Follow-up dataset-pivot thresholds (calibrated for `gemini-embedding-001`) |
| `CATALYST_MVT_WORKERS` | cpu−1 | MVT renderer process-pool cap (memory bound) |
| `PGVECTOR_*`, `CATALOG_PGVECTOR_ENABLED` | — | pgvector backend config (upload requests can mutate these in-process) |

## Conventions

- **Public vs private**: anything a caller needs is re-exported from `catalyst/__init__.py` or `catalyst/_types.py`. Treat `_internal/` as private and keep new public surface minimal.
- Result objects are frozen dataclasses (`TileResult`, `MVTResult`).
- Coordinate systems: source data is EPSG:4326 (lon/lat); MVT/tile math and spatial joins are EPSG:3857 (Web Mercator). Reprojection happens in the streaming/rendering/derive stages.
- The internal tile-partition column is `geo_parquet_tile_num`; derive-tiled output writes per-row `_bbox_*` pushdown columns and bbox-encoded tile filenames that must match `parse_parquet_bbox`.
- LLM responses are untrusted text — always parse through the defensive helpers in `suggestions.py`.
- When changing the map runtime API, bump `RUNTIME_API_LEVEL` (map.html) and `REQUIRED_RUNTIME_API_LEVEL` (index.html) together, and keep the `starlet:` message names in sync between the two templates.

## Gotchas

- **`view_dataset.html` is missing**: `GET /datasets/<dataset>.html` calls `render_template("view_dataset.html")` but no such template exists (only `index.html` and `map.html` ship; `view_mvt.html` sits in `server/` outside the template folder). The route 500s with `TemplateNotFound` until the template is added and registered in `pyproject.toml`.
- **`CATALYST_BASE_PATH` is effectively inert**: it's passed to templates as `base_path` but neither template uses it — both derive the URL prefix client-side from `window.location.pathname` (`CATALYST_BASE`). Reverse-proxy sub-path hosting works via that client-side derivation; the proxy must strip the prefix.
- **Stale sandbox comments**: prose comments in both templates claim the iframe runs with an opaque/null origin without `allow-same-origin`; the actual mechanism is `allow-same-origin` + the sibling-host origin trick described above. Trust the code.

## Packaging

`pyproject.toml` (setuptools). Package data includes the server templates (`server/templates/*.html`) and LLM prompts (`server/llm/prompt.md`) — when adding templates/prompts, register them under `[tool.setuptools.package-data]`. Publishing is automated via `.github/workflows/publish.yml`.
