# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`catalyst` is a Python package for spatial tiling, Mapbox Vector Tile (MVT) generation, and on-demand tile serving of large geospatial datasets (GeoParquet / GeoJSON). The base pipeline is: **partition a dataset into spatial tiles → build histograms → (optionally) pre-generate MVTs → serve them over HTTP**.

This repo is the **"Catalyst" demo** (VLDB 2026 demo paper). On top of the base tiling/serving package it adds an **LLM-driven catalog + styling layer**: a chat UI where a user types a natural-language request, the server semantically routes it to the best dataset, an LLM proposes a structured map style (and optionally generates the `map.html` JS), and an upload flow builds new datasets on demand. The demo-specific work lives in `_internal/server/` (`app.py`, `llm/suggestions.py`, `catalog/`) and the `templates/index.html` demo UI.

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
catalyst serve --dir datasets --port 8765                                     # Flask server (demo UI at /)
catalyst info  --dir datasets/mydata                                          # inspect a dataset
```

The README's CLI flag tables are accurate and worth consulting before adding/changing flags. The rest of the README is partly stale, though — it's still titled "Starlet", references `make` targets that don't exist (there is **no Makefile**), and its "API Endpoints" / "LLM Styling Suggestions" sections describe an older `POST /datasets/<dataset>/styles.json` endpoint that has been superseded by the `/api/chat-style` chat flow documented below. Trust this file over the README for the server/LLM surface.

### Running the demo

`catalyst serve --dir datasets` serves the demo UI (`templates/index.html`) at `/`. The LLM features require `GEMINI_API_KEY` (see LLM section). Datasets live under `datasets/<name>/`; the catalog search index must be built before semantic routing works (see Catalog section).

### Tests

There is **no test framework** (no pytest/unittest, no CI test job). Verify changes by running `catalyst serve` and exercising the endpoints/UI directly, ideally with the Playwright MCP tools.

A sample dataset (`datasets/OSM2015_33/`) and the semantic-search catalog index (`datasets/_catalog/`) are checked in so the server runs out-of-the-box; the chat/styling LLM features still require `GEMINI_API_KEY`. Larger datasets are intentionally not committed — add your own under `datasets/` and rebuild the catalog index.

## Architecture

The public API surface is deliberately small — everything user-facing lives in `catalyst/__init__.py` (the `tile`, `generate_mvt`, `build`, `create_app` functions) and `catalyst/_types.py` (`TileResult`, `MVTResult`, `Dataset`). All implementation is under `catalyst/_internal/` and is treated as private. Imports inside the public functions are **lazy** (done inside the function body) to keep CLI startup fast.

### Dataset on-disk layout

A "dataset" is a directory. Subsystems communicate through this directory structure rather than in-memory objects, so understanding it is key:

```
datasets/<name>/
  parquet_tiles/      # spatially-partitioned GeoParquet files (one per tile)
  histograms/         # 2D prefix-sum density histograms (.npy), e.g. global.npy, global_prefix.npy
  mvt/<z>/<x>/<y>.mvt # pre-generated vector tiles (optional)
  stats/attributes.json  # per-attribute statistics incl. geometry MBR (used for bbox)
datasets/_catalog/    # semantic search index over datasets (embeddings)
datasets/_uploads/    # uploaded source files (from the demo upload flow)
```

The `Dataset` class in `_types.py` is a read-only introspection wrapper that derives all metadata (tile count, bbox, zoom levels, presence of histograms/mvt/stats) by inspecting these paths.

### Tiling pipeline (`_internal/tiling/`)

Driven by `catalyst.tile()`. Flow: a `DataSource` (`GeoParquetSource` / `GeoJSONSource`) streams Arrow tables → an **assigner** maps each row to a spatial partition (`RSGroveAssigner`, the default, builds partitions via the RSGrove algorithm by reservoir-sampling centroids; `TileAssignerFromCSV` uses a legacy precomputed index) → `RoundOrchestrator` buffers rows per-partition in a `WriterPool` and writes GeoParquet files. Rows that can't fit in the current round's open file handles **spill to an overflow Parquet file and are re-processed in subsequent rounds** (this is the "round-based" design). Within each tile, rows are optionally sorted by a space-filling curve (Z-order / Hilbert) per the `--sort` flag. Attribute statistics are collected in a single pass and written to `stats/`. Finally `histogram/hist_pyramid.py` builds the density histograms.

### MVT generation (`_internal/mvt/`)

Driven by `catalyst.generate_mvt()` → `BucketMVTGenerator`. Four streaming stages: `HistogramLoader` (loads prefix-sum histogram) → `TileAssigner` (computes which z/x/y tiles are nonempty using the histogram + threshold, assigns geometries with reservoir sampling to cap features/tile) → `GeometryStreamer` (decodes WKB, reprojects EPSG:4326 → EPSG:3857) → `TileRenderer` (clip, simplify, transform to tile coords, encode `.mvt` protobuf). The renderer is parallelized — see recent commits about render speedups and avoiding redundant `make_valid`.

### Tile server (`_internal/server/`)

`create_app()` builds a Flask app (`app.py`). It is the largest, most active file on this branch and holds **both** the tile-serving routes and all the demo/LLM routes. Note that most helpers and the `_run_*_chat_turn` orchestration are **closures defined inside `create_app`** (so they capture the dataset dir and per-app caches) rather than module-level functions.

Tile-serving core: `tiler/tiler.py`'s `VectorTiler`, a **three-tier tile lookup**: (1) in-memory LRU cache (`TileCache`) → (2) pre-generated `.mvt` on disk → (3) generate on-the-fly from intersecting GeoParquet tiles (via `ParquetIndex` spatial lookup + `MVTEncoder`). Generated tiles are written to disk and promoted into the cache. One `VectorTiler` is cached per dataset. `download_service.py` serves feature downloads (csv/geojson, with optional geometry filter / MBR pushdown).

Key route groups in `app.py`:
- **Tiles & data**: `GET /<dataset>/<z>/<x>/<y>.mvt`, `GET /datasets/<dataset>.json` (metadata), `GET /api/datasets/<dataset>/stats|inspect`, feature downloads `GET|POST /datasets/<dataset>/features.<format>`, sample features (`.json`/`.geojson`).
- **Catalog / search**: `GET /api/catalog/entries`, `GET /datasets.json?q=...` (semantic search).
- **LLM demo**: `POST /api/chat-style` (main chat loop), `POST /api/query-styles`, `POST /api/generate-map-code`, `POST /api/upload-dataset` (+ `GET /api/upload-dataset/<job_id>` for async build-job polling).
- **UI**: `GET /` (demo `index.html`), `GET /map.html`, `GET /datasets/<dataset>.html` (per-dataset visualization).

> **Gotcha — the `map.html` template is not in the repo.** `index.html` embeds the map as `<iframe src="/map.html">`, and the `/map.html` route does `render_template("map.html")`, but `templates/` ships only `index.html` (and `view_mvt.html` lives in `server/`, outside the Flask `template_folder` and not in `package-data`). So `/map.html` currently raises `TemplateNotFound` and the iframe stays blank until you add a `map.html` template under `server/templates/` and register it in `pyproject.toml`. `POST /api/generate-map-code` only returns the JS *body* meant to run inside that page — it does not create the page.

### LLM styling & chat flow (`_internal/server/llm/`)

This is the heart of the Catalyst demo. The conversation orchestration lives in `llm/suggestions.py`; the HTTP glue (request parsing, turn routing, response shaping) lives in `app.py`'s `chat_style()` route and its `_run_initial_chat_turn` / `_run_followup_chat_turn` closures.

The `/api/chat-style` turn logic:
1. **No `interaction_id`/`current_dataset` yet → initial turn.** Semantic-search the catalog for the top-k datasets, then `start_style_conversation(...)` asks the LLM to pick the dataset, the attribute(s), a `style_intent`, and an initial structured `style` object.
2. **Follow-up with an existing interaction.** `_looks_like_new_dataset_request(...)` decides whether the user is pivoting to a different dataset (→ treat as a fresh initial turn) or refining the current one (→ `continue_style_conversation(...)`, threading `previous_interaction_id` so the provider keeps conversational state).
3. `generate_map_code(...)` (route `/api/generate-map-code`) asks the LLM to emit the JavaScript body for `map.html`, given the dataset summary + current structured style.

The LLM returns **strict JSON** (or fenced code); `suggestions.py` is defensive about this — `_extract_json_object` / `_extract_json_array` / `_strip_code_fences` / `_coerce_style_object` recover from chatty or fence-wrapped responses. Results are frozen-ish dataclasses (`StyleConversationResult`, `GeneratedMapCodeResult`). The structured `style` object is normalized for the client by `app.py`'s `_normalize_style_for_client` (geometry-kind inference, palette/gradient helpers, hex normalization).

**Provider-agnostic**: everything goes through `LLMFactory` (`factory.py`) — never import a concrete provider directly. Providers (`gemini_provider.py`, `ollama_provider.py`) implement the `LLMProvider` ABC and use raw REST (`urllib`), no SDKs. Selection is via the `LLM_PROVIDER` env var (`gemini` default, falls back to gemini on unknown values). `prompt.md` is shipped as package data. To add a provider: implement `LLMProvider`, register a lazy builder in `factory.py`'s `_PROVIDERS`, re-export from `__init__.py`. See `_internal/server/llm/README.md`.

Required env vars: `GEMINI_API_KEY` (Gemini), optional `OLLAMA_MODEL` (Ollama, default `llama3`).

### Catalog / semantic search (`_internal/server/catalog/`)

`CatalogRouter` (`router.py`) provides semantic dataset search over embeddings of dataset descriptors, with two backends (`SearchBackend`): `.npy` cosine-similarity (default) or `pgvector` (`pgvector_store.py`, configured via request/env). `index.py` (`build_catalog_index`, `CATALOG_FILENAME`) builds the embedding index from dataset descriptors; `embedder.py`'s `GeminiTextEmbedder` produces the embeddings. The catalog index must exist (`datasets/_catalog/`) before chat routing works — `app.py` raises a clear "Catalogue index not found" error otherwise. The checked-in `datasets/_catalog/` index covers only the sample dataset(s); after you add datasets under `datasets/`, rebuild it (`build_catalog_index`) or semantic routing won't find them.

## Conventions

- **Public vs private**: anything a caller needs is re-exported from `catalyst/__init__.py` or `catalyst/_types.py`. Treat `_internal/` as private and keep new public surface minimal.
- Result objects are frozen dataclasses (`TileResult`, `MVTResult`).
- Coordinate systems: source data is EPSG:4326 (lon/lat); MVT/tile math is EPSG:3857 (Web Mercator). Reprojection happens in the streaming/rendering stages.
- The internal tile-partition column is `geo_parquet_tile_num`.
- LLM responses are untrusted text — always parse through the defensive helpers in `suggestions.py`, never `json.loads` raw provider output.

## Packaging

`pyproject.toml` (setuptools). Package data includes the server HTML templates (`server/templates/*.html`) and LLM prompts (`server/llm/prompt.md`) — when adding new templates/prompts, register them under `[tool.setuptools.package-data]`. Publishing is automated via `.github/workflows/publish.yml`.
