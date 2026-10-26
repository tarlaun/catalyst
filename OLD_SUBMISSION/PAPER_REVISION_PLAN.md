# Catalyst — Paper Revision Plan (agent handoff)

**Audience:** an AI agent (or author) tasked with revising the Catalyst demo paper for resubmission.
**Goal:** turn the rejected VLDB 2026 demo (paper ID 2739, *"CATALYST: CATalogue-driven LLM sYstem for Spatial Tiling"*) into an acceptance by addressing every reviewer concern. The code has already been changed to back the new claims — your job is to make the paper reflect and exploit that.

---

## 0. Before you start — inputs & constraints

- **Paper source:** the repo only contains the compiled PDF `26VLDB_Demo_Catalyst.pdf` (4 pages incl. refs). **You need the LaTeX source** (`.tex` + figures) to edit. If it is not in the repo, ask the author for it or request the Overleaf project. Do not try to edit the PDF directly.
- **Format:** VLDB/PVLDB demo paper — short (≈4 pages + references), `vldb.cls` / `acmart`-style, two-column. Keep additions tight; cut hedging prose to make room.
- **Reviews to satisfy:** `Demo_reviews.pdf` (R1 Weak Reject, R2 Reject, R3 Reject) and `metareviews.pdf` (Reject). Read them first. Reviewer point IDs (e.g. `R2-D1`) are referenced throughout this plan.
- **Prior work to differentiate from:** **LASEK** = reference **[2]** (Bahadori, Sarvepalli, Eldawy, *LASEK: LLM-Assisted Style Exploration Kit*, PVLDB 18(12), 2025) — the authors' own earlier demo. Reviewers see Catalyst as "LASEK + one semantic-search layer." Defeating this perception is the single most important edit.
- **Ground truth for claims:** everything below was implemented and verified live (Gemini `gemini-2.5-flash` + `gemini-embedding-001`). See `MEMORY.md` → `catalyst-vldb-resubmission.md` for the engineering record. Use the measured numbers in §5; do not invent figures.

---

## 1. The core problem (why it was rejected)

All three reviewers + the meta-reviewer converge on: **insufficient novelty over LASEK**. Every other criticism (no scalability experiments, LLM-code security, "poor demonstration design," no geospatial reasoning) is a symptom they cite to justify that. Two reviewers (R2, R3) explicitly hand us the fix: **make cross-dataset semantic search + information aggregation/overlay the headline contribution** (it was buried in Future Work). That one repositioning resolves novelty, demo-design, and the "single-dataset only" complaints at once.

**Strategy:** Promote cross-dataset overlay from Future Work to *the* contribution; add the evidence reviewers asked for (scalability + latency table); convert the security weakness into a demonstrated feature; and curate the live demo so it needs no uploads.

---

## 2. What is now implemented (you may claim all of this truthfully)

These are done and verified in the codebase (commit on branch `cross-dataset-overlay`):

1. **Cross-dataset overlay.** `/api/chat-style` returns a `layers[]` stack of up to 3 jointly-relevant datasets (not just top-1), ordered areas→lines→points, each independently styled, with a per-layer *reason*. Two paths: LLM (`start_multilayer_conversation`, `selection_mode="llm"`) and a deterministic heuristic fallback. Rendered as stacked MapLibre layers via a new `starlet:set-layers` message; the UI shows an "Active Map Layers" panel.
2. **Sandboxed LLM-generated code.** The map runtime iframe is `sandbox="allow-scripts"` (opaque origin); generated JS runs with only `api`/`map`/`maplibregl` in scope behind a forbidden-token blocklist (no DOM/network/storage/eval); cross-frame messages are authenticated by window identity. Verified: malicious `document`/`fetch` is blocked, benign `api.*` runs.
3. **Full latency breakdown.** Chat responses report `semantic_search_ms`, `llm_ms`, `mvt_render_ms`, `total_ms`; the build pipeline records preprocessing build/index seconds (`<dataset>/preprocessing.json`).
4. **Offline / robustness.** A dependency-free `LocalTextEmbedder` fallback lets retrieval and the catalogue run with no API key (demo-booth resilience); the router self-heals when the stored index no longer matches the active embedder.
5. **Curated, co-located demo datasets.** `riverside_parks` (polygons), `riverside_highways` (lines), `riverside_intersections` (64k points) — same Riverside County, CA footprint, three geometry types — plus a global `osm_buildings`.

---

## 3. Reviewer-point → paper-edit map

| Reviewer point(s) | Concern | Paper edit (see §4) |
|---|---|---|
| R1-D3, R2-W1/D1, R3-#2, meta | Novelty vs LASEK | **E1** reframe contribution; **E2** new cross-dataset scenario; **E8** explicit "Difference from LASEK" |
| R1-W1/D1, R2-W3/D3, R3-#1, meta | No scalability/latency experiments; preprocessing time unreported | **E3** evaluation table + latency breakdown |
| R1-W2/D2, R3-#2, meta | LLM-generated-JS security | **E4** "Safe Execution" subsection; move out of Future Work |
| R2-W2/D2, meta | Poor demo design (forced uploads) | **E5** curated pre-loaded collection; uploads secondary |
| R3-W1, meta | LLM = NL→code only; no geospatial reasoning | **E6** geospatial-reasoning examples + honest limits |
| R1-D1 | Domain terms, incomplete metadata, sampling accuracy, ambiguous queries | **E7** robustness paragraph |

---

## 4. Section-by-section edits

> Anchor edits to the current structure: **Abstract**, **§1 Introduction**, **§2 System Design** (2.1 Semantic Index, 2.2 Visualization Index, *Online execution*), **§3 Demonstration Scenarios** (3.1 Upload, 3.2 Iterative Refinement, 3.3 Full Code Generation), **§4 Future Work**, **References**.

### E1 — Reframe the contribution (Abstract + §1)
- Rewrite the abstract's final third and §1's contribution list so the headline is **catalogue-driven *cross-dataset* discovery and overlay**: from one NL prompt, Catalyst retrieves *multiple* jointly-relevant datasets and composes them into a single, coherently-styled map.
- State 3 crisp contributions: (1) a semantic catalogue that retrieves and **aggregates across datasets**, (2) a hybrid dense/on-demand tiling index enabling interactive overlay at scale, (3) an LLM agent that produces **and safely executes** cross-dataset visualizations.
- Replace the single-dataset framing ("retrieves the most relevant dataset") with multi-dataset language.

### E2 — New headline scenario: Cross-Dataset Overlay (§3, make it Scenario 1)
- Add a scenario where one prompt yields a layered overlay. Use the verified example: *"visualize parks, highways and intersections in Riverside on one map"* → 3 datasets stacked (parks polygons as base, highways lines, intersection points on top), each styled by an attribute the agent chose (`PARK_TYPE`, `HWY_TYPE`, `STNAMES`), with the agent explaining why each dataset was included.
- Emphasize what LASEK could **not** do: reasoning about *which* datasets jointly answer a query and aggregating them. Mention the top-k → subset-selection step (`selection_mode`).
- Add **Figure (new):** the rendered Riverside overlay + the "Active Map Layers" panel listing the 3 datasets and the agent's per-layer reasons. (Capture from the running app at `/`.)

### E3 — Evaluation / Performance (new short subsection or table near §2.2/§3)
- Reviewers (R1-W1, R2-D3, R3-#1) demand quantified scalability and a latency breakdown including **MVT render** and **preprocessing** time. Insert a table (numbers in §5) and 2–3 sentences:
  - **Preprocessing**: per-dataset tiling+MVT build time vs. feature count (offline, one-time).
  - **Query-time latency**: semantic retrieval / LLM inference / MVT render / total, to justify "interactive."
- Honestly scope the "tens to hundreds of GB" claim: report the largest dataset you actually built and note the index design (dense precompute + on-demand sparse tiles) is what makes larger collections tractable; if you build a bigger dataset before resubmission, add its row. **Do not leave the GB claim unbacked** — either support it or soften it.

### E4 — Safe Execution of LLM code (promote from Future Work to §2/§3.3)
- Add a short "Safe Execution" paragraph describing the deployed sandbox: generated JavaScript runs in an `allow-scripts`-only iframe (opaque origin, no same-origin/DOM/cookie/storage access), is restricted to an allow-listed `api`/`map`/`maplibregl` surface, and is statically screened for forbidden tokens before execution; cross-frame messages are authenticated by window identity.
- In §4 Future Work, **remove** the "we will explore sandboxing… in future work" hedge and replace it with the harder open problems (e.g. richer static analysis, capability budgets), so the security concern reads as *addressed*, not deferred.

### E5 — Demo design without forced uploads (§3.1)
- Reframe §3.1: the demo ships a **curated, pre-processed, spatially co-located collection** (the Riverside trio + others) so attendees explore immediately by typing queries — especially cross-dataset ones. Keep upload-and-build as a *secondary* capability shown on request, not the main interaction (directly answers R2-W2/D2 and the "audience interaction: Low" scores).
- Mention offline resilience: the local-embedder fallback means the booth works without internet for retrieval/styling (a robustness selling point).

### E6 — Geospatial reasoning (§3.2 / §3.3)
- Counter R3-W1 ("NL→code translator on attribute names") with concrete reasoning examples already supported: attribute-role-aware mode selection (categorical vs. gradient), log-scale for skewed magnitudes (the existing water-area example), and **cross-dataset spatial context** (overlaying thematic layers over the same footprint). Frame overlay itself as a form of spatial reasoning (choosing complementary layers and a sensible z-order).
- Be honest about current limits (no topological joins / spatial predicates yet) and list them as near-term work — reviewers reward candor over overclaiming.

### E7 — Robustness paragraph (§2.1)
- One paragraph addressing R1-D1: how semantic retrieval handles domain-specific terminology (embedding-based matching over enriched descriptors), graceful handling of incomplete metadata (datasets without stats are skipped/retrieval still works), the reservoir-sampling cap for bounded tile sizes and its effect on visual fidelity, and **ambiguous queries** → multiple candidates → cross-dataset overlay (turning ambiguity into a feature).

### E8 — Explicit positioning vs. LASEK (§1 or Related discussion)
- Add 2–3 sentences naming LASEK [2] and stating precisely what is new: LASEK styles a *single, already-chosen* dataset; Catalyst adds (a) catalogue-scale semantic discovery, (b) **cross-dataset selection + aggregation/overlay**, and (c) safe execution of generated visualization code. The current paper's silence on LASEK is what sank it — make the delta unmistakable.

### E9 — Figures
- **Update** the timing/figure-3b panel to show the **four** metrics (semantic search, LLM, MVT render, total) — render time was the specific gap R3-#1 named.
- **Add** the overlay figure from E2.
- Ensure Figure 1 (pipeline) mentions the multi-dataset retrieval→overlay path and the sandboxed runtime.

---

## 5. Measured numbers (use these; re-measure if datasets change)

**Query-time latency** (local server, Gemini `gemini-2.5-flash`, `gemini-embedding-001` 1536-d):
- Semantic retrieval (Gemini query embedding + cosine over catalogue): ~0.28 s
- LLM multi-layer selection+styling: ~7.7–9.7 s
- MVT render (server-side, warmed representative z7 tile): <1 ms cached / single-digit–tens of ms cold
- End-to-end chat turn: ~8–10 s
- (Offline `LocalTextEmbedder` retrieval: ~1–2 ms — note as the no-network fallback.)

**Preprocessing (one-time build = spatial tiling + MVT generation):**

| Dataset | Features | Geometry | Build time | Parquet tiles | Pre-gen MVTs |
|---|---|---|---|---|---|
| riverside_parks | 529 | polygons | ~7.7 s | 11 | 34 |
| riverside_highways | 1,344 | lines | ~6.9 s | 10 | 32 |
| riverside_intersections | 64,059 | points | ~18.3 s | 11 | 28 |
| osm_buildings | 45,742 | polygons | ~31 s | 11 | 2,597 |

**On-the-fly tile sizes (z7, Riverside footprint):** parks ≈ 82 KB, highways ≈ 123 KB, intersections ≈ 2.2 MB — demonstrates the dense/sparse tradeoff.

> If you build a genuinely large (≥ several GB) dataset before resubmission, add its row to strengthen the scalability claim — that is the most valuable single addition for R1/R2/R3.

---

## 6. Suggested order of work
1. Get the `.tex` source. 2. E1 + E8 (reframe + LASEK delta) — highest leverage. 3. E2 + E9 (overlay scenario + figures). 4. E3 (evaluation table). 5. E4 (safe execution). 6. E5/E6/E7 (demo design, reasoning, robustness). 7. Tighten to page limit; re-read all three reviews and confirm each numbered point is visibly addressed.

## 7. Do-not-overclaim checklist
- [ ] No "hundreds of GB" unless a row in §5 supports it (else soften).
- [ ] Security described as *deployed sandbox + validation*, not "future work."
- [ ] Geospatial-reasoning limits stated honestly (no topological joins yet).
- [ ] Every reviewer point (R1-D1..D3, R2-D1..D3, R3-#1/#2) maps to a visible change.
- [ ] LASEK [2] explicitly cited with a stated delta.
