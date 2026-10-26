# Demo verification screenshots — starmap.cs.ucr.edu/catalyst

Taken 2026-10-25 against the live demo, after deploying the fast tile path
(ported from latest starlet) and the pushdown sidecars. Every prompt below was
typed into the live UI; all succeeded.

| # | File | Prompt | Path exercised | Result |
|---|------|--------|----------------|--------|
| 1 | `01_vegetation_by_type.png` | Color the Riverside vegetation map by vegetation type, one distinct color per type. | Suggest Style | Categorical fill, 8-class legend, full tile coverage |
| 2 | `02_fire_hazard_zones.png` | Show the fire hazard severity zones, colored from moderate to very high risk. | Suggest Style | fhsz polygons, warm palette by `haz_class` |
| 3 | `03_highways_by_type.png` | Color Riverside highways by their highway type (interstate, ramp, US highway). | Suggest Style | Lines colored by `HWY_TYPE` |
| 4 | `04_riverside_overlay_3layers.png` | Visualize parks, highways and intersections in Riverside together on one map. | Multi-layer overlay | 3-dataset stack (polygons+lines+points), "cross-dataset overlay · 3" badge |
| 5 | `05_vegetation_opacity_codegen.png` | Color each vegetation type differently and use opacity so large areas stand out... (log scale on acreage) | Generate Map Code → Run | Sandboxed script executed, per-type color + acreage opacity |
| 6 | `06_click_isolate_popup_codegen.png` | Let me click a vegetation area to fade out all other polygons and show a popup... | Generate Map Code → Run → map click | Click isolates polygon; popup shows type + 380.4 acres |
| 7 | `07_quantile_classes_codegen.png` | Classify vegetation patches into five acreage quantile classes computed from the features on screen. | Generate Map Code → Run | Computed real breaks from viewport (8.24/15.16/28.89/69.25 acres) |
| 8 | `08_parks_log_scale_codegen.png` | Color Riverside parks by acreage on a logarithmic scale... | Generate Map Code → Run | Log-interpolated green ramp + legend |
| 9 | `09_fhsz_buildings_overlay.png` | Show fire hazard zones and buildings together on one map. | Multi-layer overlay | 2-dataset overlay: hazard classes + building footprints |
| 10 | `10_buildings_by_fhsz_join.png` | Color buildings by the fire hazard severity of the zone they fall in. | derive-live (lazy spatial join) | Buildings tinted by joined `haz_class` (Malibu canyons, z14; verified 57/60 features joined per tile) |

| 11 | `11_quakes_on_fhsz_overlay.png` | Overlay recent earthquake epicenters on the fire hazard severity zones. | Multi-layer overlay | Epicenters with magnitude gradient over hazard polygons |
| 12 | `12_fhsz_quake_count_derive.png` | Count the earthquakes in each fire hazard zone and color the zones by that count. (+ white-to-red refinement) | Materialized derive (count) → cached `fhsz_earthquake_count` + follow-up refine | Choropleth 0–155 quakes/zone; derive short-circuited to the cached derived dataset |

| 13 | `13_buildings_by_fhsz_fast_cityzoom.png` | Color buildings by the fire hazard severity of the zone they fall in. | derive-live, rewritten fast path (2026-10-26) | County-wide view: full building coverage, per-class colored dots (was grey/sparse/slow) |
| 14 | `14_buildings_by_fhsz_fast_z13_malibu.png` | (same, zoomed Chatsworth ~z13) | derive-live fast path | Very High buildings blue in Santa Susana hills, unzoned valley grey |
| 15 | `15_buildings_by_fhsz_fast_footprints.png` | (same, footprint zoom ~z15) | derive-live fast path | Individual footprints colored by zone; grey exactly at zone boundary |

2026-10-26 derive-live rewrite: the join now runs inside `fast_tile.generate_tile`
(`annotate` hook — batched STRtree point-in-polygon on feature bbox centres) instead
of decoding + re-encoding the finished MVT with the pure-Python codec. Sub-pixel
packed dots are grouped per joined class so they stay colorable at city zoom; the
2500-feature sampling cap is gone (fast fallback only). Live timings:
cold 0.1–5.2s (shares the base dataset's row-group cache), warm 35–230ms round-trip;
83/83 derive-live requests 200 after the restart.

Join verification (server-side decode of `/api/derive-live/buildings/fhsz/haz_class/...` tiles):
buildings inside mapped zones carry `haz_class`; buildings in unzoned areas
(e.g. downtown Lancaster) correctly have none and render as "Other / no value".

Prompts 1–4, 9–12 (plus the join prompt 10) are now all in the example
dropdown under "Combine datasets — overlays & spatial joins". A routing bug
found during this pass was fixed: count/sum/mean derive specs were being sent
to derive-live (which only annotates values) with `value_attribute="None"`;
they now go to the materialized tiled derive.
