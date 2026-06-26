from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .factory import LLMFactory
from .provider import LLMProviderError

logger = logging.getLogger(__name__)


@dataclass
class StyleConversationResult:
    assistant_response: str
    selected_dataset: str
    selected_attributes: List[str]
    style_intent: str
    style: Dict[str, Any]
    interaction_id: Optional[str] = None


@dataclass
class LayerStyleSpec:
    """One dataset layer within a cross-dataset overlay."""
    dataset: str
    reason: str
    selected_attributes: List[str]
    style_intent: str
    style: Dict[str, Any]


@dataclass
class MultiLayerStyleResult:
    """Result of selecting and styling 1..N jointly-relevant datasets.

    If the user asked to COMBINE datasets into a new derived attribute (a spatial
    join + aggregate, e.g. "color parks by their dominant vegetation type"),
    ``derive`` holds that spec instead of ``layers``.
    """
    assistant_response: str
    layers: List[LayerStyleSpec]
    interaction_id: Optional[str] = None
    derive: Optional[Dict[str, Any]] = None


@dataclass
class GeneratedMapCodeResult:
    code: str
    assistant_response: str
    interaction_id: Optional[str] = None


_FILENAME_PROMPT_TEMPLATE = """\
You are a geospatial data visualization assistant.

Dataset: {dataset}
User query: {query}

Based on the dataset name and the user's query, suggest a list of HTML
visualization page filenames that would be useful. Each filename must:
- use lowercase snake_case
- end with .html
- be descriptive of the visualization it provides

Respond with ONLY a JSON array of strings.
"""


_START_STYLE_SYSTEM = """\
You are a geospatial visualization planner.

Your job is to choose the best dataset, the best attribute(s), and an initial style.

Return ONLY a single JSON object with exactly these top-level keys:
- assistant_response: string
- selected_dataset: string
- selected_attributes: array of strings
- style_intent: string
- style: object

The style object must have these keys:
- target_attribute: string
- style_type: string
- color_theme: object with keys "name" and "colors"
- opacity: number
- stroke_width: number
- radius: number
- legend_title: string
- notes: array of strings

Guidelines:
- Choose exactly one dataset.
- If the prompt asks for a gradient or magnitude-based map, prefer a numeric attribute.
- If the prompt asks for classes/groups/types/categories, prefer a categorical attribute.
- style_type should usually be one of:
  - fill-gradient
  - fill-categorical
  - fill-single-color
  - line-gradient
  - line-categorical
  - line-single-color
  - circle-gradient
  - circle-categorical
  - circle-single-color
- Use 3 to 6 colors when helpful.
- assistant_response should be concise and useful to a user.
Output MUST be strictly valid JSON (RFC 8259): double-quoted keys and string
values, real JSON arrays and objects. Do NOT use YAML or "key: value" lines.
No markdown fences. No prose outside the JSON object.
"""


_FOLLOWUP_STYLE_SYSTEM = """\
You are continuing an existing geospatial styling conversation.

Return ONLY a single JSON object with exactly these top-level keys:
- assistant_response: string
- selected_dataset: string
- selected_attributes: array of strings
- style_intent: string
- style: object

The style object must have these keys:
- target_attribute: string
- style_type: string
- color_theme: object with keys "name" and "colors"
- opacity: number
- stroke_width: number
- radius: number
- legend_title: string
- notes: array of strings

Important:
- Keep the same dataset unless the user clearly asks to switch datasets.
- If no dataset switch is requested, selected_dataset must stay the same.
- Use the current style hint as context and modify it based on the new request.
- Output MUST be strictly valid JSON (RFC 8259): double-quoted keys and string
  values. Do NOT use YAML or "key: value" lines. No markdown. No prose outside JSON.
"""


_MULTILAYER_SYSTEM = """\
You are a geospatial visualization planner that composes CROSS-DATASET overlays.

You are given a user's request and a ranked list of candidate datasets (each with
a relevance score and a summary of its attributes). Real analytical questions are
often answered by MORE THAN ONE dataset shown together on the same map (for
example, wildfire hotspots over population density, or vegetation types beside the
parks that contain them).

Your job: decide which of the candidates are JOINTLY relevant to the request and
return one styled layer per chosen dataset.

SPECIAL CASE — spatial join / derived attribute:
If the user asks to color/size one dataset BY A PROPERTY DERIVED FROM ANOTHER
dataset (a spatial join + aggregation) — e.g. "color parks by their most
prominent vegetation type", "shade parks by the number of intersections inside
them", "color zip codes by average building height" — then DO NOT return layers.
Instead return a JSON object with:
- assistant_response: string
- derive: object with keys:
  - target_dataset: string (the features to color; a candidate name)
  - source_dataset: string (where the property comes from; a candidate name)
  - predicate: "intersects" | "within"
  - aggregate: "dominant" | "count" | "mean" | "sum"   (dominant = most prominent / area-weighted mode)
  - value_attribute: string (the SOURCE attribute to aggregate; omit/empty for count)
  - output_attribute: string (snake_case name for the new attribute, e.g. "dominant_vegetation")
Choose value_attribute from the source dataset's attributes. Use this ONLY for
genuine combine/join requests; for plain styling or simple overlays return layers.

Otherwise (normal styling / overlay), return ONLY a single JSON object with:
- assistant_response: string (1-3 sentences; explain which datasets you combined and WHY)
- layers: array of layer objects (length 1 to 3, ordered bottom layer first, top layer last)

Each layer object must have exactly these keys:
- dataset: string (must be one of the candidate dataset names, verbatim)
- reason: string (why this dataset is relevant to the request)
- selected_attributes: array of strings
- style_intent: string
- style: object with keys:
  - target_attribute: string
  - style_type: string
  - color_theme: object with keys "name" and "colors"
  - opacity: number
  - stroke_width: number
  - radius: number
  - legend_title: string
  - notes: array of strings

Guidelines:
- Include a dataset ONLY if it genuinely helps answer the request. If just one
  dataset is relevant, return a single layer. Never pad with irrelevant datasets.
- Prefer at most 3 layers. Put broad context (areas/polygons) on the bottom and
  fine detail (points/lines) on top.
- Lower the opacity of bottom polygon layers so upper layers stay visible.
- For gradient/magnitude requests prefer a numeric attribute; for
  classes/types/categories prefer a categorical attribute.
- style_type should be one of: fill-gradient, fill-categorical, fill-single-color,
  line-gradient, line-categorical, line-single-color, circle-gradient,
  circle-categorical, circle-single-color.
- Use 3 to 6 colors when helpful. Use distinct color themes per layer so overlaid
  datasets are visually separable.
Output MUST be strictly valid JSON (RFC 8259): double-quoted keys and string values,
real JSON arrays and objects. Do NOT use YAML or "key: value" lines.
No markdown. No prose outside JSON.
"""


_MAP_CODE_SYSTEM = """\
You are generating executable JavaScript for a geospatial map runtime.

You are NOT generating a full HTML page.
You are generating ONLY the JavaScript body that will run inside map.html.

The runtime provides:
- api.setDataset(datasetName)
- api.applyStyle(styleObject)
- api.ensureDatasetLayer()
- api.fitToDataset(datasetName)
- api.addLabels(options)
- api.reset()
- api.getState()

It also provides:
- map        (MapLibre map instance)
- maplibregl (MapLibre namespace)

Important style schema rules:
- For normal structured styling with api.applyStyle(...), use:
  - renderer.mode = "single" | "categorical" | "gradient"
- For gradient mode, provide:
  - renderer.attribute
  - renderer.min
  - renderer.max
  - renderer.colors
- Do NOT use renderer.stops for the normal gradient renderer unless you are building fully custom MapLibre layers yourself.
- Prefer style_type values such as:
  - fill-gradient
  - fill-categorical
  - fill-single-color
  - line-gradient
  - line-categorical
  - line-single-color
  - circle-gradient
  - circle-categorical
  - circle-single-color

Rules:
- Return raw JavaScript only.
- Do NOT return JSON.
- Do NOT return Markdown fences.
- Do NOT explain the code before or after it.
- Call api.setDataset(...) first if a dataset is known.
- Prefer using the provided structured style via api.applyStyle(...) unless the user explicitly asks for fully custom MapLibre layers.
- Use api.fitToDataset(datasetName) instead of api.fitLayer().
- Do not invent unavailable server endpoints.
- Do not generate HTML.
- Do not wrap the code in an IIFE unless needed.
"""
def _strip_code_fences(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:javascript|js|json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_code_response(text: str) -> GeneratedMapCodeResult:
    cleaned = _strip_code_fences(text)

    if not cleaned:
        raise ValueError("LLM returned an empty response.")

    # Backward-compatible path: if the model still returns JSON, try to parse it.
    try:
        parsed = _extract_json_object(cleaned)
        code = _clean_text(parsed.get("code"))
        assistant_response = _clean_text(parsed.get("assistant_response")) or "Generated map code."
        if code:
            return GeneratedMapCodeResult(
                code=code,
                assistant_response=assistant_response,
                interaction_id=None,
            )
    except Exception:
        pass

    # New preferred path: treat the whole response as raw JavaScript.
    return GeneratedMapCodeResult(
        code=cleaned,
        assistant_response="Generated map code.",
        interaction_id=None,
    )


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"```(?:json|javascript|js)?\s*", "", str(text)).strip().rstrip("`")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"No JSON object found in LLM response: {text!r}")
    payload = cleaned[start:end + 1]
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got: {type(parsed).__name__}")
    return parsed


def _extract_json_array(text: str) -> List[Any]:
    cleaned = re.sub(r"```(?:json)?\s*", "", str(text)).strip().rstrip("`")
    match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in LLM response: {text!r}")
    parsed = json.loads(match.group())
    if not isinstance(parsed, list):
        raise ValueError("Expected JSON array")
    return parsed


def _coerce_scalar(token: str) -> Any:
    """Best-effort scalar parse for the loose 'key: value' fallback."""
    s = str(token).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", "~"):
        return ""
    if re.fullmatch(r"[+-]?\d+", s):
        return int(s)
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", s) and any(c in s for c in ".eE"):
        try:
            return float(s)
        except ValueError:
            return s
    return s


def _parse_loose_mapping(text: str) -> Dict[str, Any]:
    """Tolerant parser for the YAML-ish ``key: value`` blocks some models emit
    instead of JSON (observed with gemini-2.5-flash ignoring "JSON only").

    Handles nested mappings and simple scalar lists via indentation. Tolerates a
    missing space after the colon (``key:value``). Best effort: never raises on
    malformed input — returns whatever structure it could recover.
    """
    lines: List[tuple] = []
    for raw in str(text or "").replace("\t", "  ").splitlines():
        if not raw.strip() or raw.strip().startswith(("#", "```")):
            continue
        lines.append((len(raw) - len(raw.lstrip(" ")), raw.strip()))

    n = len(lines)
    pos = 0

    def parse_block(min_indent: int):
        nonlocal pos
        container: Any = None
        while pos < n:
            indent, content = lines[pos]
            if indent < min_indent:
                break
            if content.startswith("- "):
                if container is None:
                    container = []
                if not isinstance(container, list):
                    break
                pos += 1
                container.append(_coerce_scalar(content[2:]))
                continue
            m = re.match(r"^([^:]+):(.*)$", content)
            if not m:
                pos += 1
                continue
            if container is None:
                container = {}
            if not isinstance(container, dict):
                break
            key = m.group(1).strip()
            rest = m.group(2).strip()
            pos += 1
            if rest:
                container[key] = _coerce_scalar(rest)
            elif pos < n and lines[pos][0] > indent:
                container[key] = parse_block(indent + 1)
            else:
                container[key] = ""
        return container if container is not None else {}

    result = parse_block(0)
    return result if isinstance(result, dict) else {}


def _extract_structured_object(text: str) -> Dict[str, Any]:
    """Parse the model's structured reply. Prefers strict JSON; falls back to the
    tolerant ``key: value`` parser when the model ignores the JSON-only rule."""
    try:
        return _extract_json_object(text)
    except Exception:
        pass
    parsed = _parse_loose_mapping(text)
    if isinstance(parsed, dict) and parsed:
        logger.warning("LLM returned non-JSON structured output; recovered via loose parser.")
        return parsed
    raise ValueError(f"No JSON or structured object found in LLM response: {text!r}")


def _coerce_style_object(value: Any) -> Dict[str, Any]:
    style = value if isinstance(value, dict) else {}
    color_theme = style.get("color_theme")
    if not isinstance(color_theme, dict):
        color_theme = {"name": "custom", "colors": ["#4f83ff"]}

    colors = color_theme.get("colors")
    if not isinstance(colors, list):
        colors = ["#4f83ff"]

    return {
        "target_attribute": _clean_text(style.get("target_attribute")),
        "style_type": _clean_text(style.get("style_type")) or "fill-single-color",
        "color_theme": {
            "name": _clean_text(color_theme.get("name")) or "custom",
            "colors": [str(c) for c in colors if c is not None] or ["#4f83ff"],
        },
        "opacity": float(style.get("opacity", 0.85)),
        "stroke_width": float(style.get("stroke_width", 1.5)),
        "radius": float(style.get("radius", 4.0)),
        "legend_title": _clean_text(style.get("legend_title")),
        "notes": [str(x) for x in (style.get("notes") or [])] if isinstance(style.get("notes"), list) else [],
    }


def generate_dataset_html_suggestions(
    dataset: str,
    user_query: str,
    provider_name: str = "gemini",
) -> List[str]:
    provider = LLMFactory.get_provider(provider_name)
    prompt = _FILENAME_PROMPT_TEMPLATE.format(dataset=dataset, query=user_query)
    raw = provider.generate_response(prompt)
    parsed = _extract_json_array(raw.text)
    return [name for name in parsed if isinstance(name, str) and name.endswith(".html")]


def start_style_conversation(
    *,
    dataset: str,
    dataset_summary: Dict[str, Any],
    user_query: str,
    selected_attributes: Optional[List[str]] = None,
    style_intent: Optional[str] = None,
    provider_name: str = "gemini",
    temperature: float = 0.2,
) -> StyleConversationResult:
    provider = LLMFactory.get_provider(provider_name)

    prompt = (
        f"Dataset context:\n{json.dumps(dataset_summary, ensure_ascii=False, indent=2)}\n\n"
        f"Current dataset hint: {dataset}\n"
        f"Selected attributes hint: {json.dumps(selected_attributes or [], ensure_ascii=False)}\n"
        f"Style intent hint: {_clean_text(style_intent)}\n\n"
        f"User request:\n{user_query}\n"
    )

    raw = provider.generate_response(
        prompt,
        system_instruction=_START_STYLE_SYSTEM,
        temperature=temperature,
    )
    logger.debug("start_style_conversation raw text: %s", raw.text)

    parsed = _extract_structured_object(raw.text)
    return StyleConversationResult(
        assistant_response=_clean_text(parsed.get("assistant_response")),
        selected_dataset=_clean_text(parsed.get("selected_dataset")) or dataset,
        selected_attributes=[str(x) for x in (parsed.get("selected_attributes") or []) if x is not None],
        style_intent=_clean_text(parsed.get("style_intent")),
        style=_coerce_style_object(parsed.get("style")),
        interaction_id=raw.interaction_id,
    )


def start_multilayer_conversation(
    *,
    candidates_summary: List[Dict[str, Any]],
    user_query: str,
    max_layers: int = 3,
    provider_name: str = "gemini",
    temperature: float = 0.2,
) -> MultiLayerStyleResult:
    """Ask the LLM to compose a cross-dataset overlay from ranked candidates.

    ``candidates_summary`` is a list of {dataset, score, summary} dicts. The model
    selects the jointly-relevant subset and returns one styled layer per dataset.
    """
    provider = LLMFactory.get_provider(provider_name)

    prompt = (
        f"Candidate datasets (ranked, best first):\n"
        f"{json.dumps(candidates_summary, ensure_ascii=False, indent=2)}\n\n"
        f"Maximum layers: {max_layers}\n\n"
        f"User request:\n{user_query}\n\n"
        "Select the jointly-relevant datasets and return the overlay JSON."
    )

    raw = provider.generate_response(
        prompt,
        system_instruction=_MULTILAYER_SYSTEM,
        temperature=temperature,
    )
    logger.debug("start_multilayer_conversation raw text: %s", raw.text)

    parsed = _extract_structured_object(raw.text)
    valid_names = {str(c.get("dataset")) for c in candidates_summary}

    # Spatial-join / derived-attribute path: only honour it if both datasets are
    # real candidates (don't trust hallucinated names).
    derive_raw = parsed.get("derive")
    if isinstance(derive_raw, dict):
        target = _clean_text(derive_raw.get("target_dataset"))
        source = _clean_text(derive_raw.get("source_dataset"))
        if target in valid_names and source in valid_names and target != source:
            return MultiLayerStyleResult(
                assistant_response=_clean_text(parsed.get("assistant_response")),
                layers=[],
                interaction_id=raw.interaction_id,
                derive={
                    "target_dataset": target,
                    "source_dataset": source,
                    "predicate": _clean_text(derive_raw.get("predicate")) or "intersects",
                    "aggregate": _clean_text(derive_raw.get("aggregate")) or "dominant",
                    "value_attribute": _clean_text(derive_raw.get("value_attribute")) or None,
                    "output_attribute": _clean_text(derive_raw.get("output_attribute")) or None,
                },
            )
        logger.warning("Multilayer: dropping derive spec with non-candidate datasets %r/%r", target, source)

    layers: List[LayerStyleSpec] = []
    for item in (parsed.get("layers") or []):
        if not isinstance(item, dict):
            continue
        dataset = _clean_text(item.get("dataset"))
        if dataset not in valid_names:
            # The model must pick from the provided candidates; drop hallucinations.
            logger.warning("Multilayer: dropping non-candidate dataset %r", dataset)
            continue
        layers.append(
            LayerStyleSpec(
                dataset=dataset,
                reason=_clean_text(item.get("reason")),
                selected_attributes=[str(x) for x in (item.get("selected_attributes") or []) if x is not None],
                style_intent=_clean_text(item.get("style_intent")),
                style=_coerce_style_object(item.get("style")),
            )
        )
        if len(layers) >= max_layers:
            break

    return MultiLayerStyleResult(
        assistant_response=_clean_text(parsed.get("assistant_response")),
        layers=layers,
        interaction_id=raw.interaction_id,
    )


def continue_style_conversation(
    *,
    dataset: str,
    user_query: str,
    previous_interaction_id: str,
    selected_attributes_hint: Optional[List[str]] = None,
    current_style_hint: Optional[Dict[str, Any]] = None,
    provider_name: str = "gemini",
    temperature: float = 0.2,
) -> StyleConversationResult:
    provider = LLMFactory.get_provider(provider_name)

    prompt = (
        f"Current dataset: {dataset}\n"
        f"Selected attributes hint: {json.dumps(selected_attributes_hint or [], ensure_ascii=False)}\n"
        f"Current style hint:\n{json.dumps(current_style_hint or {}, ensure_ascii=False, indent=2)}\n\n"
        f"User follow-up request:\n{user_query}\n"
    )

    raw = provider.generate_response(
        prompt,
        previous_interaction_id=previous_interaction_id,
        system_instruction=_FOLLOWUP_STYLE_SYSTEM,
        temperature=temperature,
    )
    logger.debug("continue_style_conversation raw text: %s", raw.text)

    parsed = _extract_structured_object(raw.text)
    selected_dataset = _clean_text(parsed.get("selected_dataset")) or dataset

    return StyleConversationResult(
        assistant_response=_clean_text(parsed.get("assistant_response")),
        selected_dataset=selected_dataset,
        selected_attributes=[str(x) for x in (parsed.get("selected_attributes") or []) if x is not None],
        style_intent=_clean_text(parsed.get("style_intent")),
        style=_coerce_style_object(parsed.get("style")),
        interaction_id=raw.interaction_id or previous_interaction_id,
    )


def generate_map_code(
    *,
    dataset: str,
    dataset_summary: Dict[str, Any],
    user_query: str,
    current_style: Optional[Dict[str, Any]] = None,
    previous_interaction_id: Optional[str] = None,
    provider_name: str = "gemini",
    temperature: float = 0.2,
) -> GeneratedMapCodeResult:
    provider = LLMFactory.get_provider(provider_name)

    prompt = (
        f"Dataset:\n{dataset}\n\n"
        f"Dataset summary:\n{json.dumps(dataset_summary, ensure_ascii=False, indent=2)}\n\n"
        f"Current structured style:\n{json.dumps(current_style or {}, ensure_ascii=False, indent=2)}\n\n"
        f"User request:\n{user_query}\n\n"
        "Generate the JavaScript body for map.html."
    )

    raw = provider.generate_response(
        prompt,
        previous_interaction_id=previous_interaction_id,
        system_instruction=_MAP_CODE_SYSTEM,
        temperature=temperature,
    )
    logger.debug("generate_map_code raw text: %s", raw.text)

    result = _extract_code_response(raw.text)
    result.interaction_id = raw.interaction_id or previous_interaction_id

    if not _clean_text(result.code):
        raise ValueError("LLM did not return any code.")

    return result