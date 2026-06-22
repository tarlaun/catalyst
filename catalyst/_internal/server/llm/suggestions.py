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
No markdown. No prose outside JSON.
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
- No markdown. No prose outside JSON.
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

    parsed = _extract_json_object(raw.text)
    return StyleConversationResult(
        assistant_response=_clean_text(parsed.get("assistant_response")),
        selected_dataset=_clean_text(parsed.get("selected_dataset")) or dataset,
        selected_attributes=[str(x) for x in (parsed.get("selected_attributes") or []) if x is not None],
        style_intent=_clean_text(parsed.get("style_intent")),
        style=_coerce_style_object(parsed.get("style")),
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

    parsed = _extract_json_object(raw.text)
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