"""Request-scoped "bring your own key" credentials.

A visitor can supply their own LLM API key. The web UI hands it to the server
once; the server validates it and returns it as an HttpOnly cookie, so it lives
only in the visitor's browser and no page script (including LLM-generated map
code) can read it back. API clients may send it in headers instead. The HTTP layer validates it and installs it here for
the duration of that one request; :class:`LLMFactory` then builds the provider
from it instead of the server's environment key.

The key lives in a ``ContextVar`` — never in a module global, the environment,
a cache or a log line — so it cannot leak into another visitor's request.
"""
from __future__ import annotations

import base64
import json
import re
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Providers a visitor may bring a key for (Ollama is local-only, no key).
USER_KEY_PROVIDERS: Dict[str, Dict[str, str]] = {
    "gemini": {
        "label": "Google Gemini",
        "default_model": "gemini-flash-latest",
        "key_url": "https://aistudio.google.com/apikey",
        "key_hint": "AIza…",
    },
    "openai": {
        "label": "OpenAI (ChatGPT)",
        "default_model": "gpt-5.1",
        "key_url": "https://platform.openai.com/api-keys",
        "key_hint": "sk-…",
    },
    "anthropic": {
        "label": "Anthropic (Claude)",
        "default_model": "claude-opus-5-5",
        "key_url": "https://platform.claude.com/settings/keys",
        "key_hint": "sk-ant-…",
    },
}

# Keys are opaque printable tokens; anything else (spaces, newlines, control
# characters) is rejected before it can reach an outgoing HTTP header.
_KEY_RE = re.compile(r"^[A-Za-z0-9._\-~+/=:]{12,512}$")
_MODEL_RE = re.compile(r"^[A-Za-z0-9._:\-/@]{1,120}$")

# Interaction ids minted for providers without server-side conversation state.
LOCAL_INTERACTION_PREFIX = "local-"


def new_local_interaction_id() -> str:
    return LOCAL_INTERACTION_PREFIX + uuid.uuid4().hex[:20]


@dataclass
class RequestCredentials:
    provider: str
    api_key: str = field(repr=False)          # never printed with the dataclass
    model: Optional[str] = None


@dataclass
class _RequestLLMState:
    credentials: Optional[RequestCredentials] = None
    errors: List[str] = field(default_factory=list)
    used_provider: Optional[str] = None
    used_model: Optional[str] = None


_STATE: ContextVar[Optional[_RequestLLMState]] = ContextVar("catalyst_llm_request_state", default=None)


class CredentialsError(ValueError):
    """The supplied provider / key / model headers are malformed."""


def parse_credentials(provider: str, api_key: str, model: str = "") -> Optional[RequestCredentials]:
    """Validate raw header values. Returns None when no key was supplied."""
    provider = (provider or "").strip().lower()
    api_key = (api_key or "").strip()
    model = (model or "").strip()
    if not api_key:
        return None
    if provider not in USER_KEY_PROVIDERS:
        raise CredentialsError(
            "Unsupported provider. Choose one of: " + ", ".join(sorted(USER_KEY_PROVIDERS))
        )
    if not _KEY_RE.match(api_key):
        raise CredentialsError("The API key does not look valid (unexpected characters or length).")
    if model and not _MODEL_RE.match(model):
        raise CredentialsError("The model name contains unexpected characters.")
    return RequestCredentials(provider=provider, api_key=api_key, model=model or None)


COOKIE_NAME = "catalyst_llm"
COOKIE_MAX_AGE = 30 * 24 * 3600


def encode_cookie(creds: RequestCredentials) -> str:
    raw = json.dumps({"p": creds.provider, "k": creds.api_key, "m": creds.model or ""},
                     separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cookie(value: str) -> Optional[RequestCredentials]:
    """Credentials from the cookie, or None when it is absent / malformed."""
    if not value or len(value) > 2048:
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode("utf-8"))
        return parse_credentials(str(data.get("p", "")), str(data.get("k", "")), str(data.get("m", "")))
    except Exception:
        return None


def key_hint(creds: RequestCredentials) -> str:
    """Last four characters, for 'key ending in …' display only."""
    return "…" + creds.api_key[-4:]


def begin_request(credentials: Optional[RequestCredentials]) -> Token:
    return _STATE.set(_RequestLLMState(credentials=credentials))


def end_request(token: Optional[Token]) -> None:
    if token is not None:
        try:
            _STATE.reset(token)
            return
        except Exception:  # token from another context: fall through to a hard clear
            pass
    _STATE.set(None)


def current_credentials() -> Optional[RequestCredentials]:
    state = _STATE.get()
    return state.credentials if state else None


def redact(text: Any) -> str:
    """Strip the current request's key (and anything key-shaped) from a message."""
    out = str(text)
    creds = current_credentials()
    if creds and creds.api_key:
        out = out.replace(creds.api_key, "[key hidden]")
    out = re.sub(r"\b(sk-[A-Za-z0-9_\-]{6})[A-Za-z0-9_\-*.]{6,}", r"\1…", out)
    out = re.sub(r"\b(AIza[A-Za-z0-9_\-]{4})[A-Za-z0-9_\-]{10,}", r"\1…", out)
    return out


def note_used(provider: str, model: Optional[str]) -> None:
    state = _STATE.get()
    if state is not None:
        state.used_provider, state.used_model = provider, model


def note_error(exc: Any) -> None:
    state = _STATE.get()
    if state is not None:
        msg = redact(exc).strip()
        if len(msg) > 400:
            msg = msg[:400].rstrip() + "…"
        state.errors.append(msg)


def clear_errors() -> None:
    """Forget earlier errors of this request (a later attempt is about to supersede them)."""
    state = _STATE.get()
    if state is not None:
        state.errors.clear()


def request_summary() -> Dict[str, Any]:
    """What the client is told about the LLM side of this request (no secrets)."""
    state = _STATE.get()
    creds = state.credentials if state else None
    return {
        "source": "user" if creds else "server",
        "provider": (state.used_provider if state else None) or (creds.provider if creds else None),
        "model": (state.used_model if state else None) or (creds.model if creds else None),
        "error": (state.errors[-1] if state and state.errors else None),
    }
