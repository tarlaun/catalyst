"""Bring-your-own-key: request-scoped credentials and the REST providers (no network)."""
import pytest

from catalyst._internal.server.llm import LLMFactory, credentials as C
from catalyst._internal.server.llm import anthropic_provider, gemini_provider, openai_provider
from catalyst._internal.server.llm._http import HTTPStatusError
from catalyst._internal.server.llm.provider import LLMProviderError

KEY = "sk-test-abcdefghijklmnop1234"


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    for var in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield
    C.end_request(None)


def test_parse_credentials_validates():
    assert C.parse_credentials("openai", "") is None
    creds = C.parse_credentials(" OpenAI ", f" {KEY} ", "gpt-x")
    assert (creds.provider, creds.api_key, creds.model) == ("openai", KEY, "gpt-x")
    assert KEY not in repr(creds)
    for bad in (("nope", KEY, ""), ("openai", "short", ""), ("openai", "has space in the key 1234", ""),
                ("openai", KEY + "\r\nX-Evil: 1", ""), ("openai", KEY, "model name")):
        with pytest.raises(C.CredentialsError):
            C.parse_credentials(*bad)


def test_factory_prefers_request_credentials_and_resets():
    with pytest.raises(LLMProviderError):
        LLMFactory.get_provider("gemini")                       # no server key, no visitor key
    token = C.begin_request(C.parse_credentials("anthropic", KEY, "claude-x"))
    provider = LLMFactory.get_provider("gemini")                # name is overridden by the visitor's key
    assert type(provider).__name__ == "AnthropicProvider" and provider._model == "claude-x"
    assert C.request_summary()["source"] == "user"
    C.end_request(token)
    assert C.current_credentials() is None
    with pytest.raises(LLMProviderError):
        LLMFactory.get_provider("gemini")


def test_redact_hides_keys():
    token = C.begin_request(C.parse_credentials("openai", KEY))
    C.note_error(f"Incorrect API key provided: {KEY}")
    assert KEY not in C.request_summary()["error"]
    C.end_request(token)
    assert "AIzaSyA1234567890abcdefghij" not in C.redact("bad AIzaSyA1234567890abcdefghij key")


def test_openai_provider_retries_without_temperature(monkeypatch):
    calls = []

    def fake(service, url, *, headers, payload=None, timeout=90):
        calls.append(dict(payload))
        assert headers["Authorization"] == f"Bearer {KEY}"
        if "temperature" in payload:
            raise HTTPStatusError("OpenAI", 400, '{"error": {"message": "Unsupported value: temperature"}}')
        return {"choices": [{"message": {"content": ' {"ok": true} '}}]}

    monkeypatch.setattr(openai_provider, "request_json", fake)
    out = openai_provider.OpenAIProvider(model="m", api_key=KEY).generate_response(
        "hi", system_instruction="sys", temperature=0.2)
    assert out.text == '{"ok": true}' and out.interaction_id.startswith(C.LOCAL_INTERACTION_PREFIX)
    assert len(calls) == 2 and "temperature" not in calls[1]
    assert calls[0]["messages"][0] == {"role": "system", "content": "sys"}


def test_anthropic_provider_request_shape_and_refusal(monkeypatch):
    seen = {}

    def fake(service, url, *, headers, payload=None, timeout=90):
        seen.update(headers=headers, payload=payload)
        return {"stop_reason": "end_turn", "content": [
            {"type": "thinking", "thinking": ""}, {"type": "text", "text": "hello"}]}

    monkeypatch.setattr(anthropic_provider, "request_json", fake)
    out = anthropic_provider.AnthropicProvider(model="claude-opus-5-5", api_key=KEY).generate_response(
        "hi", system_instruction="sys", temperature=0.2)
    assert out.text == "hello"
    assert seen["headers"]["x-api-key"] == KEY and seen["headers"]["anthropic-version"]
    assert "temperature" not in seen["payload"] and seen["payload"]["system"] == "sys"
    assert seen["payload"]["max_tokens"] > 0

    monkeypatch.setattr(anthropic_provider, "request_json",
                        lambda *a, **k: {"stop_reason": "refusal", "stop_details": {"category": "cyber"}, "content": []})
    with pytest.raises(LLMProviderError):
        anthropic_provider.AnthropicProvider(api_key=KEY).generate_response("hi")


def test_gemini_ignores_local_interaction_ids(monkeypatch):
    import io, json
    sent = {}

    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None, context=None):
        sent.update(json.loads(req.data.decode()))
        return Resp(json.dumps({"id": "abc", "steps": [{"content": [{"text": "ok"}]}]}).encode())

    monkeypatch.setattr(gemini_provider.urllib.request, "urlopen", fake_urlopen)
    provider = gemini_provider.GeminiProvider(api_key=KEY)
    provider.generate_response("hi", previous_interaction_id="local-123")
    assert "previous_interaction_id" not in sent
    provider.generate_response("hi", previous_interaction_id="real-id")
    assert sent["previous_interaction_id"] == "real-id"


def test_require_user_key_blocks_the_server_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    assert type(LLMFactory.get_provider("gemini")).__name__ == "GeminiProvider"
    monkeypatch.setenv("CATALYST_REQUIRE_USER_KEY", "1")
    with pytest.raises(LLMProviderError):
        LLMFactory.get_provider("gemini")
    token = C.begin_request(C.parse_credentials("openai", KEY))
    assert type(LLMFactory.get_provider("gemini")).__name__ == "OpenAIProvider"
    C.end_request(token)


def test_cookie_round_trip_and_garbage():
    creds = C.parse_credentials("gemini", KEY, "gemini-x")
    back = C.decode_cookie(C.encode_cookie(creds))
    assert (back.provider, back.api_key, back.model) == ("gemini", KEY, "gemini-x")
    assert C.key_hint(creds) == "…" + KEY[-4:]
    for junk in ("", "not-base64!!", "e30", "A" * 5000):
        assert C.decode_cookie(junk) is None


def test_key_endpoints_use_an_httponly_cookie(monkeypatch, tmp_path):
    from catalyst._internal.server import app as app_module
    monkeypatch.setattr(app_module, "list_models_for", lambda provider, key: ["m-1", "m-2"])
    client = app_module.create_app(str(tmp_path)).test_client()

    assert client.get("/api/llm/key").get_json() == {"saved": False}
    resp = client.post("/api/llm/key", json={"provider": "openai", "key": KEY, "base": "/catalyst"})
    body = resp.get_json()
    assert body["ok"] and body["saved"] and body["model"] == "m-1" and body["key_hint"] == "…" + KEY[-4:]
    assert KEY not in resp.get_data(as_text=True)
    cookie = resp.headers["Set-Cookie"]
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie and "Path=/catalyst/api" in cookie

    # The test client replays the cookie; the server reports the key as saved, never the key.
    client.set_cookie(C.COOKIE_NAME, cookie.split(";")[0].split("=", 1)[1], path="/")
    status = client.get("/api/llm/key")
    assert status.get_json()["saved"] is True and KEY not in status.get_data(as_text=True)
    assert client.post("/api/llm/key", json={"model": "m-2"}).get_json()["model"] == "m-2"
    cleared = client.delete("/api/llm/key", json={"base": "/catalyst"})
    assert "Max-Age=0" in cleared.headers["Set-Cookie"]

    bad = client.post("/api/llm/key", json={"provider": "openai", "key": "x y"})
    assert bad.status_code == 400 and "Set-Cookie" not in bad.headers
