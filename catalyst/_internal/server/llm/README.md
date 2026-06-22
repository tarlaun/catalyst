# LLM Integration

This package provides a provider-agnostic interface for calling large language
models. It is used by the styles endpoint to generate and iteratively refine
map styling suggestions from dataset attribute statistics.

## Architecture

```text
provider.py          — ABC + normalized response model
gemini_provider.py   — Google Gemini Interactions API (stateful multi-turn)
ollama_provider.py   — Local Ollama instance (stateless fallback)
factory.py           — LLMFactory: provider registry + env-var selection
suggestions.py       — High-level style conversation helpers

All providers implement the `LLMProvider` abstract class and raise
`LLMProviderError` on failure. The rest of the server never imports a concrete
provider directly — it always goes through `LLMFactory`.

## Choosing a Provider

Set the `LLM_PROVIDER` environment variable:

```bash
export LLM_PROVIDER=gemini   # default
export LLM_PROVIDER=ollama
```

If the variable is unset or contains an unrecognised value, the factory falls
back to `gemini` and logs a warning.

## Provider Configuration

### Gemini

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | — | API key from [Google AI Studio](https://aistudio.google.com/apikey) |

The default model is `gemini-2.0-flash`. No additional dependencies are
needed — the provider uses `urllib` from the standard library.

```bash
export GEMINI_API_KEY=your-key-here
starlet serve --dir datasets
```

### Ollama

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OLLAMA_MODEL` | No | `llama3` | Model to use for generation |

Ollama must be running locally on the default port (`11434`). Start it with:

```bash
ollama serve
```

Then pull a model if you haven't already:

```bash
ollama pull llama3
```

Run the server:

```bash
export LLM_PROVIDER=ollama
starlet serve --dir datasets
```

To use a different model:

```bash
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=mistral
ollama pull mistral
starlet serve --dir datasets
```

## Adding a New Provider

1. Create `<name>_provider.py` implementing `LLMProvider`.
2. Add a lazy builder in `factory.py`'s `_PROVIDERS` dict.
3. Re-export the class from `__init__.py`.

No changes to the endpoint or factory selection logic are needed — the new
provider is automatically available via `LLM_PROVIDER=<name>`.
