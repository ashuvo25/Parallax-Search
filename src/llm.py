"""LLM client for an OpenAI-compatible API (Groq by default, OpenRouter also works)."""
import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import AuthenticationError, NotFoundError, OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")  # leave empty for non-reasoning models
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None


def get_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError("LLM_API_KEY is not set. Add it to the .env file in the project root.")
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def list_models(client: OpenAI | None = None) -> list[str]:
    """Model IDs that are live right now for this API key."""
    client = client or get_client()
    return sorted(m.id for m in client.models.list())


def generate(messages: list[dict], client: OpenAI | None = None, model: str = MODEL,
             max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE) -> LLMResponse:
    """Send a chat request and return the answer with timing and token usage."""
    client = client or get_client()
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if REASONING_EFFORT:
        kwargs["reasoning_effort"] = REASONING_EFFORT

    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(**kwargs)
    except NotFoundError as e:
        raise RuntimeError(
            f"Model '{model}' was not found at {BASE_URL}. Free models change often. "
            f"Run `python src/llm.py --list-models` and update LLM_MODEL in .env."
        ) from e
    except AuthenticationError as e:
        raise RuntimeError("The API key was rejected. Check LLM_API_KEY in .env.") from e
    latency_ms = (time.perf_counter() - start) * 1000

    choice = resp.choices[0]
    usage = resp.usage
    return LLMResponse(
        text=(choice.message.content or "").strip(),
        model=resp.model,
        latency_ms=round(latency_ms, 1),
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
        finish_reason=choice.finish_reason,
    )


def main():
    parser = argparse.ArgumentParser(description="LLM client checks.")
    parser.add_argument("--list-models", action="store_true", help="show live model IDs")
    parser.add_argument("--ping", action="store_true", help="send a short test request")
    args = parser.parse_args()

    if args.list_models:
        models = list_models()
        print(f"{len(models)} models available at {BASE_URL}:")
        for m in models:
            marker = "  <- current LLM_MODEL" if m == MODEL else ""
            print(f"  {m}{marker}")
        if MODEL not in models:
            print(f"\nWarning: LLM_MODEL '{MODEL}' is not in this list. Update .env.")

    if args.ping:
        resp = generate([{"role": "user", "content": "Reply with the single word: ready"}])
        print(f"Model: {resp.model}")
        print(f"Reply: {resp.text}")
        print(f"Latency: {resp.latency_ms} ms | tokens in/out: "
              f"{resp.prompt_tokens}/{resp.completion_tokens}")

    if not (args.list_models or args.ping):
        parser.print_help()


if __name__ == "__main__":
    main()