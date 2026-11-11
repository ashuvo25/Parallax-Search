"""LLM client for an OpenAI-compatible API (Groq by default, OpenRouter also works).

Errors are turned into LLMError subclasses, each with the HTTP status the API should return.
"""
import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import (APIConnectionError, APIStatusError, APITimeoutError,
                    AuthenticationError, NotFoundError, OpenAI,
                    PermissionDeniedError, RateLimitError)

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")  # leave empty for non-reasoning models
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
TIMEOUT_S = float(os.getenv("LLM_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))


# ---------- errors ----------

class LLMError(RuntimeError):
    """Base class. `status` is the HTTP status the API returns for this error."""
    status = 502

    def __init__(self, message: str, retry_after: str | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMConfigError(LLMError):
    status = 503      # missing key, bad key, unknown model


class LLMRateLimited(LLMError):
    status = 429


class LLMUnavailable(LLMError):
    status = 503      # provider down or unreachable


class LLMTimeout(LLMError):
    status = 504


class LLMTokenLimit(LLMError):
    status = 502      # answer cut off before any text was produced


# ---------- client ----------

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
        raise LLMConfigError("LLM_API_KEY is not set. Add it to the .env file in the project root.")
    # the SDK retries rate limits, timeouts and 5xx errors itself, with backoff
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=TIMEOUT_S, max_retries=MAX_RETRIES)


def list_models(client: OpenAI | None = None) -> list[str]:
    client = client or get_client()
    return sorted(m.id for m in client.models.list())


def generate(messages: list[dict], client: OpenAI | None = None, model: str = MODEL,
             max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE) -> LLMResponse:
    """Send a chat request. Raises an LLMError subclass if it fails after retries."""
    client = client or get_client()
    kwargs = {"model": model, "messages": messages,
              "max_tokens": max_tokens, "temperature": temperature}
    if REASONING_EFFORT:
        kwargs["reasoning_effort"] = REASONING_EFFORT

    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(**kwargs)
    except RateLimitError as e:
        retry_after = e.response.headers.get("retry-after") if e.response is not None else None
        raise LLMRateLimited("The LLM provider's rate limit was reached. Try again shortly.",
                             retry_after=retry_after) from e
    except APITimeoutError as e:  # must come before APIConnectionError (it's a subclass)
        raise LLMTimeout(f"The LLM did not answer within {TIMEOUT_S:.0f} seconds.") from e
    except APIConnectionError as e:
        raise LLMUnavailable("Could not reach the LLM provider.") from e
    except NotFoundError as e:
        raise LLMConfigError(f"Model '{model}' was not found. Run `python src/llm.py "
                             f"--list-models` and update LLM_MODEL in .env.") from e
    except (AuthenticationError, PermissionDeniedError) as e:
        raise LLMConfigError("The LLM API key was rejected. Check LLM_API_KEY in .env.") from e
    except APIStatusError as e:
        if e.status_code >= 500:
            raise LLMUnavailable(f"The LLM provider had an error ({e.status_code}).") from e
        raise LLMError(f"The LLM provider rejected the request ({e.status_code}).") from e
    latency_ms = (time.perf_counter() - start) * 1000

    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    if not text and choice.finish_reason == "length":
        raise LLMTokenLimit("The answer hit the token limit before any text was written. "
                            "Increase LLM_MAX_TOKENS or lower LLM_REASONING_EFFORT.")

    usage = resp.usage
    return LLMResponse(
        text=text,
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