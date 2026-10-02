import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMResult:
    output_text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


# Approximate USD pricing per 1K tokens.
# Update these values whenever you run a final benchmark,
# because provider pricing can change.
PRICING = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gemini-3.6-flash": {"input": 0.000075, "output": 0.0003},
    "mistral-small-latest": {"input": 0.0002, "output": 0.0006},
    "codestral-2508": {"input": 0.0003, "output": 0.0009},
}


def _require_api_key(
    environment_variable: str,
    provider_name: str,
) -> str:
    """
    Read a provider API key from the environment.

    API keys must never be hard-coded into source code.
    """
    api_key = os.getenv(environment_variable)

    if not api_key:
        raise RuntimeError(
            f"{provider_name} API key is not configured. "
            f"Set {environment_variable} in your .env file."
        )

    return api_key


def _cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """
    Estimate request cost from token usage.

    Values are approximate and intended for experiment analytics.
    """
    rates = PRICING.get(
        model,
        {
            "input": 0.0,
            "output": 0.0,
        },
    )

    return (
        (prompt_tokens / 1000) * rates["input"]
        + (completion_tokens / 1000) * rates["output"]
    )


def call_openai(
    prompt: str,
    model: str = "gpt-4o-mini",
) -> LLMResult:
    """
    Call an OpenAI chat-completion model.
    """
    from openai import OpenAI

    api_key = _require_api_key(
        "OPENAI_API_KEY",
        "OpenAI",
    )

    client = OpenAI(api_key=api_key)

    start = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    latency_ms = (time.time() - start) * 1000

    text = response.choices[0].message.content or ""

    usage = response.usage

    prompt_tokens = (
        usage.prompt_tokens
        if usage is not None
        else len(prompt.split())
    )

    completion_tokens = (
        usage.completion_tokens
        if usage is not None
        else len(text.split())
    )

    return LLMResult(
        output_text=text,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=_cost(
            model,
            prompt_tokens,
            completion_tokens,
        ),
    )


def call_gemini(
    prompt: str,
    model: str = "gemini-3.6-flash",
) -> LLMResult:
    """
    Call a Gemini model using Google's current Python SDK.

    Supports both:
        GOOGLE_API_KEY
    and:
        GEMINI_API_KEY

    GOOGLE_API_KEY is preferred because that is the variable
    already used in the project's .env file.
    """
    from google import genai

    # Support the existing GOOGLE_API_KEY variable.
    # GEMINI_API_KEY is also accepted for compatibility.
    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )

    if not api_key:
        raise RuntimeError(
            "Google Gemini API key is not configured. "
            "Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file."
        )

    client = genai.Client(
        api_key=api_key,
    )

    start = time.time()

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    latency_ms = (time.time() - start) * 1000

    text = response.text or ""

    usage = getattr(
        response,
        "usage_metadata",
        None,
    )

    prompt_tokens = (
        getattr(
            usage,
            "prompt_token_count",
            0,
        )
        if usage is not None
        else 0
    )

    completion_tokens = (
        getattr(
            usage,
            "candidates_token_count",
            0,
        )
        if usage is not None
        else 0
    )

    # Some provider responses may not expose usage.
    # Keep deterministic fallback estimates so the
    # experiment record remains complete.
    if prompt_tokens == 0:
        prompt_tokens = len(prompt.split())

    if completion_tokens == 0:
        completion_tokens = len(text.split())

    return LLMResult(
        output_text=text,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=_cost(
            model,
            prompt_tokens,
            completion_tokens,
        ),
    )


def call_mistral(
    prompt: str,
    model: str = "mistral-small-latest",
) -> LLMResult:
    """
    Call the Mistral chat-completions API.
    """
    import requests

    api_key = _require_api_key(
        "MISTRAL_API_KEY",
        "Mistral",
    )

    start = time.time()

    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        },
        timeout=60,
    )

    latency_ms = (time.time() - start) * 1000

    # Raise a useful error instead of failing later
    # while trying to read an unsuccessful response.
    response.raise_for_status()

    data = response.json()

    text = (
        data["choices"][0]["message"]["content"]
        or ""
    )

    usage = data.get("usage") or {}

    prompt_tokens = usage.get(
        "prompt_tokens",
        len(prompt.split()),
    )

    completion_tokens = usage.get(
        "completion_tokens",
        len(text.split()),
    )

    return LLMResult(
        output_text=text,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=_cost(
            model,
            prompt_tokens,
            completion_tokens,
        ),
    )


def call_mock(
    prompt: str,
    model: str = "mock-model",
) -> LLMResult:
    """
    Deterministic local provider used only for pipeline testing.

    This is NOT real LLM output and must not be used as benchmark
    evidence. It allows the evaluation pipeline to be tested without
    an external API key.
    """
    start = time.time()

    text = (
        "This is a mock response generated for pipeline testing. "
        f"Prompt received: {prompt}"
    )

    latency_ms = (time.time() - start) * 1000

    prompt_tokens = len(prompt.split())
    completion_tokens = len(text.split())

    return LLMResult(
        output_text=text,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=0.0,
    )


# One routing layer for every supported provider.
MODEL_ROUTER = {
    "gpt-4o-mini": call_openai,
    "gemini-3.6-flash": call_gemini,
    "mistral-small-latest": call_mistral,
    "codestral-2508": call_mistral,
    "mock-model": call_mock,
}


def call_model(
    model_name: str,
    prompt: str,
) -> LLMResult:
    """
    Send a prompt through the selected model provider.
    """
    if model_name not in MODEL_ROUTER:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Add it to MODEL_ROUTER."
        )

    return MODEL_ROUTER[model_name](
        prompt,
        model_name,
    )