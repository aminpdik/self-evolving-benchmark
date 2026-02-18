"""Minimal OpenAI API–compatible HTTP client.

Loads connection settings from a local .env file and exposes helper methods for calling
OpenAI-compatible chat completion endpoints."""


from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root
# Load .env from current project folder
env_path = Path(__file__).resolve().parent / ".env"
print("Looking for .env at:", env_path)
print("Exists?", env_path.exists())
load_dotenv(env_path)



@dataclass
class OpenAICompatConfig:
    """Configuration for OpenAICompatClient.

Attributes:
    base_url: Base URL of the OpenAI-compatible server (no trailing slash required).
    api_key: API key used for Authorization Bearer header (may be empty for local servers).
    model: Model identifier string passed in the payload.
    timeout_s: HTTP request timeout in seconds."""

    base_url: str
    api_key: str
    model: str
    timeout_s: int = 60


class OpenAICompatClient:
    """
    Minimal OpenAI-API-compatible client.

    Supports:
      - POST {BASE_URL}/v1/chat/completions

    Optionally attempts:
      - POST {BASE_URL}/v1/responses (fallback)
    """

    def __init__(self, cfg: OpenAICompatConfig):
        """Create an OpenAICompatClient.

Args:
    cfg: OpenAICompatConfig containing base_url, api_key, model name, and timeout.

Returns:
    None.

How it works:
    Normalizes the base URL, stores the config, and creates a requests.Session for connection reuse."""

        self.cfg = cfg
        self.base_url = cfg.base_url.rstrip("/")
        self.session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        """Build HTTP headers for requests.

Returns:
    A dict containing Content-Type and (if provided) an Authorization Bearer token."""

        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        return headers

    def chat_completions(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 800,
        response_format_json: bool = False,
        retries: int = 3,
    ) -> str:
        """Call an OpenAI-compatible /v1/chat/completions endpoint.

Args:
    messages: Chat messages in OpenAI format, e.g. [{"role":"system","content":"..."}, ...].
    temperature: Sampling temperature.
    max_tokens: Maximum tokens to generate for the assistant.
    response_format_json: Flag indicating the caller expects JSON (kept for compatibility; not enforced here).
    retries: Number of retry attempts on errors.

Returns:
    Assistant message content as a string (data["choices"][0]["message"]["content"]).

How it works:
    Sends a POST request to {base_url}/v1/chat/completions with the configured model.
    Retries on exceptions or HTTP errors using a simple backoff sleep."""
        url = f"{self.base_url}/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Some endpoints support response_format={"type":"json_object"}
        # if response_format_json:
        #     payload["response_format"] = {"type": "json_object"}

        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                r = self.session.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.cfg.timeout_s,
                )
                if r.status_code >= 400:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                time.sleep(0.8 * (attempt + 1))

        # # Fallback to /v1/responses for endpoints that prefer it
        # try:
        #     return self.responses_api(messages, temperature=temperature, max_tokens=max_tokens, retries=retries)
        # except Exception as e:
        #     raise RuntimeError(f"chat_completions failed: {last_err}; responses fallback failed: {e}") from e

    def responses_api(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_output_tokens: int = 800,
        max_tokens: Optional[int] = None,
        retries: int = 3,
    ) -> str:
        """Fallback call to an OpenAI-compatible /v1/responses endpoint.

Args:
    messages: Chat messages (will be flattened into a single input string).
    temperature: Sampling temperature.
    max_output_tokens: Max tokens for the responses API.
    max_tokens: Optional alias overriding max_output_tokens.
    retries: Number of retry attempts.

Returns:
    Best-effort extracted text from the responses API JSON structure.

How it works:
    Flattens messages into "ROLE: content" lines, posts to /v1/responses,
    then tries to extract output[0].content[0].text."""
        url = f"{self.base_url}/v1/responses"
        input_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])

        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "input": input_text,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens if max_tokens is None else max_tokens,
        }

        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                r = self.session.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.cfg.timeout_s,
                )
                if r.status_code >= 400:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
                data = r.json()
                # Common response shapes: output[].content[].text
                if "output" in data and data["output"]:
                    out0 = data["output"][0]
                    if "content" in out0 and out0["content"]:
                        c0 = out0["content"][0]
                        if "text" in c0:
                            return c0["text"]
                return str(data)[:2000]
            except Exception as e:
                last_err = e
                time.sleep(0.8 * (attempt + 1))

        raise RuntimeError(f"responses_api failed: {last_err}")


def load_client_from_env() -> OpenAICompatClient:
    """Load OpenAICompatClient configuration from environment variables.

Environment variables:
    BASE_URL: Base API URL (default: https://api.openai.com).
    API_KEY: API key (default: empty).
    MODEL: Model identifier (default: gpt-4o-mini).
    TIMEOUT: Request timeout in seconds (default: 60).

Returns:
    An initialized OpenAICompatClient."""

    base_url = os.getenv("BASE_URL", "https://api.openai.com").strip()
    api_key = os.getenv("API_KEY", "").strip()
    model = os.getenv("MODEL", "gpt-4o-mini").strip()
    timeout_s = int(os.getenv("TIMEOUT", "60"))
    cfg = OpenAICompatConfig(base_url=base_url, api_key=api_key, model=model, timeout_s=timeout_s)
    return OpenAICompatClient(cfg)