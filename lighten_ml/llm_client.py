"""Lighten LLM client wrapper for textual understanding and reasoning.

Supports Together.ai Chat Completions API by default, with configurable
base URL and model. Falls back gracefully if API key is not provided.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import json
import time
import hashlib
import logging
import requests

DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.together.xyz/v1")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
DEFAULT_TIMEOUT = 60
DEFAULT_RATE_LIMIT_TPS = 5
DEFAULT_CACHE_SIZE = 1024

logger = logging.getLogger(__name__)

class TokenBucket:
    def __init__(self, tokens_per_second: float, max_tokens: float):
        self.tokens_per_second = tokens_per_second
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.last_refill_time = time.time()

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill_time
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.tokens_per_second)
        self.last_refill_time = now

    def consume(self, tokens: int = 1):
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

class LightenLLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        rate_limit_tps: int = DEFAULT_RATE_LIMIT_TPS,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY") or os.environ.get("LLM_API_KEY")
        self.model = model or DEFAULT_MODEL
        self.base_url = base_url or DEFAULT_BASE_URL
        self.timeout = timeout
        self._enabled = bool(self.api_key)

        # Initialize rate limiter and cache
        self.rate_limiter = TokenBucket(rate_limit_tps, rate_limit_tps)
        self.cache = {}
        self.cache_size = cache_size

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _get_cache_key(self, instructions: str, text: str) -> str:
        """Generate a cache key from instructions and text."""
        return hashlib.sha256((instructions + text).encode()).hexdigest()

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
        response_format: Optional[Dict[str, Any]] = None,
        retries: int = 2,
        retry_delay: float = 1.5,
    ) -> str:
        """Call chat completions and return assistant content string.
        If the call fails or LLM is disabled, raises RuntimeError.
        """
        if not self.enabled:
            raise RuntimeError("LLM client not enabled: missing API key")

        # Check cache first
        cache_key = self._get_cache_key(json.dumps(messages), "")
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Wait for rate limiter
        while not self.rate_limiter.consume():
            time.sleep(0.1)

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    result = data["choices"][0]["message"]["content"]

                    # Update cache
                    if len(self.cache) >= self.cache_size:
                        self.cache.pop(next(iter(self.cache))) # Remove oldest item
                    self.cache[cache_key] = result

                    return result
                else:
                    last_exc = RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:500]}")
            except Exception as e:  # noqa: BLE001
                last_exc = e
            time.sleep(retry_delay)
        assert last_exc is not None
        raise last_exc

    def extract_json(self, instructions: str, text: str, schema_hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ask the LLM to extract structured JSON from text.
        Returns an empty dict if parsing fails.
        """
        sys_msg = {
            "role": "system",
            "content": (
                "You are a clinical NLP assistant. Read clinical notes and extract structured evidence per instructions. "
                "Return STRICT JSON only, with no prose."
            ),
        }
        user_msg = {
            "role": "user",
            "content": (
                f"Instructions:\n{instructions}\n\nClinical Text:\n'''\n{text}\n'''\n\n"
                "Return ONLY valid JSON."
            ),
        }
        response_format = {"type": "json_object"}
        try:
            content = self.chat([sys_msg, user_msg], response_format=response_format)
            return json.loads(content)
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM extract_json failed: %s", e)
            try:
                # Attempt to salvage JSON if model returned extra text
                start = content.find('{') if 'content' in locals() else -1
                end = content.rfind('}') if 'content' in locals() else -1
                if start != -1 and end != -1:
                    return json.loads(content[start : end + 1])
            except Exception:
                pass
            return {}
