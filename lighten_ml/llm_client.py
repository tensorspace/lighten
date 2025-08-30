"""Lighten LLM client wrapper for textual understanding and reasoning.

Supports Together.ai Chat Completions API by default, with configurable
base URL and model. Falls back gracefully if API key is not provided.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import requests

DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.together.xyz/v1")
DEFAULT_MODEL = os.environ.get(
    "LLM_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
)
DEFAULT_TIMEOUT = 60
DEFAULT_RATE_LIMIT_TPS = 5
DEFAULT_CACHE_SIZE = 1024
DEFAULT_CACHE_PATH = os.environ.get("LLM_CACHE_PATH", "llm_cache.json")

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
        self.tokens = min(
            self.max_tokens, self.tokens + elapsed * self.tokens_per_second
        )
        self.last_refill_time = now

    def consume(self, tokens: int = 1):
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def wait_for_token(self):
        while not self.consume():
            time.sleep(0.1)


class LightenLLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        rate_limit_tps: int = DEFAULT_RATE_LIMIT_TPS,
        cache_size: int = DEFAULT_CACHE_SIZE,
        cache_path: Optional[str] = None,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("TOGETHER_API_KEY")
            or os.environ.get("LLM_API_KEY")
        )
        self.model = model or DEFAULT_MODEL
        self.base_url = base_url or DEFAULT_BASE_URL
        self.timeout = timeout
        self._enabled = bool(self.api_key)

        # Initialize rate limiter and cache
        self.rate_limiter = TokenBucket(rate_limit_tps, rate_limit_tps)
        self.cache: Dict[str, Any] = {}
        self.cache_size = cache_size
        self.cache_path = cache_path or DEFAULT_CACHE_PATH
        self.retries = 5  # Increased from 2 to 5
        self._load_cache()

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

    def _load_cache(self):
        """Load the cache from a JSON file if it exists."""
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    self.cache = json.load(f)
                logger.info(
                    f"LLM cache loaded successfully from {self.cache_path}. Contains {len(self.cache)} items."
                )
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(
                    f"Could not load LLM cache from {self.cache_path}: {e}. Starting with an empty cache."
                )
                self.cache = {}

    def _save_cache(self):
        """Save the current cache to a JSON file."""
        if self.cache_path:
            try:
                with open(self.cache_path, "w") as f:
                    json.dump(self.cache, f, indent=2)
                logger.debug(f"LLM cache saved to {self.cache_path}.")
            except IOError as e:
                logger.error(f"Could not save LLM cache to {self.cache_path}: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
        response_format: Optional[Dict[str, Any]] = None,
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
            logger.info(f"LLM response found in cache for model {self.model}.")
            return self.cache[cache_key]

        # Wait for rate limiter
        logger.info(f"Waiting for rate limiter for model {self.model}...")
        self.rate_limiter.wait_for_token()

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        logger.info(f"Sending request to LLM API (model: {self.model})...")

        last_exc: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                resp = requests.post(
                    url, headers=self._headers(), json=payload, timeout=self.timeout
                )
                resp.raise_for_status()  # Raise an exception for bad status codes

                logger.info("LLM API request successful.")
                data = resp.json()
                result = data["choices"][0]["message"]["content"]

                # Update cache
                if len(self.cache) >= self.cache_size:
                    self.cache.pop(next(iter(self.cache)))  # Remove oldest item
                self.cache[cache_key] = result
                self._save_cache()  # Save cache after each new entry

                return result
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"LLM API request failed (attempt {attempt + 1}/{self.retries}): {e}"
                )
                if attempt < self.retries - 1:
                    delay = retry_delay * (2**attempt) + random.uniform(
                        0, 1
                    )  # Exponential backoff with jitter
                    logger.warning(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"LLM API request failed after {self.retries} retries."
                    )
                    raise RuntimeError(
                        f"LLM API request failed after {self.retries} retries."
                    )

    def extract_json(
        self, instructions: str, text: str, schema_hint: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
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
                start = content.find("{") if "content" in locals() else -1
                end = content.rfind("}") if "content" in locals() else -1
                if start != -1 and end != -1:
                    return json.loads(content[start : end + 1])
            except Exception:
                pass
            return {}
