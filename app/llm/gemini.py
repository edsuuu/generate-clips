"""Provider Gemini via REST com rate limiting e cascata de fallback de modelos.

- Aceita uma cascata de modelos (default: gemini-flash-latest → 2.5-flash-lite → 3.1-flash-lite).
- Antes de cada chamada, consulta o RateLimiter: se passou do RPM/TPM, aguarda
  (até max_wait); se passou do RPD, marca o modelo como esgotado e tenta o próximo.
- Em 429/5xx, faz backoff exponencial e/ou troca para o próximo modelo.
"""

from __future__ import annotations

import os
import time

import httpx

from app.llm.base import LLMProvider
from app.llm.rate_limit import limiter
from app.support.config import settings
from app.support.logger import logger


BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Cascata default — sobrescreva via env GEMINI_FALLBACK_MODELS=modeloA,modeloB
DEFAULT_FALLBACKS = [
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
]


def _estimate_tokens(text: str) -> int:
    """Heurística: 1 token ≈ 4 chars (válido para PT-BR/EN)."""
    return max(1, len(text) // 4)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        fallback_models: list[str] | None = None,
    ):
        key = api_key or settings.gemini_api_key
        if not key:
            raise ValueError("GEMINI_API_KEY não configurada")
        self.api_key = key

        primary = model or settings.gemini_model

        if fallback_models is None:
            env_cascade = os.environ.get("GEMINI_FALLBACK_MODELS", "").strip()
            fallback_models = (
                [m.strip() for m in env_cascade.split(",") if m.strip()]
                if env_cascade else list(DEFAULT_FALLBACKS)
            )

        # Mantém ordem, sem duplicar o primário
        self.models = [primary] + [m for m in fallback_models if m != primary]
        self._exhausted: set[str] = set()

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        est_tokens = _estimate_tokens(system) + _estimate_tokens(user) + 2048
        last_error: Exception | None = None

        for model in self.models:
            if model in self._exhausted:
                continue

            if not limiter.acquire(model, est_tokens, max_wait=45.0):
                logger.warning(f"[gemini:{model}] sem capacidade. Próximo modelo.")
                self._exhausted.add(model)
                continue

            try:
                text = self._call(model, system, user, json_mode=json_mode)
                limiter.record(model, est_tokens)
                return text
            except httpx.HTTPStatusError as e:
                last_error = e
                code = e.response.status_code
                body = e.response.text[:300]
                if code == 429:
                    logger.warning(f"[gemini:{model}] 429 (rate). Próximo modelo.")
                    self._exhausted.add(model)
                    continue
                if code in (500, 502, 503, 504):
                    logger.warning(f"[gemini:{model}] {code} — retry com backoff (2s).")
                    time.sleep(2.0)
                    try:
                        text = self._call(model, system, user, json_mode=json_mode)
                        limiter.record(model, est_tokens)
                        return text
                    except Exception as e2:
                        last_error = e2
                        continue
                raise RuntimeError(f"Gemini erro {code}: {body}") from e
            except Exception as e:
                last_error = e
                logger.warning(f"[gemini:{model}] erro inesperado: {e}. Próximo modelo.")
                continue

        raise RuntimeError(
            f"Todos os modelos Gemini falharam ou esgotaram cota. Último: {last_error}"
        )

    def _call(self, model: str, system: str, user: str, *, json_mode: bool) -> str:
        url = f"{BASE_URL}/models/{model}:generateContent"

        generation_config: dict = {"temperature": 0.3}
        if json_mode:
            generation_config["responseMimeType"] = "application/json"

        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation_config,
        }

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "X-goog-api-key": self.api_key,
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini {model} sem candidates: {data}")

        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)
