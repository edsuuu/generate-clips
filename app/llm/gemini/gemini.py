"""Provider Gemini via REST com rate limiting e cascata de fallback de modelos.

- Aceita uma cascata ampla de modelos de texto e distribui a carga entre eles.
- Antes de cada chamada, consulta o RateLimiter: se passou do RPM/TPM, aguarda
  (até max_wait); se passou do RPD, marca o modelo como esgotado e tenta o próximo.
- Em 429/5xx, faz backoff exponencial e/ou troca para o próximo modelo.
"""

from __future__ import annotations

import os
import threading
import time

import httpx

from app.llm import LLMProvider
from app.llm.gemini.models import TEXT_MODEL_POOL, build_model_cascade
from app.llm.gemini.rate_limit import limiter
from app.support.config import settings
from app.support.logger import logger


BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

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

        if fallback_models is None:
            env_cascade = os.environ.get("GEMINI_FALLBACK_MODELS", "").strip()
            fallback_models = (
                [m.strip() for m in env_cascade.split(",") if m.strip()]
                if env_cascade else None
            )

        primary = model or settings.gemini_model
        self.models = build_model_cascade(primary, fallback_models, TEXT_MODEL_POOL)
        self._exhausted: set[str] = set()
        self._rotation_index = 0
        self._rotation_lock = threading.Lock()

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        est_tokens = _estimate_tokens(system) + _estimate_tokens(user) + 2048
        last_error: Exception | None = None
        if len(self._exhausted) >= len(self.models):
            logger.warning("[gemini] todos os modelos estavam marcados como esgotados; resetando cache local.")
            self._exhausted.clear()

        ordered_models = self._ordered_models()
        logger.info(f"[gemini] ordem desta chamada: {', '.join(ordered_models)}")

        for model in ordered_models:
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

    def _ordered_models(self) -> list[str]:
        if len(self.models) <= 1:
            return list(self.models)

        with self._rotation_lock:
            start = self._rotation_index % len(self.models)
            self._rotation_index += 1

        return self.models[start:] + self.models[:start]

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
