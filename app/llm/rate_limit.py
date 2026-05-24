"""Rate limiter por modelo Gemini.

Mantém contadores em memória (sliding window) de RPM/TPM/RPD por modelo.
- Antes de uma chamada: `acquire(model, est_tokens)` decide se libera, aguarda
  ou indica que o modelo está esgotado (retorna False).
- Após a chamada: `record(model, tokens_consumidos)` registra o uso real.

Os limites default vêm da tabela de cotas free de 2026-05 (RPM/TPM/RPD).
Podem ser sobrescritos via env `GEMINI_LIMITS_JSON` se a cota mudar.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from app.support.logger import logger


# Defaults baseados na tabela de cotas free Gemini.
# RPM = requests/minute, TPM = tokens/minute, RPD = requests/day
DEFAULT_LIMITS: dict[str, dict[str, int]] = {
    # Fast text models
    "gemini-flash-latest":    {"rpm": 5,  "tpm": 250_000, "rpd": 20},
    "gemini-3.5-flash":       {"rpm": 5,  "tpm": 250_000, "rpd": 20},
    "gemini-3-flash":         {"rpm": 5,  "tpm": 250_000, "rpd": 20},
    "gemini-2.5-flash":       {"rpm": 5,  "tpm": 250_000, "rpd": 20},
    "gemini-2.0-flash":       {"rpm": 5,  "tpm": 250_000, "rpd": 20},
    # Mais folgados (lite)
    "gemini-3.1-flash-lite":  {"rpm": 15, "tpm": 250_000, "rpd": 500},
    "gemini-2.5-flash-lite":  {"rpm": 10, "tpm": 250_000, "rpd": 20},
}


def get_limits() -> dict[str, dict[str, int]]:
    """Lê os limites; permite override via env GEMINI_LIMITS_JSON."""
    override = os.environ.get("GEMINI_LIMITS_JSON")
    if override:
        try:
            extra = json.loads(override)
            return {**DEFAULT_LIMITS, **extra}
        except Exception as e:
            logger.warning(f"GEMINI_LIMITS_JSON inválido ({e}). Usando defaults.")
    return dict(DEFAULT_LIMITS)


@dataclass
class _ModelState:
    requests_minute: deque = field(default_factory=deque)  # timestamps (s) últimos 60s
    tokens_minute: deque = field(default_factory=deque)    # (ts, tokens) últimos 60s
    requests_day: deque = field(default_factory=deque)     # timestamps últimos 86400s
    lock: threading.Lock = field(default_factory=threading.Lock)


class GeminiRateLimiter:
    """Rate limiter thread-safe em memória."""

    def __init__(self, limits: dict[str, dict[str, int]] | None = None):
        self.limits = limits or get_limits()
        self._states: dict[str, _ModelState] = {}
        self._global_lock = threading.Lock()

    def _state(self, model: str) -> _ModelState:
        with self._global_lock:
            if model not in self._states:
                self._states[model] = _ModelState()
            return self._states[model]

    def _model_limits(self, model: str) -> dict[str, int]:
        # match exato; senão fallback genérico (5/250K/20)
        return self.limits.get(model, {"rpm": 5, "tpm": 250_000, "rpd": 20})

    def _cleanup(self, st: _ModelState, now: float) -> None:
        while st.requests_minute and now - st.requests_minute[0] > 60:
            st.requests_minute.popleft()
        while st.tokens_minute and now - st.tokens_minute[0][0] > 60:
            st.tokens_minute.popleft()
        while st.requests_day and now - st.requests_day[0] > 86_400:
            st.requests_day.popleft()

    def _tokens_in_minute(self, st: _ModelState) -> int:
        return sum(tk for _, tk in st.tokens_minute)

    def can_acquire(self, model: str, est_tokens: int) -> tuple[bool, Optional[float], str]:
        """Verifica sem aguardar. Retorna (ok, wait_seconds, reason)."""
        limits = self._model_limits(model)
        st = self._state(model)
        now = time.monotonic()

        with st.lock:
            self._cleanup(st, now)

            rpd = len(st.requests_day)
            if rpd >= limits["rpd"]:
                return False, None, f"RPD esgotado ({rpd}/{limits['rpd']})"

            rpm = len(st.requests_minute)
            if rpm >= limits["rpm"]:
                wait = 60 - (now - st.requests_minute[0]) + 0.1
                return False, max(wait, 0.5), f"RPM cheio ({rpm}/{limits['rpm']})"

            tpm_used = self._tokens_in_minute(st)
            if tpm_used + est_tokens > limits["tpm"]:
                if st.tokens_minute:
                    wait = 60 - (now - st.tokens_minute[0][0]) + 0.1
                    return False, max(wait, 0.5), (
                        f"TPM próximo do limite ({tpm_used}+{est_tokens} > {limits['tpm']})"
                    )
                return False, 60.0, "TPM excedido"

            return True, None, "ok"

    def acquire(self, model: str, est_tokens: int, max_wait: float = 30.0) -> bool:
        """Aguarda até liberar (RPM/TPM) ou retorna False se RPD esgotou
        ou demora mais do que max_wait."""
        waited = 0.0
        while True:
            ok, wait, reason = self.can_acquire(model, est_tokens)
            if ok:
                return True
            if wait is None:
                logger.warning(f"[{model}] limite hard: {reason}")
                return False
            if waited + wait > max_wait:
                logger.warning(
                    f"[{model}] {reason} (precisaria esperar {wait:.1f}s, max {max_wait}s)"
                )
                return False
            logger.info(f"[{model}] {reason} — aguardando {wait:.1f}s")
            time.sleep(wait)
            waited += wait

    def record(self, model: str, tokens_used: int) -> None:
        st = self._state(model)
        now = time.monotonic()
        with st.lock:
            st.requests_minute.append(now)
            st.requests_day.append(now)
            st.tokens_minute.append((now, max(0, tokens_used)))
            self._cleanup(st, now)

    def reset(self, model: str | None = None) -> None:
        with self._global_lock:
            if model is None:
                self._states.clear()
            else:
                self._states.pop(model, None)


# Singleton
limiter = GeminiRateLimiter()
