import json
import re
from abc import ABC, abstractmethod
from typing import Any

from app.support.config import settings
from app.support.logger import logger


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """Retorna o texto da resposta. Se json_mode=True, deve retornar JSON válido."""

    def complete_json(self, system: str, user: str) -> Any:
        raw = self.complete(system, user, json_mode=True)
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(text: str) -> Any:
        text = text.strip()
        # Remove cercas de código se vierem
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Tenta encontrar o primeiro bloco JSON na string
            match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(1))


def get_provider(name: str | None = None) -> LLMProvider:
    name = (name or settings.llm_provider or "auto").lower()

    if name in {"local", "ollama"}:
        from app.llm.ollama import OllamaProvider
        return OllamaProvider()
    if name in {"gemini", "google"}:
        from app.llm.gemini import GeminiProvider
        return GeminiProvider()
    if name == "auto":
        return _build_auto_provider()

    raise ValueError(
        f"Provider '{name}' desconhecido. Use: local, gemini, auto"
    )


def _build_auto_provider() -> LLMProvider:
    """Tenta construir Gemini como primário; se sem chave, retorna local direto."""
    from app.llm.ollama import OllamaProvider
    from app.llm.auto import AutoProvider

    try:
        from app.llm.gemini import GeminiProvider
        primary = GeminiProvider()
    except (ValueError, ImportError) as e:
        logger.warning(f"Gemini indisponível ({e}). Usando Ollama local direto.")
        return OllamaProvider()

    return AutoProvider(primary=primary, fallback=OllamaProvider())


__all__ = ["LLMProvider", "get_provider"]
