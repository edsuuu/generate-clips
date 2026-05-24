import json
import re
from abc import ABC, abstractmethod
from typing import Any

    
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
