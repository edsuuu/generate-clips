import httpx

from app.llm.base import LLMProvider
from app.support.config import settings


class OllamaProvider(LLMProvider):
    name = "local"

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or settings.ollama_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.3},
        }
        if json_mode:
            payload["format"] = "json"

        with httpx.Client(timeout=600.0) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["message"]["content"]
