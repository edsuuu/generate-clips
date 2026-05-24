from app.llm.base import LLMProvider
from app.support.config import settings


class GPTProvider(LLMProvider):
    name = "gpt"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Pacote 'openai' não instalado. Rode: pip install openai"
            ) from e

        key = api_key or settings.openai_api_key
        if not key:
            raise ValueError("OPENAI_API_KEY não configurada")

        self.client = OpenAI(api_key=key)
        self.model = model or settings.openai_model

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
