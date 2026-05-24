from app.llm.base import LLMProvider
from app.support.config import settings


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "Pacote 'anthropic' não instalado. Rode: pip install anthropic"
            ) from e

        key = api_key or settings.anthropic_api_key
        if not key:
            raise ValueError("ANTHROPIC_API_KEY não configurada")

        self.client = Anthropic(api_key=key)
        self.model = model or settings.claude_model

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        if json_mode:
            system = (
                system
                + "\n\nResponda SOMENTE com JSON válido, sem texto antes ou depois."
            )

        msg = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text  # type: ignore[union-attr]
