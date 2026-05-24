from app.llm import LLMProvider
from app.support.logger import logger


class AutoProvider(LLMProvider):
    """Tenta um provider primário; se falhar, cai para o fallback.

    Útil para 'gemini com fallback para Ollama local' — se a API estiver
    fora do ar, sem quota ou sem internet, o pipeline ainda completa.
    """

    name = "auto"

    def __init__(self, primary: LLMProvider, fallback: LLMProvider):
        self.primary = primary
        self.fallback = fallback
        self._primary_failed = False

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        if not self._primary_failed:
            try:
                return self.primary.complete(system, user, json_mode=json_mode)
            except Exception as e:
                logger.warning(
                    f"Provider primário ({self.primary.name}) falhou: {e}. "
                    f"Caindo para fallback ({self.fallback.name})."
                )
                self._primary_failed = True

        return self.fallback.complete(system, user, json_mode=json_mode)
