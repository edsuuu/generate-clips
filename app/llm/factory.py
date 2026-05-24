from app.llm.base import LLMProvider
from app.support.config import settings
from app.support.logger import logger


def get_provider(name: str | None = None) -> LLMProvider:
    name = (name or settings.llm_provider or "auto").lower()

    if name in {"local", "ollama"}:
        from app.llm.ollama import OllamaProvider
        return OllamaProvider()
    if name in {"claude", "anthropic"}:
        from app.llm.claude import ClaudeProvider
        return ClaudeProvider()
    if name in {"gemini", "google"}:
        from app.llm.gemini import GeminiProvider
        return GeminiProvider()
    if name in {"gpt", "openai"}:
        from app.llm.gpt import GPTProvider
        return GPTProvider()
    if name == "auto":
        return _build_auto_provider()

    raise ValueError(
        f"Provider '{name}' desconhecido. Use: local, claude, gemini, gpt, auto"
    )


def _build_auto_provider() -> LLMProvider:
    """Tenta construir Gemini como primário; se sem chave, retorna local direto."""
    from app.llm.auto import AutoProvider
    from app.llm.ollama import OllamaProvider

    try:
        from app.llm.gemini import GeminiProvider
        primary = GeminiProvider()
    except (ValueError, ImportError) as e:
        logger.warning(f"Gemini indisponível ({e}). Usando Ollama local direto.")
        return OllamaProvider()

    return AutoProvider(primary=primary, fallback=OllamaProvider())
