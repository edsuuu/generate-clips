"""Catalogo de modelos Gemini usados pelo projeto."""

from __future__ import annotations

TEXT_MODEL_POOL = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-pro",
]

# Validator: apenas modelos multimodais completos; sem lite.
MULTIMODAL_MODEL_POOL = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
]

MODEL_ALIASES = {
    "gemini-2-flash": "gemini-2.0-flash",
    "gemini-2-flash-lite": "gemini-2.0-flash-lite",
}


def normalize_model_name(name: str) -> str:
    normalized = (name or "").strip().lower()
    return MODEL_ALIASES.get(normalized, normalized)


def build_model_cascade(
    primary: str | None,
    fallbacks: list[str] | None,
    defaults: list[str],
) -> list[str]:
    ordered = [primary] if primary else []
    if fallbacks is not None:
        ordered.extend(fallbacks)
    else:
        ordered.extend(defaults)

    seen: set[str] = set()
    models: list[str] = []
    for model in ordered:
        normalized = normalize_model_name(model)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        models.append(normalized)
    return models
