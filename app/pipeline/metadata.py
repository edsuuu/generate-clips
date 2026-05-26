from app.llm import LLMProvider
from app.support.logger import logger
from app.support.types import CutMetadata, Highlight, Transcript

SYSTEM_PROMPT = """Você é um social media especialista em copy para Shorts, Reels e TikTok.
Receberá o trecho de um vídeo e deve criar:

1. **título**: até 60 caracteres, magnético, com gancho ou pergunta. Em PT-BR.
2. **descrição**: 1-3 frases para a legenda do post, instigante, em PT-BR. Pode ter emoji.
3. **hashtags**: lista de 10 a 15 hashtags em PT-BR (sem #) focadas em alcance,
   tendência e relevância ao tema. Misture hashtags amplas (#viral, #fyp) com nichos específicos.

Responda SOMENTE com JSON:
{
  "title": "...",
  "description": "...",
  "hashtags": ["tag1", "tag2", ...]
}"""


USER_TEMPLATE = """Tema geral do vídeo de origem: {video_title}

Trecho ({duration:.0f}s) a ser publicado:
\"\"\"
{text}
\"\"\"

Gere título, descrição e hashtags."""


def _extract_text_in_range(transcript: Transcript, start: float, end: float) -> str:
    parts: list[str] = []
    for seg in transcript.segments:
        if seg.end < start or seg.start > end:
            continue
        parts.append(seg.text.strip())
    return " ".join(parts)


class MetadataGenerator:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate(
        self,
        video_title: str,
        transcript: Transcript,
        highlight: Highlight,
    ) -> CutMetadata:
        text = _extract_text_in_range(transcript, highlight.start, highlight.end)
        user = USER_TEMPLATE.format(
            video_title=video_title,
            duration=highlight.duration,
            text=text,
        )

        data = self.provider.complete_json(SYSTEM_PROMPT, user)

        title = str(data.get("title", "")).strip()[:80]
        description = str(data.get("description", "")).strip()
        hashtags_raw = data.get("hashtags", [])
        hashtags = [
            str(h).lstrip("#").strip().replace(" ", "") for h in hashtags_raw if str(h).strip()
        ]

        logger.info(f"Metadados gerados: {title}")
        return CutMetadata(title=title, description=description, hashtags=hashtags)
