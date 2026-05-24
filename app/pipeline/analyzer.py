from app.llm.base import LLMProvider
from app.support.logger import logger
from app.support.config import settings
from app.support.types import Highlight, Transcript


SYSTEM_PROMPT = """Você é um editor especialista em conteúdo viral para TikTok, Reels e Shorts.
Sua tarefa é analisar a transcrição de um vídeo longo e selecionar os MELHORES momentos
para virarem cortes curtos de alto potencial de retenção e viralização.

Critérios para escolher um momento:
- Alta retenção: começa com gancho forte (pergunta, afirmação polêmica, estatística, história)
- Impacto emocional, humor, curiosidade ou informação valiosa que se sustenta sozinha
- Faz sentido fora de contexto (não dependente de algo anterior do vídeo)
- Tem um clímax/resolução dentro do trecho
- Frases completas: corte só em fronteiras naturais de fala (início/fim de frase)

REGRAS DURAS:
1. Cada momento DEVE ter entre {min}s e {max}s de duração.
2. Os momentos DEVEM estar em ordem temporal crescente (start1 < start2 < ...).
3. Pode haver gap CURTO entre cortes (mínimo {min_gap}s entre o fim de um e o início do próximo).
   Não precisa cobrir o vídeo inteiro nem deixar grandes intervalos vazios.
4. Selecione entre {min_cuts} e {max_cuts} momentos. PRIORIZE QUANTIDADE quando o vídeo tem
   muito conteúdo aproveitável — não deixe momentos fortes de fora por preguiça.
5. Não invente timestamps: use exatamente os ranges que aparecem na transcrição.

Responda SOMENTE com um JSON neste formato (sem comentários, sem markdown):
{{
  "highlights": [
    {{"start": 12.5, "end": 58.2, "score": 9, "reason": "gancho forte + dado surpreendente"}},
    {{"start": 62.0, "end": 105.0, "score": 8, "reason": "história emocional com clímax"}}
  ]
}}"""


USER_TEMPLATE = """Vídeo de {duration:.0f} segundos. Idioma: {language}.

Transcrição com timestamps (formato: [start-end] texto):
{transcript}

Selecione os melhores momentos respeitando todas as regras.
Lembre: priorize quantidade ({min_cuts}-{max_cuts} cortes) quando há conteúdo aproveitável."""


def _format_transcript(transcript: Transcript, max_chars: int = 120000) -> str:
    lines = []
    total = 0
    for seg in transcript.segments:
        line = f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text.strip()}"
        if total + len(line) > max_chars:
            lines.append("... (transcrição truncada)")
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


class Analyzer:
    def __init__(
        self,
        provider: LLMProvider,
        min_cuts: int | None = None,
        max_cuts: int | None = None,
        min_gap: float | None = None,
    ):
        self.provider = provider
        self.min_cuts = min_cuts if min_cuts is not None else settings.min_cuts
        self.max_cuts = max_cuts if max_cuts is not None else settings.max_cuts
        self.min_gap = min_gap if min_gap is not None else settings.min_gap_between_cuts

    def select_highlights(self, transcript: Transcript) -> list[Highlight]:
        system = SYSTEM_PROMPT.format(
            min=settings.min_cut_duration,
            max=settings.max_cut_duration,
            min_gap=self.min_gap,
            min_cuts=self.min_cuts,
            max_cuts=self.max_cuts,
        )
        user = USER_TEMPLATE.format(
            duration=transcript.duration,
            language=transcript.language,
            transcript=_format_transcript(transcript),
            min_cuts=self.min_cuts,
            max_cuts=self.max_cuts,
        )

        logger.info(
            f"Analisando transcrição via {self.provider.name} "
            f"(alvo: {self.min_cuts}-{self.max_cuts} cortes, gap min {self.min_gap}s)..."
        )
        data = self.provider.complete_json(system, user)

        raw = data.get("highlights", []) if isinstance(data, dict) else data
        highlights = [
            Highlight(
                start=float(h["start"]),
                end=float(h["end"]),
                score=float(h.get("score", 0)),
                reason=str(h.get("reason", "")),
            )
            for h in raw
        ]

        validated = self._validate_and_clean(highlights, transcript.duration)
        logger.info(
            f"{len(raw)} highlights propostos, {len(validated)} válidos após filtragem"
        )
        return validated[: self.max_cuts]

    def _validate_and_clean(
        self, highlights: list[Highlight], video_duration: float
    ) -> list[Highlight]:
        """Garante: dentro do vídeo, dentro de min/max, ordem crescente, gap mínimo."""
        cleaned: list[Highlight] = []
        for h in highlights:
            start = max(0.0, h.start)
            end = min(video_duration, h.end)
            duration = end - start

            if duration < settings.min_cut_duration:
                continue
            if duration > settings.max_cut_duration:
                end = start + settings.max_cut_duration

            cleaned.append(Highlight(start=start, end=end, score=h.score, reason=h.reason))

        # Ordena por start. Quando dois cortes invadem o gap mínimo, mantém o de
        # maior score e descarta o outro.
        cleaned.sort(key=lambda x: x.start)
        ordered: list[Highlight] = []
        for h in cleaned:
            if ordered and h.start < ordered[-1].end + self.min_gap:
                if h.score > ordered[-1].score:
                    ordered[-1] = h
                continue
            ordered.append(h)
        return ordered
