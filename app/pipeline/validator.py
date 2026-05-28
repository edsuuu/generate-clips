"""Valida a transcrição cruzando com o áudio original via LLM multimodal.

Usa Gemini Flash (multimodal) para ouvir o áudio + ler a transcrição do Whisper
e devolver uma lista de correções. As correções são aplicadas segmento a segmento,
preservando os word timestamps via redistribuição proporcional.

Se o provider configurado não suportar áudio, faz fallback gracioso (log + segue
sem validar).
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.llm.gemini import MULTIMODAL_MODEL_POOL, build_model_cascade
from app.llm.gemini import limiter as rate_limiter
from app.support.config import settings
from app.support.logger import logger
from app.support.types import Segment, Transcript, Word


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


SYSTEM_PROMPT = """Você é um revisor de transcrição em português brasileiro.

Receberá:
1. O ÁUDIO original do vídeo (anexado como arquivo).
2. A TRANSCRIÇÃO produzida pelo Whisper, segmento por segmento, com índices.

Sua tarefa é OUVIR o áudio e CORRIGIR erros de transcrição: palavras trocadas,
omissões, ortografia, gírias mal transcritas, nomes próprios escritos errado.

REGRAS:
- Mantenha a estrutura: número de segmentos e ordem inalterados.
- Corrija APENAS quando tiver certeza após ouvir o áudio. Em caso de dúvida, mantenha o original.
- Preserve pontuação natural e capitalização adequada.
- Não invente conteúdo nem adicione comentários.

Responda SOMENTE com JSON neste formato:
{
  "corrections": [
    {"segment_index": 3, "corrected_text": "texto corrigido completo do segmento"},
    {"segment_index": 7, "corrected_text": "outro segmento corrigido"}
  ]
}

Inclua APENAS segmentos que realmente precisam de correção. Se tudo estiver correto,
retorne {"corrections": []}."""


@dataclass
class _Correction:
    segment_index: int
    corrected_text: str


class TranscriptValidator:
    """Valida a transcrição comparando com o áudio via Gemini multimodal."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        fallback_models: list[str] | None = None,
    ):
        self.api_key = api_key or settings.gemini_api_key
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY não configurada — validator desativado")
        primary = model or settings.gemini_model

        if fallback_models is None:
            env_cascade = os.environ.get("GEMINI_MULTIMODAL_FALLBACKS", "").strip()
            fallback_models = (
                [m.strip() for m in env_cascade.split(",") if m.strip()] if env_cascade else None
            )
        self.models = build_model_cascade(primary, fallback_models, MULTIMODAL_MODEL_POOL)

    def validate(self, audio_path: Path, transcript: Transcript) -> Transcript:
        """Retorna o Transcript corrigido. Se falhar, retorna o original."""
        compressed_audio = self._compress_for_upload(audio_path)
        try:
            corrections = self._ask_llm(compressed_audio, transcript)
            if not corrections:
                logger.info("Validação: nenhuma correção sugerida")
                return transcript
            logger.info(f"Validação: {len(corrections)} segmento(s) corrigido(s)")
            return self._apply_corrections(transcript, corrections)
        except Exception as e:
            logger.warning(f"Validação de transcrição falhou ({e}). Mantendo original.")
            return transcript
        finally:
            if compressed_audio != audio_path:
                compressed_audio.unlink(missing_ok=True)

    def _compress_for_upload(self, audio_path: Path) -> Path:
        """Comprime para MP3 mono 24kbps — fica leve para envio inline (~3KB/s)."""
        out = audio_path.with_name(audio_path.stem + ".validate.mp3")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "24k",
            str(out),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"Falha ao comprimir áudio para validação: {result.stderr[-300:]}")
            return audio_path
        return out

    def _ask_llm(self, audio_path: Path, transcript: Transcript) -> list[_Correction]:
        audio_bytes = audio_path.read_bytes()
        size_mb = len(audio_bytes) / (1024 * 1024)
        if size_mb > 18:
            raise ValueError(f"Áudio muito grande para envio inline ({size_mb:.1f} MB)")

        mime = "audio/mp3" if audio_path.suffix.lower() == ".mp3" else "audio/wav"
        user_text = self._build_transcript_prompt(transcript)

        est_tokens = self._estimate_request_tokens(user_text, size_mb)

        payload_base = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": mime,
                                "data": base64.b64encode(audio_bytes).decode("ascii"),
                            }
                        },
                        {"text": user_text},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        logger.info(
            f"Validando transcrição com Gemini ({size_mb:.1f} MB áudio, "
            f"{len(transcript.segments)} segmentos)..."
        )

        data = self._query_with_fallback(payload_base, est_tokens)

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini retornou sem candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()

        # Remove cercas eventuais
        if text.startswith("```"):
            text = text.strip("`")
            text = text.split("\n", 1)[1] if "\n" in text else text
            if text.endswith("```"):
                text = text[:-3]

        parsed = json.loads(text)
        return [
            _Correction(
                segment_index=int(c["segment_index"]),
                corrected_text=str(c["corrected_text"]).strip(),
            )
            for c in parsed.get("corrections", [])
            if c.get("corrected_text", "").strip()
        ]

    def _build_transcript_prompt(self, transcript: Transcript) -> str:
        lines = [f"[{i}] {seg.text.strip()}" for i, seg in enumerate(transcript.segments)]
        transcript_text = "\n".join(lines)
        return (
            f"TRANSCRIÇÃO atual ({len(transcript.segments)} segmentos):\n\n"
            f"{transcript_text}\n\n"
            "Ouça o áudio anexado e devolva as correções no JSON especificado."
        )

    def _estimate_request_tokens(self, user_text: str, size_mb: float) -> int:
        return (
            _estimate_tokens(SYSTEM_PROMPT)
            + _estimate_tokens(user_text)
            + int(size_mb * 1500)
            + 4096
        )

    def _query_with_fallback(self, payload_base: dict, est_tokens: int) -> dict:
        last_error: Exception | None = None
        for model in self.models:
            if not rate_limiter.acquire(model, est_tokens, max_wait=45.0):
                logger.warning(f"[validator:{model}] sem capacidade. Próximo modelo.")
                continue

            try:
                data = self._request_model(model, payload_base)
                rate_limiter.record(model, est_tokens)
                return data
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429:
                    logger.warning(f"[validator:{model}] 429. Próximo modelo.")
                    continue
                raise
            except Exception as e:
                last_error = e
                logger.warning(f"[validator:{model}] erro: {e}. Próximo modelo.")
                continue
        raise RuntimeError(f"Todos os modelos esgotaram. Último: {last_error}")

    def _request_model(self, model: str, payload_base: dict) -> dict:
        url = f"{settings.gemini_base_url}/models/{model}:generateContent"
        with httpx.Client(timeout=240.0) as client:
            resp = client.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "X-goog-api-key": self.api_key,
                },
                json=payload_base,
            )
            resp.raise_for_status()
            data: dict = resp.json()
            return data

    def _apply_corrections(
        self, transcript: Transcript, corrections: list[_Correction]
    ) -> Transcript:
        """Aplica correções por segmento. Para preservar word timestamps,
        redistribuímos os timestamps das palavras antigas proporcionalmente
        às palavras novas."""
        index_to_corr = {c.segment_index: c for c in corrections}
        new_segments: list[Segment] = []

        for i, seg in enumerate(transcript.segments):
            if i not in index_to_corr:
                new_segments.append(seg)
                continue
            new_text = index_to_corr[i].corrected_text
            new_words = self._redistribute_words(seg, new_text)
            new_segments.append(
                Segment(
                    text=new_text,
                    start=seg.start,
                    end=seg.end,
                    words=new_words,
                )
            )

        return Transcript(
            language=transcript.language,
            duration=transcript.duration,
            segments=new_segments,
        )

    def _redistribute_words(self, seg: Segment, new_text: str) -> list[Word]:
        """Distribui timestamps proporcionalmente nas palavras novas.

        Mantém as fronteiras originais (start, end) e divide igualmente entre
        as palavras corrigidas — perde precisão palavra-a-palavra mas garante
        sincronia de legenda dentro do segmento.
        """
        tokens = [t for t in new_text.replace("\n", " ").split() if t]
        if not tokens:
            return []
        duration = max(0.0, seg.end - seg.start)
        per_word = duration / len(tokens) if tokens else 0.0
        words: list[Word] = []
        for j, tok in enumerate(tokens):
            w_start = seg.start + j * per_word
            w_end = seg.start + (j + 1) * per_word if j < len(tokens) - 1 else seg.end
            words.append(Word(text=tok, start=w_start, end=w_end))
        return words
