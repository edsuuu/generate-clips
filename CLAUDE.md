# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que o projeto faz

Pipeline em Python que baixa um vídeo do YouTube, transcreve com Whisper, valida a transcrição cruzando com o áudio via Gemini multimodal, seleciona os melhores momentos com LLM, corta com **face tracking + active speaker detection** (crop dinâmico 9:16), queima legendas estilo TikTok e gera título/descrição/hashtags. Roda **exclusivamente como API HTTP** (FastAPI) com SSE + WebSocket — não há mais CLI.

Este repositório é **só Python** — não tem código Laravel e **não recriar** a menos que o usuário peça explicitamente. Na operação real, um app Laravel externo é o **orquestrador e dono do banco**: ele chama os endpoints granulares da API, e o Python roda **stateless** (sem banco), só executando as etapas, gravando em MinIO/disco e devolvendo o resultado via callback.

## Comandos comuns

Sempre rodar com o venv local — o `.venv/bin/pip` pode ter shebang quebrado por causa de moves passados; use `.venv/bin/python -m pip` se acontecer.

```bash
# API HTTP (porta 8765 default) — sem banco, stateless
.venv/bin/python main.py
AUTO_POST_RELOAD=1 .venv/bin/python main.py   # com auto-reload em mudança de código

# Validar imports após mexer em algum módulo
.venv/bin/python -c "from app.api.main import app; import main; print('ok')"

# Reiniciar API quando travar a porta
lsof -ti :8765 | xargs kill -9; .venv/bin/python main.py

# Acompanhar job via SSE (o job_id volta no 202 do endpoint que disparou)
curl -sN http://localhost:8765/jobs/<job_id>/events
```

A API **não usa banco**. Persistência fica em arquivos locais (`output/`) + MinIO. Para o modo API, MinIO precisa estar acessível (default `http://127.0.0.1:9000`, bucket `auto-post`):

```bash
docker run -d -p 9000:9000 -p 9001:9001 --name minio \
  -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"
```

Ollama (fallback do `auto` provider) precisa estar rodando para o fallback funcionar:

```bash
ollama serve &       # se não estiver subido
ollama list          # confere que tem gemma2:9b
```

## Arquitetura — o que precisa entender para ser produtivo

A orquestração é feita pelos **workflows granulares** em **`app/pipeline/workflows/video.py`** (`ingest_video`, `recommend_cuts`, `render_cuts`, `subtitle_full_video`), que reusam as **classes de etapa** em `app/pipeline/` (`Downloader`, `Transcriber`, `Analyzer`, `FaceTracker`, `Cutter`, `Subtitler`) — não duplique lógica de etapa fora desses módulos. Cada workflow faz uma fatia do pipeline, sobe artefatos pro MinIO e dispara callback.

O entry point `main.py` (raiz) só sobe o uvicorn apontando para `app.api.main:app` — não tem lógica de pipeline.

### Camadas independentes

- **`app/pipeline/`** — cada arquivo é uma etapa pura. Recebem inputs tipados (`Transcript`, `Highlight`, `VideoInfo`) e devolvem o próximo. Sem efeitos colaterais além de arquivos em `output/`. Etapas novas são compostas dentro dos workflows em `app/pipeline/workflows/video.py`.
- **`app/llm/`** — providers de IA atrás da interface `LLMProvider`. `get_provider(name)` (definido em `app/llm/__init__.py`) é o único ponto de entrada. O `auto` provider tenta Gemini primeiro e cai para Ollama em falha — não chame `GeminiProvider()` direto fora do factory, isso quebra o fallback.
- **`app/api/`** — FastAPI. `main.py` define rotas, `jobs.py` roda o pipeline em thread separada e publica eventos via `bus` (pub/sub em memória), `schemas.py` são os DTOs Pydantic. O `bus.publish()` é o que alimenta SSE e WebSocket simultaneamente.
- **`app/db/`** — SQLAlchemy + MySQL. Models em `models.py` (`Job`, `Cut`). Sem migrations (usa `Base.metadata.create_all()` em `init_db()`).
- **`app/support/`** — config (Pydantic Settings lendo `.env`), tipos do domínio, logger.

### Gemini — rate limiting + cascata

`app/llm/gemini/gemini.py` tem cascata de modelos. Antes de cada chamada, consulta `app/llm/gemini/rate_limit.py` (singleton `limiter`) que mantém sliding window de RPM/TPM/RPD por modelo. Em rate limit ou 429, troca pro próximo modelo da cascata. Tabela de cotas free está em `DEFAULT_LIMITS` no `rate_limit.py` — atualize lá se a cota mudar. Override via env `GEMINI_LIMITS_JSON` (JSON dict).

O `TranscriptValidator` em `app/pipeline/validator.py` usa Gemini multimodal (precisa aceitar áudio) e tem uma cascata **separada** com só modelos multimodais (`MULTIMODAL_FALLBACKS`). Não use Gemini lite no validator.

### Face tracking + crop dinâmico

`app/pipeline/face_tracker.py` usa **MediaPipe Tasks API** (não a antiga `mp.solutions`, que foi removida em 0.10.x). Modelos `.task`/`.tflite` ficam em `models/` (na raiz, não em `app/models/`). Se o face tracker reclamar de modelo não encontrado, confira `MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"` em `face_tracker.py` — esse é o path correto desde a reorganização para `app/`.

Active Speaker Detection: para cada frame amostrado, calcula `lip_activity × audio_energy + size_bonus` por face e escolhe a maior pontuação como speaker. Trajetória suavizada via Savitzky-Golay (scipy). Quando não detecta face, mantém última posição conhecida.

O `Cutter.cut_dynamic()` renderiza frame-a-frame com OpenCV usando a trajetória, depois faz mux do áudio original via ffmpeg. **Não** dá pra fazer só com `crop=W:H:x:y` do ffmpeg porque a trajetória é arbitrária.

### Gotchas conhecidos (NÃO repetir esses bugs)

1. **`initial_prompt` do Whisper vaza para a transcrição** se for longo/específico. Deixe `WHISPER_INITIAL_PROMPT=""` no `.env`. Se precisar, use frase curta e neutra ("Batalha de rima."). Também mantive `condition_on_previous_text=False` por mesma razão.
2. **ffmpeg do `brew install ffmpeg` padrão NÃO tem libass.** O filtro `subtitles` falha com "No such filter". Tem que usar o tap `homebrew-ffmpeg/ffmpeg`.
3. **Filtro subtitles no ffmpeg 8.1+ exige `subtitles=filename=arquivo.ass`** (não `subtitles=arquivo.ass`). E precisa rodar com `cwd` no diretório do `.ass` para evitar problemas de escape do `:`.
4. **Fonte para ASS:** usar **`Arial Black`** (libass acha via coretext no macOS). Fontes obscuras tipo Montserrat falham silenciosamente — a legenda fica invisível.
5. **`yt-dlp` pode bloquear com "Sign in to confirm you're not a bot"** quando bate muito. Por isso o `Downloader` extrai `video_id` da URL e prioriza cache local (`output/<id>/source.*` + `meta.json`) antes de bater no YouTube.
6. **`.venv` pode ter shebang antigo apontando para `python/.venv/`** (antes da reorganização). Use `.venv/bin/python -m pip` se `.venv/bin/pip` falhar com "bad interpreter".
7. **Cortes obrigatoriamente entre 60s e 80s** (1:00 a 1:20). Configurável em `MIN_CUT_DURATION`/`MAX_CUT_DURATION` do `.env` mas esse é o range pedido pelo usuário; não mudar default sem ser solicitado.

### Persistência

Estado vive em duas camadas:
- **`output/<youtube_id>/`** — arquivos físicos (vídeo, áudio, transcripts, cortes, .ass, .json). Serve de cache: existência do `source.*` + `meta.json` pula o download.
- **MySQL** — tabelas `jobs` (status, progresso, webhook config, payload do resultado) e `cuts` (uma row por corte gerado). Usado pela API para histórico e SSE snapshots iniciais.

### Webhook

Quando `webhook_url` é fornecido na criação do job, ao final do pipeline o `_fire_webhook()` em `app/api/jobs.py` faz POST do payload com `{job_id, success, result, error}`. O header de auth é configurável (`webhook_header` no payload do job, default `Authorization`) — útil porque cada cliente usa um nome diferente (Authorization, X-API-Key, X-Webhook-Token, etc).
