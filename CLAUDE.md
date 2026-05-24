# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que o projeto faz

Pipeline em Python que baixa um vídeo do YouTube, transcreve com Whisper, valida a transcrição cruzando com o áudio via Gemini multimodal, seleciona os melhores momentos com LLM, corta com **face tracking + active speaker detection** (crop dinâmico 9:16), queima legendas estilo TikTok e gera título/descrição/hashtags. Roda como CLI (`main.py process URL`) ou como API HTTP com SSE + WebSocket (`main.py serve`).

A camada Laravel foi removida — **não recriar** a menos que o usuário peça explicitamente. Foco é só Python.

## Comandos comuns

Sempre rodar com o venv local — o `.venv/bin/pip` pode ter shebang quebrado por causa de moves passados; use `.venv/bin/python -m pip` se acontecer.

```bash
# CLI direto
.venv/bin/python main.py process "https://www.youtube.com/watch?v=..."
.venv/bin/python main.py process URL --no-validate --no-subtitles --no-face-tracking  # modo rápido para debug
.venv/bin/python main.py process URL --subtitle-only                                    # só legenda o vídeo inteiro

# API HTTP (porta 8765 default)
.venv/bin/python main.py serve

# Subir DB schema (cria tabelas se não existirem)
.venv/bin/python -c "from app.db import init_db; init_db()"

# Validar imports após mexer em algum módulo
.venv/bin/python -c "from app.pipeline.runner import PipelineRunner; from app.api.main import app; print('ok')"

# Reiniciar API quando travar a porta
lsof -ti :8765 | xargs kill -9; .venv/bin/python main.py serve

# Acompanhar job via SSE
curl -sN http://localhost:8765/jobs/<id>/events

# Status pontual
curl http://localhost:8765/jobs/<id>
```

MySQL roda em Docker (container `mysql_container`, porta 3306, root/root). Database `auto_post` precisa existir:

```bash
docker exec mysql_container mysql -uroot -proot -e "CREATE DATABASE IF NOT EXISTS auto_post CHARACTER SET utf8mb4;"
```

Ollama (fallback do `auto` provider) precisa estar rodando para o fallback funcionar:

```bash
ollama serve &       # se não estiver subido
ollama list          # confere que tem gemma2:9b
```

## Arquitetura — o que precisa entender para ser produtivo

O coração é o **`PipelineRunner`** em `app/pipeline/runner.py`. Ele orquestra TODAS as etapas e emite `ProgressEvent(stage, percent, message, detail)` em cada passo. CLI e API consomem o **mesmo runner** — não duplique lógica de pipeline no `main.py` ou em `app/api/`.

Ordem das etapas (com pesos no progresso 0–100): `download(5) → transcribe(22) → validate(5) → analyze(3) → face_track(15) → cut(15) → subtitle(20) → metadata(15)`. Se mexer em alguma etapa, atualize `WEIGHTS` em `runner.py` para o `percent` continuar somando 100.

### Camadas independentes

- **`app/pipeline/`** — cada arquivo é uma etapa pura. Recebem inputs tipados (`Transcript`, `Highlight`, `VideoInfo`) e devolvem o próximo. Sem efeitos colaterais além de arquivos em `output/`. Adicione etapas novas registrando no `runner.py` (e no dict `WEIGHTS`).
- **`app/llm/`** — providers de IA atrás da interface `LLMProvider`. `factory.get_provider(name)` é o único ponto de entrada. O `auto` provider tenta Gemini primeiro e cai para Ollama em falha — não chame `GeminiProvider()` direto fora do factory, isso quebra o fallback.
- **`app/api/`** — FastAPI. `main.py` define rotas, `jobs.py` roda o pipeline em thread separada e publica eventos via `bus` (pub/sub em memória), `schemas.py` são os DTOs Pydantic. O `bus.publish()` é o que alimenta SSE e WebSocket simultaneamente.
- **`app/db/`** — SQLAlchemy + MySQL. Models em `models.py` (`Job`, `Cut`). Sem migrations (usa `Base.metadata.create_all()` em `init_db()`).
- **`app/support/`** — config (Pydantic Settings lendo `.env`), tipos do domínio, logger.

### Gemini — rate limiting + cascata

`app/llm/gemini.py` tem cascata de modelos. Antes de cada chamada, consulta `app/llm/rate_limit.py` (singleton `limiter`) que mantém sliding window de RPM/TPM/RPD por modelo. Em rate limit ou 429, troca pro próximo modelo da cascata. Tabela de cotas free está em `DEFAULT_LIMITS` no `rate_limit.py` — atualize lá se a cota mudar. Override via env `GEMINI_LIMITS_JSON` (JSON dict).

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
