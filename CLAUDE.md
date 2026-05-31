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

A API **não usa banco**. Cada job grava arquivos efêmeros em `settings.temp_dir` (`/tmp/auto-post/<job_id>/`, limpo no fim) e sobe o que é durável pro MinIO. Para o modo API, MinIO precisa estar acessível (default `http://127.0.0.1:9000`, bucket `auto-post`):

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

- **`app/pipeline/`** — cada arquivo é uma etapa pura. Recebem inputs tipados (`Transcript`, `Highlight`, `VideoInfo`) e devolvem o próximo. Sem efeitos colaterais além de arquivos no `temp_dir` do job. Etapas novas são compostas dentro dos workflows em `app/pipeline/workflows/video.py`.
- **`app/llm/`** — providers de IA atrás da interface `LLMProvider`. `get_provider(name)` (definido em `app/llm/__init__.py`) é o único ponto de entrada. O `auto` provider tenta Gemini primeiro e cai para Ollama em falha — não chame `GeminiProvider()` direto fora do factory, isso quebra o fallback.
- **`app/api/`** — FastAPI. `main.py` define rotas e dispara cada workflow em thread separada via `_run_background`. `events.py` tem o `bus` (pub/sub em memória) + helper `emit()`; `schemas.py` são os DTOs Pydantic. O `bus.publish()` é o que alimenta SSE e WebSocket simultaneamente. **Não há banco** — o Laravel é o dono da persistência.
- **`app/storage/`** — wrapper do MinIO (`MinioStorageProvider`) usado pelos workflows pra subir originals, HLS, cortes e legendas.
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
5. **`yt-dlp` pode bloquear com "Sign in to confirm you're not a bot"** quando bate muito. Por isso o `Downloader` extrai `video_id` da URL e prioriza cache local (`<temp_dir do job>/<id>/source.*` + `meta.json`) antes de bater no YouTube.
6. **`.venv` pode ter shebang antigo apontando para `python/.venv/`** (antes da reorganização). Use `.venv/bin/python -m pip` se `.venv/bin/pip` falhar com "bad interpreter".
7. **Cortes obrigatoriamente entre 60s e 80s** (1:00 a 1:20). Configurável em `MIN_CUT_DURATION`/`MAX_CUT_DURATION` do `.env` mas esse é o range pedido pelo usuário; não mudar default sem ser solicitado.
8. **MediaPipe GPU delegate (Metal) aborta o processo no macOS.** Criar o `FaceDetector`/`FaceLandmarker` com `delegate=GPU` funciona, mas no primeiro `.detect()` o `FaceLandmarker` faz `abort()` em C++ (`unsupported ImageFrame format`) — crash **não capturável** por try/except, derruba a API. Por isso `FACE_TRACKING_DELEGATE=auto` mapeia pra **CPU**; `gpu` é opt-in com aviso. O ganho de GPU vem do ffmpeg (VideoToolbox), não do face tracking.

### Aceleração por GPU no Linux/NVIDIA (CUDA)

No Linux com GPU NVIDIA o pipeline roda tudo na GPU:
- **Encode/Decode**: `FFMPEG_ENCODER=auto` pega `h264_nvenc` (VBR + `-cq`) e `FFMPEG_HWACCEL=auto` injeta `-hwaccel cuda`. Precisa de ffmpeg compilado com nvenc/cuda (`ffmpeg -encoders | grep nvenc`).
- **Transcrição**: `faster-whisper` (CTranslate2) com `device=cuda`, `compute_type=float16` — detectado automaticamente.
- **Face tracking**: MediaPipe usa delegate GPU (OpenGL/EGL) no Linux — o crash do delegate Metal é exclusivo do macOS.
- **Libs CUDA do faster-whisper**: o CTranslate2 carrega `libcublas.so.12`/`libcudnn.so.9` por *soname* via `dlopen` e **não** descobre os wheels `nvidia-*-cu12` sozinho (diferente do PyTorch). Falha com "Library libcublas.so.12 is not found". A correção está em `app/support/cuda_bootstrap.py` (`ensure_cuda_libs()`, chamado no topo do `main.py` antes de qualquer import de ctranslate2): ele coloca os dirs `site-packages/nvidia/*/lib` no `LD_LIBRARY_PATH` e dá **re-exec** do processo (o loader só lê `LD_LIBRARY_PATH` na inicialização — setar em runtime não basta). É idempotente (não entra em loop de re-exec). **Não** remova essa chamada nem instale `faster-whisper` esperando que ele ache as libs sozinho — instale `nvidia-cublas-cu12` e `nvidia-cudnn-cu12` (já no `requirements.txt` com marker Linux/x86_64).

### Aceleração por GPU (Apple Silicon / VideoToolbox)

O pipeline empurra o processamento de vídeo pra GPU do Mac via ffmpeg VideoToolbox, centralizado em `app/support/ffmpeg.py`:
- **Encode**: `build_video_encode_profile()` já usa `h264_videotoolbox` no macOS (`FFMPEG_ENCODER=auto`).
- **Decode**: `build_decode_args()` injeta `-hwaccel videotoolbox` antes do `-i` em todos os comandos de decode (`FFMPEG_HWACCEL=auto`). Sem `-hwaccel_output_format` de propósito, pra os frames voltarem à memória e o filtro `subtitles`/libass continuar funcionando.
- **Corte dinâmico**: `Cutter._cut_dynamic()` manda os frames croppados do OpenCV **crus (bgr24) direto ao encoder de hardware via `pipe:0`** e muxa o áudio no mesmo passo — `run_with_progress(..., stdin_frames=...)`. Não existe mais o arquivo intermediário `.novideo.mp4` em mp4v (software) nem o double-encode. **Não** reintroduza o `cv2.VideoWriter`.
- **Transcrição**: `app/pipeline/transcriber.py` tem dois backends (`WHISPER_BACKEND=auto|mlx|faster`). `auto` usa **`mlx-whisper`** (GPU Metal) no macOS Apple Silicon quando instalado, senão `faster-whisper` (CTranslate2, só CPU/CUDA — não tem Metal). O `mlx-whisper` está nas deps com marker `sys_platform == 'darwin' and platform_machine == 'arm64'`, então o CI Linux ignora; o import é lazy dentro do método pra não quebrar fora do Mac. MLX 0.4.x **não tem beam search** (usa greedy `temperature=0`) nem `vad_filter` — a supressão de alucinação fica com os thresholds `no_speech`/`logprob`/`compression_ratio`. Mapa modelo→repo MLX em `_MLX_MODEL_REPOS`; override via `WHISPER_MLX_MODEL`.

### Persistência

A API é **stateless** — nada de banco. Estado de job vive só em memória durante a execução:
- **`settings.temp_dir/<job_id>/`** — arquivos efêmeros (vídeo baixado, áudio, transcripts, cortes, .ass). Limpo no `finally` de cada workflow. Dentro disso o `Downloader` cacheia por `<youtube_id>`: existência do `source.*` + `meta.json` pula o download.
- **MinIO** — artefatos duráveis (originals, HLS, cortes legendados). É o que o Laravel consome.
- **`bus`** (`app/api/events.py`) — pub/sub em memória do progresso. Mantém o último snapshot por job pra clientes SSE/WS que conectam tarde.

### Webhook

Quando `callback_url` vem no payload do job, ao final do workflow o `_post_webhook()` em `app/pipeline/workflows/video.py` faz POST do resultado completo (com `event`, `status`, `files`, `payloads`, etc.). O header de auth é configurável (`callback_header` no payload, default `Authorization`) — útil porque cada cliente usa um nome diferente (Authorization, X-API-Key, X-Webhook-Token, etc.). Em falha, loga warning e segue (a menos que `WEBHOOK_FAIL_JOB_ON_ERROR=true`).
