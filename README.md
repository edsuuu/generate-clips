# auto-post

Transforma vídeos longos do YouTube em **cortes curtos legendados** para Shorts, Reels e TikTok — com **face tracking dinâmico**, **detecção do speaker ativo** e **API HTTP** para integração com outros sistemas.

```
URL do YouTube  →  download  →  transcrição  →  análise  →  cortes  →  legendas  →  metadados
```

Pode rodar como **CLI** (`python main.py process URL`) ou como **API** (`python main.py serve` → POST /jobs).

---

## Estrutura do projeto

Pensada no estilo Laravel: `main.py` é o entry point (como `artisan`), e tudo em `app/` segue divisão por responsabilidade.

```
auto-post/
├── README.md
├── pyproject.toml          # ≈ composer.json
├── requirements.txt
├── .env / .env.example     # variáveis (chaves de API, DB, paths)
├── .gitignore
├── main.py                 # ≈ artisan — entry point CLI (Typer)
│
├── app/                    # ≈ app/ do Laravel — todo o código
│   ├── pipeline/           # ≈ app/Services — etapas do pipeline
│   │   ├── runner.py           # orquestrador (callback de progresso)
│   │   ├── downloader.py       # baixa do YouTube + cache local
│   │   ├── transcriber.py      # transcrição (faster-whisper large-v3)
│   │   ├── analyzer.py         # escolhe os melhores momentos via LLM
│   │   ├── face_tracker.py     # face tracking + active speaker detection
│   │   ├── cutter.py           # corta + crop dinâmico 9:16
│   │   ├── subtitler.py        # legendas estilo TikTok (queima na imagem)
│   │   └── metadata.py         # gera título + descrição + hashtags
│   │
│   ├── llm/                # ≈ app/Integrations — providers de IA
│   │   ├── __init__.py         # base + factory combinados
│   │   ├── auto.py             # tenta Gemini, fallback para local
│   │   ├── ollama.py           # Ollama local
│   │   └── gemini/             # subpacote Gemini (models, rate_limit, etc.)
│   │
│   ├── api/                # ≈ routes/ + controllers — FastAPI
│   │   ├── main.py             # app + endpoints REST/SSE/WS
│   │   ├── jobs.py             # worker async + pub/sub + webhook
│   │   └── schemas.py          # DTOs Pydantic (request/response)
│   │
│   ├── db/                 # ≈ database/ — SQLAlchemy + MySQL
│   │   ├── session.py          # engine + sessionmaker + Base
│   │   └── models.py           # Job, Cut
│   │
│   └── support/            # ≈ app/Support — config, tipos, logger
│       ├── config.py           # settings via Pydantic + .env
│       ├── types.py            # dataclasses do domínio
│       └── logger.py           # rich logger
│
├── models/                 # modelos pré-treinados (MediaPipe .task/.tflite)
└── output/                 # vídeos processados (≈ storage/app)
```

---

## Pré-requisitos

- **Python 3.11+** (testado com 3.12)
- **ffmpeg com libass** — necessário para queimar legendas
  - macOS: `brew tap homebrew-ffmpeg/ffmpeg && brew install homebrew-ffmpeg/ffmpeg/ffmpeg`
  - O `brew install ffmpeg` padrão **não tem libass** e falha nas legendas.
- **MySQL 8+** (pode ser via Docker)
- **Ollama** (opcional — usado como fallback se Gemini falhar)
  - `brew install ollama` → `ollama pull gemma2:9b`
- Chave de API do **Gemini** (recomendado, é o default)

---

## Instalação

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copie e ajuste `.env`:

```bash
cp .env.example .env
# Edite e cole GEMINI_API_KEY e credenciais do MySQL
```

Crie o database e tabelas:

```bash
# Cria o database (se ainda não existir)
mysql -uroot -proot -e "CREATE DATABASE IF NOT EXISTS auto_post CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

# Cria as tabelas
.venv/bin/python -c "from app.db import init_db; init_db()"
```

Os modelos do MediaPipe já estão em `models/`. Se faltarem:

```bash
mkdir -p models && cd models
curl -O https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
curl -O https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

---

## Como usar — CLI

### Pipeline completo

```bash
python main.py process "https://www.youtube.com/watch?v=..."
```

Faz tudo: download → transcrição → análise → cortes verticais com face tracking → legendas → metadados (título/descrição/hashtags em PT-BR).

### Flags úteis

```bash
# Quantidade de cortes (default: 6-20, gap mínimo 1s)
python main.py process URL --min-cuts 8 --max-cuts 25 --min-gap 1.5

# Forçar provider de IA
python main.py process URL --llm gemini       # Google Gemini
python main.py process URL --llm local        # Ollama (gemma2:9b)

# Apenas legendar o vídeo inteiro (sem cortar)
python main.py process URL --subtitle-only

# Pular partes para testar mais rápido
python main.py process URL --no-subtitles --no-metadata
python main.py process URL --no-face-tracking   # crop centralizado estático
python main.py process URL --no-vertical        # mantém proporção original

# Exportar resultado completo em JSON
python main.py process URL --json-result resultado.json
```

---

## Como usar — API HTTP

Sobe o servidor:

```bash
python main.py serve                 # 0.0.0.0:8765 (configurável via .env)
python main.py serve --port 9000     # outra porta
python main.py serve --reload        # auto-reload em dev
```

Docs interativas: <http://localhost:8765/docs>

### 1) Criar um job

```bash
curl -X POST http://localhost:8765/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=...",
    "llm": "gemini",
    "min_cuts": 6,
    "max_cuts": 15,
    "webhook_url": "https://meu-app.com/api/auto-post-callback",
    "webhook_token": "Bearer xpto-token-aqui",
    "webhook_header": "Authorization"
  }'
```

Resposta `202 Accepted`:

```json
{
  "id": "660b847a-28fd-43df-a309-0d95eeec6d7c",
  "status": "pending",
  "progress": 0,
  "stage": "pending",
  ...
}
```

O pipeline roda em background. Se `webhook_url` for fornecido, o servidor faz `POST` nesse endpoint ao concluir, enviando o `webhook_token` no header `webhook_header` (default `Authorization`):

```json
{
  "job_id": "660b847a-...",
  "success": true,
  "result": { "video": {...}, "cuts": [...] },
  "error": null
}
```

### 2) Acompanhar progresso — opção A: polling REST

```bash
curl http://localhost:8765/jobs/<job_id>
```

### 3) Acompanhar progresso — opção B: SSE (Server-Sent Events)

```bash
curl -N http://localhost:8765/jobs/<job_id>/events
```

Eventos típicos:

```
event: snapshot
data: {"job_id": "...", "stage": "transcribe", "percent": 12.5, ...}

event: cut
data: {"stage": "cut", "percent": 67.2, "message": "Corte PT3 OK", "index": 3, "total": 6}

event: done
data: {"stage": "done", "percent": 100, "message": "Concluído", "result": {...}}
```

No browser:

```javascript
const es = new EventSource(`http://localhost:8765/jobs/${jobId}/events`);
es.onmessage = (e) => console.log(JSON.parse(e.data));
es.addEventListener("done", () => es.close());
```

### 4) Acompanhar progresso — opção C: WebSocket

```bash
# precisa do `websocat` ou `wscat`
websocat ws://localhost:8765/jobs/<job_id>/ws
```

Recebe eventos no formato:

```json
{"type": "progress", "stage": "subtitle", "percent": 82.5, "message": "Legenda PT4 OK"}
```

No browser:

```javascript
const ws = new WebSocket(`ws://localhost:8765/jobs/${jobId}/ws`);
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

### 5) Listar jobs

```bash
curl http://localhost:8765/jobs?limit=20
```

---

## Saída em disco

```
output/
└── <video_id>/
    ├── source.mp4              # original na melhor qualidade
    ├── meta.json               # cache (title, duration)
    ├── full_subtitled.mp4      # gerado por --subtitle-only
    └── cuts/
        ├── PT1.mp4             # corte 1 (1080x1920, legendado, face tracking)
        ├── PT1.json            # título, descrição, hashtags, score
        ├── PT2.mp4
        ├── PT2.json
        └── ...
```

Os cortes:

- Têm nome sequencial `PT1`, `PT2`, ... em ordem cronológica (nunca volta no tempo)
- Duração entre **60s e 80s** (≈ 1min a 1min20s — configurável)
- Verticais 1080×1920 (9:16) com **crop seguindo o rosto do speaker**
- **Legendas estilo TikTok** queimadas (3 palavras visíveis, palavra ativa em amarelo, fonte Arial Black)
- Cada `.json` traz título chamativo, descrição com emoji e 10–15 hashtags em PT-BR

---

## Configuração (`.env`)

| Variável | Default | Descrição |
|---|---|---|
| `LLM_PROVIDER` | `auto` | `auto` (Gemini→local), `gemini`, `local` |
| `GEMINI_API_KEY` | — | Sua chave do Google AI Studio |
| `GEMINI_MODEL` | `gemini-flash-latest` | Modelo primário do Gemini |
| `GEMINI_FALLBACK_MODELS` | vazio | Se vazio, gira entre `gemini-flash-latest`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.0-flash`, `gemini-2.0-flash-lite` e `gemini-2.5-pro` |
| `GEMINI_MULTIMODAL_FALLBACKS` | vazio | Cascata do validator de áudio; usa só modelos multimodais completos |
| `GEMINI_LIMITS_JSON` | vazio | Override por projeto/conta dos limites RPM/TPM/RPD do Gemini |
| `OLLAMA_MODEL` | `gemma2:9b` | Modelo Ollama (fallback local) |
| `WHISPER_MODEL` | `large-v3` | Modelo Whisper |
| `WHISPER_LANGUAGE` | `pt` | Idioma da transcrição |
| `MIN_CUTS` / `MAX_CUTS` | `6` / `20` | Quantidade-alvo de cortes |
| `MIN_CUT_DURATION` / `MAX_CUT_DURATION` | `60` / `80` | Duração de cada corte (s) |
| `MIN_GAP_BETWEEN_CUTS` | `1.0` | Gap mínimo entre cortes (s) |
| `FACE_TRACKING_ENABLED` | `true` | Liga/desliga face tracking |
| `FACE_TRACKING_SAMPLE_FPS` | `6` | Frames/segundo amostrados |
| `FFMPEG_ENCODER` | `auto` | `h264_videotoolbox` no macOS, `h264_nvenc` no Windows/NVIDIA e fallback `libx264` |
| `FFMPEG_CRF` | `23` | CRF usado no fallback `libx264` |
| `FFMPEG_PRESET` | `veryfast` | Preset do fallback `libx264` |
| `FFMPEG_VIDEO_BITRATE` | `5M` | Bitrate alvo do `h264_videotoolbox` |
| `FFMPEG_NVENC_PRESET` | `p4` | Preset usado no encoder `h264_nvenc` |
| `FFMPEG_MAX_CONCURRENT_RENDERS` | `1` | Máximo de renders FFmpeg simultâneos no processo local |
| `DB_HOST` / `DB_PORT` | `127.0.0.1` / `3306` | MySQL |
| `DB_DATABASE` / `DB_USER` / `DB_PASSWORD` | `auto_post` / `root` / `root` | MySQL |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8765` | Servidor HTTP |
| `WEBHOOK_TIMEOUT_SECONDS` | `30` | Timeout do POST de callback/webhook |
| `WEBHOOK_FAIL_JOB_ON_ERROR` | `false` | Se `true`, falha do webhook derruba o job; por padrão só loga warning |

---

## Fluxo interno

1. **Runner** (`app/pipeline/runner.py`) — orquestra todas as etapas e emite `ProgressEvent(stage, percent, message)` em cada passo. CLI e API consomem o mesmo runner.
2. **Downloader** — yt-dlp em **melhor qualidade**. Reusa `output/<id>/source.*` se já existir (cache local — não bate no YouTube).
3. **Transcriber** — Whisper large-v3 com `beam_size=10`, prompt PT-BR, word timestamps, VAD filter.
4. **Analyzer** — LLM seleciona 6–20 melhores momentos (60–80s). Valida ordem temporal e gap mínimo, escolhe o de maior score em caso de conflito.
5. **FaceTracker** — para cada highlight, amostra frames a 6fps, MediaPipe detecta faces + landmarks dos lábios, librosa mede energia do áudio. Decide o **speaker ativo** combinando movimento dos lábios × energia × tamanho da face. Suaviza trajetória com Savitzky-Golay.
6. **Cutter** — render frame-a-frame com OpenCV usando a trajetória dinâmica; mux do áudio original via ffmpeg. Output 1080×1920 H.264.
7. **Subtitler** — gera `.ass` com 3 palavras por frame (palavra ativa destacada em amarelo). ffmpeg queima via `subtitles=filename=...` (precisa libass), preferindo `h264_videotoolbox` no Mac, `h264_nvenc` em Windows com NVIDIA e caindo para `libx264 veryfast` quando necessário.
8. **MetadataGenerator** — LLM gera título chamativo + descrição com emoji + 10–15 hashtags em PT-BR para cada corte.

---

## Por onde começar (lendo o código)

1. **`main.py`** — entry point CLI. Mostra como o Runner é chamado.
2. **`app/pipeline/runner.py`** — leia a função `run()`: vê todas as etapas em ordem.
3. **`app/support/types.py`** — dataclasses do domínio (`Transcript`, `Highlight`, `Cut`).
4. **`app/support/config.py`** — todas as configs disponíveis.
5. **`app/pipeline/*.py`** — cada arquivo é uma etapa do pipeline.
6. **`app/api/main.py`** — endpoints HTTP, SSE, WebSocket.
7. **`app/api/jobs.py`** — worker async + pub/sub + webhook.
8. **`app/llm/__init__.py`** — interface de provider de IA e factory.

---

## Roadmap

- [ ] Cache de transcrição (não retranscrever vídeos já processados)
- [ ] Agendamento de posts (YouTube Shorts, Instagram Reels, Facebook Reels, TikTok)
- [ ] Publicação via APIs oficiais quando disponível; automação alternativa quando não
