import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from app.pipeline.runner import PipelineOptions, PipelineRunner, ProgressEvent
from app.support.config import settings

app = typer.Typer(
    help="auto-post — gera cortes curtos legendados a partir de vídeos do YouTube",
    no_args_is_help=True,
)
console = Console()


@app.command("process")
def process(
    url: str = typer.Argument(..., help="URL do vídeo do YouTube"),
    llm: str = typer.Option(
        settings.llm_provider, "--llm",
        help="Provider LLM: auto | local | claude | gemini | gpt",
    ),
    output_dir: Path = typer.Option(
        settings.output_dir, "--output-dir", "-o",
        help="Diretório de saída",
    ),
    min_cuts: int = typer.Option(
        settings.min_cuts, "--min-cuts", help="Mínimo de cortes a gerar"
    ),
    max_cuts: int = typer.Option(
        settings.max_cuts, "--max-cuts", help="Máximo de cortes a gerar"
    ),
    min_gap: float = typer.Option(
        settings.min_gap_between_cuts, "--min-gap",
        help="Gap mínimo (s) entre o fim de um corte e o início do próximo",
    ),
    no_subtitles: bool = typer.Option(False, "--no-subtitles", help="Pula legendas"),
    no_vertical: bool = typer.Option(
        False, "--no-vertical", help="Mantém proporção original (sem 9:16)"
    ),
    no_metadata: bool = typer.Option(
        False, "--no-metadata", help="Pula geração de título/descrição/hashtags"
    ),
    no_face_tracking: bool = typer.Option(
        False, "--no-face-tracking",
        help="Desativa face tracking (crop centralizado estático)",
    ),
    no_validate: bool = typer.Option(
        False, "--no-validate",
        help="Pula a validação da transcrição com áudio (mais rápido)",
    ),
    subtitle_only: bool = typer.Option(
        False, "--subtitle-only",
        help="Pula análise/cortes/metadata e gera APENAS o vídeo completo legendado",
    ),
    json_result: Path | None = typer.Option(
        None, "--json-result", help="Escreve resultado completo em JSON neste arquivo"
    ),
):
    """Processa um vídeo do YouTube e gera cortes curtos PT1, PT2, ..."""
    console.rule(f"[bold cyan]auto-post[/] — {url}")

    def on_progress(ev: ProgressEvent) -> None:
        console.print(
            f"[dim]{ev.percent:5.1f}%[/] [cyan]{ev.stage:12}[/] {ev.message}"
        )

    runner = PipelineRunner(on_progress=on_progress)
    result = runner.run(
        url=url,
        options=PipelineOptions(
            llm=llm,
            output_dir=output_dir,
            min_cuts=min_cuts,
            max_cuts=max_cuts,
            min_gap=min_gap,
            no_subtitles=no_subtitles,
            no_vertical=no_vertical,
            no_metadata=no_metadata,
            no_face_tracking=no_face_tracking,
            no_validate=no_validate,
            subtitle_only=subtitle_only,
        ),
    )

    if result.get("mode") == "cuts":
        _print_summary(result)
    console.rule("[bold green]Concluído")

    if json_result is not None:
        json_result.parent.mkdir(parents=True, exist_ok=True)
        json_result.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )


@app.command("serve")
def serve(
    host: str = typer.Option(settings.api_host, "--host"),
    port: int = typer.Option(settings.api_port, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload em mudança de código"),
):
    """Sobe a API HTTP (FastAPI + uvicorn)."""
    import uvicorn
    uvicorn.run("app.api.main:app", host=host, port=port, reload=reload)


@app.command("version")
def version():
    """Mostra a versão do auto-post."""
    console.print("auto-post 0.3.0")


def _print_summary(result: dict) -> None:
    video_title = result["video"]["title"]
    table = Table(title=f"Cortes gerados — {video_title}")
    table.add_column("Nome", style="cyan", no_wrap=True)
    table.add_column("Início", justify="right")
    table.add_column("Fim", justify="right")
    table.add_column("Dur.", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Título", style="green")

    for c in result["cuts"]:
        table.add_row(
            c["name"],
            f"{c['start_seconds']:.1f}s",
            f"{c['end_seconds']:.1f}s",
            f"{c['duration_seconds']:.1f}s",
            f"{c['score']:.0f}",
            c["title"] or "—",
        )
    console.print(table)


if __name__ == "__main__":
    app()
