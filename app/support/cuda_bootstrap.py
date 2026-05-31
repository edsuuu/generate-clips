"""Garante que as libs CUDA dos pacotes pip da NVIDIA fiquem visíveis ao loader.

O `ctranslate2` (backend do faster-whisper) carrega `libcublas.so.12` /
`libcudnn.so.9` por *soname* via `dlopen`. Quando essas libs vêm dos wheels
`nvidia-cublas-cu12` / `nvidia-cudnn-cu12`, elas ficam em
`site-packages/nvidia/<pkg>/lib/` — fora do path padrão do linker — e a
inferência na GPU falha com "Library libcublas.so.12 is not found".

Diferente do PyTorch, o `ctranslate2` não injeta esses diretórios no
`LD_LIBRARY_PATH` sozinho. Como o dynamic loader lê o `LD_LIBRARY_PATH` apenas
na inicialização do processo, definir a env em runtime não basta: é preciso
re-exec do processo já com o path correto. Esta função faz isso de forma
idempotente (sem loop de re-exec) e só age no Linux com os wheels presentes.

Importe e chame `ensure_cuda_libs()` o mais cedo possível — antes de qualquer
import de `ctranslate2`/`faster_whisper`.
"""

from __future__ import annotations

import glob
import os
import sys


def _nvidia_lib_dirs() -> list[str]:
    try:
        import nvidia  # namespace package dos wheels nvidia-*-cu12
    except ImportError:
        return []

    dirs: list[str] = []
    # nvidia é namespace package: __file__ é None, use __path__.
    for root in getattr(nvidia, "__path__", []):
        dirs.extend(d for d in glob.glob(os.path.join(root, "*", "lib")) if os.path.isdir(d))
    return sorted(set(dirs))


def ensure_cuda_libs() -> None:
    """Coloca os dirs de lib dos wheels NVIDIA no LD_LIBRARY_PATH via re-exec.

    No-op fora do Linux ou quando os wheels não estão instalados (ex.: macOS
    com MLX, ou ambiente CPU puro).
    """
    if sys.platform != "linux":
        return

    lib_dirs = _nvidia_lib_dirs()
    if not lib_dirs:
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = current.split(os.pathsep) if current else []

    missing = [d for d in lib_dirs if d not in current_parts]
    if not missing:
        # Já estão no path (provavelmente após um re-exec) — evita loop infinito.
        return

    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([*missing, *current_parts])
    # Re-exec para o dynamic loader reler o LD_LIBRARY_PATH atualizado.
    os.execv(sys.executable, [sys.executable, *sys.argv])
