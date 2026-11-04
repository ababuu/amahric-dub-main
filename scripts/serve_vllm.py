"""Launch the local vLLM server the dialogue-adaptation stage talks to.

Dialogue adaptation runs on Qwen-2.5-Instruct served by vLLM on the same GPU host as
the rest of the pipeline, so this launcher and the client must agree on two numbers:
the checkpoint, and how much VRAM the server may pre-allocate. Both come from
:meth:`app.config.Settings.vllm_server_arguments`, which is the same settings object
the adaptation stage reads.

Usage::

    python scripts/serve_vllm.py            # exec the server in the foreground
    python scripts/serve_vllm.py --print    # print the command and exit

The command it runs is equivalent to::

    vllm serve Qwen/Qwen2.5-14B-Instruct \\
        --host 127.0.0.1 --port 8000 \\
        --gpu-memory-utilization 0.90 --max-num-seqs 8 --dtype auto
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings, get_settings  # noqa: E402


def build_command(settings: Settings) -> list[str]:
    """Return the argv that starts the server for ``settings``."""

    return ["vllm", "serve", *settings.vllm_server_arguments()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/serve_vllm.py",
        description="Serve the local Qwen dialogue model with vLLM.",
    )
    parser.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="print the command instead of running it",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    command = build_command(settings)

    if args.print_only:
        print(" ".join(command))
        return 0

    if shutil.which("vllm") is None:
        print(
            "vllm is not on PATH. Install the server in the GPU environment:\n"
            "    pip install vllm\n"
            "then re-run this launcher.",
            file=sys.stderr,
        )
        return 1

    print("starting: " + " ".join(command), file=sys.stderr, flush=True)
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
