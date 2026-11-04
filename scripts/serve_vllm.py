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
    """Return the argv that starts the server for ``settings``, or ``[]`` if vLLM is absent.

    Three launch paths, in order:

    * the ``vllm`` console script, which is what ``pip install vllm`` normally puts on
      ``PATH``;
    * the module form, for an install whose script directory is not on ``PATH`` (a
      ``--user`` install, or a venv whose ``bin`` was not activated) - the same server,
      reached through the interpreter that has the package;
    * nothing, which the caller reports with install instructions.

    The module form deliberately uses ``vllm.entrypoints.openai.api_server`` rather
    than ``vllm serve``: it is the entry point that has existed across every released
    version, and every argument this launcher passes is a flag rather than a positional
    subcommand.
    """

    arguments = settings.vllm_server_arguments()
    executable = shutil.which("vllm")
    if executable:
        return [executable, "serve", *arguments]

    try:
        import vllm  # noqa: F401
    except ImportError:
        return []
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server", *arguments]


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
        print(
            " ".join(command)
            if command
            else "vllm serve " + " ".join(settings.vllm_server_arguments())
        )
        return 0

    if not command:
        print(
            "vLLM is not installed in this environment.\n"
            "If pip stopped with `uninstall-no-record-file` for PyJWT, the copy on this\n"
            "image was installed by apt and has no pip metadata. Install a pip-managed\n"
            "copy over it, then retry:\n"
            "    pip install --ignore-installed PyJWT\n"
            "    pip install vllm\n"
            "then: python scripts/serve_vllm.py",
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
