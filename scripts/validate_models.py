#!/usr/bin/env python3
"""Initialise every model the pipeline needs, one at a time, and report.

The stages load their models lazily, so a missing checkpoint, a rejected Hugging
Face token or an incompatible package only shows up partway through a run - after
other models have already been downloaded. This script loads each model by itself,
prints what happened, and releases it before moving on, so a failure names exactly
one component.

Which models are checked follows the configuration:

* Dialogue adaptation always talks to the local Qwen model served by vLLM at
  ``TRANSLATION_BASE_URL``; the step probes it with one JSON-mode request.
* ``TTS_ENGINE`` selects ``omnivoice`` (the default), or ``mms``.

Only the selected engine is loaded. Checking the other one would download many
gigabytes a run never touches and, worse, would leave the engine it *does* use
unchecked, so the pre-flight would pass and the run would still fail. To check the
other configuration, set the environment variable::

    TTS_ENGINE=mms python scripts/validate_models.py

::

    python scripts/validate_models.py
    python scripts/validate_models.py --only served
    python scripts/validate_models.py --only separation,mms

Each step drives the stage's *own* loader (``separation``'s model, ``diarization``'s
pipeline loader, ``transcription``'s model loader, ``translation``'s backend,
``tts``'s engines), so what is validated here is the same code path a real run
takes. The translation and synthesis steps go one step further than loading and run
one probe line through the model, because a checkpoint that loads can still reject
every input. ``scripts/test_gpu.py`` is the end-to-end run.

Models are dropped and CUDA's cache is emptied between steps, so the whole
validation fits on a single 48 GB card even though every model is loaded.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

# Invoking this file directly puts ``scripts/`` on ``sys.path`` rather than the
# project root, so the ``app`` package is made importable explicitly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEPARATOR = "-" * 60


def _gib(num_bytes: float) -> str:
    return f"{num_bytes / 1024**3:.1f} GiB"


def _vram() -> str:
    """Return the current CUDA allocation, or an empty string without CUDA."""

    try:
        import torch
    except ImportError:
        return ""

    if not torch.cuda.is_available():
        return ""
    free, total = torch.cuda.mem_get_info()
    return f"vram {_gib(total - free)} used of {_gib(total)}"


def _release() -> None:
    """Drop every cached engine and free what CUDA is holding."""

    from app.pipeline import tts

    tts.reset_engine_cache()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - no runtime installed
        pass


def _report(title: str, value: str) -> None:
    print(f"      {title:<18} {value}", flush=True)


# ---------------------------------------------------------------------------
# One loader per model
# ---------------------------------------------------------------------------


def load_torch() -> None:
    """Report the compute runtime every model below depends on."""

    import torch

    _report("torch", torch.__version__)
    _report("cuda available", str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        _report("device", torch.cuda.get_device_name(0))
        _report("cuda version", torch.version.cuda or "unknown")


def load_separation() -> None:
    """Load the configured BS-Roformer model and verify CUDA selection."""

    from app.pipeline import separation

    settings = separation.get_settings()
    torch = separation._require_cuda()  # noqa: SLF001
    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise RuntimeError(
            f"could not import audio-separator or one of its dependencies ({exc})"
        ) from exc

    cache_dir = settings.model_cache_dir / "audio-separator"
    cache_dir.mkdir(parents=True, exist_ok=True)
    separator = Separator(
        model_file_dir=str(cache_dir),
        output_dir=str(settings.work_dir),
        output_format="WAV",
        sample_rate=separation.REQUIRED_SAMPLE_RATE,
        use_soundfile=True,
        chunk_duration=None,
    )
    try:
        device = separation._configure_cuda_device(  # noqa: SLF001
            torch, separator, settings.device
        )
        _report("model", separation.MODEL_FILENAME)
        _report("GPU", torch.cuda.get_device_name(device))
        _report("native chunking", "BS-Roformer overlap-add")
        separator.load_model(model_filename=separation.MODEL_FILENAME)
        _report(
            "model inference device",
            separation._verify_model_cuda(separator),  # noqa: SLF001
        )
    finally:
        del separator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_pyannote() -> None:
    """Load the gated pyannote Community-1 diarization pipeline."""

    from app.pipeline import diarization

    settings = diarization.get_settings()
    token = settings.huggingface_token
    if not token:
        raise RuntimeError(
            "HUGGINGFACE_TOKEN is not set; the Community-1 checkpoint is gated, "
            "and the token's account must have accepted its conditions"
        )

    _report("model", settings.diarization_model)
    # The stage's own loader, so the device move and the "pipeline is None" case
    # are exercised exactly as a diarization run would exercise them.
    pipeline = diarization._load_pipeline(settings, token)  # noqa: SLF001
    _report("device", str(getattr(pipeline, "device", settings.device)))


def load_whisper() -> None:
    """Load the faster-whisper transcription model onto the configured device."""

    from app.pipeline import transcription

    settings = transcription.get_settings()
    _report("model", settings.transcription_model)
    _report("compute type", settings.transcription_compute_type)

    model = transcription._load_model(settings)  # noqa: SLF001
    _report("loaded", type(model).__name__)


def load_served_model() -> None:
    """Reach the local vLLM server and confirm it answers a JSON-mode request.

    A reachable endpoint that returns the model's own name is the smallest thing that
    proves the server is up, the configured checkpoint is loaded, and the client's
    request shape is accepted - which is worth knowing before a run reaches the
    adaptation stage and submits a whole film's windows.
    """

    import asyncio
    import json

    from app.pipeline import translation

    settings = translation.get_settings()
    _report("base url", settings.translation_base_url)
    _report("model", settings.translation_model)
    _report("batching", translation.describe_backend(settings))

    client = translation._client_for(settings)  # noqa: SLF001

    async def probe() -> str:
        try:
            return await client.complete(
                system="Reply with a JSON object.",
                user='Return exactly {"ok": true} and nothing else.',
            )
        finally:
            await client.aclose()

    try:
        reply = asyncio.run(probe())
    except translation.ApiClientError as exc:
        raise RuntimeError(
            f"the local vLLM server is not reachable at "
            f"{settings.translation_base_url}: {exc}"
        ) from exc
    json.loads(reply)
    _report("loaded", "answered a JSON-mode probe")


def load_mms() -> None:
    """Load MMS-TTS Amharic and synthesize one word.

    Synthesis is the step worth validating here, because it exercises the two things
    a load alone would not: the romanisation the checkpoint requires, and the
    duration predictor producing a waveform at all.
    """

    from app.pipeline import tts

    settings = tts.get_settings()
    _report("model", settings.tts_model)
    _report("device", settings.device)
    _report("seed / rate", f"{settings.mms_seed} / {settings.mms_speaking_rate:g}")

    if settings.tts_engine != "mms":
        raise RuntimeError(
            f"TTS_ENGINE is {settings.tts_engine!r}, so the MMS engine is not what a "
            "run would use; validate 'omnivoice' instead"
        )

    engine = tts.load_mms_engine(settings=settings)
    romanized = engine.romanize("ሰላም")
    _report("romanised", romanized)

    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as directory:
        written = engine.synthesize(
            text="ሰላም", destination=_Path(directory) / "probe.wav"
        )
        if not written.is_file() or written.stat().st_size == 0:
            raise RuntimeError("MMS-TTS wrote no audio for the probe line")
    _report("loaded", f"synthesized at {engine.sample_rate} Hz")


def load_omnivoice() -> None:
    """Load OmniVoice and synthesize one word in a cloned voice.

    Cloning is exercised, not just loading: the project's whole reason for choosing
    this engine is that it speaks as a specific voice, and a load alone would not show
    whether the reference was accepted.
    """

    from app.pipeline import tts

    settings = tts.get_settings()
    _report("model", settings.omnivoice_model)
    _report("device", settings.device)
    _report("steps / guidance", f"{settings.omnivoice_steps} / {settings.omnivoice_guidance_scale:g}")

    if settings.tts_engine != "omnivoice":
        raise RuntimeError(
            f"TTS_ENGINE is {settings.tts_engine!r}, so the OmniVoice engine is not "
            "what a run would use; validate that engine instead"
        )

    engine = tts.load_omnivoice_engine(settings=settings)

    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as directory:
        import numpy as _np
        import soundfile as _sf

        root = _Path(directory)
        # A reference is required: this engine has no voice of its own to fall back on,
        # which is precisely the difference from the single-voice engines.
        reference = root / "reference.wav"
        rate = 24_000
        seconds = 5.0
        tone = 0.4 * _np.sin(
            2.0 * _np.pi * 150.0 * _np.arange(int(seconds * rate)) / rate
        )
        _sf.write(str(reference), tone.astype(_np.float32), rate, subtype="PCM_16")

        written = engine.synthesize(
            text="ሰላም",
            voice_reference=reference,
            destination=root / "probe.wav",
        )
        if not written.is_file() or written.stat().st_size == 0:
            raise RuntimeError("OmniVoice wrote no audio for the probe line")

    _report("loaded", f"synthesized a cloned line at {engine.sample_rate} Hz")


#: Steps that every configuration needs, in pipeline order.
STEPS: tuple[tuple[str, str, object], ...] = (
    ("torch", "compute runtime", load_torch),
    ("separation", "audio-separator BS-Roformer vocals/instrumental", load_separation),
    ("pyannote", "pyannote Community-1 diarization", load_pyannote),
    ("whisper", "faster-whisper transcription", load_whisper),
)

#: Steps that depend on the configured backend and engine. Only the ones a run will
#: actually use are loaded: validating the other pair would download many gigabytes
#: the run never touches, and - worse - would leave the models it *does* use
#: unchecked, so the pre-flight would pass and the run would still fail.
CONFIGURED_STEPS: tuple[tuple[str, str, object, str], ...] = (
    (
        "served",
        "local vLLM adaptation (TRANSLATION_BASE_URL)",
        load_served_model,
        "vllm",
    ),
    (
        "omnivoice",
        "OmniVoice cloned-voice synthesis (TTS_ENGINE=omnivoice)",
        load_omnivoice,
        "omnivoice",
    ),
    ("mms", "MMS-TTS Amharic synthesis (TTS_ENGINE=mms)", load_mms, "mms"),
)


#: How to make a step that the current configuration does not select part of it.
ACTIVATION: dict[str, str] = {
    "served": "start the vLLM server at TRANSLATION_BASE_URL",
    "omnivoice": "set TTS_ENGINE=omnivoice",
    "mms": "set TTS_ENGINE=mms",
}


def configured_steps() -> tuple[tuple[str, str, object], ...]:
    """Return every step this configuration needs, in run order.

    The selection is taken from the settings themselves - ``TRANSLATION_BACKEND`` and
    ``TTS_ENGINE`` - so what is validated is what a run would load.
    """

    from app.pipeline import translation, tts

    settings = translation.get_settings()
    engine = tts.get_settings().tts_engine

    selected: list[tuple[str, str, object]] = list(STEPS)
    for name, description, loader, wanted in CONFIGURED_STEPS:
        # Dialogue adaptation is always local; only the speech engine is a choice.
        if name in ("mms", "omnivoice") and wanted != engine:
            continue
        selected.append((name, description, loader))
    return tuple(selected)


def all_step_names() -> set[str]:
    """Return every step name that exists, configured or not."""

    return {name for name, _, _ in STEPS} | {
        name for name, _, _, _ in CONFIGURED_STEPS
    }


def _cuda_available() -> bool | None:
    """Return CUDA availability, or ``None`` when PyTorch is not installed."""

    try:
        import torch
    except ImportError:
        return None
    return torch.cuda.is_available()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load each pipeline model on its own and report the result."
    )
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "comma-separated subset to validate "
            f"(one or more of: {', '.join(name for name, _, _ in configured_steps())})"
        ),
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help=(
            "run the model-loading steps even without CUDA. They download real "
            "weights and are meant for the GPU worker, not a development machine."
        ),
    )
    args = parser.parse_args(argv)

    steps = configured_steps()

    selected = {name for name, _, _ in steps}
    if args.only:
        requested = {part.strip() for part in args.only.split(",") if part.strip()}
        unknown = sorted(requested - selected - all_step_names())
        if unknown:
            print(f"unknown step(s): {', '.join(unknown)}", file=sys.stderr)
            print(
                f"known steps: {', '.join(sorted(all_step_names()))}",
                file=sys.stderr,
            )
            return 2
        # A step that exists but belongs to the other configuration is worth
        # distinguishing from a typo: the names are in this file's help and in the
        # README, so being told *how* to select one is the difference between a dead
        # end and a one-line fix.
        other = sorted(requested - selected)
        if other:
            print(
                f"not part of this configuration: {', '.join(other)}", file=sys.stderr
            )
            for name in other:
                print(f"  {name}: {ACTIVATION[name]}", file=sys.stderr)
            return 2
        selected = requested

    # Everything but the runtime report fetches weights, so refuse to do that on a
    # machine without a GPU: this is the check that stops a development box from
    # quietly filling its disk with checkpoints.
    if (selected - {"torch"}) and not args.allow_cpu and _cuda_available() is not True:
        print(
            "refusing to load models: torch reports no CUDA device on this machine.",
            file=sys.stderr,
        )
        print(
            "The steps below download real weights and are meant for the GPU worker.",
            file=sys.stderr,
        )
        print(
            "Run this on the pod, or pass --allow-cpu if you really mean to do it here.",
            file=sys.stderr,
        )
        return 2

    print("Amharic dubbing pipeline - model initialisation check")
    print(SEPARATOR)
    print("Models are loaded one at a time and released before the next one.\n")

    results: list[tuple[str, bool, float, str]] = []
    for name, description, loader in steps:
        if name not in selected:
            continue
        print(f"[{name}] {description}", flush=True)
        started = time.perf_counter()
        error = ""
        healthy = True
        try:
            loader()  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001 - every failure is reported
            healthy = False
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started

        vram = _vram()
        if vram:
            _report("memory", vram)
        if healthy:
            print(f"      OK ({elapsed:.1f} s)", flush=True)
        else:
            print(f"      FAIL ({elapsed:.1f} s)  {error}", flush=True)
        results.append((name, healthy, elapsed, error))
        _release()

    print(SEPARATOR)
    failed = [name for name, healthy, _, _ in results if not healthy]
    print(f"validated {len(results)} of {len(STEPS)} step(s) in {sum(r[2] for r in results):.1f} s")
    if failed:
        print(f"Result: FAIL ({', '.join(failed)})")
        print(
            "\nFix the first failure before starting the end-to-end run: see the "
            "component named above, not the model that happened to load after it."
        )
        return 1

    print("Result: OK - every selected model initialised")
    print("\nNext: python scripts/test_gpu.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
