"""Central configuration for the Amharic dubbing pipeline.

All runtime configuration is read from environment variables so that the exact
same code runs locally and on the RunPod GPU worker without any changes.

Design rules
------------
* A local ``.env`` file is loaded when present, but it never overrides variables
  that are already defined in the real environment. This lets the RunPod worker
  environment inject values that win over a developer's local file.
* Secrets (``HUGGINGFACE_TOKEN``) are optional. The project must stay runnable
  without them so that the scaffold, the health check, and the tests work before
  any credentials exist.
* Secrets are *never* hard-coded and are *never* logged or printed.

Environment variables
---------------------
``HUGGINGFACE_TOKEN`` Token used to download gated Hugging Face models
                      (e.g. pyannote speaker diarization).
``INPUT_DIR``         Directory holding source videos. Defaults to ``data/input``.
``WORK_DIR``          Scratch directory for intermediate artifacts.
``OUTPUT_DIR``        Directory for final dubbed videos and manifests.
``MODEL_CACHE_DIR``   Where model weights are downloaded at runtime.
``DIARIZATION_MODEL`` Hugging Face pipeline id used for speaker diarization.
``TRANSCRIPTION_MODEL``
                      faster-whisper model used for transcription.
``TRANSCRIPTION_COMPUTE_TYPE``
                      CTranslate2 compute type (e.g. ``float16`` on GPU).
``TRANSCRIPTION_LANGUAGE``
                      Source language code; unset means detect automatically.
``TRANSLATION_MODEL``
                      Qwen-2.5-Instruct checkpoint served by the local vLLM server.
``TRANSLATION_BASE_URL``
                      OpenAI-compatible base URL of the local vLLM server.
``TRANSLATION_API_KEY``
                      Optional bearer token for that server; local vLLM usually
                      ignores it, so it defaults to an empty string.
``TRANSLATION_WINDOW_LINES`` / ``TRANSLATION_WINDOW_OVERLAP``
                      Size of one chronological context window, and how many lines
                      consecutive windows share so a scene is never cut mid-exchange.
``TRANSLATION_MAX_CONCURRENCY``
                      Number of windows in flight at once - the client-side ceiling
                      that keeps a high-throughput batch from spiking VRAM.
``TRANSLATION_GPU_MEMORY_UTILIZATION``
                      Fraction of VRAM the vLLM server may pre-allocate; exported to
                      the server launcher, not sent with a request.
``TRANSLATION_TEMPERATURE`` / ``TRANSLATION_MAX_NEW_TOKENS`` / ``TRANSLATION_TIMEOUT_SECONDS``
                      Sampling and request bounds for one window.
``SCRIPT_PATH``       Flat ``.txt`` ground-truth screenplay used for character names
                      and dialogue alignment. Absent means labels stay ``SPEAKER_XX``.
``TTS_MAX_INPUT_CHARS``
                      Longest text sent to OmniVoice in one inference pass; longer
                      lines are split at punctuation and stitched back together.
``TTS_MAX_CONCURRENCY``
                      Number of synthesis requests in flight at once.
``TIMING_ACTIVE_ZONE_MARGIN_DB`` / ``TIMING_ACTIVE_ZONE_MIN_SECONDS``
                      How the active speech zone inside an original line is detected
                      so the dub is padded with silence rather than time-stretched.
``VOICE_PROFILE_DIR`` Directory holding per-speaker voice profiles.
``DIALOGUE_BIBLE_PATH``
                      Persistent character/consistency state (names, address
                      forms, register, recurring spellings) handed to dialogue
                      adaptation. Absent means an empty bible.
``VOICE_REFERENCE_MIN_DURATION`` / ``..._TARGET_DURATION`` / ``..._MAX_DURATION``
                      Preferred length of a voice-cloning reference, in seconds.
``TTS_MODEL``         Amharic speech adapter used by the TTS stage.``DIARIZATION_MIN_SPEAKERS`` / ``DIARIZATION_MAX_SPEAKERS``
                      Optional bounds on how many speakers diarization may find.
                      A cast list turns these into the strongest single guard
                      against a character splitting into several clusters.
``TTS_MAX_PAUSE_SECONDS``
                      Longest pause rendered around a synthesized line.
``TTS_MIN_LINE_SECONDS``
                      Shortest original window that can be dubbed; shorter lines
                      are skipped and reported rather than sent to the engine.
``TTS_CONTINUE_ON_FAILURE``
                      Skip and report a line an engine *fails* on instead of ending
                      the run. Off by default, because a failure should be visible.
``DEVICE``            Compute device hint, ``cuda`` by default.
``ALLOW_TORCH_COMPILE``
                      Set to ``1`` to re-enable TorchInductor. Compilation is off
                      by default because a pod whose PyTorch CUDA build predates
                      its GPU aborts at NVRTC compile time.
``LOG_LEVEL``         Logging verbosity, ``INFO`` by default.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Optional

try:  # ``python-dotenv`` is part of the base requirements.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - keeps the module importable without it
    load_dotenv = None  # type: ignore[assignment]


#: Repository root (the directory that contains ``app/``).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "input"
DEFAULT_WORK_DIR = PROJECT_ROOT / "data" / "working"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
DEFAULT_MODEL_CACHE_DIR = PROJECT_ROOT / "models_cache"

#: pyannote Community-1 speaker diarization pipeline on the Hugging Face Hub.
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

#: Optional bounds on the number of speakers diarization may report, or ``None``
#: to let the model decide. Supplying them is the strongest single guard against
#: a character splitting into several clusters or merging with another, so a cast
#: count - or a generous range around one - is worth configuring for a film.
DEFAULT_DIARIZATION_MIN_SPEAKERS: Optional[int] = None
DEFAULT_DIARIZATION_MAX_SPEAKERS: Optional[int] = None

#: Whether a cluster whose own turns hold two voices is split apart. On by default: a
#: character merged into a neighbour's cluster is cloned with that neighbour's voice for
#: the whole film, which is audible everywhere they speak. The pass is conservative and
#: reports every split it makes; turn it off to keep exactly the model's own clustering.
DEFAULT_DIARIZATION_SPLIT_MIXED_SPEAKERS = True

#: faster-whisper model used to transcribe the dialogue stem.
DEFAULT_TRANSCRIPTION_MODEL = "large-v3"

#: CTranslate2 compute type. ``float16`` targets the A40 GPU; CPU runs need
#: ``int8`` or ``float32``, because CTranslate2 does not support fp16 on CPU.
DEFAULT_TRANSCRIPTION_COMPUTE_TYPE = "float16"

#: Whether a run round-trips its own dialogue through an Amharic ASR model to measure
#: pronunciation. Off by default: it loads a second model and transcribes every clip.
#: It is the only automated check that can see content the speech model invented - a
#: stray word at the start of a line, a fragment of the English prompt the voice was
#: cloned from, a mispronunciation - because none of that is the film's original audio,
#: so no comparison with the original can find it.
DEFAULT_QC_PRONUNCIATION = False

#: The local dialogue-adaptation model: Qwen-2.5-Instruct, served by vLLM on the
#: GPU host. Every heavy operation - including this inference - runs on the rented
#: GPU, because the host CPU cannot carry a 7B-to-14B model at film scale.
#:
#: The checkpoint name is the one the server was launched with; the client never
#: downloads weights, so a mismatch is reported by vLLM rather than hidden here.
DEFAULT_TRANSLATION_MODEL = "Qwen/Qwen2.5-14B-Instruct"

#: OpenAI-compatible endpoint of the local vLLM server. Loopback by default: the
#: server and the pipeline run in the same GPU pod, and nothing about a film is
#: sent to a third party.
DEFAULT_TRANSLATION_BASE_URL = "http://127.0.0.1:8000/v1"

#: Environment variable holding the optional bearer token for that server. A local
#: vLLM deployment normally ignores the key, so an empty value is valid and no
#: hosted-vendor credential is needed anywhere in this pipeline.
TRANSLATION_API_KEY_ENV = "TRANSLATION_API_KEY"

#: Dialogue lines in one chronological context window. Large enough that the model
#: sees a scene as a conversation rather than a list of sentences, small enough that
#: one rejection is cheap and the JSON response stays inside a predictable budget.
DEFAULT_TRANSLATION_WINDOW_LINES = 24

#: Lines consecutive windows share. The overlap is what keeps an exchange from being
#: cut in half at a window boundary; the duplicate lines it produces are removed by
#: :func:`app.pipeline.translation.deduplicate_records` before anything downstream.
DEFAULT_TRANSLATION_WINDOW_OVERLAP = 5

#: How many windows may be in flight at once. This is the client-side half of the
#: VRAM throttle: the server pre-allocates its cache (``--gpu-memory-utilization``),
#: and this bounds how much request state can land on it at the same moment.
#:
#: Four rather than eight because the same GPU also runs the separator, diarizer,
#: transcriber and speech engine. Every concurrent window needs its own KV cache, and
#: on a 48 GiB card a 14B checkpoint at 0.72 utilization leaves room for about four
#: windows of a 24-line payload - asking for eight is what turns into
#: "No available memory for the cache blocks" at server start.
DEFAULT_TRANSLATION_MAX_CONCURRENCY = 4

#: Fraction of VRAM the vLLM server may pre-allocate. Handed to the server launcher
#: as ``--gpu-memory-utilization``; it is never sent with a chat request.
#:
#: 0.72, not 0.90, because this is a *shared* card. vLLM reserves its fraction for the
#: entire run, while the pipeline still has to load faster-whisper large-v3 (~3 GiB),
#: BS-Roformer (~3 GiB), pyannote (~2 GiB) and OmniVoice (~6 GiB) beside it. At 0.90 on
#: a 48 GiB A40 that left under 2 GiB free and transcription died with
#: ``CUDA failed with error out of memory`` before it could load a single weight. At
#: 0.72 the 14B checkpoint still fits with a usable KV cache, and the rest of the
#: pipeline has roughly 13 GiB to work in.
#:
#: Raise it only on a card that is dedicated to the model server.
DEFAULT_TRANSLATION_GPU_MEMORY_UTILIZATION = 0.72

#: Sampling temperature for adaptation. Zero keeps the same film producing the same
#: Amharic twice, which is what makes two runs comparable.
DEFAULT_TRANSLATION_TEMPERATURE = 0.0

#: Longest JSON response one window may generate, in tokens. A 24-line window needs
#: well under this; the bound exists so a runaway generation cannot hold a slot.
DEFAULT_TRANSLATION_MAX_NEW_TOKENS = 2048

#: Seconds one window request may take before it is failed rather than left pending.
DEFAULT_TRANSLATION_TIMEOUT_SECONDS = 300.0

#: Syllables per second an Amharic performer delivers, used to turn a span of time into
#: a syllable budget. Measured at about 4.6 syllables/second on real synthesized output,
#: and kept a little slower than that on purpose: asking a line to fit when it
#: comfortably can is harmless, while asking one to fit when it cannot leads to it being
#: cut. ``app.pipeline.dialogue_context.DEFAULT_SYLLABLES_PER_SECOND`` holds the same
#: prior for the adaptation prompt, and a test keeps the two in step.
DEFAULT_TRANSLATION_SYLLABLES_PER_SECOND = 4.0

#: The flat ``.txt`` screenplay the run is given as ground truth. It is optional: a run
#: without one keeps the diarization labels and lets the model infer characters.
DEFAULT_SCRIPT_PATH = DEFAULT_INPUT_DIR / "script.txt"

#: Directory holding the per-speaker voice profiles consumed by the TTS stage.
DEFAULT_VOICE_PROFILE_DIR = DEFAULT_WORK_DIR / "voices"

#: Persistent consistency state for dialogue adaptation: who the characters are,
#: how they address one another, and how their recurring names and terms are
#: spelled. Kept beside the voice profiles because both are per-film state that
#: outlives a single stage - unlike an artifact, it is meant to be edited by hand
#: between runs and reused, which is what stops a character drifting over 90-180
#: minutes. A missing file is an empty bible, not an error.
DEFAULT_DIALOGUE_BIBLE_FILENAME = "dialogue_bible.json"
DEFAULT_DIALOGUE_BIBLE_PATH = DEFAULT_WORK_DIR / DEFAULT_DIALOGUE_BIBLE_FILENAME

#: Preferred length of a voice-cloning reference, in seconds. A short, clean
#: prompt limits the amount of reference dialogue OmniVoice can echo; longer
#: references remain available when no suitable shorter interval exists.
DEFAULT_VOICE_REFERENCE_MIN_DURATION = 4.0
DEFAULT_VOICE_REFERENCE_TARGET_DURATION = 5.0
DEFAULT_VOICE_REFERENCE_MAX_DURATION = 10.0

#: Root of the manually provided per-character voice samples, one directory per
#: character: ``voice_samples/<Character_Name>/sample_01.wav`` with the matching
#: ``sample_01.txt`` transcript beside it. The directory name is matched to the
#: character names resolved from the screenplay, so it is the same spelling the
#: dialogue and the voice profiles use.
DEFAULT_VOICE_SAMPLE_DIR = PROJECT_ROOT / "voice_samples"

#: Legacy sample mapping, currently unused by the production pipeline.
DEFAULT_VOICE_SAMPLE_MAP = ""

#: Voice designs for characters the film gives no usable reference, e.g.
#: ``SPEAKER_02=female,young adult,high pitch``. Empty by default: every such character
#: is then given a distinct design from
#: :data:`app.pipeline.tts.DESIGNED_VOICE_POOL`, derived from its speaker id so a re-run
#: of the same film produces the same voices. Entries here override that for the
#: speakers they name.
DEFAULT_DESIGNED_VOICE_MAP = ""

#: Amharic speech engine consumed by :mod:`app.pipeline.tts`.
#:
#: ``"omnivoice"`` is k2-fsa's OmniVoice: a zero-shot model covering hundreds of
#: languages that clones a voice from a short reference and accepts a per-line rate.
#: Cloning from each character's own audio is what keeps one voice per character for a
#: whole film, and the rate is what lets a line be asked for at the length its window
#: allows instead of being stretched into it.
#: ``"mms"`` is Meta's MMS-TTS Amharic: a VITS model trained on Amharic alone. It is a
#: **single-speaker** model - one voice for the whole film, with no cloning and no
#: per-character identity - and it needs its text romanised first (see the module).
DEFAULT_TTS_ENGINE = "omnivoice"

#: The Amharic TTS checkpoint. Which model this id should name depends on
#: ``TTS_ENGINE``: an MMS-TTS checkpoint for ``mms``. The ``omnivoice`` engine names
#: its model through its own setting.
DEFAULT_TTS_MODEL = "facebook/mms-tts-amh"

#: The OmniVoice checkpoint used when ``TTS_ENGINE=omnivoice``.
#:
#: Amharic-tuned OmniVoice. The Lab-et page documents Amharic voice cloning; its
#: example loads the same weights through the canonical african-low-resource id.
#: Its published objective and human-evaluation metrics are still TBD, so quality
#: must be judged from generated output rather than assumed from the fine-tuning.
DEFAULT_OMNIVOICE_MODEL = "african-low-resource/omnivoice-amharic"

#: Diffusion steps for OmniVoice. The model's own default is 32; 16 is the documented
#: faster setting. Quality is traded for speed here, so the default is left alone.
DEFAULT_OMNIVOICE_STEPS = 32

#: Classifier-free guidance scale. The model's default is 2.0.
DEFAULT_OMNIVOICE_GUIDANCE_SCALE = 2.0

#: Precision for the OmniVoice weights. ``float16`` halves the footprint; the model
#: card documents it, so it is what a run uses.
DEFAULT_OMNIVOICE_TORCH_DTYPE = "float16"

#: Sample rate MMS-TTS produces, in Hz. VITS models are trained per language and this
#: one is trained at 16 kHz, so the pipeline resamples its output rather than
#: pretending it is at a rate the model never produced.
DEFAULT_MMS_SAMPLE_RATE = 16_000

#: Fixed seed for the MMS duration predictor. VITS samples its rhythm, so the same
#: line would otherwise come out a slightly different length every run - which would
#: make a dub unreproducible and a baseline meaningless.
DEFAULT_MMS_SEED = 0

#: Speaking rate asked of the MMS duration predictor. ``1.0`` is the model's natural
#: pace, and the value reaches the predictor through the forward call that reads it.
#: It sets one delivery speed for the whole film rather than fitting each line: a line
#: is still fitted to its own window by ``timing.py``.
DEFAULT_MMS_SPEAKING_RATE = 1.0

#: Longest pause rendered before and after a synthesized line, in seconds. The
#: dialogue model estimates the pauses of a scene and normally stays below 1.5
#: seconds; this bound keeps a wildly wrong estimate from becoming a hole in the
#: dub.
DEFAULT_TTS_MAX_PAUSE_SECONDS = 2.0

#: Shortest original window that can be dubbed, in seconds. Below this the original
#: is a fragment - a breath, a click, a sliver of a mis-diarized turn - rather than a
#: spoken line: there is no room for a word in the time it occupied, any Amharic
#: written for it is unintelligible, and the speech engine can fail on it outright.
#: A vocoder can be driven to an empty mel spectrogram by input that short, which
#: trips a convolution inside it, so such a line must be skipped and reported rather
#: than attempted.
DEFAULT_TTS_MIN_LINE_SECONDS = 0.30

#: Whether a line an engine *fails* on is recorded and skipped instead of ending the
#: run. Off by default, because hiding a failure is worse than stopping: a stage error
#: names the line, and a dub with a hole in it is only acceptable when it was asked
#: for. Set it on a long run where finishing matters more than a perfect first pass.
DEFAULT_TTS_CONTINUE_ON_FAILURE = False

#: Whether the TTS stage asks a cloning engine for a per-line length before the timing
#: stage falls back to stretching. Asking a model for a duration changes how the line is
#: *spoken*; time-stretching changes audio that has already been spoken, which is why
#: the first is worth doing whenever an engine can honour it. Exposed as a flag so a
#: comparison run can switch it off and attribute a difference to it.
DEFAULT_TTS_REQUEST_RATE = True

#: Longest text handed to OmniVoice in one inference pass.
#:
#: The engine has a hard per-call character ceiling, and a line that crosses it fails
#: outright. The threshold is kept below the documented limit so a punctuation-split
#: segment still has room, and the split itself happens at natural punctuation so the
#: seam is a pause a listener already expects.
DEFAULT_TTS_MAX_INPUT_CHARS = 220

#: How many synthesis requests may be in flight at once. OmniVoice runs on the GPU, so
#: this is a VRAM throttle as much as a throughput setting: the semaphore keeps a
#: film-scale batch from turning into an out-of-memory failure.
DEFAULT_TTS_MAX_CONCURRENCY = 4

#: How far a line's delivery may be time-stretched to fit the window of the
#: original line, as a tempo factor. A factor below 1 slows the line down, above
#: 1 speeds it up, and both are pitch-preserving. The band is deliberately narrow:
#: a line that needs more than this is reported as not fitting rather than being
#: mangled to fit. FFmpeg's ``atempo`` filter only accepts factors between 0.5 and
#: 2.0, which bounds any configuration here.
#:
#: The ceiling is 1.45 rather than 1.25 because it was measured, not chosen. Replaying
#: the failed run's own line timings through the placement policy below, with the
#: Amharic the adaptation stage produced for this project, the worst line landed after
#: the actor spoke by:
#:
#: * 1.25 - 9.24s (the old default, with strict separation)
#: * 1.35 - 6.13s
#: * 1.45 - 4.96s
#: * 1.55 - 3.98s
#:
#: Below 1.25 the lines have nowhere to go and the whole dub falls behind the picture;
#: past 1.45 the delivery starts to sound hurried for what is left to gain. This is
#: deliberately a *band*, not a target: a line that fits is left at its natural pace,
#: and only the lines that cannot fit are delivered faster. FFmpeg's ``atempo`` filter
#: only accepts factors between 0.5 and 2.0, which bounds any configuration here.
DEFAULT_TIMING_MIN_TEMPO = 0.80
DEFAULT_TIMING_MAX_TEMPO = 1.45

#: How far a line may run into the next one before the next one is moved instead, in
#: seconds.
#:
#: Two voices a fraction of a second apart is what a conversation already sounds like -
#: the listener hears a natural hand-over (and the window includes the next line's own
#: leading pause, so the *speech* overlap is shorter still). Two voices *seconds* apart
#: is unintelligible, and moving a line that far breaks lip-sync, so the overrun is paid
#: first in a small overlap and only then in position. Measured with the real timing
#: stage on the failed run's own numbers, the worst line's drift falls from 6.13s to
#: 4.13s when this goes from 0.30 to 0.45.
DEFAULT_TIMING_MAX_OVERLAP_SECONDS = 0.45

#: Silence kept between one dubbed line and the next, in seconds.
#:
#: A line may run past its own original window into the silence that follows it, which is
#: what keeps a language longer than English from being squashed, but it must stop this
#: far short of the next line. Below the guard, two voices are audible at once and the
#: dub stops being intelligible; the guard is small because the silence between film
#: lines is usually short and worth using.
DEFAULT_TIMING_MIN_LINE_GAP = 0.12

#: Whether a line that will not fit its available time is cut short, with a fade, so that
#: it never overlaps the next line.
#:
#: **Off by default, deliberately.** Cutting is the one thing that damages the
#: performance: the listener hears a word clipped off the end, and on a real run where
#: the text was too long it happened to 31 of 38 lines, which sounds broken rather than
#: fast. The pipeline asks for concise text in its first translation pass. When the text
#: fits, this never fires and the setting is irrelevant. When it does not, the run says
#: so instead: a line that
#: overruns is reported, and is heard as a brief overlap rather than as a missing word.
#: Set it true to trade a clipped word for a guaranteed gap between lines.
DEFAULT_TIMING_TRIM_TO_FIT = False

#: How far above the line's own noise floor a frame must rise to count as speech when
#: the *active vocal zone* of the original line is measured.
#:
#: The original line's timestamps are the diarizer's idea of a turn, not the actor's
#: first and last phoneme: a turn usually opens and closes with silence. Fitting the
#: Amharic to the turn would stretch it into that silence; fitting it to the zone that
#: actually contains speech, then padding the rest with silence, keeps the cadence.
DEFAULT_TIMING_ACTIVE_ZONE_MARGIN_DB = 12.0

#: A detected zone shorter than this is a detection failure - a line that is mostly a
#: breath or a click - and the whole original window is used instead of trusting it.
DEFAULT_TIMING_ACTIVE_ZONE_MIN_SECONDS = 0.08

#: Level of the dubbed dialogue in the final mix, in dB relative to the clips the
#: TTS stage produced. Never a boost by default: the dialogue was generated at a
#: consistent level, so raising it here would only invite clipping.
DEFAULT_MIX_DIALOGUE_GAIN_DB = 0.0

#: How far the music and effects are ducked while dialogue is playing, in dB. The
#: reduction is smoothed, so it follows the dialogue instead of gating it, and it
#: is applied to the bed only - never to the dialogue itself.
DEFAULT_MIX_DUCK_DB = 6.0


def load_env_file(path: Optional[Path] = None) -> None:
    """Load a ``.env`` file without overriding existing environment variables.

    Missing files and a missing ``python-dotenv`` installation are both ignored
    on purpose: the project has to start even when nothing is configured.
    """

    if load_dotenv is None:
        return

    env_path = Path(path) if path is not None else PROJECT_ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)


#: Whether :func:`apply_torch_compile_policy` installed ``TORCH_COMPILE_DISABLE``
#: itself, so a configuration that opts compilation back in can undo exactly that.
_compile_guard_installed = False


def apply_torch_compile_policy() -> None:
    """Keep TorchInductor off unless ``ALLOW_TORCH_COMPILE=1``.

    Inductor compiles kernels for the visible GPU's compute capability through
    NVRTC. On a pod whose PyTorch CUDA build predates the GPU, that compilation
    aborts with ``nvrtc: error: invalid value for --gpu-architecture (-arch)``
    even though eager CUDA inference itself works. No stage of this pipeline
    needs compilation, so it stays off; set ``ALLOW_TORCH_COMPILE=1`` to opt in.
    """

    global _compile_guard_installed

    if _read_env("ALLOW_TORCH_COMPILE", "0") == "1":
        if _compile_guard_installed:
            os.environ.pop("TORCH_COMPILE_DISABLE", None)
            _compile_guard_installed = False
        return

    if os.environ.get("TORCH_COMPILE_DISABLE") == "1":
        return

    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    _compile_guard_installed = True


def _read_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return a trimmed environment value, falling back to ``default``."""

    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _read_path(name: str, default: Path) -> Path:
    """Return an environment value as a :class:`~pathlib.Path`.

    Relative paths are resolved against the project root so that behaviour does
    not depend on the current working directory.
    """

    raw = _read_env(name)
    if raw is None:
        return default

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _read_int(name: str, default: int) -> int:
    """Return an environment value as a positive integer, or ``default``.

    A malformed value is a configuration mistake and fails loudly rather than
    being silently replaced by the default.
    """

    raw = _read_env(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc

    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def _read_nonnegative_int(name: str, default: int) -> int:
    """Return an environment value as a non-negative integer, or ``default``.

    Zero is a legitimate value for settings that use it to disable an operation,
    such as translation rewrites, and for ordinary random seeds.
    """

    raw = _read_env(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc

    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value}")
    return value


def _read_optional_int(name: str, default: Optional[int]) -> Optional[int]:
    """Return an environment value as an optional positive integer.

    Unlike :func:`_read_int`, an unset variable is a legitimate value here: the
    speaker-count bounds of diarization are optional, and "not configured" has to
    stay distinguishable from any number.
    """

    raw = _read_env(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc

    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


#: Values accepted (case-insensitively) for a boolean environment flag.
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def _read_bool(name: str, default: bool) -> bool:
    """Return an environment value as a boolean, or ``default``."""

    raw = _read_env(name)
    if raw is None:
        return default

    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {raw!r}")


def _read_float(name: str, default: float) -> float:
    """Return an environment value as a finite positive float, or ``default``."""

    raw = _read_env(name)
    if raw is None:
        return default

    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc

    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {raw!r}")
    if value <= 0.0:
        raise ValueError(f"{name} must be a positive number, got {value}")
    return value


def _read_signed_float(name: str, default: float) -> float:
    """Return an environment value as a finite float of either sign, or ``default``.

    Levels are naturally signed - a gain of ``0`` means "leave it alone" and a
    negative one attenuates - so they cannot use :func:`_read_float`.
    """

    raw = _read_env(name)
    if raw is None:
        return default

    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc

    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {raw!r}")
    return value


@dataclass(frozen=True)
class Settings:
    """Resolved, immutable runtime settings for the dubbing pipeline."""

    input_dir: Path
    work_dir: Path
    output_dir: Path
    model_cache_dir: Path
    huggingface_token: Optional[str] = None
    #: Optional bearer token for the local vLLM server. A loopback deployment normally
    #: ignores it, so an unset value is a working configuration, not a missing secret.
    translation_api_key: Optional[str] = None
    device: str = "cuda"
    log_level: str = "INFO"
    #: Hugging Face pipeline id used by :mod:`app.pipeline.diarization`.
    diarization_model: str = DEFAULT_DIARIZATION_MODEL
    #: Optional speaker-count bounds for diarization; ``None`` lets it decide.
    diarization_min_speakers: Optional[int] = DEFAULT_DIARIZATION_MIN_SPEAKERS
    diarization_max_speakers: Optional[int] = DEFAULT_DIARIZATION_MAX_SPEAKERS
    #: Whether a cluster holding two voices is split, so the second one is not cloned
    #: with the first one's reference.
    diarization_split_mixed_speakers: bool = DEFAULT_DIARIZATION_SPLIT_MIXED_SPEAKERS
    #: faster-whisper model and CTranslate2 compute type for transcription.
    transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL
    transcription_compute_type: str = DEFAULT_TRANSCRIPTION_COMPUTE_TYPE
    #: Round-trip every delivered clip through an Amharic ASR model and report the
    #: character error rate against the text that was synthesized. Off by default
    #: because it loads a second model and transcribes every clip.
    qc_pronunciation: bool = DEFAULT_QC_PRONUNCIATION
    #: Source language code for transcription; ``None`` detects it automatically.
    transcription_language: Optional[str] = None
    #: Dialogue adaptation settings for :mod:`app.pipeline.translation`, which talks
    #: to a local Qwen-2.5-Instruct served by vLLM on the GPU host.
    translation_model: str = DEFAULT_TRANSLATION_MODEL
    translation_base_url: str = DEFAULT_TRANSLATION_BASE_URL
    #: Lines per chronological window and how many of them adjacent windows share.
    translation_window_lines: int = DEFAULT_TRANSLATION_WINDOW_LINES
    translation_window_overlap: int = DEFAULT_TRANSLATION_WINDOW_OVERLAP
    #: Client-side concurrency ceiling, paired with the server's own VRAM reservation.
    translation_max_concurrency: int = DEFAULT_TRANSLATION_MAX_CONCURRENCY
    translation_gpu_memory_utilization: float = (
        DEFAULT_TRANSLATION_GPU_MEMORY_UTILIZATION
    )
    translation_temperature: float = DEFAULT_TRANSLATION_TEMPERATURE
    translation_max_new_tokens: int = DEFAULT_TRANSLATION_MAX_NEW_TOKENS
    translation_timeout_seconds: float = DEFAULT_TRANSLATION_TIMEOUT_SECONDS
    translation_syllables_per_second: float = DEFAULT_TRANSLATION_SYLLABLES_PER_SECOND
    #: Flat ``.txt`` screenplay used as ground truth for character names and for the
    #: complete sentences the transcribed fragments are aligned to.
    script_path: Path = DEFAULT_SCRIPT_PATH
    #: Voice-profile settings for :mod:`app.pipeline.voice_profiles`.
    voice_profile_dir: Path = DEFAULT_VOICE_PROFILE_DIR
    voice_reference_min_duration: float = DEFAULT_VOICE_REFERENCE_MIN_DURATION
    voice_reference_target_duration: float = DEFAULT_VOICE_REFERENCE_TARGET_DURATION
    voice_reference_max_duration: float = DEFAULT_VOICE_REFERENCE_MAX_DURATION
    #: Directory of native-speaker recordings used instead of the film's own audio, and
    #: an optional ``SPEAKER=file`` list saying which voice each speaker gets.
    voice_sample_dir: Path = DEFAULT_VOICE_SAMPLE_DIR
    voice_sample_map: str = DEFAULT_VOICE_SAMPLE_MAP
    #: Optional ``SPEAKER=design`` list giving a character the film cannot describe a
    #: designed voice explicitly, e.g. ``SPEAKER_02=female,young adult,high pitch``.
    designed_voice_map: str = DEFAULT_DESIGNED_VOICE_MAP
    #: Persistent character/consistency state handed to dialogue adaptation.
    dialogue_bible_path: Path = DEFAULT_DIALOGUE_BIBLE_PATH
    #: TTS settings for :mod:`app.pipeline.tts`: the Amharic speech adapter and the
    #: pause bounds of one synthesized line.
    tts_model: str = DEFAULT_TTS_MODEL
    tts_engine: str = DEFAULT_TTS_ENGINE
    omnivoice_model: str = DEFAULT_OMNIVOICE_MODEL
    omnivoice_steps: int = DEFAULT_OMNIVOICE_STEPS
    omnivoice_guidance_scale: float = DEFAULT_OMNIVOICE_GUIDANCE_SCALE
    omnivoice_torch_dtype: str = DEFAULT_OMNIVOICE_TORCH_DTYPE
    mms_sample_rate: int = DEFAULT_MMS_SAMPLE_RATE
    mms_seed: int = DEFAULT_MMS_SEED
    mms_speaking_rate: float = DEFAULT_MMS_SPEAKING_RATE
    tts_max_pause_seconds: float = DEFAULT_TTS_MAX_PAUSE_SECONDS
    #: Shortest original window that can be dubbed at all; shorter lines are skipped
    #: and reported instead of being sent to the engine.
    tts_min_line_seconds: float = DEFAULT_TTS_MIN_LINE_SECONDS
    #: Whether a line an engine fails on is skipped and reported, rather than ending
    #: the run. Off by default: a failure should be visible, not absorbed.
    tts_continue_on_failure: bool = DEFAULT_TTS_CONTINUE_ON_FAILURE
    tts_request_rate: bool = DEFAULT_TTS_REQUEST_RATE
    #: Longest text handed to OmniVoice in one inference pass. A longer line is split
    #: at natural punctuation and the synthesized pieces are stitched back together.
    tts_max_input_chars: int = DEFAULT_TTS_MAX_INPUT_CHARS
    #: How many synthesis requests may be in flight at once. Paired with the serverside
    #: VRAM reservation so a film-scale batch cannot spike the card.
    tts_max_concurrency: int = DEFAULT_TTS_MAX_CONCURRENCY
    #: Timing settings for :mod:`app.pipeline.timing`: the tempo band a line may
    #: be stretched within to fit the time it has, and the silence kept before
    #: the next line so two voices are never heard at once.
    timing_min_tempo: float = DEFAULT_TIMING_MIN_TEMPO
    timing_max_tempo: float = DEFAULT_TIMING_MAX_TEMPO
    timing_max_overlap_seconds: float = DEFAULT_TIMING_MAX_OVERLAP_SECONDS
    timing_min_line_gap: float = DEFAULT_TIMING_MIN_LINE_GAP
    timing_trim_to_fit: bool = DEFAULT_TIMING_TRIM_TO_FIT
    #: How far above a line's own noise floor a frame must rise to count as speech when
    #: the original line's active vocal zone is measured.
    timing_active_zone_margin_db: float = DEFAULT_TIMING_ACTIVE_ZONE_MARGIN_DB
    #: A detected zone shorter than this is treated as a detection failure and the whole
    #: original window is used instead.
    timing_active_zone_min_seconds: float = DEFAULT_TIMING_ACTIVE_ZONE_MIN_SECONDS
    #: Mix settings for :mod:`app.pipeline.mixing`: the dialogue level and how far
    #: the music and effects are ducked under it.
    mix_dialogue_gain_db: float = DEFAULT_MIX_DIALOGUE_GAIN_DB
    mix_duck_db: float = DEFAULT_MIX_DUCK_DB

    # -- construction -------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Settings":
        """Build a :class:`Settings` instance from the environment."""

        load_env_file()
        apply_torch_compile_policy()
        work_dir = _read_path("WORK_DIR", DEFAULT_WORK_DIR)
        model_cache_dir = _read_path("MODEL_CACHE_DIR", DEFAULT_MODEL_CACHE_DIR)
        return cls(
            input_dir=_read_path("INPUT_DIR", DEFAULT_INPUT_DIR),
            work_dir=work_dir,
            output_dir=_read_path("OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
            model_cache_dir=model_cache_dir,
            translation_api_key=_read_env(TRANSLATION_API_KEY_ENV),
            huggingface_token=_read_env("HUGGINGFACE_TOKEN"),
            device=(_read_env("DEVICE", "cuda") or "cuda").lower(),
            log_level=(_read_env("LOG_LEVEL", "INFO") or "INFO").upper(),
            diarization_model=(
                _read_env("DIARIZATION_MODEL", DEFAULT_DIARIZATION_MODEL)
                or DEFAULT_DIARIZATION_MODEL
            ),
            diarization_min_speakers=_read_optional_int(
                "DIARIZATION_MIN_SPEAKERS", DEFAULT_DIARIZATION_MIN_SPEAKERS
            ),
            diarization_max_speakers=_read_optional_int(
                "DIARIZATION_MAX_SPEAKERS", DEFAULT_DIARIZATION_MAX_SPEAKERS
            ),
            diarization_split_mixed_speakers=_read_bool(
                "DIARIZATION_SPLIT_MIXED_SPEAKERS",
                DEFAULT_DIARIZATION_SPLIT_MIXED_SPEAKERS,
            ),
            transcription_model=(
                _read_env("TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL)
                or DEFAULT_TRANSCRIPTION_MODEL
            ),
            transcription_compute_type=(
                _read_env("TRANSCRIPTION_COMPUTE_TYPE", DEFAULT_TRANSCRIPTION_COMPUTE_TYPE)
                or DEFAULT_TRANSCRIPTION_COMPUTE_TYPE
            ),
            qc_pronunciation=_read_bool(
                "QC_PRONUNCIATION", DEFAULT_QC_PRONUNCIATION
            ),
            transcription_language=(
                (_read_env("TRANSCRIPTION_LANGUAGE") or "").lower() or None
            ),
            translation_model=(
                _read_env("TRANSLATION_MODEL", DEFAULT_TRANSLATION_MODEL)
                or DEFAULT_TRANSLATION_MODEL
            ),
            translation_base_url=(
                _read_env("TRANSLATION_BASE_URL", DEFAULT_TRANSLATION_BASE_URL)
                or DEFAULT_TRANSLATION_BASE_URL
            ),
            translation_window_lines=_read_int(
                "TRANSLATION_WINDOW_LINES", DEFAULT_TRANSLATION_WINDOW_LINES
            ),
            translation_window_overlap=_read_nonnegative_int(
                "TRANSLATION_WINDOW_OVERLAP", DEFAULT_TRANSLATION_WINDOW_OVERLAP
            ),
            translation_max_concurrency=_read_int(
                "TRANSLATION_MAX_CONCURRENCY", DEFAULT_TRANSLATION_MAX_CONCURRENCY
            ),
            translation_gpu_memory_utilization=_read_float(
                "TRANSLATION_GPU_MEMORY_UTILIZATION",
                DEFAULT_TRANSLATION_GPU_MEMORY_UTILIZATION,
            ),
            translation_temperature=_read_float(
                "TRANSLATION_TEMPERATURE", DEFAULT_TRANSLATION_TEMPERATURE
            ),
            translation_max_new_tokens=_read_int(
                "TRANSLATION_MAX_NEW_TOKENS", DEFAULT_TRANSLATION_MAX_NEW_TOKENS
            ),
            translation_timeout_seconds=_read_float(
                "TRANSLATION_TIMEOUT_SECONDS", DEFAULT_TRANSLATION_TIMEOUT_SECONDS
            ),
            translation_syllables_per_second=_read_float(
                "TRANSLATION_SYLLABLES_PER_SECOND",
                DEFAULT_TRANSLATION_SYLLABLES_PER_SECOND,
            ),
            script_path=_read_path("SCRIPT_PATH", DEFAULT_SCRIPT_PATH),
            voice_profile_dir=_read_path("VOICE_PROFILE_DIR", work_dir / "voices"),
            voice_reference_min_duration=_read_float(
                "VOICE_REFERENCE_MIN_DURATION", DEFAULT_VOICE_REFERENCE_MIN_DURATION
            ),
            voice_reference_target_duration=_read_float(
                "VOICE_REFERENCE_TARGET_DURATION", DEFAULT_VOICE_REFERENCE_TARGET_DURATION
            ),
            voice_reference_max_duration=_read_float(
                "VOICE_REFERENCE_MAX_DURATION", DEFAULT_VOICE_REFERENCE_MAX_DURATION
            ),
            voice_sample_dir=_read_path("VOICE_SAMPLE_DIR", DEFAULT_VOICE_SAMPLE_DIR),
            voice_sample_map=(
                _read_env("VOICE_SAMPLE_MAP", DEFAULT_VOICE_SAMPLE_MAP)
                or DEFAULT_VOICE_SAMPLE_MAP
            ),
            designed_voice_map=(
                _read_env("DESIGNED_VOICE_MAP", DEFAULT_DESIGNED_VOICE_MAP)
                or DEFAULT_DESIGNED_VOICE_MAP
            ),
            dialogue_bible_path=_read_path(
                "DIALOGUE_BIBLE_PATH", work_dir / DEFAULT_DIALOGUE_BIBLE_FILENAME
            ),
            tts_model=(_read_env("TTS_MODEL", DEFAULT_TTS_MODEL) or DEFAULT_TTS_MODEL),
            tts_engine=(
                _read_env("TTS_ENGINE", DEFAULT_TTS_ENGINE) or DEFAULT_TTS_ENGINE
            ).lower(),
            omnivoice_model=(
                _read_env("OMNIVOICE_MODEL", DEFAULT_OMNIVOICE_MODEL)
                or DEFAULT_OMNIVOICE_MODEL
            ),
            omnivoice_steps=_read_int("OMNIVOICE_STEPS", DEFAULT_OMNIVOICE_STEPS),
            omnivoice_guidance_scale=_read_float(
                "OMNIVOICE_GUIDANCE_SCALE", DEFAULT_OMNIVOICE_GUIDANCE_SCALE
            ),
            omnivoice_torch_dtype=(
                _read_env("OMNIVOICE_TORCH_DTYPE", DEFAULT_OMNIVOICE_TORCH_DTYPE)
                or DEFAULT_OMNIVOICE_TORCH_DTYPE
            ),
            mms_sample_rate=_read_int("MMS_SAMPLE_RATE", DEFAULT_MMS_SAMPLE_RATE),
            mms_seed=_read_nonnegative_int("MMS_SEED", DEFAULT_MMS_SEED),
            mms_speaking_rate=_read_float(
                "MMS_SPEAKING_RATE", DEFAULT_MMS_SPEAKING_RATE
            ),
            tts_max_pause_seconds=_read_float(
                "TTS_MAX_PAUSE_SECONDS", DEFAULT_TTS_MAX_PAUSE_SECONDS
            ),
            tts_min_line_seconds=_read_float(
                "TTS_MIN_LINE_SECONDS", DEFAULT_TTS_MIN_LINE_SECONDS
            ),
            tts_continue_on_failure=_read_bool(
                "TTS_CONTINUE_ON_FAILURE", DEFAULT_TTS_CONTINUE_ON_FAILURE
            ),
            tts_request_rate=_read_bool(
                "TTS_REQUEST_RATE", DEFAULT_TTS_REQUEST_RATE
            ),
            tts_max_input_chars=_read_int(
                "TTS_MAX_INPUT_CHARS", DEFAULT_TTS_MAX_INPUT_CHARS
            ),
            tts_max_concurrency=_read_int(
                "TTS_MAX_CONCURRENCY", DEFAULT_TTS_MAX_CONCURRENCY
            ),
            timing_min_tempo=_read_float("TIMING_MIN_TEMPO", DEFAULT_TIMING_MIN_TEMPO),
            timing_max_tempo=_read_float("TIMING_MAX_TEMPO", DEFAULT_TIMING_MAX_TEMPO),
            timing_max_overlap_seconds=_read_float(
                "TIMING_MAX_OVERLAP_SECONDS", DEFAULT_TIMING_MAX_OVERLAP_SECONDS
            ),
            timing_min_line_gap=_read_float(
                "TIMING_MIN_LINE_GAP", DEFAULT_TIMING_MIN_LINE_GAP
            ),
            timing_trim_to_fit=_read_bool(
                "TIMING_TRIM_TO_FIT", DEFAULT_TIMING_TRIM_TO_FIT
            ),
            timing_active_zone_margin_db=_read_float(
                "TIMING_ACTIVE_ZONE_MARGIN_DB", DEFAULT_TIMING_ACTIVE_ZONE_MARGIN_DB
            ),
            timing_active_zone_min_seconds=_read_float(
                "TIMING_ACTIVE_ZONE_MIN_SECONDS",
                DEFAULT_TIMING_ACTIVE_ZONE_MIN_SECONDS,
            ),
            mix_dialogue_gain_db=_read_signed_float(
                "MIX_DIALOGUE_GAIN_DB", DEFAULT_MIX_DIALOGUE_GAIN_DB
            ),
            mix_duck_db=_read_signed_float("MIX_DUCK_DB", DEFAULT_MIX_DUCK_DB),
        )

    # -- helpers ------------------------------------------------------------
    @property
    def translation_api_key_env(self) -> str:
        """Return the environment variable holding the local server's token."""

        return TRANSLATION_API_KEY_ENV

    def vllm_server_arguments(self) -> list[str]:
        """Return the vLLM server arguments this configuration was sized for.

        The server is launched beside the pipeline on the GPU host. Handing it the
        same model id and the same VRAM reservation the client was configured with is
        what stops the two halves from disagreeing about how much headroom exists.
        """

        return [
            "--model",
            self.translation_model,
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--gpu-memory-utilization",
            f"{self.translation_gpu_memory_utilization:.2f}",
            "--max-num-seqs",
            str(max(1, self.translation_max_concurrency)),
            "--dtype",
            "auto",
        ]

    @property
    def has_huggingface_credentials(self) -> bool:
        """``True`` when a Hugging Face token is configured."""

        return bool(self.huggingface_token)

    def missing_credentials(self) -> list[str]:
        """Return the names of the credentials the configured run still needs.

        Only the Hugging Face token: the adaptation model is served from the same GPU
        host, so no hosted-vendor key is involved anywhere in the pipeline.
        """

        missing: list[str] = []
        if not self.huggingface_token:
            missing.append("HUGGINGFACE_TOKEN")
        return missing

    def ensure_directories(self) -> None:
        """Create the input/working/output/model-cache directories if needed."""

        for directory in (
            self.input_dir,
            self.work_dir,
            self.output_dir,
            self.model_cache_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, object]:
        """Return a redacted, JSON-safe view of the settings (never secrets)."""

        return {
            "input_dir": str(self.input_dir),
            "work_dir": str(self.work_dir),
            "output_dir": str(self.output_dir),
            "model_cache_dir": str(self.model_cache_dir),
            "translation_api_key_set": bool(self.translation_api_key),
            "huggingface_token_set": self.has_huggingface_credentials,
            "device": self.device,
            "log_level": self.log_level,
            "diarization_model": self.diarization_model,
            "diarization_min_speakers": self.diarization_min_speakers,
            "diarization_max_speakers": self.diarization_max_speakers,
            "diarization_split_mixed_speakers": self.diarization_split_mixed_speakers,
            "transcription_model": self.transcription_model,
            "transcription_compute_type": self.transcription_compute_type,
            "qc_pronunciation": self.qc_pronunciation,
            "transcription_language": self.transcription_language,
            "translation_backend": "vllm",
            "translation_provider": "local",
            "translation_api_key_env": self.translation_api_key_env,
            "translation_model": self.translation_model,
            "translation_base_url": self.translation_base_url,
            "translation_window_lines": self.translation_window_lines,
            "translation_window_overlap": self.translation_window_overlap,
            "translation_max_concurrency": self.translation_max_concurrency,
            "translation_gpu_memory_utilization": (
                self.translation_gpu_memory_utilization
            ),
            "translation_temperature": self.translation_temperature,
            "translation_max_new_tokens": self.translation_max_new_tokens,
            "translation_timeout_seconds": self.translation_timeout_seconds,
            "translation_syllables_per_second": self.translation_syllables_per_second,
            "script_path": str(self.script_path),
            "voice_profile_dir": str(self.voice_profile_dir),
            "voice_reference_min_duration": self.voice_reference_min_duration,
            "voice_reference_target_duration": self.voice_reference_target_duration,
            "voice_reference_max_duration": self.voice_reference_max_duration,
            "voice_sample_dir": str(self.voice_sample_dir),
            "voice_sample_map": self.voice_sample_map,
            "designed_voice_map": self.designed_voice_map,
            "dialogue_bible_path": str(self.dialogue_bible_path),
            "tts_model": self.tts_model,
            "tts_engine": self.tts_engine,
            "omnivoice_model": self.omnivoice_model,
            "omnivoice_steps": self.omnivoice_steps,
            "omnivoice_guidance_scale": self.omnivoice_guidance_scale,
            "omnivoice_torch_dtype": self.omnivoice_torch_dtype,
            "mms_sample_rate": self.mms_sample_rate,
            "mms_seed": self.mms_seed,
            "mms_speaking_rate": self.mms_speaking_rate,
            "tts_max_pause_seconds": self.tts_max_pause_seconds,
            "tts_min_line_seconds": self.tts_min_line_seconds,
            "tts_continue_on_failure": self.tts_continue_on_failure,
            "tts_request_rate": self.tts_request_rate,
            "tts_max_input_chars": self.tts_max_input_chars,
            "tts_max_concurrency": self.tts_max_concurrency,
            "timing_min_tempo": self.timing_min_tempo,
            "timing_max_tempo": self.timing_max_tempo,
            "timing_max_overlap_seconds": self.timing_max_overlap_seconds,
            "timing_min_line_gap": self.timing_min_line_gap,
            "timing_trim_to_fit": self.timing_trim_to_fit,
            "timing_active_zone_margin_db": self.timing_active_zone_margin_db,
            "timing_active_zone_min_seconds": self.timing_active_zone_min_seconds,
            "mix_dialogue_gain_db": self.mix_dialogue_gain_db,
            "mix_duck_db": self.mix_duck_db,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide, cached :class:`Settings` instance."""

    return Settings.from_env()


__all__ = [
    "DEFAULT_OMNIVOICE_GUIDANCE_SCALE",
    "DEFAULT_OMNIVOICE_MODEL",
    "DEFAULT_OMNIVOICE_STEPS",
    "DEFAULT_OMNIVOICE_TORCH_DTYPE",
    "DEFAULT_DIALOGUE_BIBLE_FILENAME",
    "DEFAULT_DIALOGUE_BIBLE_PATH",
    "DEFAULT_DIARIZATION_MAX_SPEAKERS",
    "DEFAULT_DIARIZATION_MIN_SPEAKERS",
    "DEFAULT_DIARIZATION_MODEL",
    "DEFAULT_DIARIZATION_SPLIT_MIXED_SPEAKERS",
    "DEFAULT_MIX_DIALOGUE_GAIN_DB",
    "DEFAULT_MIX_DUCK_DB",
    "DEFAULT_MMS_SAMPLE_RATE",
    "DEFAULT_MMS_SEED",
    "DEFAULT_MMS_SPEAKING_RATE",
    "DEFAULT_TRANSCRIPTION_COMPUTE_TYPE",
    "DEFAULT_QC_PRONUNCIATION",
    "DEFAULT_TRANSCRIPTION_MODEL",
    "DEFAULT_TRANSLATION_BASE_URL",
    "DEFAULT_TRANSLATION_GPU_MEMORY_UTILIZATION",
    "DEFAULT_TRANSLATION_MAX_CONCURRENCY",
    "DEFAULT_TRANSLATION_MAX_NEW_TOKENS",
    "DEFAULT_TRANSLATION_MODEL",
    "DEFAULT_TRANSLATION_SYLLABLES_PER_SECOND",
    "DEFAULT_TRANSLATION_TEMPERATURE",
    "DEFAULT_TRANSLATION_TIMEOUT_SECONDS",
    "DEFAULT_TRANSLATION_WINDOW_LINES",
    "DEFAULT_TRANSLATION_WINDOW_OVERLAP",
    "DEFAULT_SCRIPT_PATH",
    "TRANSLATION_API_KEY_ENV",
    "DEFAULT_TTS_CONTINUE_ON_FAILURE",
    "DEFAULT_TTS_MAX_PAUSE_SECONDS",
    "DEFAULT_TTS_MIN_LINE_SECONDS",
    "DEFAULT_TTS_REQUEST_RATE",
    "DEFAULT_TTS_MAX_INPUT_CHARS",
    "DEFAULT_TTS_MAX_CONCURRENCY",
    "DEFAULT_TTS_MODEL",
    "DEFAULT_TIMING_MAX_OVERLAP_SECONDS",
    "DEFAULT_TIMING_MAX_TEMPO",
    "DEFAULT_TIMING_MIN_LINE_GAP",
    "DEFAULT_TIMING_TRIM_TO_FIT",
    "DEFAULT_TIMING_ACTIVE_ZONE_MARGIN_DB",
    "DEFAULT_TIMING_ACTIVE_ZONE_MIN_SECONDS",
    "DEFAULT_TIMING_MIN_TEMPO",
    "DEFAULT_VOICE_PROFILE_DIR",
    "DEFAULT_VOICE_REFERENCE_MAX_DURATION",
    "DEFAULT_VOICE_SAMPLE_DIR",
    "DEFAULT_VOICE_SAMPLE_MAP",
    "DEFAULT_DESIGNED_VOICE_MAP",
    "DEFAULT_VOICE_REFERENCE_MIN_DURATION",
    "DEFAULT_VOICE_REFERENCE_TARGET_DURATION",
    "PROJECT_ROOT",
    "Settings",
    "apply_torch_compile_policy",
    "get_settings",
    "load_env_file",
]


# Applied at import so the guard is in place before any stage imports torch,
# which is what lets it cover a dependency that compiles during its own import.
apply_torch_compile_policy()
