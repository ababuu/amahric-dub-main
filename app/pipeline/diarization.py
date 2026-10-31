"""Speaker diarization ("who spoke when") with **pyannote Community-1**.

This module is the only place in the pipeline that talks to ``pyannote.audio``.
It consumes the dialogue stem written by :mod:`app.pipeline.separation` and
returns the speaker turns every later stage needs::

    speech.wav -> diarize() -> [SpeakerSegment(speaker_id, start, end), ...]

Three passes, each covering something the one before it cannot see:

1. **Global**, the model's own view. Community-1 diarizes the whole dialogue stem and
   returns an exclusive annotation (one speaker per instant) plus an overlap-aware one
   used only to flag crosstalk. The speaker count is *discovered*: no count is passed
   unless the caller configures one.
2. **Evidence**, this module's own embeddings. Every turn of at least half a second is
   embedded; turns heard together are aggregated into one voiceprint, so a character
   whose lines are all short still has evidence to be judged on. Attributions are
   verified against Community-1's speaker prototypes, and a cluster whose turns hold
   two voices is split - with every near-miss reported as a candidate.
3. **Suspect windows**, where the film-wide evidence was too thin to decide. Short
   windows around ambiguous or overlapping dialogue are diarized on their own, and may
   name a character the film-wide pass missed or move one of its turns to another
   identity. Only the turns a window was opened for can be changed by it.

The identities themselves come from the final partition, are rebuilt once, and are
persisted, so one character keeps one id and one cloned voice for the whole film. A
voice too close to two identities to separate is given its own id and reported as
uncertain rather than folded into either.

Transcript text, emotions, character names, and voice-cloning embeddings belong
to :mod:`app.pipeline.transcription`, :mod:`app.pipeline.voice_profiles`, and
:mod:`app.pipeline.tts`; they must not be added here.

Contract with the ``pyannote.audio`` package (checked against 4.0.7)
-------------------------------------------------------------------
* ``Pipeline.from_pretrained(model, token=..., cache_dir=...)`` loads the
  pipeline. There is **no** device argument; the call returns ``None`` when the
  checkpoint config cannot be fetched (unknown id, or a token without access).
* ``pipeline.to(torch.device(...))`` selects the compute device.
* ``pipeline(audio_path, num_speakers=..., min_speakers=..., max_speakers=...)``
  runs diarization and returns a ``DiarizeOutput``. Its
  ``exclusive_speaker_diarization`` annotation provides non-overlapping turns.
  Speaker embeddings are aligned to the labels in ``speaker_diarization`` and are
  used to verify clean, long turns. A bare ``Annotation`` (what ``legacy=True``
  pipelines return) is rejected rather than accepted as an overlap-free result.
* ``Annotation.itertracks(yield_label=True)`` yields ``(turn, track, label)``.

Deliberate non-goals
--------------------
* **No CUDA -> CPU fallback.** The device comes from the project settings, and an
  explicit ``cuda`` request that is unavailable must fail loudly.
* **No fallback to the non-exclusive diarization.** If the exclusive annotation
  is missing or malformed, this module raises :class:`UnsupportedOutputError`
  instead of returning potentially overlapping speaker turns.
* **No model downloads at import time** (and none in the test suite).
* **No pyannote objects in the public API.** Callers only ever see
  :class:`SpeakerSegment`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.pipeline import speaker_partition as partition
from app.pipeline import speaker_identity as identity

try:  # Heavy runtime dependencies. They are guarded so that this module stays
    # importable (and testable) where the CUDA/PyTorch stack is not installed.
    import torch
    from pyannote.audio import Pipeline
except ImportError:  # pragma: no cover - only without the AI runtime installed
    torch = None  # type: ignore[assignment]
    Pipeline = None  # type: ignore[assignment]


#: ``DiarizeOutput`` field holding the annotation that is authoritative for
#: *attribution*. pyannote Community-1 assigns every instant of speech to exactly
#: one speaker there, so every region maps to a single speaker and a line is never
#: attributed to two people at once.
_EXCLUSIVE_ANNOTATION_ATTRIBUTE = "exclusive_speaker_diarization"

#: ``DiarizeOutput`` field holding the overlap-aware annotation. It is read **only**
#: to flag where two speakers talk at once - information the exclusive view cannot
#: carry, because it has to drop one of the two. Never used to attribute a line.
#: Unlike the exclusive field this one is optional: a pyannote result without it
#: simply means "no crosstalk could be reported", not an error.
_OVERLAPPING_ANNOTATION_ATTRIBUTE = "speaker_diarization"

#: Shortest turn that can carry speaker evidence on its own. pyannote's own embedding
#: backends need about half a second, and one 0.6 s fragment is as likely to be a
#: clipped word as a voice, so shorter turns are never decided on individually.
_MIN_EMBEDDED_TURN_SECONDS = 0.5

#: A turn this long is evidence by itself. Anything shorter is only acted on when its
#: neighbours corroborate it (see :func:`_evidence_groups`): a character whose whole
#: performance is five short lines still has several seconds of speech between them.
_MIN_VERIFICATION_TURN_SECONDS = 1.5

#: Turns of one speaker this close together are heard as one performance, so their
#: embeddings are aggregated into one voiceprint instead of being judged one by one.
_EVIDENCE_GROUP_GAP_SECONDS = 1.6

#: Longest performance aggregated into one voiceprint. Beyond this the group is cut, so
#: a monologue cannot outvote the short lines that need the evidence.
_EVIDENCE_GROUP_MAX_SECONDS = 12.0

_MIN_REASSIGNMENT_SIMILARITY = 0.55
_MIN_REASSIGNMENT_MARGIN = 0.15
_LOGGER = logging.getLogger(__name__)

#: Shortest turn that joins the global re-partitioning as evidence. Half a second is
#: pyannote's own embedding floor (the crop is widened to a padded window first), so this
#: is "anything the embedder can hear", not a tuned value: a character whose whole part
#: is five short lines must still contribute five voiceprints.
_UNIT_MIN_SECONDS = 0.5

#: A turn shorter than this is embedded inside a wider window, so a 0.6 s line still
#: produces a usable voiceprint. The window never reaches into another speaker's turn.
_EMBED_MIN_WINDOW_SECONDS = 1.5

#: A cluster keeps a pyannote speaker's name only when that speaker provides this much
#: of its speech; otherwise it is a voice the model did not have and gets a new id.
_CLUSTER_DOMINANCE = 0.6

#: How far a turn without evidence of its own may sit from a moved performance and
#: still follow it. Beyond this it keeps the label pyannote gave it.
_UNMEASURED_INHERIT_SECONDS = 2.0

#: Most new characters one suspect window may introduce: a few seconds of audio that
#: yields a cast is a mistake, not a discovery.
_WINDOW_MAX_NEW_IDENTITIES = 1

#: A window needs at least this much speech in a cluster before it may name a character
#: a new character from it.
_WINDOW_MIN_CLUSTER_SECONDS = 1.0

#: Shortest suspect window worth a diarization call, and the longest one allowed: these
#: are windows around ambiguous dialogue, not scenes.
_WINDOW_MIN_SPAN_SECONDS = 2.0
_WINDOW_MAX_SPAN_SECONDS = 15.0

#: How much audio either side of an ambiguous performance a suspect window covers.
_WINDOW_PAD_SECONDS = 0.75

#: Silence between two suspect performances that still belongs to the same window.
_WINDOW_JOIN_GAP_SECONDS = 1.0

#: The window modes the pipeline understands.
_WINDOW_MODES = ("all", "suspect", "off")




class DiarizationError(RuntimeError):
    """Base class for every error raised by this module."""


class MissingInputError(DiarizationError):
    """The requested input audio file does not exist."""


class InvalidInputError(DiarizationError):
    """The input path exists but is not a regular file."""


class MissingHuggingFaceTokenError(DiarizationError):
    """No Hugging Face token is configured for the gated diarization model."""


class ModelInitializationError(DiarizationError):
    """The pyannote pipeline could not be loaded or moved to the device."""


class DiarizationInferenceError(DiarizationError):
    """pyannote failed while running diarization."""


class UnsupportedOutputError(DiarizationError):
    """pyannote returned an object this module cannot convert."""


class InvalidSegmentError(DiarizationError, ValueError):
    """A speaker segment has an empty speaker id or invalid timestamps."""


class InvalidSpeakerCountError(DiarizationError, ValueError):
    """A requested speaker-count hint is not a usable positive integer."""


def _seconds(name: str, value: object) -> float:
    """Return ``value`` as a finite ``float`` number of seconds."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidSegmentError(
            f"{name} must be a number of seconds, got {type(value).__name__}"
        )

    seconds = float(value)
    if not math.isfinite(seconds):
        raise InvalidSegmentError(f"{name} must be finite, got {value!r}")
    return seconds


@dataclass(frozen=True, slots=True)
class SpeakerSegment:
    """One diarized speaker turn, in seconds.

    Instances are immutable and always valid: an empty speaker id, a negative
    ``start``, or an ``end`` that does not come strictly after ``start`` is
    rejected at construction time, so later stages never have to defend
    themselves against a malformed turn.
    """

    speaker_id: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if not isinstance(self.speaker_id, str) or not self.speaker_id.strip():
            raise InvalidSegmentError("speaker_id must be a non-empty string")

        start = _seconds("start", self.start)
        end = _seconds("end", self.end)

        if start < 0:
            raise InvalidSegmentError(f"start must be >= 0 seconds, got {start}")
        if end <= start:
            raise InvalidSegmentError(
                f"end must be greater than start ({start} seconds), got {end}"
            )

        # Normalise ints to float so the declared type is guaranteed.
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def duration(self) -> float:
        """Length of the turn in seconds."""

        return self.end - self.start


@dataclass(frozen=True, slots=True)
class CrosstalkRegion:
    """An interval in which two or more speakers are talking at the same time.

    Reported for information only: it never changes which speaker a line is
    attributed to. It exists because the exclusive diarization this pipeline
    otherwise relies on has to pick one speaker per instant, so simultaneous
    speech would otherwise be invisible - and simultaneous speech is where a dub
    most often goes wrong.
    """

    speaker_ids: tuple[str, ...]
    start: float
    end: float

    def __post_init__(self) -> None:
        speakers = tuple(str(speaker).strip() for speaker in self.speaker_ids)
        if len(speakers) < 2 or any(not speaker for speaker in speakers):
            raise InvalidSegmentError(
                "a crosstalk region needs at least two non-empty speaker ids, got "
                f"{self.speaker_ids!r}"
            )

        start = _seconds("start", self.start)
        end = _seconds("end", self.end)
        if start < 0:
            raise InvalidSegmentError(f"start must be >= 0 seconds, got {start}")
        if end <= start:
            raise InvalidSegmentError(
                f"end must be greater than start ({start} seconds), got {end}"
            )

        # Sorted and de-duplicated so two regions covering the same speakers and
        # instant compare equal however pyannote happened to order its turns.
        object.__setattr__(self, "speaker_ids", tuple(sorted(set(speakers))))
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def duration(self) -> float:
        """Length of the overlap in seconds."""

        return self.end - self.start


@dataclass(frozen=True, slots=True)
class DiarizationResult:
    """Everything diarization found: attributed turns plus what it could not settle.

    ``turns`` is the final attribution: pyannote's exclusive view, verified against its
    own speaker prototypes, then re-partitioned by :mod:`app.pipeline.speaker_partition`
    so that a voice scattered over several clusters becomes one identity.
    ``raw_turns`` is pyannote's answer untouched, kept so the two can be compared.

    Nothing about a character's identity is decided silently: ``reclustering`` reports
    the partition chosen and every one refused, ``reassignments`` every performance that
    moved, ``windows`` what each suspect window found, ``uncertain_speakers`` the voices
    that could not be told apart, and ``merges`` the identities that were one person.
    """

    turns: tuple[SpeakerSegment, ...]
    crosstalk: tuple[CrosstalkRegion, ...] = ()
    corrections: tuple["SpeakerAttributionCorrection", ...] = ()
    #: pyannote's own exclusive attribution, exactly as it came back. Kept so the final
    #: partition can always be compared with the model's.
    raw_turns: tuple[SpeakerSegment, ...] = ()
    #: The film's own similarity scale, measured from its embeddings.
    calibration: partition.Calibration | None = None
    #: The global re-partitioning: the chosen number of speakers, the score it reached,
    #: and every number of speakers that was tried and refused.
    reclustering: partition.Partition | None = None
    #: Performances that carried evidence into the re-partitioning.
    units: int = 0
    unit_seconds: float = 0.0
    #: Every performance the re-partitioning moved to a different identity.
    reassignments: tuple["SpeakerReassignment", ...] = ()
    #: Suspect windows diarized on their own, and what each one produced.
    windows: tuple["DiarizationWindow", ...] = ()
    #: Voices that could not be tied to an identity with confidence.
    uncertain_speakers: tuple["UncertainSpeaker", ...] = ()
    #: Identity pairs that are suspiciously alike but were not merged.
    merge_suspects: tuple["SpeakerMergeSuspect", ...] = ()
    #: Identity pairs that were merged as one person discovered twice.
    merges: tuple["SpeakerIdentityMerge", ...] = ()
    #: Identities that only the re-partitioning found.
    discovered_speakers: tuple[str, ...] = ()
    #: Turns too short, too overlapped, or too noisy to carry speaker evidence.
    unmeasured_turns: int = 0
    #: Where the run persisted its identities, when it did.
    registry_path: str | None = None
    #: Why the registry could not be read or written, when that happened.
    registry_error: str = ""

    @property
    def speaker_ids(self) -> tuple[str, ...]:
        """Every speaker id that appears in ``turns``, sorted."""

        return tuple(sorted({segment.speaker_id for segment in self.turns}))

    @property
    def raw_speaker_ids(self) -> tuple[str, ...]:
        """Every speaker id pyannote itself produced, sorted."""

        return tuple(sorted({segment.speaker_id for segment in self.raw_turns}))

    @property
    def crosstalk_seconds(self) -> float:
        """Total time covered by reported crosstalk."""

        return sum(region.duration for region in self.crosstalk)



@dataclass(frozen=True, slots=True)
class SpeakerAttributionCorrection:
    """A diarized turn reassigned after a high-confidence embedding comparison."""

    start: float
    end: float
    original_speaker_id: str
    corrected_speaker_id: str
    corrected_similarity: float
    similarity_margin: float


@dataclass(frozen=True, slots=True)
class SpeakerReassignment:
    """One performance moved from pyannote's cluster to the identity the evidence supports.

    Reported rather than silent: this is the operation that fixes a character whose
    dialogue was scattered over two clusters, and it has to be auditable - which
    performance moved, from where to where, and how sure the embedding was.
    """

    start: float
    end: float
    seconds: float
    original_speaker_id: str
    corrected_speaker_id: str
    #: Similarity to the voiceprint of the cluster it was moved to.
    similarity: float
    #: How far that beat the cluster it came from.
    margin: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the reassignment."""

        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "seconds": round(self.seconds, 3),
            "from": self.original_speaker_id,
            "to": self.corrected_speaker_id,
            "similarity": round(self.similarity, 4),
            "margin": round(self.margin, 4),
        }


@dataclass(frozen=True, slots=True)
class UncertainSpeaker:
    """A voice that could not be tied to an identity with confidence.

    It was given an id of its own rather than being assigned to a character it merely
    resembled, because two characters sharing a cloned voice is worse than one
    character being cloned from a voice of uncertain name.
    """

    speaker_id: str
    reason: str
    similarity: float
    margin: float
    seconds: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the uncertainty."""

        return {
            "speaker_id": self.speaker_id,
            "reason": self.reason,
            "similarity": round(self.similarity, 4),
            "margin": round(self.margin, 4),
            "seconds": round(self.seconds, 3),
        }


@dataclass(frozen=True, slots=True)
class DiarizationWindow:
    """What one suspect window's own diarization produced, and what was done with it."""

    index: int
    start: float
    end: float
    turns: int
    speakers: tuple[str, ...]
    #: ``window`` when the window's own labels were kept, ``kept`` when the current
    #: attribution was kept, ``failed`` when the window could not be diarized at all.
    mode: str
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the window."""

        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "turns": self.turns,
            "speakers": list(self.speakers),
            "mode": self.mode,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class SpeakerMergeSuspect:
    """Two identities that are suspiciously alike but were left separate."""

    speaker_ids: tuple[str, str]
    similarity: float
    reason: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the suspect pair."""

        return {
            "speakers": list(self.speaker_ids),
            "similarity": round(self.similarity, 4),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SpeakerIdentityMerge:
    """Two identities that were merged as one person discovered twice."""

    from_speaker_id: str
    to_speaker_id: str
    similarity: float
    turns: int
    seconds: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the merge."""

        return {
            "from": self.from_speaker_id,
            "to": self.to_speaker_id,
            "similarity": round(self.similarity, 4),
            "turns": self.turns,
            "seconds": round(self.seconds, 3),
        }



def _validate_audio_path(path: str | Path) -> Path:
    """Return ``path`` as a :class:`~pathlib.Path` pointing at an existing file."""

    source = Path(path)
    if not source.exists():
        raise MissingInputError(f"input audio file not found: {source}")
    if not source.is_file():
        raise InvalidInputError(f"input audio path is not a file: {source}")
    return source


def _require_huggingface_token(settings: Settings) -> str:
    """Return the configured Hugging Face token, or explain what is missing."""

    token = (settings.huggingface_token or "").strip()
    if not token:
        raise MissingHuggingFaceTokenError(
            "HUGGINGFACE_TOKEN is not set; the pyannote Community-1 diarization "
            "pipeline is a gated model and requires a Hugging Face token that has "
            "accepted its conditions (see .env.example)"
        )
    return token


def _validate_speaker_counts(
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> None:
    """Reject unusable speaker-count hints before touching the model."""

    hints = (
        ("num_speakers", num_speakers),
        ("min_speakers", min_speakers),
        ("max_speakers", max_speakers),
    )
    for name, value in hints:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise InvalidSpeakerCountError(
                f"{name} must be a positive integer, got {value!r}"
            )

    if min_speakers is not None and max_speakers is not None:
        if min_speakers > max_speakers:
            raise InvalidSpeakerCountError(
                f"min_speakers ({min_speakers}) must not exceed "
                f"max_speakers ({max_speakers})"
            )


def _runtime_summary() -> str:
    """Describe the compute stack, so a device or compiler failure is diagnosable.

    Every lookup is defensive: this runs on an error path (and against the fakes
    in the test suite), where raising would hide the failure it is meant to explain.
    """

    if torch is None:  # pragma: no cover - only without the AI runtime
        return "torch is not installed"

    parts: list[str] = []
    try:
        parts.append(f"torch {torch.__version__}")
    except Exception:  # pragma: no cover - a stubbed runtime
        parts.append("torch (version unknown)")
    try:
        cuda_build = torch.version.cuda
        parts.append(f"built for CUDA {cuda_build or 'none (CPU build)'}")
    except Exception:  # pragma: no cover - a stubbed runtime
        pass
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            parts.append(f"{torch.cuda.get_device_name(0)} (sm_{major}{minor})")
    except Exception:  # pragma: no cover - diagnostics must never fail the stage
        pass
    try:
        import pyannote.audio as pyannote_audio

        parts.append(f"pyannote.audio {pyannote_audio.__version__}")
    except Exception:  # pragma: no cover - version metadata may be absent
        pass
    return ", ".join(parts)


#: NVRTC's own wording when it rejects the architecture Inductor asked for.
_ARCH_MISMATCH_MARKERS = ("nvrtc", "gpu-architecture")


def _is_compiler_arch_failure(exc: BaseException) -> bool:
    """``True`` when a CUDA kernel could not be compiled for this GPU's arch."""

    message = str(exc).lower()
    return any(marker in message for marker in _ARCH_MISMATCH_MARKERS)


def _load_pipeline(settings: Settings, token: str) -> Any:
    """Load the pyannote pipeline into memory on the configured device."""

    if Pipeline is None or torch is None:  # pragma: no cover - missing AI runtime
        raise ModelInitializationError(
            "pyannote.audio is not installed; install the runtime dependencies "
            "(torch, torchaudio, pyannote.audio) before running diarization"
        )

    try:
        pipeline = Pipeline.from_pretrained(
            settings.diarization_model,
            token=token,
            cache_dir=settings.model_cache_dir,
        )
    except Exception as exc:
        raise ModelInitializationError(
            f"could not load the pyannote pipeline "
            f"{settings.diarization_model!r}: {type(exc).__name__}: {exc}"
        ) from exc

    # ``from_pretrained`` returns None instead of raising when the checkpoint
    # config cannot be fetched at all.
    if pipeline is None:
        raise ModelInitializationError(
            f"pyannote could not load {settings.diarization_model!r}: the pipeline "
            "is unknown, or HUGGINGFACE_TOKEN has not been granted access to it"
        )

    try:
        pipeline.to(torch.device(settings.device))
    except Exception as exc:
        raise ModelInitializationError(
            f"could not move the pyannote pipeline to device "
            f"{settings.device!r}: {type(exc).__name__}: {exc}"
        ) from exc

    _LOGGER.info("pyannote runtime: %s", _runtime_summary())
    return pipeline


def _apply_pipeline(
    pipeline: Any,
    audio_path: Path,
    *,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> Any:
    """Run diarization, wrapping any pyannote failure with context."""

    kwargs: dict[str, int] = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    try:
        return pipeline(str(audio_path), **kwargs)
    except Exception as exc:
        hint = ""
        if _is_compiler_arch_failure(exc):
            hint = (
                ". A CUDA kernel could not be compiled for this GPU's compute "
                "capability (NVRTC). TorchInductor is disabled by default "
                "(TORCH_COMPILE_DISABLE=1); if ALLOW_TORCH_COMPILE=1 or "
                "TORCH_COMPILE_DISABLE=0 is set, remove it. If not, the pod's "
                "PyTorch CUDA build is older than its GPU, so use the RunPod "
                f"PyTorch image matched to it. Runtime: {_runtime_summary()}"
            )
        raise DiarizationInferenceError(
            f"pyannote diarization failed for {audio_path}: "
            f"{type(exc).__name__}: {exc}{hint}"
        ) from exc


def _extract_annotation(result: Any) -> Any:
    """Return the exclusive diarization ``Annotation`` carried by ``result``.

    pyannote 4.x returns a ``DiarizeOutput``; its ``exclusive_speaker_diarization``
    field contains no overlapping turns, so every region is attributed to exactly
    one speaker. There is no fallback to the overlapping ``speaker_diarization``
    field: a result without a usable exclusive annotation is an error to report,
    not something to paper over.
    """

    annotation = getattr(result, _EXCLUSIVE_ANNOTATION_ATTRIBUTE, None)
    if annotation is not None and hasattr(annotation, "itertracks"):
        return annotation

    raise UnsupportedOutputError(
        f"pyannote returned {type(result).__name__} without a usable "
        f"{_EXCLUSIVE_ANNOTATION_ATTRIBUTE!r}; pyannote Community-1 exclusive "
        "diarization is required so that every region is attributed to exactly "
        "one speaker, and this module does not fall back to the non-exclusive "
        "output"
    )


def _extract_overlap_annotation(result: Any) -> Any | None:
    """Return the overlap-aware ``Annotation`` carried by ``result``, or ``None``.

    Unlike the exclusive annotation this one is optional: it is only used to flag
    crosstalk, so a result that does not carry it is a result with nothing to
    report rather than a failure.
    """

    annotation = getattr(result, _OVERLAPPING_ANNOTATION_ATTRIBUTE, None)
    if annotation is not None and hasattr(annotation, "itertracks"):
        return annotation
    return None


def _to_crosstalk(annotation: Any) -> list[CrosstalkRegion]:
    """Return the intervals of ``annotation`` where two or more speakers overlap.

    The annotation is swept as a set of boundaries rather than compared pair by
    pair: at every boundary the set of simultaneously active speakers changes, and
    each stretch with two or more of them is an overlap. Adjacent stretches with
    the same speakers are merged, so a long crosstalk is one region instead of one
    region per turn boundary inside it.
    """

    boundaries: set[float] = set()
    spans: list[tuple[str, float, float]] = []
    for turn, _, label in annotation.itertracks(yield_label=True):
        speaker = label.strip() if isinstance(label, str) else ""
        if not speaker:
            continue
        spans.append((speaker, float(turn.start), float(turn.end)))
        boundaries.add(float(turn.start))
        boundaries.add(float(turn.end))

    regions: list[CrosstalkRegion] = []
    ordered = sorted(boundaries)
    for start, end in zip(ordered, ordered[1:]):
        if end <= start:
            continue
        active = tuple(
            sorted(
                {
                    speaker
                    for speaker, span_start, span_end in spans
                    if span_start <= start and span_end >= end
                }
            )
        )
        if len(active) < 2:
            continue

        if regions and regions[-1].speaker_ids == active and regions[-1].end == start:
            merged = regions.pop()
            regions.append(CrosstalkRegion(speaker_ids=active, start=merged.start, end=end))
            continue
        regions.append(CrosstalkRegion(speaker_ids=active, start=start, end=end))

    return regions


def _to_segments(result: Any) -> list[SpeakerSegment]:
    """Convert a pyannote result into chronologically sorted segments."""

    annotation = _extract_annotation(result)

    segments: list[SpeakerSegment] = []
    for turn, _, label in annotation.itertracks(yield_label=True):
        speaker_id = label.strip() if isinstance(label, str) else ""
        segments.append(
            SpeakerSegment(
                speaker_id=speaker_id,
                start=turn.start,
                end=turn.end,
            )
        )

    segments.sort(key=lambda segment: (segment.start, segment.end, segment.speaker_id))
    return segments


def _as_embedding(value: Any) -> tuple[float, ...] | None:
    """Convert a model embedding to a finite, non-zero Python vector."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        vector = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return None
    if not vector or not all(math.isfinite(component) for component in vector):
        return None
    if sum(component * component for component in vector) <= 0:
        return None
    return vector


def _speaker_centroids(result: Any) -> dict[str, tuple[float, ...]]:
    """Read Community-1's documented label-ordered speaker embeddings."""

    annotation = getattr(result, _OVERLAPPING_ANNOTATION_ATTRIBUTE, None)
    embeddings = getattr(result, "speaker_embeddings", None)
    if annotation is None or not hasattr(annotation, "labels") or embeddings is None:
        return {}

    labels = list(annotation.labels())
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    try:
        rows = list(embeddings)
    except TypeError:
        return {}
    if len(labels) != len(rows):
        return {}

    centroids: dict[str, tuple[float, ...]] = {}
    for label, row in zip(labels, rows):
        vector = _as_embedding(row)
        if isinstance(label, str) and label.strip() and vector is not None:
            centroids[label.strip()] = vector
    return centroids


def _is_crosstalk_turn(
    segment: SpeakerSegment, crosstalk: tuple[CrosstalkRegion, ...]
) -> bool:
    """Return whether any part of this turn overlaps another speaker."""

    return any(
        segment.start < region.end and segment.end > region.start
        for region in crosstalk
    )


@dataclass(frozen=True, slots=True)
class _EvidenceGroup:
    """Turns of one speaker heard together, and the voiceprint they carry.

    ``embeddable`` holds the turns that produced an embedding; ``vector`` is their
    duration-weighted mean. A group with one long turn is that turn; a group with three
    short ones is the evidence those three give together, which is the whole point -
    a character whose lines are all under a second still has a voiceprint.
    """

    speaker_id: str
    indexes: tuple[int, ...]
    embeddable: tuple[int, ...]
    start: float
    end: float
    speech_seconds: float
    vector: tuple[float, ...] | None


def _weighted_mean(
    vectors: Sequence[tuple[float, ...]], weights: Sequence[float]
) -> tuple[float, ...] | None:
    """Return the weighted mean of normalised ``vectors``, or ``None`` if impossible."""

    if not vectors:
        return None
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        return None
    total = sum(weights)
    if total <= 0:
        total = float(len(vectors))
        weights = [1.0] * len(vectors)
    blended = [0.0] * width
    for vector, weight in zip(vectors, weights):
        for position, value in enumerate(vector):
            blended[position] += value * weight / total
    try:
        return identity.normalise(blended)
    except identity.InvalidVoiceprintError:
        return None


def _padded_span(
    segments: Sequence[SpeakerSegment],
    index: int,
    *,
    minimum: float,
    total_duration: float | None = None,
) -> tuple[float, float]:
    """Return a window around one short turn that is wide enough to embed.

    The window grows into the silence before and after the turn, never into another
    speaker's turn: the neighbours' edges are the hard limits. A 0.6 s line inside a
    two-second pause therefore yields 1.5 s of that speaker and silence, which is a
    usable voiceprint, while a line butted against another speaker keeps its own edges
    and stays marked as short evidence.
    """

    segment = segments[index]
    if segment.duration >= minimum:
        return segment.start, segment.end

    need = minimum - segment.duration
    before = segments[index - 1].end if index > 0 else 0.0
    after = (
        segments[index + 1].start
        if index + 1 < len(segments)
        else (total_duration if total_duration is not None else segment.end + minimum)
    )
    left_room = max(0.0, segment.start - before)
    right_room = max(0.0, after - segment.end)
    left = min(left_room, need / 2.0)
    right = min(right_room, need - left)
    left = min(left_room, need - right)
    return segment.start - left, segment.end + right


def _evidence_vectors(
    pipeline: Any,
    audio_path: Path,
    segments: Sequence[SpeakerSegment],
    crosstalk: tuple[CrosstalkRegion, ...],
    *,
    minimum_seconds: float = _MIN_EMBEDDED_TURN_SECONDS,
    pad_short: bool = True,
) -> tuple[dict[int, tuple[float, ...]], int, int]:
    """Embed every turn that can carry speaker evidence.

    Returns the embeddings keyed by the segment's position in ``segments``, the number
    of turns that could not carry any, and how many were embedded inside a padded
    window. Short turns are padded rather than dropped, because a character built out of
    short lines is exactly the one this stage exists to find - but a turn that cannot be
    padded safely is left unmeasured rather than measured badly.
    """

    audio = getattr(pipeline, "_audio", None)
    embedder = getattr(pipeline, "_embedding", None)
    if audio is None or embedder is None or torch is None:
        return {}, len(segments), 0
    try:
        from pyannote.core import Segment as PyannoteSegment
    except ImportError:  # pragma: no cover - pyannote always ships this
        return {}, len(segments), 0

    vectors: dict[int, tuple[float, ...]] = {}
    unmeasured = 0
    padded = 0
    for index, segment in enumerate(segments):
        if segment.duration < minimum_seconds or _is_crosstalk_turn(segment, crosstalk):
            unmeasured += 1
            continue
        span = segment
        if pad_short and segment.duration < _EMBED_MIN_WINDOW_SECONDS:
            start, end = _padded_span(
                segments, index, minimum=_EMBED_MIN_WINDOW_SECONDS
            )
            if end > start:
                span = SpeakerSegment(segment.speaker_id, start, end)
                if span.duration > segment.duration:
                    padded += 1
        vector = _embed_turn(
            audio,
            embedder,
            audio_path,
            span,
            pyannote_segment=PyannoteSegment,
        )
        if vector is None:
            unmeasured += 1
            continue
        try:
            vectors[index] = identity.normalise(vector)
        except identity.InvalidVoiceprintError:
            unmeasured += 1
    return vectors, unmeasured, padded


def _evidence_groups(
    segments: Sequence[SpeakerSegment], vectors: Mapping[int, tuple[float, ...]]
) -> list[_EvidenceGroup]:
    """Group each speaker's turns into performances and average their voiceprints.

    A group is a run of turns by the same speaker separated by less than
    :data:`_EVIDENCE_GROUP_GAP_SECONDS` and no longer than
    :data:`_EVIDENCE_GROUP_MAX_SECONDS`. Grouping is what makes short dialogue usable:
    five 0.8 s lines with two-second gaps are five turns too short to judge alone, but
    one 4 s performance with a voiceprint to match against.
    """

    groups: list[_EvidenceGroup] = []
    current: list[int] = []

    def flush() -> None:
        if not current:
            return
        embeddable = tuple(index for index in current if index in vectors)
        speech = sum(segments[index].duration for index in embeddable)
        vector = _weighted_mean(
            [vectors[index] for index in embeddable],
            [segments[index].duration for index in embeddable],
        )
        groups.append(
            _EvidenceGroup(
                speaker_id=segments[current[0]].speaker_id,
                indexes=tuple(current),
                embeddable=embeddable,
                start=segments[current[0]].start,
                end=segments[current[-1]].end,
                speech_seconds=speech,
                vector=vector,
            )
        )
        current.clear()

    for index, segment in enumerate(segments):
        if not current:
            current.append(index)
            continue
        previous = segments[current[-1]]
        same_speaker = segment.speaker_id == previous.speaker_id
        close_enough = segment.start - previous.end <= _EVIDENCE_GROUP_GAP_SECONDS
        short_enough = (
            segment.end - segments[current[0]].start <= _EVIDENCE_GROUP_MAX_SECONDS
        )
        if same_speaker and close_enough and short_enough:
            current.append(index)
        else:
            flush()
            current.append(index)
    flush()
    return groups


def _evidence_units(
    turns: Sequence[SpeakerSegment],
    vectors: Mapping[int, tuple[float, ...]],
    *,
    minimum_seconds: float = _UNIT_MIN_SECONDS,
) -> tuple[list[partition.Unit], list["_EvidenceGroup"]]:
    """Return one evidence unit per turn that can carry a voiceprint.

    A unit is a single turn, not a run of them: pooling two turns that share a *label*
    would average two voices whenever the label is the thing that is wrong - the case
    this stage exists to fix. Short turns are still usable because they were embedded
    inside a padded window; their movement is what the clustering does with all the
    short performances of one voice together.
    """

    units: list[partition.Unit] = []
    groups: list[_EvidenceGroup] = []
    for index in sorted(vectors):
        segment = turns[index]
        if segment.duration < minimum_seconds:
            continue
        units.append(
            partition.Unit(
                provisional_id=segment.speaker_id,
                seconds=segment.duration,
                vector=vectors[index],
                tag=index,
            )
        )
        groups.append(
            _EvidenceGroup(
                speaker_id=segment.speaker_id,
                indexes=(index,),
                embeddable=(index,),
                start=segment.start,
                end=segment.end,
                speech_seconds=segment.duration,
                vector=vectors[index],
            )
        )
    return units, groups


def _verify_speaker_attributions(
    centroids: Mapping[str, tuple[float, ...]],
    segments: Sequence[SpeakerSegment],
    groups: Sequence[_EvidenceGroup],
) -> tuple[list[SpeakerSegment], tuple[SpeakerAttributionCorrection, ...]]:
    """Correct attributions only where the evidence is strong and corroborated.

    Community-1's speaker embeddings are the model-space prototypes its own clustering
    used, so comparing a performance's voiceprint against them needs no extra model and
    no assumption about how many speakers there are.

    A single short fragment is never enough: it can be a clipped word, a breath, or the
    tail of the previous speaker. A decision therefore needs either one turn of at least
    :data:`_MIN_VERIFICATION_TURN_SECONDS`, or several short ones that agree - which is
    exactly the aggregation the old per-turn check could not do, and the reason a
    character made only of short lines was never verified at all.
    """

    if len(centroids) < 2:
        return list(segments), ()

    corrections: list[SpeakerAttributionCorrection] = []
    verified = list(segments)
    for group in groups:
        if group.vector is None or group.speaker_id not in centroids:
            continue
        scores = {
            speaker_id: identity.cosine_similarity(group.vector, centroid)
            for speaker_id, centroid in centroids.items()
        }
        if len(scores) < 2:
            continue
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        best_speaker, best_score = ranked[0]
        current_score = scores[group.speaker_id]
        margin = best_score - current_score
        if best_speaker == group.speaker_id:
            continue
        if best_score < _MIN_REASSIGNMENT_SIMILARITY or margin < _MIN_REASSIGNMENT_MARGIN:
            continue
        corroborated = len(group.embeddable) >= 2
        long_enough = group.speech_seconds >= _MIN_VERIFICATION_TURN_SECONDS
        if not (corroborated or long_enough):
            # One short fragment against a confident prototype: reported by omission
            # (it stays where the clustering put it) rather than acted on.
            continue
        for index in group.embeddable:
            segment = verified[index]
            verified[index] = SpeakerSegment(best_speaker, segment.start, segment.end)
            corrections.append(
                SpeakerAttributionCorrection(
                    start=segment.start,
                    end=segment.end,
                    original_speaker_id=group.speaker_id,
                    corrected_speaker_id=best_speaker,
                    corrected_similarity=best_score,
                    similarity_margin=margin,
                )
            )
    return verified, tuple(corrections)



def _embed_turn(
    audio: Any,
    embedder: Any,
    audio_path: Path,
    segment: SpeakerSegment,
    *,
    pyannote_segment: Any,
) -> tuple[float, ...] | None:
    """Return one turn's speaker embedding, or ``None`` when it cannot be taken."""

    try:
        waveform, _ = audio.crop(
            {"audio": str(audio_path)},
            pyannote_segment(segment.start, segment.end),
            mode="pad",
        )
        with torch.inference_mode():
            raw_embedding = embedder(waveform.unsqueeze(0))
    except Exception as exc:  # pragma: no cover - model-specific failure
        _LOGGER.warning(
            "Could not embed %.3f-%.3fs of %s (%s: %s); leaving the cluster unsplit",
            segment.start,
            segment.end,
            segment.speaker_id,
            type(exc).__name__,
            exc,
        )
        return None

    rows = raw_embedding.tolist() if hasattr(raw_embedding, "tolist") else raw_embedding
    try:
        return _as_embedding(rows[0])
    except (IndexError, TypeError):
        return None


def _next_speaker_id(existing: set[str]) -> str:
    """Return an unused ``SPEAKER_NN`` label after the highest one in use."""

    highest = -1
    for label in existing:
        prefix, _, index = label.rpartition("_")
        if prefix == "SPEAKER" and index.isdigit():
            highest = max(highest, int(index))
    candidate = highest + 1
    while f"SPEAKER_{candidate:02d}" in existing:
        candidate += 1
    return f"SPEAKER_{candidate:02d}"


def _name_clusters(
    units: Sequence["_EvidenceGroup"],
    labels: Sequence[int],
    k: int,
    existing_ids: Collection[str],
) -> dict[int, str]:
    """Give each cluster the name of the speaker it mostly is, or a new one.

    A cluster that mostly holds one pyannote speaker keeps that speaker's id, which is
    what makes the result readable next to the original run. A cluster separated out of
    a speaker's turns by the audio - the scattered character - has no majority to
    inherit and gets an id of its own. Two clusters can never share an id.
    """

    totals: dict[int, float] = {cluster: 0.0 for cluster in range(k)}
    counts: dict[int, int] = {cluster: 0 for cluster in range(k)}
    weighted: dict[int, dict[str, float]] = {cluster: {} for cluster in range(k)}
    for unit, label in zip(units, labels):
        totals[label] += unit.speech_seconds
        counts[label] += 1
        weighted[label][unit.speaker_id] = (
            weighted[label].get(unit.speaker_id, 0.0) + unit.speech_seconds
        )

    def dominance(cluster: int) -> float:
        shares = weighted[cluster]
        if not shares or not totals[cluster]:
            return 0.0
        return max(shares.values()) / totals[cluster]

    # Purest clusters claim their speaker's name first: a cluster that is *entirely*
    # one speaker must not lose that name to a mixed cluster that merely leans towards
    # it, whichever speaks longer.
    ordered = sorted(range(k), key=lambda cluster: (-dominance(cluster), -totals[cluster], cluster))
    assignment: dict[int, str] = {}
    for cluster in ordered:
        shares = weighted[cluster]
        if not shares or not totals[cluster]:
            continue
        dominant, dominant_seconds = max(shares.items(), key=lambda item: item[1])
        if (
            dominant_seconds / totals[cluster] >= _CLUSTER_DOMINANCE
            and dominant not in assignment.values()
        ):
            assignment[cluster] = dominant
    provisional = set(existing_ids)
    for cluster in ordered:
        if cluster in assignment:
            continue
        new_id = _next_speaker_id(provisional | set(assignment.values()))
        assignment[cluster] = new_id

    return assignment
    return assignment, plans


def _cluster_scores(
    unit: "_EvidenceGroup", voiceprints: Mapping[int, tuple[float, ...]]
) -> list[tuple[int, float]]:
    """Rank the clusters by how well they explain one performance."""

    if unit.vector is None:
        return []
    scored = [
        (cluster, identity.cosine_similarity(unit.vector, voiceprint))
        for cluster, voiceprint in voiceprints.items()
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def _apply_partition(
    turns: Sequence[SpeakerSegment],
    units: Sequence["_EvidenceGroup"],
    labels: Sequence[int],
    assignment: Mapping[int, str],
    voiceprints: Mapping[int, tuple[float, ...]],
    calibration: partition.Calibration,
) -> tuple[
    list[SpeakerSegment],
    list[str],
    tuple[SpeakerReassignment, ...],
    tuple[UncertainSpeaker, ...],
]:
    """Move each performance to the identity the evidence supports, or leave it alone.

    A performance only moves when the cluster it was put in beats the next best by
    ``calibration.margin``. That is the rule that stops a 0.6 s fragment being handed to
    a character on the strength of one embedding: the ambiguity is reported and the turn
    keeps the identity pyannote gave it.
    """

    assignments: list[str] = []
    reassignments: list[SpeakerReassignment] = []
    uncertain_seconds: dict[str, float] = {}
    uncertain_worst: dict[str, tuple[float, float]] = {}
    updated = list(turns)

    for unit, label in zip(units, labels):
        target = assignment[label]
        scores = _cluster_scores(unit, voiceprints)
        best_score = scores[0][1] if scores else -1.0
        rival = max((score for cluster, score in scores if cluster != label), default=-1.0)
        margin = best_score - rival
        if target != unit.speaker_id and margin < calibration.margin:
            assignments.append(unit.speaker_id)
            uncertain_seconds[unit.speaker_id] = (
                uncertain_seconds.get(unit.speaker_id, 0.0) + unit.speech_seconds
            )
            previous = uncertain_worst.get(unit.speaker_id)
            if previous is None or margin < previous[1]:
                uncertain_worst[unit.speaker_id] = (best_score, margin)
            continue
        assignments.append(target)
        if target == unit.speaker_id:
            continue
        for index in unit.indexes:
            segment = updated[index]
            updated[index] = SpeakerSegment(target, segment.start, segment.end)
        reassignments.append(
            SpeakerReassignment(
                start=unit.start,
                end=unit.end,
                seconds=unit.speech_seconds,
                original_speaker_id=unit.speaker_id,
                corrected_speaker_id=target,
                similarity=best_score,
                margin=margin,
            )
        )

    # Turns without evidence of their own follow the nearest moved performance of the
    # same pyannote speaker, so a short fragment inside a moved performance is not left
    # behind; anything further away keeps the label the model gave it.
    measured = {index for unit in units for index in unit.indexes}
    movers = [
        (unit, unit_id)
        for unit, unit_id in zip(units, assignments)
        if unit_id != unit.speaker_id
    ]
    for index, segment in enumerate(updated):
        if index in measured:
            continue
        nearest: tuple[float, str] | None = None
        for unit, unit_id in movers:
            if unit.speaker_id != segment.speaker_id:
                continue
            distance = min(abs(segment.start - unit.start), abs(segment.end - unit.end))
            if distance <= _UNMEASURED_INHERIT_SECONDS and (
                nearest is None or distance < nearest[0]
            ):
                nearest = (distance, unit_id)
        if nearest is not None:
            updated[index] = SpeakerSegment(nearest[1], segment.start, segment.end)

    uncertain = tuple(
        UncertainSpeaker(
            speaker_id=speaker_id,
            reason=(
                "the evidence did not separate this voice from another one; the "
                "performance keeps pyannote's identity rather than being moved on a "
                "guess"
            ),
            similarity=uncertain_worst[speaker_id][0],
            margin=uncertain_worst[speaker_id][1],
            seconds=seconds,
        )
        for speaker_id, seconds in sorted(uncertain_seconds.items())
    )
    return updated, assignments, tuple(reassignments), uncertain


@dataclass(frozen=True, slots=True)
class _Window:
    """A short stretch of film around dialogue the evidence cannot settle."""

    index: int
    start: float
    end: float
    #: Every turn inside the window, used as context.
    turn_indexes: tuple[int, ...]
    #: The turns the window was opened for. Only these may be changed by it.
    suspect_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _LocalCluster:
    """One speaker as a window's own diarization heard them."""

    speaker_id: str
    indexes: tuple[int, ...]
    seconds: float
    vector: tuple[float, ...] | None


def _window_mode(value: str) -> str:
    """Return the validated window mode, or raise for a value that is not one."""

    mode = (value or "").strip().lower()
    if mode not in _WINDOW_MODES:
        raise InvalidInputError(
            f"DIARIZATION_SCENE_MODE must be one of {', '.join(_WINDOW_MODES)}, "
            f"got {value!r}"
        )
    return mode


def _suspect_windows(
    turns: Sequence[SpeakerSegment],
    units: Sequence["_EvidenceGroup"],
    labels: Sequence[int],
    voiceprints: Mapping[int, tuple[float, ...]],
    calibration: partition.Calibration,
    crosstalk: tuple[CrosstalkRegion, ...],
) -> list[_Window]:
    """Return short windows around the dialogue the evidence could not settle.

    Two things make a performance worth a second look: its two best clusters are too
    close to call, or it overlaps another speaker - the one case where the exclusive
    annotation necessarily drops a voice. The windows cover only those performances plus
    a little context, never a whole scene: diarizing a minute of film again asks the same
    question that already failed.
    """

    suspect_indexes: set[int] = set()
    for unit, label in zip(units, labels):
        scores = _cluster_scores(unit, voiceprints)
        if not scores:
            continue
        rival = max(
            (score for cluster, score in scores if cluster != label), default=-1.0
        )
        if scores[0][1] - rival < calibration.margin:
            suspect_indexes.update(unit.indexes)
    for index, segment in enumerate(turns):
        if _is_crosstalk_turn(segment, crosstalk):
            suspect_indexes.add(index)
    if not suspect_indexes:
        return []

    spans = sorted(
        (max(0.0, turns[index].start - _WINDOW_PAD_SECONDS),
         turns[index].end + _WINDOW_PAD_SECONDS)
        for index in suspect_indexes
    )
    merged: list[list[float]] = []
    for start, end in spans:
        if (
            merged
            and start - merged[-1][1] <= _WINDOW_JOIN_GAP_SECONDS
            and end - merged[-1][0] <= _WINDOW_MAX_SPAN_SECONDS
        ):
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    windows: list[_Window] = []
    for start, end in merged:
        if end - start < _WINDOW_MIN_SPAN_SECONDS:
            continue
        inside = tuple(
            index
            for index, segment in enumerate(turns)
            if segment.start < end and segment.end > start
        )
        if not inside:
            continue
        windows.append(
            _Window(
                index=len(windows),
                start=start,
                end=end,
                turn_indexes=inside,
                suspect_indexes=tuple(
                    index for index in inside if index in suspect_indexes
                ),
            )
        )
    return windows


def _window_argument(
    pipeline: Any, audio_path: Path, window: _Window
) -> tuple[Any, float, str] | None:
    """Return the argument that hands pyannote exactly this window, its time base, and
    which kind of argument it is.

    The waveform is preferred because the model's times are then relative to a slice
    this module cut, so the base is simply the window's own start. If the loaded
    pipeline exposes no audio reader, a file mapping with ``offset``/``duration`` is
    used and the model's times are checked before they are trusted.
    """

    try:
        from pyannote.core import Segment as PyannoteSegment
    except ImportError:  # pragma: no cover - pyannote always ships this
        return None

    audio = getattr(pipeline, "_audio", None)
    if audio is not None:
        try:
            waveform, sample_rate = audio.crop(
                {"audio": str(audio_path)},
                PyannoteSegment(window.start, window.end),
                mode="pad",
            )
        except Exception as exc:  # pragma: no cover - model-specific failure
            _LOGGER.warning(
                "Window %d (%.2f-%.2fs) could not be cut from the dialogue stem "
                "(%s: %s); falling back to a file slice",
                window.index,
                window.start,
                window.end,
                type(exc).__name__,
                exc,
            )
        else:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                window.start,
                "waveform",
            )

    return (
        {
            "audio": str(audio_path),
            "offset": window.start,
            "duration": window.end - window.start,
        },
        0.0,
        "offset",
    )


def _diarize_window(
    pipeline: Any, audio_path: Path, window: _Window
) -> list[SpeakerSegment] | None:
    """Diarize one suspect window on its own, or return ``None`` when that is unusable.

    No speaker-count hint is passed: the point of a window is to discover who is in it
    without being told. A window the model cannot handle is reported, never guessed at.
    """

    prepared = _window_argument(pipeline, audio_path, window)
    if prepared is None:
        return None
    argument, base, kind = prepared

    try:
        result = pipeline(argument)
    except Exception as exc:
        _LOGGER.warning(
            "Window %d (%.2f-%.2fs) could not be diarized on its own (%s: %s); "
            "keeping the current attribution for it",
            window.index,
            window.start,
            window.end,
            type(exc).__name__,
            exc,
        )
        return None

    try:
        local = _to_segments(result)
    except (UnsupportedOutputError, InvalidSegmentError) as exc:
        _LOGGER.warning(
            "Window %d (%.2f-%.2fs) returned a result this module cannot read (%s); "
            "keeping the current attribution for it",
            window.index,
            window.start,
            window.end,
            exc,
        )
        return None
    if not local:
        return None

    duration = window.end - window.start
    shift = base
    if kind == "offset" and max(segment.end for segment in local) <= duration + 1e-6:
        # A file slice given an offset should come back on the film's timeline, but if
        # the model returned window-relative times anyway, place them back here rather
        # than trusting the offset to have been honoured.
        shift = window.start
    if shift:
        local = [
            SpeakerSegment(
                segment.speaker_id, segment.start + shift, segment.end + shift
            )
            for segment in local
        ]

    if any(
        segment.end <= window.start - 0.5 or segment.start >= window.end + 0.5
        for segment in local
    ):
        _LOGGER.warning(
            "Window %d (%.2f-%.2fs) came back with turns outside the window; "
            "keeping the current attribution for it",
            window.index,
            window.start,
            window.end,
        )
        return None
    return local


def _vectors_by_time(
    segments: Sequence[SpeakerSegment],
    vectors: Mapping[int, tuple[float, ...]],
) -> dict[tuple[float, float], tuple[float, ...]]:
    """Re-key per-turn embeddings by the turn's interval.

    Scene-level diarization re-cuts a scene's turns, so positions into the global turn
    list stop meaning anything once a scene has been replaced. Times do not: a voiceprint
    belongs to the audio between two instants, whoever's list it came from.
    """

    timed: dict[tuple[float, float], tuple[float, ...]] = {}
    for index, segment in enumerate(segments):
        vector = vectors.get(index)
        if vector is not None:
            timed[(round(segment.start, 6), round(segment.end, 6))] = vector
    return timed


def _label_prototypes(
    segments: Sequence[SpeakerSegment],
    vectors: Mapping[tuple[float, float], tuple[float, ...]],
) -> dict[str, tuple[tuple[float, ...], float, int]]:
    """Return each speaker's voiceprint, the seconds behind it, and its turn count."""

    grouped: dict[str, list[tuple[SpeakerSegment, tuple[float, ...]]]] = {}
    for segment in segments:
        vector = vectors.get((round(segment.start, 6), round(segment.end, 6)))
        if vector is not None:
            grouped.setdefault(segment.speaker_id, []).append((segment, vector))
    prototypes: dict[str, tuple[tuple[float, ...], float, int]] = {}
    for speaker_id, entries in grouped.items():
        vector = _weighted_mean(
            [vector for _, vector in entries],
            [segment.duration for segment, _ in entries],
        )
        if vector is None:
            continue
        prototypes[speaker_id] = (
            vector,
            sum(segment.duration for segment, _ in entries),
            len(entries),
        )
    return prototypes


def _local_clusters(
    segments: Sequence[SpeakerSegment], vectors: Mapping[int, tuple[float, ...]]
) -> dict[str, _LocalCluster]:
    """Group one scene's turns by the labels its own diarization gave them."""

    grouped: dict[str, list[int]] = {}
    for index, segment in enumerate(segments):
        grouped.setdefault(segment.speaker_id, []).append(index)
    clusters: dict[str, _LocalCluster] = {}
    for speaker_id, indexes in grouped.items():
        embeddable = [index for index in indexes if index in vectors]
        clusters[speaker_id] = _LocalCluster(
            speaker_id=speaker_id,
            indexes=tuple(indexes),
            seconds=sum(segments[index].duration for index in embeddable),
            vector=_weighted_mean(
                [vectors[index] for index in embeddable],
                [segments[index].duration for index in embeddable],
            )
            if embeddable
            else None,
        )
    return clusters


def _reconcile_windows(
    pipeline: Any,
    audio_path: Path,
    turns: Sequence[SpeakerSegment],
    windows: Sequence[_Window],
    voiceprints: Mapping[str, tuple[float, ...]],
    calibration: partition.Calibration,
    *,
    short_turn: float,
) -> tuple[
    list[SpeakerSegment],
    tuple[DiarizationWindow, ...],
    tuple[UncertainSpeaker, ...],
    tuple[str, ...],
    dict[tuple[float, float], tuple[float, ...]],
]:
    """Diarize each suspect window on its own and let it correct the attribution.

    A window may change only the turns it was opened for - the ambiguous ones, and the
    ones under crosstalk - so a noisy few seconds cannot rewrite a scene it merely
    borders. Within that remit it can do both things the film-wide pass can do: name a
    character the current attribution lacks, and *move* dialogue from one existing
    identity to another when its own diarization separates them.

    A local cluster that matches no identity closely enough becomes a new one, reported
    as uncertain rather than folded into whichever identity it most resembles.
    """

    reports: list[DiarizationWindow] = []
    uncertain: list[UncertainSpeaker] = []
    replacements: dict[int, list[SpeakerSegment]] = {}
    replaced: set[int] = set()
    accepted_vectors: dict[tuple[float, float], tuple[float, ...]] = {}
    discovered: list[str] = []
    catalogue: dict[str, tuple[float, ...]] = dict(voiceprints)

    for window in windows:
        window_turns = [turns[index] for index in window.turn_indexes]
        current = {segment.speaker_id for segment in window_turns}
        local = _diarize_window(pipeline, audio_path, window)
        if local is None:
            reports.append(
                DiarizationWindow(
                    index=window.index,
                    start=window.start,
                    end=window.end,
                    turns=len(window_turns),
                    speakers=tuple(sorted(current)),
                    mode="failed",
                    detail="the window could not be diarized on its own",
                )
            )
            continue

        local_vectors, _, _ = _evidence_vectors(
            pipeline, audio_path, local, (), minimum_seconds=short_turn
        )
        if not local_vectors:
            reports.append(
                DiarizationWindow(
                    index=window.index,
                    start=window.start,
                    end=window.end,
                    turns=len(window_turns),
                    speakers=tuple(sorted(current)),
                    mode="kept",
                    detail="the window's turns were too short to embed",
                )
            )
            continue

        mapping: dict[str, str] = {}
        ambiguous_labels: set[str] = set()
        minted = 0
        for label, cluster in sorted(_local_clusters(local, local_vectors).items()):
            if cluster.vector is None or cluster.seconds < _WINDOW_MIN_CLUSTER_SECONDS:
                continue
            ranked = sorted(
                (
                    (speaker_id, identity.cosine_similarity(cluster.vector, voiceprint))
                    for speaker_id, voiceprint in catalogue.items()
                ),
                key=lambda item: (-item[1], item[0]),
            )
            if not ranked:
                continue
            best_id, best_score = ranked[0]
            rival = max((score for _, score in ranked[1:]), default=-1.0)
            margin = best_score - rival
            if best_score >= calibration.same_similarity and margin >= calibration.margin:
                mapping[label] = best_id
                continue
            if minted >= _WINDOW_MAX_NEW_IDENTITIES:
                ambiguous_labels.add(label)
                continue
            new_id = _next_speaker_id(set(catalogue) | set(mapping.values()))
            catalogue[new_id] = cluster.vector
            mapping[label] = new_id
            minted += 1
            discovered.append(new_id)
            uncertain.append(
                UncertainSpeaker(
                    speaker_id=new_id,
                    reason=(
                        "a suspect window separates this voice but no identity in the "
                        "film matches it closely enough; it keeps its own id rather "
                        "than being assigned to one it merely resembles"
                    ),
                    similarity=best_score,
                    margin=margin,
                    seconds=cluster.seconds,
                )
            )

        affected = [
            index
            for index in window.suspect_indexes
            if any(
                segment.start < turns[index].end and segment.end > turns[index].start
                for segment in local
            )
        ]
        if not affected:
            reports.append(
                DiarizationWindow(
                    index=window.index,
                    start=window.start,
                    end=window.end,
                    turns=len(window_turns),
                    speakers=tuple(sorted(current)),
                    mode="kept",
                    detail="the window's own diarization did not cover the suspect turns",
                )
            )
            continue

        span_start = min(turns[index].start for index in affected)
        span_end = max(turns[index].end for index in affected)
        replacement: list[SpeakerSegment] = []
        changes: set[tuple[int, str]] = set()
        undecided = False
        for segment in local:
            if segment.end <= span_start or segment.start >= span_end:
                continue
            start = max(segment.start, span_start)
            end = min(segment.end, span_end)
            if end <= start:
                continue
            mapped = mapping.get(segment.speaker_id)
            if mapped is None:
                if segment.speaker_id in ambiguous_labels:
                    undecided = True
                mapped = next(
                    (
                        turns[index].speaker_id
                        for index in affected
                        if turns[index].start < end and turns[index].end > start
                    ),
                    None,
                )
            if mapped is None:
                continue
            replacement.append(SpeakerSegment(mapped, start, end))
            for index in affected:
                if (
                    turns[index].start < end
                    and turns[index].end > start
                    and turns[index].speaker_id != mapped
                ):
                    changes.add((index, mapped))

        if not replacement or undecided or not changes:
            reports.append(
                DiarizationWindow(
                    index=window.index,
                    start=window.start,
                    end=window.end,
                    turns=len(window_turns),
                    speakers=tuple(sorted(current)),
                    mode="kept",
                    detail=(
                        "the window could not decide between identities for its "
                        "ambiguous turns"
                        if undecided
                        else "the window agreed with the current attribution"
                    ),
                )
            )
            continue

        replacements[window.index] = replacement
        replaced.update(affected)
        accepted_vectors.update(_vectors_by_time(local, local_vectors))
        reports.append(
            DiarizationWindow(
                index=window.index,
                start=window.start,
                end=window.end,
                turns=len(replacement),
                speakers=tuple(sorted({segment.speaker_id for segment in replacement})),
                mode="window",
                detail=(
                    f"the window moved {len(changes)} suspect turn(s) to the identity "
                    "its own diarization supports"
                ),
            )
        )

    if not replacements:
        return (
            list(turns),
            tuple(reports),
            tuple(uncertain),
            tuple(discovered),
            accepted_vectors,
        )

    merged: list[SpeakerSegment] = [
        segment for index, segment in enumerate(turns) if index not in replaced
    ]
    for window in windows:
        if window.index in replacements:
            merged.extend(replacements[window.index])
    merged.sort(key=lambda segment: (segment.start, segment.end, segment.speaker_id))
    return (
        merged,
        tuple(reports),
        tuple(uncertain),
        tuple(discovered),
        accepted_vectors,
    )


def _suspect_pairs(
    voiceprints: Mapping[str, tuple[float, ...]],
    seconds: Mapping[str, float],
    calibration: partition.Calibration,
) -> tuple[SpeakerMergeSuspect, ...]:
    """Report identities that are suspiciously alike but were deliberately kept apart.

    A pair this close was either merged already (above the film's same-voice threshold)
    or left alone because the evidence for one voice was not decisive. The second case
    is exactly what a listener hears as two characters sharing a voice, so it is
    reported even though nothing is done about it.
    """

    threshold = calibration.same_similarity - calibration.margin
    identifiers = sorted(voiceprints)
    suspects: list[SpeakerMergeSuspect] = []
    for position, first in enumerate(identifiers):
        for second in identifiers[position + 1 :]:
            similarity = identity.cosine_similarity(
                voiceprints[first], voiceprints[second]
            )
            if similarity < threshold:
                continue
            suspects.append(
                SpeakerMergeSuspect(
                    speaker_ids=(first, second),
                    similarity=similarity,
                    reason=(
                        "the two voiceprints are close but not close enough to merge; "
                        "one of them may be a fragment of the other's voice"
                    ),
                )
            )
    return tuple(suspects)


def _rebuild_registry(
    turns: Sequence[SpeakerSegment],
    vectors: Mapping[tuple[float, float], tuple[float, ...]],
    settings: Settings,
    *,
    registry_path: str | Path | None,
) -> tuple[identity.SpeakerRegistry, str | None, str]:
    """Rebuild every identity from the final partition and persist it once.

    One ``observe`` per identity, carrying the total evidence behind it: observing the
    same audio again from a window is what made the previous run report 256 s of speech
    for a 247 s film.
    """

    registry = identity.SpeakerRegistry()
    for speaker_id, (vector, seconds, count) in sorted(
        _label_prototypes(turns, vectors).items()
    ):
        registry.observe(speaker_id, vector, seconds, turns=count)
    written, error = _save_registry(registry, settings, registry_path=registry_path)
    return registry, written, error


def _resolve_speaker_counts(
    settings: Settings,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> tuple[int | None, int | None, int | None]:
    """Fill unset speaker-count hints from ``settings``.

    An explicit argument always wins, so a caller that knows better for one film
    is never overridden by the environment. Supplying the bounds at all is the
    single strongest guard against a character splitting into several clusters -
    which sounds, downstream, like one character being several people.
    """

    if min_speakers is None:
        min_speakers = settings.diarization_min_speakers
    if max_speakers is None:
        max_speakers = settings.diarization_max_speakers
    return num_speakers, min_speakers, max_speakers


def _effective_calibration(
    calibration: partition.Calibration, settings: Settings
) -> partition.Calibration:
    """Return the film's own calibration, or the configured fallback when it is unusable.

    A film whose provisional clusters do not separate its voices at all - a cast of one,
    or a soundtrack so noisy the embeddings agree with everything - cannot be its own
    reference. Then, and only then, the configured similarity and margin are used, so an
    operator still has a knob.
    """

    if calibration.reliable:
        return calibration
    return replace(
        calibration,
        same_similarity=settings.diarization_identity_min_similarity,
        margin=settings.diarization_identity_min_margin,
    )


def diarize_detailed(
    audio_path: str | Path,
    *,
    settings: Settings | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    registry_path: str | Path | None = None,
) -> DiarizationResult:
    """Diarize ``audio_path``, then re-partition the evidence into characters.

    Four steps, each seeing something the previous one cannot:

    1. **pyannote**, once, over the whole dialogue stem. Its exclusive annotation is the
       starting attribution and is kept untouched as ``raw_turns``. The speaker count is
       discovered: no count is passed unless the caller configured bounds.
    2. **Evidence**: every turn of at least half a second is embedded - inside a padded
       window when the turn itself is too short - and turns heard together are averaged
       into one performance, so a character made of short lines has a voiceprint.
    3. **Re-partition**: those performances are clustered *together*, not per pyannote
       cluster, because a voice can be spread over several of them. The number of
       speakers, and whether any number of speakers is believable at all, comes from the
       film's own similarity scale (:mod:`app.pipeline.speaker_partition`). Every
       candidate and every rejection is reported.
    4. **Suspect windows**: short windows around dialogue the re-partitioning could not
       settle (ambiguous, or overlapping another speaker) are diarized on their own, and
       may name a character the film-wide pass missed.

    Identities are rebuilt from the final partition and persisted once, so a character
    keeps one id and one voice for the whole film and across re-runs.
    """

    resolved = settings if settings is not None else get_settings()

    source = _validate_audio_path(audio_path)
    token = _require_huggingface_token(resolved)
    num_speakers, min_speakers, max_speakers = _resolve_speaker_counts(
        resolved, num_speakers, min_speakers, max_speakers
    )
    _validate_speaker_counts(num_speakers, min_speakers, max_speakers)
    mode = _window_mode(resolved.diarization_scene_mode)
    short_turn = resolved.diarization_short_turn_seconds

    pipeline = _load_pipeline(resolved, token)
    result = _apply_pipeline(
        pipeline,
        source,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    overlap_annotation = _extract_overlap_annotation(result)
    crosstalk = () if overlap_annotation is None else tuple(_to_crosstalk(overlap_annotation))
    raw_turns = _to_segments(result)

    centroids = _speaker_centroids(result)
    if len(raw_turns) >= 2:
        vectors, unmeasured, _padded = _evidence_vectors(
            pipeline, source, raw_turns, crosstalk, minimum_seconds=short_turn
        )
    else:
        vectors, unmeasured = {}, 0

    # Verification against pyannote's own prototypes is independent of the partition: it
    # fixes a turn the clustering put in the wrong cluster by comparing it with the
    # model-space centroid of the speaker it claims to be.
    turns, corrections = _verify_speaker_attributions(
        centroids, raw_turns, _evidence_groups(raw_turns, vectors)
    )
    units, unit_groups = _evidence_units(turns, vectors)
    calibration = _effective_calibration(partition.calibrate(units), resolved)
    provisional_ids = {unit.provisional_id for unit in units}

    if resolved.diarization_split_mixed_speakers:
        reclustering = partition.repartition(
            units,
            calibration,
            max_extra=resolved.diarization_max_extra_speakers,
        )
    else:
        # The switch keeps pyannote's own clustering, for a run that wants the model's
        # answer untouched. Calibration and the registry still describe what it found.
        order: dict[str, int] = {}
        provisional_labels = tuple(
            order.setdefault(unit.provisional_id, len(order)) for unit in units
        )
        reclustering = partition.Partition(
            labels=provisional_labels,
            k=len(order),
            silhouette=0.0,
            stability=1.0,
            accepted=False,
            reason="global re-partitioning is switched off",
            candidates=(),
        )

    labels = list(reclustering.labels)
    voiceprints, cluster_seconds, _counts = partition.cluster_voiceprints(
        units, labels, reclustering.k
    )
    assignment = _name_clusters(unit_groups, labels, reclustering.k, provisional_ids)

    merges_found = partition.merge_indistinguishable(
        voiceprints,
        cluster_seconds,
        # Two identities are merged only when they are almost certainly one person: the
        # midpoint of the film's own scale is a *matching* threshold, not a merging one.
        threshold=resolved.diarization_identity_merge_similarity,
    )
    # The merge report has to describe the identities *before* they were folded
    # together, so capture them before the renumbering below overwrites everything.
    pre_names = dict(assignment)
    pre_seconds = dict(cluster_seconds)
    pre_counts = dict(_counts)
    aliases = {drop: keep for keep, drop, _ in merges_found}

    def resolve(cluster: int) -> int:
        seen = 0
        while cluster in aliases and seen < len(aliases) + 1:
            cluster = aliases[cluster]
            seen += 1
        return cluster

    surviving = sorted({resolve(cluster) for cluster in labels})
    renumbered = {cluster: position for position, cluster in enumerate(surviving)}
    labels = [renumbered[resolve(cluster)] for cluster in labels]
    assignment = {
        renumbered[cluster]: assignment[cluster] for cluster in surviving
    }
    k = len(surviving)
    voiceprints, cluster_seconds, cluster_units = partition.cluster_voiceprints(
        units, labels, k
    )
    merges = tuple(
        SpeakerIdentityMerge(
            from_speaker_id=pre_names[drop],
            to_speaker_id=pre_names[keep],
            similarity=float(similarity),
            turns=pre_counts.get(drop, 0),
            seconds=pre_seconds.get(drop, 0.0),
        )
        for keep, drop, similarity in merges_found
    )
    discovered = [
        assignment[cluster]
        for cluster in sorted(assignment)
        if assignment[cluster] not in provisional_ids
    ]

    turns, _unit_assignments, reassignments, uncertain_partition = _apply_partition(
        turns,
        unit_groups,
        labels,
        assignment,
        voiceprints,
        calibration,
    )

    identity_voiceprints: dict[str, tuple[float, ...]] = {}
    for cluster, voiceprint in voiceprints.items():
        identity_voiceprints[assignment[cluster]] = voiceprint

    windows: tuple[DiarizationWindow, ...] = ()
    uncertain_windows: tuple[UncertainSpeaker, ...] = ()
    discovered_windows: tuple[str, ...] = ()
    window_vectors: dict[tuple[float, float], tuple[float, ...]] = {}
    if mode != "off" and units and resolved.diarization_split_mixed_speakers:
        suspect = _suspect_windows(
            turns, unit_groups, labels, voiceprints, calibration, crosstalk
        )
        if suspect:
            (
                turns,
                windows,
                uncertain_windows,
                discovered_windows,
                window_vectors,
            ) = _reconcile_windows(
                pipeline,
                source,
                turns,
                suspect,
                identity_voiceprints,
                calibration,
                short_turn=short_turn,
            )

    timed_vectors = _vectors_by_time(turns, vectors)
    timed_vectors.update(window_vectors)
    prototypes = _label_prototypes(turns, timed_vectors)
    _registry, registry_written, registry_error = _rebuild_registry(
        turns, timed_vectors, resolved, registry_path=registry_path
    )
    suspects = _suspect_pairs(
        {speaker_id: entry[0] for speaker_id, entry in prototypes.items()},
        {speaker_id: entry[1] for speaker_id, entry in prototypes.items()},
        calibration,
    )

    for merge in merges:
        _LOGGER.info(
            "Merged %s into %s (%.2f alike): one person found twice",
            merge.from_speaker_id,
            merge.to_speaker_id,
            merge.similarity,
        )
    for reassignment in reassignments:
        _LOGGER.info(
            "Moved %.2f-%.2fs from %s to %s (similarity %.2f, margin %.2f)",
            reassignment.start,
            reassignment.end,
            reassignment.original_speaker_id,
            reassignment.corrected_speaker_id,
            reassignment.similarity,
            reassignment.margin,
        )

    return DiarizationResult(
        turns=tuple(turns),
        crosstalk=crosstalk,
        corrections=corrections,
        raw_turns=tuple(raw_turns),
        calibration=calibration,
        reclustering=reclustering,
        units=len(units),
        unit_seconds=sum(unit.seconds for unit in units),
        reassignments=reassignments,
        windows=windows,
        uncertain_speakers=uncertain_partition + uncertain_windows,
        merge_suspects=suspects,
        merges=merges,
        discovered_speakers=tuple(dict.fromkeys(discovered + list(discovered_windows))),
        unmeasured_turns=unmeasured,
        registry_path=registry_written,
        registry_error=registry_error,
    )


def _registry_target(
    settings: Settings, registry_path: str | Path | None
) -> Path | None:
    """Return where this run should persist its identities, or ``None`` to skip it."""

    if not settings.diarization_registry_enabled:
        return None
    if registry_path is not None:
        return Path(registry_path)
    if settings.diarization_registry_path is not None:
        return Path(settings.diarization_registry_path)
    return Path(settings.work_dir) / identity.DEFAULT_REGISTRY_FILENAME


def _save_registry(
    registry: identity.SpeakerRegistry,
    settings: Settings,
    *,
    registry_path: str | Path | None,
) -> tuple[str | None, str]:
    """Persist the run's identities, returning where they went and any error."""

    target = _registry_target(settings, registry_path)
    if target is None:
        return None, ""
    try:
        written = registry.save(target)
    except (OSError, identity.IdentityError) as exc:
        _LOGGER.warning("Speaker registry %s could not be written: %s", target, exc)
        return None, str(exc)
    return str(written), ""



def diarize(
    audio_path: str | Path,
    *,
    settings: Settings | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[SpeakerSegment]:
    """Diarize ``audio_path`` into chronologically sorted speaker turns.

    Parameters
    ----------
    audio_path:
        Dialogue audio (typically the ``*_speech.wav`` stem written by
        :func:`app.pipeline.separation.separate_stems`).
    settings:
        Project settings override; defaults to :func:`app.config.get_settings`.
        The model id, device, model cache, Hugging Face token and the fallback
        speaker-count bounds all come from here - nothing is hard-coded in this
        module.
    num_speakers, min_speakers, max_speakers:
        Optional hints for pyannote's clustering. Leave unset to fall back to the
        configured bounds, or to let pyannote estimate the count when those are
        unset too. ``num_speakers`` wins over the bounds.

    Returns
    -------
    list[SpeakerSegment]
        One immutable segment per speaker turn, sorted by ``start``, after
        high-confidence speaker-embedding corrections. An empty list means
        pyannote found no speech. Call :func:`diarize_detailed` instead to also
        retrieve crosstalk and correction details.

    Raises
    ------
    MissingInputError, InvalidInputError
        The input is not an existing regular file.
    MissingHuggingFaceTokenError
        ``HUGGINGFACE_TOKEN`` is not configured.
    ModelInitializationError
        The pyannote pipeline could not be loaded or moved to the device.
    DiarizationInferenceError
        pyannote failed while running diarization.
    UnsupportedOutputError
        pyannote returned an object this module cannot convert.
    """

    return list(
        diarize_detailed(
            audio_path,
            settings=settings,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        ).turns
    )


__all__ = [
    "CrosstalkRegion",
    "DiarizationError",
    "DiarizationInferenceError",
    "DiarizationResult",
    "InvalidInputError",
    "InvalidSegmentError",
    "InvalidSpeakerCountError",
    "MissingHuggingFaceTokenError",
    "MissingInputError",
    "ModelInitializationError",
    "DiarizationWindow",
    "SpeakerAttributionCorrection",
    "SpeakerIdentityMerge",
    "SpeakerMergeSuspect",
    "SpeakerReassignment",
    "SpeakerSegment",
    "UncertainSpeaker",
    "UnsupportedOutputError",
    "diarize",
    "diarize_detailed",
]
