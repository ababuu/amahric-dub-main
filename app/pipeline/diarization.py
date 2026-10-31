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
3. **Scenes**, where short dialogue is actually rescued. The film is cut into scenes at
   its silences, each scene is diarized on its own, and the voices found in it are named
   against a :class:`~app.pipeline.speaker_identity.SpeakerRegistry`. A voice that
   matches an identity reuses its id and updates its voiceprint, so one character keeps
   one id and one cloned voice across the film; a voice that matches nothing keeps its
   own id; a voice too close to two identities to separate is given its own id and
   reported as uncertain rather than folded into either.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
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

# Splitting a cluster that holds two voices. Clustering separates speakers on one
# film well enough for the count to be right most of the time, but a character who
# speaks little, quietly, or mostly under another's turn ends up in a neighbour's
# cluster - and is then cloned in that neighbour's voice. The pass below looks for
# exactly that: a cluster whose own turns fall into two groups that are each
# consistent and clearly unlike the other.
#
# Thresholds reviewed against pyannote's speaker embeddings, where a voice agrees with
# itself well above 0.6 and with a different voice below 0.5: group at 0.55, require
# each group to sit at least 0.55 from its own centroid, require the groups to be no
# more than 0.50 alike, and require a 0.10 separation between the two. The evidence a
# turn can carry is no longer the gate - every turn of at least half a second counts,
# which is what lets a character who only ever says one short line be recovered - and
# every rejection is reported as a candidate instead of being passed over, because a
# merge that was *nearly* caught is the one worth seeing in a manifest.
_SPLIT_CLUSTER_SIMILARITY = 0.55
_SPLIT_MAX_ACROSS_SIMILARITY = 0.50
_SPLIT_MIN_WITHIN_SIMILARITY = 0.55
_SPLIT_MIN_SEPARATION = 0.10
_SPLIT_MIN_TURNS = 3
_SPLIT_MIN_SECONDS = 4.0
_SPLIT_MIN_CLUSTER_TURNS = 5

#: A scene needs at least this much speech in a cluster before a new identity may be
#: minted from it. A shorter cluster keeps the label the global pass gave it: inventing
#: a character from a clipped word is worse than leaving one character in the wrong
#: voice, and the run says which clusters it left alone.
_SCENE_MIN_CLUSTER_SECONDS = 1.0

#: Shortest scene worth a diarization call of its own.
_SCENE_MIN_SPAN_SECONDS = 3.0

#: The scene modes the pipeline understands.
_SCENE_MODES = ("all", "suspect", "off")



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

    ``turns`` is the exclusive, non-overlapping attribution with high-confidence
    speaker-embedding corrections applied. ``crosstalk`` is additive information
    about where speakers overlapped, and is never used to attribute a line.
    ``corrections`` records every changed turn and the scores that justified it.

    The remaining fields exist so that nothing about a character's identity is decided
    silently: ``splits`` and ``merges`` say which identities changed, ``scenes`` what
    each scene's own diarization found, ``split_candidates`` the clusters that looked
    like two voices and were left alone anyway, ``uncertain_speakers`` the voices that
    could not be told apart from another one, and ``merge_suspects`` the pairs that are
    suspiciously alike without being merged.
    """

    turns: tuple[SpeakerSegment, ...]
    crosstalk: tuple[CrosstalkRegion, ...] = ()
    corrections: tuple["SpeakerAttributionCorrection", ...] = ()
    #: Clusters that were found to hold two voices and were split apart.
    splits: tuple["SpeakerClusterSplit", ...] = ()
    #: Clusters that looked like two voices and were *not* split, with the reason.
    split_candidates: tuple["SpeakerSplitCandidate", ...] = ()
    #: What each scene's own diarization produced, and what was kept from the global
    #: pass instead.
    scenes: tuple["SceneDiarization", ...] = ()
    #: Voices that could not be tied to an identity with confidence. They were given an
    #: id of their own rather than being folded into somebody else's.
    uncertain_speakers: tuple["UncertainSpeaker", ...] = ()
    #: Identity pairs that are suspiciously alike but were not merged.
    merge_suspects: tuple["SpeakerMergeSuspect", ...] = ()
    #: Identity pairs that were merged as one person discovered twice.
    merges: tuple["SpeakerIdentityMerge", ...] = ()
    #: Identities that only scene-level diarization found.
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
class SpeakerClusterSplit:
    """One cluster that held two voices, and the one that was split off it.

    Reported rather than silent: a run that invents a character has to be able to say
    it did, which speaker it came from, and how sure the embeddings were.
    """

    speaker_id: str
    new_speaker_id: str
    turns: int
    seconds: float
    within_similarity: float
    across_similarity: float


@dataclass(frozen=True, slots=True)
class SpeakerSplitCandidate:
    """A cluster whose turns looked like two voices, and why it was left alone.

    The split pass is deliberately conservative, so the interesting case is not the
    split it made but the one it declined: ``reason`` says whether the groups were too
    alike, too small, or too inconsistent to act on, and the similarities are reported
    so the decision can be argued with without re-running the film.
    """

    speaker_id: str
    turns: int
    seconds: float
    within_similarity: float
    across_similarity: float
    reason: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the candidate."""

        return {
            "speaker_id": self.speaker_id,
            "turns": self.turns,
            "seconds": round(self.seconds, 3),
            "within_similarity": round(self.within_similarity, 4),
            "across_similarity": round(self.across_similarity, 4),
            "reason": self.reason,
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
class SceneDiarization:
    """What one scene's own diarization produced, and what was done with it."""

    index: int
    start: float
    end: float
    turns: int
    speakers: tuple[str, ...]
    #: ``scene`` when the scene's own labels were kept, ``global`` when the global pass
    #: was kept, ``failed`` when the scene could not be diarized at all.
    mode: str
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the scene."""

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


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Return cosine similarity, or -1 for incompatible embedding dimensions."""

    return identity.cosine_similarity(left, right)


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


def _evidence_vectors(
    pipeline: Any,
    audio_path: Path,
    segments: Sequence[SpeakerSegment],
    crosstalk: tuple[CrosstalkRegion, ...],
    *,
    minimum_seconds: float = _MIN_EMBEDDED_TURN_SECONDS,
) -> tuple[dict[int, tuple[float, ...]], int]:
    """Embed every turn that can carry speaker evidence.

    Returns the embeddings keyed by the segment's position in ``segments``, and the
    number of turns that could not carry any - too short to embed, or overlapping
    another speaker, or a turn the embedder refused. Those are *counted*, never
    guessed at: a decision is only ever made from real evidence.
    """

    audio = getattr(pipeline, "_audio", None)
    embedder = getattr(pipeline, "_embedding", None)
    if audio is None or embedder is None or torch is None:
        return {}, len(segments)
    try:
        from pyannote.core import Segment as PyannoteSegment
    except ImportError:  # pragma: no cover - pyannote always ships this
        return {}, len(segments)

    vectors: dict[int, tuple[float, ...]] = {}
    unmeasured = 0
    for index, segment in enumerate(segments):
        if segment.duration < minimum_seconds or _is_crosstalk_turn(segment, crosstalk):
            unmeasured += 1
            continue
        vector = _embed_turn(
            audio,
            embedder,
            audio_path,
            segment,
            pyannote_segment=PyannoteSegment,
        )
        if vector is None:
            unmeasured += 1
            continue
        try:
            vectors[index] = identity.normalise(vector)
        except identity.InvalidVoiceprintError:
            unmeasured += 1
    return vectors, unmeasured


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


def _agglomerate(
    vectors: list[tuple[float, ...]], *, threshold: float
) -> list[list[int]]:
    """Group ``vectors`` by average-linkage similarity, stopping below ``threshold``.

    Deterministic: the most similar pair is merged first, and a tie is broken by the
    lowest indices, so the same film always yields the same groups.
    """

    groups: list[list[int]] = [[index] for index in range(len(vectors))]

    while len(groups) > 1:
        best_similarity = -1.0
        best_pair: tuple[int, int] | None = None
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                total = 0.0
                for first in groups[left]:
                    for second in groups[right]:
                        total += _cosine_similarity(vectors[first], vectors[second])
                similarity = total / (len(groups[left]) * len(groups[right]))
                if similarity > best_similarity + 1e-12:
                    best_similarity = similarity
                    best_pair = (left, right)
        if best_pair is None or best_similarity < threshold:
            break
        left, right = best_pair
        groups[left] = groups[left] + groups[right]
        del groups[right]

    return groups


def _cross_similarity(
    vectors: list[tuple[float, ...]], left: list[int], right: list[int]
) -> float:
    """Return the mean similarity between the two groups."""

    total = 0.0
    for first in left:
        for second in right:
            total += _cosine_similarity(vectors[first], vectors[second])
    return total / (len(left) * len(right))


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


def _mean_similarity_to_centroid(
    vectors: Sequence[tuple[float, ...]], indices: Sequence[int]
) -> float:
    """Return how tightly ``indices`` agree with their own centroid.

    Cohesion, not the worst pair: one clipped turn would otherwise decide that a group
    of six clean turns is inconsistent.
    """

    if len(indices) < 2:
        return 1.0
    centroid = _weighted_mean(
        [vectors[index] for index in indices], [1.0] * len(indices)
    )
    if centroid is None:
        return -1.0
    return sum(
        identity.cosine_similarity(vectors[index], centroid) for index in indices
    ) / len(indices)


def _split_mixed_clusters(
    segments: Sequence[SpeakerSegment],
    vectors: Mapping[int, tuple[float, ...]],
) -> tuple[
    list[SpeakerSegment],
    tuple[SpeakerClusterSplit, ...],
    tuple[SpeakerSplitCandidate, ...],
]:
    """Split a speaker cluster whose own turns hold two voices.

    Community-1 clusters the film's speech once. A character who speaks little, quietly
    or mostly under someone else's turn can land inside that neighbour's cluster, and is
    then cloned with the neighbour's voice for the whole film. This looks for that case
    with the pipeline's own embedding model: a cluster whose turns fall into two groups
    that are each internally consistent and clearly unlike each other, with enough of
    the smaller voice to be worth a character of its own.

    Every turn of at least half a second contributes, so a character made only of short
    lines can be recovered; the evidence is aggregated into groups rather than judged
    turn by turn. A mis-split invents a character, so the tests stay conservative - and
    a cluster that *nearly* qualified is returned as a
    :class:`SpeakerSplitCandidate` with the numbers that stopped it, instead of being
    passed over in silence.
    """

    if not segments:
        return list(segments), (), ()

    by_speaker: dict[str, list[int]] = {}
    for index, segment in enumerate(segments):
        if index in vectors:
            by_speaker.setdefault(segment.speaker_id, []).append(index)

    splittable = {
        speaker_id: indices
        for speaker_id, indices in by_speaker.items()
        if len(indices) >= _SPLIT_MIN_CLUSTER_TURNS
    }
    if not splittable:
        return list(segments), (), ()

    splits: list[SpeakerClusterSplit] = []
    candidates: list[SpeakerSplitCandidate] = []
    updated = list(segments)
    taken = {segment.speaker_id for segment in segments}

    for speaker_id in sorted(splittable):
        indices = splittable[speaker_id]
        cluster_vectors = [vectors[index] for index in indices]
        groups = _agglomerate(cluster_vectors, threshold=_SPLIT_CLUSTER_SIMILARITY)
        if len(groups) < 2:
            # One voice: the silence here is the answer, and a "candidate" for every
            # correctly clustered speaker would drown the real ones.
            continue

        groups.sort(
            key=lambda group: -sum(
                segments[indices[position]].duration for position in group
            )
        )
        bigger, smaller = groups[0], groups[1]
        smaller_seconds = sum(
            segments[indices[position]].duration for position in smaller
        )
        total_seconds = sum(segments[index].duration for index in indices)
        within = min(
            _mean_similarity_to_centroid(cluster_vectors, bigger),
            _mean_similarity_to_centroid(cluster_vectors, smaller),
        )
        across = _cross_similarity(cluster_vectors, bigger, smaller)

        reason = ""
        if len(groups) > 2:
            reason = (
                f"the cluster's turns fall into {len(groups)} groups, not two; "
                "left as one speaker rather than guessing which are characters"
            )
        elif len(smaller) < _SPLIT_MIN_TURNS:
            reason = (
                f"the smaller voice has {len(smaller)} turn(s); "
                f"{_SPLIT_MIN_TURNS} are needed to stand alone"
            )
        elif smaller_seconds < _SPLIT_MIN_SECONDS:
            reason = (
                f"the smaller voice has {smaller_seconds:.2f}s of speech; "
                f"{_SPLIT_MIN_SECONDS:.1f}s are needed to stand alone"
            )
        elif within < _SPLIT_MIN_WITHIN_SIMILARITY:
            reason = (
                f"a group agrees with its own centroid only {within:.2f} of the "
                f"{_SPLIT_MIN_WITHIN_SIMILARITY:.2f} needed; not one voice"
            )
        elif across > _SPLIT_MAX_ACROSS_SIMILARITY:
            reason = (
                f"the two groups are {across:.2f} alike, above the "
                f"{_SPLIT_MAX_ACROSS_SIMILARITY:.2f} that makes them different voices"
            )
        elif within - across < _SPLIT_MIN_SEPARATION:
            reason = (
                f"the groups differ by only {within - across:.2f}, below the "
                f"{_SPLIT_MIN_SEPARATION:.2f} separation needed to be sure"
            )

        if reason:
            candidates.append(
                SpeakerSplitCandidate(
                    speaker_id=speaker_id,
                    turns=len(smaller),
                    seconds=smaller_seconds,
                    within_similarity=within,
                    across_similarity=across,
                    reason=reason,
                )
            )
            _LOGGER.info(
                "Kept %s as one speaker: %s (turns %d, %.2fs, within %.2f, across %.2f)",
                speaker_id,
                reason,
                len(smaller),
                smaller_seconds,
                within,
                across,
            )
            continue

        new_speaker_id = _next_speaker_id(taken)
        taken.add(new_speaker_id)
        for position in smaller:
            index = indices[position]
            segment = updated[index]
            updated[index] = SpeakerSegment(new_speaker_id, segment.start, segment.end)
        splits.append(
            SpeakerClusterSplit(
                speaker_id=speaker_id,
                new_speaker_id=new_speaker_id,
                turns=len(smaller),
                seconds=smaller_seconds,
                within_similarity=within,
                across_similarity=across,
            )
        )

    return updated, tuple(splits), tuple(candidates)



@dataclass(frozen=True, slots=True)
class _Scene:
    """One stretch of the film that is heard as a scene, and the turns inside it."""

    index: int
    start: float
    end: float
    turn_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _LocalCluster:
    """One speaker as a scene's own diarization heard them."""

    speaker_id: str
    indexes: tuple[int, ...]
    seconds: float
    vector: tuple[float, ...] | None


def _scene_mode(value: str) -> str:
    """Return the validated scene mode, or raise for a value that is not one."""

    mode = (value or "").strip().lower()
    if mode not in _SCENE_MODES:
        raise InvalidInputError(
            f"DIARIZATION_SCENE_MODE must be one of {', '.join(_SCENE_MODES)}, got {value!r}"
        )
    return mode


def _segment_scenes(
    turns: Sequence[SpeakerSegment], *, gap: float, max_seconds: float
) -> list[_Scene]:
    """Cut the film into scenes at the silences between its turns.

    A film's dialogue already carries its own scene boundaries: people stop talking when
    a scene ends. A gap of ``gap`` seconds, or a stretch longer than ``max_seconds``,
    starts a new scene. The boundaries fall between turns, so a scene always holds whole
    turns and its diarization result can be spliced back without re-timing anything.
    """

    if not turns:
        return []
    scenes: list[_Scene] = []
    current: list[int] = [0]
    start = turns[0].start
    for index in range(1, len(turns)):
        previous = turns[index - 1]
        segment = turns[index]
        if segment.start - previous.end >= gap or segment.end - start > max_seconds:
            scenes.append(
                _Scene(
                    index=len(scenes),
                    start=start,
                    end=previous.end,
                    turn_indexes=tuple(current),
                )
            )
            current = [index]
            start = segment.start
        else:
            current.append(index)
    scenes.append(
        _Scene(
            index=len(scenes),
            start=start,
            end=turns[-1].end,
            turn_indexes=tuple(current),
        )
    )
    return scenes


def _scene_argument(
    pipeline: Any, audio_path: Path, scene: _Scene
) -> tuple[Any, float, str] | None:
    """Return the argument that hands pyannote exactly this scene, the time base of the
    times it will return, and which kind of argument it is.

    The waveform is preferred because the model's times are then relative to a slice
    this module cut, so the base is simply the scene's own start. If the loaded pipeline
    exposes no audio reader, a file mapping with ``offset``/``duration`` is used and the
    model's times are checked before they are trusted.
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
                PyannoteSegment(scene.start, scene.end),
                mode="pad",
            )
        except Exception as exc:  # pragma: no cover - model-specific failure
            _LOGGER.warning(
                "Scene %d (%.2f-%.2fs) could not be cut from the dialogue stem "
                "(%s: %s); falling back to a file slice",
                scene.index,
                scene.start,
                scene.end,
                type(exc).__name__,
                exc,
            )
        else:
            return (
                {"waveform": waveform, "sample_rate": sample_rate},
                scene.start,
                "waveform",
            )

    return (
        {
            "audio": str(audio_path),
            "offset": scene.start,
            "duration": scene.end - scene.start,
        },
        0.0,
        "offset",
    )


def _diarize_scene(
    pipeline: Any, audio_path: Path, scene: _Scene
) -> list[SpeakerSegment] | None:
    """Diarize one scene on its own, or return ``None`` when that is not usable.

    No speaker-count hint is passed: the whole point of the scene pass is to discover
    who is in the scene without being told, including a character the global pass never
    separated. A scene the model cannot handle is reported, never guessed at.
    """

    prepared = _scene_argument(pipeline, audio_path, scene)
    if prepared is None:
        return None
    argument, base, kind = prepared

    try:
        result = pipeline(argument)
    except Exception as exc:
        _LOGGER.warning(
            "Scene %d (%.2f-%.2fs) could not be diarized on its own (%s: %s); "
            "keeping the global attribution for it",
            scene.index,
            scene.start,
            scene.end,
            type(exc).__name__,
            exc,
        )
        return None

    try:
        local = _to_segments(result)
    except (UnsupportedOutputError, InvalidSegmentError) as exc:
        _LOGGER.warning(
            "Scene %d (%.2f-%.2fs) returned a result this module cannot read (%s); "
            "keeping the global attribution for it",
            scene.index,
            scene.start,
            scene.end,
            exc,
        )
        return None
    if not local:
        return None

    duration = scene.end - scene.start
    shift = base
    if kind == "offset" and max(segment.end for segment in local) <= duration + 1e-6:
        # A file slice given an offset should come back on the film's timeline, but if
        # the model returned scene-relative times anyway, place them back here rather
        # than trusting the offset to have been honoured.
        shift = scene.start
    if shift:
        local = [
            SpeakerSegment(
                segment.speaker_id, segment.start + shift, segment.end + shift
            )
            for segment in local
        ]

    if any(
        segment.end <= scene.start - 0.5 or segment.start >= scene.end + 0.5
        for segment in local
    ):
        _LOGGER.warning(
            "Scene %d (%.2f-%.2fs) came back with turns outside the scene; "
            "keeping the global attribution for it",
            scene.index,
            scene.start,
            scene.end,
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


def _reconcile_scenes(
    pipeline: Any,
    audio_path: Path,
    turns: Sequence[SpeakerSegment],
    scenes: Sequence[_Scene],
    registry: identity.SpeakerRegistry,
    *,
    mode: str,
    suspect_speakers: frozenset[str],
    min_similarity: float,
    min_margin: float,
) -> tuple[
    list[SpeakerSegment],
    tuple[SceneDiarization, ...],
    tuple[UncertainSpeaker, ...],
    tuple[str, ...],
    dict[tuple[float, float], tuple[float, ...]],
]:
    """Diarize each scene on its own and name the characters it finds.

    A scene's own result is only kept when it names a character the global pass missed
    for that scene. Anything else would re-litigate a decision the global pass already
    made on far more audio, and scene-sized diarization is noisier than a film-sized
    one - so it is allowed to *add* identities, never to shuffle the ones already named.

    A scene voice that matches an existing identity reuses its id, which is how a
    character heard in three different scenes ends up with one name and one voice. A
    voice that matches nothing is given a new id. A voice that is too close to two
    identities to separate is given a new id too, and reported as uncertain: assigning
    it to either one would clone a character in another character's voice.
    """

    reports: list[SceneDiarization] = []
    uncertain: list[UncertainSpeaker] = []
    replacements: dict[int, list[SpeakerSegment]] = {}
    known = set(registry)
    discovered: list[str] = []
    accepted_vectors: dict[tuple[float, float], tuple[float, ...]] = {}

    for scene in scenes:
        scene_turns = [turns[index] for index in scene.turn_indexes]
        before = {segment.speaker_id for segment in scene_turns}
        span = scene.end - scene.start
        if span < _SCENE_MIN_SPAN_SECONDS:
            reports.append(
                SceneDiarization(
                    index=scene.index,
                    start=scene.start,
                    end=scene.end,
                    turns=len(scene_turns),
                    speakers=tuple(sorted(before)),
                    mode="global",
                    detail=f"only {span:.1f}s of film; too short to diarize on its own",
                )
            )
            continue
        if mode == "suspect" and not (before & suspect_speakers):
            reports.append(
                SceneDiarization(
                    index=scene.index,
                    start=scene.start,
                    end=scene.end,
                    turns=len(scene_turns),
                    speakers=tuple(sorted(before)),
                    mode="global",
                    detail="no suspected merge in this scene",
                )
            )
            continue

        local = _diarize_scene(pipeline, audio_path, scene)
        if local is None:
            reports.append(
                SceneDiarization(
                    index=scene.index,
                    start=scene.start,
                    end=scene.end,
                    turns=len(scene_turns),
                    speakers=tuple(sorted(before)),
                    mode="failed",
                    detail="the scene could not be diarized on its own",
                )
            )
            continue

        local_vectors, _ = _evidence_vectors(pipeline, audio_path, local, ())
        clusters = _local_clusters(local, local_vectors)
        mapping: dict[str, str] = {}
        for label, cluster in sorted(clusters.items()):
            if cluster.vector is None or cluster.seconds < _SCENE_MIN_CLUSTER_SECONDS:
                continue
            decision = registry.identify(
                cluster.vector,
                min_similarity=min_similarity,
                min_margin=min_margin,
            )
            if decision.matched and decision.speaker_id is not None:
                mapping[label] = decision.speaker_id
                registry.observe(
                    decision.speaker_id,
                    cluster.vector,
                    cluster.seconds,
                    turns=len(cluster.indexes),
                    scenes=1,
                )
                continue
            ambiguous = decision.outcome == identity.AMBIGUOUS
            minted = registry.mint(
                cluster.vector,
                cluster.seconds,
                turns=len(cluster.indexes),
                scenes=1,
                uncertain=ambiguous,
            )
            mapping[label] = minted.speaker_id
            if minted.speaker_id not in known:
                known.add(minted.speaker_id)
                discovered.append(minted.speaker_id)
            if ambiguous:
                uncertain.append(
                    UncertainSpeaker(
                        speaker_id=minted.speaker_id,
                        reason=(
                            "too close to an existing identity to separate; given its own "
                            "id rather than reusing either"
                        ),
                        similarity=decision.similarity,
                        margin=decision.margin,
                        seconds=cluster.seconds,
                    )
                )

        resolved = {mapping[label] for label in mapping}
        new_characters = resolved - before
        if not new_characters:
            reports.append(
                SceneDiarization(
                    index=scene.index,
                    start=scene.start,
                    end=scene.end,
                    turns=len(scene_turns),
                    speakers=tuple(sorted(before)),
                    mode="global",
                    detail="the scene's own diarization found no character the global "
                    "pass missed",
                )
            )
            continue

        def _global_label_for(segment: SpeakerSegment) -> str | None:
            for candidate in scene_turns:
                if candidate.start < segment.end and candidate.end > segment.start:
                    return candidate.speaker_id
            return None

        replacement: list[SpeakerSegment] = []
        for segment in local:
            mapped = mapping.get(segment.speaker_id) or _global_label_for(segment)
            if mapped is None:
                continue
            replacement.append(
                SpeakerSegment(mapped, segment.start, segment.end)
            )
        if not replacement:
            continue
        replacements[scene.index] = replacement
        accepted_vectors.update(_vectors_by_time(local, local_vectors))
        reports.append(
            SceneDiarization(
                index=scene.index,
                start=scene.start,
                end=scene.end,
                turns=len(replacement),
                speakers=tuple(
                    sorted({segment.speaker_id for segment in replacement})
                ),
                mode="scene",
                detail=(
                    "the scene's own diarization separated "
                    f"{len(resolved - before)} character(s) the global pass merged"
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

    merged: list[SpeakerSegment] = []
    for scene in scenes:
        if scene.index in replacements:
            merged.extend(replacements[scene.index])
        else:
            merged.extend(turns[index] for index in scene.turn_indexes)
    merged.sort(key=lambda segment: (segment.start, segment.end, segment.speaker_id))
    return (
        merged,
        tuple(reports),
        tuple(uncertain),
        tuple(discovered),
        accepted_vectors,
    )


def _merge_duplicate_identities(
    turns: Sequence[SpeakerSegment],
    vectors: Mapping[tuple[float, float], tuple[float, ...]],
    registry: identity.SpeakerRegistry,
    *,
    merge_similarity: float,
    suspect_similarity: float,
) -> tuple[list[SpeakerSegment], tuple[SpeakerIdentityMerge, ...], tuple[SpeakerMergeSuspect, ...]]:
    """Merge identities that are the same person, and report the ones that are close.

    Scene-level diarization can name one character twice - a voice heard in two scenes
    may not match its own earlier voiceprint - so pairs that are almost identical are
    merged. A pair that is merely *similar* is reported instead of merged: a wrong merge
    gives two characters one voice, which is exactly the failure this module exists to
    prevent.
    """

    prototypes = _label_prototypes(turns, vectors)
    if len(prototypes) < 2:
        return list(turns), (), ()

    speaker_ids = sorted(prototypes)
    pairs: list[tuple[float, str, str]] = []
    for position, first in enumerate(speaker_ids):
        for second in speaker_ids[position + 1 :]:
            similarity = identity.cosine_similarity(
                prototypes[first][0], prototypes[second][0]
            )
            pairs.append((similarity, first, second))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    survivor: dict[str, str] = {speaker_id: speaker_id for speaker_id in speaker_ids}

    def resolve(speaker_id: str) -> str:
        while survivor[speaker_id] != speaker_id:
            speaker_id = survivor[speaker_id]
        return speaker_id

    merges: list[SpeakerIdentityMerge] = []
    suspects: list[SpeakerMergeSuspect] = []
    for similarity, first, second in pairs:
        left, right = resolve(first), resolve(second)
        if left == right:
            continue
        if similarity >= merge_similarity:
            left_seconds = prototypes[left][1]
            right_seconds = prototypes[right][1]
            keep, drop = (left, right) if left_seconds >= right_seconds else (right, left)
            survivor[drop] = keep
            registry.merge(drop, keep)
            merges.append(
                SpeakerIdentityMerge(
                    from_speaker_id=drop,
                    to_speaker_id=keep,
                    similarity=similarity,
                    turns=prototypes[drop][2],
                    seconds=prototypes[drop][1],
                )
            )
            continue
        if similarity >= suspect_similarity:
            suspects.append(
                SpeakerMergeSuspect(
                    speaker_ids=(left, right),
                    similarity=similarity,
                    reason=(
                        "the two voiceprints are unusually alike; they were left as "
                        "separate speakers rather than risk one voice for two characters"
                    ),
                )
            )

    if not merges:
        return list(turns), (), tuple(suspects)

    renamed = [
        SpeakerSegment(
            survivor.get(segment.speaker_id, segment.speaker_id),
            segment.start,
            segment.end,
        )
        for segment in turns
    ]
    return renamed, tuple(merges), tuple(suspects)


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


def diarize_detailed(
    audio_path: str | Path,
    *,
    settings: Settings | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    registry_path: str | Path | None = None,
) -> DiarizationResult:
    """Diarize ``audio_path`` into turns, identities, and what could not be settled.

    The pipeline has three passes, and each one exists because the previous one cannot
    see something:

    1. **Global.** Community-1 diarizes the whole dialogue stem. Its speaker count is
       automatic: no count is passed unless the caller configured bounds.
    2. **Evidence.** Every turn of at least half a second is embedded, turns heard
       together are aggregated into one voiceprint, attributions are verified against
       Community-1's own prototypes, and a cluster whose turns hold two voices is split.
       Short utterances are aggregated rather than discarded, and any cluster that
       *nearly* qualified is reported instead of being passed over.
    3. **Scenes.** The film is cut into scenes at its silences, each scene is diarized
       on its own, and the characters found in it are named against a registry of
       identities. That is what finds a character whose whole performance is a handful
       of short lines in three different scenes, and it is what keeps those lines in
       one voice.

    Speaker-count hints that are left unset fall back to
    ``DIARIZATION_MIN_SPEAKERS`` / ``DIARIZATION_MAX_SPEAKERS``; an explicit
    argument always wins over the configured value. Neither is required: the number of
    speakers is discovered, not declared.
    """

    resolved = settings if settings is not None else get_settings()

    source = _validate_audio_path(audio_path)
    token = _require_huggingface_token(resolved)
    num_speakers, min_speakers, max_speakers = _resolve_speaker_counts(
        resolved, num_speakers, min_speakers, max_speakers
    )
    _validate_speaker_counts(num_speakers, min_speakers, max_speakers)
    mode = _scene_mode(resolved.diarization_scene_mode)
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
    turns = _to_segments(result)

    scenes: list[_Scene] = []
    if mode != "off":
        scenes = _segment_scenes(
            turns,
            gap=resolved.diarization_scene_gap_seconds,
            max_seconds=resolved.diarization_scene_max_seconds,
        )

    centroids = _speaker_centroids(result)
    vectors: dict[int, tuple[float, ...]] = {}
    if _needs_evidence(
        turns,
        centroids,
        scenes,
        mode=mode,
        split=resolved.diarization_split_mixed_speakers,
        short_turn=short_turn,
    ):
        vectors, unmeasured = _evidence_vectors(
            pipeline, source, turns, crosstalk, minimum_seconds=short_turn
        )
    else:
        unmeasured = sum(
            1
            for segment in turns
            if segment.duration < short_turn or _is_crosstalk_turn(segment, crosstalk)
        )

    groups = _evidence_groups(turns, vectors)
    turns, corrections = _verify_speaker_attributions(centroids, turns, groups)

    splits: tuple[SpeakerClusterSplit, ...] = ()
    candidates: tuple[SpeakerSplitCandidate, ...] = ()
    if resolved.diarization_split_mixed_speakers:
        turns, splits, candidates = _split_mixed_clusters(turns, vectors)
        for split in splits:
            _LOGGER.info(
                "Split %d turn(s) / %.2fs out of %s into %s (within %.2f, across %.2f)",
                split.turns,
                split.seconds,
                split.speaker_id,
                split.new_speaker_id,
                split.within_similarity,
                split.across_similarity,
            )

    scene_reports: tuple[SceneDiarization, ...] = ()
    uncertain: tuple[UncertainSpeaker, ...] = ()
    discovered: tuple[str, ...] = ()
    merges: tuple[SpeakerIdentityMerge, ...] = ()
    suspects: tuple[SpeakerMergeSuspect, ...] = ()
    registry_path_written: str | None = None
    registry_error = ""

    if mode != "off" and len(scenes) > 1 and vectors:
        registry, registry_error = _load_registry(
            resolved, registry_path=registry_path
        )
        timed_vectors = _vectors_by_time(turns, vectors)
        # The global pass names the cast; the registry carries those names forward and
        # is what scene-level diarization matches against.
        for speaker_id, (vector, seconds, count) in _label_prototypes(
            turns, timed_vectors
        ).items():
            registry.observe(speaker_id, vector, seconds, turns=count)
        turns, scene_reports, uncertain, discovered, scene_vectors = _reconcile_scenes(
            pipeline,
            source,
            turns,
            scenes,
            registry,
            mode=mode,
            suspect_speakers=frozenset(
                candidate.speaker_id for candidate in candidates
            ),
            min_similarity=resolved.diarization_identity_min_similarity,
            min_margin=resolved.diarization_identity_min_margin,
        )
        timed_vectors.update(scene_vectors)
        turns, merges, suspects = _merge_duplicate_identities(
            turns,
            timed_vectors,
            registry,
            merge_similarity=resolved.diarization_identity_merge_similarity,
            suspect_similarity=identity.DEFAULT_SUSPECT_SIMILARITY,
        )
        registry_path_written, saved_error = _save_registry(
            registry, resolved, registry_path=registry_path
        )
        registry_error = registry_error or saved_error
        _LOGGER.info(
            "Speaker registry: %d identit%s, %d scene(s) diarized on their own",
            len(registry),
            "y" if len(registry) == 1 else "ies",
            sum(1 for report in scene_reports if report.mode == "scene"),
        )

    return DiarizationResult(
        turns=tuple(turns),
        crosstalk=crosstalk,
        corrections=corrections,
        splits=splits,
        split_candidates=candidates,
        scenes=scene_reports,
        uncertain_speakers=uncertain,
        merge_suspects=suspects,
        merges=merges,
        discovered_speakers=discovered,
        unmeasured_turns=unmeasured,
        registry_path=registry_path_written,
        registry_error=registry_error,
    )


def _needs_evidence(
    turns: Sequence[SpeakerSegment],
    centroids: Mapping[str, tuple[float, ...]],
    scenes: Sequence[_Scene],
    *,
    mode: str,
    split: bool,
    short_turn: float,
) -> bool:
    """Return whether anything downstream can use a speaker embedding.

    Embeddings are the expensive part of this module, and a single-scene film needs no
    scene pass and no registry; the check exists so a run is not made slower for
    information nothing will read.
    """

    if len(centroids) >= 2:
        return True
    if split:
        counts: dict[str, int] = {}
        for segment in turns:
            if segment.duration >= short_turn:
                counts[segment.speaker_id] = counts.get(segment.speaker_id, 0) + 1
        if any(count >= _SPLIT_MIN_CLUSTER_TURNS for count in counts.values()):
            return True
    return mode != "off" and len(scenes) > 1


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


def _load_registry(
    settings: Settings, *, registry_path: str | Path | None
) -> tuple[identity.SpeakerRegistry, str]:
    """Read the run's identities, or start empty and say why if that fails."""

    target = _registry_target(settings, registry_path)
    if target is None:
        return identity.SpeakerRegistry(), ""
    try:
        registry = identity.SpeakerRegistry.load(target)
    except (identity.UnreadableRegistryError, OSError) as exc:
        _LOGGER.warning(
            "Speaker registry %s could not be read (%s); starting a new one", target, exc
        )
        return identity.SpeakerRegistry(), str(exc)
    return registry, ""


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
    "SceneDiarization",
    "SpeakerAttributionCorrection",
    "SpeakerClusterSplit",
    "SpeakerIdentityMerge",
    "SpeakerMergeSuspect",
    "SpeakerSegment",
    "SpeakerSplitCandidate",
    "UncertainSpeaker",
    "UnsupportedOutputError",
    "diarize",
    "diarize_detailed",
]
