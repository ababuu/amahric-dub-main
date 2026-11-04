"""Dialogue adaptation into dubbing-ready spoken Amharic, on a local GPU LLM.

This module is the only place in the pipeline that talks to a language model. It
turns the transcribed source-language lines produced by
:mod:`app.pipeline.transcription` into Amharic dialogue that OmniVoice can perform::

    [TranscriptSegment, ...] -> adapt_dialogue() -> [AdaptedDialogue, ...]

The model is **Qwen-2.5-Instruct served by vLLM** on the same GPU host as the rest of
the pipeline. There is no hosted vendor, no API key to configure and no fallback
provider: a run either reaches the local server or fails, and it fails with the reason.

Why the model does five jobs in one request
-------------------------------------------
A dub is not a translation. The Amharic has to be *aligned* to the script's complete
sentences even though transcription hands over fragments, has to fit a duration measured
in seconds rather than words, has to carry the physical noises the screenplay wrote
inline, and has to tell the voice engine how the line is delivered. Asking five separate
models to do that would let them disagree with each other, so one JSON document is sent
and one JSON document comes back - see :data:`SYSTEM_PROMPT` for the contract.

Chronological windows, and why they overlap
-------------------------------------------
A film is too long for one request, and a single random sentence has no context at all:
pronouns, who is answering whom, and the register of an exchange all live in the lines
around it. Dialogue is therefore cut into sequential windows of
``TRANSLATION_WINDOW_LINES`` lines that share ``TRANSLATION_WINDOW_OVERLAP`` lines with
their neighbours, and the windows are submitted **concurrently**:

    window 0: lines  0..23
    window 1: lines 19..42
    window 2: lines 38..61

The overlap is what stops a scene from being severed at a boundary - the last lines of
one window are read again with the first lines of the next as context. It also means a
line is translated more than once, so :func:`deduplicate_records` merges the parallel
answers back into one entry per line before anything downstream sees them.

VRAM is throttled from both ends: the client holds an :class:`asyncio.Semaphore` of
``TRANSLATION_MAX_CONCURRENCY`` (``asyncio.gather`` alone would launch every window at
once), and the vLLM server is launched with ``--gpu-memory-utilization`` (see
:meth:`app.config.Settings.vllm_server_arguments`).

Who owns what
-------------
The application owns identity and timing. Every input line has a stable id
(``dialogue_000001``, ...) and the model must echo its ``start_timestamp`` and
``end_timestamp`` unchanged. A response whose timestamps do not match the film is
rejected rather than trusted: the model cannot be allowed to move a line on the
timeline. The character is the screenplay's, normalised to one strict spelling, because
it is the key a ``voice_samples/<Character>/`` directory is looked up by.

Contract with the ``openai`` package
------------------------------------
* ``AsyncOpenAI(base_url=..., api_key=...)`` builds the client for vLLM's
  OpenAI-compatible endpoint.
* ``client.chat.completions.create(model=..., messages=[...],
  response_format={"type": "json_object"}, temperature=..., max_tokens=...)`` returns
  an object whose ``choices[0].message.content`` holds the JSON text.
* Any test injects a client implementing :class:`CompletionClient` instead, so no test
  reaches the network.

Deliberate non-goals
--------------------
* **No hosted providers.** DeepSeek, Gemini and NLLB are gone; adding one back would be
  a new client, not a configuration key.
* **No re-translation pass.** The prompt is the length constraint; a window that comes
  back malformed is retried once with the parser's complaint, and then the run stops.
* **No vendor objects in the public API.** Callers only ever see
  :class:`AdaptedDialogue`.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from app.config import Settings, get_settings
from app.pipeline.amharic_text import has_pronounceable_text
from app.pipeline.dialogue_context import SyllableBudget, syllable_budget
from app.pipeline.script_parse import ScriptLine, normalize_character_name
from app.pipeline.transcription import DialogueSentence, TranscriptSegment

#: Prefix and width of the application-generated dialogue ids, e.g.
#: ``dialogue_000001``. The model must echo these back unchanged.
DIALOGUE_ID_PREFIX = "dialogue_"
DIALOGUE_ID_DIGITS = 6

#: The closed emotion vocabulary. A line is spoken in exactly one of these, and the
#: value is handed to OmniVoice as a performance instruction, so a descriptive string
#: ("quietly furious, but hiding it") is not an answer - it is a sentence the engine
#: would try to read aloud.
EMOTIONS: tuple[str, ...] = ("neutral", "angry", "sad", "excited", "whisper", "fearful")

#: The only inline tags allowed in the Amharic text. They are physical noises the
#: voice engine can actually produce; anything else is stripped, because an emotional
#: descriptor inside the text is spoken as words.
INLINE_SOUND_TAGS: tuple[str, ...] = ("sigh", "breath", "gasp", "laughter")

#: Intensity implied by each emotion, used for the manifest and for engines that read
#: an intensity but no emotion. Never invented per line: it is a property of the mood.
EMOTION_INTENSITY: Mapping[str, float] = {
    "neutral": 0.45,
    "angry": 0.85,
    "sad": 0.55,
    "excited": 0.80,
    "whisper": 0.30,
    "fearful": 0.70,
}

#: Delivery wording recorded beside each emotion. The voice engine takes the emotion;
#: this is what a human director reads in the manifest.
EMOTION_DELIVERY: Mapping[str, str] = {
    "neutral": "natural",
    "angry": "forceful",
    "sad": "subdued",
    "excited": "animated",
    "whisper": "hushed",
    "fearful": "trembling",
}

#: Slack allowed between the timestamp the model echoes and the film's own, in seconds.
TIMESTAMP_TOLERANCE = 0.05

#: Attempts one window gets. The second attempt is the same window plus the parser's
#: complaint, which is what recovers a truncated or unquoted JSON reply without a
#: film-long retry loop.
MAX_WINDOW_ATTEMPTS = 2

#: How many script lines are shown beyond a window's own span, so the model can see the
#: complete sentence a fragment belongs to and the cue that follows it.
SCRIPT_EXCERPT_PADDING = 8

_SOUND_TAG_PATTERN = re.compile(r"\[([^\[\]]{1,40})\]")
_WHITESPACE = re.compile(r"\s+")
_DIARIZATION_LABEL = re.compile(r"^SPEAKER[\s_-]?\d+$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TranslationError(RuntimeError):
    """Base class for every error raised by this module."""


class ConfigurationError(TranslationError, ValueError):
    """A setting or an injected client makes adaptation impossible."""


class ApiClientError(TranslationError):
    """The local vLLM server could not be reached."""


class ApiRequestError(TranslationError):
    """The server answered with an error."""


class ApiAuthenticationError(ApiRequestError):
    """The server rejected the configured token."""


class MalformedResponseError(TranslationError):
    """The reply was not the JSON object the prompt asked for."""


class InvalidModelOutputError(TranslationError):
    """The reply parsed, but described lines the film does not have."""


class DialogueIdError(InvalidModelOutputError, ValueError):
    """A requested line id is missing, duplicated, or unknown."""


class InvalidTranscriptSegmentsError(TranslationError, ValueError):
    """The dialogue handed to this module is not a sequence of dialogue units."""


class InvalidSegmentError(TranslationError, ValueError):
    """One dialogue unit cannot be described as a line."""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _finite_number(name: str, value: object) -> float:
    """Return ``value`` as a finite ``float``."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidSegmentError(f"{name} must be a number, got {type(value).__name__}")

    number = float(value)
    if not math.isfinite(number):
        raise InvalidSegmentError(f"{name} must be finite, got {value!r}")
    return number


def _required_text(name: str, value: object) -> str:
    """Return ``value`` stripped, or fail when it is not usable text."""

    if not isinstance(value, str) or not value.strip():
        raise InvalidSegmentError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class PerformanceBeat:
    """How one part of a sentence is performed, on the original timeline.

    Sentences used to arrive as several timed parts, and the manifest still records
    them when a caller supplies them by hand. The local model returns one mood per
    line, so a line adapted here carries no beats; the type stays because it is part
    of the artifact schema the manifest and the TTS record read.
    """

    start: float
    end: float
    emotion: str
    intensity: float
    delivery: str

    def __post_init__(self) -> None:
        start = _finite_number("start", self.start)
        end = _finite_number("end", self.end)
        if start < 0:
            raise InvalidSegmentError(f"start must be >= 0 seconds, got {start}")
        if end <= start:
            raise InvalidSegmentError(
                f"end must be greater than start ({start} seconds), got {end}"
            )

        emotion = _required_text("emotion", self.emotion)
        delivery = _required_text("delivery", self.delivery)
        intensity = _finite_number("intensity", self.intensity)
        if not 0.0 <= intensity <= 1.0:
            raise InvalidSegmentError(
                f"intensity must be between 0.0 and 1.0, got {intensity}"
            )

        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "emotion", emotion)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "delivery", delivery)

    def as_dict(self) -> dict[str, Any]:
        """Return the beat as a manifest-friendly mapping."""

        return {
            "start": self.start,
            "end": self.end,
            "emotion": self.emotion,
            "intensity": self.intensity,
            "delivery": self.delivery,
        }


@dataclass(frozen=True, slots=True)
class AdaptedDialogue:
    """One dubbing-ready Amharic line plus its performance direction.

    One line is one **sentence** - the unit the model was asked to write and the unit
    the voice speaks - not a transcriber's pause-delimited segment.

    Instances are immutable and always valid, and the timing fields are always the
    application-owned values from the original line, never the model's arithmetic.
    ``speaker_id`` holds the *character name* resolved from the screenplay, which is
    what makes ``voice_samples/<Character>/`` a folder lookup rather than a mapping.
    """

    speaker_id: str
    start: float
    end: float
    source_text: str
    amharic: str
    emotion: str
    intensity: float
    delivery: str
    pause_before: float
    pause_after: float
    beats: tuple[PerformanceBeat, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.speaker_id, str) or not self.speaker_id.strip():
            raise InvalidSegmentError("speaker_id must be a non-empty string")

        start = _finite_number("start", self.start)
        end = _finite_number("end", self.end)
        if start < 0:
            raise InvalidSegmentError(f"start must be >= 0 seconds, got {start}")
        if end <= start:
            raise InvalidSegmentError(
                f"end must be greater than start ({start} seconds), got {end}"
            )

        source_text = _required_text("source_text", self.source_text)
        amharic = _required_text("amharic", self.amharic)
        emotion = _required_text("emotion", self.emotion)
        delivery = _required_text("delivery", self.delivery)

        intensity = _finite_number("intensity", self.intensity)
        if not 0.0 <= intensity <= 1.0:
            raise InvalidSegmentError(
                f"intensity must be between 0.0 and 1.0, got {intensity}"
            )

        pause_before = _finite_number("pause_before", self.pause_before)
        pause_after = _finite_number("pause_after", self.pause_after)
        if pause_before < 0:
            raise InvalidSegmentError(
                f"pause_before must be >= 0 seconds, got {pause_before}"
            )
        if pause_after < 0:
            raise InvalidSegmentError(
                f"pause_after must be >= 0 seconds, got {pause_after}"
            )

        beats = tuple(self.beats)
        for index, beat in enumerate(beats):
            if not isinstance(beat, PerformanceBeat):
                raise InvalidSegmentError(
                    f"beats[{index}] is {type(beat).__name__}, expected a PerformanceBeat"
                )

        # Normalise so the declared types are guaranteed.
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "source_text", source_text)
        object.__setattr__(self, "amharic", amharic)
        object.__setattr__(self, "emotion", emotion)
        object.__setattr__(self, "delivery", delivery)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "pause_before", pause_before)
        object.__setattr__(self, "pause_after", pause_after)
        object.__setattr__(self, "beats", beats)

    @property
    def character(self) -> str:
        """The normalised character name that will speak this line."""

        return self.speaker_id

    @property
    def duration(self) -> float:
        """Length of the original line in seconds."""

        return self.end - self.start

    @property
    def sound_tags(self) -> tuple[str, ...]:
        """The inline physical sound tags carried by the Amharic, in order."""

        return tuple(_SOUND_TAG_PATTERN.findall(self.amharic))


def _dialogue_id(position: int) -> str:
    """Return the stable id of the ``position``-th line (0-based)."""

    return f"{DIALOGUE_ID_PREFIX}{position + 1:0{DIALOGUE_ID_DIGITS}d}"


def _validate_segments(
    segments: Iterable[TranscriptSegment | DialogueSentence],
) -> list[TranscriptSegment | DialogueSentence]:
    """Return the dialogue units as a chronological, validated list."""

    if isinstance(segments, (str, bytes)) or not isinstance(segments, Iterable):
        raise InvalidTranscriptSegmentsError(
            "segments must be an iterable of TranscriptSegment or DialogueSentence "
            f"objects, got {type(segments).__name__}"
        )

    items = list(segments)
    for index, item in enumerate(items):
        if not isinstance(item, (TranscriptSegment, DialogueSentence)):
            raise InvalidTranscriptSegmentsError(
                f"segments[{index}] is {type(item).__name__}, expected a "
                "TranscriptSegment or DialogueSentence"
            )
    return sorted(items, key=lambda item: (item.start, item.end, item.speaker_id))


# ---------------------------------------------------------------------------
# Windows, and merging them back
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Window:
    """One concurrent request's slice of the film, in line positions."""

    index: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ConfigurationError(f"window index must be >= 0, got {self.index!r}")
        for name in ("start", "end"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigurationError(f"window {name} must be >= 0, got {value!r}")
        if self.end <= self.start:
            raise ConfigurationError(
                f"a window must cover at least one line, got [{self.start}, {self.end})"
            )

    @property
    def length(self) -> int:
        """How many lines this window covers."""

        return self.end - self.start

    def edge_distance(self, position: int) -> int:
        """How far ``position`` sits from the nearer edge of the window.

        A line in the middle of a window was read with the most context around it, so
        its answer is the one worth keeping when the same line comes back twice.
        """

        return min(position - self.start, self.end - 1 - position)


def overlapping_windows(
    count: int, *, size: int, overlap: int
) -> tuple[Window, ...]:
    """Return the chronological windows covering ``count`` lines.

    ``size`` lines per window, ``overlap`` lines shared with the next one. The overlap
    must be smaller than the window, or the plan would never advance.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ConfigurationError(f"line count must be >= 0, got {count!r}")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ConfigurationError(f"window size must be >= 1, got {size!r}")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
        raise ConfigurationError(f"window overlap must be >= 0, got {overlap!r}")
    if overlap >= size:
        raise ConfigurationError(
            f"window overlap ({overlap}) must be smaller than the window size ({size}), "
            "or the windows never advance"
        )
    if count == 0:
        return ()

    windows: list[Window] = []
    start = 0
    while True:
        end = min(count, start + size)
        windows.append(Window(index=len(windows), start=start, end=end))
        if end >= count:
            return tuple(windows)
        start = end - overlap


@dataclass(frozen=True, slots=True)
class WindowRecord:
    """One model answer for one line, tagged with the window that produced it."""

    position: int
    window: Window
    dialogue: AdaptedDialogue

    def __post_init__(self) -> None:
        if not isinstance(self.window, Window):
            raise InvalidModelOutputError(
                f"window must be a Window, got {type(self.window).__name__}"
            )
        if not isinstance(self.dialogue, AdaptedDialogue):
            raise InvalidModelOutputError(
                f"dialogue must be AdaptedDialogue, got {type(self.dialogue).__name__}"
            )
        if not self.window.start <= self.position < self.window.end:
            raise InvalidModelOutputError(
                f"line {self.position} is outside window [{self.window.start}, "
                f"{self.window.end})"
            )


def _record_origin(position: int, window: Window) -> tuple[int, int]:
    """Sort key for competing answers: most context first, then earliest window."""

    return (-window.edge_distance(position), window.index)


def deduplicate_records(
    records: Iterable[WindowRecord], *, total: int
) -> tuple[AdaptedDialogue, ...]:
    """Merge overlapping windows into exactly one answer per line.

    Overlapping windows translate some lines twice. The duplicates are not identical -
    the model saw each copy with different surrounding lines - so keeping an arbitrary
    one would make the result depend on which request happened to finish first. The
    answer is kept from the window in which the line sat **furthest from an edge**,
    where its context was most complete; ties go to the earlier window, which makes the
    merge deterministic.

    Raises
    ------
    DialogueIdError
        A line position was never answered, or the same window answered it twice.
    InvalidModelOutputError
        A line was answered with a different character by different windows.
    """

    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ConfigurationError(f"total must be >= 0, got {total!r}")

    chosen: dict[int, WindowRecord] = {}
    seen: set[tuple[int, int]] = set()
    for record in records:
        if not isinstance(record, WindowRecord):
            raise InvalidModelOutputError(
                f"expected a WindowRecord, got {type(record).__name__}"
            )
        if record.position >= total:
            raise DialogueIdError(
                f"line {record.position} was answered but the film has {total} line(s)"
            )
        key = (record.window.index, record.position)
        if key in seen:
            raise DialogueIdError(
                f"window {record.window.index} answered line {record.position} twice"
            )
        seen.add(key)

        incumbent = chosen.get(record.position)
        if incumbent is None or _record_origin(record.position, record.window) < (
            _record_origin(record.position, incumbent.window)
        ):
            chosen[record.position] = record

    missing = [position for position in range(total) if position not in chosen]
    if missing:
        shown = ", ".join(str(position) for position in missing[:10])
        raise DialogueIdError(
            f"{len(missing)} line(s) were never answered (positions {shown}"
            f"{', ...' if len(missing) > 10 else ''})"
        )

    ordered = [chosen[position] for position in range(total)]
    for position in range(1, total):
        previous = ordered[position - 1].dialogue
        current = ordered[position].dialogue
        if current.start < previous.start - 1e-6:
            raise InvalidModelOutputError(
                f"line {position} starts before the line before it; the timeline is "
                "not chronological"
            )
    return tuple(record.dialogue for record in ordered)


# ---------------------------------------------------------------------------
# Model output: parsing, sanitising, and validation
# ---------------------------------------------------------------------------


def normalize_emotion(value: str) -> str:
    """Return the closed-vocabulary emotion ``value`` names.

    Raises
    ------
    InvalidModelOutputError
        ``value`` is not one of :data:`EMOTIONS`.
    """

    if not isinstance(value, str):
        raise InvalidModelOutputError(
            f"overall_emotion must be a string, got {type(value).__name__}"
        )
    emotion = value.strip().lower()
    if emotion not in EMOTIONS:
        raise InvalidModelOutputError(
            f"overall_emotion {value!r} is not one of {', '.join(EMOTIONS)}; the "
            "vocabulary is closed because the value is sent to the voice engine as a "
            "performance instruction"
        )
    return emotion


def sanitize_inline_tags(text: str) -> str:
    """Return ``text`` with only the physical inline tags kept.

    The prompt allows ``[sigh]``, ``[breath]``, ``[gasp]`` and ``[laughter]`` inside the
    Amharic. Anything else a model writes in brackets - ``[angry]``, ``[softly]``,
    ``[voice breaking]`` - is an emotion description, and a voice engine reads it as
    words. Those are removed here rather than trusted to survive the prompt.
    """

    if not isinstance(text, str):
        raise InvalidModelOutputError(
            f"amharic_text_with_tags must be a string, got {type(text).__name__}"
        )

    def replace(match: re.Match[str]) -> str:
        tag = match.group(1).strip().lower().rstrip(".!")
        return f"[{tag}]" if tag in INLINE_SOUND_TAGS else " "

    cleaned = _SOUND_TAG_PATTERN.sub(replace, text)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    return cleaned


def parse_response(content: str, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Return the model's ``lines`` payload keyed by id.

    The reply must be one JSON object with a ``lines`` array, and every requested id
    must appear exactly once. A missing, duplicated or unknown id fails the window: a
    line that silently disappears is a line missing from the dub.

    Raises
    ------
    MalformedResponseError, DialogueIdError
    """

    if not isinstance(content, str) or not content.strip():
        raise MalformedResponseError("the model returned an empty reply")

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise MalformedResponseError(f"the reply is not JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise MalformedResponseError(
            f"the reply must be a JSON object, got {type(payload).__name__}"
        )

    entries = payload.get("lines")
    if not isinstance(entries, list):
        raise MalformedResponseError("the reply must hold a 'lines' array")

    requested = set(ids)
    found: dict[str, dict[str, Any]] = {}
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise MalformedResponseError(
                f"lines[{position}] must be a JSON object, got {type(entry).__name__}"
            )
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise MalformedResponseError(f"lines[{position}] has no string id")
        identifier = identifier.strip()
        if identifier not in requested:
            raise DialogueIdError(
                f"the model answered {identifier!r}, which was not asked for"
            )
        if identifier in found:
            raise DialogueIdError(f"the model answered {identifier!r} twice")
        found[identifier] = entry

    missing = [identifier for identifier in ids if identifier not in found]
    if missing:
        raise DialogueIdError(
            f"the model did not answer {len(missing)} line(s), starting with "
            f"{missing[0]!r}"
        )
    return found


def _coerce_timestamp(
    value: object, name: str, expected: float, identifier: str
) -> float:
    """Return ``value``, or fail when the model moved the line on the timeline."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidModelOutputError(
            f"{identifier}: {name} must be a number, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise InvalidModelOutputError(f"{identifier}: {name} must be finite")
    if abs(number - expected) > TIMESTAMP_TOLERANCE:
        raise InvalidModelOutputError(
            f"{identifier}: {name} came back as {number:.3f}s but the film has "
            f"{expected:.3f}s; the application owns timing and a moved line would "
            "break lip-sync"
        )
    return expected


def _resolve_character(
    raw: object,
    *,
    fallback: str,
    script_characters: Sequence[str],
) -> str:
    """Return the character name for one line.

    The screenplay's own names win: a name the script does not contain is a model
    invention and would create a voice profile for a character who does not exist.
    """

    if isinstance(raw, str) and raw.strip():
        try:
            normalized = normalize_character_name(raw)
        except ValueError:
            normalized = ""
        if normalized:
            if not script_characters or normalized in script_characters:
                return normalized

    if not _DIARIZATION_LABEL.match(fallback.strip()):
        return fallback.strip()
    return fallback.strip()


def _record_to_dialogue(
    entry: Mapping[str, Any],
    *,
    identifier: str,
    source: TranscriptSegment | DialogueSentence,
    script_characters: Sequence[str],
) -> AdaptedDialogue:
    """Validate one model answer and turn it into an :class:`AdaptedDialogue`."""

    start = _coerce_timestamp(
        entry.get("start_timestamp"), "start_timestamp", source.start, identifier
    )
    end = _coerce_timestamp(
        entry.get("end_timestamp"), "end_timestamp", source.end, identifier
    )

    emotion = normalize_emotion(entry.get("overall_emotion", ""))
    text = sanitize_inline_tags(entry.get("amharic_text_with_tags", ""))
    if not text or not has_pronounceable_text(text):
        raise InvalidModelOutputError(
            f"{identifier}: amharic_text_with_tags holds nothing a voice can say"
        )

    character = _resolve_character(
        entry.get("character"), fallback=source.speaker_id, script_characters=script_characters
    )

    return AdaptedDialogue(
        speaker_id=character,
        start=start,
        end=end,
        source_text=source.text,
        amharic=text,
        emotion=emotion,
        intensity=EMOTION_INTENSITY[emotion],
        delivery=EMOTION_DELIVERY[emotion],
        pause_before=0.0,
        pause_after=0.0,
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@runtime_checkable
class CompletionClient(Protocol):
    """One JSON-mode chat completion. Implemented by vLLM and by tests."""

    async def complete(self, *, system: str, user: str) -> str:
        """Return the raw assistant reply for one system/user pair."""


class VllmChatClient:
    """A JSON-mode client for a Qwen model served by vLLM.

    The client is deliberately thin: one request, no retries, no provider failover.
    vLLM's OpenAI-compatible endpoint is on loopback, so every failure here is a
    server-side fact the run should surface rather than paper over.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 300.0,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._api_key = (api_key or "").strip() or "EMPTY"
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client: Any | None = None

    def _openai(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - only without the client
            raise ConfigurationError(
                "the local dialogue model needs the 'openai' package for its "
                f"OpenAI-compatible client ({type(exc).__name__}: {exc})"
            ) from exc

        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            max_retries=0,
        )
        return self._client

    async def complete(self, *, system: str, user: str) -> str:
        client = self._openai()
        try:
            response = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - mapped to this module's errors
            raise _as_request_error(exc) from exc
        return _response_content(response)

    async def aclose(self) -> None:
        """Close the underlying HTTP client, when one was created."""

        client = self._client
        self._client = None
        if client is not None:
            await client.close()


def _as_request_error(exc: Exception) -> TranslationError:
    """Map an ``openai`` exception onto this module's error vocabulary."""

    name = type(exc).__name__
    if name in {"APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout"}:
        return ApiClientError(
            f"the local dialogue model could not be reached ({name}: {exc}); start the "
            "vLLM server before running the translation stage"
        )
    if name in {"AuthenticationError", "PermissionDeniedError"}:
        return ApiAuthenticationError(f"the local dialogue model rejected the token: {exc}")
    return ApiRequestError(f"the local dialogue model failed with {name}: {exc}")


def _response_content(response: Any) -> str:
    """Return the assistant text of a chat completion, or fail clearly."""

    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, Mapping):
        choices = response.get("choices")
    if not choices:
        raise MalformedResponseError("the model returned no choices")

    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, Mapping):
        message = first.get("message")
    content = getattr(message, "content", None)
    if content is None and isinstance(message, Mapping):
        content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise MalformedResponseError("the model returned an empty message")
    return content


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

#: The adaptation contract. Every rule here exists because a dub without it is wrong
#: in a way the rest of the pipeline cannot repair: a literal translation that cannot
#: be spoken in the time available, an emotion the engine cannot act on, a name that
#: does not match a voice-sample folder, or a sound tag that is read aloud as a word.
SYSTEM_PROMPT = """\
You are the dialogue adaptation engine of a professional Amharic movie dubbing studio.
Your output is spoken by a neural voice, not read as a subtitle. It must sound like
something a native Amharic speaker would actually say in that moment, and it must fit
the time the original performance occupied.

INPUT
You receive one JSON object:
{
  "script": [ {"line": 1, "character": "Jack", "text": "..."} ],
  "lines":  [ {"id": "dialogue_000001", "character": "SPEAKER_00",
               "start_timestamp": 12.34, "end_timestamp": 14.10,
               "duration_seconds": 1.76, "syllable_budget": 7, "text": "..."} ]
}
"script" is the ground-truth screenplay for this stretch of the film, in order. Its
lines are complete sentences and its parentheticals are physical stage directions.
"lines" are the transcriber's fragments, which may cut a sentence in half, repeat a
word, or merge two speakers. Align each fragment to the script before translating.

TASK - for every entry of "lines", produce exactly one entry of "lines" in your reply:
1. ALIGNMENT: match the fragment to the complete sentence(s) the script gives at that
   point in the exchange. Translate the complete thought, not the broken fragment.
2. CHARACTER: if the script names the speaker, that name; otherwise the name most
   consistent with the screenplay. Remove every screenplay extension - (O.S.), (V.O.),
   (O.C.), (CONT'D), (CONTINUED) - and write the name in strict title case with letters
   and digits only: "JACK (V.O.)" -> "Jack", "MARY-JANE (CONT'D)" -> "MaryJane".
   Never invent a character who is not in the screenplay.
3. AMHARIC: natural, spoken, performable Amharic in Fidel script. Never translate word
   for word. Keep the meaning, the subtext, the humor, the rudeness, the register and
   the relationship. Do not add information, do not explain a joke, do not sanitize.
   Everyday English loanwords that Amharic speakers really use stay, written in Fidel
   script. Names, places, brands and technical terms stay as they are.
4. TEMPORAL CONSTRAINT: duration_seconds is how long the original line lasted, and
   syllable_budget is roughly how many Fidel syllables fit in it. Write the Amharic as
   tight as the natural spoken rhythm allows - drop repeated information, asides,
   qualifiers and examples before you drop the point - and never exceed the budget
   when a shorter phrasing still carries the meaning. Never sacrifice the information
   the scene turns on: a name, a number, a threat, a promise or a reveal stays.
5. INLINE PHYSICAL SOUNDS: if the script marks a physical noise at a position in the
   line - (sighs), (breathes), (gasps), (laughs) - insert the matching tag INSIDE the
   Amharic text at that exact position. The only legal tags are [sigh], [breath],
   [gasp] and [laughter]. Never put an emotion, an adjective or a stage direction
   inside the text: the voice would speak it as words.
6. OVERALL EMOTION: judge the line's global tone from the exchange and choose exactly
   one value from this closed list: "neutral", "angry", "sad", "excited", "whisper",
   "fearful". No other value, no combinations, no descriptions.

OUTPUT
Reply with one JSON object and nothing else:
{
  "lines": [
    {
      "id": "dialogue_000001",
      "character": "Jack",
      "start_timestamp": 12.34,
      "end_timestamp": 14.10,
      "overall_emotion": "angry",
      "amharic_text_with_tags": "ጃክ አትጫወትብኝም። [sigh] ሁሉንም ነገር አየሁ።"
    }
  ]
}

RULES FOR THE JSON
- Exactly one entry per input line, in the same order, with the same "id".
- Copy "start_timestamp" and "end_timestamp" from the input UNCHANGED.
- "amharic_text_with_tags" must contain Fidel script and may contain only the four
  tags above.
- Do not add keys, commentary, markdown or code fences.
"""


def _script_excerpt(
    script: Sequence[ScriptLine],
    *,
    start: int,
    end: int,
    total: int,
) -> list[dict[str, Any]]:
    """Return the screenplay lines that correspond to a dialogue window.

    The screenplay has no timestamps, so its position is inferred from how far into the
    film the window sits. The excerpt is padded on both sides: the sentence the first
    fragment belongs to usually begins above it.
    """

    if not script or total <= 0:
        return []

    length = len(script)
    first = int(length * start / total) - SCRIPT_EXCERPT_PADDING
    last = int(length * end / total) + SCRIPT_EXCERPT_PADDING + 1
    first = max(0, min(first, length - 1))
    last = max(first + 1, min(length, last))
    return [
        {"line": line.index + 1, "character": line.character, "text": line.text}
        for line in script[first:last]
    ]


def _window_payload(
    lines: Sequence[TranscriptSegment | DialogueSentence],
    window: Window,
    *,
    script: Sequence[ScriptLine],
    settings: Settings,
) -> str:
    """Render one window as the user message the model receives."""

    payload_lines: list[dict[str, Any]] = []
    for position in range(window.start, window.end):
        line = lines[position]
        window_seconds = max(line.end - line.start, 0.0)
        budget: SyllableBudget = syllable_budget(
            max(window_seconds, 0.01),
            rate=settings.translation_syllables_per_second,
        )
        payload_lines.append(
            {
                "id": _dialogue_id(position),
                "character": line.speaker_id,
                "start_timestamp": round(line.start, 3),
                "end_timestamp": round(line.end, 3),
                "duration_seconds": round(window_seconds, 3),
                "syllable_budget": budget.syllables,
                "text": line.text,
            }
        )

    payload = {
        "film": {"dialogue_lines": len(lines)},
        "script": _script_excerpt(
            script, start=window.start, end=window.end, total=len(lines)
        ),
        "lines": payload_lines,
    }
    return json.dumps(payload, ensure_ascii=False)


def _window_system_prompt(base: str, bible_block: str) -> str:
    """Return the system prompt, with any continuity notes appended."""

    if not bible_block:
        return base
    return (
        f"{base}\n\nCONTINUITY NOTES (how these characters speak; follow them)\n"
        f"{bible_block}\n"
    )


# ---------------------------------------------------------------------------
# Running a window
# ---------------------------------------------------------------------------


async def _complete_window(
    client: CompletionClient,
    *,
    system: str,
    user: str,
    window: Window,
    ids: Sequence[str],
    lines: Sequence[TranscriptSegment | DialogueSentence],
    script_characters: Sequence[str],
) -> list[WindowRecord]:
    """Ask for one window, retrying once with the parser's complaint."""

    complaint = ""
    last_error: TranslationError | None = None
    for attempt in range(MAX_WINDOW_ATTEMPTS):
        message = user if not complaint else (
            f"{user}\n\nYour previous reply was rejected: {complaint}\n"
            "Reply again with the complete JSON object."
        )
        try:
            content = await client.complete(system=system, user=message)
            entries = parse_response(content, ids)
            records = [
                WindowRecord(
                    position=window.start + offset,
                    window=window,
                    dialogue=_record_to_dialogue(
                        entries[identifier],
                        identifier=identifier,
                        source=lines[window.start + offset],
                        script_characters=script_characters,
                    ),
                )
                for offset, identifier in enumerate(ids)
            ]
        except (MalformedResponseError, DialogueIdError, InvalidModelOutputError) as exc:
            last_error = exc
            complaint = str(exc)
            continue

        return records

    if last_error is None:  # pragma: no cover - the loop always sets it or returns
        raise MalformedResponseError(f"window {window.index} produced no answer")
    raise type(last_error)(
        f"window {window.index} ({window.start}-{window.end - 1}) failed after "
        f"{MAX_WINDOW_ATTEMPTS} attempt(s): {last_error}"
    ) from last_error


async def _run_windows(
    lines: Sequence[TranscriptSegment | DialogueSentence],
    windows: Sequence[Window],
    *,
    client: CompletionClient,
    settings: Settings,
    script: Sequence[ScriptLine],
    script_characters: Sequence[str],
    bible_block: str,
    on_progress: Callable[[str], None] | None,
) -> list[WindowRecord]:
    """Submit every window, bounded by the client-side concurrency ceiling."""

    semaphore = asyncio.Semaphore(settings.translation_max_concurrency)
    system = _window_system_prompt(SYSTEM_PROMPT, bible_block)

    if on_progress is not None:
        on_progress(
            f"  adaptation: {len(windows)} window(s) of <= "
            f"{settings.translation_window_lines} line(s), "
            f"{settings.translation_window_overlap}-line overlap, "
            f"{settings.translation_max_concurrency} concurrent"
        )

    completed = 0
    total = len(windows)
    lock = asyncio.Lock()

    async def one(window: Window) -> list[WindowRecord]:
        nonlocal completed
        async with semaphore:
            user = _window_payload(lines, window, script=script, settings=settings)
            records = await _complete_window(
                client,
                system=system,
                user=user,
                window=window,
                ids=[_dialogue_id(position) for position in range(window.start, window.end)],
                lines=lines,
                script_characters=script_characters,
            )
        if on_progress is not None:
            async with lock:
                completed += 1
                on_progress(
                    f"  adaptation: window {completed}/{total} done "
                    f"(lines {window.start + 1}-{window.end})"
                )
        return records

    results = await asyncio.gather(*(one(window) for window in windows))
    return [record for batch in results for record in batch]


def _client_for(settings: Settings) -> VllmChatClient:
    """Return the configured client for the local vLLM server."""

    if not isinstance(settings.translation_base_url, str) or not (
        settings.translation_base_url.strip()
    ):
        raise ConfigurationError("TRANSLATION_BASE_URL must name the local vLLM server")
    return VllmChatClient(
        base_url=settings.translation_base_url,
        model=settings.translation_model,
        api_key=settings.translation_api_key,
        temperature=settings.translation_temperature,
        max_tokens=settings.translation_max_new_tokens,
        timeout=settings.translation_timeout_seconds,
    )


def _validate_settings(settings: Settings) -> tuple[int, int, int]:
    """Return ``(window, overlap, concurrency)``, or say what is unusable."""

    window = settings.translation_window_lines
    overlap = settings.translation_window_overlap
    concurrency = settings.translation_max_concurrency
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ConfigurationError(
            f"TRANSLATION_WINDOW_LINES must be a positive integer, got {window!r}"
        )
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
        raise ConfigurationError(
            f"TRANSLATION_WINDOW_OVERLAP must be a non-negative integer, got {overlap!r}"
        )
    if overlap >= window:
        raise ConfigurationError(
            f"TRANSLATION_WINDOW_OVERLAP ({overlap}) must be smaller than "
            f"TRANSLATION_WINDOW_LINES ({window})"
        )
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ConfigurationError(
            f"TRANSLATION_MAX_CONCURRENCY must be a positive integer, got {concurrency!r}"
        )
    if not math.isfinite(settings.translation_temperature) or not (
        0.0 <= settings.translation_temperature <= 2.0
    ):
        raise ConfigurationError(
            "TRANSLATION_TEMPERATURE must be between 0.0 and 2.0, got "
            f"{settings.translation_temperature!r}"
        )
    return window, overlap, concurrency


def _script_character_names(script: Sequence[ScriptLine]) -> tuple[str, ...]:
    """Return the distinct character names of the screenplay, in order."""

    return tuple(dict.fromkeys(line.character for line in script))


async def adapt_dialogue_async(
    dialogue: Iterable[TranscriptSegment | DialogueSentence],
    *,
    script: Sequence[ScriptLine] | None = None,
    settings: Settings | None = None,
    client: CompletionClient | None = None,
    bible: Any | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[AdaptedDialogue, ...]:
    """Adapt every line to Amharic, concurrently, and return them in film order.

    Parameters
    ----------
    dialogue:
        Transcribed lines or assembled sentences, in chronological order. Their text is
        translated; their identity and timing are never changed by the model.
    script:
        The parsed screenplay. It supplies the character names and the complete
        sentences fragments are aligned to. Absent means the model works from the
        transcript alone.
    settings:
        Project settings override; the window size, concurrency and endpoint live here.
    client:
        A :class:`CompletionClient` to use instead of the configured vLLM server, which
        is how the stage is tested without a GPU.
    bible:
        Optional continuity state with an ``as_prompt_block(speaker_ids)`` method, as
        :class:`app.pipeline.dialogue_context.CharacterBible` provides.
    on_progress:
        Receives one line per window - planned, then completed.

    Raises
    ------
    InvalidTranscriptSegmentsError, InvalidSegmentError, ConfigurationError
    ApiClientError, ApiRequestError, ApiAuthenticationError
    MalformedResponseError, DialogueIdError, InvalidModelOutputError
    """

    lines = _validate_segments(dialogue)
    if not lines:
        return ()

    resolved = settings if settings is not None else get_settings()
    window_size, overlap, _concurrency = _validate_settings(resolved)

    script_lines = tuple(script or ())
    script_characters = _script_character_names(script_lines)
    bible_block = ""
    if bible is not None and hasattr(bible, "as_prompt_block"):
        try:
            bible_block = str(
                bible.as_prompt_block(
                    sorted({line.speaker_id for line in lines})
                )
            ).strip()
        except Exception:  # noqa: BLE001 - continuity notes are optional context
            bible_block = ""

    windows = overlapping_windows(len(lines), size=window_size, overlap=overlap)

    injected = client
    active = injected if injected is not None else _client_for(resolved)
    try:
        records = await _run_windows(
            lines,
            windows,
            client=active,
            settings=resolved,
            script=script_lines,
            script_characters=script_characters,
            bible_block=bible_block,
            on_progress=on_progress,
        )
    finally:
        if injected is None and isinstance(active, VllmChatClient):
            await active.aclose()

    return deduplicate_records(records, total=len(lines))


def _run_sync(coroutine: Any) -> Any:
    """Run ``coroutine`` to completion, even from inside a running event loop."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    # A notebook or a server already owns the loop. Running the pipeline's own loop in
    # a worker thread keeps the caller's loop free and still gives this stage one.
    import concurrent.futures
    import threading

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coroutine)
        return future.result()


def adapt_dialogue(
    dialogue: Iterable[TranscriptSegment | DialogueSentence],
    *,
    script: Sequence[ScriptLine] | None = None,
    settings: Settings | None = None,
    client: CompletionClient | None = None,
    bible: Any | None = None,
    on_progress: Callable[[str], None] | None = None,
    **legacy: Any,
) -> tuple[AdaptedDialogue, ...]:
    """Synchronous entry point for :func:`adapt_dialogue_async`.

    Legacy keyword arguments from the retired hosted backends are rejected loudly
    rather than ignored, so a stale caller cannot believe a setting still applies.
    """

    if legacy:
        names = ", ".join(sorted(legacy))
        raise ConfigurationError(
            f"adapt_dialogue no longer accepts {names}; dialogue adaptation runs on the "
            "local Qwen model through TRANSLATION_* settings, and the hosted backends "
            "and their rewrite passes were removed"
        )

    return _run_sync(
        adapt_dialogue_async(
            dialogue,
            script=script,
            settings=settings,
            client=client,
            bible=bible,
            on_progress=on_progress,
        )
    )


def describe_backend(settings: Settings | None = None) -> str:
    """Return one line saying which model, where, and how it is batched."""

    resolved = settings if settings is not None else get_settings()
    return (
        f"local vLLM {resolved.translation_model} at {resolved.translation_base_url} "
        f"({resolved.translation_window_lines}-line windows, "
        f"{resolved.translation_window_overlap}-line overlap, "
        f"{resolved.translation_max_concurrency} concurrent, "
        f"JSON mode)"
    )


__all__ = [
    "AdaptedDialogue",
    "ApiAuthenticationError",
    "ApiClientError",
    "ApiRequestError",
    "CompletionClient",
    "ConfigurationError",
    "DIALOGUE_ID_DIGITS",
    "DIALOGUE_ID_PREFIX",
    "DialogueIdError",
    "EMOTIONS",
    "EMOTION_DELIVERY",
    "EMOTION_INTENSITY",
    "INLINE_SOUND_TAGS",
    "InvalidModelOutputError",
    "InvalidSegmentError",
    "InvalidTranscriptSegmentsError",
    "MAX_WINDOW_ATTEMPTS",
    "MalformedResponseError",
    "PerformanceBeat",
    "SYSTEM_PROMPT",
    "TIMESTAMP_TOLERANCE",
    "TranslationError",
    "VllmChatClient",
    "Window",
    "WindowRecord",
    "adapt_dialogue",
    "adapt_dialogue_async",
    "deduplicate_records",
    "describe_backend",
    "normalize_emotion",
    "overlapping_windows",
    "parse_response",
    "sanitize_inline_tags",
]
