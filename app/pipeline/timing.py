"""Timing alignment for dubbed dialogue.

Each synthesized clip is placed against the *active vocal zone* of the line it
replaces, not against the diarizer's whole turn::

    original turn:  [ ... silence ... ][ actor speaking ][ ... silence ... ]
    synthesized:              [ Amharic speech ]      -> placed on the actor's zone
    delivered:      [ ... silence ... ][ Amharic ][ ... silence ... ]

* the **speech** is left at its natural cadence whenever it fits. Time-stretching is
  applied only when the line cannot fit even with the silence after it, and then only
  within the configured tempo band - a dub that is uniformly sped up is the artefact
  this stage exists to avoid;
* the window's own silence is **written around** the speech, so the line still lands on
  the picture even when the Amharic is shorter than the performance it replaces;
* a line that is still too long is **reported** as not fitting rather than mangled to
  fit, and **moves the next line later** instead of being spoken over it, so two
  voices are never heard at once unless the source put them there.

The result of this stage is one aligned WAV per line at the pipeline sample rate,
plus where to place it in the timeline, which is exactly what
:mod:`app.pipeline.mixing` consumes. Lines are produced one at a time by
:func:`iter_aligned`, so mixing can start consuming before the last line is fitted.

Deliberate non-goals
--------------------
* **No resampling for change of speed.** Speed is changed with FFmpeg's
  ``atempo``, which preserves pitch; resampling a clip to change its length would
  shift every voice up or down, which is never wanted.
* **No fitting across the tempo band.** Nothing here silently rewrites what the
  dialogue model asked for: a clamped stretch is recorded on the clip.
* **No padding *with* stretched speech.** A short line keeps its natural delivery; the
  gap that follows is silence, not stretched audio.
* **No touching of the clip files.** The clips remain the TTS stage's artifacts;
  the aligned audio is written beside them.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.config import Settings, get_settings
from app.pipeline.dialogue_context import MINIMUM_PLACEMENT_WINDOW
from app.pipeline.tts import TtsClip
from app.pipeline.voice_profiles import portable_path

#: Every stage after separation works at the pipeline's 48 kHz, so no stage has to
#: resample another stage's output. ``tests/test_timing.py`` keeps this in step
#: with ``separation.REQUIRED_SAMPLE_RATE`` and ``video.PIPELINE_SAMPLE_RATE``.
PIPELINE_SAMPLE_RATE = 48_000

#: Directory (under the work directory) holding the aligned clips.
TIMING_DIRECTORY_NAME = "timing"

#: Sub-directory of :data:`TIMING_DIRECTORY_NAME` holding the aligned lines.
ALIGNED_DIRECTORY_NAME = "aligned"

#: A residual this small is not worth a filter: it is below both the ear's sense
#: of lip-sync and FFmpeg's own rounding of a tempo change.
EXACT_FIT_TOLERANCE_SECONDS = 0.03

#: ``atempo`` only accepts factors inside this range, so a configuration outside
#: it is rejected instead of producing an unusable filter argument.
ATEMPO_MIN = 0.5
ATEMPO_MAX = 2.0

#: Length of the fade applied to the tail of a line that has to be cut short, in
#: seconds. Long enough that the cut is inaudible, short enough not to lose a syllable.
TRIM_FADE_SECONDS = 0.025

#: Below this frame level there is only noise, whatever the local floor says. Without
#: it a region of digital silence would make every frame look "active" against a floor
#: of minus infinity.
SILENCE_FLOOR_DBFS = -70.0

#: Resolution of the active-zone scan. Short enough to see the gaps between words,
#: long enough that a single sample cannot define the start of speech.
ACTIVE_ZONE_FRAME_SECONDS = 0.02


class TimingError(RuntimeError):
    """Base class for every error raised by this module."""


class ConfigurationError(TimingError, ValueError):
    """The tempo bounds contradict FFmpeg or each other."""


class InvalidClipError(TimingError, ValueError):
    """A clip is not the :class:`~app.pipeline.tts.TtsClip` this stage needs."""


class MissingInputError(TimingError):
    """A clip's audio file, or the configured output directory, is unusable."""


class MissingFfmpegError(TimingError):
    """FFmpeg is not on ``PATH``."""


class StretchError(TimingError):
    """FFmpeg failed to stretch a line, or produced something unusable."""


class InvalidAudioError(TimingError):
    """A clip, or a stretched result, is not usable audio."""


@dataclass(frozen=True, slots=True)
class AlignedClip:
    """One line, stretched to fit and placed on the source timeline.

    ``start`` is where the file begins, already accounting for the leading pause,
    so ``start`` is earlier than the original line's own start by exactly
    ``rendered_pause_before``. ``required_tempo`` is what an exact fit would have
    needed and ``tempo`` is what was applied: the two differ only for a line that
    did not fit, which is why both are kept. ``drift`` is how far the line had to
    move later than the original performance because the line before it ran long.
    """

    index: int
    clip: TtsClip
    audio_path: Path
    sample_rate: int
    start: float
    tempo: float
    required_tempo: float
    speech_duration: float
    original_window: float
    rendered_pause_before: float
    rendered_pause_after: float
    notes: tuple[str, ...] = ()
    #: Seconds of speech this line was allowed before it would run into the next one.
    #: Equal to ``original_window`` when the line was considered alone.
    available: float | None = None
    #: Seconds cut off the end, with a fade, because the line could not fit even at the
    #: tempo limit. Non-zero means the translation was longer than the time available.
    #: Stays zero unless ``TIMING_TRIM_TO_FIT`` is on.
    trimmed: float = 0.0
    #: Seconds this line runs past its available time. Non-zero means it will overlap the
    #: next line; it is reported rather than cut away.
    overrun: float = 0.0
    #: Seconds this line was moved later than the original performance started, because the
    #: line before it ran long. ``0.0`` means it landed exactly where the actor spoke.
    #: A drifted line is *late* rather than doubling another voice, which is the smaller
    #: cost: two voices at once is unintelligible, a late line is merely a little behind.
    drift: float = 0.0
    #: Absolute start and end of the *active vocal zone* inside the original window: the
    #: part of the actor's turn that actually contains speech, rather than the diarizer's
    #: turn boundaries. The dub is placed against this, not against the whole window.
    active_zone_start: float = 0.0
    active_zone_end: float = 0.0
    #: Silence padded before and after the fitted speech, in seconds. This is the time
    #: the original line spent not speaking; it is filled rather than stretched.
    silence_before: float = 0.0
    silence_after: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise InvalidClipError(
                f"index must be a non-negative integer, got {self.index!r}"
            )
        if not isinstance(self.clip, TtsClip):
            raise InvalidClipError(
                f"clip must be a TtsClip, got {type(self.clip).__name__}"
            )

        object.__setattr__(self, "audio_path", Path(self.audio_path))

        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise InvalidClipError(
                f"sample_rate must be an integer, got {self.sample_rate!r}"
            )
        if self.sample_rate < 1:
            raise InvalidClipError(
                f"sample_rate must be positive, got {self.sample_rate}"
            )

        for name in ("start", "tempo", "required_tempo", "rendered_pause_before",
                     "rendered_pause_after", "drift", "overrun", "trimmed"):
            value = getattr(self, name)
            number = _finite(name, value)
            if number < 0:
                raise InvalidClipError(f"{name} must be >= 0, got {number}")
            object.__setattr__(self, name, number)

        for name in ("speech_duration", "original_window"):
            number = _finite(name, getattr(self, name))
            if number <= 0:
                raise InvalidClipError(f"{name} must be positive, got {number}")
            object.__setattr__(self, name, number)

        for name in (
            "active_zone_start",
            "active_zone_end",
            "silence_before",
            "silence_after",
        ):
            number = _finite(name, getattr(self, name))
            if number < 0:
                raise InvalidClipError(f"{name} must be >= 0, got {number}")
            object.__setattr__(self, name, number)

        if isinstance(self.notes, list):
            object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def speaker_id(self) -> str:
        """The diarization speaker id this line belongs to."""

        return self.clip.speaker_id

    @property
    def residual(self) -> float:
        """How much longer than its window the delivered speech is, in seconds.

        Negative means the line underruns its window, positive means it overruns.
        """

        return self.speech_duration - self.original_window

    @property
    def fits(self) -> bool:
        """``True`` when the speech matches its window within the tolerance."""

        return abs(self.residual) <= EXACT_FIT_TOLERANCE_SECONDS

    @property
    def duration(self) -> float:
        """Length of the delivered file: speech plus its rendered pauses."""

        return (
            self.rendered_pause_before + self.speech_duration + self.rendered_pause_after
        )

    @property
    def end(self) -> float:
        """Where the delivered file stops on the source timeline."""

        return self.start + self.duration

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the aligned line."""

        return {
            "index": self.index,
            "speaker_id": self.speaker_id,
            "audio_path": portable_path(self.audio_path),
            "sample_rate": self.sample_rate,
            "start": self.start,
            "end": self.end,
            "original_start": self.clip.start,
            "original_end": self.clip.end,
            "original_window": self.original_window,
            "tempo": self.tempo,
            "required_tempo": self.required_tempo,
            "speech_duration": self.speech_duration,
            "residual": self.residual,
            "fits": self.fits,
            "available": self.available,
            "trimmed": self.trimmed,
            "overrun": self.overrun,
            "drift": self.drift,
            "active_zone_start": self.active_zone_start,
            "active_zone_end": self.active_zone_end,
            "silence_before": self.silence_before,
            "silence_after": self.silence_after,
            "rendered_pause_before": self.rendered_pause_before,
            "rendered_pause_after": self.rendered_pause_after,
            "duration": self.duration,
            "notes": list(self.notes),
        }


def _finite(name: str, value: Any) -> float:
    """Return ``value`` as a finite float, or raise :class:`InvalidClipError`."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidClipError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not np.isfinite(number):
        raise InvalidClipError(f"{name} must be a finite number, got {value!r}")
    return number


def _run_ffmpeg(arguments: list[str]) -> None:
    """Run ``ffmpeg`` with ``arguments`` and fail clearly when it does not work."""

    executable = shutil.which("ffmpeg")
    if executable is None:
        raise MissingFfmpegError(
            "ffmpeg was not found on PATH; aligning dubbed dialogue needs FFmpeg "
            "(see the runtime requirements in README.md)"
        )

    process = subprocess.run(  # noqa: S603 - argv list, no shell involved
        [executable, *arguments],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stderr or "").strip().splitlines()
        raise StretchError(
            f"ffmpeg failed to stretch a dubbed line (exit code "
            f"{process.returncode}): {detail[-1] if detail else 'no output'}"
        )


def _read_mono_file(path: Path, *, label: str) -> tuple[np.ndarray, int]:
    """Read an audio file into a mono float32 array, with its sample rate."""

    if not path.is_file():
        raise MissingInputError(f"{label} was not found: {path}")

    try:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:
        raise InvalidAudioError(f"could not read {label} at {path}: {exc}") from exc

    if data.size == 0 or data.shape[0] == 0:
        raise InvalidAudioError(f"{label} at {path} contains no samples")
    if sample_rate < 1:
        raise InvalidAudioError(f"{label} at {path} has an invalid sample rate")

    # The clips are mono, but a stereo one would still be usable: generated
    # dialogue is placed as a single centred channel, so channels are averaged.
    return np.ascontiguousarray(data.mean(axis=1, dtype=np.float32)), int(sample_rate)


def _write_mono(path: Path, samples: np.ndarray, sample_rate: int, *, label: str) -> Path:
    """Write ``samples`` as mono PCM-16 WAV and return the path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(
            str(path), np.asarray(samples, dtype=np.float32), sample_rate,
            format="WAV", subtype="PCM_16",
        )
    except (OSError, RuntimeError) as exc:
        raise StretchError(f"could not write the {label} to {path}: {exc}") from exc
    return path


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _speech_region(
    clip: TtsClip, samples: np.ndarray, sample_rate: int
) -> np.ndarray:
    """Return the line's speech, without its rendered pauses, from the clip file.

    The clip is ``[pause_before][speech][pause_after]``, so the speech is a slice
    at a known offset. A clip whose file is shorter than its metadata claims is
    used as far as it goes rather than rejected: the metadata comes from the TTS
    stage's own measurement of the same file, so any disagreement is a fraction
    of a sample.
    """

    first = int(round(clip.rendered_pause_before * sample_rate))
    count = int(round(clip.speech_duration * sample_rate))
    if count < 1:
        raise InvalidClipError(
            f"the clip for line {clip.index} has no speech to align "
            f"({clip.speech_duration:.4f}s)"
        )

    region = samples[first : first + count]
    if region.size == 0:
        raise InvalidClipError(
            f"the clip for line {clip.index} is shorter than its own leading pause"
        )
    return np.ascontiguousarray(region, dtype=np.float32)


def _stretched_speech(
    clip: TtsClip,
    region: np.ndarray,
    sample_rate: int,
    tempo: float,
    destination: Path,
) -> tuple[np.ndarray, int]:
    """Return the line's speech stretched by ``tempo``, at the pipeline rate.

    The speech is written out and read back through FFmpeg, which does both jobs
    in one pass: the tempo change (pitch-preserving, and only when one is needed)
    and the sample-rate conversion to the pipeline rate.
    """

    source = _write_mono(destination, region, sample_rate, label="clip speech")
    output = destination.with_name(f"{destination.stem}.stretched.wav")
    arguments = [
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
    ]
    if tempo != 1.0:
        # atempo is the pitch-preserving stretcher; the length of the result is
        # the input length divided by the factor.
        arguments += ["-af", f"atempo={tempo:.6f}"]
    arguments += [
        "-ar",
        str(PIPELINE_SAMPLE_RATE),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]

    try:
        _run_ffmpeg(arguments)
    finally:
        source.unlink(missing_ok=True)

    try:
        samples, rate = _read_mono_file(output, label="the stretched line")
    finally:
        output.unlink(missing_ok=True)

    if rate != PIPELINE_SAMPLE_RATE:
        raise InvalidAudioError(
            f"ffmpeg returned {rate} Hz for a stretched line, expected "
            f"{PIPELINE_SAMPLE_RATE} Hz"
        )
    if samples.size == 0:
        raise StretchError(f"stretching line {clip.index} produced no audio")
    return samples, rate


def _fade_out(
    samples: np.ndarray,
    sample_rate: int,
    seconds: float = TRIM_FADE_SECONDS,
) -> np.ndarray:
    """Return ``samples`` with its tail faded to silence.

    Used only when a line has to be cut short. Cutting mid-waveform is a click; a fade
    over a few tens of milliseconds is inaudible, and the speech that is kept is not
    altered at all.
    """

    count = min(int(round(seconds * sample_rate)), samples.shape[0])
    if count <= 1:
        return samples
    faded = np.array(samples, dtype=np.float32, copy=True)
    faded[-count:] *= np.linspace(1.0, 0.0, count, dtype=np.float32)
    return faded


def _audio_start(clip: TtsClip) -> float:
    """Return where a clip's file begins on the source timeline."""

    lead = min(clip.rendered_pause_before, clip.dialogue.start)
    return clip.dialogue.start - lead


def _region_samples(
    path: Path, start: float, end: float, *, label: str
) -> tuple[np.ndarray, int]:
    """Read only ``[start, end)`` of a long file, without loading the whole thing.

    A film's dialogue stem is far larger than memory, so the region is seeked to and
    read directly. ``soundfile`` handles the container; no FFmpeg pass is needed for a
    WAV stem.
    """

    try:
        with sf.SoundFile(str(path)) as handle:
            rate = int(handle.samplerate)
            first = max(0, int(round(start * rate)))
            count = max(0, int(round((end - start) * rate)))
            handle.seek(first)
            data = handle.read(frames=count, dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:
        raise InvalidAudioError(f"could not read {label} at {path}: {exc}") from exc

    if data.size == 0 or rate < 1:
        raise InvalidAudioError(f"{label} at {path} holds no samples in that window")
    return np.ascontiguousarray(data.mean(axis=1, dtype=np.float32)), rate


def detect_active_zone(
    path: Path,
    start: float,
    end: float,
    *,
    margin_db: float = 12.0,
    minimum_seconds: float = 0.08,
    frame_seconds: float = 0.02,
) -> tuple[float, float]:
    """Return the part of ``[start, end)`` that actually contains speech.

    A diarized turn is a *speaker's* interval, not the actor's first and last phoneme:
    the turn normally opens and closes with room tone. Fitting Amharic to the turn -
    which is what the stage used to do - stretches it into that silence and makes the
    cadence wrong. The active zone is found from the audio instead, with a burstiness
    test: a frame is active when it rises ``margin_db`` above the noise floor of the
    window itself, so steady music or hum does not read as speech.

    Falls back to the whole window when the region cannot be read or the zone it finds
    is implausibly short; a diagnostic must never break a run.
    """

    if end <= start:
        return start, end

    try:
        samples, rate = _region_samples(Path(path), start, end, label="the dialogue stem")
    except TimingError:
        return start, end

    frame = max(1, int(round(frame_seconds * rate)))
    usable = (samples.shape[0] // frame) * frame
    if usable < frame:
        return start, end
    frames = samples[:usable].reshape(-1, frame)

    energy = np.sqrt(np.mean(np.square(frames, dtype=np.float32), axis=1) + 1e-12)
    levels_db = 20.0 * np.log10(np.maximum(energy, 1e-6))

    floor_db = float(np.percentile(levels_db, 15.0))
    threshold = max(floor_db + float(margin_db), SILENCE_FLOOR_DBFS)
    active = np.flatnonzero(levels_db > threshold)
    if active.size == 0:
        return start, end

    first = int(active[0]) * frame_seconds
    last = (int(active[-1]) + 1) * frame_seconds
    if last - first < minimum_seconds:
        return start, end
    # Clamp to the requested window; the region was read for exactly that span.
    return start + first, min(end, start + last)


def align_clip(
    clip: TtsClip,
    destination: str | Path,
    *,
    settings: Settings | None = None,
    room: float | None = None,
    active_zone: tuple[float, float] | None = None,
) -> AlignedClip:
    """Place one clip inside its window, against the actor's active vocal zone.

    Parameters
    ----------
    clip:
        A clip from :func:`app.pipeline.tts.synthesize_dialogue`.
    destination:
        File to write the aligned line to. Created, with its parents, when missing.
    settings:
        Project settings override; the tempo band comes from here.
    room:
        How many seconds of speech this line may occupy before it would run into the
        next one, including any silence between them. ``None`` means "no more than its
        own window", which is what a single clip considered alone can know.
    active_zone:
        Absolute ``(start, end)`` of the speech inside the original window, from
        :func:`detect_active_zone`. ``None`` uses the whole window.

    Returns
    -------
    AlignedClip

    Notes
    -----
    **The speech is never stretched just to fill a window.** Time-stretching is what
    makes a dub sound sped up or slurred, and it was being applied because a diarized
    turn includes silence the actor never spoke into. The clip is placed at the start of
    the active zone, and the window's own silence is written around it. A line is
    compressed only when it cannot fit even with the silence after it, and then only
    within the configured tempo band; anything still too long is reported rather than
    mangled.
    """

    resolved = settings if settings is not None else get_settings()
    minimum, maximum = _tempo_bounds(resolved)

    if not isinstance(clip, TtsClip):
        raise InvalidClipError(f"expected a TtsClip, got {type(clip).__name__}")

    samples, sample_rate = _read_mono_file(
        Path(clip.audio_path), label=f"the clip for line {clip.index}"
    )
    region = _speech_region(clip, samples, sample_rate)

    window = clip.dialogue.duration
    speech = clip.speech_duration

    zone_start, zone_end = (
        active_zone if active_zone is not None else (clip.dialogue.start, clip.dialogue.end)
    )
    zone_start = max(clip.dialogue.start, min(zone_start, clip.dialogue.end))
    zone_end = max(zone_start, min(zone_end, clip.dialogue.end))

    # The room a line may use is never smaller than its own window: the window is where
    # the actor actually spoke, and no line should be squashed below the performance it
    # replaces just because the line in front of it ran long. ``room`` can come back
    # smaller than the window when the two original performances overlapped.
    limit = (
        window
        if room is None
        else max(window, float(room), MINIMUM_PLACEMENT_WINDOW)
    )

    required = speech / max(limit, MINIMUM_PLACEMENT_WINDOW)
    if speech <= limit + EXACT_FIT_TOLERANCE_SECONDS:
        # It fits somewhere in the time it has: leave the cadence alone entirely.
        tempo = 1.0
    else:
        tempo = _clamp(required, minimum, maximum)

    if tempo == 1.0:
        stretched = np.ascontiguousarray(region, dtype=np.float32)
        rate = sample_rate
        if rate != PIPELINE_SAMPLE_RATE:
            # Only the sample rate needs normalising; no tempo filter is applied.
            destination = Path(destination)
            temporary = destination.with_name(f"{destination.stem}.rate.wav")
            stretched, rate = _stretched_speech(
                clip, region, sample_rate, 1.0, temporary
            )
    else:
        stretched, rate = _stretched_speech(
            clip,
            region,
            sample_rate,
            tempo,
            Path(destination).with_name(f"{Path(destination).stem}.speech.wav"),
        )
    speech_duration = stretched.shape[0] / rate

    notes: list[str] = []
    trimmed = 0.0
    overrun = 0.0
    if speech_duration > limit + EXACT_FIT_TOLERANCE_SECONDS:
        overrun = speech_duration - limit
        if resolved.timing_trim_to_fit:
            # Cutting is opt-in, because it is the one outcome that damages the
            # performance: the listener hears a word clipped off the end. Two voices
            # briefly at once is the lesser evil, so an over-long line is reported and
            # left intact unless the run has asked for the guarantee.
            keep = int(round(limit * rate))
            if keep > 0:
                trimmed = speech_duration - keep / rate
                stretched = _fade_out(stretched[:keep], rate)
                speech_duration = keep / rate
                overrun = 0.0
                notes.append(
                    f"the line needed {required:.3f}x the {limit:.3f}s available but "
                    f"the tempo limit is {maximum:g}x; the last {trimmed:.3f}s were cut "
                    "with a fade so the next line is not spoken over - the Amharic is "
                    "longer than its time"
                )
        else:
            notes.append(
                f"the line is {overrun:.3f}s longer than the {limit:.3f}s it has before "
                "the next line and it is left intact; the next line is moved later "
                "instead of the two being spoken over each other - the Amharic is "
                "longer than its time"
            )

    # Placement: the speech starts where the actor's speech started. The file may begin
    # earlier by the line's own leading pause - that pause is silence, and it is the
    # performance's own entry - clipped so the file never starts before time zero.
    lead = min(max(clip.rendered_pause_before, 0.0), max(zone_start, 0.0))
    file_start = zone_start - lead

    # Everything the window has left after the speech is silence. Writing it out is what
    # keeps the dub in sync with the picture without touching the speech itself. A
    # pause the dialogue stage asked for is a *minimum*: it is silence too, and the
    # cascade only measures speech, so keeping it can never make two voices collide.
    speech_end = zone_start + speech_duration
    window_trail = max(0.0, clip.dialogue.end - speech_end)
    trail = max(window_trail, max(clip.rendered_pause_after, 0.0))
    silence_before = max(0.0, zone_start - clip.dialogue.start)
    silence_after = trail

    if zone_start > clip.dialogue.start + 1e-9:
        notes.append(
            f"the line is placed at the start of the actor's active speech "
            f"({zone_start:.3f}s, {silence_before:.3f}s into the window) and the "
            f"{silence_before:.3f}s before it is silence, not compressed audio"
        )
    if lead < clip.rendered_pause_before - 1e-9:
        notes.append(
            f"the {clip.rendered_pause_before:.3f}s leading pause was trimmed to "
            f"{lead:.3f}s so the line could keep its original start of "
            f"{clip.dialogue.start:.3f}s"
        )
    if tempo == 1.0:
        notes.append("the cadence is left untouched; no time-stretching was applied")
    elif trimmed == 0.0 and abs(speech_duration - window) > EXACT_FIT_TOLERANCE_SECONDS:
        notes.append(
            f"an exact fit needed a tempo of {required:.3f}, outside "
            f"[{minimum:g}, {maximum:g}]; clamped to {tempo:.3f}, so the line runs "
            f"{abs(speech_duration - window):.3f}s "
            f"{'long' if speech_duration > window else 'short'} of its window"
        )

    pieces = [stretched]
    if lead:
        pieces.insert(0, np.zeros(int(round(lead * rate)), dtype=np.float32))
    if trail:
        pieces.append(np.zeros(int(round(trail * rate)), dtype=np.float32))

    _write_mono(
        Path(destination),
        np.concatenate(pieces),
        rate,
        label="aligned line",
    )

    return AlignedClip(
        index=clip.index,
        clip=clip,
        audio_path=Path(destination),
        sample_rate=rate,
        start=file_start,
        tempo=tempo,
        required_tempo=required,
        speech_duration=speech_duration,
        original_window=window,
        rendered_pause_before=lead,
        rendered_pause_after=trail,
        notes=tuple(notes),
        available=limit,
        trimmed=trimmed,
        overrun=overrun,
        active_zone_start=zone_start,
        active_zone_end=zone_end,
        silence_before=silence_before,
        silence_after=silence_after,
    )


def _tempo_bounds(settings: Settings) -> tuple[float, float]:
    """Return the validated ``(minimum, maximum)`` tempo band."""

    minimum = _finite("TIMING_MIN_TEMPO", settings.timing_min_tempo)
    maximum = _finite("TIMING_MAX_TEMPO", settings.timing_max_tempo)

    if minimum < ATEMPO_MIN or maximum > ATEMPO_MAX:
        raise ConfigurationError(
            f"the tempo band [{minimum:g}, {maximum:g}] is outside the range "
            f"FFmpeg's atempo filter supports, [{ATEMPO_MIN:g}, {ATEMPO_MAX:g}]"
        )
    if minimum > maximum:
        raise ConfigurationError(
            f"TIMING_MIN_TEMPO ({minimum:g}) must not exceed TIMING_MAX_TEMPO "
            f"({maximum:g})"
        )
    return minimum, maximum


def resolve_timing_directory(
    output_dir: str | Path | None = None,
    *,
    settings: Settings | None = None,
) -> Path:
    """Return the directory that holds the aligned lines.

    An explicit ``output_dir`` wins, so a run can keep its artifacts together;
    otherwise the work directory is used.
    """

    resolved = settings if settings is not None else get_settings()
    root = Path(output_dir) if output_dir is not None else Path(resolved.work_dir)
    return root / TIMING_DIRECTORY_NAME / ALIGNED_DIRECTORY_NAME


def _rooms(ordered: list[TtsClip], guard: float) -> list[float | None]:
    """Return how long each line may speak before it would meet the next one."""

    rooms: list[float | None] = []
    for position, clip in enumerate(ordered):
        if position + 1 >= len(ordered):
            rooms.append(None)
            continue
        following = ordered[position + 1]
        deadline = _audio_start(following) - guard
        rooms.append(deadline - clip.dialogue.start - clip.rendered_pause_after)
    return rooms


def _place(
    clip: AlignedClip,
    *,
    previous: AlignedClip | None,
    previous_speech_end: float,
    guard: float,
    overlap: float,
) -> tuple[AlignedClip, float]:
    """Move one placed line later if the line before it is still sounding.

    Everything here is measured between *speech* boundaries rather than file boundaries.
    A clip carries its rendered pauses with it, and silence over silence is inaudible, so
    a line is not moved to avoid an overlap nobody can hear.
    """

    same_speaker = previous is not None and previous.speaker_id == clip.speaker_id
    allowed = 0.0 if same_speaker else overlap
    # The constraint is on speech, not the files: pauses are inaudible over silence.
    # Even genuine source crosstalk is capped so translated dialogue stays clear.
    floor = previous_speech_end + guard - clip.rendered_pause_before - allowed
    start = max(clip.start, floor)
    drift = start - clip.start
    if drift > 1e-6:
        why = (
            "the same character had not finished, and one voice cannot hand over to "
            "itself"
            if same_speaker
            else "the line before it ran past its own time"
        )
        clip = replace(
            clip,
            start=start,
            drift=drift,
            notes=clip.notes
            + (
                f"{why}, so this line starts {drift:.3f}s later rather than being "
                "spoken over",
            ),
        )
    speech_end = clip.start + clip.rendered_pause_before + clip.speech_duration
    return clip, speech_end


def iterate_aligned(
    clips: Iterable[TtsClip],
    *,
    output_dir: str | Path | None = None,
    settings: Settings | None = None,
    speech_stem: str | Path | None = None,
) -> Iterator[AlignedClip]:
    """Fit and place every clip, yielding each one as soon as it is written.

    Yielding rather than returning a list is what lets :mod:`app.pipeline.mixing` start
    consuming the dub while later lines are still being fitted. The room each line has
    and the placement of each line only depend on metadata and on the line before it, so
    nothing here needs the whole film in memory.

    Parameters
    ----------
    clips:
        Clips from :func:`app.pipeline.tts.synthesize_dialogue`, in order.
    output_dir:
        Directory to keep the aligned lines in. Defaults to
        ``<WORK_DIR>/timing/aligned``.
    settings:
        Project settings override; defaults to :func:`app.config.get_settings`.
    speech_stem:
        The original clean speech stem. When supplied, the *active vocal zone* of each
        original line is measured from it, so the dub is aligned to the actor's speech
        rather than to the diarizer's turn boundaries. Absent means the whole window is
        used, which is the old behaviour.

    Yields
    ------
    AlignedClip
        One per input clip, in order, never overlapping another line more than the
        configured hand-over allowance.
    """

    resolved = settings if settings is not None else get_settings()
    _tempo_bounds(resolved)

    ordered = list(clips)
    if not ordered:
        return

    directory = resolve_timing_directory(output_dir, settings=resolved)
    directory.mkdir(parents=True, exist_ok=True)

    guard = resolved.timing_min_line_gap
    rooms = _rooms(ordered, guard)

    previous: AlignedClip | None = None
    previous_speech_end = 0.0
    for clip, room in zip(ordered, rooms):
        if not isinstance(clip, TtsClip):
            raise InvalidClipError(f"expected a TtsClip, got {type(clip).__name__}")

        active_zone: tuple[float, float] | None = None
        if speech_stem is not None:
            active_zone = detect_active_zone(
                Path(speech_stem),
                clip.dialogue.start,
                clip.dialogue.end,
                margin_db=resolved.timing_active_zone_margin_db,
                minimum_seconds=resolved.timing_active_zone_min_seconds,
                frame_seconds=ACTIVE_ZONE_FRAME_SECONDS,
            )

        destination = directory / f"{Path(clip.audio_path).stem}.wav"
        aligned = align_clip(
            clip,
            destination,
            settings=resolved,
            room=room,
            active_zone=active_zone,
        )
        aligned, previous_speech_end = _place(
            aligned,
            previous=previous,
            previous_speech_end=previous_speech_end,
            guard=guard,
            overlap=resolved.timing_max_overlap_seconds,
        )
        previous = aligned
        yield aligned


async def align_dialogue_stream(
    clips: Iterable[TtsClip],
    *,
    output_dir: str | Path | None = None,
    settings: Settings | None = None,
    speech_stem: str | Path | None = None,
) -> AsyncIterator[AlignedClip]:
    """Async view of :func:`iterate_aligned`, for streaming into the mixing stage.

    The fitting work is blocking CPU/FFmpeg work, so each line is produced on a worker
    thread and yielded to the caller as soon as it exists. Stage 9 can therefore start
    mixing while the last lines are still being fitted.
    """

    iterator = iterate_aligned(
        clips, output_dir=output_dir, settings=settings, speech_stem=speech_stem
    )

    while True:
        try:
            item = await asyncio.to_thread(next, iterator)
        except StopIteration:
            return
        yield item


def align_dialogue(
    clips: list[TtsClip] | tuple[TtsClip, ...],
    *,
    output_dir: str | Path | None = None,
    settings: Settings | None = None,
    speech_stem: str | Path | None = None,
) -> list[AlignedClip]:
    """Fit every clip to its original window, in the order it was given.

    Parameters
    ----------
    clips:
        Clips from :func:`app.pipeline.tts.synthesize_dialogue`.
    output_dir:
        Directory to keep the aligned lines in. Defaults to
        ``<WORK_DIR>/timing/aligned``.
    settings:
        Project settings override; defaults to :func:`app.config.get_settings`.
    speech_stem:
        The original clean speech stem, used to measure each line's active vocal zone.

    Returns
    -------
    list[AlignedClip]
        One entry per input clip, in the same order. An empty input returns an
        empty list without touching FFmpeg.

    Raises
    ------
    ConfigurationError
        The tempo band is not usable.
    InvalidClipError
        An entry is not a :class:`~app.pipeline.tts.TtsClip`.
    MissingInputError
        A clip's audio file does not exist.
    MissingFfmpegError, StretchError, InvalidAudioError
        FFmpeg is missing, failed, or produced audio that is not the pipeline
        format.
    """

    return list(
        iterate_aligned(
            clips,
            output_dir=output_dir,
            settings=settings,
            speech_stem=speech_stem,
        )
    )


__all__ = [
    "ACTIVE_ZONE_FRAME_SECONDS",
    "ALIGNED_DIRECTORY_NAME",
    "ATEMPO_MAX",
    "ATEMPO_MIN",
    "EXACT_FIT_TOLERANCE_SECONDS",
    "PIPELINE_SAMPLE_RATE",
    "SILENCE_FLOOR_DBFS",
    "TIMING_DIRECTORY_NAME",
    "AlignedClip",
    "ConfigurationError",
    "InvalidAudioError",
    "InvalidClipError",
    "MissingFfmpegError",
    "MissingInputError",
    "StretchError",
    "TRIM_FADE_SECONDS",
    "TimingError",
    "align_clip",
    "align_dialogue",
    "align_dialogue_stream",
    "detect_active_zone",
    "iterate_aligned",
    "resolve_timing_directory",
]
