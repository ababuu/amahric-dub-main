"""Quality control for a dubbed run: measure it, do not just listen to it.

This module answers one question about a run's output - *how good is it, and
where exactly is it worst?* - so that a change to any stage can be judged instead
of guessed at. It sits beside the pipeline rather than in it, consuming what the
stages produced::

    [AlignedClip] + [AdaptedDialogue] + [CrosstalkRegion]
        -> build_qc_report() -> QcReport

What it measures, and why each one
----------------------------------
* **Duration fit.** The dominant complaint about machine-dubbed audio in the one
  large human study of professional dubbing is an unnatural speaking rate - "too
  slow, too fast, or too uneven" - so the report carries the distribution of fit
  residuals and tempo factors, not a single average that would hide a bad tail.
* **Speaking rate, in syllables per second.** One Fidel character is one syllable,
  which is what the script encodes, so syllable counts are available without a
  phonemiser or a grapheme-to-phoneme model - neither of which exists for Amharic.
  A rate outside the plausible band for speech is a line no performer could
  deliver, however well it "fits".
* **Crosstalk.** Simultaneous speech is exactly what exclusive diarization cannot
  describe, so it is reported here as a known uncertainty rather than discovered
  later by listening.
* **Pronunciation (opt-in).** A round trip through an Amharic ASR model gives a
  character error rate against the text that was synthesized, which is the only
  automated check that the dub is *intelligible* rather than merely present. It
  needs a model, so it is an injected callable and is simply absent when no model
  is available - reported as not measured, never as a pass.

Deliberate non-goals
--------------------
* **No models of its own.** Nothing here imports torch, ASR or an embedding
  model. Pronunciation and speaker similarity are injected callables, so the whole
  model-free report runs anywhere, including in the test suite.
* **No thresholds that fail a run.** The report states what was measured. Deciding
  what is good enough is a project decision, not a module's.
* **No audio decoding.** Everything numeric comes from the metadata the stages
  already produced; only an injected transcriber reads a file.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any

from app.pipeline.amharic_text import (
    HOMOPHONE_FAMILIES,
    SYLLABLE_RANGES,
    InvalidTextError,
    count_syllables,
    has_latin,
    has_pronounceable_text,
    latin_spans,
    syllable_sequence,
)
from app.pipeline.diarization import CrosstalkRegion
from app.pipeline.timing import EXACT_FIT_TOLERANCE_SECONDS, AlignedClip
from app.pipeline.translation import AdaptedDialogue
from app.pipeline.tts import TtsClip

#: A line whose residual is inside this fraction of its window is called a *close*
#: fit. Ten percent is the tolerance the dubbing literature quotes when it reports
#: "how many lines are within tolerance", so the figure is comparable to published
#: work rather than invented here.
CLOSE_FIT_RATIO = 0.10

#: A spoken delivery lives in roughly this band. Outside it a line is either
#: impossible to perform or unnaturally drawn out, so those lines are counted and
#: listed instead of being averaged away. The band is deliberately wide: it exists
#: to catch the pathological, not to grade ordinary variation.
PLAUSIBLE_SYLLABLES_PER_SECOND = (1.5, 9.0)

#: The name this module has always exposed for the comparison form of a string.
normalise_for_comparison = syllable_sequence

#: How far a generated line's median pitch may sit from its character's reference before
#: the clone is called into question. Translation and a new language move pitch a little;
#: a quarter of the reference's own pitch is far beyond that and is what a clone that
#: never took the reference looks like.
VOICE_PITCH_DRIFT = 0.25

#: Two references at least this far apart are two different voices. Ten percent is about
#: the distance between two adult voices of the same sex that a listener would still call
#: different people.
DISTINCT_REFERENCE_PITCH = 0.10

#: Two characters whose generated voices are closer than this, while their references
#: were further apart, were spoken by one voice.
COLLAPSED_VOICE_PITCH = 0.04

#: Longest analysis window per line, in seconds: enough to measure a pitch, short enough
#: that a film's worth of lines stays cheap.
_PITCH_WINDOW_SECONDS = 1.2

#: Analysis frame and hop for the pitch tracker, in seconds.
_PITCH_FRAME_SECONDS = 0.04
_PITCH_HOP_SECONDS = 0.02


class QcError(RuntimeError):
    """Base class for every error raised by this module."""


class InvalidMeasurementError(QcError, ValueError):
    """An injected measurement returned something unusable."""


def character_error_rate(expected: str, actual: str) -> float:
    """Return the character error rate between two syllable sequences.

    The rate is the Levenshtein distance divided by the length of ``expected``, so
    ``0.0`` is an exact match and ``1.0`` means as many edits as there were
    characters. An empty ``expected`` has no rate and raises.
    """

    source = syllable_sequence(expected)
    target = syllable_sequence(actual)
    if not source:
        raise InvalidMeasurementError(
            "a character error rate needs expected text with at least one syllable"
        )
    if not target and source:
        return 1.0

    previous = list(range(len(target) + 1))
    for row, source_character in enumerate(source, start=1):
        current = [row]
        for column, target_character in enumerate(target, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (source_character != target_character),
                )
            )
        previous = current

    return previous[-1] / len(source)


@dataclass(frozen=True, slots=True)
class LineMetrics:
    """What quality control can say about one delivered line.

    Every value comes from the metadata the earlier stages already recorded, so
    building one costs no audio decoding and no model.
    """

    index: int
    speaker_id: str
    amharic: str
    syllables: int
    original_window: float
    speech_duration: float
    tempo: float
    required_tempo: float
    residual: float
    notes: tuple[str, ...] = ()

    @property
    def speaker(self) -> str:
        """Alias for the diarized speaker id."""

        return self.speaker_id

    @property
    def fits(self) -> bool:
        """``True`` when the delivered speech matches its window exactly."""

        return abs(self.residual) <= EXACT_FIT_TOLERANCE_SECONDS

    @property
    def residual_ratio(self) -> float:
        """The residual as a fraction of the window, or ``0.0`` for an empty window."""

        if self.original_window <= 0:
            return 0.0
        return self.residual / self.original_window

    @property
    def close_fit(self) -> bool:
        """``True`` when the line landed within :data:`CLOSE_FIT_RATIO` of its window."""

        return abs(self.residual_ratio) <= CLOSE_FIT_RATIO

    @property
    def syllables_per_second(self) -> float:
        """How fast the line is *delivered*, in syllables per second."""

        if self.speech_duration <= 0:
            return 0.0
        return self.syllables / self.speech_duration

    @property
    def natural_speech_duration(self) -> float:
        """How long the line would have run before it was stretched to fit.

        Dividing the applied tempo back out recovers the delivery the engine
        produced, whatever the fitter did to it afterwards - which is what makes the
        natural speaking rate measurable from a run that had to stretch its lines.
        """

        return self.speech_duration * self.tempo

    @property
    def natural_syllables_per_second(self) -> float:
        """The rate the engine actually speaks at, before any stretching.

        This is the figure the syllable budget should be computed from. Computing it
        from the delivered duration instead would read a stretched line as evidence
        that the engine speaks slower than it does, and the error compounds.
        """

        natural = self.natural_speech_duration
        if natural <= 0:
            return 0.0
        return self.syllables / natural

    @property
    def plausible_rate(self) -> bool:
        """``True`` when the delivery rate is inside the plausible band."""

        low, high = PLAUSIBLE_SYLLABLES_PER_SECOND
        return low <= self.syllables_per_second <= high

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of this line's measurements."""

        return {
            "index": self.index,
            "speaker_id": self.speaker_id,
            "syllables": self.syllables,
            "original_window": round(self.original_window, 3),
            "speech_duration": round(self.speech_duration, 3),
            "tempo": round(self.tempo, 4),
            "required_tempo": round(self.required_tempo, 4),
            "residual": round(self.residual, 3),
            "residual_ratio": round(self.residual_ratio, 4),
            "fits": self.fits,
            "close_fit": self.close_fit,
            "syllables_per_second": round(self.syllables_per_second, 3),
            "natural_syllables_per_second": round(
                self.natural_syllables_per_second, 3
            ),
            "plausible_rate": self.plausible_rate,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class DurationFitReport:
    """The distribution of how well the delivered lines fit their windows."""

    lines: int
    fitted: int
    close_fits: int
    unfitted: int
    stretched: int
    implausible_rate: int
    mean_absolute_residual: float
    worst_residual: float
    mean_tempo: float
    slowest_tempo: float
    fastest_tempo: float

    @property
    def close_fit_ratio(self) -> float:
        """Fraction of lines inside :data:`CLOSE_FIT_RATIO` of their window."""

        if not self.lines:
            return 0.0
        return self.close_fits / self.lines

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the distribution."""

        return {
            "lines": self.lines,
            "fitted": self.fitted,
            "close_fits": self.close_fits,
            "close_fit_ratio": round(self.close_fit_ratio, 4),
            "unfitted": self.unfitted,
            "stretched": self.stretched,
            "implausible_rate": self.implausible_rate,
            "mean_absolute_residual": round(self.mean_absolute_residual, 3),
            "worst_residual": round(self.worst_residual, 3),
            "mean_tempo": round(self.mean_tempo, 4),
            "slowest_tempo": round(self.slowest_tempo, 4),
            "fastest_tempo": round(self.fastest_tempo, 4),
            "close_fit_tolerance": CLOSE_FIT_RATIO,
            "exact_fit_tolerance_seconds": EXACT_FIT_TOLERANCE_SECONDS,
        }


@dataclass(frozen=True, slots=True)
class PronunciationReport:
    """How intelligible the dubbed lines are, judged by an ASR round trip."""

    measured: int
    unmeasured: int
    mean_cer: float
    worst_cer: float
    worst_index: int | None
    model: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the measurement."""

        return {
            "measured": self.measured,
            "unmeasured": self.unmeasured,
            "mean_cer": round(self.mean_cer, 4),
            "worst_cer": round(self.worst_cer, 4),
            "worst_index": self.worst_index,
            "model": self.model,
        }


@dataclass(frozen=True, slots=True)
class VoiceIdentityReport:
    """Whether each character actually sounds like the actor they were cloned from.

    Everything upstream can be right about *who* is speaking and still produce a dub in
    which every character shares one voice: a cloning engine that ignores its reference
    does exactly that, and nothing else in the pipeline looks at the audio it returns.
    This compares the pitch of a character's generated lines with the pitch of the
    reference they were cloned from - the pipeline already estimates that for every
    profile - and with each other, so the failure is reported instead of heard.
    """

    #: Median pitch of the generated lines, per speaker. ``None`` when unmeasurable.
    generated_pitch: dict[str, float | None]
    #: The reference pitch each character was cloned from.
    reference_pitch: dict[str, float]
    #: Speakers whose generated pitch is far from their own reference.
    drifted: tuple[str, ...] = ()
    #: Pairs of characters with different references whose generated voices match.
    collapsed: tuple[tuple[str, str], ...] = ()

    @property
    def ok(self) -> bool:
        """``True`` when every character sounds like the actor they were cloned from."""

        return not self.drifted and not self.collapsed

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the measurement."""

        return {
            "generated_pitch_hz": {
                speaker_id: (
                    None if pitch is None else round(pitch, 1)
                )
                for speaker_id, pitch in sorted(self.generated_pitch.items())
            },
            "reference_pitch_hz": {
                speaker_id: round(pitch, 1)
                for speaker_id, pitch in sorted(self.reference_pitch.items())
            },
            "drifted": list(self.drifted),
            "collapsed": [list(pair) for pair in self.collapsed],
        }

    def summary(self) -> str:
        """Return the one-line summary a run prints."""

        if not self.generated_pitch:
            return "voice identity not measured (no generated audio)"
        measured = sum(
            1 for pitch in self.generated_pitch.values() if pitch is not None
        )
        if self.ok:
            return (
                f"{measured} character(s) match the pitch of the actor they were cloned "
                "from, and no two of them share a voice"
            )
        parts: list[str] = []
        if self.collapsed:
            shown = ", ".join(
                f"{first} and {second}" for first, second in self.collapsed
            )
            parts.append(
                f"{shown} have different reference voices but were generated with the "
                "same one - their lines sound like one character"
            )
        if self.drifted:
            parts.append(
                f"{', '.join(self.drifted)} do not sound like the reference they were "
                "cloned from"
            )
        return "; ".join(parts)


@dataclass(frozen=True, slots=True)
class QcReport:
    """Everything quality control could say about one run."""

    duration: DurationFitReport
    lines: tuple[LineMetrics, ...]
    crosstalk_regions: int
    crosstalk_seconds: float
    pronunciation: PronunciationReport | None = None
    #: Line indexes whose time is shared with simultaneous speech, where the speaker
    #: attribution cannot be checked from timing alone. The claim "this character said
    #: this" rests on the diarized turn, and a turn that overlaps another turn is one the
    #: exclusive view had to arbitrate. Reported rather than acted on: a dub must still
    #: say the line, and dropping it would leave a hole in the dialogue.
    attribution_uncertain: tuple[int, ...] = ()
    #: Whether the generated voices are the actors' own, measured from the audio itself.
    voice_identity: VoiceIdentityReport | None = None

    @property
    def worst_lines(self) -> tuple[LineMetrics, ...]:
        """The lines furthest from their window, worst first."""

        return tuple(
            sorted(self.lines, key=lambda line: abs(line.residual), reverse=True)
        )

    @property
    def implausible_lines(self) -> tuple[LineMetrics, ...]:
        """Lines whose delivery rate falls outside the plausible band."""

        return tuple(line for line in self.lines if not line.plausible_rate)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the whole report."""

        return {
            "duration": self.duration.as_dict(),
            "pronunciation": (
                None if self.pronunciation is None else self.pronunciation.as_dict()
            ),
            "crosstalk": {
                "regions": self.crosstalk_regions,
                "seconds": round(self.crosstalk_seconds, 3),
                "attribution_uncertain": list(self.attribution_uncertain),
            },
            "lines": [line.as_dict() for line in self.lines],
            "voice_identity": (
                None if self.voice_identity is None else self.voice_identity.as_dict()
            ),
        }

    def summary(self) -> str:
        """Return the one-line summary a run prints."""

        duration = self.duration
        parts = [
            f"{duration.close_fits}/{duration.lines} line(s) within "
            f"{CLOSE_FIT_RATIO:.0%} of their window",
        ]
        if duration.unfitted:
            parts.append(f"{duration.unfitted} not fitted")
        if duration.implausible_rate:
            parts.append(f"{duration.implausible_rate} at an implausible rate")
        if duration.stretched:
            parts.append(f"{duration.stretched} stretched")
        if self.crosstalk_regions:
            parts.append(
                f"{self.crosstalk_regions} crosstalk region(s) "
                f"({self.crosstalk_seconds:.2f}s)"
            )
        if self.attribution_uncertain:
            # Named, not merely counted: these are the lines where the dub's claim about
            # who is speaking comes from a turn the diarizer had to arbitrate, so they
            # are the first place to listen when a character sounds wrong.
            shown = ", ".join(str(index) for index in self.attribution_uncertain[:8])
            if len(self.attribution_uncertain) > 8:
                shown += f", and {len(self.attribution_uncertain) - 8} more"
            parts.append(
                f"{len(self.attribution_uncertain)} line(s) inside simultaneous speech "
                f"whose speaker cannot be verified from timing alone ({shown})"
            )
        if self.pronunciation is None:
            parts.append("pronunciation not measured")
        else:
            parts.append(
                f"pronunciation CER {self.pronunciation.mean_cer:.3f} over "
                f"{self.pronunciation.measured} line(s)"
            )
        if self.voice_identity is not None:
            parts.append(f"voice identity {self.voice_identity.summary()}")
        return ", ".join(parts)


def measure_lines(alignment: Iterable[AlignedClip]) -> tuple[LineMetrics, ...]:
    """Measure every aligned line, in the order they were aligned."""

    metrics: list[LineMetrics] = []
    for aligned in alignment:
        dialogue = aligned.clip.dialogue
        metrics.append(
            LineMetrics(
                index=aligned.index,
                speaker_id=dialogue.speaker_id,
                amharic=dialogue.amharic,
                syllables=count_syllables(dialogue.amharic),
                original_window=aligned.original_window,
                speech_duration=aligned.speech_duration,
                tempo=aligned.tempo,
                required_tempo=aligned.required_tempo,
                residual=aligned.residual,
                notes=tuple(aligned.notes),
            )
        )
    return tuple(metrics)


def summarise_duration(lines: Iterable[LineMetrics]) -> DurationFitReport:
    """Summarise the fit of the measured lines."""

    measured = tuple(lines)
    if not measured:
        return DurationFitReport(
            lines=0,
            fitted=0,
            close_fits=0,
            unfitted=0,
            stretched=0,
            implausible_rate=0,
            mean_absolute_residual=0.0,
            worst_residual=0.0,
            mean_tempo=0.0,
            slowest_tempo=0.0,
            fastest_tempo=0.0,
        )

    residuals = [line.residual for line in measured]
    tempos = [line.tempo for line in measured]
    return DurationFitReport(
        lines=len(measured),
        fitted=sum(1 for line in measured if line.fits),
        close_fits=sum(1 for line in measured if line.close_fit),
        unfitted=sum(1 for line in measured if not line.fits),
        stretched=sum(1 for line in measured if line.tempo != 1.0),
        implausible_rate=sum(1 for line in measured if not line.plausible_rate),
        mean_absolute_residual=fmean(abs(residual) for residual in residuals),
        worst_residual=max(residuals, key=abs),
        mean_tempo=fmean(tempos),
        slowest_tempo=min(tempos),
        fastest_tempo=max(tempos),
    )


def measure_pronunciation(
    dialogue: Iterable[AdaptedDialogue],
    clips: Iterable[TtsClip],
    *,
    transcribe: Callable[[Path], str],
    model: str = "unknown",
) -> PronunciationReport:
    """Measure the round-trip character error rate of the delivered lines.

    ``transcribe`` is handed the path of a *converted* clip - the audio that was
    actually voiced - and must return what it heard. Injecting it keeps this
    module free of any model dependency: an Amharic ASR model in production, a
    fixed string in a test.

    Clips are matched to dialogue by index. A clip that cannot be measured - a
    transcriber that fails, or an empty transcript - is counted as unmeasured
    rather than as an error, so one bad line cannot make the run look better or
    worse than it is.
    """

    expected = dict(enumerate(dialogue))
    rates: list[tuple[int, float]] = []
    unmeasured = 0

    for clip in clips:
        index = clip.index
        line = expected.get(index)
        if line is None:
            unmeasured += 1
            continue

        path = Path(clip.audio_path)
        try:
            heard = transcribe(path)
        except Exception:  # a measurement must never fail a run
            unmeasured += 1
            continue

        if not isinstance(heard, str):
            raise InvalidMeasurementError(
                f"the transcriber returned {type(heard).__name__} for line "
                f"{index}, expected the transcript as a string"
            )
        if not heard.strip():
            unmeasured += 1
            continue

        rates.append((index, character_error_rate(line.amharic, heard)))

    if not rates:
        return PronunciationReport(
            measured=0,
            unmeasured=unmeasured,
            mean_cer=0.0,
            worst_cer=0.0,
            worst_index=None,
            model=model,
        )

    worst_index, worst = max(rates, key=lambda item: item[1])
    return PronunciationReport(
        measured=len(rates),
        unmeasured=unmeasured,
        mean_cer=fmean(rate for _, rate in rates),
        worst_cer=worst,
        worst_index=worst_index,
        model=model,
    )


def measure_voice_identity(
    clips: Iterable[TtsClip],
    *,
    reference_pitch: dict[str, float],
) -> VoiceIdentityReport:
    """Compare the generated voices with the references they were cloned from.

    Pitch is measured on the generated audio itself, not taken from the engine's own
    metadata: the question is what the dub sounds like. It is the one identity cue that
    needs no model and no GPU, which is why it is the check that can run on every film.

    Two failures are reported. A character whose generated pitch is far from its own
    reference was not cloned from it. Two characters whose references are far apart but
    whose generated voices match were cloned into one voice - the failure a listener
    hears as "everyone sounds the same".
    """

    pitches: dict[str, list[float]] = {}
    for clip in clips:
        speaker_id = clip.clip.speaker_id if hasattr(clip, "clip") else clip.speaker_id
        pitch = _median_pitch(clip.audio_path)
        if pitch is None:
            continue
        pitches.setdefault(speaker_id, []).append(pitch)

    generated: dict[str, float | None] = {
        speaker_id: (float(median(values)) if values else None)
        for speaker_id, values in pitches.items()
    }
    for speaker_id in reference_pitch:
        generated.setdefault(speaker_id, None)

    drifted: list[str] = []
    for speaker_id, pitch in generated.items():
        reference = reference_pitch.get(speaker_id)
        if pitch is None or reference is None or reference <= 0:
            continue
        if abs(pitch - reference) / reference > VOICE_PITCH_DRIFT:
            drifted.append(speaker_id)

    collapsed: list[tuple[str, str]] = []
    identifiers = sorted(generated)
    for position, first in enumerate(identifiers):
        for second in identifiers[position + 1 :]:
            first_pitch = generated[first]
            second_pitch = generated[second]
            first_reference = reference_pitch.get(first)
            second_reference = reference_pitch.get(second)
            if None in (first_pitch, second_pitch):
                continue
            if not first_reference or not second_reference:
                continue
            reference_gap = abs(first_reference - second_reference) / min(
                first_reference, second_reference
            )
            generated_gap = abs(first_pitch - second_pitch) / min(
                first_pitch, second_pitch
            )
            if (
                reference_gap >= DISTINCT_REFERENCE_PITCH
                and generated_gap < COLLAPSED_VOICE_PITCH
            ):
                collapsed.append((first, second))

    return VoiceIdentityReport(
        generated_pitch=generated,
        reference_pitch=dict(reference_pitch),
        drifted=tuple(drifted),
        collapsed=tuple(collapsed),
    )


def _median_pitch(path: str | Path) -> float | None:
    """Return the median fundamental frequency of one generated line, in Hz.

    A plain autocorrelation tracker, chosen because it needs nothing but numpy: the
    measurement's job is to notice a voice that is not the one it should be, and a
    gross error in pitch is exactly that. Frames below the line's own loudness are
    ignored, so silence and pause padding do not enter the median.
    """

    import numpy as np
    import soundfile as sf

    try:
        samples, rate = sf.read(str(path), always_2d=False)
    except (OSError, RuntimeError):
        return None
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = np.asarray(samples, dtype=np.float64)
    if rate <= 0 or samples.size < rate * 0.1:
        return None

    frame = int(_PITCH_FRAME_SECONDS * rate)
    hop = int(_PITCH_HOP_SECONDS * rate)
    limit = int(_PITCH_WINDOW_SECONDS * rate)
    window = samples[: min(samples.size, limit)]
    if window.size < frame * 2:
        return None

    lowest = max(2, int(rate / 400.0))
    highest = min(frame - 1, int(rate / 60.0))
    if highest <= lowest:
        return None

    measured: list[float] = []
    for start in range(0, window.size - frame, hop):
        piece = window[start : start + frame]
        if float(np.sqrt(np.mean(piece**2))) < 0.01:
            continue
        centred = piece - piece.mean()
        correlation = np.correlate(centred, centred, mode="full")[frame - 1 :]
        if correlation[0] <= 0:
            continue
        lag = lowest + int(np.argmax(correlation[lowest:highest]))
        if correlation[lag] < 0.4 * correlation[0]:
            continue
        measured.append(rate / lag)
    if len(measured) < 5:
        return None
    return float(median(measured))


def build_qc_report(
    alignment: Iterable[AlignedClip],
    *,
    dialogue: Iterable[AdaptedDialogue] = (),
    clips: Iterable[TtsClip] = (),
    crosstalk: Iterable[CrosstalkRegion] = (),
    transcribe: Callable[[Path], str] | None = None,
    transcription_model: str = "unknown",
    reference_pitch: dict[str, float] | None = None,
) -> QcReport:
    """Build the model-free report, measuring pronunciation when asked to.

    ``transcribe`` is the only thing that needs a model; without it the report is
    complete except for pronunciation, which is then reported as not measured
    rather than silently omitted. ``reference_pitch`` - each character's reference
    pitch, from their voice profile - turns on the check that the generated voices are
    the actors' own.
    """

    line_metrics = measure_lines(alignment)
    regions = tuple(crosstalk)
    clip_list = tuple(clips)

    pronunciation: PronunciationReport | None = None
    if transcribe is not None:
        pronunciation = measure_pronunciation(
            dialogue, clip_list, transcribe=transcribe, model=transcription_model
        )

    voice_identity: VoiceIdentityReport | None = None
    if reference_pitch:
        voice_identity = measure_voice_identity(
            clip_list, reference_pitch=reference_pitch
        )

    return QcReport(
        duration=summarise_duration(line_metrics),
        lines=line_metrics,
        crosstalk_regions=len(regions),
        crosstalk_seconds=sum(region.duration for region in regions),
        pronunciation=pronunciation,
        attribution_uncertain=_attribution_uncertain(alignment, regions),
        voice_identity=voice_identity,
    )


def _attribution_uncertain(
    alignment: Iterable[AlignedClip],
    regions: tuple[CrosstalkRegion, ...],
) -> tuple[int, ...]:
    """Return the indexes of lines that share time with simultaneous speech.

    Attribution is temporal: each diarized turn is transcribed on its own and everything
    it contains is credited to that turn's speaker. Where two turns overlap, the exclusive
    view has already picked a winner, and nothing in the pipeline checks that pick against
    the actual voice. That is a real limit rather than a bug to paper over, so the lines
    it affects are named.
    """

    if not regions:
        return ()

    uncertain: list[int] = []
    for line in alignment:
        start = line.clip.dialogue.start
        end = line.clip.dialogue.end
        for region in regions:
            if min(end, region.end) - max(start, region.start) > 1e-6:
                uncertain.append(line.index)
                break
    return tuple(uncertain)


__all__ = [
    "COLLAPSED_VOICE_PITCH",
    "DISTINCT_REFERENCE_PITCH",
    "VOICE_PITCH_DRIFT",
    # Re-exported from ``amharic_text``: the syllable is the unit of *both* a
    # delivery rate here and a timing budget in ``dialogue_context``, and it would
    # be surprising for a caller measuring a run to reach into another module for it.
    # The Roman-script helpers come with it, because whether a borrowed word can be
    # read by the engine is a measurement's business too.
    "HOMOPHONE_FAMILIES",
    "PLAUSIBLE_SYLLABLES_PER_SECOND",
    "SYLLABLE_RANGES",
    "DurationFitReport",
    "InvalidMeasurementError",
    "InvalidTextError",
    "LineMetrics",
    "PronunciationReport",
    "QcError",
    "QcReport",
    "VoiceIdentityReport",
    "build_qc_report",
    "character_error_rate",
    "count_syllables",
    "has_latin",
    "has_pronounceable_text",
    "latin_spans",
    "measure_lines",
    "measure_pronunciation",
    "measure_voice_identity",
    "normalise_for_comparison",
    "summarise_duration",
]
