"""Tests for Stage 8's active-vocal-zone alignment.

The point of the stage is that a dub is *placed* inside the actor's speech and padded
with silence, rather than time-stretched to fill a diarized turn. These tests build
real WAV files and check the audio that is written.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.pipeline.timing import (
    EXACT_FIT_TOLERANCE_SECONDS,
    PIPELINE_SAMPLE_RATE,
    align_clip,
    detect_active_zone,
    iterate_aligned,
)
from app.pipeline.translation import AdaptedDialogue
from app.pipeline.tts import PerformanceControls, TtsClip

RATE = PIPELINE_SAMPLE_RATE


def _settings(root, **overrides) -> Settings:
    base = Settings(
        input_dir=root / "input",
        work_dir=root / "work",
        output_dir=root / "output",
        model_cache_dir=root / "models",
    )
    values = {field: getattr(base, field) for field in base.__dataclass_fields__}
    values.update(overrides)
    return Settings(**values)


def _dialogue(*, start: float = 1.0, end: float = 3.0) -> AdaptedDialogue:
    return AdaptedDialogue(
        speaker_id="Jack",
        start=start,
        end=end,
        source_text="Get in the car.",
        amharic="ጃክ ግባ።",
        emotion="neutral",
        intensity=0.45,
        delivery="natural",
        pause_before=0.0,
        pause_after=0.0,
    )


def _clip(
    path,
    *,
    speech: float,
    rate: int = RATE,
    lead: float = 0.0,
    trailer: float = 0.0,
    dialogue=None,
) -> TtsClip:
    dialogue = dialogue or _dialogue()
    count = int(round(speech * rate))
    samples = np.sin(
        2 * np.pi * 220.0 * np.arange(count, dtype=np.float32) / rate
    ) * 0.3
    pieces = []
    if lead:
        pieces.append(np.zeros(int(round(lead * rate)), dtype=np.float32))
    pieces.append(samples)
    if trailer:
        pieces.append(np.zeros(int(round(trailer * rate)), dtype=np.float32))
    sf.write(str(path), np.concatenate(pieces), rate)

    return TtsClip(
        index=0,
        dialogue=dialogue,
        performance=PerformanceControls.from_dialogue(dialogue, seed=0),
        audio_path=path,
        take_path=path,
        performance_reference_path=None,
        voice_reference_path=None,
        sample_rate=rate,
        speech_duration=speech,
        rendered_pause_before=lead,
        rendered_pause_after=trailer,
        performance_engine="omnivoice",
        style_engine="none",
    )


def _stem(path, *, duration: float = 6.0, speech=(2.0, 3.0)) -> None:
    samples = np.zeros(int(round(duration * RATE)), dtype=np.float32)
    first, last = speech
    count = int(round((last - first) * RATE))
    tone = np.sin(2 * np.pi * 180.0 * np.arange(count) / RATE) * 0.5
    samples[int(round(first * RATE)) : int(round(first * RATE)) + count] = tone
    sf.write(str(path), samples, RATE)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_the_active_zone_is_the_speech_inside_the_turn(tmp_path) -> None:
    stem = tmp_path / "speech.wav"
    _stem(stem, speech=(2.0, 3.0))

    start, end = detect_active_zone(stem, 1.0, 4.0)

    assert start == pytest.approx(2.0, abs=0.05)
    assert end == pytest.approx(3.0, abs=0.05)


def test_a_region_without_speech_falls_back_to_the_whole_window(tmp_path) -> None:
    stem = tmp_path / "silence.wav"
    sf.write(str(stem), np.zeros(RATE * 4, dtype=np.float32), RATE)

    assert detect_active_zone(stem, 1.0, 3.0) == (1.0, 3.0)


def test_an_unreadable_stem_falls_back_rather_than_failing(tmp_path) -> None:
    """A diagnostic must never break a run."""

    assert detect_active_zone(tmp_path / "absent.wav", 1.0, 3.0) == (1.0, 3.0)


def test_a_zone_shorter_than_the_floor_is_ignored(tmp_path) -> None:
    stem = tmp_path / "click.wav"
    samples = np.zeros(RATE * 4, dtype=np.float32)
    samples[RATE * 2 : RATE * 2 + 40] = 0.8
    sf.write(str(stem), samples, RATE)

    assert detect_active_zone(stem, 1.0, 3.0, minimum_seconds=0.1) == (1.0, 3.0)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def test_a_short_line_is_padded_with_silence_not_stretched(tmp_path) -> None:
    clip = _clip(tmp_path / "clip.wav", speech=1.0)

    aligned = align_clip(
        clip,
        tmp_path / "aligned.wav",
        settings=_settings(tmp_path),
        active_zone=(1.4, 2.6),
    )

    assert aligned.tempo == pytest.approx(1.0)
    assert aligned.active_zone_start == pytest.approx(1.4)
    assert aligned.active_zone_end == pytest.approx(2.6)
    # 0.4s of silence before the actor's first phoneme, then the 1.0s line, then the
    # rest of the window.
    assert aligned.silence_before == pytest.approx(0.4)
    assert aligned.silence_after == pytest.approx(0.6)
    assert aligned.speech_duration == pytest.approx(1.0, abs=1e-3)
    # The file starts at the actor's first phoneme, so the window's own lead-in is not
    # part of it; what is written is the line plus the silence that follows it.
    assert aligned.start == pytest.approx(1.4)
    assert aligned.duration == pytest.approx(1.6, abs=1e-3)
    assert aligned.end == pytest.approx(3.0, abs=1e-3)
    assert any("cadence is left untouched" in note for note in aligned.notes)


def test_the_written_file_fills_the_window(tmp_path) -> None:
    clip = _clip(tmp_path / "clip.wav", speech=0.8)
    destination = tmp_path / "aligned.wav"

    aligned = align_clip(
        clip,
        destination,
        settings=_settings(tmp_path),
        active_zone=(1.5, 2.5),
    )

    samples, rate = sf.read(str(destination), dtype="float32")
    assert rate == RATE
    # 1.5s -> 3.0s: the line and the rest of the window, filled with silence.
    assert samples.shape[0] / rate == pytest.approx(1.5, abs=0.02)
    assert aligned.end == pytest.approx(3.0, abs=0.02)


def test_an_overlong_line_is_compressed_only_when_it_cannot_fit(tmp_path, monkeypatch) -> None:
    """Compression is the fallback, and it still respects the tempo band."""

    import app.pipeline.timing as timing

    calls: list[float] = []

    def fake_stretched(clip, region, sample_rate, tempo, destination):
        calls.append(tempo)
        wanted = int(round(region.shape[0] / tempo))
        return np.zeros(wanted, dtype=np.float32), RATE

    monkeypatch.setattr(timing, "_stretched_speech", fake_stretched)
    clip = _clip(tmp_path / "clip.wav", speech=4.0)

    aligned = align_clip(
        clip,
        tmp_path / "aligned.wav",
        settings=_settings(tmp_path),
        room=2.0,
        active_zone=(1.0, 3.0),
    )

    # An exact fit would need 2.0x, which is outside the configured band, so the line
    # is compressed to the band limit and the remainder is reported, not hidden.
    assert calls == [pytest.approx(1.45)]
    assert aligned.tempo == pytest.approx(1.45)
    assert aligned.overrun > 0.0


def test_a_line_that_fits_the_room_is_not_touched(tmp_path, monkeypatch) -> None:
    import app.pipeline.timing as timing

    calls: list[float] = []

    def fake_stretched(clip, region, sample_rate, tempo, destination):
        calls.append(tempo)
        return region, RATE

    monkeypatch.setattr(timing, "_stretched_speech", fake_stretched)
    clip = _clip(tmp_path / "clip.wav", speech=2.5)

    align_clip(
        clip,
        tmp_path / "aligned.wav",
        settings=_settings(tmp_path),
        room=3.0,
        active_zone=(1.0, 3.0),
    )

    assert calls == []


def test_a_leading_pause_is_clipped_to_the_active_zone(tmp_path) -> None:
    clip = _clip(tmp_path / "clip.wav", speech=1.0, lead=0.5)
    dialogue = _dialogue(start=1.0, end=3.0)
    clip = _clip(
        tmp_path / "clip_with_pause.wav",
        speech=1.0,
        lead=0.5,
        dialogue=dialogue,
    )

    aligned = align_clip(
        clip,
        tmp_path / "aligned.wav",
        settings=_settings(tmp_path),
        active_zone=(1.2, 2.8),
    )

    # The file may start earlier than the zone only by the line's own pause: the pause
    # is silence, and the speech itself still begins exactly at the zone.
    assert aligned.start == pytest.approx(0.7)
    assert aligned.start + aligned.rendered_pause_before == pytest.approx(1.2)
    assert aligned.rendered_pause_before == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_lines_are_yielded_one_at_a_time(tmp_path) -> None:
    first = _clip(tmp_path / "a.wav", speech=1.0)
    second = _clip(tmp_path / "b.wav", speech=1.0)

    stream = iterate_aligned(
        [first, second],
        output_dir=tmp_path,
        settings=_settings(tmp_path),
    )

    produced = []
    for aligned in stream:
        produced.append(aligned)
        assert aligned.audio_path.is_file()
    assert len(produced) == 2


def test_the_stem_drives_the_zone_of_each_line(tmp_path) -> None:
    stem = tmp_path / "speech.wav"
    _stem(stem, duration=6.0, speech=(1.5, 2.5))
    clip = _clip(tmp_path / "clip.wav", speech=0.5)

    aligned = list(
        iterate_aligned(
            [clip],
            output_dir=tmp_path,
            settings=_settings(tmp_path),
            speech_stem=stem,
        )
    )[0]

    assert aligned.active_zone_start == pytest.approx(1.5, abs=0.05)
    assert aligned.tempo == pytest.approx(1.0)
    assert aligned.silence_before > 0.0


def test_an_empty_input_streams_nothing(tmp_path) -> None:
    assert list(iterate_aligned([], settings=_settings(tmp_path))) == []
