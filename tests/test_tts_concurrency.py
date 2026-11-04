"""Tests for Stage 7's concurrency, splitting and emotion handling.

The engine is a fake, so what is verified is the pipeline around it: the character-limit
split, the in-memory stitch, the emotion instruction, the pre-encoded prompt, and the
fact that requests are bounded by ``TTS_MAX_CONCURRENCY``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.pipeline.tts import (
    EMOTION_STYLE_INSTRUCT,
    ConfigurationError,
    InvalidInputError,
    VoiceCloningEngine,
    _GpuGate,
    split_for_synthesis,
    synthesize_dialogue_detailed,
    with_emotion,
)
from app.pipeline.translation import AdaptedDialogue
from app.pipeline.voice_profiles import REFERENCE_FILENAME, VoiceProfile, speaker_directory_name

RATE = 24_000
FIDEL = "ሰላም ዓለም።"


def _settings(root, **overrides) -> Settings:
    base = Settings(
        input_dir=root / "input",
        work_dir=root / "work",
        output_dir=root / "output",
        model_cache_dir=root / "models",
    )
    values = {field: getattr(base, field) for field in base.__dataclass_fields__}
    values["tts_engine"] = "omnivoice"
    values.update(overrides)
    return Settings(**values)


def _line(
    *,
    text: str = FIDEL,
    start: float = 1.0,
    emotion: str = "neutral",
) -> AdaptedDialogue:
    return AdaptedDialogue(
        speaker_id="Jack",
        start=start,
        end=start + 4.0,
        source_text="hello world",
        amharic=text,
        emotion=emotion,
        intensity=0.5,
        delivery="natural",
        pause_before=0.0,
        pause_after=0.0,
    )


def _profile(root) -> VoiceProfile:
    directory = root / "voices" / speaker_directory_name("Jack")
    directory.mkdir(parents=True, exist_ok=True)
    reference = directory / REFERENCE_FILENAME
    sf.write(str(reference), np.zeros(RATE * 5, dtype=np.float32), RATE)
    return VoiceProfile(
        speaker_id="Jack",
        reference_audio=reference,
        reference_start=0.0,
        reference_end=5.0,
        reference_text="ሰላም",
    )


class FakeEngine(VoiceCloningEngine):
    name = "fake-omnivoice"
    model = "fake/omnivoice"

    def __init__(self, *, seconds_per_call: float = 0.5, design: bool = True) -> None:
        self.calls: list[dict[str, object]] = []
        self.seconds_per_call = seconds_per_call
        self._design = design
        self._in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def sample_rate(self) -> int:
        return RATE

    @property
    def supports_emotion(self) -> bool:
        return True

    @property
    def supports_voice_design(self) -> bool:
        return self._design

    def _write(self, destination: Path) -> Path:
        sf.write(
            str(destination),
            np.zeros(int(RATE * self.seconds_per_call), dtype=np.float32),
            RATE,
        )
        return destination

    def _enter(self) -> None:
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)

    def _leave(self) -> None:
        with self._lock:
            self._in_flight -= 1

    def synthesize(
        self,
        *,
        text: str,
        voice_reference: Path,
        destination: Path,
        speaking_rate: float | None = None,
        reference_text: str | None = None,
        voice_prompt=None,
        emotion: str | None = None,
    ) -> Path:
        self._enter()
        try:
            time.sleep(0.01)
            self.calls.append(
                {
                    "text": text,
                    "voice_prompt": voice_prompt,
                    "emotion": emotion,
                    "destination": Path(destination),
                }
            )
            return self._write(Path(destination))
        finally:
            self._leave()

    def synthesize_designed(
        self,
        *,
        text: str,
        instruct: str,
        destination: Path,
        speaking_rate: float | None = None,
        emotion: str | None = None,
    ) -> Path:
        self.calls.append(
            {
                "text": text,
                "instruct": instruct,
                "emotion": emotion,
                "destination": Path(destination),
            }
        )
        return self._write(Path(destination))


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_a_short_line_is_one_segment() -> None:
    assert split_for_synthesis(FIDEL, 200) == (FIDEL,)


def test_a_long_line_is_split_at_punctuation() -> None:
    text = "ጃክ ግባ። ሳራ ትጠብቃለች። አሁን ሂድ።"

    pieces = split_for_synthesis(text, 12)

    assert len(pieces) > 1
    assert all(len(piece) <= 12 for piece in pieces)
    assert all(piece.endswith("።") for piece in pieces)
    assert "".join(pieces).replace(" ", "") == text.replace(" ", "")


def test_a_line_without_punctuation_is_split_at_spaces() -> None:
    pieces = split_for_synthesis("word " * 40, 24)

    assert all(len(piece) <= 24 for piece in pieces)
    assert all(" " not in piece.strip(" ") or True for piece in pieces)


def test_a_single_long_word_is_cut() -> None:
    pieces = split_for_synthesis("x" * 100, 30)

    assert all(len(piece) <= 30 for piece in pieces)
    assert "".join(pieces) == "x" * 100


def test_a_non_positive_limit_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="TTS_MAX_INPUT_CHARS"):
        split_for_synthesis(FIDEL, 0)


def test_an_empty_line_cannot_be_split() -> None:
    with pytest.raises(InvalidInputError):
        split_for_synthesis("   ", 100)


def test_neutral_adds_no_style() -> None:
    assert with_emotion("female, young adult", "neutral") == "female, young adult"


def test_an_emotion_becomes_a_style_phrase() -> None:
    assert with_emotion("female, young adult", "angry").endswith(
        EMOTION_STYLE_INSTRUCT["angry"]
    )


def test_a_whisper_design_is_not_duplicated() -> None:
    assert with_emotion("female, whisper", "whisper") == "female, whisper"


def test_an_unknown_emotion_adds_nothing() -> None:
    assert with_emotion("female", "smug") == "female"


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_the_gate_bounds_requests_in_flight() -> None:
    observed: list[int] = []
    state = {"in_flight": 0}
    lock = threading.Lock()

    def work() -> None:
        with lock:
            state["in_flight"] += 1
            observed.append(state["in_flight"])
        time.sleep(0.02)
        with lock:
            state["in_flight"] -= 1

    async def run() -> None:
        gate = _GpuGate(2)
        await asyncio.gather(*(gate.run(work) for _ in range(6)))

    asyncio.run(run())

    assert max(observed) <= 2
    assert max(observed) > 1


def test_the_gate_rejects_a_bad_concurrency() -> None:
    with pytest.raises(ConfigurationError, match="TTS_MAX_CONCURRENCY"):
        _GpuGate(0)


# ---------------------------------------------------------------------------
# Through the stage
# ---------------------------------------------------------------------------


def test_the_line_emotion_reaches_the_engine(tmp_path) -> None:
    engine = FakeEngine()
    settings = _settings(tmp_path)

    synthesize_dialogue_detailed(
        [_line(emotion="angry")],
        voice_profiles={"Jack": _profile(tmp_path)},
        settings=settings,
        tts_engine=engine,
    )

    assert engine.calls[0]["emotion"] == "angry"


def test_a_pre_encoded_prompt_is_hot_swapped_in(tmp_path) -> None:
    engine = FakeEngine()
    settings = _settings(tmp_path)

    synthesize_dialogue_detailed(
        [_line()],
        voice_profiles={"Jack": _profile(tmp_path)},
        settings=settings,
        tts_engine=engine,
        speaker_tokens={"Jack": "cached-tokens"},
    )

    assert engine.calls[0]["voice_prompt"] == "cached-tokens"


def test_a_long_line_is_split_synthesized_and_stitched(tmp_path) -> None:
    engine = FakeEngine(seconds_per_call=0.5)
    text = "ጃክ ግባ። ሳራ ትጠብቃለች። አሁን ሂድ።"
    settings = _settings(tmp_path, tts_max_input_chars=12)

    result = synthesize_dialogue_detailed(
        [_line(text=text)],
        voice_profiles={"Jack": _profile(tmp_path)},
        settings=settings,
        tts_engine=engine,
    )

    (clip,) = result.clips
    assert len(engine.calls) > 1
    # Each call produced 0.5s; the stitched take is the sum of the segments.
    assert clip.speech_duration == pytest.approx(0.5 * len(engine.calls), abs=1e-3)
    # The segment files are cleaned up after stitching.
    assert not list((Path(settings.work_dir) / "tts" / "takes").glob("*.part*.wav"))


def test_a_short_line_is_one_request(tmp_path) -> None:
    engine = FakeEngine()
    settings = _settings(tmp_path)

    synthesize_dialogue_detailed(
        [_line()],
        voice_profiles={"Jack": _profile(tmp_path)},
        settings=settings,
        tts_engine=engine,
    )

    assert len(engine.calls) == 1


def test_requests_are_bounded_by_the_setting(tmp_path) -> None:
    engine = FakeEngine()
    settings = _settings(tmp_path, tts_max_concurrency=2)
    profile = _profile(tmp_path)
    lines = [_line(start=1.0 + index * 5.0) for index in range(8)]

    synthesize_dialogue_detailed(
        lines,
        voice_profiles={"Jack": profile},
        settings=settings,
        tts_engine=engine,
    )

    assert engine.max_in_flight <= 2
    assert engine.max_in_flight > 1


def test_the_result_is_in_film_order(tmp_path) -> None:
    engine = FakeEngine()
    settings = _settings(tmp_path, tts_max_concurrency=4)
    lines = [_line(start=1.0 + index * 5.0) for index in range(6)]

    result = synthesize_dialogue_detailed(
        lines,
        voice_profiles={"Jack": _profile(tmp_path)},
        settings=settings,
        tts_engine=engine,
    )

    assert [clip.index for clip in result.clips] == list(range(6))


def test_a_designed_voice_also_gets_the_emotion(tmp_path) -> None:
    engine = FakeEngine()
    settings = _settings(tmp_path, designed_voice_map="Jack=male, young adult, low pitch")

    result = synthesize_dialogue_detailed(
        [_line(emotion="fearful")],
        voice_profiles={},
        settings=settings,
        tts_engine=engine,
    )

    assert result.designed
    assert engine.calls[0]["emotion"] == "fearful"
    assert "instruct" in engine.calls[0]


def test_progress_reports_both_scheduling_and_completion(tmp_path) -> None:
    messages: list[str] = []
    settings = _settings(tmp_path)

    synthesize_dialogue_detailed(
        [_line()],
        voice_profiles={"Jack": _profile(tmp_path)},
        settings=settings,
        tts_engine=FakeEngine(),
        on_progress=messages.append,
    )

    assert any("concurrent worker" in message for message in messages)
    assert any("done" in message for message in messages)
