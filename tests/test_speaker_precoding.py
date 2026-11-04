"""Tests for Stage 6's pre-encoded speaker profiles.

The encoder is injected, so no OmniVoice weights are loaded and no GPU is needed. What
is verified is the directory convention, the pairing rule, the in-memory cache, and
that the tokens reach the synthesis stage unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.pipeline.voice_profiles import (
    InvalidProfileError,
    MissingVoiceSampleError,
    OmniVoiceProfileEncoder,
    SpeakerProfile,
    SpeakerProfileBank,
    precompute_speaker_profiles,
    reset_speaker_profile_cache,
)


class FakeEncoder:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    def encode(self, *, reference_audio, reference_text):
        self.calls.append((reference_audio.name, reference_text))
        return f"tokens:{reference_text}"

    def close(self) -> None:
        self.closed = True


def _settings(root, **overrides) -> Settings:
    base = Settings(
        input_dir=root / "input",
        work_dir=root / "work",
        output_dir=root / "output",
        model_cache_dir=root / "models",
        voice_sample_dir=root / "voice_samples",
    )
    values = {field: getattr(base, field) for field in base.__dataclass_fields__}
    values.update(overrides)
    return Settings(**values)


def _sample(directory, *, seconds: float = 5.0, text: str = "ሰላም ጓደኛዬ።") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(directory / "sample_01.wav"),
        np.zeros(int(24_000 * seconds), dtype=np.float32),
        24_000,
    )
    (directory / "sample_01.txt").write_text(text, encoding="utf-8")


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    reset_speaker_profile_cache()


def test_a_complete_pair_is_encoded_once(tmp_path) -> None:
    _sample(tmp_path / "voice_samples" / "Jack")
    encoder = FakeEncoder()

    bank = precompute_speaker_profiles(
        ["Jack"], settings=_settings(tmp_path), encoder=encoder
    )

    assert list(bank.profiles) == ["Jack"]
    assert bank.tokens() == {"Jack": "tokens:ሰላም ጓደኛዬ።"}
    assert len(encoder.calls) == 1
    assert encoder.closed is False  # an injected encoder is the caller's to close


def test_the_directory_name_is_matched_case_insensitively(tmp_path) -> None:
    _sample(tmp_path / "voice_samples" / "MARY-JANE")

    bank = precompute_speaker_profiles(
        ["MaryJane"], settings=_settings(tmp_path), encoder=FakeEncoder()
    )

    assert "MaryJane" in bank


def test_a_character_without_a_directory_is_missing_but_not_fatal(tmp_path) -> None:
    _sample(tmp_path / "voice_samples" / "Jack")

    bank = precompute_speaker_profiles(
        ["Jack", "Sarah"], settings=_settings(tmp_path), encoder=FakeEncoder()
    )

    assert list(bank.profiles) == ["Jack"]
    assert bank.missing == ("Sarah",)


def test_a_half_pair_is_reported(tmp_path) -> None:
    directory = tmp_path / "voice_samples" / "Jack"
    directory.mkdir(parents=True)
    sf.write(str(directory / "sample_01.wav"), np.zeros(2400, dtype=np.float32), 24_000)

    bank = precompute_speaker_profiles(
        ["Jack"], settings=_settings(tmp_path), encoder=FakeEncoder()
    )

    assert bank.profiles == {}
    assert bank.missing == ("Jack",)
    assert any("no complete" in note for note in bank.notes)


def test_an_empty_transcript_is_not_a_pair(tmp_path) -> None:
    _sample(tmp_path / "voice_samples" / "Jack", text="   ")

    bank = precompute_speaker_profiles(
        ["Jack"], settings=_settings(tmp_path), encoder=FakeEncoder()
    )

    assert bank.profiles == {}
    assert any("empty" in note for note in bank.notes)


def test_a_missing_sample_directory_leaves_everyone_to_the_film(tmp_path) -> None:
    bank = precompute_speaker_profiles(
        ["Jack", "Sarah"], settings=_settings(tmp_path), encoder=FakeEncoder()
    )

    assert bank.profiles == {}
    assert bank.missing == ("Jack", "Sarah")
    assert any("no character sample directories" in note for note in bank.notes)


def test_the_same_sample_is_not_encoded_twice(tmp_path) -> None:
    _sample(tmp_path / "voice_samples" / "Jack")
    encoder = FakeEncoder()
    settings = _settings(tmp_path)

    first = precompute_speaker_profiles(["Jack"], settings=settings, encoder=encoder)
    second = precompute_speaker_profiles(["Jack"], settings=settings, encoder=encoder)

    assert len(encoder.calls) == 1
    assert second.tokens() == first.tokens()


def test_the_cache_can_be_bypassed(tmp_path) -> None:
    _sample(tmp_path / "voice_samples" / "Jack")
    encoder = FakeEncoder()
    settings = _settings(tmp_path)

    precompute_speaker_profiles(["Jack"], settings=settings, encoder=encoder)
    precompute_speaker_profiles(
        ["Jack"], settings=settings, encoder=encoder, use_cache=False
    )

    assert len(encoder.calls) == 2


def test_a_reference_longer_than_a_cloning_prompt_is_rejected(tmp_path) -> None:
    _sample(tmp_path / "voice_samples" / "Jack", seconds=45.0)

    with pytest.raises(InvalidProfileError, match="at most"):
        precompute_speaker_profiles(
            ["Jack"], settings=_settings(tmp_path), encoder=FakeEncoder()
        )


def test_a_reference_shorter_than_a_cloning_prompt_is_rejected(tmp_path) -> None:
    _sample(tmp_path / "voice_samples" / "Jack", seconds=0.5)

    with pytest.raises(InvalidProfileError, match="at least"):
        precompute_speaker_profiles(
            ["Jack"], settings=_settings(tmp_path), encoder=FakeEncoder()
        )


def test_an_unreadable_transcript_is_reported(tmp_path) -> None:
    directory = tmp_path / "voice_samples" / "Jack"
    _sample(directory)
    (directory / "sample_01.txt").write_bytes(b"\xff\xfe\x00")

    with pytest.raises(MissingVoiceSampleError, match="transcript"):
        precompute_speaker_profiles(
            ["Jack"], settings=_settings(tmp_path), encoder=FakeEncoder()
        )


def test_no_characters_means_no_work(tmp_path) -> None:
    encoder = FakeEncoder()

    bank = precompute_speaker_profiles([], settings=_settings(tmp_path), encoder=encoder)

    assert bank == SpeakerProfileBank()
    assert encoder.calls == []


def test_the_bank_converts_to_the_stages_voice_profiles(tmp_path) -> None:
    _sample(tmp_path / "voice_samples" / "Jack", seconds=5.0)

    bank = precompute_speaker_profiles(
        ["Jack"], settings=_settings(tmp_path), encoder=FakeEncoder()
    )

    profiles = bank.voice_profiles()

    profile = profiles["Jack"]
    assert profile.speaker_id == "Jack"
    assert profile.reference_text == "ሰላም ጓደኛዬ።"
    assert profile.reference_duration == pytest.approx(5.0, abs=1e-3)
    assert "pre-encoded" in profile.selection_reason


def test_a_profile_is_json_safe_and_hides_its_tokens(tmp_path) -> None:
    _sample(tmp_path / "voice_samples" / "Jack")

    bank = precompute_speaker_profiles(
        ["Jack"], settings=_settings(tmp_path), encoder=FakeEncoder()
    )

    payload = bank.profiles["Jack"].as_dict()

    assert payload["pre_encoded"] is True
    assert "tokens" not in payload


def test_a_profile_needs_the_words_its_reference_says() -> None:
    with pytest.raises(InvalidProfileError, match="transcript"):
        SpeakerProfile(
            character="Jack",
            reference_audio="sample_01.wav",
            reference_text="  ",
            tokens=None,
        )


def test_the_default_encoder_is_named_for_the_engine(tmp_path) -> None:
    _sample(tmp_path / "voice_samples" / "Jack")
    settings = _settings(tmp_path)
    encoder = OmniVoiceProfileEncoder(
        model=settings.omnivoice_model, device=settings.device
    )

    assert encoder.name == "omnivoice"
