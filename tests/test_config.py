"""Tests for the environment-driven configuration system.

These tests never touch real credentials and never require a GPU, FFmpeg, or any
model weights - the configuration layer must work on a bare machine.
"""

from __future__ import annotations

import os

import pytest

from app.config import (
    DEFAULT_DIALOGUE_BIBLE_FILENAME,
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_MIX_DIALOGUE_GAIN_DB,
    DEFAULT_MIX_DUCK_DB,
    DEFAULT_OMNIVOICE_MODEL,
    DEFAULT_TIMING_MAX_TEMPO,
    DEFAULT_TIMING_MIN_LINE_GAP,
    DEFAULT_TRANSLATION_BASE_URL,
    DEFAULT_TRANSLATION_GPU_MEMORY_UTILIZATION,
    DEFAULT_TRANSLATION_MAX_CONCURRENCY,
    DEFAULT_TRANSLATION_MODEL,
    DEFAULT_TRANSLATION_SYLLABLES_PER_SECOND,
    DEFAULT_TRANSLATION_WINDOW_LINES,
    DEFAULT_TRANSLATION_WINDOW_OVERLAP,
    DEFAULT_TIMING_MIN_TEMPO,
    DEFAULT_TRANSCRIPTION_COMPUTE_TYPE,
    DEFAULT_TRANSCRIPTION_MODEL,
    DEFAULT_TTS_MAX_INPUT_CHARS,
    DEFAULT_TTS_MIN_LINE_SECONDS,
    DEFAULT_TTS_MAX_PAUSE_SECONDS,
    DEFAULT_TTS_MODEL,
    DEFAULT_VOICE_REFERENCE_MAX_DURATION,
    DEFAULT_VOICE_REFERENCE_MIN_DURATION,
    DEFAULT_VOICE_REFERENCE_TARGET_DURATION,
    TRANSLATION_API_KEY_ENV,
    Settings,
    apply_torch_compile_policy,
    get_settings,
)


def test_defaults_are_project_relative(_clean_env: None) -> None:
    settings = Settings.from_env()

    assert settings.input_dir.name == "input"
    assert settings.work_dir.name == "working"
    assert settings.output_dir.name == "output"
    assert settings.model_cache_dir.name == "models_cache"
    assert settings.device == "cuda"
    assert settings.log_level == "INFO"
    assert (
        settings.omnivoice_model
        == DEFAULT_OMNIVOICE_MODEL
        == "african-low-resource/omnivoice-amharic"
    )


def test_missing_credentials_do_not_break_construction(_clean_env: None) -> None:
    settings = Settings.from_env()

    assert settings.translation_api_key is None
    assert settings.huggingface_token is None
    assert settings.has_huggingface_credentials is False
    # Dialogue adaptation is served from the same GPU host, so no hosted-vendor key
    # exists anywhere in the pipeline. Only the gated model download needs a token.
    assert settings.missing_credentials() == ["HUGGINGFACE_TOKEN"]


def test_the_local_server_needs_no_api_key(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    """A loopback vLLM deployment is a working configuration with no secret at all."""

    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_test")
    monkeypatch.setenv("TRANSLATION_API_KEY", "")

    settings = Settings.from_env()

    assert settings.translation_api_key is None
    assert settings.missing_credentials() == []
    assert settings.translation_api_key_env == TRANSLATION_API_KEY_ENV


def test_an_optional_local_token_is_read(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv("TRANSLATION_API_KEY", "local-token")

    assert Settings.from_env().translation_api_key == "local-token"


def test_the_local_llm_defaults_are_a_loopback_vllm_server(_clean_env: None) -> None:
    settings = Settings.from_env()

    assert settings.translation_model == DEFAULT_TRANSLATION_MODEL
    assert "qwen" in settings.translation_model.lower()
    assert settings.translation_base_url == DEFAULT_TRANSLATION_BASE_URL
    assert settings.translation_base_url.startswith("http://127.0.0.1")
    assert settings.translation_window_lines == DEFAULT_TRANSLATION_WINDOW_LINES
    assert settings.translation_window_overlap == DEFAULT_TRANSLATION_WINDOW_OVERLAP
    assert (
        settings.translation_window_overlap < settings.translation_window_lines
    )
    assert settings.translation_max_concurrency == DEFAULT_TRANSLATION_MAX_CONCURRENCY
    # The card is shared with the rest of the pipeline, so the server must not claim
    # almost all of it: at 0.90 on a 48 GiB A40, transcription could not load at all.
    assert (
        settings.translation_gpu_memory_utilization
        == DEFAULT_TRANSLATION_GPU_MEMORY_UTILIZATION
        == 0.72
    )
    assert settings.translation_gpu_memory_utilization < 0.80


def test_the_server_arguments_match_the_client_settings(_clean_env: None) -> None:
    """The server and the client must agree on the VRAM reservation and the batch."""

    settings = Settings.from_env()

    arguments = settings.vllm_server_arguments()

    assert arguments[0] == "--model"
    assert arguments[1] == settings.translation_model
    assert "--gpu-memory-utilization" in arguments
    assert f"{settings.translation_gpu_memory_utilization:.2f}" in arguments
    assert str(settings.translation_max_concurrency) in arguments


def test_an_explicit_endpoint_still_wins(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    """Someone pointing at a proxy or a mirror means it."""

    monkeypatch.setenv("TRANSLATION_BASE_URL", "http://10.0.0.5:9000/v1")
    monkeypatch.setenv("TRANSLATION_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    settings = Settings.from_env()

    assert settings.translation_base_url == "http://10.0.0.5:9000/v1"
    assert settings.translation_model == "Qwen/Qwen2.5-7B-Instruct"


def test_the_screenplay_path_is_configurable(
    monkeypatch: pytest.MonkeyPatch, tmp_path, _clean_env: None
) -> None:
    script = tmp_path / "movie.txt"
    script.write_text("JACK\nHello.\n", encoding="utf-8")
    monkeypatch.setenv("SCRIPT_PATH", str(script))

    assert Settings.from_env().script_path == script


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("INPUT_DIR", str(tmp_path / "in"))
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_test_token")
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("LOG_LEVEL", "debug")

    settings = Settings.from_env()

    assert settings.input_dir == tmp_path / "in"
    assert settings.work_dir == tmp_path / "work"
    assert settings.output_dir == tmp_path / "out"
    assert settings.model_cache_dir == tmp_path / "models"
    assert settings.device == "cpu"
    assert settings.log_level == "DEBUG"
    assert settings.missing_credentials() == []


def test_torch_compile_is_disabled_unless_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inductor can abort when the pod's CUDA build predates the GPU."""

    monkeypatch.delenv("ALLOW_TORCH_COMPILE", raising=False)
    monkeypatch.delenv("TORCH_COMPILE_DISABLE", raising=False)

    apply_torch_compile_policy()

    assert os.environ["TORCH_COMPILE_DISABLE"] == "1"


def test_torch_compile_can_be_opted_back_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TORCH_COMPILE_DISABLE", raising=False)
    monkeypatch.setenv("ALLOW_TORCH_COMPILE", "1")

    apply_torch_compile_policy()

    assert "TORCH_COMPILE_DISABLE" not in os.environ


def test_ensure_directories_creates_all_paths(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("INPUT_DIR", str(tmp_path / "input"))
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "working"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "models_cache"))
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    settings = Settings.from_env()
    settings.ensure_directories()

    assert settings.input_dir.is_dir()
    assert settings.work_dir.is_dir()
    assert settings.output_dir.is_dir()
    assert settings.model_cache_dir.is_dir()


def test_as_dict_never_exposes_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSLATION_API_KEY", "super-secret-value")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_secret_value")

    payload = Settings.from_env().as_dict()

    assert payload["translation_api_key_set"] is True
    assert payload["huggingface_token_set"] is True
    assert "super-secret-value" not in str(payload)
    assert "hf_secret_value" not in str(payload)


def test_diarization_model_is_configurable(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    assert Settings.from_env().diarization_model == DEFAULT_DIARIZATION_MODEL

    monkeypatch.setenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")

    settings = Settings.from_env()
    assert settings.diarization_model == "pyannote/speaker-diarization-3.1"
    assert settings.as_dict()["diarization_model"] == "pyannote/speaker-diarization-3.1"


def test_diarization_speaker_counts_are_optional_and_configurable(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    defaults = Settings.from_env()
    assert defaults.diarization_min_speakers is None
    assert defaults.diarization_max_speakers is None

    monkeypatch.setenv("DIARIZATION_MIN_SPEAKERS", "4")
    monkeypatch.setenv("DIARIZATION_MAX_SPEAKERS", "12")

    settings = Settings.from_env()
    assert settings.diarization_min_speakers == 4
    assert settings.diarization_max_speakers == 12
    assert settings.as_dict()["diarization_min_speakers"] == 4
    assert settings.as_dict()["diarization_max_speakers"] == 12


def test_a_malformed_speaker_count_is_rejected(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv("DIARIZATION_MAX_SPEAKERS", "0")
    with pytest.raises(ValueError, match="positive integer"):
        Settings.from_env()

    monkeypatch.setenv("DIARIZATION_MAX_SPEAKERS", "several")
    with pytest.raises(ValueError, match="integer"):
        Settings.from_env()


def test_transcription_settings_are_configurable(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    defaults = Settings.from_env()
    assert defaults.transcription_model == DEFAULT_TRANSCRIPTION_MODEL
    assert defaults.transcription_compute_type == DEFAULT_TRANSCRIPTION_COMPUTE_TYPE
    assert defaults.transcription_language is None

    monkeypatch.setenv("TRANSCRIPTION_MODEL", "medium")
    monkeypatch.setenv("TRANSCRIPTION_COMPUTE_TYPE", "int8")
    monkeypatch.setenv("TRANSCRIPTION_LANGUAGE", "AM")

    settings = Settings.from_env()
    assert settings.transcription_model == "medium"
    assert settings.transcription_compute_type == "int8"
    assert settings.transcription_language == "am"  # normalised to lower case

    payload = settings.as_dict()
    assert payload["transcription_model"] == "medium"
    assert payload["transcription_compute_type"] == "int8"
    assert payload["transcription_language"] == "am"


def test_translation_settings_are_configurable(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    defaults = Settings.from_env()
    assert defaults.translation_model == DEFAULT_TRANSLATION_MODEL
    assert defaults.translation_base_url == DEFAULT_TRANSLATION_BASE_URL
    assert defaults.translation_window_lines == DEFAULT_TRANSLATION_WINDOW_LINES

    monkeypatch.setenv("TRANSLATION_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    monkeypatch.setenv("TRANSLATION_BASE_URL", "http://127.0.0.1:8001/v1")
    monkeypatch.setenv("TRANSLATION_WINDOW_LINES", "32")
    monkeypatch.setenv("TRANSLATION_WINDOW_OVERLAP", "6")
    monkeypatch.setenv("TRANSLATION_MAX_CONCURRENCY", "16")
    monkeypatch.setenv("TRANSLATION_GPU_MEMORY_UTILIZATION", "0.85")
    monkeypatch.setenv("TRANSLATION_TEMPERATURE", "0.2")

    settings = Settings.from_env()
    assert settings.translation_model == "Qwen/Qwen2.5-7B-Instruct"
    assert settings.translation_base_url == "http://127.0.0.1:8001/v1"
    assert settings.translation_window_lines == 32
    assert settings.translation_window_overlap == 6
    assert settings.translation_max_concurrency == 16
    assert settings.translation_gpu_memory_utilization == 0.85
    assert settings.translation_temperature == 0.2

    payload = settings.as_dict()
    assert payload["translation_model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert payload["translation_window_lines"] == 32
    assert payload["translation_window_overlap"] == 6
    # Which model and how it was batched are part of the manifest: a later run
    # cannot be compared with this one without them.
    assert payload["translation_provider"] == "local"
    assert payload["translation_backend"] == "vllm"


def test_malformed_integer_setting_is_rejected(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv("TRANSLATION_WINDOW_LINES", "0")
    with pytest.raises(ValueError, match="positive integer"):
        Settings.from_env()

    monkeypatch.setenv("TRANSLATION_WINDOW_LINES", "lots")
    with pytest.raises(ValueError, match="integer"):
        Settings.from_env()


def test_voice_profile_settings_are_configurable(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    defaults = Settings.from_env()
    assert defaults.voice_profile_dir == defaults.work_dir / "voices"
    assert defaults.voice_reference_min_duration == DEFAULT_VOICE_REFERENCE_MIN_DURATION
    assert (
        defaults.voice_reference_target_duration == DEFAULT_VOICE_REFERENCE_TARGET_DURATION
    )
    assert defaults.voice_reference_max_duration == DEFAULT_VOICE_REFERENCE_MAX_DURATION
    assert defaults.voice_reference_min_duration == 4.0
    assert defaults.voice_reference_target_duration == 5.0
    assert defaults.voice_reference_max_duration == 10.0

    monkeypatch.setenv("VOICE_PROFILE_DIR", "custom/voices")
    monkeypatch.setenv("VOICE_REFERENCE_MIN_DURATION", "2.5")
    monkeypatch.setenv("VOICE_REFERENCE_TARGET_DURATION", "8")
    monkeypatch.setenv("VOICE_REFERENCE_MAX_DURATION", "12.5")

    settings = Settings.from_env()
    assert settings.voice_profile_dir.name == "voices"
    assert settings.voice_profile_dir.parent.name == "custom"
    assert settings.voice_reference_min_duration == 2.5
    assert settings.voice_reference_target_duration == 8.0
    assert settings.voice_reference_max_duration == 12.5

    payload = settings.as_dict()
    assert payload["voice_reference_min_duration"] == 2.5
    assert payload["voice_reference_target_duration"] == 8.0
    assert payload["voice_reference_max_duration"] == 12.5
    assert str(payload["voice_profile_dir"]).endswith("voices")


def test_voice_profile_dir_follows_a_custom_work_dir(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv("WORK_DIR", "custom-work")

    settings = Settings.from_env()

    assert settings.voice_profile_dir == settings.work_dir / "voices"


def test_the_dialogue_bible_path_follows_the_work_directory(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    """Consistency state is per-film working data, so it lives with the work dir."""

    defaults = Settings.from_env()
    assert defaults.dialogue_bible_path == (
        defaults.work_dir / DEFAULT_DIALOGUE_BIBLE_FILENAME
    )

    monkeypatch.setenv("WORK_DIR", "scratch")
    monkeypatch.setenv("DIALOGUE_BIBLE_PATH", "scratch/bible.json")

    settings = Settings.from_env()
    assert settings.dialogue_bible_path.name == "bible.json"
    assert settings.as_dict()["dialogue_bible_path"].endswith("bible.json")


def test_tts_settings_are_configurable(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    defaults = Settings.from_env()
    assert defaults.tts_model == DEFAULT_TTS_MODEL
    assert defaults.tts_max_pause_seconds == DEFAULT_TTS_MAX_PAUSE_SECONDS
    assert defaults.tts_min_line_seconds == DEFAULT_TTS_MIN_LINE_SECONDS
    # A failure should be visible, so absorbing one is opt-in.
    assert defaults.tts_continue_on_failure is False

    monkeypatch.setenv("TTS_MODEL", "someone/other-adapter")
    monkeypatch.setenv("TTS_MAX_PAUSE_SECONDS", "1.5")
    monkeypatch.setenv("TTS_MIN_LINE_SECONDS", "0.45")
    monkeypatch.setenv("TTS_CONTINUE_ON_FAILURE", "true")

    settings = Settings.from_env()
    assert settings.tts_model == "someone/other-adapter"
    assert settings.tts_max_pause_seconds == 1.5
    assert settings.tts_min_line_seconds == 0.45
    assert settings.tts_continue_on_failure is True

    payload = settings.as_dict()
    assert payload["tts_model"] == "someone/other-adapter"
    assert payload["tts_max_pause_seconds"] == 1.5
    assert payload["tts_min_line_seconds"] == 0.45
    assert payload["tts_continue_on_failure"] is True


def test_malformed_float_setting_is_rejected(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv("VOICE_REFERENCE_TARGET_DURATION", "longish")
    with pytest.raises(ValueError, match="must be a number"):
        Settings.from_env()

    monkeypatch.setenv("VOICE_REFERENCE_TARGET_DURATION", "0")
    with pytest.raises(ValueError, match="positive number"):
        Settings.from_env()

    monkeypatch.setenv("VOICE_REFERENCE_TARGET_DURATION", "inf")
    with pytest.raises(ValueError, match="finite"):
        Settings.from_env()


def test_malformed_boolean_setting_is_rejected(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv("TTS_CONTINUE_ON_FAILURE", "maybe")
    with pytest.raises(ValueError, match="boolean flag"):
        Settings.from_env()


def test_timing_settings_are_configurable(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv("TIMING_MIN_TEMPO", "0.9")
    monkeypatch.setenv("TIMING_MAX_TEMPO", "1.1")

    settings = Settings.from_env()

    assert settings.timing_min_tempo == 0.9
    assert settings.timing_max_tempo == 1.1
    payload = settings.as_dict()
    assert payload["timing_min_tempo"] == 0.9
    assert payload["timing_max_tempo"] == 1.1


def test_timing_defaults_allow_both_directions(
    _clean_env: None,
) -> None:
    settings = Settings.from_env()

    assert settings.timing_min_tempo == DEFAULT_TIMING_MIN_TEMPO
    assert settings.timing_max_tempo == DEFAULT_TIMING_MAX_TEMPO
    assert settings.timing_min_tempo < 1.0 < settings.timing_max_tempo


def test_a_malformed_tempo_is_rejected(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv("TIMING_MAX_TEMPO", "fast")
    with pytest.raises(ValueError, match="must be a number"):
        Settings.from_env()

    monkeypatch.setenv("TIMING_MAX_TEMPO", "0")
    with pytest.raises(ValueError, match="positive number"):
        Settings.from_env()


def test_mix_settings_accept_signed_levels(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    """A level has to be able to be zero or negative, unlike a duration."""

    monkeypatch.setenv("MIX_DIALOGUE_GAIN_DB", "-3.5")
    monkeypatch.setenv("MIX_DUCK_DB", "0")

    settings = Settings.from_env()

    assert settings.mix_dialogue_gain_db == -3.5
    assert settings.mix_duck_db == 0.0
    payload = settings.as_dict()
    assert payload["mix_dialogue_gain_db"] == -3.5
    assert payload["mix_duck_db"] == 0.0


def test_mix_defaults_are_sane(_clean_env: None) -> None:
    settings = Settings.from_env()

    assert settings.mix_dialogue_gain_db == DEFAULT_MIX_DIALOGUE_GAIN_DB
    assert settings.mix_duck_db == DEFAULT_MIX_DUCK_DB
    assert settings.mix_duck_db > 0.0


def test_a_malformed_mix_level_is_rejected(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv("MIX_DUCK_DB", "quiet please")
    with pytest.raises(ValueError, match="must be a number"):
        Settings.from_env()

    monkeypatch.setenv("MIX_DUCK_DB", "inf")
    with pytest.raises(ValueError, match="finite"):
        Settings.from_env()


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_the_syllable_prior_matches_the_adaptation_prompt() -> None:
    """Two copies of one prior, so they must not drift apart."""

    from app.pipeline.dialogue_context import DEFAULT_SYLLABLES_PER_SECOND

    assert (
        DEFAULT_TRANSLATION_SYLLABLES_PER_SECOND == DEFAULT_SYLLABLES_PER_SECOND
    )


def test_the_synthesis_split_threshold_leaves_room_for_a_segment() -> None:
    """A line over this is split before OmniVoice's own ceiling is reached."""

    assert 80 <= DEFAULT_TTS_MAX_INPUT_CHARS <= 400


def test_the_line_gap_is_positive_and_small() -> None:
    """It exists to keep two voices apart without giving up usable silence."""

    assert 0.0 < DEFAULT_TIMING_MIN_LINE_GAP <= 0.5
