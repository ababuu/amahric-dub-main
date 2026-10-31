"""Tests for :mod:`app.pipeline.diarization`.

``pyannote.audio`` is never imported or executed here: the pyannote pipeline and
the ``torch.device`` factory it is moved with are both replaced by in-memory
fakes. The tests therefore need no Hugging Face token, no CUDA device, no model
weights, and no audio backend. The input "audio" is an empty placeholder file
because pyannote itself is mocked and never opens it.
"""

from __future__ import annotations

import contextlib
import json
import dataclasses
import sys
import types
from pathlib import Path

import pytest

from app.config import DEFAULT_DIARIZATION_MODEL, Settings
from app.pipeline import diarization
from app.pipeline.diarization import (
    CrosstalkRegion,
    DiarizationError,
    DiarizationInferenceError,
    DiarizationResult,
    InvalidInputError,
    InvalidSegmentError,
    InvalidSpeakerCountError,
    MissingHuggingFaceTokenError,
    MissingInputError,
    ModelInitializationError,
    SpeakerSegment,
    UnsupportedOutputError,
    diarize,
    diarize_detailed,
)


class FakeTurn:
    """Stand-in for a ``pyannote.core.Segment``."""

    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end

    @property
    def duration(self) -> float:
        return self.end - self.start


def _scene_base(argument: object) -> float:
    """Return the film time a cropped waveform was cut at, or 0 for a whole-file call."""

    waveform = argument.get("waveform") if isinstance(argument, dict) else None
    return float(getattr(waveform, "base", 0.0) or 0.0)


class FakeAnnotation:
    """Stand-in for ``pyannote.core.Annotation``."""

    def __init__(self, turns: list[tuple[float, float, object]]) -> None:
        self.turns = list(turns)
        self.yield_label: bool | None = None

    def itertracks(self, *, yield_label: bool = False):
        self.yield_label = yield_label
        for start, end, speaker_id in self.turns:
            yield FakeTurn(start, end), "track", speaker_id

    def labels(self):
        return sorted({speaker_id for _, _, speaker_id in self.turns})

    def shifted(self, delta: float) -> "FakeAnnotation":
        """Return the same turns moved by ``delta`` seconds."""

        return FakeAnnotation(
            [(start + delta, end + delta, speaker_id) for start, end, speaker_id in self.turns]
        )


class FakeDiarizeOutput:
    """Stand-in for pyannote's ``DiarizeOutput``.

    Attributes are only created when a value is passed, so a test can model a
    pipeline that exposes just the exclusive annotation, or neither.
    """

    def __init__(
        self,
        speaker_diarization: object | None = None,
        exclusive_speaker_diarization: object | None = None,
        speaker_embeddings: object | None = None,
    ) -> None:
        if speaker_diarization is not None:
            self.speaker_diarization = speaker_diarization
        if exclusive_speaker_diarization is not None:
            self.exclusive_speaker_diarization = exclusive_speaker_diarization
        if speaker_embeddings is not None:
            self.speaker_embeddings = speaker_embeddings

    def shifted(self, delta: float) -> "FakeDiarizeOutput":
        """Return the same result with its annotation times moved by ``delta``."""

        def move(annotation: object) -> object:
            shift = getattr(annotation, "shifted", None)
            return shift(delta) if shift is not None else annotation

        return FakeDiarizeOutput(
            speaker_diarization=move(getattr(self, "speaker_diarization", None)),
            exclusive_speaker_diarization=move(
                getattr(self, "exclusive_speaker_diarization", None)
            ),
            speaker_embeddings=getattr(self, "speaker_embeddings", None),
        )


class FakePipeline:
    """Stand-in for ``pyannote.audio.Pipeline`` that records every interaction."""

    # Recorded state, reset by :func:`_patch`.
    requests: list[dict[str, object]] = []
    devices: list[object] = []
    calls: list[tuple[str, dict[str, object]]] = []
    instances: list["FakePipeline"] = []

    # Scripted behaviour, set by individual tests.
    load_error: Exception | None = None
    move_error: Exception | None = None
    infer_error: Exception | None = None
    returns_none: bool = False
    result: object = None
    audio: object = None
    embedding: object = None
    #: Results consumed in call order, for the global pass followed by the scene passes.
    results: list[object] = []
    #: The argument every call received: a path for the global pass, a mapping for a
    #: scene slice.
    files: list[object] = []

    def __init__(self, model: str, *, token: str | None = None, cache_dir=None) -> None:
        self.model = model
        self.token = token
        self.cache_dir = cache_dir
        self.device: object = None
        self._audio = FakePipeline.audio
        self._embedding = FakePipeline.embedding
        FakePipeline.instances.append(self)

    @classmethod
    def from_pretrained(cls, model: str, *, token=None, cache_dir=None):
        cls.requests.append({"model": model, "token": token, "cache_dir": cache_dir})
        if cls.load_error is not None:
            raise cls.load_error
        if cls.returns_none:
            return None
        return cls(model, token=token, cache_dir=cache_dir)

    def to(self, device):
        if FakePipeline.move_error is not None:
            raise FakePipeline.move_error
        self.device = device
        FakePipeline.devices.append(device)
        return self

    def __call__(self, audio_path: str, **kwargs):
        FakePipeline.calls.append((audio_path, kwargs))
        FakePipeline.files.append(audio_path)
        if FakePipeline.infer_error is not None:
            raise FakePipeline.infer_error
        if FakePipeline.results:
            result = FakePipeline.results.pop(0)
            shift = _scene_base(audio_path)
            if shift and hasattr(result, "shifted"):
                # A cropped waveform is timed from zero, exactly as pyannote times the
                # slice it was given; the scripted result is in film time.
                return result.shifted(-shift)
            return result
        return FakePipeline.result


class FakeTorch:
    """Minimal stand-in for the only part of ``torch`` this module uses."""

    requested: list[str] = []

    @staticmethod
    def device(name: str) -> str:
        FakeTorch.requested.append(name)
        return f"torch.device({name})"

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the fakes and reset all recorded state."""

    FakePipeline.requests = []
    FakePipeline.devices = []
    FakePipeline.calls = []
    FakePipeline.instances = []
    FakePipeline.load_error = None
    FakePipeline.move_error = None
    FakePipeline.infer_error = None
    FakePipeline.returns_none = False
    FakePipeline.result = None
    FakePipeline.audio = None
    FakePipeline.embedding = None
    FakePipeline.results = []
    FakePipeline.files = []
    FakeTorch.requested = []

    monkeypatch.setattr(diarization, "Pipeline", FakePipeline)
    monkeypatch.setattr(diarization, "torch", FakeTorch)


def _settings(
    tmp_path: Path,
    *,
    device: str = "cuda",
    token: str | None = "hf_test_token",
    model: str = DEFAULT_DIARIZATION_MODEL,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> Settings:
    return Settings(
        input_dir=tmp_path / "input",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        model_cache_dir=tmp_path / "models",
        huggingface_token=token,
        device=device,
        diarization_model=model,
        diarization_min_speakers=min_speakers,
        diarization_max_speakers=max_speakers,
    )


def _audio(tmp_path: Path, name: str = "speech.wav") -> Path:
    """Create the placeholder input file (pyannote is mocked, so it is never read)."""

    path = tmp_path / name
    path.write_bytes(b"")
    return path


def _pipeline_output(turns) -> FakeDiarizeOutput:
    """Build the ``DiarizeOutput`` pyannote returns for ``turns``.

    Only the exclusive annotation carries the result this module consumes.
    """

    return FakeDiarizeOutput(exclusive_speaker_diarization=FakeAnnotation(turns))


# ---------------------------------------------------------------------------
# segment conversion, ordering, and speakers
# ---------------------------------------------------------------------------


def test_exclusive_turns_are_converted_to_speaker_segments(monkeypatch, tmp_path):
    _patch(monkeypatch)
    annotation = FakeAnnotation([(12.43, 16.82, "SPEAKER_00")])
    FakePipeline.result = FakeDiarizeOutput(exclusive_speaker_diarization=annotation)

    segments = diarize(_audio(tmp_path), settings=_settings(tmp_path))

    assert segments == [SpeakerSegment(speaker_id="SPEAKER_00", start=12.43, end=16.82)]
    assert annotation.yield_label is True  # labels must be requested, not silently dropped


def test_segments_are_sorted_chronologically(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output(
        [
            (17.10, 20.54, "SPEAKER_01"),
            (12.43, 16.82, "SPEAKER_00"),
            (30.00, 31.00, "SPEAKER_00"),
        ]
    )

    segments = diarize(_audio(tmp_path), settings=_settings(tmp_path))

    assert [(segment.speaker_id, segment.start) for segment in segments] == [
        ("SPEAKER_00", 12.43),
        ("SPEAKER_01", 17.10),
        ("SPEAKER_00", 30.00),
    ]


def test_multiple_speakers_are_preserved_with_their_timestamps(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output(
        [
            (0.50, 2.25, "SPEAKER_00"),
            (2.25, 4.00, "SPEAKER_02"),
            (4.00, 6.75, "SPEAKER_01"),
        ]
    )

    segments = diarize(_audio(tmp_path), settings=_settings(tmp_path))

    assert [segment.speaker_id for segment in segments] == [
        "SPEAKER_00",
        "SPEAKER_02",
        "SPEAKER_01",
    ]
    assert [segment.duration for segment in segments] == [1.75, 1.75, 2.75]


def test_no_speech_yields_an_empty_list(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([])

    assert diarize(_audio(tmp_path), settings=_settings(tmp_path)) == []


# ---------------------------------------------------------------------------
# pyannote output: the exclusive annotation is authoritative
# ---------------------------------------------------------------------------


def test_exclusive_annotation_wins_when_both_are_present(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = FakeDiarizeOutput(
        speaker_diarization=FakeAnnotation([(1.0, 2.0, "SPEAKER_00")]),
        exclusive_speaker_diarization=FakeAnnotation([(9.0, 10.0, "SPEAKER_09")]),
    )

    assert diarize(_audio(tmp_path), settings=_settings(tmp_path)) == [
        SpeakerSegment("SPEAKER_09", 9.0, 10.0)
    ]


def test_overlapping_speaker_diarization_is_not_used(monkeypatch, tmp_path):
    _patch(monkeypatch)
    # The non-exclusive field contains two overlapping tracks. The exclusive field
    # resolves them to one speaker, and that is what must come back.
    FakePipeline.result = FakeDiarizeOutput(
        speaker_diarization=FakeAnnotation(
            [(1.0, 3.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")]
        ),
        exclusive_speaker_diarization=FakeAnnotation([(1.0, 4.0, "SPEAKER_00")]),
    )

    segments = diarize(_audio(tmp_path), settings=_settings(tmp_path))

    assert segments == [SpeakerSegment("SPEAKER_00", 1.0, 4.0)]
    assert len(segments) == 1  # the same audio is not attributed twice


def test_exclusive_annotation_is_read_from_a_pipeline_that_only_has_it(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = FakeDiarizeOutput(
        exclusive_speaker_diarization=FakeAnnotation([(5.0, 6.0, "SPEAKER_01")])
    )

    assert diarize(_audio(tmp_path), settings=_settings(tmp_path)) == [
        SpeakerSegment("SPEAKER_01", 5.0, 6.0)
    ]


def test_regular_speaker_diarization_alone_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    # Only the non-exclusive field is present: not good enough, and no fallback.
    FakePipeline.result = FakeDiarizeOutput(
        speaker_diarization=FakeAnnotation([(1.0, 2.0, "SPEAKER_00")])
    )

    with pytest.raises(UnsupportedOutputError, match="exclusive_speaker_diarization"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_output_without_the_exclusive_annotation_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = FakeDiarizeOutput()

    with pytest.raises(UnsupportedOutputError, match="exclusive_speaker_diarization"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_malformed_exclusive_annotation_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = FakeDiarizeOutput(exclusive_speaker_diarization=object())

    with pytest.raises(UnsupportedOutputError, match="exclusive_speaker_diarization"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_bare_annotation_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    # A ``legacy=True`` pipeline returns the bare, non-exclusive annotation.
    FakePipeline.result = FakeAnnotation([(1.0, 2.0, "SPEAKER_00")])

    with pytest.raises(UnsupportedOutputError, match="exclusive_speaker_diarization"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_unexpected_pyannote_output_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = object()

    with pytest.raises(UnsupportedOutputError, match="exclusive_speaker_diarization"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


# ---------------------------------------------------------------------------
# invalid segments
# ---------------------------------------------------------------------------


def test_empty_speaker_id_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([(1.0, 2.0, "")])

    with pytest.raises(InvalidSegmentError, match="speaker_id"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_whitespace_only_speaker_id_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([(1.0, 2.0, "   ")])

    with pytest.raises(InvalidSegmentError, match="speaker_id"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_non_string_speaker_id_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([(1.0, 2.0, None)])

    with pytest.raises(InvalidSegmentError, match="speaker_id"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_end_before_start_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([(5.0, 2.0, "SPEAKER_00")])

    with pytest.raises(InvalidSegmentError, match="greater than start"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_zero_length_turn_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([(2.0, 2.0, "SPEAKER_00")])

    with pytest.raises(InvalidSegmentError, match="greater than start"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_negative_start_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([(-1.0, 3.0, "SPEAKER_00")])

    with pytest.raises(InvalidSegmentError, match="start must be"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_non_finite_timestamp_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([(0.0, float("inf"), "SPEAKER_00")])

    with pytest.raises(InvalidSegmentError, match="finite"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_speaker_segment_normalises_integers_to_float():
    segment = SpeakerSegment(speaker_id="SPEAKER_00", start=1, end=3)

    assert isinstance(segment.start, float)
    assert isinstance(segment.end, float)
    assert segment.duration == 2.0


def test_speaker_segment_is_immutable():
    segment = SpeakerSegment("SPEAKER_00", 0.0, 1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        segment.speaker_id = "SPEAKER_01"


def test_speaker_segment_rejects_non_string_speaker_id():
    with pytest.raises(InvalidSegmentError, match="speaker_id"):
        SpeakerSegment(speaker_id=None, start=0.0, end=1.0)


def test_speaker_segment_rejects_non_numeric_start():
    with pytest.raises(InvalidSegmentError, match="number of seconds"):
        SpeakerSegment(speaker_id="SPEAKER_00", start="0", end=1.0)


def test_segment_and_count_errors_are_also_value_errors():
    assert issubclass(InvalidSegmentError, ValueError)
    assert issubclass(InvalidSpeakerCountError, ValueError)
    assert issubclass(InvalidSegmentError, DiarizationError)


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def test_missing_input_file_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)

    with pytest.raises(MissingInputError, match="not found"):
        diarize(tmp_path / "does-not-exist.wav", settings=_settings(tmp_path))

    assert FakePipeline.requests == []  # the model is never touched


def test_directory_input_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)
    directory = tmp_path / "a-directory"
    directory.mkdir()

    with pytest.raises(InvalidInputError, match="not a file"):
        diarize(directory, settings=_settings(tmp_path))

    assert FakePipeline.requests == []


# ---------------------------------------------------------------------------
# configuration forwarding
# ---------------------------------------------------------------------------


def test_pipeline_is_loaded_with_the_configured_model_token_and_cache(monkeypatch, tmp_path):
    _patch(monkeypatch)
    settings = _settings(tmp_path, model="pyannote/some-other-model")
    FakePipeline.result = _pipeline_output([])

    diarize(_audio(tmp_path), settings=settings)

    assert FakePipeline.requests == [
        {
            "model": "pyannote/some-other-model",
            "token": "hf_test_token",
            "cache_dir": settings.model_cache_dir,
        }
    ]


def test_pipeline_is_moved_to_the_configured_device(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([])

    diarize(_audio(tmp_path), settings=_settings(tmp_path, device="cuda:1"))

    # Exactly one device is requested, and it is the configured one: there is no
    # silent fall back to cpu.
    assert FakeTorch.requested == ["cuda:1"]
    assert FakePipeline.devices == ["torch.device(cuda:1)"]


def test_audio_path_is_passed_to_the_pipeline(monkeypatch, tmp_path):
    _patch(monkeypatch)
    source = _audio(tmp_path)
    FakePipeline.result = _pipeline_output([])

    diarize(source, settings=_settings(tmp_path))

    assert FakePipeline.calls == [(str(source), {})]


def test_speaker_count_hints_are_forwarded(monkeypatch, tmp_path):
    _patch(monkeypatch)
    source = _audio(tmp_path)
    FakePipeline.result = _pipeline_output([])

    diarize(source, settings=_settings(tmp_path), min_speakers=2, max_speakers=6)

    assert FakePipeline.calls == [(str(source), {"min_speakers": 2, "max_speakers": 6})]


def test_unusable_speaker_count_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)

    with pytest.raises(InvalidSpeakerCountError, match="num_speakers"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path), num_speakers=0)

    assert FakePipeline.requests == []


def test_min_speakers_above_max_speakers_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)

    with pytest.raises(InvalidSpeakerCountError, match="must not exceed"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path), min_speakers=5, max_speakers=2)

    assert FakePipeline.requests == []


# ---------------------------------------------------------------------------
# authentication and model failures
# ---------------------------------------------------------------------------


def test_missing_huggingface_token_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)

    with pytest.raises(MissingHuggingFaceTokenError, match="HUGGINGFACE_TOKEN"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path, token=None))

    assert FakePipeline.requests == []


def test_blank_huggingface_token_is_rejected(monkeypatch, tmp_path):
    _patch(monkeypatch)

    with pytest.raises(MissingHuggingFaceTokenError, match="HUGGINGFACE_TOKEN"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path, token="   "))

    assert FakePipeline.requests == []


def test_missing_pyannote_runtime_is_reported(monkeypatch, tmp_path):
    _patch(monkeypatch)
    monkeypatch.setattr(diarization, "Pipeline", None)

    with pytest.raises(ModelInitializationError, match="not installed"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_pipeline_load_failure_is_wrapped(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.load_error = RuntimeError("gated repo, access denied")

    with pytest.raises(ModelInitializationError, match="could not load") as excinfo:
        diarize(_audio(tmp_path), settings=_settings(tmp_path))

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_unavailable_checkpoint_is_reported(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.returns_none = True

    with pytest.raises(ModelInitializationError, match="could not load"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path))


def test_device_move_failure_is_wrapped(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.move_error = RuntimeError("CUDA is not available")

    with pytest.raises(ModelInitializationError, match="could not move") as excinfo:
        diarize(_audio(tmp_path), settings=_settings(tmp_path))

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_inference_failure_is_wrapped(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.infer_error = RuntimeError("boom")

    with pytest.raises(DiarizationInferenceError, match="diarization failed") as excinfo:
        diarize(_audio(tmp_path), settings=_settings(tmp_path))

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_a_cuda_architecture_compile_failure_explains_itself(monkeypatch, tmp_path):
    """NVRTC rejecting the GPU arch is actionable, so say what to change."""

    _patch(monkeypatch)
    FakePipeline.infer_error = RuntimeError(
        "nvrtc: error: invalid value for --gpu-architecture (-arch)"
    )

    with pytest.raises(DiarizationInferenceError, match="compute capability") as excinfo:
        diarize(_audio(tmp_path), settings=_settings(tmp_path))

    assert "ALLOW_TORCH_COMPILE" in str(excinfo.value)


# ---------------------------------------------------------------------------
# crosstalk from the overlap-aware view
# ---------------------------------------------------------------------------


def _crosstalk_output(exclusive, overlapping) -> FakeDiarizeOutput:
    """Build the ``DiarizeOutput`` pyannote actually returns for both views."""

    return FakeDiarizeOutput(
        exclusive_speaker_diarization=FakeAnnotation(exclusive),
        speaker_diarization=FakeAnnotation(overlapping),
    )


def test_diarize_detailed_reports_both_turns_and_crosstalk(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _crosstalk_output(
        exclusive=[(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")],
        overlapping=[(0.0, 2.0, "SPEAKER_00"), (1.0, 3.0, "SPEAKER_01")],
    )

    result = diarize_detailed(_audio(tmp_path), settings=_settings(tmp_path))

    assert isinstance(result, DiarizationResult)
    # Attribution still comes from the exclusive view, and only from it.
    assert result.turns == (
        SpeakerSegment("SPEAKER_00", 0.0, 2.0),
        SpeakerSegment("SPEAKER_01", 2.0, 4.0),
    )
    assert result.crosstalk == (
        CrosstalkRegion(("SPEAKER_00", "SPEAKER_01"), start=1.0, end=2.0),
    )
    assert result.speaker_ids == ("SPEAKER_00", "SPEAKER_01")
    assert result.crosstalk_seconds == 1.0


def test_confident_clean_turns_are_reassigned_but_short_and_overlapping_turns_are_not(
    monkeypatch, tmp_path
):
    _patch(monkeypatch)

    class FakeWaveform:
        def unsqueeze(self, _dimension):
            return self

    class FakeAudio:
        def __init__(self):
            self.crops = []

        def crop(self, _audio_file, segment, *, mode):
            assert mode == "pad"
            self.crops.append((segment.start, segment.end))
            return FakeWaveform(), 16000

    FakePipeline.audio = FakeAudio()
    FakePipeline.embedding = lambda _waveform: [[0.1, 1.0]]
    FakePipeline.result = FakeDiarizeOutput(
        speaker_diarization=FakeAnnotation(
            [(0.0, 2.0, "SPEAKER_00"), (2.0, 5.0, "SPEAKER_00"), (4.0, 4.5, "SPEAKER_01")]
        ),
        exclusive_speaker_diarization=FakeAnnotation(
            [
                (0.0, 2.0, "SPEAKER_00"),
                (2.1, 2.3, "SPEAKER_00"),
                (3.0, 5.0, "SPEAKER_00"),
            ]
        ),
        speaker_embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    core_module = types.ModuleType("pyannote.core")
    core_module.Segment = FakeTurn
    monkeypatch.setitem(sys.modules, "pyannote.core", core_module)
    # This test is about the verification pass alone; the suspect-window pass is
    # exercised separately and would legitimately rewrite the overlapped turn.
    settings = dataclasses.replace(_settings(tmp_path), diarization_scene_mode="off")

    result = diarize_detailed(_audio(tmp_path), settings=settings)

    assert result.turns == (
        SpeakerSegment("SPEAKER_01", 0.0, 2.0),
        SpeakerSegment("SPEAKER_00", 2.1, 2.3),
        SpeakerSegment("SPEAKER_00", 3.0, 5.0),
    )
    assert len(result.corrections) == 1
    correction = result.corrections[0]
    assert correction.original_speaker_id == "SPEAKER_00"
    assert correction.corrected_speaker_id == "SPEAKER_01"
    assert correction.corrected_similarity == pytest.approx(1.0 / (1.01**0.5))
    assert correction.similarity_margin == pytest.approx(
        1.0 / (1.01**0.5) - 0.1 / (1.01**0.5)
    )
    assert FakePipeline.audio.crops == [(0.0, 2.0)]


def test_diarize_returns_only_the_turns(monkeypatch, tmp_path):
    """The existing contract is unchanged: crosstalk is additive information."""

    _patch(monkeypatch)
    FakePipeline.result = _crosstalk_output(
        exclusive=[(0.0, 2.0, "SPEAKER_00")],
        overlapping=[(0.0, 2.0, "SPEAKER_00"), (0.5, 1.5, "SPEAKER_01")],
    )

    assert diarize(_audio(tmp_path), settings=_settings(tmp_path)) == [
        SpeakerSegment("SPEAKER_00", 0.0, 2.0)
    ]


def test_a_result_without_the_overlap_view_reports_no_crosstalk(monkeypatch, tmp_path):
    """A missing overlap view means "nothing to report", not an error."""

    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([(0.0, 2.0, "SPEAKER_00")])

    result = diarize_detailed(_audio(tmp_path), settings=_settings(tmp_path))

    assert result.crosstalk == ()
    assert result.turns == (SpeakerSegment("SPEAKER_00", 0.0, 2.0),)


def test_three_simultaneous_speakers_are_one_region(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _crosstalk_output(
        exclusive=[(0.0, 4.0, "SPEAKER_00")],
        overlapping=[
            (0.0, 4.0, "SPEAKER_00"),
            (1.0, 3.0, "SPEAKER_01"),
            (1.5, 2.5, "SPEAKER_02"),
        ],
    )

    result = diarize_detailed(_audio(tmp_path), settings=_settings(tmp_path))

    assert result.crosstalk == (
        CrosstalkRegion(("SPEAKER_00", "SPEAKER_01"), start=1.0, end=1.5),
        CrosstalkRegion(("SPEAKER_00", "SPEAKER_01", "SPEAKER_02"), start=1.5, end=2.5),
        CrosstalkRegion(("SPEAKER_00", "SPEAKER_01"), start=2.5, end=3.0),
    )


def test_touching_regions_with_the_same_speakers_are_merged(monkeypatch, tmp_path):
    """A long stretch of crosstalk is one region, not one per turn boundary."""

    _patch(monkeypatch)
    FakePipeline.result = _crosstalk_output(
        exclusive=[(0.0, 6.0, "SPEAKER_00")],
        overlapping=[
            (0.0, 6.0, "SPEAKER_00"),
            (1.0, 2.5, "SPEAKER_01"),
            (2.5, 4.0, "SPEAKER_01"),
        ],
    )

    result = diarize_detailed(_audio(tmp_path), settings=_settings(tmp_path))

    assert result.crosstalk == (
        CrosstalkRegion(("SPEAKER_00", "SPEAKER_01"), start=1.0, end=4.0),
    )


def test_non_overlapping_speakers_report_no_crosstalk(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _crosstalk_output(
        exclusive=[(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")],
        overlapping=[(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")],
    )

    assert diarize_detailed(_audio(tmp_path), settings=_settings(tmp_path)).crosstalk == ()


def test_a_crosstalk_region_needs_two_speakers():
    with pytest.raises(InvalidSegmentError, match="at least two"):
        CrosstalkRegion(("SPEAKER_00",), start=0.0, end=1.0)

    with pytest.raises(InvalidSegmentError, match="at least two"):
        CrosstalkRegion(("SPEAKER_00", "  "), start=0.0, end=1.0)


def test_a_crosstalk_region_validates_its_interval():
    with pytest.raises(InvalidSegmentError, match="greater than start"):
        CrosstalkRegion(("A", "B"), start=2.0, end=2.0)

    with pytest.raises(InvalidSegmentError, match="start must be"):
        CrosstalkRegion(("A", "B"), start=-1.0, end=1.0)


def test_a_crosstalk_region_normalises_speaker_ids():
    region = CrosstalkRegion(("b", "A", "A"), start=0, end=1)

    assert region.speaker_ids == ("A", "b")
    assert region.duration == 1.0


def test_crosstalk_regions_are_deduplicated_case_sensitively(monkeypatch, tmp_path):
    """Repeated labels are collapsed, so a region cannot claim a speaker twice."""

    _patch(monkeypatch)
    FakePipeline.result = _crosstalk_output(
        exclusive=[(0.0, 3.0, "SPEAKER_00")],
        overlapping=[(0.0, 1.0, "SPEAKER_00"), (0.5, 1.5, "SPEAKER_00")],
    )

    assert diarize_detailed(_audio(tmp_path), settings=_settings(tmp_path)).crosstalk == ()


# ---------------------------------------------------------------------------
# speaker-count hints from settings
# ---------------------------------------------------------------------------


def test_configured_speaker_counts_are_used_as_a_fallback(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([(0.0, 1.0, "SPEAKER_00")])

    diarize(
        _audio(tmp_path),
        settings=_settings(tmp_path, min_speakers=4, max_speakers=9),
    )

    assert FakePipeline.calls[0][1] == {"min_speakers": 4, "max_speakers": 9}


def test_explicit_speaker_counts_win_over_the_configured_ones(monkeypatch, tmp_path):
    """A caller that knows better for one film is never overridden by the env."""

    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([(0.0, 1.0, "SPEAKER_00")])

    diarize(
        _audio(tmp_path),
        settings=_settings(tmp_path, min_speakers=4, max_speakers=9),
        min_speakers=2,
        max_speakers=3,
    )

    assert FakePipeline.calls[0][1] == {"min_speakers": 2, "max_speakers": 3}


def test_num_speakers_still_wins_over_everything(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([(0.0, 1.0, "SPEAKER_00")])

    diarize(
        _audio(tmp_path),
        settings=_settings(tmp_path, min_speakers=4, max_speakers=9),
        num_speakers=5,
    )

    assert FakePipeline.calls[0][1] == {"num_speakers": 5, "min_speakers": 4, "max_speakers": 9}


def test_configured_counts_are_validated_before_the_model_is_touched(monkeypatch, tmp_path):
    _patch(monkeypatch)
    FakePipeline.result = _pipeline_output([(0.0, 1.0, "SPEAKER_00")])

    with pytest.raises(InvalidSpeakerCountError, match="must not exceed"):
        diarize(_audio(tmp_path), settings=_settings(tmp_path, min_speakers=9, max_speakers=4))

    assert FakePipeline.requests == []


# ---------------------------------------------------------------------------
# a cluster that holds two voices
# ---------------------------------------------------------------------------


def _embedding_output(turns: list[tuple[float, float, str]], vectors: dict[float, list[float]]):
    """Script the pipeline so each turn is embedded as ``vectors[start]``."""

    class FakeWaveform:
        def __init__(self, vector: list[float]) -> None:
            self.vector = vector

        def unsqueeze(self, _dimension):
            return self

    class FakeAudio:
        def __init__(self) -> None:
            self.crops: list[tuple[float, float]] = []

        def crop(self, _audio_file, segment, *, mode):
            assert mode == "pad"
            self.crops.append((segment.start, segment.end))
            return FakeWaveform(vectors[segment.start]), 16000

    FakePipeline.audio = FakeAudio()
    FakePipeline.embedding = lambda waveform: [waveform.vector]
    FakePipeline.result = _pipeline_output(turns)
    return FakePipeline.audio


def _install_fake_pyannote_core(monkeypatch: pytest.MonkeyPatch) -> None:
    core_module = types.ModuleType("pyannote.core")
    core_module.Segment = FakeTurn
    monkeypatch.setitem(sys.modules, "pyannote.core", core_module)


class _Waveform:
    """A crop the embedder turns back into the vector the test scripted.

    ``base`` is the film time the crop starts at, which is what pyannote's own
    window-relative timestamps are measured from.
    """

    def __init__(self, vector: list[float], base: float = 0.0) -> None:
        self.vector = vector
        self.base = base

    def unsqueeze(self, _dimension):  # noqa: ANN201
        return self


class _ScriptedAudio:
    """Return a scripted embedding for a turn, and a window slice for anything else."""

    def __init__(self, vectors: dict[tuple[float, float], list[float]]) -> None:
        self.vectors = vectors
        self.turns: list[tuple[float, float]] = []
        self.padded: list[tuple[float, float]] = []
        self.slices: list[tuple[float, float]] = []

    def crop(self, _audio_file, segment, *, mode):  # noqa: ANN001, ANN201
        assert mode == "pad"
        key = (round(segment.start, 6), round(segment.end, 6))
        if key in self.vectors:
            self.turns.append(key)
            return _Waveform(self.vectors[key], base=segment.start), 16000
        # A turn shorter than the embedding window is embedded inside a padded window;
        # pyannote hears the turn, so that is the voiceprint the window returns. Window
        # slices are longer than any padded window and are left untouched.
        if segment.end - segment.start <= 1.6:
            # Only scripted turns that fit entirely inside the padded window speak for
            # it; the most specific (shortest) such turn wins.
            fits = [
                (end - start, (start, end))
                for (start, end) in self.vectors
                if start >= segment.start - 1e-6 and end <= segment.end + 1e-6
            ]
            if fits:
                fits.sort()
                self.padded.append(key)
                return _Waveform(self.vectors[fits[0][1]], base=segment.start), 16000
        self.slices.append(key)
        return _Waveform([0.0, 0.0, 0.0], base=segment.start), 16000


def _scripted(
    global_turns: list[tuple[float, float, str]],
    local_results: list[list[tuple[float, float, str]]],
    vectors: dict[tuple[float, float], list[float]],
) -> _ScriptedAudio:
    """Script the global pass and then each suspect window, in call order."""

    FakePipeline.results = [_pipeline_output(global_turns)] + [
        _pipeline_output(turns) for turns in local_results
    ]
    audio = _ScriptedAudio(vectors)
    FakePipeline.audio = audio
    FakePipeline.embedding = lambda waveform: [waveform.vector]
    return audio


def _centroid_output(
    exclusive: list[tuple[float, float, str]],
    overlapping: list[tuple[float, float, str]],
    embeddings: list[list[float]],
) -> FakeDiarizeOutput:
    return FakeDiarizeOutput(
        speaker_diarization=FakeAnnotation(overlapping),
        exclusive_speaker_diarization=FakeAnnotation(exclusive),
        speaker_embeddings=embeddings,
    )


# ---------------------------------------------------------------------------
# global re-partitioning: the scattered character
# ---------------------------------------------------------------------------

_VA = [1.0, 0.0, 0.0, 0.0]
_VB = [0.0, 1.0, 0.0, 0.0]
_VC = [0.0, 0.0, 1.0, 0.0]


def _scattered_film():
    """A, B and C where C's lines carry A's label three times and B's once.

    This is the shape the artifacts showed: one character's dialogue spread over both
    provisional clusters, in short performances, which no per-cluster split can undo.
    """

    turns = [
        (0.0, 5.0, "SPEAKER_00"),
        (6.0, 11.0, "SPEAKER_01"),
        (12.0, 13.1, "SPEAKER_00"),
        (15.0, 16.1, "SPEAKER_00"),
        (18.0, 19.1, "SPEAKER_00"),
        (40.0, 45.0, "SPEAKER_00"),
        (46.0, 51.0, "SPEAKER_01"),
        (60.0, 61.1, "SPEAKER_01"),
        (80.0, 85.0, "SPEAKER_00"),
        (86.0, 91.0, "SPEAKER_01"),
    ]
    vectors = {
        (0.0, 5.0): _VA,
        (6.0, 11.0): _VB,
        (12.0, 13.1): _VC,
        (15.0, 16.1): _VC,
        (18.0, 19.1): _VC,
        (40.0, 45.0): _VA,
        (46.0, 51.0): _VB,
        (60.0, 61.1): _VC,
        (80.0, 85.0): _VA,
        (86.0, 91.0): _VB,
    }
    return turns, vectors


def test_a_voice_scattered_over_two_clusters_becomes_one_identity(monkeypatch, tmp_path):
    """The three-character regression, at the level where the bug lives.

    pyannote reports two speakers. Character C speaks for 4.4 s in four short
    performances, three of which carry SPEAKER_00 and one SPEAKER_01. The re-partitioning
    must produce three identities, put every one of C's performances in the same one, and
    leave A and B apart.
    """

    _patch(monkeypatch)
    turns, vectors = _scattered_film()
    _scripted(turns, [], vectors)
    _install_fake_pyannote_core(monkeypatch)

    result = diarize_detailed(_audio(tmp_path), settings=_settings(tmp_path))

    # pyannote's own answer is kept, untouched, for comparison.
    assert result.raw_speaker_ids == ("SPEAKER_00", "SPEAKER_01")
    # The evidence supports a third character - discovered, not declared.
    assert result.speaker_ids == ("SPEAKER_00", "SPEAKER_01", "SPEAKER_02")
    assert result.discovered_speakers == ("SPEAKER_02",)

    by_start = {round(segment.start, 2): segment.speaker_id for segment in result.turns}
    scattered = [12.0, 15.0, 18.0, 60.0]
    assert {by_start[start] for start in scattered} == {"SPEAKER_02"}
    # A and B are not merged, and keep pyannote's ids.
    assert {by_start[start] for start in (0.0, 40.0, 80.0)} == {"SPEAKER_00"}
    assert {by_start[start] for start in (6.0, 46.0, 86.0)} == {"SPEAKER_01"}

    # Every move is recorded with the evidence that justified it.
    assert len(result.reassignments) == 4
    assert {item.corrected_speaker_id for item in result.reassignments} == {"SPEAKER_02"}
    assert all(item.margin > 0 for item in result.reassignments)

    # The partition that produced this is reported, including what was refused.
    assert result.reclustering is not None
    assert result.reclustering.accepted is True
    assert result.reclustering.k == 3
    assert any(candidate.chosen for candidate in result.reclustering.candidates)


def test_the_registry_is_rebuilt_once_from_the_final_partition(monkeypatch, tmp_path):
    """Evidence is counted once per identity: the previous run reported more speech
    than the film contained."""

    _patch(monkeypatch)
    turns, vectors = _scattered_film()
    _scripted(turns, [], vectors)
    _install_fake_pyannote_core(monkeypatch)

    result = diarize_detailed(_audio(tmp_path), settings=_settings(tmp_path))

    registry = json.loads(
        Path(result.registry_path).read_text(encoding="utf-8")
    )
    asserted = sum(speaker["evidence_seconds"] for speaker in registry["speakers"])
    assert asserted == pytest.approx(result.unit_seconds, abs=1e-3)
    assert len(registry["speakers"]) == 3


def test_a_single_voice_is_not_turned_into_a_cast(monkeypatch, tmp_path):
    """Noisy short performances of one voice must not become new characters."""

    _patch(monkeypatch)
    import numpy as np

    rng = np.random.default_rng(3)
    turns = []
    vectors = {}
    for index in range(8):
        start = 3.0 * index
        turns.append((start, start + 1.2, "SPEAKER_00"))
        noise = rng.normal(0.0, 0.25, size=4)
        vectors[(start, start + 1.2)] = [1.0 + noise[0], noise[1], noise[2], noise[3]]
    _scripted(turns, [], vectors)
    _install_fake_pyannote_core(monkeypatch)

    result = diarize_detailed(_audio(tmp_path), settings=_settings(tmp_path))

    assert result.speaker_ids == ("SPEAKER_00",)
    assert result.reassignments == ()
    assert result.reclustering is not None
    assert result.reclustering.candidates
    assert any(
        "bis" in candidate.reason or "reproduc" in candidate.reason or "silhouette" in candidate.reason
        for candidate in result.reclustering.candidates
    )


def test_an_undecidable_performance_keeps_pyannotes_identity(monkeypatch, tmp_path):
    """Never silently assign an uncertain voice: it stays put and is reported."""

    _patch(monkeypatch)
    turns = [
        (0.0, 5.0, "SPEAKER_00"),
        (6.0, 11.0, "SPEAKER_01"),
        (20.0, 25.0, "SPEAKER_00"),
        (26.0, 31.0, "SPEAKER_01"),
        (40.0, 45.0, "SPEAKER_00"),
        (46.0, 51.0, "SPEAKER_01"),
        (60.0, 62.0, "SPEAKER_00"),
    ]
    nearly = [0.55, 0.62, 0.0, 0.0]
    vectors = {
        (0.0, 5.0): _VA,
        (6.0, 11.0): _VB,
        (20.0, 25.0): _VA,
        (26.0, 31.0): _VB,
        (40.0, 45.0): _VA,
        (46.0, 51.0): _VB,
        (60.0, 62.0): nearly,
    }
    _scripted(turns, [], vectors)
    _install_fake_pyannote_core(monkeypatch)

    result = diarize_detailed(_audio(tmp_path), settings=_settings(tmp_path))

    assert result.turns[-1].speaker_id == "SPEAKER_00"
    assert result.uncertain_speakers
    assert any(
        item.speaker_id == "SPEAKER_00" for item in result.uncertain_speakers
    )
    assert all(item.reason for item in result.uncertain_speakers)


def test_the_repartition_can_be_switched_off(monkeypatch, tmp_path):
    """A run that wants pyannote's clustering exactly can have it."""

    _patch(monkeypatch)
    turns, vectors = _scattered_film()
    _scripted(turns, [], vectors)
    _install_fake_pyannote_core(monkeypatch)
    settings = dataclasses.replace(
        _settings(tmp_path), diarization_split_mixed_speakers=False
    )

    result = diarize_detailed(_audio(tmp_path), settings=settings)

    assert result.speaker_ids == ("SPEAKER_00", "SPEAKER_01")
    assert result.reassignments == ()
    assert result.reclustering is not None
    assert "switched off" in result.reclustering.reason


def test_too_few_performances_keep_the_model_labels(monkeypatch, tmp_path):
    _patch(monkeypatch)
    turns = [(0.0, 5.0, "SPEAKER_00"), (10.0, 15.0, "SPEAKER_01")]
    _scripted(turns, [], {(0.0, 5.0): _VA, (10.0, 15.0): _VB})
    _install_fake_pyannote_core(monkeypatch)

    result = diarize_detailed(_audio(tmp_path), settings=_settings(tmp_path))

    assert result.speaker_ids == ("SPEAKER_00", "SPEAKER_01")
    assert result.reclustering is not None
    assert result.reclustering.accepted is False
    assert "reliable performance" in result.reclustering.reason


# ---------------------------------------------------------------------------
# suspect windows
# ---------------------------------------------------------------------------


def test_a_suspect_window_can_move_dialogue_between_existing_identities(
    monkeypatch, tmp_path
):
    """A window is allowed to reassign, not only to add a new character."""

    _patch(monkeypatch)
    turns = [
        (0.0, 5.0, "SPEAKER_00"),
        (6.0, 11.0, "SPEAKER_01"),
        (20.0, 25.0, "SPEAKER_00"),
        (26.0, 31.0, "SPEAKER_01"),
        (40.0, 45.0, "SPEAKER_00"),
        (46.0, 51.0, "SPEAKER_01"),
        (60.0, 62.0, "SPEAKER_00"),
    ]
    halfway = [0.6, 0.6, 0.0, 0.0]
    vectors = {
        (0.0, 5.0): _VA,
        (6.0, 11.0): _VB,
        (20.0, 25.0): _VA,
        (26.0, 31.0): _VB,
        (40.0, 45.0): _VA,
        (46.0, 51.0): _VB,
        (60.0, 62.0): halfway,
        # The window's own diarization hears that turn as two clean fragments of B.
        (60.0, 61.0): _VB,
        (61.0, 62.0): _VB,
    }
    _scripted(
        turns,
        [[(60.0, 61.0, "SPEAKER_00"), (61.0, 62.0, "SPEAKER_00")]],
        vectors,
    )
    _install_fake_pyannote_core(monkeypatch)

    result = diarize_detailed(_audio(tmp_path), settings=_settings(tmp_path))

    assert result.windows, "an ambiguous turn should open a suspect window"
    assert [item.mode for item in result.windows] == ["window"]
    moved = [segment for segment in result.turns if segment.start >= 60.0]
    assert {segment.speaker_id for segment in moved} == {"SPEAKER_01"}


def test_a_window_that_cannot_be_diarized_keeps_the_attribution(monkeypatch, tmp_path):
    _patch(monkeypatch)
    turns = [
        (0.0, 5.0, "SPEAKER_00"),
        (6.0, 11.0, "SPEAKER_01"),
        (20.0, 25.0, "SPEAKER_00"),
        (26.0, 31.0, "SPEAKER_01"),
        (40.0, 45.0, "SPEAKER_00"),
        (46.0, 51.0, "SPEAKER_01"),
        (60.0, 62.0, "SPEAKER_00"),
    ]
    vectors = {
        (0.0, 5.0): _VA,
        (6.0, 11.0): _VB,
        (20.0, 25.0): _VA,
        (26.0, 31.0): _VB,
        (40.0, 45.0): _VA,
        (46.0, 51.0): _VB,
        (60.0, 62.0): [0.6, 0.6, 0.0, 0.0],
    }
    _scripted(turns, [], vectors)
    FakePipeline.results = [
        _pipeline_output(turns),
        object(),
    ]
    _install_fake_pyannote_core(monkeypatch)

    result = diarize_detailed(_audio(tmp_path), settings=_settings(tmp_path))

    assert [item.mode for item in result.windows] == ["failed"]
    assert result.turns[-1].speaker_id == "SPEAKER_00"


def test_the_window_pass_can_be_switched_off(monkeypatch, tmp_path):
    _patch(monkeypatch)
    turns = [(0.0, 5.0, "SPEAKER_00"), (20.0, 25.0, "SPEAKER_01")]
    audio = _scripted(turns, [], {(0.0, 5.0): _VA, (20.0, 25.0): _VB})
    _install_fake_pyannote_core(monkeypatch)
    settings = dataclasses.replace(_settings(tmp_path), diarization_scene_mode="off")

    result = diarize_detailed(_audio(tmp_path), settings=settings)

    assert result.windows == ()
    assert audio.slices == []

def test_the_artifact_scene_keeps_character_c_in_one_identity(monkeypatch, tmp_path):
    """The regression the artifacts describe, at the timings they measured.

    Two speakers are reported. Character C speaks six short lines - 19.8 s, 105-111 s and
    214.6 s - five of which pyannote labels SPEAKER_00 and one SPEAKER_01. C must end up
    as one identity, and A and B must stay apart.
    """

    _patch(monkeypatch)
    turns = [
        (0.0, 2.0, "SPEAKER_00"),
        (3.0, 5.0, "SPEAKER_01"),
        (14.97, 19.31, "SPEAKER_01"),
        (19.77, 20.43, "SPEAKER_00"),   # C
        (20.0, 22.0, "SPEAKER_00"),
        (24.63, 25.13, "SPEAKER_00"),
        (105.75, 106.29, "SPEAKER_00"),  # C
        (107.78, 109.10, "SPEAKER_00"),  # C
        (109.89, 110.47, "SPEAKER_00"),  # C
        (110.66, 111.46, "SPEAKER_01"),  # C
        (112.23, 113.89, "SPEAKER_01"),
        (114.09, 116.35, "SPEAKER_01"),
        (116.67, 121.35, "SPEAKER_01"),
        (201.52, 203.9, "SPEAKER_00"),
        (213.57, 214.37, "SPEAKER_00"),
        (214.59, 215.51, "SPEAKER_00"),  # C
    ]
    characters = {
        (0.0, 2.0): _VA,
        (3.0, 5.0): _VB,
        (14.97, 19.31): _VB,
        (20.0, 22.0): _VA,
        (24.63, 25.13): _VA,
        (112.23, 113.89): _VB,
        (114.09, 116.35): _VB,
        (116.67, 121.35): _VB,
        (201.52, 203.9): _VA,
        (213.57, 214.37): _VA,
    }
    for start, end, _label in turns:
        if (start, end) not in characters:
            characters[(start, end)] = _VC

    _scripted(turns, [], characters)
    _install_fake_pyannote_core(monkeypatch)

    result = diarize_detailed(_audio(tmp_path), settings=_settings(tmp_path))

    assert result.raw_speaker_ids == ("SPEAKER_00", "SPEAKER_01")
    by_start = {round(segment.start, 2): segment.speaker_id for segment in result.turns}
    # C's lines, all of them, one identity - including the one pyannote gave to B.
    assert {by_start[start] for start in (19.77, 105.75, 107.78, 109.89, 110.66, 214.59)} == {
        "SPEAKER_02"
    }
    # A and B keep their own identities and are not merged with each other or with C.
    assert {by_start[start] for start in (0.0, 20.0, 24.63, 201.52, 213.57)} == {"SPEAKER_00"}
    assert {by_start[start] for start in (3.0, 14.97, 112.23, 114.09, 116.67)} == {"SPEAKER_01"}
    assert len(result.reassignments) == 6
