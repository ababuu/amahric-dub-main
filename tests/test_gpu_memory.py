"""Tests for the GPU-memory policy between stages.

The stages share one card with the vLLM server, so the question this module answers is
"can the next model even be loaded?" - and it has to answer without a GPU present.
"""

from __future__ import annotations

import pytest

import app.models as models
from app.pipeline import orchestrator


def test_release_is_a_no_op_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """A machine with no GPU must still be able to run the whole stage sequence."""

    monkeypatch.setattr(models, "release_gpu_memory", lambda: 0.0)

    assert models.release_gpu_memory() == 0.0


def test_gpu_free_gib_is_none_without_cuda() -> None:
    assert models.gpu_free_gib() is None


def test_release_never_raises_when_torch_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):  # noqa: ANN001, ANN202
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    assert models.release_gpu_memory() == 0.0


def test_a_small_release_is_not_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 40 MiB flush is noise; printing it would bury the stage lines."""

    monkeypatch.setattr(orchestrator, "release_gpu_memory", lambda: 40.0)
    messages: list[str] = []

    orchestrator._free_gpu_memory("separation", messages.append)

    assert messages == []


def test_a_real_release_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator, "release_gpu_memory", lambda: 3072.0)
    messages: list[str] = []

    orchestrator._free_gpu_memory("separation", messages.append)

    assert messages == ["  gpu: released 3.0 GiB after separation"]


def test_an_out_of_memory_failure_names_the_real_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bare CUDA error sends people to the wrong place; the hint must not."""

    monkeypatch.setattr(orchestrator, "gpu_free_gib", lambda: (1.7, 47.5))
    seconds: dict[str, float] = {}

    def boom() -> None:
        raise RuntimeError("CUDA failed with error out of memory")

    with pytest.raises(orchestrator.StageError) as info:
        orchestrator._stage("transcription", seconds, lambda _: None, boom)

    message = str(info.value)
    assert "1.7 GiB of 47.5 GiB" in message
    assert "TRANSLATION_GPU_MEMORY_UTILIZATION" in message
    assert "TRANSLATION_MAX_CONCURRENCY" in message


def test_an_ordinary_failure_gets_no_memory_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orchestrator, "gpu_free_gib", lambda: (40.0, 47.5))
    seconds: dict[str, float] = {}

    def boom() -> None:
        raise ValueError("something unrelated")

    with pytest.raises(orchestrator.StageError) as info:
        orchestrator._stage("separation", seconds, lambda _: None, boom)

    assert "TRANSLATION_GPU_MEMORY_UTILIZATION" not in str(info.value)
    assert isinstance(info.value.__cause__, ValueError)
