"""Shared model loading, caching, and device management.

This module exists so a heavyweight checkpoint is loaded **once per process**, no
matter how many stages need it. Two stages need OmniVoice: the voice-profile stage
encodes a reference into a speaker prompt, and the synthesis stage generates speech
with it. Loading two copies of the same multi-gigabyte model would not just be slow -
it would put two copies in VRAM on the same card.

Design rules
------------
* Weights are downloaded at runtime into ``MODEL_CACHE_DIR`` (see
  :mod:`app.config`); they are never committed to the repository.
* Models are loaded lazily and cached per process, keyed by everything that shapes
  one (checkpoint, device, precision), so two configurations do not share weights
  that were loaded differently.
* GPU/CPU selection is centralized here.

Deliberate non-goals
--------------------
* **No import at module load.** ``torch`` and the model packages are imported inside
  the loader, so this module stays importable on a machine with no GPU stack - which
  is what the test suite relies on.
* **No eviction policy.** A run holds one checkpoint per stage; unloading it early
  would only make the next stage pay for the load again.
"""

from __future__ import annotations

from typing import Any

#: Loaded OmniVoice networks, keyed by ``(checkpoint, device, dtype)``.
_OMNIVOICE_NETWORKS: dict[tuple[str, str, str], Any] = {}


def load_omnivoice_network(
    model: str,
    *,
    device: str = "cuda",
    dtype: str = "float16",
    model_class: Any | None = None,
    torch_module: Any | None = None,
) -> Any:
    """Return the process-wide OmniVoice network for these settings.

    Parameters
    ----------
    model:
        Hugging Face checkpoint id, e.g. ``african-low-resource/omnivoice-amharic``.
    device:
        ``device_map`` passed to ``from_pretrained`` (``cuda``, ``cpu``, ...).
    dtype:
        Torch dtype *name*, resolved on the imported ``torch`` module.
    model_class, torch_module:
        Supplied by a caller that has already resolved them (which is also how the
        test suite injects a fake without importing the real runtime).

    Raises
    ------
    ImportError
        The ``omnivoice`` or ``torch`` package is not installed.
    ValueError
        ``dtype`` is not a torch dtype, or ``model`` is empty.
    """

    if not isinstance(model, str) or not model.strip():
        raise ValueError("the OmniVoice checkpoint must be a non-empty id")

    key = (model.strip(), device, dtype)
    cached = _OMNIVOICE_NETWORKS.get(key)
    if cached is not None:
        return cached

    if torch_module is None:
        import torch as torch_module_imported

        torch_module = torch_module_imported
    if model_class is None:
        from omnivoice import OmniVoice

        model_class = OmniVoice

    torch_dtype = getattr(torch_module, dtype, None)
    if torch_dtype is None:
        raise ValueError(f"{dtype!r} is not a torch dtype")

    network = model_class.from_pretrained(
        key[0], device_map=device, dtype=torch_dtype
    )
    _OMNIVOICE_NETWORKS[key] = network
    return network


def cached_models() -> tuple[tuple[str, str, str], ...]:
    """Return the keys of the models currently cached, for diagnostics."""

    return tuple(sorted(_OMNIVOICE_NETWORKS))


def reset_model_cache() -> None:
    """Drop every cached network. Used by tests and by a device/model switch."""

    _OMNIVOICE_NETWORKS.clear()


def release_gpu_memory() -> float:
    """Return unreferenced GPU memory to the driver, and report how much, in MiB.

    Every stage here loads a model into VRAM and mostly does not need it afterwards: the
    separator's weights are useless once the stems exist, pyannote's once the turns do,
    whisper's once the transcript does. Their *objects* are dropped when the stage
    returns, but PyTorch's caching allocator keeps the segments mapped so the next
    allocation is cheap - which is exactly wrong on a card that also hosts the vLLM
    server. `empty_cache` is what turns those segments back into memory the driver can
    hand to another allocator (CTranslate2, for instance, does not use PyTorch's).

    Cached models in this module are deliberately *kept*: OmniVoice is loaded by the
    voice-profile stage and needed again by synthesis, and reloading it would cost far
    more than the memory it frees.

    Returns
    -------
    float
        MiB actually released, or ``0.0`` when there is no CUDA runtime or nothing was
        cached. Never raises: a failed cache flush must not end a run.
    """

    import gc

    gc.collect()

    try:
        import torch
    except ImportError:  # pragma: no cover - no torch installed
        return 0.0
    if not torch.cuda.is_available():  # pragma: no cover - CPU-only machine
        return 0.0

    try:
        before = float(torch.cuda.memory_reserved())
        torch.cuda.empty_cache()
        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
        if ipc_collect is not None:
            ipc_collect()
        after = float(torch.cuda.memory_reserved())
    except Exception:  # noqa: BLE001 - a diagnostics call must never break a run
        return 0.0
    return max(0.0, (before - after) / (1024.0 * 1024.0))


def gpu_free_gib() -> tuple[float, float] | None:
    """Return ``(free, total)`` GPU memory in GiB, or ``None`` without CUDA.

    This is the number that decides whether the next model can be loaded at all, so a
    failure is reported with it rather than leaving someone to guess.
    """

    try:
        import torch
    except ImportError:  # pragma: no cover - no torch installed
        return None
    if not torch.cuda.is_available():  # pragma: no cover - CPU-only machine
        return None
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001 - a diagnostics call must never break a run
        return None
    gib = 1024.0 * 1024.0 * 1024.0
    return free / gib, total / gib


__all__ = [
    "cached_models",
    "gpu_free_gib",
    "load_omnivoice_network",
    "release_gpu_memory",
    "reset_model_cache",
]
