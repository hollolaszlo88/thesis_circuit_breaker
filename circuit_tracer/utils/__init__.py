import os

import torch
from circuit_tracer.utils.create_graph_files import create_graph_files as create_graph_files


def get_default_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU.

    MPS requires a PyTorch build with Metal support; many Anaconda ``pytorch`` installs
    are CPU-only on macOS, in which case this returns CPU until you reinstall from
    https://pytorch.org/get-started/locally/
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_mps_memory_if_needed(device: torch.device) -> None:
    """Set ``PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`` when unset so large models + transcoders can load.

    PyTorch caps MPS allocations by default; Gemma-scale weights plus transcoder stacks can
    exceed that cap by a small margin. ``0.0`` removes the cap (may increase memory pressure);
    set the variable yourself before import if you want a different policy. See
    https://pytorch.org/docs/stable/notes/mps.html
    """
    if device.type != "mps":
        return
    if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" in os.environ:
        return
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def safetensors_mmap_device(device: torch.device) -> str:
    """Device string for :func:`safetensors.safe_open` (MPS is not a valid mmap load target).

    Load on CPU for MPS targets, then move tensors with ``.to(device=...)``.
    """
    if device.type == "mps":
        return "cpu"
    return str(device)


__all__ = [
    "configure_mps_memory_if_needed",
    "create_graph_files",
    "get_default_device",
    "safetensors_mmap_device",
]
