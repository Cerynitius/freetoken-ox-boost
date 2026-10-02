"""Named timing regions for kernel-level attribution (off by default).

``FREETOKEN_REGIONS=1`` wraps each region in a ``torch.profiler.record_function("R:<name>")``;
an eager decode profile (``--cuda-graph-max-bs 0`` + ``FREETOKEN_PROFILE_DECODE``) then lets
``naive-stack/regions_report.py`` attribute every GPU kernel to its innermost region via the
launch correlation ids. Zero cost when disabled; never active during CUDA-graph capture.
"""
import contextlib
import os

import torch

ENABLED = os.environ.get("FREETOKEN_REGIONS", "0") == "1"
_NULL = contextlib.nullcontext()


def region(name: str):
    if not ENABLED or torch.cuda.is_current_stream_capturing():
        return _NULL
    return torch.profiler.record_function("R:" + name)
