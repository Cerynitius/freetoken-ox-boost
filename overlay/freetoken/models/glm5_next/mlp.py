"""Clamped-SwiGLU MLP for GLM-5.3's dense layers and shared experts.

HF reference (Glm5NextTextMLP / shared experts): ``silu(clamp(gate, max=limit)) *
clamp(up, -limit, limit)``. Serves through the dsv4 fused kernel, which is bit-exact
to that formula (verified). Weight names match GlmDsaGatedMLP (gate/up/down_proj).
"""

from __future__ import annotations

import os

import torch
from freetoken.kernel.triton.dsv4.swiglu import fused_swiglu
from freetoken.models.glm_moe_dsa.mlp import GlmDsaGatedMLP
from freetoken.utils import nvtx_annotate


# Opt-in (default OFF), read once at import: FREETOKEN_GLM5_MLP_FUSED=1 runs a decode token's
# gate GEMV + up GEMV + both split-K reduces + clamped SwiGLU as ONE launch (fp8 W8A16
# projections, M == 1); bit-identical -- see kernel/triton/fp8_gemv_fused.py.
_MLP_FUSED = os.environ.get("FREETOKEN_GLM5_MLP_FUSED", "0") == "1"
if _MLP_FUSED:
    from freetoken.kernel.triton.fp8_gemv_fused import gate_up_swiglu_fused


class Glm5ClampedMLP(GlmDsaGatedMLP):
    def __init__(self, hidden_size: int, intermediate_size: int, quant: str = "none",
                 limit: float = 10.0):
        super().__init__(hidden_size, intermediate_size, quant=quant)
        self.__dict__["_limit"] = float(limit)  # plain attr, excluded from state_dict walk

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _MLP_FUSED:
            h = gate_up_swiglu_fused(x, self.gate_proj, self.up_proj, self._limit)
            if h is not None:
                return self.down_proj.forward(h)
        gate = self.gate_proj.forward(x)
        up = self.up_proj.forward(x)
        return self.down_proj.forward(fused_swiglu(gate, up, self._limit, x.dtype))


__all__ = ["Glm5ClampedMLP"]
