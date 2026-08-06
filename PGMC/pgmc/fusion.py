"""The selected PGMC logit fusion."""

from typing import Dict

import torch


def fuse_pgmc_logits(
    zero_shot: torch.Tensor,
    components: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> torch.Tensor:
    """Direct weighted sum selected from the three variants in the original logs."""
    fused = float(weights.get("zero_shot", 1.0)) * zero_shot
    for name, logits in components.items():
        fused = fused + float(weights.get(name, 0.0)) * logits
    return fused

