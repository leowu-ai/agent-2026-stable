from __future__ import annotations

from typing import Dict, Mapping

from torch import Tensor, nn

from models.g2p_toolbank import G2PHypergraphToolBank


MAGNIFICATIONS = ("10x", "20x", "40x")


class MultiScaleG2PAgent(nn.Module):
    """Three independent scale-specific structured evidence models."""

    def __init__(self, models: Mapping[str, G2PHypergraphToolBank]) -> None:
        super().__init__()
        missing = [scale for scale in MAGNIFICATIONS if scale not in models]
        if missing:
            raise ValueError(f"missing scale-specific model(s): {missing}")
        self.models = nn.ModuleDict({scale: models[scale] for scale in MAGNIFICATIONS})

    def infer_case(
        self, features_by_scale: Mapping[str, Tensor]
    ) -> Dict[str, Dict[str, object]]:
        missing = [scale for scale in MAGNIFICATIONS if scale not in features_by_scale]
        if missing:
            raise KeyError(f"missing multiscale features: {missing}")
        return {
            scale: self.models[scale](features_by_scale[scale])
            for scale in MAGNIFICATIONS
        }
