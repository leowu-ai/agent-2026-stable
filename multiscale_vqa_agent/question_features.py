from __future__ import annotations

import torch
from torch import Tensor


class QuestionFeatureAdapter:
    """Normalizes a precomputed text embedding for question-guided retrieval.

    In the full system this feature can be produced by the pathology text encoder.
    The public forward example supplies a random tensor so no text preprocessing
    or external model download is required.
    """

    @staticmethod
    def normalize(feature: Tensor) -> Tensor:
        value = feature.float().reshape(-1)
        return value / value.norm().clamp_min(1e-6)
