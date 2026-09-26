from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from .schemas import EvidenceGroup, PatchCandidate


class MultiScaleRetrievalAgent:
    """Question/prototype-guided patch retrieval with coarse-to-fine linkage.

    The production agent can use spatial coordinates for exact parent/child
    matching. This data-free forward path preserves the same acquisition logic
    using index linkage, feature-level deduplication, and a small global bypass.
    """

    def __init__(
        self,
        top_patches_per_source: int = 4,
        max_evidence_groups: int = 4,
        global_bypass_per_magnification: int = 2,
        feature_cosine_threshold: float = 0.95,
        question_weight: float = 0.25,
    ) -> None:
        self.top_patches_per_source = int(top_patches_per_source)
        self.max_evidence_groups = int(max_evidence_groups)
        self.global_bypass_per_magnification = int(global_bypass_per_magnification)
        self.feature_cosine_threshold = float(feature_cosine_threshold)
        self.question_weight = float(question_weight)

    @staticmethod
    def _linked_range(
        parent_index: int,
        parent_count: int,
        child_count: int,
    ) -> Tensor:
        start = int(parent_index * child_count / parent_count)
        end = int((parent_index + 1) * child_count / parent_count)
        end = max(start + 1, min(end, child_count))
        return torch.arange(start, end)

    @staticmethod
    def _minmax(values: Tensor) -> Tensor:
        lo = values.min()
        hi = values.max()
        if float((hi - lo).abs()) < 1e-8:
            return torch.zeros_like(values)
        return (values - lo) / (hi - lo)

    def _score(
        self,
        attention: Tensor,
        patch_features: Optional[Tensor],
        question_feature: Optional[Tensor],
    ) -> tuple[Tensor, str]:
        base = self._minmax(attention)
        if patch_features is None or question_feature is None:
            return base, "phenotype_attention"
        patches = F.normalize(patch_features.float(), dim=-1)
        query = F.normalize(question_feature.float().reshape(1, -1), dim=-1)
        if patches.shape[-1] != query.shape[-1]:
            return base, "phenotype_attention"
        similarity = self._minmax((patches @ query.t()).squeeze(-1))
        weight = min(max(self.question_weight, 0.0), 1.0)
        return (1.0 - weight) * base + weight * similarity, "hybrid_question_prototype"

    def _is_feature_duplicate(
        self,
        index: int,
        used: List[int],
        patch_features: Optional[Tensor],
    ) -> bool:
        if patch_features is None or not used:
            return False
        candidate = F.normalize(patch_features[index].float(), dim=-1)
        previous = F.normalize(patch_features[used].float(), dim=-1)
        similarity = previous @ candidate
        return bool((similarity >= self.feature_cosine_threshold).any())

    def retrieve(
        self,
        scale_results: Dict[str, Dict[str, object]],
        phenotype_index: int,
        magnification: str,
        parent_groups: Optional[List[EvidenceGroup]] = None,
        patch_features: Optional[Tensor] = None,
        question_feature: Optional[Tensor] = None,
    ) -> List[EvidenceGroup]:
        attention = scale_results[magnification]["phenotype_attention"][
            0, phenotype_index
        ].detach()
        if patch_features is not None and patch_features.ndim == 3:
            patch_features = patch_features[0].detach()
        combined, source = self._score(attention, patch_features, question_feature)

        if not parent_groups:
            ranked = torch.argsort(combined, descending=True).tolist()
            chosen: List[int] = []
            for index in ranked:
                if self._is_feature_duplicate(index, chosen, patch_features):
                    continue
                chosen.append(int(index))
                if len(chosen) >= min(
                    self.top_patches_per_source, self.max_evidence_groups
                ):
                    break
            return [
                EvidenceGroup(
                    group_id=i,
                    score=float(combined[index]),
                    evidence_source=source,
                    patches={
                        magnification: PatchCandidate(
                            magnification=magnification,
                            patch_index=int(index),
                            score=float(combined[index]),
                            source=source,
                        )
                    },
                )
                for i, index in enumerate(chosen)
            ]

        parent_scale = next(reversed(parent_groups[0].patches))
        parent_count = scale_results[parent_scale]["phenotype_attention"].shape[-1]
        child_count = combined.numel()

        groups: List[EvidenceGroup] = []
        used: List[int] = []
        for parent in parent_groups[: self.max_evidence_groups]:
            parent_patch = parent.patches[parent_scale]
            candidates = self._linked_range(
                parent_patch.patch_index, parent_count, child_count
            ).to(combined.device)
            ranked = candidates[torch.argsort(combined[candidates], descending=True)]
            chosen = None
            for index in ranked.tolist():
                if int(index) in used:
                    continue
                if self._is_feature_duplicate(int(index), used, patch_features):
                    continue
                chosen = int(index)
                break
            if chosen is None:
                continue
            used.append(chosen)
            patches = dict(parent.patches)
            patches[magnification] = PatchCandidate(
                magnification=magnification,
                patch_index=chosen,
                score=float(combined[chosen]),
                source=source,
                parent_patch_index=parent_patch.patch_index,
            )
            groups.append(
                EvidenceGroup(
                    group_id=len(groups),
                    score=float(combined[chosen]),
                    evidence_source=source,
                    parent_group_id=parent.group_id,
                    patches=patches,
                )
            )

        # Preserve a small global bypass so question-relevant evidence is not
        # forced to remain inside a weak parent trajectory.
        if len(groups) < self.max_evidence_groups:
            ranked_global = torch.argsort(combined, descending=True).tolist()
            bypass = 0
            for index in ranked_global:
                if bypass >= self.global_bypass_per_magnification:
                    break
                if int(index) in used:
                    continue
                if self._is_feature_duplicate(int(index), used, patch_features):
                    continue
                used.append(int(index))
                groups.append(
                    EvidenceGroup(
                        group_id=len(groups),
                        score=float(combined[index]),
                        evidence_source=f"{source}_global_bypass",
                        patches={
                            magnification: PatchCandidate(
                                magnification=magnification,
                                patch_index=int(index),
                                score=float(combined[index]),
                                source=f"{source}_global_bypass",
                            )
                        },
                    )
                )
                bypass += 1
                if len(groups) >= self.max_evidence_groups:
                    break
        return groups
