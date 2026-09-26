from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from .schemas import EvidenceGroup, PatchCandidate


class MultiScaleRetrievalAgent:
    """Question/prototype-guided patch retrieval with coarse-to-fine linkage."""

    def __init__(
        self,
        top_patches_per_source: int = 4,
        max_evidence_groups: int = 4,
        global_bypass_per_magnification: int = 2,
        same_scale_iou_threshold: float = 0.50,
        feature_cosine_threshold: float = 0.95,
        question_weight: float = 0.25,
    ) -> None:
        self.top_patches_per_source = int(top_patches_per_source)
        self.max_evidence_groups = int(max_evidence_groups)
        self.global_bypass_per_magnification = int(global_bypass_per_magnification)
        self.same_scale_iou_threshold = float(same_scale_iou_threshold)
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

    @staticmethod
    def _box_iou(box_a: Tensor, box_b: Tensor) -> Tensor:
        left_top = torch.maximum(box_a[:2], box_b[:2])
        right_bottom = torch.minimum(box_a[2:], box_b[2:])
        wh = (right_bottom - left_top).clamp_min(0)
        intersection = wh[0] * wh[1]
        area_a = (box_a[2] - box_a[0]).clamp_min(0) * (box_a[3] - box_a[1]).clamp_min(0)
        area_b = (box_b[2] - box_b[0]).clamp_min(0) * (box_b[3] - box_b[1]).clamp_min(0)
        return intersection / (area_a + area_b - intersection).clamp_min(1e-6)

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

    def _is_duplicate(
        self,
        index: int,
        used: List[int],
        patch_features: Optional[Tensor],
        patch_boxes: Optional[Tensor],
    ) -> bool:
        if not used:
            return False
        if patch_boxes is not None:
            candidate_box = patch_boxes[index].float()
            for previous_index in used:
                if float(self._box_iou(candidate_box, patch_boxes[previous_index].float())) >= self.same_scale_iou_threshold:
                    return True
        if patch_features is not None:
            candidate = F.normalize(patch_features[index].float(), dim=-1)
            previous = F.normalize(patch_features[used].float(), dim=-1)
            similarity = previous @ candidate
            if bool((similarity >= self.feature_cosine_threshold).any()):
                return True
        return False

    @staticmethod
    def _box_value(patch_boxes: Optional[Tensor], index: int):
        if patch_boxes is None:
            return None
        return [float(value) for value in patch_boxes[index].detach().cpu().tolist()]

    def retrieve(
        self,
        scale_results: Dict[str, Dict[str, object]],
        phenotype_index: int,
        magnification: str,
        parent_groups: Optional[List[EvidenceGroup]] = None,
        patch_features: Optional[Tensor] = None,
        patch_boxes: Optional[Tensor] = None,
        question_feature: Optional[Tensor] = None,
    ) -> List[EvidenceGroup]:
        attention = scale_results[magnification]["phenotype_attention"][
            0, phenotype_index
        ].detach()
        if patch_features is not None and patch_features.ndim == 3:
            patch_features = patch_features[0].detach()
        if patch_boxes is not None:
            patch_boxes = patch_boxes.detach()
            if patch_boxes.ndim != 2 or patch_boxes.shape != (attention.numel(), 4):
                raise ValueError("patch_boxes must have shape [num_patches, 4]")
        combined, source = self._score(attention, patch_features, question_feature)

        if not parent_groups:
            ranked = torch.argsort(combined, descending=True).tolist()
            chosen: List[int] = []
            for index in ranked:
                if self._is_duplicate(index, chosen, patch_features, patch_boxes):
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
                            box=self._box_value(patch_boxes, int(index)),
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
                if self._is_duplicate(int(index), used, patch_features, patch_boxes):
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
                box=self._box_value(patch_boxes, chosen),
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

        if len(groups) < self.max_evidence_groups:
            ranked_global = torch.argsort(combined, descending=True).tolist()
            bypass = 0
            for index in ranked_global:
                if bypass >= self.global_bypass_per_magnification:
                    break
                if int(index) in used:
                    continue
                if self._is_duplicate(int(index), used, patch_features, patch_boxes):
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
                                box=self._box_value(patch_boxes, int(index)),
                                source=f"{source}_global_bypass",
                            )
                        },
                    )
                )
                bypass += 1
                if len(groups) >= self.max_evidence_groups:
                    break
        return groups
