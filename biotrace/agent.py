"""Minimal BioTrace evidence-acquisition controller for synthetic forward testing.

The production Patho-R1 observer, Qwen reasoner, RAG knowledge base, and full
prompting stack are deliberately excluded. This module keeps only the paper-level
state transitions so reviewers can inspect the forward data flow without exposing
benchmark artifacts or deployment-specific prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional

import torch
from torch import Tensor, nn

from .evidence_space import BioTraceEvidenceSpace


class EvidenceStatus(str, Enum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    CONFLICTING = "conflicting"
    INSUFFICIENT = "insufficient"
    UNAVAILABLE = "unavailable"


@dataclass
class EvidenceObservation:
    source: str
    magnification: str
    concept_index: int
    patch_index: Optional[int]
    relevance: float
    reliability_context: float
    role: str


@dataclass
class EvidenceState:
    status: EvidenceStatus = EvidenceStatus.INSUFFICIENT
    observations: List[EvidenceObservation] = field(default_factory=list)
    inspected_magnifications: List[str] = field(default_factory=list)

    def add(self, observation: EvidenceObservation) -> None:
        self.observations.append(observation)
        if observation.magnification not in self.inspected_magnifications:
            self.inspected_magnifications.append(observation.magnification)


class BioTraceAgent(nn.Module):
    """Paper-aligned forward skeleton for evidence acquisition and verification."""

    def __init__(
        self,
        evidence_space: BioTraceEvidenceSpace,
        top_pathways: int = 3,
        genes_per_pathway: int = 2,
    ) -> None:
        super().__init__()
        self.evidence_space = evidence_space
        self.top_pathways = int(top_pathways)
        self.genes_per_pathway = int(genes_per_pathway)

    @staticmethod
    def _confidence(scores: Tensor) -> float:
        probs = torch.softmax(scores, dim=-1)
        top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
        if top2.shape[-1] == 1:
            return float(top2[0, 0].detach())
        return float((top2[0, 0] - top2[0, 1]).clamp(0, 1).detach())

    @staticmethod
    def _demo_evidence_score(state: EvidenceState) -> float:
        """Synthetic-only evidence score used to exercise adaptive control flow.

        This is not the production verifier. It intentionally depends on the
        currently accumulated evidence so the example can stop early when a
        synthetic sample is already well supported, or request additional
        evidence when it is not.
        """
        if not state.observations:
            return 0.0

        direct = [
            obs for obs in state.observations if obs.role == "direct_phenotype"
        ]
        morphology = [
            obs for obs in state.observations
            if obs.role == "visible_morphology_placeholder"
        ]
        supportive = [
            obs for obs in state.observations if obs.role == "supportive_biology"
        ]

        direct_score = max(
            (
                obs.relevance
                * (0.5 + 0.5 * max(0.0, min(1.0, obs.reliability_context)))
                for obs in direct
            ),
            default=0.0,
        )
        morphology_score = 0.25 * max(
            (obs.relevance for obs in morphology), default=0.0
        )
        observed_scales = {
            obs.magnification for obs in morphology
            if obs.magnification in {"10x", "20x", "40x"}
        }
        coverage_score = 0.08 * len(observed_scales)
        supportive_score = 0.12 if supportive else 0.0

        return float(direct_score + morphology_score + coverage_score + supportive_score)

    @classmethod
    def _demo_verify(cls, state: EvidenceState) -> EvidenceStatus:
        """Demo-only adaptive verifier; production verification is not released here."""
        if not state.observations:
            return EvidenceStatus.INSUFFICIENT

        score = cls._demo_evidence_score(state)
        if score >= 0.50:
            return EvidenceStatus.SUFFICIENT
        if score >= 0.28:
            return EvidenceStatus.PARTIAL
        return EvidenceStatus.INSUFFICIENT

    def _add_phenotype_evidence(
        self, state: EvidenceState, cache: Dict[str, Tensor], scale: str
    ) -> int:
        scores = cache["phenotype_scores"]
        index = int(torch.argmax(scores[0]).item())
        state.add(EvidenceObservation(
            source="structured_phenotype",
            magnification=scale,
            concept_index=index,
            patch_index=None,
            relevance=float(torch.softmax(scores, -1)[0, index].detach()),
            reliability_context=self._confidence(scores),
            role="direct_phenotype",
        ))
        return index

    @staticmethod
    def _linked_candidates(parent_index: int, parent_count: int, child_count: int) -> Tensor:
        """Synthetic parent/child linkage for the random forward example only."""
        start = int(parent_index * child_count / parent_count)
        end = int((parent_index + 1) * child_count / parent_count)
        end = max(start + 1, min(end, child_count))
        return torch.arange(start, end)

    def _add_visual_evidence(
        self,
        state: EvidenceState,
        cache: Dict[str, Tensor],
        scale: str,
        phenotype_index: int,
        parent_patch: Optional[int] = None,
        parent_count: Optional[int] = None,
    ) -> int:
        response = cache["phenotype_spatial_response"][0, phenotype_index]
        if parent_patch is None or parent_count is None:
            patch_index = int(torch.argmax(response).item())
        else:
            candidates = self._linked_candidates(
                parent_patch, parent_count, response.numel()
            ).to(response.device)
            local = int(torch.argmax(response[candidates]).item())
            patch_index = int(candidates[local].item())

        state.add(EvidenceObservation(
            source=f"synthetic_visual_patch_{patch_index}",
            magnification=scale,
            concept_index=phenotype_index,
            patch_index=patch_index,
            relevance=float(response[patch_index].detach()),
            reliability_context=0.5,
            role="visible_morphology_placeholder",
        ))
        return patch_index

    def _add_supportive_biology(
        self,
        state: EvidenceState,
        cache: Dict[str, Tensor],
        phenotype_index: int,
    ) -> Dict[str, List[int]]:
        relation = cache["pathway_phenotype_relations"][:, phenotype_index]
        pathways = torch.topk(
            relation.abs(), k=min(self.top_pathways, relation.numel())
        ).indices.tolist()
        gp = cache["gene_pathway_relations"]
        genes: List[int] = []
        for pathway_index in pathways:
            ranked_genes = torch.topk(
                gp[:, pathway_index],
                k=min(self.genes_per_pathway, gp.shape[0]),
            ).indices.tolist()
            genes.extend(ranked_genes)
            state.add(EvidenceObservation(
                source="structured_pathway",
                magnification="40x",
                concept_index=int(pathway_index),
                patch_index=None,
                relevance=float(relation[pathway_index].abs().detach()),
                reliability_context=0.5,
                role="supportive_biology",
            ))
            for gene_index in ranked_genes:
                state.add(EvidenceObservation(
                    source="structured_gene",
                    magnification="40x",
                    concept_index=int(gene_index),
                    patch_index=None,
                    relevance=float(gp[gene_index, pathway_index].detach()),
                    reliability_context=0.5,
                    role="supportive_biology",
                ))
        return {"pathways": [int(x) for x in pathways], "genes": [int(x) for x in genes]}

    def forward(self, multiscale_features: Mapping[str, Tensor]) -> Dict[str, object]:
        # The scale-specific structured models run once and their outputs are cached.
        cache = self.evidence_space(multiscale_features)
        state = EvidenceState()

        # Round 0: compact 10x structured phenotype evidence.
        phenotype_index = self._add_phenotype_evidence(state, cache["10x"], "10x")
        state.status = self._demo_verify(state)

        # If unresolved, inspect morphology from coarse to fine. The verifier is
        # re-evaluated after each acquisition, so the path may stop at any scale.
        parent_patch: Optional[int] = None
        parent_count: Optional[int] = None
        for scale in ("10x", "20x", "40x"):
            if state.status == EvidenceStatus.SUFFICIENT:
                break
            patch_index = self._add_visual_evidence(
                state,
                cache[scale],
                scale,
                phenotype_index,
                parent_patch=parent_patch,
                parent_count=parent_count,
            )
            parent_patch = patch_index
            parent_count = cache[scale]["phenotype_spatial_response"].shape[-1]
            state.status = self._demo_verify(state)

        # Pathway/Gene evidence is an optional support stage at 40x and is queried
        # only if the accumulated phenotype/morphology evidence remains unresolved.
        support = {"pathways": [], "genes": []}
        if "40x" in state.inspected_magnifications and state.status != EvidenceStatus.SUFFICIENT:
            support = self._add_supportive_biology(state, cache["40x"], phenotype_index)
            state.status = self._demo_verify(state)

        return {
            "status": state.status.value,
            "evidence_score": self._demo_evidence_score(state),
            "selected_phenotype": phenotype_index,
            "support": support,
            "evidence_state": state,
            "scale_cache": cache,
        }
