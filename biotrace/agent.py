"""Minimal BioTrace evidence-acquisition controller for synthetic forward testing.

The production Patho-R1 observer, Qwen reasoner, RAG knowledge base, and full
prompting stack are deliberately excluded. This module keeps only the paper-level
state transitions so reviewers can inspect the forward data flow without exposing
benchmark artifacts or deployment-specific prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping

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
    def _demo_verify(state: EvidenceState) -> EvidenceStatus:
        """Deterministic stand-in used only to exercise the synthetic state machine."""
        if not state.observations:
            return EvidenceStatus.INSUFFICIENT
        if "40x" in state.inspected_magnifications and any(
            obs.role == "supportive_biology" for obs in state.observations
        ):
            return EvidenceStatus.SUFFICIENT
        return EvidenceStatus.PARTIAL

    def _add_phenotype_evidence(
        self, state: EvidenceState, cache: Dict[str, Tensor], scale: str
    ) -> int:
        scores = cache["phenotype_scores"]
        index = int(torch.argmax(scores[0]).item())
        state.add(EvidenceObservation(
            source="structured_phenotype",
            magnification=scale,
            concept_index=index,
            relevance=float(torch.softmax(scores, -1)[0, index].detach()),
            reliability_context=self._confidence(scores),
            role="direct_phenotype",
        ))
        return index

    def _add_visual_evidence(
        self,
        state: EvidenceState,
        cache: Dict[str, Tensor],
        scale: str,
        phenotype_index: int,
    ) -> None:
        response = cache["phenotype_spatial_response"][0, phenotype_index]
        top_patch = int(torch.argmax(response).item())
        state.add(EvidenceObservation(
            source=f"synthetic_visual_patch_{top_patch}",
            magnification=scale,
            concept_index=phenotype_index,
            relevance=float(response[top_patch].detach()),
            reliability_context=0.5,
            role="visible_morphology_placeholder",
        ))

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
                relevance=float(relation[pathway_index].abs().detach()),
                reliability_context=0.5,
                role="supportive_biology",
            ))
            for gene_index in ranked_genes:
                state.add(EvidenceObservation(
                    source="structured_gene",
                    magnification="40x",
                    concept_index=int(gene_index),
                    relevance=float(gp[gene_index, pathway_index].detach()),
                    reliability_context=0.5,
                    role="supportive_biology",
                ))
        return {"pathways": [int(x) for x in pathways], "genes": [int(x) for x in genes]}

    def forward(self, multiscale_features: Mapping[str, Tensor]) -> Dict[str, object]:
        cache = self.evidence_space(multiscale_features)
        state = EvidenceState()

        # Round 0: compact 10x structured phenotype evidence.
        phenotype_index = self._add_phenotype_evidence(state, cache["10x"], "10x")
        state.status = self._demo_verify(state)

        # Coarse-to-fine refinement; there is no fixed multiscale fusion.
        support = {"pathways": [], "genes": []}
        for scale in ("20x", "40x"):
            if state.status == EvidenceStatus.SUFFICIENT:
                break
            self._add_visual_evidence(state, cache[scale], scale, phenotype_index)
            if scale == "40x":
                support = self._add_supportive_biology(state, cache[scale], phenotype_index)
            state.status = self._demo_verify(state)

        return {
            "status": state.status.value,
            "selected_phenotype": phenotype_index,
            "support": support,
            "evidence_state": state,
            "scale_cache": cache,
        }
