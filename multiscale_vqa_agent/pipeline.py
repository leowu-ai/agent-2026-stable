from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import torch
from torch import Tensor

from .agent_memory import EvidenceObservation, WorkingMemory
from .fusion import FusionAgent
from .g2p_runtime import MultiScaleG2PAgent
from .knowledge_rag import KnowledgeRAG
from .pathology import PathologyAgent
from .planner import EvidencePlanner
from .registry import ToolBankRegistry
from .relation import RelationReasoningAgent
from .retrieval import MultiScaleRetrievalAgent
from .verifier import EvidenceVerifierAgent


class BioTracePipeline(torch.nn.Module):
    """Question-driven evidence acquisition and verification."""

    def __init__(
        self,
        g2p_agent: MultiScaleG2PAgent,
        registry: ToolBankRegistry,
        planner: EvidencePlanner,
        retrieval: MultiScaleRetrievalAgent,
        pathology: PathologyAgent,
        verifier: EvidenceVerifierAgent,
        fusion: FusionAgent,
        knowledge: Optional[KnowledgeRAG] = None,
        *,
        top_pathways: int = 3,
        genes_per_pathway: int = 2,
        max_rounds: int = 8,
    ) -> None:
        super().__init__()
        self.g2p_agent = g2p_agent
        self.registry = registry
        self.planner = planner
        self.retrieval = retrieval
        self.pathology = pathology
        self.verifier = verifier
        self.fusion = fusion
        self.knowledge = knowledge or KnowledgeRAG()
        self.relations = RelationReasoningAgent(
            registry,
            top_pathways=top_pathways,
            genes_per_pathway=genes_per_pathway,
        )
        self.max_rounds = int(max_rounds)

    def forward(
        self,
        case_id: str,
        question: str,
        choices: List[str],
        features_by_scale: Mapping[str, Tensor],
        question_feature: Optional[Tensor] = None,
        image_urls_by_scale: Optional[Mapping[str, List[str]]] = None,
        patch_boxes_by_scale: Optional[Mapping[str, Tensor]] = None,
    ) -> Dict[str, Any]:
        plan = self.planner.plan(case_id, question, choices)
        scale_results = self.g2p_agent.infer_case(features_by_scale)
        target = plan.target_phenotypes[0]
        phenotype_index = self.registry.phenotype_to_index[target]

        knowledge = self.knowledge.retrieve(
            question, choices, plan.target_phenotypes
        )
        memory = WorkingMemory(
            case_id=case_id,
            question=question,
            choices=list(choices),
            plan=plan.to_dict(),
            knowledge=knowledge,
        )

        self._initialize_structured_state(
            memory, scale_results, phenotype_index, target
        )
        relation_evidence = self.relations.reason(target, scale_results)

        groups_by_scale: Dict[str, Any] = {}
        for round_index in range(self.max_rounds):
            available = self._available_actions(memory, relation_evidence)
            decision = self.verifier.verify(
                memory,
                available,
                pathway_candidates=relation_evidence["pathways"],
                gene_candidates=relation_evidence["genes"],
            )
            memory.update_verifier(decision)
            action = decision["next_action"]
            memory.record_action(
                round_index,
                action,
                target=decision.get("target"),
                reason=decision.get("reason", ""),
            )

            if action in {"answer", "unavailable"}:
                break

            if action.startswith("inspect_") and action.endswith("x"):
                scale = action.replace("inspect_", "")
                parent = None
                if scale == "20x":
                    parent = groups_by_scale.get("10x")
                elif scale == "40x":
                    parent = groups_by_scale.get("20x") or groups_by_scale.get("10x")
                groups = self.retrieval.retrieve(
                    scale_results,
                    phenotype_index,
                    scale,
                    parent_groups=parent,
                    patch_features=features_by_scale[scale],
                    patch_boxes=(patch_boxes_by_scale or {}).get(scale),
                    question_feature=question_feature,
                )
                groups_by_scale[scale] = groups
                observation = self.pathology.describe(
                    question,
                    groups,
                    scale,
                    visual_guidance=knowledge[
                        "scale_specific_visual_guidance"
                    ].get(scale, []),
                    image_urls=(image_urls_by_scale or {}).get(scale, []),
                )
                memory.add_observation(
                    EvidenceObservation(
                        round_index=round_index,
                        action=action,
                        evidence_type="morphology",
                        evidence_role="direct",
                        magnification=scale,
                        target_type="phenotype",
                        target_name=target,
                        visual_description="; ".join(
                            observation.get("visible_findings", [])
                        ),
                        structured_support={
                            "image_quality": observation.get("image_quality"),
                        },
                        group_ids=[g.group_id for g in groups],
                    )
                )
                continue

            if action == "inspect_pathway":
                pathway = self._next_pathway(memory, relation_evidence)
                if pathway is None:
                    continue
                memory.add_observation(
                    EvidenceObservation(
                        round_index=round_index,
                        action=action,
                        evidence_type="pathway",
                        evidence_role="supportive",
                        magnification="40x",
                        target_type="pathway",
                        target_name=pathway["name"],
                        structured_support=pathway,
                    )
                )
                continue

            if action == "inspect_gene":
                gene = self._next_gene(memory, relation_evidence)
                if gene is None:
                    continue
                memory.add_observation(
                    EvidenceObservation(
                        round_index=round_index,
                        action=action,
                        evidence_type="gene",
                        evidence_role="supportive",
                        magnification="40x",
                        target_type="gene",
                        target_name=gene["name"],
                        structured_support=gene,
                    )
                )

        if not memory.final_verifier:
            memory.final_verifier = {
                "evidence_sufficient": False,
                "evidence_state": "unavailable",
                "missing_evidence_type": "unavailable",
                "conflict_detected": False,
                "next_action": "unavailable",
                "target": None,
                "reason": "Acquisition budget exhausted.",
                "decision_source": "budget_guard",
            }

        answer = self.fusion.answer(memory)
        return {
            "answer": answer,
            "plan": plan.to_dict(),
            "structured_evidence": memory.structured_evidence,
            "relation_evidence": relation_evidence,
            "working_memory": memory.to_dict(),
            "scale_results": scale_results,
        }

    def _initialize_structured_state(
        self,
        memory: WorkingMemory,
        scale_results: Dict[str, Dict[str, object]],
        phenotype_index: int,
        target: str,
    ) -> None:
        per_scale = {}
        scores = []
        for scale in ("10x", "20x", "40x"):
            logit = scale_results[scale]["phenotype_logits"][phenotype_index][0]
            score = float(torch.sigmoid(logit).detach())
            per_scale[scale] = score
            scores.append(score)

        initial_score = per_scale["10x"]
        agreement = 1.0 - min(
            max(torch.tensor(scores).std(unbiased=False).item(), 0.0), 1.0
        )
        confidence = 0.65 * initial_score + 0.35 * agreement
        reliability = agreement

        choice_index = next(
            (index for index, choice in enumerate(memory.choices)
             if choice.strip().casefold() == target.strip().casefold()),
            0,
        )

        memory.structured_candidate = {
            "phenotype": target,
            "choice_index": choice_index,
        }
        memory.structured_confidence = float(confidence)
        memory.structured_reliability = float(reliability)
        memory.structured_evidence = {
            "target_phenotype": target,
            "initial_magnification": "10x",
            "per_scale_prediction": per_scale,
            "cross_scale_agreement": float(agreement),
            "candidate_choice_index": choice_index,
            "semantics": "WSI-derived phenotype prediction; not clinical ground truth.",
        }
        memory.add_observation(
            EvidenceObservation(
                round_index=0,
                action="round0",
                evidence_type="phenotype",
                evidence_role="direct",
                magnification="10x",
                target_type="phenotype",
                target_name=target,
                structured_support=memory.structured_evidence,
            )
        )

    @staticmethod
    def _available_actions(
        memory: WorkingMemory,
        relation_evidence: Dict[str, Any],
    ) -> List[str]:
        actions = ["answer"]
        for scale in ("10x", "20x", "40x"):
            if scale not in memory.inspected_magnifications:
                actions.append(f"inspect_{scale}")
        if relation_evidence["pathways"] and not memory.inspected_pathways:
            actions.append("inspect_pathway")
        if relation_evidence["genes"] and not memory.inspected_genes:
            actions.append("inspect_gene")
        actions.append("unavailable")
        return actions

    @staticmethod
    def _next_pathway(
        memory: WorkingMemory,
        relation_evidence: Dict[str, Any],
    ):
        for row in relation_evidence["pathways"]:
            if row["name"] not in memory.inspected_pathways:
                return row
        return None

    @staticmethod
    def _next_gene(
        memory: WorkingMemory,
        relation_evidence: Dict[str, Any],
    ):
        for row in relation_evidence["genes"]:
            if row["name"] not in memory.inspected_genes:
                return row
        return None
