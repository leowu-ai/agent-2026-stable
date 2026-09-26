#!/usr/bin/env python3
"""Run one synthetic BioTrace end-to-end forward pass.

The example uses random patch embeddings and synthetic concept names. It does not
read a WSI, patient record, benchmark file, answer label, split, checkpoint, or
molecular measurement.
"""

from __future__ import annotations

import torch

from models import G2PHypergraphToolBank
from multiscale_vqa_agent import BioTracePipeline
from multiscale_vqa_agent.clients import OpenAICompatibleClient
from multiscale_vqa_agent.fusion import FusionAgent
from multiscale_vqa_agent.g2p_runtime import MultiScaleG2PAgent
from multiscale_vqa_agent.knowledge_rag import KnowledgeRAG
from multiscale_vqa_agent.pathology import PathologyAgent
from multiscale_vqa_agent.planner import EvidencePlanner
from multiscale_vqa_agent.registry import ToolBankRegistry
from multiscale_vqa_agent.retrieval import MultiScaleRetrievalAgent
from multiscale_vqa_agent.verifier import EvidenceVerifierAgent


def build_pipeline() -> BioTracePipeline:
    torch.manual_seed(7)

    genes = [f"Gene-{i}" for i in range(8)]
    pathways = [f"Pathway-{i}" for i in range(4)]
    phenotypes = ["Phenotype-A", "Phenotype-B", "Phenotype-C"]

    h_prior = torch.tensor(
        [
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=torch.float32,
    )
    r_prior = torch.tensor(
        [
            [0.6, 0.2, 0.1],
            [0.3, 0.5, 0.2],
            [0.1, 0.4, 0.6],
            [0.4, 0.1, 0.5],
        ],
        dtype=torch.float32,
    )

    models = {
        scale: G2PHypergraphToolBank(
            feature_dim=512,
            hidden_dim=512,
            phenotype_names=phenotypes,
            gene_names=genes,
            pathway_names=pathways,
            H_prior=h_prior,
            R_prior=r_prior,
        )
        for scale in ("10x", "20x", "40x")
    }
    g2p = MultiScaleG2PAgent(models)
    registry = ToolBankRegistry(phenotypes, pathways, genes)

    disabled = OpenAICompatibleClient(enabled=False)
    return BioTracePipeline(
        g2p_agent=g2p,
        registry=registry,
        planner=EvidencePlanner(registry, disabled),
        retrieval=MultiScaleRetrievalAgent(
            top_patches_per_source=4,
            max_evidence_groups=4,
        ),
        pathology=PathologyAgent(disabled),
        verifier=EvidenceVerifierAgent(disabled),
        fusion=FusionAgent(disabled),
        knowledge=KnowledgeRAG(),
        top_pathways=3,
        genes_per_pathway=2,
    )


def main() -> None:
    pipeline = build_pipeline()
    pipeline.eval()

    features = {
        "10x": torch.randn(1, 24, 512),
        "20x": torch.randn(1, 48, 512),
        "40x": torch.randn(1, 96, 512),
    }
    question = "Which synthetic phenotype is best supported by the slide evidence?"
    choices = ["Phenotype-A", "Phenotype-B", "Phenotype-C"]
    question_feature = torch.randn(512)

    with torch.no_grad():
        result = pipeline.forward(
            case_id="synthetic-case",
            question=question,
            choices=choices,
            features_by_scale=features,
            question_feature=question_feature,
        )

    memory = result["working_memory"]
    print("BioTrace synthetic forward: OK")
    print("final arbitration backend:", result["answer"]["backend"])
    print("verifier:", memory["final_verifier"]["evidence_state"])
    print(
        "acquisition:",
        " -> ".join(row["action"] for row in memory["action_history"]),
    )
    print("inspected magnifications:", memory["inspected_magnifications"])
    print("supportive pathways:", memory["inspected_pathways"])
    print("supportive genes:", memory["inspected_genes"])


if __name__ == "__main__":
    main()
