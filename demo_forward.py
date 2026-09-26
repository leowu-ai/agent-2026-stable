#!/usr/bin/env python3
"""Run one fully synthetic BioTrace forward pass.

No WSI, patient identifier, benchmark question, label, split, checkpoint, or RNA
value is loaded by this script.
"""

from __future__ import annotations

import torch

from biotrace import BioTraceAgent, BioTraceEvidenceSpace


def main() -> None:
    torch.manual_seed(7)
    feature_dim = 512
    hidden_dim = 128
    num_genes = 8
    num_pathways = 4
    num_phenotypes = 3

    # Synthetic prior support only; it does not encode a real pathway definition.
    gene_pathway_support = torch.tensor(
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

    model = BioTraceEvidenceSpace(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_genes=num_genes,
        num_pathways=num_pathways,
        num_phenotypes=num_phenotypes,
        gene_pathway_support=gene_pathway_support,
    )
    agent = BioTraceAgent(model, top_pathways=3, genes_per_pathway=2)
    agent.eval()

    sample = {
        "10x": torch.randn(1, 12, feature_dim),
        "20x": torch.randn(1, 24, feature_dim),
        "40x": torch.randn(1, 48, feature_dim),
    }

    with torch.no_grad():
        result = agent(sample)

    print("BioTrace synthetic forward: OK")
    print("status:", result["status"])
    print("selected phenotype index:", result["selected_phenotype"])
    print("supportive pathway indices:", result["support"]["pathways"])
    print("supportive gene indices:", result["support"]["genes"])
    for scale in ("10x", "20x", "40x"):
        cache = result["scale_cache"][scale]
        print(
            f"{scale}: gene={tuple(cache['gene_scores'].shape)} "
            f"pathway={tuple(cache['pathway_scores'].shape)} "
            f"phenotype={tuple(cache['phenotype_scores'].shape)}"
        )


if __name__ == "__main__":
    main()
