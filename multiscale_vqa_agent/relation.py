from __future__ import annotations

from typing import Any, Dict, List

import torch

from .registry import ToolBankRegistry


class RelationReasoningAgent:
    """Rank patient-aware Phenotype <- Pathway <- Gene support."""

    def __init__(
        self,
        registry: ToolBankRegistry,
        top_pathways: int = 3,
        genes_per_pathway: int = 2,
    ) -> None:
        self.registry = registry
        self.top_pathways = int(top_pathways)
        self.genes_per_pathway = int(genes_per_pathway)

    def reason(
        self,
        phenotype_name: str,
        scale_results: Dict[str, Dict[str, object]],
    ) -> Dict[str, Any]:
        phenotype_index = self.registry.phenotype_to_index[phenotype_name]
        scales = [scale for scale in ("10x", "20x", "40x") if scale in scale_results]

        relations = torch.stack(
            [scale_results[s]["R"][:, phenotype_index].detach() for s in scales],
            dim=0,
        )
        pathway_pred = torch.stack(
            [scale_results[s]["pathway_pred"][0].detach() for s in scales],
            dim=0,
        )
        mean_relation = relations.mean(dim=0)
        agreement = 1.0 / (1.0 + relations.std(dim=0, unbiased=False))
        activity = pathway_pred.abs().mean(dim=0)
        pathway_score = mean_relation.abs() * agreement * (1.0 + activity)

        k = min(self.top_pathways, pathway_score.numel())
        pathway_indices = torch.topk(pathway_score, k=k).indices.tolist()

        h_values = torch.stack(
            [scale_results[s]["H"].detach() for s in scales], dim=0
        ).mean(dim=0)
        gene_pred = torch.stack(
            [scale_results[s]["gene_pred"][0].detach() for s in scales],
            dim=0,
        ).abs().mean(dim=0)

        pathways: List[Dict[str, Any]] = []
        genes: Dict[int, float] = {}
        for pathway_index in pathway_indices:
            gene_score = h_values[:, pathway_index].abs() * (1.0 + gene_pred)
            gk = min(self.genes_per_pathway, gene_score.numel())
            gene_indices = torch.topk(gene_score, k=gk).indices.tolist()
            for gene_index in gene_indices:
                genes[gene_index] = max(
                    genes.get(gene_index, 0.0),
                    float(gene_score[gene_index]),
                )
            pathways.append(
                {
                    "index": int(pathway_index),
                    "name": self.registry.pathways[pathway_index],
                    "score": float(pathway_score[pathway_index]),
                    "relation": float(mean_relation[pathway_index]),
                    "genes": [
                        {
                            "index": int(g),
                            "name": self.registry.genes[g],
                            "score": float(gene_score[g]),
                        }
                        for g in gene_indices
                    ],
                }
            )

        ranked_genes = sorted(genes.items(), key=lambda item: item[1], reverse=True)
        return {
            "phenotype": phenotype_name,
            "pathways": pathways,
            "genes": [
                {
                    "index": int(index),
                    "name": self.registry.genes[index],
                    "score": float(score),
                }
                for index, score in ranked_genes
            ],
        }
