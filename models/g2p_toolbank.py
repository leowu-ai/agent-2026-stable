"""Inference-side Gene -> Pathway -> Phenotype evidence model.

This file keeps the core model path used by BioTrace while omitting dataset
construction, target generation, training loops, and patient-level supervision.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


class G2PHypergraphToolBank(nn.Module):
    """Scale-specific structured evidence model.

    Gene concepts are grounded directly in WSI patch features. A fixed
    Gene -> Pathway membership prior aggregates gene evidence into pathway
    context. Pathway -> Phenotype relation strengths are learned model
    parameters. All outputs at inference are derived from WSI features.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        phenotype_names: Iterable[str],
        gene_names: Iterable[str],
        pathway_names: Iterable[str],
        H_prior: Tensor,
        R_prior: Optional[Tensor] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.phenotype_names = list(phenotype_names)
        self.gene_names = list(gene_names)
        self.pathway_names = list(pathway_names)

        self.num_genes = len(self.gene_names)
        self.num_pathways = len(self.pathway_names)
        self.num_phenotypes = len(self.phenotype_names)

        H_prior = torch.as_tensor(H_prior, dtype=torch.float32)
        if H_prior.shape != (self.num_genes, self.num_pathways):
            raise ValueError("H_prior must have shape [num_genes, num_pathways]")
        self.register_buffer("H_prior", H_prior)

        if R_prior is None:
            R_prior = torch.zeros(self.num_pathways, self.num_phenotypes)
        R_prior = torch.as_tensor(R_prior, dtype=torch.float32)
        if R_prior.shape != (self.num_pathways, self.num_phenotypes):
            raise ValueError("R_prior must have shape [num_pathways, num_phenotypes]")
        eps = 1e-4
        init = R_prior.clamp(-1 + eps, 1 - eps)
        self.R_theta = nn.Parameter(torch.atanh(init))

        self.patch_projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.gene_identity_embeddings = nn.Parameter(
            torch.randn(self.num_genes, self.hidden_dim) * 0.02
        )
        self.gene_query_projection = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        nn.init.eye_(self.gene_query_projection.weight)
        self.gene_query_delta = nn.Parameter(
            torch.zeros(self.num_genes, self.hidden_dim)
        )
        self.gene_query_norm = nn.LayerNorm(self.hidden_dim)

        self.pathway_prototypes = nn.Parameter(
            torch.randn(self.num_pathways, self.hidden_dim) * 0.02
        )
        self.phenotype_prototypes = nn.Parameter(
            torch.randn(self.num_phenotypes, self.hidden_dim) * 0.02
        )

        self.gene_to_pathway_update = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.pathway_gate = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, 1),
        )
        self.pathway_to_phenotype_update = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.alpha_pathway = nn.Parameter(torch.tensor(-2.0))

        self.gene_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, 1)
        )
        self.pathway_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, 1)
        )
        self.phenotype_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, 1)
                )
                for _ in range(self.num_phenotypes)
            ]
        )

    def gene_queries(self) -> Tensor:
        return self.gene_query_norm(
            self.gene_query_projection(self.gene_identity_embeddings)
            + self.gene_query_delta
        )

    def gene_pathway_weights(self) -> Tensor:
        """Fixed biological Gene -> Pathway membership."""
        return self.H_prior

    def pathway_phenotype_weights(self) -> Tensor:
        """Trainable Pathway -> Phenotype relation strengths."""
        return torch.tanh(self.R_theta)

    @staticmethod
    def _cross_attention(
        queries: Tensor, patches: Tensor, patch_mask: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        if patches.ndim != 3:
            raise ValueError("patches must be [B, N, D]")
        if queries.ndim == 2:
            queries = queries.unsqueeze(0).expand(patches.shape[0], -1, -1)
        scores = torch.matmul(queries, patches.transpose(-1, -2))
        scores = scores / math.sqrt(patches.shape[-1])
        if patch_mask is not None:
            scores = scores.masked_fill(
                ~patch_mask.bool().unsqueeze(1),
                torch.finfo(scores.dtype).min,
            )
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, patches), attention

    def _gene_to_pathway(self, z_gene: Tensor) -> Tensor:
        weights = self.H_prior
        weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-6)
        return torch.einsum("gp,bgh->bph", weights, z_gene)

    def _pathway_to_phenotype(self, z_pathway: Tensor, z_pheno: Tensor) -> Tensor:
        relations = self.pathway_phenotype_weights()
        denom = relations.abs().sum(dim=0).clamp_min(1.0)
        message = torch.einsum("pt,bph->bth", relations, z_pathway)
        message = message / denom.view(1, -1, 1)
        scale = 0.5 * torch.sigmoid(self.alpha_pathway)
        return z_pheno + scale * self.pathway_to_phenotype_update(message)

    def forward(
        self,
        features: Tensor,
        patch_mask: Optional[Tensor] = None,
    ) -> Dict[str, object]:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected features [B, N, {self.feature_dim}], "
                f"got {tuple(features.shape)}"
            )

        patches = self.patch_projector(features.float())

        gene_queries = self.gene_queries()
        z_gene, gene_attention = self._cross_attention(
            gene_queries, patches, patch_mask
        )
        gene_pred = self.gene_head(z_gene).squeeze(-1)

        z_pathway_wsi, pathway_attention = self._cross_attention(
            self.pathway_prototypes, patches, patch_mask
        )
        z_pathway_gene = self._gene_to_pathway(z_gene)
        gate = torch.sigmoid(
            self.pathway_gate(torch.cat([z_pathway_wsi, z_pathway_gene], dim=-1))
        )
        z_pathway = z_pathway_wsi + gate * self.gene_to_pathway_update(
            z_pathway_gene
        )
        pathway_pred = self.pathway_head(z_pathway).squeeze(-1)

        z_pheno, phenotype_attention = self._cross_attention(
            self.phenotype_prototypes, patches, patch_mask
        )
        z_pheno = self._pathway_to_phenotype(z_pathway, z_pheno)
        phenotype_logits: List[Tensor] = [
            head(z_pheno[:, index]).squeeze(-1)
            for index, head in enumerate(self.phenotype_heads)
        ]

        return {
            "gene_pred": gene_pred,
            "gene_attention": gene_attention,
            "pathway_pred": pathway_pred,
            "pathway_attention": pathway_attention,
            "phenotype_logits": phenotype_logits,
            "phenotype_attention": phenotype_attention,
            "H": self.gene_pathway_weights(),
            "R": self.pathway_phenotype_weights(),
            "gene_embeddings": z_gene,
            "pathway_embeddings": z_pathway,
            "phenotype_embeddings": z_pheno,
        }
