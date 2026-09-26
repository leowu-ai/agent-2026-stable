"""Core BioTrace evidence-space forward pass.

This module intentionally contains only the inference-side computation needed to
inspect the structured Gene -> Pathway -> Phenotype evidence space. Dataset
construction, target generation, RNA/ssGSEA supervision, training loops,
checkpoints, and benchmark-specific adapters are not part of this review artifact.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping

import torch
from torch import Tensor, nn


class _ScaleEvidenceBlock(nn.Module):
    """One scale-specific structured evidence model."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_genes: int,
        num_pathways: int,
        num_phenotypes: int,
        gene_pathway_support: Tensor,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.patch_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gene_prototypes = nn.Parameter(torch.randn(num_genes, hidden_dim) * 0.02)
        self.pathway_prototypes = nn.Parameter(torch.randn(num_pathways, hidden_dim) * 0.02)
        self.phenotype_prototypes = nn.Parameter(torch.randn(num_phenotypes, hidden_dim) * 0.02)

        # The biological prior constrains which Gene -> Pathway edges may contribute.
        self.register_buffer(
            "gene_pathway_support", gene_pathway_support.float().clamp(0, 1)
        )
        self.gene_pathway_strength = nn.Parameter(torch.zeros(num_genes, num_pathways))
        self.pathway_phenotype_strength = nn.Parameter(
            torch.randn(num_pathways, num_phenotypes) * 0.02
        )

        self.gene_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
            for _ in range(num_genes)
        ])
        self.pathway_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
            for _ in range(num_pathways)
        ])
        self.phenotype_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
            for _ in range(num_phenotypes)
        ])

    @staticmethod
    def _ground(prototypes: Tensor, patches: Tensor) -> tuple[Tensor, Tensor]:
        scores = torch.einsum("ch,bnh->bcn", prototypes, patches)
        attention = torch.softmax(scores / math.sqrt(patches.shape[-1]), dim=-1)
        representation = torch.einsum("bcn,bnh->bch", attention, patches)
        return attention, representation

    @staticmethod
    def _predict(representations: Tensor, heads: nn.ModuleList) -> Tensor:
        values = [
            head(representations[:, index]).squeeze(-1)
            for index, head in enumerate(heads)
        ]
        return torch.stack(values, dim=-1)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected [B, N, {self.feature_dim}] features, got {tuple(features.shape)}"
            )
        patches = self.patch_projector(features.float())
        gene_attn, gene_direct = self._ground(self.gene_prototypes, patches)
        pathway_attn, pathway_direct = self._ground(self.pathway_prototypes, patches)
        phenotype_attn, phenotype_direct = self._ground(self.phenotype_prototypes, patches)

        gp = self.gene_pathway_support * torch.sigmoid(self.gene_pathway_strength)
        gp_norm = gp.sum(dim=0, keepdim=True).clamp_min(1e-6)
        pathway_from_gene = torch.einsum("gp,bgh->bph", gp / gp_norm, gene_direct)
        pathway_repr = pathway_direct + pathway_from_gene

        pp = torch.tanh(self.pathway_phenotype_strength)
        pp_norm = pp.abs().sum(dim=0, keepdim=True).clamp_min(1.0)
        phenotype_from_pathway = torch.einsum("pt,bph->bth", pp / pp_norm, pathway_repr)
        phenotype_repr = phenotype_direct + phenotype_from_pathway

        return {
            "gene_scores": self._predict(gene_direct, self.gene_heads),
            "pathway_scores": self._predict(pathway_repr, self.pathway_heads),
            "phenotype_scores": self._predict(phenotype_repr, self.phenotype_heads),
            "gene_spatial_response": gene_attn,
            "pathway_spatial_response": pathway_attn,
            "phenotype_spatial_response": phenotype_attn,
            "gene_pathway_relations": gp,
            "pathway_phenotype_relations": pp,
        }


class BioTraceEvidenceSpace(nn.Module):
    """Compact paper-aligned multiscale structured evidence space."""

    SCALES = ("10x", "20x", "40x")

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_genes: int,
        num_pathways: int,
        num_phenotypes: int,
        gene_pathway_support: Tensor,
    ) -> None:
        super().__init__()
        if tuple(gene_pathway_support.shape) != (num_genes, num_pathways):
            raise ValueError(
                "gene_pathway_support must have shape [num_genes, num_pathways]"
            )
        self.blocks = nn.ModuleDict({
            scale: _ScaleEvidenceBlock(
                feature_dim,
                hidden_dim,
                num_genes,
                num_pathways,
                num_phenotypes,
                gene_pathway_support,
            )
            for scale in self.SCALES
        })

    def forward(
        self, multiscale_features: Mapping[str, Tensor]
    ) -> Dict[str, Dict[str, Tensor]]:
        missing = [scale for scale in self.SCALES if scale not in multiscale_features]
        if missing:
            raise KeyError(f"missing magnification(s): {missing}")
        return {
            scale: self.blocks[scale](multiscale_features[scale])
            for scale in self.SCALES
        }
