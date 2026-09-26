from __future__ import annotations

from typing import Dict, Iterable, List


class ToolBankRegistry:
    """Concept inventory exposed to the Agent."""

    def __init__(
        self,
        phenotype_names: Iterable[str],
        pathway_names: Iterable[str],
        gene_names: Iterable[str],
    ) -> None:
        self.phenotypes = list(phenotype_names)
        self.pathways = list(pathway_names)
        self.genes = list(gene_names)
        self.phenotype_to_index = {
            name: index for index, name in enumerate(self.phenotypes)
        }

    def concept_inventory(self) -> Dict[str, List[str]]:
        return {
            "phenotypes": list(self.phenotypes),
            "pathways": list(self.pathways),
            "genes": list(self.genes),
        }
