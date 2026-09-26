from __future__ import annotations

from typing import Any, Dict, List


class KnowledgeRAG:
    """Answer-free pathology knowledge interface.

    The full system can replace this with an external knowledge source. The
    public implementation stores only generic evidence semantics and scale-role
    guidance; it contains no patient, benchmark, or answer data.
    """

    SCALE_GUIDANCE = {
        "10x": [
            "Inspect global architecture, tumor distribution, and broad tissue organization."
        ],
        "20x": [
            "Inspect intermediate structural patterns and local tissue relationships."
        ],
        "40x": [
            "Inspect fine morphology and cytologic detail when the question requires it."
        ],
    }

    def retrieve(
        self,
        question: str,
        choices: List[str],
        target_phenotypes: List[str],
    ) -> Dict[str, Any]:
        del question, choices
        return {
            "target_phenotypes": list(target_phenotypes),
            "evidence_rules": [
                "Direct phenotype evidence is evaluated before supportive Pathway/Gene evidence.",
                "Visual evidence must be tied to visible H&E morphology.",
                "Pathway/Gene outputs are WSI-derived supportive predictions.",
            ],
            "scale_specific_visual_guidance": dict(self.SCALE_GUIDANCE),
            "limitations": [
                "WSI-derived biological evidence is not a measured molecular assay."
            ],
        }
