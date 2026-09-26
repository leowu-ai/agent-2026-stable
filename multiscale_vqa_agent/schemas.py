from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


MAGNIFICATIONS = ("10x", "20x", "40x")


@dataclass
class ExecutionPlan:
    case_id: str
    question: str
    choices: List[str]
    target_phenotypes: List[str]
    preferred_magnification: str = "10x"
    required_evidence: List[str] = field(default_factory=lambda: ["phenotype", "morphology"])
    answer_mode: str = "single_choice"
    evidence_route: str = "phenotype_direct"
    use_pathology_agent: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatchCandidate:
    magnification: str
    patch_index: int
    score: float
    x: Optional[float] = None
    y: Optional[float] = None
    source: str = "phenotype_attention"
    parent_patch_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceGroup:
    group_id: int
    score: float
    patches: Dict[str, PatchCandidate] = field(default_factory=dict)
    evidence_source: str = "phenotype"
    parent_group_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "score": self.score,
            "evidence_source": self.evidence_source,
            "parent_group_id": self.parent_group_id,
            "patches": {scale: patch.to_dict() for scale, patch in self.patches.items()},
        }
