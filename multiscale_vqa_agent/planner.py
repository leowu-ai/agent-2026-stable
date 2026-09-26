from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .clients import OpenAICompatibleClient, parse_json_response
from .registry import ToolBankRegistry
from .schemas import ExecutionPlan


PROMPT_PATH = Path(__file__).with_name("prompts") / "planner.txt"


class EvidencePlanner:
    """Question -> evidence-requirement interface."""

    def __init__(
        self,
        registry: ToolBankRegistry,
        client: Optional[OpenAICompatibleClient] = None,
    ) -> None:
        self.registry = registry
        self.client = client or OpenAICompatibleClient(enabled=False)
        self.system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def plan(
        self,
        case_id: str,
        question: str,
        choices: List[str],
    ) -> ExecutionPlan:
        parsed = None
        if self.client.enabled:
            payload = {
                "question": question,
                "choices": choices,
                "concept_inventory": self.registry.concept_inventory(),
            }
            raw = self.client.chat(
                self.system_prompt,
                json.dumps(payload, ensure_ascii=False),
                max_tokens=350,
            )
            parsed = parse_json_response(raw)

        target = self._resolve_target(parsed)
        return ExecutionPlan(
            case_id=case_id,
            question=question,
            choices=list(choices),
            target_phenotypes=[target],
            preferred_magnification=str(
                (parsed or {}).get("preferred_magnification") or "10x"
            ),
            required_evidence=list(
                (parsed or {}).get("required_evidence")
                or ["phenotype", "morphology"]
            ),
            answer_mode=str((parsed or {}).get("answer_mode") or "single_choice"),
            evidence_route=str(
                (parsed or {}).get("evidence_route") or "phenotype_direct"
            ),
        )

    def _resolve_target(self, parsed) -> str:
        if isinstance(parsed, dict):
            targets = parsed.get("target_phenotypes")
            if isinstance(targets, list):
                for target in targets:
                    if str(target) in self.registry.phenotype_to_index:
                        return str(target)
        return self.registry.phenotypes[0]
