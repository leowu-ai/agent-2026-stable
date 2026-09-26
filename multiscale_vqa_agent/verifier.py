from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent_memory import WorkingMemory
from .clients import OpenAICompatibleClient, parse_json_response


PROMPT_PATH = Path(__file__).with_name("prompts") / "verifier.txt"
EVIDENCE_STATES = {
    "sufficient",
    "partial",
    "conflicting",
    "insufficient",
    "unavailable",
}


class EvidenceVerifierAgent:
    """Evidence-sufficiency controller.

    With a language-model client enabled, verification is adaptive and follows
    the evidence-state/action contract used by BioTrace. When the client is
    disabled, a label-free structural fallback is used only to exercise the
    forward interfaces on synthetic data.
    """

    def __init__(
        self,
        client: Optional[OpenAICompatibleClient] = None,
    ) -> None:
        self.client = client or OpenAICompatibleClient(enabled=False)
        self.system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def verify(
        self,
        memory: WorkingMemory,
        available_actions: List[str],
        pathway_candidates: Optional[List[Dict[str, Any]]] = None,
        gene_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if self.client.enabled:
            payload = {
                "plan": memory.plan,
                "structured_evidence": memory.structured_evidence,
                "observations": [
                    observation.to_dict() for observation in memory.observations
                ],
                "current_missing_evidence": memory.current_missing_evidence,
                "available_actions": available_actions,
                "pathway_candidates": pathway_candidates or [],
                "gene_candidates": gene_candidates or [],
            }
            raw = self.client.chat(
                self.system_prompt,
                json.dumps(payload, ensure_ascii=False),
                max_tokens=450,
            )
            normalized = self._normalize(parse_json_response(raw), available_actions)
            if normalized is not None:
                return normalized
        return self._fallback(memory, available_actions)

    @staticmethod
    def _normalize(
        parsed: Optional[Dict[str, Any]],
        available_actions: List[str],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(parsed, dict):
            return None
        state = str(parsed.get("evidence_state", "insufficient")).lower()
        action = str(parsed.get("next_action", "")).lower()
        if state not in EVIDENCE_STATES or action not in available_actions:
            return None
        sufficient = bool(parsed.get("evidence_sufficient"))
        if action == "answer" and not sufficient:
            return None
        return {
            "evidence_sufficient": sufficient,
            "evidence_state": "sufficient" if sufficient else state,
            "missing_evidence_type": str(
                parsed.get("missing_evidence_type") or "none"
            ),
            "conflict_detected": bool(parsed.get("conflict_detected")),
            "next_action": action,
            "target": parsed.get("target"),
            "reason": str(parsed.get("reason") or "")[:500],
            "decision_source": "language_verifier",
        }

    @staticmethod
    def _fallback(
        memory: WorkingMemory,
        available_actions: List[str],
    ) -> Dict[str, Any]:
        # No answer labels, option semantics, confidence thresholds, or
        # benchmark-specific rules are used here. The fallback simply walks the
        # evidence hierarchy so the synthetic example can exercise every
        # interface without an external language-model service.
        for action, missing in (
            ("inspect_10x", "coarse_visual"),
            ("inspect_20x", "intermediate_visual"),
            ("inspect_40x", "fine_visual"),
            ("inspect_pathway", "pathway_support"),
            ("inspect_gene", "gene_support"),
        ):
            if action in available_actions:
                return {
                    "evidence_sufficient": False,
                    "evidence_state": (
                        "partial" if memory.observations else "insufficient"
                    ),
                    "missing_evidence_type": missing,
                    "conflict_detected": False,
                    "next_action": action,
                    "target": None,
                    "reason": f"Exercise the next available {missing} interface.",
                    "decision_source": "synthetic_structural_fallback",
                }

        if "answer" in available_actions:
            return {
                "evidence_sufficient": False,
                "evidence_state": "partial",
                "missing_evidence_type": "none",
                "conflict_detected": False,
                "next_action": "answer",
                "target": None,
                "reason": (
                    "No additional synthetic evidence action remains; "
                    "proceed to the final arbitration interface with uncertainty."
                ),
                "decision_source": "synthetic_structural_fallback",
            }

        return {
            "evidence_sufficient": False,
            "evidence_state": "unavailable",
            "missing_evidence_type": "unavailable",
            "conflict_detected": False,
            "next_action": "unavailable",
            "target": None,
            "reason": "No further evidence action is available.",
            "decision_source": "synthetic_structural_fallback",
        }
