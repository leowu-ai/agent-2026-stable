from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent_memory import WorkingMemory
from .clients import OpenAICompatibleClient, parse_json_response


PROMPT_PATH = Path(__file__).with_name("prompts") / "fusion_arbiter.txt"


def indexed_choices(choices: List[str]) -> List[Dict[str, str]]:
    return [
        {"id": chr(ord("A") + index), "text": text}
        for index, text in enumerate(choices)
    ]


class FusionAgent:
    """Final evidence arbitration interface."""

    def __init__(
        self,
        client: Optional[OpenAICompatibleClient] = None,
    ) -> None:
        self.client = client or OpenAICompatibleClient(enabled=False)
        self.system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def answer(self, memory: WorkingMemory) -> Dict[str, Any]:
        options = indexed_choices(memory.choices)
        if self.client.enabled:
            payload = {
                "question": memory.question,
                "choices": options,
                "structured_evidence": memory.structured_evidence,
                "observations": [
                    observation.to_dict() for observation in memory.observations
                ],
                "verifier": memory.final_verifier,
                "limitations": memory.knowledge.get("limitations", []),
            }
            raw = self.client.chat(
                self.system_prompt,
                json.dumps(payload, ensure_ascii=False),
                max_tokens=350,
            )
            parsed = parse_json_response(raw)
            if isinstance(parsed, dict):
                valid = {row["id"]: row["text"] for row in options}
                answer_id = str(parsed.get("answer_id", "")).upper()
                if answer_id in valid:
                    return {
                        "answer_id": answer_id,
                        "answer": valid[answer_id],
                        "confidence": float(parsed.get("confidence", 0.0)),
                        "explanation": str(parsed.get("explanation", "")),
                        "limitations": str(parsed.get("limitations", "")),
                        "backend": "language_reasoner",
                    }

        candidate = memory.structured_candidate or {}
        index = int(candidate.get("choice_index", 0))
        index = max(0, min(index, len(options) - 1))
        selected = options[index]
        return {
            "answer_id": selected["id"],
            "answer": selected["text"],
            "confidence": round(float(memory.structured_confidence), 6),
            "explanation": "Selected the synthetic structured candidate after evidence verification.",
            "limitations": "Random-feature demo; no clinical interpretation is intended.",
            "backend": "synthetic",
        }
