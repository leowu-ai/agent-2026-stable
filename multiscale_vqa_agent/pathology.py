from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .clients import OpenAICompatibleClient, parse_json_response
from .schemas import EvidenceGroup


PROMPT_PATH = Path(__file__).with_name("prompts") / "pathology_observer.txt"


class PathologyAgent:
    """Patho-R1 style morphology observer interface."""

    def __init__(
        self,
        client: Optional[OpenAICompatibleClient] = None,
    ) -> None:
        self.client = client or OpenAICompatibleClient(enabled=False)
        self.system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def describe(
        self,
        question: str,
        groups: List[EvidenceGroup],
        magnification: str,
        visual_guidance: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        metadata = [group.to_dict() for group in groups]

        if self.client.enabled:
            payload = {
                "inspection_question": question,
                "magnification": magnification,
                "scale_specific_visual_guidance": list(visual_guidance or []),
                "selected_regions": metadata,
            }
            raw = self.client.chat(
                self.system_prompt,
                json.dumps(payload, ensure_ascii=False),
                max_tokens=300,
            )
            parsed = parse_json_response(raw)
            if isinstance(parsed, dict):
                return {
                    "backend": "pathology_model",
                    "visible_findings": list(parsed.get("visible_findings") or [])[:5],
                    "image_quality": parsed.get("image_quality", "limited"),
                    "groups": metadata,
                }

        return {
            "backend": "synthetic",
            "visible_findings": [
                f"{len(groups)} question-relevant region group(s) selected at {magnification}; "
                "pixel-level morphology is unavailable in the random-feature demo."
            ],
            "image_quality": "limited",
            "groups": metadata,
        }
