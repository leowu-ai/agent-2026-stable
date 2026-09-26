from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def parse_json_response(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


class OpenAICompatibleClient:
    """Optional client used by planner/verifier/fusion interfaces.

    The synthetic demo runs with this client disabled and therefore does not
    require model services, network access, credentials, or local paths.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        enabled: bool = False,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.model = model or ""
        self.api_key = api_key or os.environ.get("BIOTRACE_API_KEY", "")
        self.enabled = bool(enabled and self.base_url and self.model)

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Optional[str]:
        if not self.enabled:
            return None
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key or 'EMPTY'}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
            return None
