"""Helpers to look up local clean-air initiatives using the project's GPT wrapper.

Provides:
- find_initiatives(city) -> list[dict]: parsed candidates with name and handle
- cli entrypoint for quick testing

This module uses `app.lib.gpt_advisor.generate_action_plan` for the GPT call.
"""

from __future__ import annotations
import json
import re
from typing import List, Dict

try:
    from app.lib.gpt_advisor import generate_action_plan
except Exception:
    # relative import fallback for script use
    try:
        from lib.gpt_advisor import generate_action_plan
    except Exception:
        raise

PROMPT = (
    "Provide a JSON array of local clean-air initiatives or community groups focused on air quality in {city}. "
    "Return an array of objects with keys `name` and `handle` (handle may be empty). "
    "If you are unsure, return an empty array. Only output JSON if possible."
)


def _extract_first_json_array(text: str) -> List[Dict]:
    if not text:
        return []
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    m = re.search(r"(\[.*?\])", text, flags=re.DOTALL)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(1))
        if isinstance(parsed, list):
            return parsed
    except Exception:
        return []
    return []


def find_initiatives(city: str) -> List[Dict[str, str]]:
    prompt = PROMPT.format(city=city)
    resp = generate_action_plan(
        city=city,
        sensitive_groups=[],
        user_notes="",
        pollen_summary=None,
        air_quality_summary=None,
        custom_prompt=prompt,
        temperature=0.2,
    )
    if isinstance(resp, dict):
        content = resp.get("content") or resp.get("result") or ""
    else:
        content = str(resp)

    candidates = _extract_first_json_array(content)
    parsed = []
    for item in candidates:
        if isinstance(item, dict):
            name = (item.get("name") or item.get("title") or "").strip()
            handle = (item.get("handle") or item.get("handle_tag") or "").strip()
            parsed.append({"name": name, "handle": handle})
        else:
            parsed.append({"name": str(item), "handle": ""})
    return parsed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="your city")
    args = parser.parse_args()
    res = find_initiatives(args.city)
    print(json.dumps(res, indent=2))
