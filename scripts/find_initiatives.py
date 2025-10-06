#!/usr/bin/env python3
"""
scripts/find_initiatives.py

Call the project's GPT helper with a focused prompt to return a JSON array of
local clean-air initiatives (name + handle). Print the raw GPT response and
attempt to parse and pretty-print the JSON candidates.

Usage:
    python scripts/find_initiatives.py --city "Houston"

Note: Requires OPENAI_API_KEY set in your environment or a .env file at the repo root.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys

# Ensure repo root is on sys.path so we can import app.lib.gpt_advisor
repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from app.lib.gpt_advisor import generate_action_plan
except Exception as exc:
    raise RuntimeError("Could not import generate_action_plan from app.lib.gpt_advisor") from exc

PROMPT_TEMPLATE = (
    "Provide a JSON array of local clean-air initiatives or community groups focused on air quality in {city}. "
    "Return an array of objects with keys `name` and `handle` (handle may be empty). "
    "If you are unsure, return an empty array. Only return JSON if possible."
)


def extract_first_json_array(text: str) -> list:
    """Try to parse text as JSON array, fallback to extracting the first [...] substring."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # search for first JSON array
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="your city", help="City name to look up initiatives for")
    args = parser.parse_args()

    city = args.city
    prompt = PROMPT_TEMPLATE.format(city=city)

    print(f"Asking GPT for initiatives in: {city}\n")

    try:
        resp = generate_action_plan(
            city=city,
            sensitive_groups=[],
            user_notes="",
            pollen_summary=None,
            air_quality_summary=None,
            custom_prompt=prompt,
            temperature=0.2,
        )
    except Exception as exc:
        print("Error calling GPT helper:", exc)
        sys.exit(2)

    if isinstance(resp, dict):
        content = resp.get("content") or resp.get("result") or ""
    else:
        content = str(resp)

    print("--- RAW GPT RESPONSE ---\n")
    print(content)
    print("\n--- END RAW RESPONSE ---\n")

    candidates = extract_first_json_array(content)
    if not candidates:
        print("No JSON candidates found in GPT response.")
        sys.exit(0)

    print("Parsed initiative candidates:\n")
    for idx, item in enumerate(candidates, start=1):
        if isinstance(item, dict):
            name = item.get("name") or item.get("title") or ""
            handle = item.get("handle") or item.get("handle_tag") or ""
        else:
            # If items are plain strings, show them
            name = str(item)
            handle = ""
        print(f"{idx}. Name: {name}\n   Handle: {handle}\n")


if __name__ == "__main__":
    main()
