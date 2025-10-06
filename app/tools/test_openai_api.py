"""Quick diagnostic script to verify OPENAI_API_KEY works with Chat Completions."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import requests
from dotenv import load_dotenv


# Attempt to load environment variables from app/.env if present
DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH)

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
ENDPOINT = "https://api.openai.com/v1/chat/completions"
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DEFAULT_PROMPT = "Reply with a short confirmation that the API key is working."


def run(prompt: str = DEFAULT_PROMPT) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing or empty in the environment")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a diagnostic assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 120,
        "temperature": 0.1,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(ENDPOINT, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    user_prompt = " ".join(sys.argv[1:]).strip()
    if not user_prompt:
        user_prompt = DEFAULT_PROMPT
    try:
        result = run(user_prompt)
        choice = result.get("choices", [{}])[0].get("message", {}).get("content", "<no content>")
        print("✓ OpenAI response:")
        print(json.dumps(result, indent=2)[:2000])
        print("\nAssistant message:\n", choice)
    except Exception as exc:  # noqa: BLE001
        print("✗ OpenAI call failed:", exc)
        sys.exit(1)
