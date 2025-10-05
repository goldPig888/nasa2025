"""Wrapper for generating personalized recommendations using GPT-style models."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

try:  # Prefer the official OpenAI client when available
    import openai  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    openai = None  # type: ignore

if openai is not None:
    OPENAI_RATE_LIMIT_ERROR = getattr(openai, "RateLimitError", None)
    if OPENAI_RATE_LIMIT_ERROR is None and hasattr(openai, "error"):
        OPENAI_RATE_LIMIT_ERROR = getattr(openai.error, "RateLimitError", None)
else:  # pragma: no cover - handled at runtime
    OPENAI_RATE_LIMIT_ERROR = None


def _resolve_openai_settings() -> Tuple[str, str]:
    """Load the most up-to-date OpenAI credentials from the environment."""

    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    load_dotenv(override=False)

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
    return api_key, model


_ALLERGEN_SYNONYM_MAP = {
    "ragweed": "ragweed",
    "rag-weed": "ragweed",
    "grass": "grass",
    "grasses": "grass",
    "tree": "tree",
    "trees": "tree",
    "weed": "weed",
    "weeds": "weed",
    "cedar": "cedar",
    "juniper": "juniper",
    "pine": "pine",
    "oak": "oak",
    "birch": "birch",
    "maple": "maple",
    "elm": "elm",
    "mold": "mold",
}


def _format_readable_list(items: Sequence[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _extract_allergen_focus(
    sensitive_groups: Sequence[str],
    user_notes: str,
) -> List[str]:
    focus: set[str] = set()

    def _scan_text(text: str, require_trigger: bool = False) -> None:
        lowered = text.lower()
        if require_trigger and not any(token in lowered for token in ("allerg", "pollen", "hay fever", "hayfever")):
            return
        for token in re.split(r"[^a-z]+", lowered):
            if not token:
                continue
            mapped = _ALLERGEN_SYNONYM_MAP.get(token)
            if mapped:
                focus.add(mapped)

    for group in sensitive_groups:
        if not group:
            continue
        _scan_text(group)

    if user_notes:
        _scan_text(user_notes, require_trigger=True)

    return sorted(focus)


def _build_prompt(
    city: str,
    sensitive_groups: Sequence[str],
    user_notes: str,
    pollen_summary: Optional[str],
    air_quality_summary: Optional[str],
    custom_prompt: Optional[str],
) -> str:
    parts = [
        "You are a wellness assistant providing actionable environmental health guidance.",
        f"Current city: {city or 'Unknown City'}.",
    ]

    allergen_focus = _extract_allergen_focus(sensitive_groups, user_notes)
    focus_list = _format_readable_list(allergen_focus)

    if sensitive_groups:
        parts.append("Sensitive groups: " + ", ".join(sensitive_groups) + ".")
    if user_notes:
        parts.append("User notes: " + user_notes.strip())
    if pollen_summary:
        parts.append("Pollen outlook: " + pollen_summary)
    elif allergen_focus:
        parts.append(
            "Pollen outlook: Personal allergen focus with no forecast — "
            + focus_list
            + "."
        )
    if air_quality_summary:
        parts.append("Air quality outlook: " + air_quality_summary)
    if allergen_focus:
        parts.append("Allergen tags to emphasize: " + ", ".join(allergen_focus) + ".")

    structure_clauses = [
        "Begin directly with section headings; do not restate 'Personalized Action Steps' or add an introduction.",
        "Include a bold heading 'Pollution Action Steps' followed immediately by exactly three numbered items covering scheduling, filtration, protective gear, hydration, and medication adherence.",
        "Number each item as '1.', '2.', '3.' and keep them under about 25 words.",
        "Add a bold heading 'Pollen & Allergies Action Steps' only when a pollen forecast or allergen focus is available; when you include it, provide exactly three numbered items that call out the specific allergens.",
        "If there is no pollen forecast and no allergen focus, omit the pollen section entirely and do not mention pollen elsewhere.",
    ]

    if allergen_focus:
        structure_clauses.append(
            "When you include the pollen section, reference "
            + focus_list
            + " in the numbered guidance."
        )

    structure_text = " ".join(structure_clauses)

    if custom_prompt:
        instructions = custom_prompt.strip()
        if instructions and not instructions.endswith(('.', '!', '?')):
            instructions += "."
        instructions += " " + structure_text
    else:
        instructions = structure_text

    parts.append("Instructions: " + instructions)
    return "\n".join(parts)


def _is_rate_limit_error(error: Exception) -> bool:
    text = str(error).lower()
    if "insufficient_quota" in text or "insufficient quota" in text:
        return False

    if OPENAI_RATE_LIMIT_ERROR and isinstance(error, OPENAI_RATE_LIMIT_ERROR):
        return True

    status = getattr(error, "status", None) or getattr(error, "status_code", None)
    if status == 429 and "quota" not in text:
        return True

    return ("rate limit" in text or "too many requests" in text) and "quota" not in text


def _invoke_openai_chat(
    api_key: str,
    model_name: str,
    messages,
    *,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, str]:
    if openai is None:
        raise RuntimeError("openai package is not installed")

    if hasattr(openai, "OpenAI"):
        client_cls = getattr(openai, "OpenAI")
        client = client_cls(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        used_model = getattr(response, "model", model_name)
        return content, used_model

    openai.api_key = api_key  # type: ignore[attr-defined]
    response = openai.ChatCompletion.create(  # type: ignore[attr-defined]
        model=model_name,
        messages=messages,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
    content = str(response["choices"][0]["message"]["content"]).strip()
    used_model = response.get("model", model_name)
    return content, used_model


def _fallback_plan(
    city: str,
    sensitive_groups: Sequence[str],
    user_notes: str,
    pollen_summary: Optional[str],
    air_quality_summary: Optional[str],
    error_note: Optional[str] = None,
) -> Dict[str, str]:
    allergen_focus = _extract_allergen_focus(sensitive_groups, user_notes)

    lines = [
        f"**Location:** {city or 'Unknown city'}",
    ]
    if sensitive_groups:
        lines.append("**Sensitive groups:** " + ", ".join(sensitive_groups))
    if user_notes:
        lines.append("**User notes:** " + user_notes.strip())
    if air_quality_summary:
        lines.append("**Air quality snapshot:** " + air_quality_summary)
    if pollen_summary:
        lines.append("**Pollen snapshot:** " + pollen_summary)
    elif allergen_focus:
        lines.append(
            "**Pollen snapshot:** No forecast available; personal focus on "
            + ", ".join(allergen_focus)
            + "."
        )

    pollution_steps = [
        "1. Spend more time indoors during high AQI hours and shift errands to the cleanest forecast period.",
        "2. Run HEPA filtration, seal windows, and limit ventilation when particulate levels spike.",
        "3. Use respirators or prescribed inhalers before outdoor exertion and hydrate every few hours.",
    ]

    body_lines = lines + [
        "",
        "**Pollution Action Steps**",
        *pollution_steps,
    ]

    include_pollen_section = bool(pollen_summary or allergen_focus)

    if include_pollen_section:
        if allergen_focus:
            focus_list = _format_readable_list(allergen_focus)
            pollen_steps = [
                f"1. Check daily {focus_list} alerts and pre-treat per your allergy plan.",
                "2. Rinse off, shower, and wipe pets after outdoor time to clear lingering pollen.",
                "3. Run bedroom HEPA filters on high and dry laundry indoors until counts ease.",
            ]
            section_title = (
                "**Pollen & Allergies Action Steps (focus: "
                + ", ".join(allergen_focus)
                + ")**"
            )
        else:
            pollen_steps = [
                "1. Share elevated pollen alerts with allergy-prone friends or neighbors.",
                "2. Delay mowing or yardwork midday to limit pollen release and support local air quality.",
                "3. Keep windows shut and run filters to help guests who react to seasonal pollen.",
            ]
            section_title = "**Pollen & Allergies Action Steps**"

        body_lines += [
            "",
            section_title,
            *pollen_steps,
        ]

    body = "\n".join(body_lines)
    if error_note:
        body += f"\n\n_Using local fallback guidance (API unavailable: {error_note})._"
    else:
        body += "\n\n_Using local fallback guidance._"

    return {"content": body, "model": "fallback"}


def generate_action_plan(
    city: str,
    sensitive_groups: Sequence[str],
    user_notes: str,
    pollen_summary: Optional[str],
    air_quality_summary: Optional[str],
    custom_prompt: Optional[str] = None,
    temperature: float = 0.4,
    max_tokens: int = 350,
) -> Dict[str, str]:
    api_key, model_name = _resolve_openai_settings()

    if not api_key:
        return _fallback_plan(
            city=city,
            sensitive_groups=sensitive_groups,
            user_notes=user_notes,
            pollen_summary=pollen_summary,
            air_quality_summary=air_quality_summary,
            error_note="missing OPENAI_API_KEY",
        )

    prompt = _build_prompt(
        city=city,
        sensitive_groups=sensitive_groups,
        user_notes=user_notes,
        pollen_summary=pollen_summary,
        air_quality_summary=air_quality_summary,
        custom_prompt=custom_prompt,
    )

    if openai is None:
        return _fallback_plan(
            city=city,
            sensitive_groups=sensitive_groups,
            user_notes=user_notes,
            pollen_summary=pollen_summary,
            air_quality_summary=air_quality_summary,
            error_note="openai client unavailable",
        )

    messages = [
        {"role": "system", "content": "You are a concise environmental health coach."},
        {"role": "user", "content": prompt},
    ]

    attempts = 2
    last_error_note: Optional[str] = None
    last_exception: Optional[Exception] = None

    for attempt in range(attempts):
        try:
            content, used_model = _invoke_openai_chat(
                api_key,
                model_name,
                messages,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
            if content:
                return {"content": content, "model": used_model}
            last_error_note = "Empty response"
            break
        except Exception as exc:  # pragma: no cover - network path
            message_lower = str(exc).lower()
            is_rate_limited = _is_rate_limit_error(exc)
            if is_rate_limited and attempt < attempts - 1:
                last_error_note = "429 rate limit"
                time.sleep(1.5 * (attempt + 1))
                continue
            last_exception = exc
            if "insufficient_quota" in message_lower or "insufficient quota" in message_lower:
                last_error_note = "insufficient quota"
            elif is_rate_limited:
                last_error_note = "429 rate limit"
            else:
                last_error_note = str(exc)
            break

    if last_exception:
        print("[advisor] OpenAI request failed:", last_exception)

    error_note = last_error_note or (str(last_exception) if last_exception else "Unknown error")
    return _fallback_plan(
        city=city,
        sensitive_groups=sensitive_groups,
        user_notes=user_notes,
        pollen_summary=pollen_summary,
        air_quality_summary=air_quality_summary,
        error_note=error_note,
    )
