"""
lead_capture.py — end-of-call lead extraction + HOT/WARM/COLD classification.

WHY THIS EXISTS
---------------
The old lead path had two structural bugs:

 1. `create_lead()` built its record ONLY from `AppointmentState`, which is populated
    solely by the model choosing to call `update_booking_details()`. If the model just
    talked (the normal case), every structured field stayed null and the information the
    caller actually said ended up as free text in `notes`. Nothing ever read the
    transcript, and nothing ever looked at the SIP caller ID.

 2. A lead was only written if the model chose to call `create_lead()`. Calls that ended
    any other way — hangup, transfer, confirmed booking, silence timeout, error — wrote
    no lead at all.

This module fixes both: it reads the FULL transcript, extracts every field, resolves the
phone from spoken digits OR caller ID, classifies the lead, and ALWAYS returns a record —
falling back to deterministic rules if the LLM extraction fails. It is pure/injectable so
it can be unit-tested without a live call.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

LEAD_QUALITIES = ("HOT", "WARM", "COLD")

# ---------------------------------------------------------------------------
# The lead record schema (documentation + validation target)
# ---------------------------------------------------------------------------
LEAD_RECORD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # identity / bookkeeping
        "lead_id": {"type": "string", "description": "LEAD###### — assigned on save"},
        "type": {"type": "string", "const": "PUNO_LEAD"},
        "created_at": {"type": "string", "description": "YYYY-MM-DD HH:MM:SS"},
        "call_id": {"type": ["string", "null"]},
        # classification (NEW — present on every lead)
        "lead_quality": {"type": "string", "enum": list(LEAD_QUALITIES)},
        "reasoning": {"type": "string", "description": "one or two sentences justifying lead_quality"},
        # extracted caller details — null ONLY when never mentioned
        "name": {"type": ["string", "null"]},
        "phone": {"type": ["string", "null"], "description": "10-digit; spoken digits win over caller ID"},
        "phone_source": {"type": "string", "enum": ["spoken", "caller_id", "none"]},
        "intent": {"type": ["string", "null"], "description": "short label, e.g. 'birthday party'"},
        "package_interest": {"type": ["string", "null"]},
        "preferred_date": {"type": ["string", "null"], "description": "YYYY-MM-DD"},
        "preferred_slot": {"type": ["string", "null"]},
        "adults_7plus": {"type": ["integer", "null"]},
        "kids_under7": {"type": ["integer", "null"]},
        "headcount_total": {"type": ["integer", "null"]},
        "notes": {"type": "string", "description": "free text ONLY for what has no typed field"},
        # call outcome
        "outcome": {
            "type": "string",
            "enum": ["Booked", "Lead created", "Transferred to human", "Info only", "Not interested"],
        },
        "booking_id": {"type": ["string", "null"]},
        "follow_up_needed": {"type": "boolean"},
        "language": {"type": ["string", "null"], "enum": ["en", "hi", "hinglish", None]},
        # provenance / diagnostics
        "turn_count": {"type": ["integer", "null"]},
        "caller_turn_count": {"type": ["integer", "null"]},
        "extraction_source": {"type": "string", "enum": ["llm", "rule_based_fallback"]},
        "payment_status": {"type": "string", "const": "n/a"},
    },
    "required": ["lead_quality", "reasoning", "phone_source", "notes", "outcome",
                 "extraction_source", "payment_status"],
}

# The subset the LLM is asked to produce (bookkeeping fields are added by us).
_LLM_FIELDS = (
    "name", "phone_spoken", "intent", "package_interest", "preferred_date",
    "preferred_slot", "adults_7plus", "kids_under7", "notes",
    "lead_quality", "reasoning", "caller_said_same_number", "wrong_number_or_spam",
)


# ---------------------------------------------------------------------------
# Transcript rendering
# ---------------------------------------------------------------------------
def render_transcript(turns: list[dict], max_chars: int = 24000) -> str:
    """Render conversation turns as 'CALLER:' / 'AGENT:' lines for the prompt.

    Keeps the TAIL when over-long: the end of a call carries the commitment signals
    that decide lead quality. Truncation is logged by the caller.
    """
    lines: list[str] = []
    for t in turns or []:
        if not isinstance(t, dict):
            continue
        role = str(t.get("role") or "").lower()
        text = str(t.get("text") or "").strip()
        if not text:
            continue
        who = "CALLER" if ("user" in role or "caller" in role) else "AGENT"
        lines.append(f"{who}: {text}")
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = "...[earlier turns truncated]...\n" + out[-max_chars:]
    return out


def count_turns(turns: list[dict]) -> tuple[int, int]:
    """(total_turns, caller_turns)."""
    total = caller = 0
    for t in turns or []:
        if not isinstance(t, dict):
            continue
        if not str(t.get("text") or "").strip():
            continue
        total += 1
        role = str(t.get("role") or "").lower()
        if "user" in role or "caller" in role:
            caller += 1
    return total, caller


# ---------------------------------------------------------------------------
# Phone resolution
# ---------------------------------------------------------------------------
def _digits10(raw: str) -> str:
    d = re.sub(r"\D", "", str(raw or ""))
    if len(d) > 10:
        d = d[-10:]
    return d if len(d) == 10 else ""


def resolve_phone(
    *,
    state_phone: Optional[str] = None,
    llm_spoken_phone: Optional[str] = None,
    caller_id: Optional[str] = None,
) -> tuple[Optional[str], str]:
    """Return (phone, source).

    Precedence, highest first:
      1. `state_phone`  — digits the caller actually dictated and the agent confirmed.
      2. `llm_spoken_phone` — digits the extractor found in the transcript.
      3. `caller_id` — the SIP calling number. This is the fix for "same number as this
         call", and also a general fallback: if the telephony layer knows the number
         there is no reason to leave the field null.
    """
    for cand, src in ((state_phone, "spoken"), (llm_spoken_phone, "spoken"), (caller_id, "caller_id")):
        d = _digits10(cand or "")
        if d:
            return d, src
    return None, "none"


# ---------------------------------------------------------------------------
# Deterministic classification (fallback + sanity check)
# ---------------------------------------------------------------------------
_READY_RE = re.compile(
    r"\b(book (it|me|kar)|confirm|kar do|kar dijiye|karwa d|final|pakka|done|"
    r"yes book|go ahead|slot (lock|book)|reserve)\b", re.IGNORECASE)
_WRONG_NUMBER_RE = re.compile(
    r"\b(wrong number|galat number|galti se|not interested|nahi chahiye|"
    r"kuch nahi|no thanks|remove my number|do not call)\b", re.IGNORECASE)


def classify_rule_based(
    *,
    booking_id: Optional[str] = None,
    name: Optional[str] = None,
    phone: Optional[str] = None,
    preferred_date: Optional[str] = None,
    package_interest: Optional[str] = None,
    headcount_total: Optional[int] = None,
    intent: Optional[str] = None,
    caller_turns: int = 0,
    transcript_text: str = "",
    transferred: bool = False,
) -> tuple[str, str]:
    """Deterministic HOT/WARM/COLD with reasoning. Used when the LLM is unavailable,
    and as the tie-break when the LLM returns something unusable."""
    t = transcript_text or ""

    if booking_id:
        return "HOT", f"Booking {booking_id} was confirmed on the call."

    if _WRONG_NUMBER_RE.search(t):
        return "COLD", "Caller indicated a wrong number or explicitly declined interest."

    if caller_turns <= 1:
        return "COLD", f"Caller spoke only {caller_turns} time(s) — no intent signal before the call ended."

    score = 0
    signals: list[str] = []
    if preferred_date:
        score += 2; signals.append("gave a specific date")
    if package_interest:
        score += 2; signals.append("named a package/activity")
    if phone:
        score += 1; signals.append("contact number available")
    if name:
        score += 1; signals.append("gave a name")
    if headcount_total:
        score += 1; signals.append("gave a headcount")
    if intent:
        score += 1; signals.append(f"intent '{intent}'")
    if _READY_RE.search(t):
        score += 4; signals.append("expressed readiness to book")
    if transferred:
        score += 1; signals.append("asked for a human specialist")
    # Engagement is itself a WARM signal: a caller who held a real back-and-forth
    # showed genuine interest even when the extractor captured no typed fields
    # (e.g. the LLM was unavailable). Without this a chatty, interested caller
    # scored COLD purely because nothing had been parsed out of the call.
    if caller_turns >= 3:
        score += 2; signals.append(f"engaged conversation ({caller_turns} caller turns)")
    if caller_turns >= 6:
        score += 1; signals.append("sustained conversation")

    detail = "; ".join(signals) if signals else "no concrete booking signals"
    if score >= 7:
        return "HOT", f"Ready to book: {detail}."
    if score >= 3:
        return "WARM", f"Genuine interest but no commitment: {detail}."
    return "COLD", f"Vague enquiry with little follow-through: {detail}."


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """You are a precise CRM data extractor for PUNO Jaipur, an entertainment centre in India.

Below is the FULL transcript of one inbound phone call between a CALLER and the AGENT.
Extract the caller's details into JSON. This feeds a sales CRM, so accuracy matters more
than completeness — but anything the caller actually said MUST end up in its own field.

TRANSCRIPT
----------
{transcript}
----------

CONTEXT ALREADY KNOWN (may be empty; trust it over your own reading):
{known}

RULES
1. Put every detail in its OWN typed field. Do NOT describe a detail in "notes" if a field
   exists for it. If the caller says "I'm Sam", name MUST be "Sam" — not a sentence in notes.
2. Use null ONLY when the caller genuinely never mentioned that detail anywhere in the call.
   Never use null just because the caller was vague — record your best literal reading.
3. The conversation is Hindi / English / Hinglish. Extract meaning regardless of language,
   but write field VALUES in English (except a person's name, which stays as spoken).
4. "phone_spoken": only digits the CALLER actually dictated in the conversation. If they
   said something like "same number", "yahi number", "isi number par" WITHOUT dictating
   digits, leave phone_spoken null and set caller_said_same_number true. Never invent digits.
5. "preferred_date": resolve relative dates to YYYY-MM-DD using TODAY = {today}.
   "tomorrow"/"kal" = the day after TODAY. If no date was mentioned, null.
6. "adults_7plus" / "kids_under7": integers. PUNO splits guests at age 7. If the caller gave
   only a total ("3-4 people"), put the lower bound in adults_7plus and leave kids_under7 null.
7. "intent": a SHORT label — e.g. "birthday party", "booking enquiry", "pricing enquiry",
   "group booking", "membership", "complaint", "wrong number".
8. "package_interest": the package or activity named. PUNO packages are "Tourist Pass -
   Weekdays", "Tourist Pass - Weekends", "Junior Fun n' Play"; activities include bowling,
   trampoline, VR, arcade, Mad Wheels, junior zone, birthday party. If the caller only ASKED
   what packages exist without naming one, use "asked for package information".
9. "notes": ONLY information that has no field of its own (special requests, context,
   objections). Do NOT repeat name/phone/date/package here.
10. "wrong_number_or_spam": true only for a genuine wrong number, silent/dead line, or spam.

LEAD QUALITY — assign exactly one:
- "HOT": clear intent AND a specific date or package AND reachable contact, and the caller
  was ready to book / asked to confirm / a booking was made.
- "WARM": genuine interest — asked real questions or gave partial details (name, date,
  package or headcount) — but did not commit.
- "COLD": no real intent — vague or generic question with no follow-through, wrong number,
  spam, or the caller ended the call almost immediately.
Put a one-or-two sentence justification in "reasoning", citing what the caller actually said.

Return ONLY a JSON object with exactly these keys:
{{"name": string|null, "phone_spoken": string|null, "intent": string|null,
"package_interest": string|null, "preferred_date": string|null, "preferred_slot": string|null,
"adults_7plus": integer|null, "kids_under7": integer|null, "notes": string,
"lead_quality": "HOT"|"WARM"|"COLD", "reasoning": string,
"caller_said_same_number": boolean, "wrong_number_or_spam": boolean}}
No markdown, no code fence, no commentary — just the JSON object."""


def build_extraction_prompt(transcript_text: str, known: dict | None, today: str) -> str:
    known_txt = json.dumps({k: v for k, v in (known or {}).items() if v not in (None, "", [])},
                           ensure_ascii=False) or "{}"
    return EXTRACTION_PROMPT.format(
        transcript=transcript_text or "(no transcript captured)",
        known=known_txt,
        today=today,
    )


def parse_extraction_json(raw: str) -> dict:
    """Parse the model's reply into a dict, tolerating fences/prose around the JSON."""
    if not raw:
        raise ValueError("empty extraction response")
    s = str(raw).strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        return json.loads(s[start:end + 1])
    raise ValueError(f"could not parse extraction JSON from: {s[:200]}")


def _clean_int(v) -> Optional[int]:
    if v is None or v is True or v is False:
        return None
    try:
        n = int(str(v).strip())
        return n if n >= 0 else None
    except Exception:
        return None


def _clean_str(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("null", "none", "n/a", "unknown", "not mentioned", ""):
        return None
    return s


def normalise_lead_quality(v) -> Optional[str]:
    s = str(v or "").strip().upper()
    return s if s in LEAD_QUALITIES else None


# ---------------------------------------------------------------------------
# Default LLM backend (injectable — tests pass their own generate_fn)
# ---------------------------------------------------------------------------
LEAD_EXTRACTION_MODEL = os.environ.get("LEAD_EXTRACTION_MODEL", "gemini-2.5-flash")
# Default 10s, not 20s: this runs during job shutdown, where the whole process has a
# bounded grace period. A timeout here is not a lost lead — build_lead_record catches
# it and falls back to the deterministic classifier — so failing fast beats hanging.
LEAD_EXTRACTION_TIMEOUT_S = float(os.environ.get("LEAD_EXTRACTION_TIMEOUT_S", "10"))
# Extraction is pure field-copying out of a transcript against a fixed schema; it does
# not benefit from a reasoning pass. gemini-2.5-flash thinks by default, and measured
# on a real 30-turn PUNO call that cost 20.5s versus 4.0s with thinking off, for the
# same extracted fields. That 20s was the whole problem: it blew every end-of-call
# budget the agent has.
LEAD_EXTRACTION_THINKING_BUDGET = int(os.environ.get("LEAD_EXTRACTION_THINKING_BUDGET", "0"))


def _gemini_generate(prompt: str) -> str:
    """Default extraction backend: a plain text generate_content call."""
    from google.genai import types as _t
    from genai_routing import _genai_client_for

    # Routed through the shared decision point so this cannot drift from the rest
    # of the project (see genai_routing.py). LEAD_EXTRACTION_MODEL is a plain text
    # model, so today this still honours GOOGLE_GENAI_USE_VERTEXAI as before.
    client = _genai_client_for(LEAD_EXTRACTION_MODEL)

    cfg_kwargs: dict[str, Any] = {
        "temperature": 0.1,
        "response_mime_type": "application/json",
        # LEAD_EXTRACTION_TIMEOUT_S was declared but never applied, so a stalled
        # request had no ceiling at all. google-genai takes http timeouts in ms.
        "http_options": _t.HttpOptions(timeout=int(LEAD_EXTRACTION_TIMEOUT_S * 1000)),
    }
    if LEAD_EXTRACTION_THINKING_BUDGET >= 0:
        try:
            cfg_kwargs["thinking_config"] = _t.ThinkingConfig(
                thinking_budget=LEAD_EXTRACTION_THINKING_BUDGET
            )
        except Exception:  # model/SDK without thinking support — plain call is fine
            pass

    resp = client.models.generate_content(
        model=LEAD_EXTRACTION_MODEL,
        contents=prompt,
        config=_t.GenerateContentConfig(**cfg_kwargs),
    )
    return getattr(resp, "text", "") or ""


# ---------------------------------------------------------------------------
# Orchestrator — ALWAYS returns a complete record
# ---------------------------------------------------------------------------
def build_lead_record(
    *,
    turns: list[dict],
    state: dict | None = None,
    caller_id: str = "",
    today: str = "",
    booking_id: Optional[str] = None,
    outcome: Optional[str] = None,
    transferred: bool = False,
    language: Optional[str] = None,
    call_id: Optional[str] = None,
    generate_fn: Optional[Callable[[str], str]] = None,
) -> dict:
    """Build the end-of-call lead record for EVERY call.

    Never raises and never returns None: if the LLM extraction fails we fall back to
    the fields the tools already captured plus a deterministic classification, so a
    lead is always written.
    """
    state = dict(state or {})
    transcript_text = render_transcript(turns)
    total_turns, caller_turns = count_turns(turns)

    known = {
        "name": state.get("name"),
        "phone_confirmed_by_agent": state.get("phone"),
        "package": state.get("package_interest"),
        "date": state.get("preferred_date"),
        "slot": state.get("preferred_slot"),
        "adults_7plus": state.get("adults_7plus"),
        "kids_under7": state.get("kids_under7"),
        "booking_id": booking_id,
    }

    llm: dict = {}
    source = "rule_based_fallback"
    if transcript_text.strip():
        try:
            fn = generate_fn or _gemini_generate
            llm = parse_extraction_json(fn(build_extraction_prompt(transcript_text, known, today)))
            source = "llm"
        except Exception as e:
            logger.error("[Lead] LLM extraction failed (%s) — using deterministic fallback.", e)
            llm = {}
    else:
        logger.warning("[Lead] No transcript captured — deterministic fallback only.")

    # --- merge: prefer tool-confirmed state, fall back to the extractor ------
    name = _clean_str(state.get("name")) or _clean_str(llm.get("name"))
    intent = _clean_str(llm.get("intent")) or _clean_str(state.get("intent"))
    package = _clean_str(state.get("package_interest")) or _clean_str(llm.get("package_interest"))
    date = _clean_str(state.get("preferred_date")) or _clean_str(llm.get("preferred_date"))
    slot = _clean_str(state.get("preferred_slot")) or _clean_str(llm.get("preferred_slot"))
    adults = _clean_int(state.get("adults_7plus"))
    if adults is None:
        adults = _clean_int(llm.get("adults_7plus"))
    kids = _clean_int(state.get("kids_under7"))
    if kids is None:
        kids = _clean_int(llm.get("kids_under7"))
    headcount = (adults or 0) + (kids or 0) or None

    phone, phone_source = resolve_phone(
        state_phone=state.get("phone"),
        llm_spoken_phone=llm.get("phone_spoken"),
        caller_id=caller_id,
    )

    # --- classification: LLM label, validated; rules decide otherwise -------
    quality = normalise_lead_quality(llm.get("lead_quality"))
    reasoning = _clean_str(llm.get("reasoning")) or ""
    rule_quality, rule_reason = classify_rule_based(
        booking_id=booking_id, name=name, phone=phone, preferred_date=date,
        package_interest=package, headcount_total=headcount, intent=intent,
        caller_turns=caller_turns, transcript_text=transcript_text, transferred=transferred,
    )
    if booking_id:
        # A confirmed booking is HOT by definition — never let the model downgrade it.
        quality, reasoning = "HOT", reasoning or rule_reason
        if not reasoning.lower().startswith("booking"):
            reasoning = f"Booking {booking_id} confirmed on the call. {reasoning}".strip()
    elif llm.get("wrong_number_or_spam") is True:
        quality, reasoning = "COLD", reasoning or "Wrong number or spam call."
    elif quality is None:
        quality, reasoning = rule_quality, rule_reason
    if not reasoning:
        reasoning = rule_reason

    if not outcome:
        outcome = ("Booked" if booking_id else
                   "Transferred to human" if transferred else
                   "Not interested" if quality == "COLD" else "Lead created")

    return {
        "type": "PUNO_LEAD",
        "call_id": call_id,
        "lead_quality": quality,
        "reasoning": reasoning,
        "name": name,
        "phone": phone,
        "phone_source": phone_source,
        "intent": intent,
        "package_interest": package,
        "preferred_date": date,
        "preferred_slot": slot,
        "adults_7plus": adults,
        "kids_under7": kids,
        "headcount_total": headcount,
        "notes": _clean_str(llm.get("notes")) or "",
        "outcome": outcome,
        "booking_id": booking_id,
        "follow_up_needed": quality in ("HOT", "WARM"),
        "language": language,
        "turn_count": total_turns,
        "caller_turn_count": caller_turns,
        "extraction_source": source,
        "payment_status": "n/a",
    }
