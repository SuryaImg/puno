import asyncio
import inspect
import json
import logging
import os
import time
import re
import datetime
import wave
from collections.abc import AsyncIterator
from typing import Optional, Annotated

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    function_tool,
)
from livekit.agents.llm import ChatMessage
from livekit.plugins import google
from livekit.plugins.google.realtime import RealtimeModel
from google.genai import types as genai_types
# NOTE: `from google.cloud import texttospeech` removed — the separate Cloud TTS
# engine it configured was the cause of the robotic voice (see AgentSession).
from tools import normalize_unicode_digits
from tools import (
    AppointmentState,
    CallTranscript,
    SkipQueue,
    normalize_hindi_phone_number,
    normalize_time_slot,
    detect_language,
    language_label,
    resolve_caller_language,
)
from storage import (
    get_appointment_state,
    save_appointment_state as _save_appointment_state_original,
    save_call_transcript,
    close_pool,
    _get_redis,
    # --- PUNO Jaipur (v2.3) persistence ---
    generate_booking_id,
    save_puno_booking,
    save_puno_lead,
    save_puno_call_log,
    save_dummy_sms,
)
import puno_data
from puno_data import get_package, package_choices_line
from call_guardrails import (
    mentions_payment_link,
    payment_correction_line,
    FixedLineDedup,
    should_fire_checkin,
    mentions_audibility_question,
    reengagement_line,
    stall_recovery_line,
    gender_drifted,
    hindi_self_gender,
    promises_booking_modification,
    booking_modification_correction_line,
    leaked_instruction,
    is_near_duplicate,
)

load_dotenv()
logger = logging.getLogger(__name__)


def _is_gemini31_model(name: str) -> bool:
    m = str(name or "").lower()
    return "gemini-3" in m or "3.1" in m


from genai_routing import (
    _model_uses_gemini_api,
    _genai_client_for,
    _is_permanent_model_error,
)


def _prefer_google_api_key() -> str:
    """Keep a single Gemini/Google key so google-genai does not warn and pick one.

    When both GOOGLE_API_KEY and GEMINI_API_KEY are set the SDK logs
    "Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY."
    This process used to copy the same value into both names, which always
    triggered that warning. Prefer GOOGLE_API_KEY and drop GEMINI_API_KEY.
    """
    google_key = (os.environ.get("GOOGLE_API_KEY") or "").strip()
    gemini_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    chosen = google_key or gemini_key
    if chosen:
        os.environ["GOOGLE_API_KEY"] = chosen
    if "GEMINI_API_KEY" in os.environ:
        os.environ.pop("GEMINI_API_KEY", None)
        if chosen:
            logger.info(
                "Using GOOGLE_API_KEY only (unset GEMINI_API_KEY to avoid dual-key warning)"
            )
    return chosen


_prefer_google_api_key()

# Primary Live model is GEMINI_MODEL; if session.start() fails we retry with this.
_GEMINI_FALLBACK_MODEL = "gemini-live-2.5-flash-native-audio"
# Strong refs for fire-and-forget tool work (SMS, instruction refresh) so the
# event loop cannot GC the task mid-flight. Also copied onto the agent._bg_tasks
# set when an agent is available, so disconnect cancels them.
_TOOL_BG_TASKS: set = set()

# --- Single source of truth for "what time is it right now" -----------------
# Arora Hospital operates on India Standard Time regardless of what timezone
# the host server happens to be running in (a common source of bugs: a VM
# running in UTC or another region silently shifts every "is this slot in the
# past", "is same-day", and "is OPD still open" check by hours). IST has no
# daylight-saving changes, so a fixed +5:30 offset is used rather than relying
# on the OS/IANA timezone database being installed and correct.
_IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
# OPD is 9:00 AM–7:00 PM IST inclusive. Used by datetime validation so a
# caller who says "8 baje" / "raat ke 12" cannot be booked outside hours.
_OPD_OPEN_MINUTES = 9 * 60
_OPD_CLOSE_MINUTES = 19 * 60


def _now_ist() -> datetime.datetime:
    """Current wall-clock moment in India Standard Time, as a naive datetime
    (tzinfo stripped) so it drops in everywhere existing naive-datetime
    arithmetic/comparisons already expect. Always use this instead of
    datetime.datetime.now() for anything involving the current real-world
    time — never trust the server host's local timezone."""
    return datetime.datetime.now(datetime.timezone.utc).astimezone(_IST).replace(tzinfo=None)


def _today_ist() -> datetime.date:
    """Current calendar date in India Standard Time. Use instead of
    datetime.date.today()."""
    return _now_ist().date()


# Spoken/number-word ages up to the only boundary that matters for eligibility
# (10 vs 11). Kept small on purpose — we only need to tell "10 and under" from
# "11 and over", not parse arbitrary ages.
_AGE_WORD_MAP = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    # Common Hindi/Hinglish number words
    "ek": 1, "do": 2, "teen": 3, "char": 4, "char": 4, "paanch": 5, "panch": 5,
    "chhe": 6, "che": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10, "dus": 10,
    "gyarah": 11, "gyaarah": 11, "baarah": 12, "barah": 12,
}

# Words that indicate the age was given in MONTHS (or younger) — always an
# eligible infant. Matched on the normalized age string.
_MONTH_WORDS = ("month", "months", "mahine", "mahina", "maah", "mah", "महीने", "महीना",
                "week", "weeks", "hafte", "hafta", "din", "days", "day",
                "newborn", "new born", "neonate", "infant")


def _extract_age_years(age_raw) -> Optional[float]:
    """Best-effort parse of a spoken/typed child age into YEARS.

    Returns a float number of years, or None if it genuinely can't be
    determined. Deliberately conservative and small in scope: its only job is
    to let the eligibility gate distinguish "10 and under" (eligible) from
    "11 and over" (not). Anything expressed in months/weeks/days is treated as
    well under a year (returns 0.5) since this is a children's hospital and such
    infants are always age-eligible.

    NOTE: this is used ONLY for the non-emergency booking age gate. It never
    blocks emergency handling.
    """
    if age_raw is None:
        return None
    from tools import normalize_unicode_digits
    s = normalize_unicode_digits(str(age_raw)).strip().lower()
    if not s:
        return None

    # Expressed in months/weeks/days → infant, always eligible.
    if any(w in s for w in _MONTH_WORDS):
        return 0.5

    # First numeric token (handles "5", "5 years", "5.5", "5 saal", "5yr").
    m = re.search(r"\d+(?:\.\d+)?", s)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass

    # Fall back to number-words ("five", "paanch", "eleven", "gyarah").
    for tok in re.split(r"[^a-z\u0900-\u097f]+", s):
        if tok in _AGE_WORD_MAP:
            return float(_AGE_WORD_MAP[tok])

    return None

_HOSPITAL_CONTACT_NUMBERS = frozenset({
    "9549101235",  # Emergency
    "7891452498",  # Appointments / insurance
    "9799346653",  # Appointments / insurance
})

_SAME_NUMBER_RE = re.compile(
    r"(?:"
    r"yah[ií]\s*(?:w[aá]l[aá]\s*)?(?:number|no\.?|mobile)|"
    r"is[ií]\s*(?:w[aá]l[aá]\s*)?(?:number|no\.?)|"
    r"us[ií]\s*(?:number|no\.?)|"
    r"same\s+(?:number|mobile)|"
    r"this\s+number|"
    r"(?:calling|caller)(?:'s)?\s+number|"
    r"jis(?:se)?\s+(?:maine\s+)?(?:call|phone)|"
    r"number\s+yah[ií]|"
    r"यही\s*(?:वाला\s*)?(?:नंबर|नम्बर)|"
    r"इसी\s*(?:नंबर|नम्बर)|"
    r"उसी\s*(?:नंबर|नम्बर)|"
    r"जिस(?:से)?\s+(?:मैंने\s+)?(?:कॉल|call)|"
    r"कॉल\s+किया\s*(?:है|हूं|हूँ)?"
    r")",
    re.I,
)

_NOT_EMERGENCY_RE = re.compile(
    r"(?:"
    r"not\s+an?\s+emergency|"
    r"no\s+emergency|"
    r"emergency\s*(?:nahi+|nahin|mat)|"
    r"normal\s+(?:dikha|visit|check|appointment)|"
    r"routine\s+(?:visit|check)|"
    r"इमरजेंसी\s*(?:में\s*)?(?:दिखाने\s*की\s*)?(?:जरूरत\s*)?नहीं|"
    r"एमरजेंसी\s*(?:में\s*)?नहीं|"
    r"नॉर्मल\s*दिखा|"
    r"इतना\s*भी\s*तेज\s*नहीं"
    r")",
    re.I,
)

_STRONG_EMERGENCY_RE = re.compile(
    r"(?:"
    r"unconscious|unresponsive|not breathing|gasping|blue lips|"
    r"choking|seizure|convulsion|\bfits\b|severe bleeding|"
    r"poison(?:ing|ed)?|emergency hai|emergency\s+hai|"
    r"बेहोश|बेहोशी|सांस\s*नहीं|दम\s*घु|"
    r"दौरा|मिर्गी|इमरजेंसी\s*है|एमरजेंसी\s*है"
    r")",
    re.I,
)

_ID_ASK_RE = re.compile(
    r"(?:"
    r"appointment\s*id|apt(?:ointment)?\s*id|booking\s*id|"
    r"\btoken\b|reference\s*(?:id|number)|"
    r"अपॉइंटमेंट\s*आईडी|आईडी\s*बता|id\s*bata"
    r")",
    re.I,
)

_STRONG_CONFIRM_RE = re.compile(
    r"(?:"
    r"confirm|कंफर्म|कन्फर्म|"
    r"book\s*(?:kar|kr|it)|बुक\s*कर|"
    r"कर\s*द(?:ो|ीज(?:िए|ए)|ें)|"
    r"sab\s*(?:kuch\s*)?sahi|सब\s*कुछ\s*सही"
    r")",
    re.I,
)


def _ten_digit_mobile(raw: str) -> str:
    """Return a valid 10-digit Indian mobile, or ''."""
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) > 10 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if len(digits) == 10 and digits[0] in "6789":
        return digits
    return ""


def _is_hospital_contact_number(phone: str) -> bool:
    d = _ten_digit_mobile(phone) or re.sub(r"\D", "", phone or "")
    if len(d) > 10 and d.startswith("91"):
        d = d[2:]
    return d in _HOSPITAL_CONTACT_NUMBERS


def _utterance_requests_caller_phone(text: str) -> bool:
    return bool(_SAME_NUMBER_RE.search(text or ""))


def _sip_caller_mobile(userdata: dict | None) -> str:
    if not isinstance(userdata, dict):
        return ""
    return _ten_digit_mobile(userdata.get("caller_phone") or "")


def _apply_sip_caller_phone(state, userdata: dict | None) -> str:
    """Store the SIP CLI as the booking number. Returns the 10-digit mobile or ''."""
    mobile = _sip_caller_mobile(userdata)
    if not mobile or _is_hospital_contact_number(mobile):
        return ""
    state.patient_phone = mobile
    state.phone_number = mobile
    state.patient_phone_confirmed = False
    if isinstance(getattr(state, "step_attempts", None), dict):
        state.step_attempts["patient_phone"] = 0
    return mobile



_DATA_CACHE = {"value": None, "updated_at": 0.0, "lock": None}
_CACHE_TTL = 120.0

# --- PUNO Jaipur persona/prompt (v2.3) ------------------------------------
# The full system prompt now lives in puno_prompt.py (persona, call flow,
# guardrails, system actions) with the verified knowledge base baked in from
# puno_data.py. build_agent_instructions() below fills [Stored Fields] and
# [Missing Fields] each turn. Kept under the original name so every downstream
# caller (build_agent_instructions / update / refresh) stays unchanged.
from puno_prompt import PUNO_VOICE_INSTRUCTIONS as _human_voice_instructions

def _language_switch_supplement(from_code: str | None, to_code: str | None) -> str:
    """Extra directive appended when the caller just switched language."""
    if not from_code or not to_code or from_code == to_code:
        return ""
    to_label = language_label(to_code)
    from_label = language_label(from_code)
    lines = [
        f"\nCRITICAL LANGUAGE SWITCH: The caller just moved from {from_label} to {to_label}.",
        f"Your NEXT reply MUST be entirely in {to_label}.",
        "Ignore the language of your previous replies and the Hindi greeting.",
    ]
    if to_code == "en":
        lines.append(
            "Do NOT use any Hindi words — no 'ji', 'aap', 'bata dijiye', 'baje', "
            "'aaj', 'theek hai', 'haan', or Hindi grammar."
        )
    elif to_code == "hi":
        lines.append("Reply in natural spoken Hindi — do not reply in English.")
    elif to_code == "hinglish":
        lines.append(
            "Reply in the same natural Hindi-English mix the caller is using — "
            "not pure Hindi and not pure English unless they just switched to one."
        )
    return " ".join(lines) + "\n"


def _caller_language_reply_hint(state: AppointmentState) -> str:
    """Short suffix for tool results so the model renders in the caller's language."""
    code = getattr(state, "caller_language", None)
    if not code:
        return " Speak in the caller's own language."
    return f" Speak in {language_label(code)} — in your own natural words, not a fixed script."


def build_agent_instructions(
    state: AppointmentState,
    *,
    language_switch_from: str | None = None,
    lang_reinforce_seq: int = 0,
) -> str:
    """Compose the full PUNO Jaipur system instructions for `state`.

    The persona, call flow, guardrails and baked-in knowledge base live in
    puno_prompt.PUNO_VOICE_INSTRUCTIONS (imported above as _human_voice_instructions).
    This builder only fills the two live-state placeholders — [Stored Fields] and
    [Missing Fields] — from the five PUNO booking fields (name, phone, headcount,
    date/slot, package), then appends the ACTIVE CONVERSATION LANGUAGE directive.
    Signature is unchanged so update_agent_instructions / refresh callers keep working.
    """
    base_instr = _human_voice_instructions

    # --- LANGUAGE LOCK (behaviour unchanged from the hospital build) --------
    _lang_code = getattr(state, "caller_language", None)
    _switch_from = language_switch_from
    if _lang_code:
        _lang_directive = (
            f"\n\n=== ACTIVE CONVERSATION LANGUAGE: {language_label(_lang_code)} ===\n"
            f"The caller is speaking {language_label(_lang_code)}. Every reply you give from now on "
            f"MUST be in {language_label(_lang_code)}, including anything a tool result asks you to "
            f"communicate. This field is recomputed from the caller's own speech on every single turn, "
            f"so it already reflects any switch they have just made — trust it over anything earlier in "
            f"the conversation. If the caller speaks a different language on their next turn, this will "
            f"change automatically and you follow it immediately, without them having to ask.\n"
            + _language_switch_supplement(_switch_from, _lang_code)
        )
        if lang_reinforce_seq > 0:
            _lang_directive += (
                f"\nREMINDER (caller turn #{lang_reinforce_seq}): Reply in "
                f"{language_label(_lang_code)} on THIS turn. Do not slip back into another "
                f"language from your greeting or a previous reply if the caller is in "
                f"{language_label(_lang_code)}.\n"
            )
    else:
        _lang_directive = (
            "\n\n=== ACTIVE CONVERSATION LANGUAGE: FOLLOW THE CALLER'S AUDIO ===\n"
            "No text-based language signal is available for this turn. Do NOT treat that as a "
            "reason to fall back to Hindi or to your greeting's language. Listen to the caller "
            "and reply in whatever language they are actually speaking — Hindi to Hindi, English "
            "to English, Hinglish to Hinglish — switching the moment they switch, without being "
            "asked.\n"
        )

    # --- PUNO live state: the five Step-4 booking fields -------------------
    stored: list[str] = []
    missing: list[str] = []

    booking_intent = getattr(state, "booking_intent_detected", False)

    # 1. Full name (reuses patient_name / caller_name)
    name = getattr(state, "patient_name", None) or getattr(state, "caller_name", None)
    if name:
        stored.append(f"- Name: {name} (ALREADY CAPTURED - DO NOT ASK AGAIN)")
    elif booking_intent:
        missing.append("- Full name")

    # 2. Phone number (reuses patient_phone)
    phone = getattr(state, "patient_phone", None)
    if phone:
        spaced = ", ".join(list(str(phone)))
        confirmed = getattr(state, "patient_phone_confirmed", False)
        stored.append(
            f"- Phone Number: {spaced} (confirmed={confirmed}) (ALREADY CAPTURED - DO NOT ASK AGAIN)"
        )
    elif booking_intent:
        missing.append("- Phone Number (read it back digit by digit to confirm; the confirmation SMS goes here)")

    # 3. Headcount split under-7 / 7+
    kids = getattr(state, "kids_under7", None)
    adults = getattr(state, "adults_7plus", None)
    if kids is not None or adults is not None:
        stored.append(
            f"- Headcount: {adults or 0} aged 7+ and {kids or 0} under 7 (ALREADY CAPTURED - DO NOT ASK AGAIN)"
        )
    elif booking_intent:
        missing.append("- Headcount, split as under 7 and 7-and-over (drives package eligibility)")

    # 4. Date + time slot (reuses appointment_date / appointment_time)
    visit_date = getattr(state, "appointment_date", None)
    slot = getattr(state, "appointment_time", None)
    if visit_date:
        stored.append(f"- Visit Date: {visit_date} (ALREADY CAPTURED - DO NOT ASK AGAIN)")
    elif booking_intent:
        missing.append("- Preferred date")
    if slot:
        stored.append(f"- Time Slot: {slot} (ALREADY CAPTURED - DO NOT ASK AGAIN)")
    elif booking_intent:
        missing.append("- Preferred time slot")

    # 5. Package / activities of interest
    pkg = getattr(state, "package_name", None) or getattr(state, "package_id", None)
    if pkg:
        stored.append(f"- Package / Activities: {pkg} (ALREADY CAPTURED - DO NOT ASK AGAIN)")
    elif booking_intent:
        missing.append("- Package or activities of interest")

    # booking_id, once a booking has succeeded on this call
    bid = getattr(state, "booking_id", None)
    if bid:
        stored.append(f"- Booking ID: {bid} (BOOKING CONFIRMED - the confirmation SMS has fired)")

    stored_str = "\n".join(stored) if stored else "- (nothing captured yet)"
    missing_str = "\n".join(missing) if missing else "- (nothing outstanding)"

    updated_instr = base_instr.replace("[Stored Fields]", stored_str).replace("[Missing Fields]", missing_str)
    updated_instr += _lang_directive
    return updated_instr


def build_turn_snapshot(state: AppointmentState) -> str:
    """Compact 'what I already know' line injected into the live chat context.

    THE RE-ASKING BUG. Captured details only reached the model through
    build_agent_instructions(), and that rebuild is fire-and-forget with a 0.5s
    delay plus a Gemini websocket restart. The reply for the turn in which a
    detail was captured is therefore generated from instructions that do NOT
    contain it yet — so the agent asked for the name, the phone number or the
    age a second time, immediately after being told. Injecting this snapshot in
    on_user_turn_completed() closes that window: it lands in the context the
    model is about to read, with no restart and no added latency.
    """
    known = []
    _name = getattr(state, "patient_name", None) or getattr(state, "caller_name", None)
    if _name:
        known.append(f"name = {_name}")
    if getattr(state, "patient_phone", None):
        known.append(f"phone = {state.patient_phone}")
    _adults = getattr(state, "adults_7plus", None)
    _kids = getattr(state, "kids_under7", None)
    if _adults is not None or _kids is not None:
        known.append(f"headcount = {_adults or 0} aged 7+, {_kids or 0} under 7")
    if getattr(state, "package_name", None):
        known.append(f"package = {state.package_name}")
    elif getattr(state, "reason", None):
        known.append(f"activities of interest = {state.reason}")
    if getattr(state, "appointment_date", None):
        known.append(f"date = {state.appointment_date}")
    if getattr(state, "appointment_time", None):
        known.append(f"time slot = {state.appointment_time}")
    if getattr(state, "booking_id", None):
        known.append(f"booking CONFIRMED, id = {state.booking_id}")

    parts = []
    if known:
        parts.append(
            "[LIVE BOOKING STATE — retain all details, follow this state, never re-ask for something listed here] "
            + "; ".join(known) + "."
        )

    # Fields already asked about at least once. Asking a third time is what
    # makes a call feel like an interrogation, so name them explicitly.
    asked = [
        f for f, c in (getattr(state, "ask_counts", None) or {}).items()
        if c >= 2 and not getattr(state, f, None)
    ]
    if asked:
        parts.append(
            "[ALREADY ASKED TWICE, still unanswered] " + ", ".join(asked) +
            ". Do not ask again in the same words — either move on to the next "
            "needed detail and come back to it once at the end, or ask it once "
            "in clearly different, shorter words."
        )

    return " ".join(parts)


async def update_agent_instructions(
    agent,
    state: AppointmentState,
    *,
    force: bool = False,
    language_switch_from: str | None = None,
    lang_reinforce_seq: int = 0,
) -> bool:
    """Apply freshly-built instructions to `agent`, skipping no-op updates.

    Returns True when instructions were pushed to the live session.

    Routine mid-session updates are skipped on Gemini 3.1 (mutable_chat_context
    is false and restarts are risky). Language switches use force=True so the
    ACTIVE CONVERSATION LANGUAGE block reaches the model before it replies.
    """
    if not agent:
        return False

    model_name = getattr(agent, "_model_name", "") or getattr(agent, "model", "")
    if hasattr(agent, "_opts") and hasattr(agent._opts, "model"):
        model_name = agent._opts.model
    mn = str(model_name).lower()
    if "gemini-3" in mn or "3.1" in mn:
        # Gemini 3.1: update_instructions() tears down the realtime websocket via
        # _mark_restart_needed(), which aborts the reply activity_end just asked
        # the model to generate. Track language/state locally instead.
        logger.debug(
            "Skipping update_instructions on Gemini 3.1 (would abort live generation)."
        )
        return False
    if not force and "gemini-3" in mn:
        logger.debug("Skipping mid-session update_instructions for Gemini 3.1 model.")
        return False

    updated_instr = build_agent_instructions(
        state,
        language_switch_from=language_switch_from,
        lang_reinforce_seq=lang_reinforce_seq,
    )

    _prev = getattr(agent, "_last_applied_instructions", None)
    if _prev == updated_instr:
        return False

    try:
        await agent.update_instructions(updated_instr)
        agent._last_applied_instructions = updated_instr
        if force and language_switch_from:
            logger.info(
                "Language switch instructions applied: %s -> %s",
                language_switch_from or "unset",
                getattr(state, "caller_language", None),
            )
        elif force and lang_reinforce_seq:
            logger.info(
                "Language reinforce #%d applied (%s)",
                lang_reinforce_seq,
                getattr(state, "caller_language", None),
            )
        else:
            logger.info("Dynamically updated agent instructions with current state.")
        return True
    except Exception as e:
        logger.warning("update_instructions failed: %s", e)
        return False

def get_booking_state_label(state: AppointmentState) -> str:
    # PUNO booking-flow labels (internal state/logging only — never spoken).
    if state.booking_status == "BOOKED" or state.booking_status == "confirmed":
        return "Booking confirmed"
    if state.booking_status == "READY_FOR_CONFIRMATION" or state.booking_status == "ready_for_confirmation":
        return "Awaiting caller confirmation"
    _name = state.patient_name or state.caller_name
    if not _name or not state.patient_phone:
        return "Collecting contact details"
    if (state.adults_7plus is None) and (state.kids_under7 is None):
        return "Collecting headcount"
    if not state.package_id:
        return "Selecting package"
    if not state.appointment_date:
        return "Selecting date"
    if not state.appointment_time:
        return "Selecting time slot"
    return "Ready to confirm"

async def save_appointment_state(user_id: str, state: AppointmentState) -> None:
    """
    Persist appointment state to Redis.

    NOTE (BUG FIX): This function used to reflectively walk the call stack and
    call `agent.update_instructions(...)` (a FULL system-instructions replacement)
    every single time state was saved. `save_appointment_state` is invoked many
    times per turn (inside update_appointment_details, verify_availability, and
    confirm_appointment — including inside internal retry/alternate-doctor loops).
    On a realtime/native-audio model, replacing the system instructions mid-call
    can force the underlying model session to drop or reinterpret its running
    conversational context, which is the most likely cause of the agent
    re-greeting the caller and "forgetting" that a booking summary was already
    given mid-call (see transcript analysis). Instruction refresh is now done
    explicitly and only ONCE per tool call, from within the PUNO tools themselves,
    instead of on every internal state save.
    """
    state.booking_state = get_booking_state_label(state)
    await _save_appointment_state_original(user_id, state)


def _agent_from_ctx(ctx) -> Optional[object]:
    """Best-effort current PunoVoiceAgent from a tool RunContext."""
    try:
        if hasattr(ctx, "session") and ctx.session is not None:
            agent = getattr(ctx.session, "current_agent", None)
            if agent is not None:
                return agent
    except Exception:
        pass
    try:
        userdata = getattr(ctx, "userdata", None)
        if isinstance(userdata, dict):
            session = userdata.get("session")
            if session is not None:
                return getattr(session, "current_agent", None)
    except Exception:
        pass
    return None


def _spawn_agent_bg(ctx, coro) -> Optional[asyncio.Task]:
    """Run a coroutine in the background without blocking the current tool turn.

    Tracked on the agent (and a module-level set) so disconnect can cancel it
    and so asyncio cannot garbage-collect a still-running task.
    """
    try:
        task = asyncio.create_task(coro)
    except Exception as e:
        logger.warning("Could not spawn background tool task: %s", e)
        return None
    _TOOL_BG_TASKS.add(task)
    task.add_done_callback(_TOOL_BG_TASKS.discard)

    def _log_bg_exc(t: asyncio.Task) -> None:
        if t.cancelled():
            return
        try:
            exc = t.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            logger.warning("Background tool task failed: %s", exc)

    task.add_done_callback(_log_bg_exc)
    agent = _agent_from_ctx(ctx)
    if agent is not None:
        bg = getattr(agent, "_bg_tasks", None)
        if isinstance(bg, set):
            bg.add(task)
            task.add_done_callback(bg.discard)
    return task


async def refresh_agent_instructions(ctx: RunContext, state: AppointmentState) -> None:
    """
    Refresh system instructions with the latest state WITHOUT blocking the turn.

    `update_instructions()` can restart the Gemini realtime websocket. Awaiting
    it inside a tool used to stall speech until that restart finished. The
    0.5s delay lets the current reply queue first (same rationale as
    PunoVoiceAgent._background_update_instructions).
    """
    async def _run() -> None:
        try:
            await asyncio.sleep(0.5)
            agent = _agent_from_ctx(ctx)
            if agent:
                await update_agent_instructions(agent, state)
            else:
                logger.error("Failed to refresh agent instructions: Could not find agent in ctx.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Failed to refresh agent instructions: %s", exc)

    _spawn_agent_bg(ctx, _run())

def _get_transcript(ud: dict) -> CallTranscript:
    if "transcript" not in ud:
        ud["transcript"] = CallTranscript(user_id=ud.get("user_id", "unknown"))
    return ud["transcript"]


_ADULT_ONLY_CONDITIONS = [
    (r"\b(labou?r pain|labou?r|in labou?r|contractions|water broke|"
     r"prasav|prasuti|delivery pain|childbirth|giving birth)\b", 12, "labour/childbirth"),
    (r"\b(pregnan\w*|pregnancy|garbh\w*|expecting a baby|conceiv\w*|"
     r"trimester|antenatal|prenatal|miscarriage|abortion)\b", 10, "pregnancy"),
    (r"\b(menopause|perimenopause|rajonivritti)\b", 35, "menopause"),
    (r"\b(period pain|menstrual\w*|menstruation|periods|mahwari|"
     r"masik dharm)\b", 8, "menstruation"),
    (r"\b(prostate|prostat\w*|erectile\w*|impoten\w*|vasectomy)\b", 18, "prostate/adult urology"),
    (r"\b(infertility|ivf|sperm count|tubectomy|hysterectomy)\b", 15, "fertility"),
    (r"\b(dementia|alzheimer\w*|parkinson\w*|cataract surgery|"
     r"osteoporosis|arthritis of old age)\b", 40, "geriatric condition"),
]

# Devanagari equivalents, matched by substring (word boundaries are unreliable
# against combining matras — the same problem documented in tools.py).
_ADULT_ONLY_DEVANAGARI = [
    (("प्रसव", "प्रसव पीड़ा", "जचगी", "जच्चगी", "बच्चा होने"), 12, "labour/childbirth"),
    (("गर्भवती", "गर्भावस्था", "प्रेगनेंट", "प्रेग्नेंट"), 10, "pregnancy"),
    (("रजोनिवृत्ति",), 35, "menopause"),
    (("माहवारी", "मासिक धर्म", "पीरियड्स"), 8, "menstruation"),
    (("प्रोस्टेट",), 18, "prostate/adult urology"),
]


def validate_and_normalize_appointment_datetime(date_str: str, time_str: str) -> tuple[str, str, Optional[str]]:
    date_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", date_str.strip())
    if not date_match:
        return date_str, time_str, None
        
    year = int(date_match.group(1))
    month = date_match.group(2)
    day = date_match.group(3)
    
    time_clean = time_str.strip().upper()
    time_match = re.match(r"^(\d{1,2}):(\d{2})\s*(AM|PM)$", time_clean)
    if not time_match:
        time_match_simple = re.match(r"^(\d{1,2})\s*(AM|PM)$", time_clean)
        if time_match_simple:
            hour = int(time_match_simple.group(1))
            minute = 0
            am_pm = time_match_simple.group(2)
        else:
            time_match_24 = re.match(r"^(\d{1,2}):(\d{2})$", time_clean)
            if time_match_24:
                hour = int(time_match_24.group(1))
                minute = int(time_match_24.group(2))
                am_pm = None
            else:
                return date_str, time_str, ("INVALID_TIME: could not parse a specific clock time. "
                    "Ask the caller for a specific time (e.g. 10:00 AM) in their own language.")
    else:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        am_pm = time_match.group(3)
        
    if am_pm == "PM" and hour < 12:
        hour += 12
    elif am_pm == "AM" and hour == 12:
        hour = 0
        
    try:
        appt_dt = datetime.datetime(year, int(month), int(day), hour, minute)
    except ValueError:
        return date_str, time_str, ("INVALID_DATETIME: the date or time is not valid. "
            "Ask the caller to restate the date and time in their own language.")
        
    now = _now_ist()
    diff = (appt_dt - now).total_seconds()

    # POLICY: a slot must not be in the PAST, and it must fall inside OPD
    # hours (9:00 AM–7:00 PM). A future time later today is bookable. The
    # previous 30-minute-buffer rule contradicted the instructions and,
    # combined with verify_availability's finders (which only skip
    # strictly-past slots), caused the agent to PROPOSE a same-day time and
    # then REJECT it at confirm — an endless "koi aur time?" loop. A
    # 60-second grace absorbs clock rounding so "abhi ka" (right now) still
    # books.
    slot_minutes = hour * 60 + minute
    if slot_minutes < _OPD_OPEN_MINUTES or slot_minutes > _OPD_CLOSE_MINUTES:
        return date_str, time_str, (
            "OUTSIDE_OPD_HOURS: OPD runs 9:00 AM to 7:00 PM. "
            "Do NOT store or book this time. Ask the caller — in their own language — "
            "to pick a time in that window. If they asked for after 7:00 PM, offer "
            "7:00 PM today if that slot is still upcoming, otherwise tomorrow morning."
        )
    if diff < -60:
        return date_str, time_str, ("PAST_TIME: the requested time is in the past. "
            "Ask the caller — in their own language — to choose a time from now onwards.")

    normalized_time = appt_dt.strftime("%I:%M %p").lstrip('0')
    normalized_date = appt_dt.strftime("%Y-%m-%d")
    return normalized_date, normalized_time, None


_RETRY_TASKS: set = set()


async def _iter_pcm_frames(
    pcm: bytes,
    *,
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2,
    frame_ms: int = 20,
) -> AsyncIterator[rtc.AudioFrame]:
    bytes_per_sample = sample_width * channels
    samples_per_frame = max(1, int(sample_rate * frame_ms / 1000))
    frame_bytes = samples_per_frame * bytes_per_sample
    offset = 0
    while offset < len(pcm):
        chunk = pcm[offset : offset + frame_bytes]
        offset += frame_bytes
        n = len(chunk) // bytes_per_sample
        if n <= 0:
            break
        yield rtc.AudioFrame(
            data=chunk,
            sample_rate=sample_rate,
            num_channels=channels,
            samples_per_channel=n,
        )
        await asyncio.sleep(0)


# Fixed-line PCM cache: (text, voice, live_model) -> pcm bytes. Fixed lines come
# from small closed sets (re-engagement lines, fillers, goodbye, payment
# correction), so caching removes both the repeat synthesis latency and any
# chance of the same line being rendered by two different engines on one call.
_FIXED_LINE_PCM_CACHE: dict[tuple, bytes] = {}
_FIXED_LINE_LIVE_TIMEOUT_S = float(os.environ.get("FIXED_LINE_LIVE_TIMEOUT_S", "12.0"))


def _configured_voice() -> str:
    """The one place the TTS voice name is resolved. Every synthesis path uses this."""
    return (os.environ.get("GEMINI_VOICE") or "Vindemiatrix").strip() or "Vindemiatrix"


def _synthesize_fixed_line_pcm(text: str, voice: str) -> bytes:
    """Render a fixed line in the SAME voice AND SAME engine as the conversation.

    ISSUE 1 ROOT CAUSE (voice/persona mismatch): this used to call batch TTS
    (gemini-2.5-*-tts) directly, while the conversation itself is voiced by the
    gemini-3.1 native-audio LIVE model. Both were handed the same voice NAME
    (Vindemiatrix), so the config looked correct — but a batch TTS render and a
    live native-audio render of the same voice do not sound like the same person.
    Every check-in, filler, goodbye and correction was therefore spoken in a
    noticeably different voice from the rest of the call.

    Now the live model is tried FIRST (same engine as the call), exactly like the
    greeting path, and batch TTS is only a logged fallback. Results are cached per
    (text, voice, live-model).
    """
    voice = (voice or "").strip() or _configured_voice()
    text = (text or "").strip()
    if not text:
        raise RuntimeError("empty fixed line")

    live_model = _live_greeting_model()
    key = (text, voice, live_model)
    cached = _FIXED_LINE_PCM_CACHE.get(key)
    if cached:
        return cached

    # Preferred: the same Live engine that voices the conversation.
    try:
        async def _live():
            return await asyncio.wait_for(
                _synthesize_greeting_pcm_live(text, voice, live_model),
                timeout=_FIXED_LINE_LIVE_TIMEOUT_S,
            )

        pcm = _run_async(_live)
        if pcm:
            _FIXED_LINE_PCM_CACHE[key] = pcm
            logger.info(
                "[Issue1] Fixed line rendered by LIVE model %s (voice=%s): %.40s",
                live_model, voice, text,
            )
            return pcm
    except Exception as e:
        logger.warning(
            "[Issue1] Live-model fixed-line synthesis failed (%s) — falling back to "
            "batch TTS; this line may not match the conversation voice: %.40s",
            e, text,
        )

    pcm = _synthesize_fixed_line_pcm_tts(text, voice)
    if pcm:
        _FIXED_LINE_PCM_CACHE[key] = pcm
    return pcm


def _synthesize_fixed_line_pcm_tts(text: str, voice: str) -> bytes:
    """LAST-RESORT batch TTS for short fixed lines.

    NOTE: this renders through client.models.generate_content() against
    _GREETING_TTS_MODELS, which is a DIFFERENT engine from the live native-audio
    model voicing the call. It shares the voice *name* but not the timbre — see
    the note at _resolve_greeting_cache_wav. Only used when the live-model
    synthesis in _synthesize_fixed_line_pcm() fails.
    """
    voice = (voice or "").strip() or _configured_voice()
    text = (text or "").strip()
    if not text:
        raise RuntimeError("empty fixed line")
    use_v = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() == "true"
    gcp_cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if gcp_cred and os.path.exists(gcp_cred):
        try:
            with open(gcp_cred, "r", encoding="utf-8") as _f:
                _p = json.load(_f).get("project_id")
                if _p:
                    os.environ["GOOGLE_CLOUD_PROJECT"] = _p
        except Exception:
            pass
    # Every entry in _GREETING_TTS_MODELS is a TTS model, so they all route the
    # same way; the Vertex-vs-Gemini-API decision lives in genai_routing.py.
    client = _genai_client_for(_GREETING_TTS_MODELS[0])
    prompt = (
        "Say exactly these words in a warm natural conversational voice:\n"
        f"{text}"
    )
    last_err: Optional[BaseException] = None
    for tts_model in _GREETING_TTS_MODELS:
        try:
            response = client.models.generate_content(
                model=tts_model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=genai_types.SpeechConfig(
                        voice_config=genai_types.VoiceConfig(
                            prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                                voice_name=voice,
                            )
                        )
                    ),
                ),
            )
            pcm = _extract_tts_pcm(response)
            if pcm:
                return pcm
            last_err = RuntimeError(f"{tts_model} returned no audio")
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Fixed line TTS failed: {last_err}")


async def _speak_fixed_line_audio(
    session,
    text: str,
    allow_interruptions: bool = True,
    add_to_chat_ctx: bool = False,
) -> bool:
    voice = _configured_voice()
    try:
        pcm = await asyncio.to_thread(_synthesize_fixed_line_pcm, text, voice)
    except Exception as e:
        logger.warning("Fixed line audio synthesis failed: %s", e)
        return False

    last_err: Optional[BaseException] = None
    for attempt in range(1, 51):
        try:
            handle = session.say(
                text,
                audio=_iter_pcm_frames(pcm),
                allow_interruptions=allow_interruptions,
                add_to_chat_ctx=add_to_chat_ctx,
            )
            if handle is not None:
                return True
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.02)
    return False


def speak_fixed_line(
    session,
    text: str,
    allow_interruptions: bool = True,
    add_to_chat_ctx: bool = False,
    *,
    retries: int = 50,
    retry_delay: float = 0.02,
) -> bool:
    """Speak a fixed line through the NATIVE-AUDIO model.

    ROBOTIC-VOICE FIX: session.say() renders text via a separate TTS plugin.
    This agent deliberately has no `tts=` on its AgentSession (the native-audio
    realtime model produces speech itself). Verified against the installed
    plugin: the Gemini RealtimeModel reports supports_say=False, so with no TTS
    attached say() RAISES RuntimeError. generate_reply() is therefore the only
    delivery path — say() is kept solely as a fallback for the day a TTS is
    reintroduced, and must never be relied on.

    GREETING-LATENCY: the instruction is deliberately terse. Everything in it
    is prompt the model reads before emitting its first audio frame, so padding
    it directly delays the first word the caller hears.

    SILENT-FAILURE FIX — this is why the greeting sometimes never played:
    the underlying Gemini session pushes events onto a channel with
    send_nowait() and *suppresses* ChanClosed. If the realtime websocket has
    not finished connecting at the moment we ask for the greeting, the request
    is dropped on the floor with no exception and no log — the agent simply
    sits mute until the caller speaks (which does go through a different code
    path, hence "it only talks after I say hello"). session.start() returning
    does NOT guarantee the realtime session is live.

    So: verify a SpeechHandle actually came back, and retry briefly if not.
    The retries are cheap (a few hundred ms) and only ever run in the failure
    case.

    Returns True if the line was successfully scheduled. Never raises — a
    filler failing to play must not take down the call.
    """
    if session is None:
        return False

    # BUG 3 FIX — fixed-line dedup. Check-ins, thinking fillers and the greeting
    # all flow through here; a scheduling race could speak the SAME line twice
    # back-to-back ("repeats its own previous line verbatim"). Suppress an exact
    # repeat inside a short window, scoped per session via userdata.
    try:
        _ud_dd = getattr(session, "userdata", None)
        if isinstance(_ud_dd, dict):
            _dedup = _ud_dd.get("_fixed_line_dedup")
            if _dedup is None:
                _dedup = FixedLineDedup(
                    window_s=float(os.environ.get("FIXED_LINE_DEDUP_WINDOW_S", "8.0"))
                )
                _ud_dd["_fixed_line_dedup"] = _dedup
            if not _dedup.should_speak(text):
                logger.warning(
                    "[Bug3] Suppressed duplicate fixed line within dedup window: %r",
                    (text or "")[:60],
                )
                return True  # intentionally handled — do not let callers retry it
    except Exception as _dd_err:
        logger.debug("fixed-line dedup check failed (continuing): %s", _dd_err)

    # Gemini 3.1 Live rejects generate_reply() outright (mutable_chat_context
    # is False). Calling it anyway does not play a filler — it ABORTS the
    # in-flight Gemini turn, which is what made the agent go silent mid-call
    # ("failed to generate a reply" → thinking → listening, no audio).
    try:
        ud = getattr(session, "userdata", None) or {}
        model = (ud.get("realtime_model") or "") if isinstance(ud, dict) else ""
        if "3.1" in model.lower():
            # Gemini 3.1 Live cannot use generate_reply() for fixed lines; play
            # pre-synthesized audio via session.say(audio=...) like the opener.

            async def _run_audio():
                await _speak_fixed_line_audio(session, text, allow_interruptions, add_to_chat_ctx)

            try:
                _t = asyncio.create_task(_run_audio())
                _RETRY_TASKS.add(_t)
                _t.add_done_callback(_RETRY_TASKS.discard)
            except RuntimeError:
                logger.error("No event loop for fixed line audio: '%s'", text[:40])
                return False
            return True
    except Exception:
        pass

    def _try_once():
        """Returns True if the line was scheduled."""
        # VERBATIM-DELIVERY FIX. The instruction used to be
        # "Say exactly, warmly: '<text>'", which the native-audio model treated
        # as a suggestion: on a real call the fixed opener came out as
        # "Yes yes. I am sorry I am not sure I heard you. How may I help you?
        # Are you trying to book an appointment?" — a greeting plus an apology
        # plus two invented questions. There is no TTS on this session so we
        # cannot bypass the model; the instruction therefore has to close off
        # every direction the model was wandering in.
        handle = session.generate_reply(
            instructions=(
                "Speak the following line and NOTHING else, word for word, in a warm "
                "natural voice:\n"
                f"<<<{text}>>>\n"
                "Output ONLY the text between the markers. Do not translate it, do not "
                "reword it, do not shorten it. Add NOTHING before it — no 'yes yes', no "
                "'hello', no apology, no 'sorry I couldn't hear you'. Add NOTHING after "
                "it — no extra question, no offer, no follow-up. Ignore the live state "
                "and the conversation so far for this one turn; this is a fixed line, "
                "not a reply.\n"
                "CRITICAL: if the caller has already said something (for example they "
                "said \"hello?\" into a silent line before you started speaking), you "
                "still say ONLY the line above. Do NOT answer them, do NOT apologise, "
                "and above all do NOT say any form of \"sorry, I couldn't hear you\" / "
                "\"main sun nahi paayi\". Their words get answered on your NEXT turn, "
                "not this one."
            ),
            allow_interruptions=allow_interruptions,
        )
        return handle is not None

    def _session_running() -> bool:
        """Best-effort check that generate_reply() will not throw.

        LATENCY FIX: the caller of this function greets as soon as
        `session.current_agent` exists, which happens early inside
        session.start() — well before the session is running. Attempt 1 then
        always failed with "AgentSession isn't running" and we paid the retry
        delay on every single call. Probing here lets the retry loop spin
        tightly until the session is genuinely ready instead.

        Returns True when we cannot tell, so an unknown plugin version degrades
        to the old behaviour rather than never greeting.
        """
        for attr in ("_started", "started", "_running", "running"):
            try:
                v = getattr(session, attr)
            except Exception:
                continue
            if isinstance(v, bool):
                return v
        return True

    # Fast path: try once immediately
    try:
        if _try_once():
            return True
    except Exception as e:
        logger.warning("generate_reply attempt 1/%d failed: %s", retries, e)

    # Slow path: the realtime session was not ready. Retry on the event loop
    # (never time.sleep here — this runs inside async code and blocking it
    # would stall audio for the whole worker).
    async def _retry_loop():
        # BUGA: with retries=40/delay=0.1 this polls for up to ~4s and fires the
        # instant the session is live, which is what removes the 2.5s greeting
        # gap. Log only the outcome, not every poll.
        for attempt in range(2, retries + 1):
            await asyncio.sleep(retry_delay)
            try:
                if _try_once():
                    logger.info("Fixed line scheduled on attempt %d.", attempt)
                    return
            except Exception as e:
                logger.warning(
                    "generate_reply attempt %d/%d failed: %s", attempt, retries, e
                )
        logger.error(
            "Could not schedule fixed line after %d attempts — falling back to "
            "voice-pinned audio synthesis.", retries
        )
        # ISSUE 1 HARDENING: this used to be a bare `session.say(text, ...)` with no
        # `audio=`, which delegates rendering to a session-level `tts=` plugin and is
        # therefore the ONE audio path in this file that would not be pinned to
        # GEMINI_VOICE. It is unreachable under the shipped 3.1 config (and raises
        # anyway, since the realtime model reports supports_say=False), but leaving an
        # unvoiced say() in the tree is exactly how a wrong-voice regression gets
        # reintroduced. Route it through the same voice-pinned synthesis every other
        # fixed line uses, so no code path can ever speak in a different voice.
        try:
            await _speak_fixed_line_audio(
                session, text, allow_interruptions, add_to_chat_ctx
            )
        except Exception as e:
            logger.error("Fixed line could not be spoken at all: '%s': %s", text[:40], e)

    try:
        _t = asyncio.create_task(_retry_loop())
        _RETRY_TASKS.add(_t)
        _t.add_done_callback(_RETRY_TASKS.discard)
    except RuntimeError:
        # No running loop (shouldn't happen in the agent, but be safe).
        logger.error("No event loop available to retry fixed line: '%s'", text[:40])
    return False


_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_GREETING_CACHE_WAV = os.path.join(_AGENT_DIR, ".greeting_cache.wav")
_GREETING_CACHE_META = os.path.join(_AGENT_DIR, ".greeting_cache.meta")
_DEFAULT_GREETING_TEXT = (
    "Hi! This is Rahi from PUNO Jaipur. How may I assist you?"
)
_GREETING_TTS_MODELS = (
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
    "gemini-3.1-flash-tts-preview",
)


def _live_greeting_model() -> str:
    return (
        os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-live-preview").strip()
        or "gemini-3.1-flash-live-preview"
    )


def _resolve_greeting_cache_wav(text: str, voice: str, model: str = "") -> str:
    """Return the cached opener WAV if it matches this call's wording/voice/model.

    Gemini native-audio often produces no audio for generate_reply() until the
    caller speaks, so the agent stays silent on pickup. Playing a pre-rendered
    WAV is what lets Jasmine start the conversation. The WAV must come from the
    same Live model as the call, not from Gemini TTS — TTS shares the voice
    *name* but does not sound like the live agent.
    """
    if not os.path.isfile(_GREETING_CACHE_WAV):
        return ""
    cached_text = text.strip()
    cached_voice = (voice or "").strip()
    cached_model = (model or "").strip()
    try:
        with open(_GREETING_CACHE_META, encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
        if lines and lines[0].strip() != cached_text:
            logger.warning(
                "Greeting cache text mismatch; not using %s", _GREETING_CACHE_WAV
            )
            return ""
        # ISSUE 1: validation used to be skipped whenever EITHER side was blank,
        # so an unlabelled cache entry passed silently. Require both.
        if not (len(lines) > 1 and lines[1].strip()) or not cached_voice:
            logger.warning("Greeting cache voice unverifiable; not using WAV")
            return ""
        if True:
            if lines[1].strip() != cached_voice:
                logger.warning(
                    "Greeting cache voice mismatch (%s vs %s); not using WAV",
                    lines[1].strip(),
                    cached_voice,
                )
                return ""
        if cached_model:
            stamped = lines[4].strip() if len(lines) > 4 else ""
            # The guard that MATTERS here is keeping batch-TTS audio out: TTS shares
            # the voice name but does not sound like the live agent. Requiring an
            # EXACT live-model match was too strict — when the configured model was
            # briefly unreachable the fallback rendered the WAV, the marker no longer
            # matched, and the cache was rejected and re-synthesised on EVERY call.
            if not stamped.startswith("live:"):
                logger.warning(
                    "Greeting cache was not rendered by a live model (%s); not using WAV",
                    stamped or "unmarked",
                )
                return ""
            if stamped != f"live:{cached_model}":
                logger.info(
                    "Greeting cache was rendered by %s rather than the configured %s — "
                    "reusing it (both are live models).",
                    stamped[len("live:"):], cached_model,
                )
    except FileNotFoundError:
        # ISSUE 1: this used to `pass`, which meant a MISSING .meta disabled every
        # check above and any stale WAV on disk was played to callers - in whatever
        # voice/engine/wording it happened to hold. Fail closed instead.
        logger.warning(
            "Greeting cache meta missing (%s) - refusing to play unvalidated WAV.",
            _GREETING_CACHE_META,
        )
        return ""
    except Exception as e:
        logger.warning("Could not read greeting cache meta: %s", e)
        return ""
    return _GREETING_CACHE_WAV


def _write_pcm_wav(path: str, pcm: bytes, rate: int = 24000) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def _pcm_bytes(data) -> bytes:
    if not data:
        return b""
    if isinstance(data, str):
        import base64
        return base64.b64decode(data)
    return bytes(data)


def _pcm_from_live_message(msg) -> bytes:
    # Prefer the SDK convenience field. Using it AND model_turn.parts would
    # duplicate the same PCM and make the greeting play at half speed / double length.
    data = getattr(msg, "data", None)
    if data:
        return _pcm_bytes(data)
    sc = getattr(msg, "server_content", None)
    turn = getattr(sc, "model_turn", None) if sc is not None else None
    chunks: list[bytes] = []
    for part in (getattr(turn, "parts", None) or []):
        inline = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
        if inline is None:
            continue
        pcm = _pcm_bytes(getattr(inline, "data", None))
        if pcm:
            chunks.append(pcm)
    return b"".join(chunks)


def _live_turn_done(msg) -> bool:
    sc = getattr(msg, "server_content", None)
    if sc is None:
        return False
    return bool(
        getattr(sc, "turn_complete", False) or getattr(sc, "generation_complete", False)
    )


def _run_async(factory):
    """Run an async factory from sync code, even if a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(factory())).result(timeout=90)


async def _synthesize_greeting_pcm_live(text: str, voice: str, model: str) -> bytes:
    """Capture the opener from Gemini Live native audio — the same engine/voice as calls."""
    # ROOT FIX: this used to read GOOGLE_GENAI_USE_VERTEXAI directly and so forced a
    # Vertex endpoint for native-audio/Gemini-3 Live models, which are Gemini-API-only.
    # That returned websocket 1008 "Publisher model ... not found" on every attempt and
    # is what made the greeting re-synthesise on every call. Routing is now decided in
    # exactly one place, the same way the realtime session decides it.
    client = _genai_client_for(model)
    is_gemini_3 = "gemini-3" in (model or "").lower()
    config_kw = dict(
        response_modalities=["AUDIO"],
        temperature=0.6,
        speech_config=genai_types.SpeechConfig(
            voice_config=genai_types.VoiceConfig(
                prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                    voice_name=voice,
                )
            )
        ),
        enable_affective_dialog=not is_gemini_3,
        output_audio_transcription=genai_types.AudioTranscriptionConfig(),
        system_instruction=(
            "You are Rahi, a warm, upbeat front-desk host for an entertainment "
            "centre, on a live phone call. When given a fixed line, say ONLY that "
            "line in your natural live voice. Do not greet twice, do not add a "
            "question, do not apologise, do not offer extra help."
        ),
    )
    if is_gemini_3:
        config_kw["thinking_config"] = genai_types.ThinkingConfig(
            thinking_level=genai_types.ThinkingLevel.MINIMAL,
        )
    config = genai_types.LiveConnectConfig(**config_kw)
    prompt = (
        "Speak the following line and NOTHING else, word for word, in a warm "
        "natural live phone voice:\n"
        f"<<<{text}>>>\n"
        "Output ONLY the text between the markers. Do not translate it, do not "
        "reword it, do not shorten it. Add NOTHING before it and NOTHING after it."
    )

    async def _one_attempt(send_text: bool) -> tuple[bytes, str]:
        chunks: list[bytes] = []
        transcript: list[str] = []
        async with client.aio.live.connect(model=model, config=config) as session:
            if send_text:
                await session.send_realtime_input(text=prompt)
            else:
                await session.send_client_content(
                    turns={"role": "user", "parts": [{"text": prompt}]},
                    turn_complete=True,
                )
            async for msg in session.receive():
                pcm = _pcm_from_live_message(msg)
                if pcm:
                    chunks.append(pcm)
                sc = getattr(msg, "server_content", None)
                ot = getattr(sc, "output_transcription", None) if sc is not None else None
                piece = getattr(ot, "text", None) if ot is not None else None
                if piece:
                    transcript.append(piece)
                if _live_turn_done(msg):
                    break
        return b"".join(chunks), "".join(transcript).strip()

    last_err: Optional[BaseException] = None
    best: Optional[bytes] = None
    max_bytes = 24_000 * 2 * 8  # 8s of 24 kHz 16-bit mono — opener is one short line
    for attempt in range(1, 5):
        send_text = attempt <= 3
        try:
            pcm, spoken = await asyncio.wait_for(_one_attempt(send_text), timeout=45)
            dur = len(pcm) / (24000 * 2)
            logger.info(
                "Live greeting attempt %d: %.2fs, transcript=%r",
                attempt,
                dur,
                spoken[:160],
            )
            if len(pcm) < 24000:
                last_err = RuntimeError(f"{model} returned too little audio ({len(pcm)} bytes)")
                continue
            if len(pcm) <= max_bytes:
                logger.info(
                    "Captured live greeting audio from %s (%d bytes, %.2fs)",
                    model,
                    len(pcm),
                    dur,
                )
                return pcm
            if best is None or len(pcm) < len(best):
                best = pcm
            last_err = RuntimeError(
                f"{model} opener too long ({dur:.1f}s, transcript={spoken[:80]!r})"
            )
        except Exception as e:
            last_err = e
            logger.warning(
                "Live greeting capture via %s failed (attempt %d): %s",
                model,
                attempt,
                e,
            )
            # A missing/denied/unsupported model fails identically every time, so
            # the remaining attempts are pure latency — this is what turned one bad
            # model into a 32-second startup. Abandon THIS model immediately; the
            # caller (_synthesize_greeting_pcm) still tries the fallback model.
            if _is_permanent_model_error(e):
                logger.error(
                    "Model %s is permanently unavailable (%s) — skipping its remaining "
                    "retries and moving to the fallback model.", model, e,
                )
                break
    if best is not None and len(best) >= 24000:
        logger.warning(
            "Using shortest live greeting capture anyway (%.2fs)",
            len(best) / (24000 * 2),
        )
        return best
    raise RuntimeError(f"Live greeting capture failed: {last_err}")


# meta can stamp real provenance instead of an assumed 'live:' marker.
# Records which engine actually produced the last greeting PCM, so the cache
_LAST_GREETING_ENGINE = ""

def _synthesize_greeting_pcm(text: str, voice: str, model: str) -> bytes:
    """Render the opener with the live agent voice, then TTS only as a last resort."""
    models = []
    for name in (model, _GEMINI_FALLBACK_MODEL):
        n = (name or "").strip()
        if n and n not in models:
            models.append(n)
    last_err: Optional[BaseException] = None
    for live_model in models:
        try:
            global _LAST_GREETING_ENGINE
            _LAST_GREETING_ENGINE = "live:" + live_model
            return _run_async(
                lambda m=live_model: _synthesize_greeting_pcm_live(text, voice, m)
            )
        except Exception as e:
            last_err = e
            logger.warning("Live greeting synth failed on %s: %s", live_model, e)

    # TTS fallback for the opener — routed through the same single decision point.
    client = _genai_client_for(_GREETING_TTS_MODELS[0])
    prompt = (
        "Speak in a warm, upbeat, friendly voice as an entertainment-centre "
        "front-desk host. Say exactly these words and nothing else:\n"
        f"{text}"
    )
    _LAST_GREETING_ENGINE = ""  # live attempts exhausted
    for tts_model in _GREETING_TTS_MODELS:
        try:
            response = client.models.generate_content(
                model=tts_model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=genai_types.SpeechConfig(
                        voice_config=genai_types.VoiceConfig(
                            prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                                voice_name=voice,
                            )
                        )
                    ),
                ),
            )
            pcm = _extract_tts_pcm(response)
            _LAST_GREETING_ENGINE = "tts:" + tts_model
            if pcm:
                logger.warning(
                    "Fell back to TTS greeting audio via %s — voice may not match live agent",
                    tts_model,
                )
                return pcm
            last_err = RuntimeError(f"{tts_model} returned no audio")
        except Exception as e:
            last_err = e
            logger.warning("Greeting TTS via %s failed: %s", tts_model, e)
    raise RuntimeError(f"Greeting synthesis failed: {last_err}")


def _extract_tts_pcm(response) -> bytes:
    """Pull raw 24 kHz PCM out of a Gemini TTS generate_content response."""
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            pcm = _pcm_bytes(
                getattr(
                    getattr(part, "inline_data", None) or getattr(part, "inlineData", None),
                    "data",
                    None,
                )
            ) if (getattr(part, "inline_data", None) or getattr(part, "inlineData", None)) else b""
            if pcm:
                return pcm
    return b""


def _ensure_greeting_cache_wav(text: str, voice: str, model: str = "") -> str:
    """Return a matching opener WAV, rebuilding it if wording/voice/live-model changed."""
    text = (text or "").strip()
    voice = (voice or "").strip() or _configured_voice()
    model = (model or "").strip() or _live_greeting_model()
    if not text:
        return ""
    existing = _resolve_greeting_cache_wav(text, voice, model)
    if existing:
        return existing
    try:
        pcm = _synthesize_greeting_pcm(text, voice, model)
        tmp = f"{_GREETING_CACHE_WAV}.{os.getpid()}.tmp"
        _write_pcm_wav(tmp, pcm)
        os.replace(tmp, _GREETING_CACHE_WAV)
        with open(_GREETING_CACHE_META, "w", encoding="utf-8") as f:
            _engine = _LAST_GREETING_ENGINE or ("live:" + model)
            if not _engine.startswith("live:"):
                logger.warning(
                    "Greeting rendered by %s, not the live model - caching real provenance "
                    "so it is rebuilt once live synthesis works.", _engine)
            f.write(text + chr(10) + voice + chr(10) + "24000" + chr(10) + "1" + chr(10) + _engine + chr(10))
        logger.info("Wrote live greeting cache WAV (%s)", _GREETING_CACHE_WAV)
        return _GREETING_CACHE_WAV
    except Exception as e:
        logger.warning("Could not synthesize greeting cache: %s", e)
        try:
            tmp = f"{_GREETING_CACHE_WAV}.{os.getpid()}.tmp"
            if os.path.isfile(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return ""


async def _iter_wav_frames(path: str, frame_ms: int = 20) -> AsyncIterator[rtc.AudioFrame]:
    """Yield PCM frames from a WAV so session.say() can play it without TTS."""
    with wave.open(path, "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        pcm = wf.readframes(wf.getnframes())
    bytes_per_sample = sample_width * channels
    samples_per_frame = max(1, int(rate * frame_ms / 1000))
    frame_bytes = samples_per_frame * bytes_per_sample
    offset = 0
    while offset < len(pcm):
        chunk = pcm[offset : offset + frame_bytes]
        offset += frame_bytes
        n = len(chunk) // bytes_per_sample
        if n <= 0:
            break
        yield rtc.AudioFrame(
            data=chunk,
            sample_rate=rate,
            num_channels=channels,
            samples_per_channel=n,
        )
        await asyncio.sleep(0)


async def _schedule_opening_greeting(session, userdata: dict) -> bool:
    """Start the fixed opener once per answered call — agent speaks first.

    Primary path plays the cached WAV through session.say(audio=...), which
    does not depend on Gemini generate_reply() (native-audio often stays mute
    until the caller talks). generate_reply() is the fallback if the cache
    is missing or say() cannot start.

    Used from Agent.on_enter and the _greet_asap backup. The userdata flag
    prevents a double greeting.
    """
    if not userdata or userdata.get("greeting_scheduled"):
        return False
    if session is None:
        return False
    text = (userdata.get("greeting_text") or "").strip() or _DEFAULT_GREETING_TEXT
    interruptible = bool(userdata.get("greeting_interruptible", True))
    wav = (userdata.get("greeting_wav") or "").strip()
    userdata["greeting_scheduled"] = True

    if wav and os.path.isfile(wav):
        last_err: Optional[BaseException] = None
        for attempt in range(1, 51):
            try:
                handle = session.say(
                    text,
                    audio=_iter_wav_frames(wav),
                    allow_interruptions=interruptible,
                    add_to_chat_ctx=True,
                )
                if handle is not None:
                    logger.info(
                        "Opening greeting playing from cache (attempt %d): %s",
                        attempt,
                        wav,
                    )
                    return True
            except Exception as e:
                last_err = e
                await asyncio.sleep(0.02)
        logger.warning(
            "Cached greeting could not start (%s); falling back to generate_reply.",
            last_err,
        )

    speak_fixed_line(session, text, allow_interruptions=interruptible)
    logger.info("Opening greeting scheduled via generate_reply")
    return True


_MAX_CONFIRM_GATE_ATTEMPTS = 2


async def _execute_end_call_drop(userdata: dict):
    if userdata.get("_drop_in_progress"):
        return
    userdata["_drop_in_progress"] = True

    # 1. If agent is currently thinking or just starting speech, wait a short moment (max 0.6s)
    # for speech state to register
    for _ in range(6):
        if userdata.get("is_agent_speaking"):
            break
        await asyncio.sleep(0.1)

    # 2. Wait until the agent has finished speaking the closing line — BOUNDED.
    #
    #    ISSUE 4 ROOT CAUSE. This used to be an unbounded `while
    #    userdata.get("is_agent_speaking"): await sleep(0.05)`. That flag is set from
    #    the agent_state_changed event, which we already established lags or MISSES
    #    the "stopped speaking" transition on Gemini 3.1 native audio. Whenever that
    #    event was missed the loop spun forever and the call was never disconnected —
    #    the agent said goodbye and then simply stayed on the line.
    #
    #    Two independent escapes now: the ground-truth agent-audio stamp (if no agent
    #    audio has arrived for a moment, speech has finished whatever the flag says),
    #    and a hard ceiling.
    _speech_max = float(os.environ.get("END_CALL_SPEECH_MAX_WAIT_S", "15.0"))
    _audio_quiet = float(os.environ.get("END_CALL_AUDIO_QUIET_S", "1.2"))
    _wait_t0 = time.monotonic()
    while userdata.get("is_agent_speaking"):
        _last_audio = float(userdata.get("last_agent_audio_monotonic") or 0.0)
        _since_audio = (time.monotonic() - _last_audio) if _last_audio else None
        if _since_audio is not None and _since_audio > _audio_quiet:
            logger.warning(
                "[Issue4] is_agent_speaking still True but no agent audio for %.1fs — "
                "treating the closing line as finished (stale state event).", _since_audio,
            )
            break
        if time.monotonic() - _wait_t0 > _speech_max:
            logger.error(
                "[Issue4] Waited %.1fs for is_agent_speaking to clear and it never did — "
                "forcing the disconnect so the call cannot hang open.", _speech_max,
            )
            break
        await asyncio.sleep(0.05)

    # 3. Short beat after the greeting completes before cutting the call
    await asyncio.sleep(1.0)

    # Write the lead BEFORE teardown (room.disconnect() -> session close clears the
    # transcript store the extractor reads) — but NEVER let it hold the call open.
    # ISSUE 4, second cause: this await was unbounded, and finalize does a transcript
    # drain plus an LLM extraction, so a slow model kept the caller connected for
    # seconds after goodbye. Shielded so the save still completes in the background
    # if we stop waiting for it.
    try:
        _lead_task = asyncio.ensure_future(finalize_lead_for_call(userdata, reason="end_call"))
        _TOOL_BG_TASKS.add(_lead_task)
        _lead_task.add_done_callback(_TOOL_BG_TASKS.discard)
        await asyncio.wait_for(
            asyncio.shield(_lead_task),
            timeout=float(os.environ.get("END_CALL_LEAD_MAX_S", "4.0")),
        )
    except asyncio.TimeoutError:
        logger.warning(
            "[Issue4] Lead finalize exceeded its budget — disconnecting now; the save "
            "continues in the background."
        )
    except Exception as _le:
        logger.error("[Lead] finalize on end_call failed: %s", _le)

    room = userdata.get("room")
    if not room:
        return

    logger.info("Agent finished closing greeting. Disconnecting room cleanly and immediately.")

    try:
        await room.disconnect()
    except Exception as e:
        logger.warning("room.disconnect() failed: %s", e)

    try:
        from livekit import api
        api_client = api.LiveKitAPI()
        try:
            await api_client.room.delete_room(api.DeleteRoomRequest(room=room.name))
        finally:
            await api_client.aclose()
    except Exception as e:
        logger.error("Failed to delete room %s: %s", room.name, e)


@function_tool
async def end_call(ctx: RunContext) -> str:
    """Disconnect the call after finishing the closing greetings.

    Call this tool when:
      1. the caller has signalled they are finished (said goodbye, bye, 'that's all',
         'nothing else', 'no thanks', 'bas itna hi', or answered no to 'anything else?'), AND
      2. you are speaking your closing greeting: "Thanks for calling PUNO Jaipur. Have a great day!"
         (or in Hindi: "PUNO Jaipur mein call karne ke liye dhanyavaad, aapka din shubh ho!").
    """
    try:
        _st = ctx.userdata.get("state")
        if _st is not None:
            _st.call_wrapup_ok = True
    except Exception:
        pass

    asyncio.create_task(_execute_end_call_drop(ctx.userdata))
    return (
        "Done. The call will disconnect cleanly 1 second after you finish speaking the closing greeting. "
        "Produce no further speech after the greeting."
    )


# ===========================================================================
# PUNO Jaipur tools (v2.3)
#
# Info tools read verified facts from puno_data.py; the four system actions
# (confirm_booking / create_lead / transfer_to_human / log_call_summary)
# implement spec §6. There is NO payment channel anywhere in this flow — a
# confirmed booking fires a *dummy* SMS immediately and nothing is ever charged.
# These are the ONLY tools registered on the agent (see the tools=[...] list);
# the hospital-era tools this project was converted from have been removed.
# ===========================================================================


def _puno_state(ctx: RunContext) -> AppointmentState:
    return ctx.userdata["state"]


async def _puno_persist(ctx: RunContext) -> None:
    """Save state + transcript (best-effort) after a PUNO tool mutates state."""
    try:
        await save_appointment_state(ctx.userdata["user_id"], _puno_state(ctx))
    except Exception as e:
        logger.warning("PUNO: save_appointment_state failed: %s", e)
    try:
        await save_call_transcript(ctx.userdata["user_id"], _get_transcript(ctx.userdata))
    except Exception:
        pass


def _puno_total_headcount(state: AppointmentState) -> int:
    return int(state.adults_7plus or 0) + int(state.kids_under7 or 0)


# ===========================================================================
# END-OF-CALL LEAD FINALIZER — runs on EVERY call, exactly once.
#
# The old behaviour saved a lead only if the model happened to call create_lead(),
# and built the record purely from AppointmentState (i.e. only fields the model had
# separately pushed through update_booking_details). Calls that ended any other way
# wrote nothing, and details the caller clearly said came back null.
#
# This runs unconditionally at end of call, extracts from the FULL transcript,
# resolves the phone from spoken digits OR the SIP caller ID, and classifies the
# lead HOT/WARM/COLD. It never raises and always writes a record.
# ===========================================================================

async def finalize_lead_for_call(userdata: dict, *, reason: str = "call_ended") -> Optional[str]:
    """Build + save the end-of-call lead. Idempotent: only the first call does work."""
    if not isinstance(userdata, dict):
        return None
    if userdata.get("_lead_finalized"):
        logger.debug("[Lead] finalize skipped (already done, reason=%s)", reason)
        return userdata.get("_lead_id")
    userdata["_lead_finalized"] = True  # claim BEFORE any await — prevents double-fire

    state = userdata.get("state")
    _call_id = userdata.get("call_id") or ""
    try:
        from lead_capture import build_lead_record
        try:
            from call_recording import get_conversation, transcript_turn_count

            # TRANSCRIPT DRAIN. The caller's LAST turn only lands in the store once
            # the realtime provider emits a FINAL input transcription, and the SDK
            # documents that a provider may withhold that final until its reply has
            # finished generating. Snapshotting at t=0 of the disconnect therefore
            # drops the most valuable turn on exactly the calls that matter most —
            # "same number, book me for Saturday" *click* — leaving the fields null
            # and the lead misclassified. Wait until the store stops growing (or the
            # cap elapses) before reading. This still completes long before
            # session.aclose()/ctx.shutdown(), which is when the store is cleared,
            # and _lead_finalized was already claimed above so the backstop cannot
            # double-fire while we wait.
            _max_wait = float(os.environ.get("LEAD_TRANSCRIPT_DRAIN_MAX_S", "1.5"))
            _quiet_for = float(os.environ.get("LEAD_TRANSCRIPT_DRAIN_QUIET_S", "0.35"))
            _poll = 0.1
            _last = transcript_turn_count(_call_id)
            _waited = _stable = 0.0
            while _waited < _max_wait:
                await asyncio.sleep(_poll)
                _waited += _poll
                _now = transcript_turn_count(_call_id)
                if _now != _last:
                    _last, _stable = _now, 0.0
                else:
                    _stable += _poll
                    if _stable >= _quiet_for:
                        break
            logger.info(
                "[Lead] transcript drained: %d turns after %.1fs (reason=%s)",
                _last, _waited, reason,
            )
            turns = get_conversation(_call_id)
        except Exception as _te:
            logger.warning("[Lead] could not read transcript store: %s", _te)
            turns = []

        snapshot = {}
        if state is not None:
            snapshot = {
                "name": getattr(state, "patient_name", None) or getattr(state, "caller_name", None),
                "phone": getattr(state, "patient_phone", None),
                "package_interest": getattr(state, "package_name", None) or getattr(state, "reason", None),
                "preferred_date": getattr(state, "appointment_date", None),
                "preferred_slot": getattr(state, "appointment_time", None),
                "adults_7plus": getattr(state, "adults_7plus", None),
                "kids_under7": getattr(state, "kids_under7", None),
                "intent": getattr(state, "call_category", None),
            }

        record = await asyncio.to_thread(
            build_lead_record,
            turns=turns,
            state=snapshot,
            caller_id=_sip_caller_mobile(userdata),
            today=_today_ist().strftime("%Y-%m-%d"),
            booking_id=getattr(state, "booking_id", None) if state is not None else None,
            outcome=getattr(state, "call_outcome", None) if state is not None else None,
            transferred=bool(getattr(state, "transferred_to_human", False)) if state is not None else False,
            language=getattr(state, "caller_language", None) if state is not None else None,
            call_id=userdata.get("call_id"),
        )
        record["end_reason"] = reason

        lead_id = await save_puno_lead(record)
        userdata["_lead_id"] = lead_id
        if state is not None:
            try:
                state.lead_id = lead_id
            except Exception:
                pass
        logger.info(
            "[Lead] saved %s quality=%s source=%s phone=%s(%s) name=%r turns=%s reason=%s",
            lead_id, record.get("lead_quality"), record.get("extraction_source"),
            record.get("phone"), record.get("phone_source"), record.get("name"),
            record.get("turn_count"), reason,
        )
        return lead_id
    except Exception as e:
        logger.error("[Lead] finalize FAILED (%s) — attempting minimal record.", e)
        # Absolute last resort: never lose the call entirely.
        try:
            minimal = {
                "type": "PUNO_LEAD",
                "lead_quality": "COLD",
                "reasoning": f"Lead finalization failed: {e}",
                "phone": _sip_caller_mobile(userdata) or None,
                "phone_source": "caller_id" if _sip_caller_mobile(userdata) else "none",
                "notes": "",
                "outcome": "Info only",
                "extraction_source": "rule_based_fallback",
                "payment_status": "n/a",
                "end_reason": reason,
                "call_id": userdata.get("call_id"),
            }
            return await save_puno_lead(minimal)
        except Exception as e2:
            logger.error("[Lead] minimal lead save also failed: %s", e2)
            return None


# --- Info tools -------------------------------------------------------------
@function_tool
async def get_puno_info(ctx: RunContext) -> str:
    """Location, timings, parking and contact for PUNO Jaipur. Use for
    'where are you', 'what time do you open', 'is there parking', 'phone number'."""
    state = _puno_state(ctx)
    info = (
        f"Spoken address (do NOT read the full address aloud): \"{puno_data.LOCATION['spoken_address']}\" — "
        f"offer the full address and Google Maps pin on WhatsApp. "
        f"Timings: {puno_data.LOCATION['timings']}. "
        f"Parking: {puno_data.LOCATION['parking']}. "
        f"Phone / WhatsApp: {puno_data.CONTACT['phone']}. Website: {puno_data.CONTACT['website']}."
    )
    return info + _caller_language_reply_hint(state)


@function_tool
async def get_packages(ctx: RunContext) -> str:
    """The three PUNO packages with prices (GST inclusive). Use for pricing /
    'how much' / 'what packages do you have'. Summarise the 2-3 that fit; do not read all aloud."""
    state = _puno_state(ctx)
    lines = []
    for p in puno_data.PACKAGES:
        lines.append(f"{p['name']}: {p['price_str']}, {p['duration']} — includes {p['includes']}. Validity: {p['validity']}")
    body = "The three packages (all GST inclusive): " + " | ".join(lines) + \
        " Anything beyond these three — say 'let me confirm the live price for you'."
    return body + _caller_language_reply_hint(state)


@function_tool
async def get_activities(ctx: RunContext) -> str:
    """The attractions at PUNO Jaipur. Use for 'what activities/games do you have'.
    Summarise the top few relevant ones; offer the full list on WhatsApp."""
    state = _puno_state(ctx)
    lines = []
    for a in puno_data.ATTRACTIONS:
        note = f" ({a['notes']})" if a["notes"] else ""
        lines.append(f"{a['name']}: {a['desc']}{note}")
    body = "Attractions: " + " | ".join(lines)
    return body + _caller_language_reply_hint(state)


@function_tool
async def check_activity(
    ctx: RunContext,
    activity: Annotated[str, "The game or activity the caller asked about, in their own words."],
) -> str:
    """Check whether a specific game/activity is actually offered at PUNO Jaipur.

    ALWAYS use this before answering "do you have <X>?" for any game or activity you
    are not certain about. It checks the real attraction list rather than your own
    assumption, and tells you exactly how to answer — including when to decline."""
    state = _puno_state(ctx)
    res = puno_data.activity_availability(activity)
    status = res["status"]
    logger.info("[Issue2] check_activity(%r) -> %s (%s)", activity, status, res.get("matched"))

    if status == "available":
        return (
            f"YES — '{activity}' is available: it is part of {res['matched']}. "
            "Confirm it warmly and briefly, then offer the next step."
            + _caller_language_reply_hint(state)
        )
    if status == "not_offered":
        alts = ", ".join(res["alternatives"][:2])
        return (
            f"NO — PUNO Jaipur does NOT offer '{activity}'. This is definitive: the attraction "
            f"list is complete. Tell the caller clearly and politely that we don't have it, then "
            f"offer 1-2 alternatives we DO have ({alts}). Do NOT say 'maybe', do NOT say you will "
            f"check, and do NOT imply we might have it."
            + _caller_language_reply_hint(state)
        )
    return (
        f"UNCERTAIN — '{activity}' may or may not map to one of our zones, and you do NOT have a "
        "confirmed answer. Do NOT guess either way. Say you don't have that exact detail, offer to "
        "note it down for the team to confirm, and continue helping with what you do know."
        + _caller_language_reply_hint(state)
    )


@function_tool
async def get_offers(ctx: RunContext) -> str:
    """Current PUNO offers. ALWAYS add that offers change and you'll confirm the live price."""
    state = _puno_state(ctx)
    lines = [f"{o['name']}: {o['desc']}" for o in puno_data.OFFERS]
    body = "Offers (these change — always say you'll confirm the live price): " + " | ".join(lines)
    return body + _caller_language_reply_hint(state)


@function_tool
async def get_membership(ctx: RunContext) -> str:
    """PUNO VIP 'Infinity Club' membership cards. If the caller asks how Golden/Purple
    map to these cards, or for terms beyond this table, capture interest and hand off to sales."""
    state = _puno_state(ctx)
    lines = [f"{c['name']}: {c['price']} (value {c['value']}) — {c['benefits']}" for c in puno_data.MEMBERSHIP_CARDS]
    body = (
        f"PUNO VIP Infinity Club, tiers {puno_data.MEMBERSHIP_TIERS}. Cards: " + " | ".join(lines) +
        " For how Golden/Purple map to these cards, or any terms beyond this, hand off to sales — do not improvise."
    )
    return body + _caller_language_reply_hint(state)


@function_tool
async def get_birthday_info(ctx: RunContext) -> str:
    """Birthday / party package info. Party bookings are NEVER completed on this call —
    collect interest and route to the Party Specialist; frame prices as 'starting from'."""
    state = _puno_state(ctx)
    b = puno_data.BIRTHDAY
    body = (
        f"Birthday parties include: {b['included']}. {b['minimum']} " +
        " ".join(b["options"]) +
        ". Always frame party prices as 'starting from, depending on inclusions' and confirm the exact "
        "quote with the Party Specialist — do not complete a party booking on this call."
    )
    return body + _caller_language_reply_hint(state)


# --- Data collection --------------------------------------------------------
@function_tool
async def update_booking_details(
    ctx: RunContext,
    name: Annotated[str, "The caller's full name for the booking."] = "",
    phone: Annotated[str, "10-digit contact number. The confirmation SMS goes here."] = "",
    kids_under7: Annotated[int, "Number of visitors UNDER 7 years old."] = -1,
    adults_7plus: Annotated[int, "Number of visitors aged 7 and OVER."] = -1,
    date: Annotated[str, "Preferred visit date (YYYY-MM-DD or as the caller said it)."] = "",
    slot: Annotated[str, "Preferred time slot, e.g. '4 PM'."] = "",
    package: Annotated[str, "Package of interest: a package id or name, or an activity."] = "",
) -> str:
    """Save any of the five booking fields the caller has provided (name, phone,
    headcount split under-7/7+, date, slot, package). Call it as details arrive.
    Only pass the fields the caller actually gave. Returns a short read-back."""
    state = _puno_state(ctx)
    state.booking_intent_detected = True

    if name.strip():
        state.patient_name = name.strip()
        if not state.caller_name:
            state.caller_name = name.strip()
    if phone.strip():
        norm = normalize_hindi_phone_number(phone) or phone.strip()
        state.patient_phone = norm
    if kids_under7 >= 0:
        state.kids_under7 = int(kids_under7)
    if adults_7plus >= 0:
        state.adults_7plus = int(adults_7plus)
    if date.strip():
        state.appointment_date = date.strip()
    if slot.strip():
        state.appointment_time = slot.strip()
    if package.strip():
        pkg = get_package(package)
        if pkg:
            state.package_id = pkg["id"]
            state.package_name = pkg["name"]
        else:
            # An activity of interest rather than a named package.
            state.reason = package.strip()

    await _puno_persist(ctx)

    parts = []
    nm = state.patient_name or state.caller_name
    if nm:
        parts.append(f"name {nm}")
    if state.patient_phone:
        parts.append("phone " + " ".join(list(str(state.patient_phone))))
    if state.adults_7plus is not None or state.kids_under7 is not None:
        parts.append(f"{state.adults_7plus or 0} aged 7-plus and {state.kids_under7 or 0} under 7")
    if state.appointment_date:
        parts.append(f"date {state.appointment_date}")
    if state.appointment_time:
        parts.append(f"slot {state.appointment_time}")
    if state.package_name:
        parts.append(f"package {state.package_name}")
    readback = "Saved. So far I have: " + ", ".join(parts) + "." if parts else "Saved."
    return (
        readback + " Read back only what is new, once, in your own words, then ask for the next "
        "missing detail (at most two at a time)." + _caller_language_reply_hint(state)
    )


# --- §6 System action: confirm_booking -------------------------------------
@function_tool
async def confirm_booking(
    ctx: RunContext,
    package_id: Annotated[str, "Package id or name, e.g. 'tourist_weekday'. Optional if already saved."] = "",
    date: Annotated[str, "Visit date. Optional if already saved."] = "",
    slot: Annotated[str, "Time slot. Optional if already saved."] = "",
    adults_7plus: Annotated[int, "Visitors aged 7+. Optional if already saved (-1 = leave unchanged)."] = -1,
    kids_under7: Annotated[int, "Visitors under 7. Optional if already saved (-1 = leave unchanged)."] = -1,
) -> str:
    """Confirm and write the booking (spec §5A.3). Call ONLY after the caller has
    heard the summary and said yes. On success it writes the Booked record, fires
    the dummy confirmation SMS immediately, and returns a booking_id — ONLY THEN
    may you speak the confirmation line. On failure, do NOT confirm: hand over to a
    human. There is no payment of any kind in this flow."""
    state = _puno_state(ctx)
    transcript = _get_transcript(ctx.userdata)
    user_id = ctx.userdata["user_id"]

    # Idempotency: already booked on this call → return the same booking_id.
    if state.booking_id:
        return (
            f"Already confirmed (booking id {state.booking_id}). Do not book again. Reassure the caller "
            "their booking stands and ask if there's anything else." + _caller_language_reply_hint(state)
        )

    # Merge any newly-passed fields.
    if package_id.strip():
        pkg = get_package(package_id)
        if pkg:
            state.package_id = pkg["id"]
            state.package_name = pkg["name"]
    if date.strip():
        state.appointment_date = date.strip()
    if slot.strip():
        state.appointment_time = slot.strip()
    if adults_7plus >= 0:
        state.adults_7plus = int(adults_7plus)
    if kids_under7 >= 0:
        state.kids_under7 = int(kids_under7)

    # Required fields — ask for whatever is missing WITHOUT booking.
    name = state.patient_name or state.caller_name
    missing = []
    if not name:
        missing.append("the name for the booking")
    if not state.patient_phone:
        missing.append("a contact phone number")
    if not state.package_id:
        missing.append("which package")
    if not state.appointment_date:
        missing.append("the date")
    if not state.appointment_time:
        missing.append("the time slot")
    if _puno_total_headcount(state) <= 0:
        missing.append("how many people, split into under 7 and 7-and-over")
    if missing:
        state.booking_status = "DRAFT"
        await _puno_persist(ctx)
        return (
            "Not booked yet — still need: " + ", ".join(missing) +
            ". Ask for just those (at most two at a time). Do NOT say the booking is confirmed." +
            _caller_language_reply_hint(state)
        )

    try:
        pkg = get_package(state.package_id) or {}
        headcount = _puno_total_headcount(state)
        unit_price = int(pkg.get("price") or 0)
        total_price = unit_price * max(1, headcount)
        state.total_price = total_price

        booking_id = await generate_booking_id()

        record = {
            "booking_id": booking_id,
            "name": name,
            "phone": state.patient_phone,
            "package_id": state.package_id,
            "package_name": state.package_name,
            "unit_price": unit_price,
            "adults_7plus": int(state.adults_7plus or 0),
            "kids_under7": int(state.kids_under7 or 0),
            "headcount": headcount,
            "total_price": total_price,
            "visit_date": state.appointment_date,
            "time_slot": state.appointment_time,
            "is_peak_day": bool(state.is_peak_day) if state.is_peak_day is not None else None,
            "status": "Booked",
            "payment_status": "n/a",
        }
        # Fail-soft persistence — returns the booking_id it actually wrote.
        booking_id = await save_puno_booking(record)

        # Fire the dummy confirmation SMS immediately (no payment dependency, no link).
        sms_body = (
            f"PUNO Jaipur: Booking confirmed! {state.package_name} on {state.appointment_date} "
            f"at {state.appointment_time} for {headcount} guest(s). Ref {booking_id}. "
            f"Settle at the front desk on arrival. See you soon!"
        )
        await save_dummy_sms(booking_id, state.patient_phone, sms_body)

        # Commit state.
        state.booking_id = booking_id
        state.booking_status = "BOOKED"
        state.payment_status = "n/a"
        state.call_outcome = "Booked"
        await save_appointment_state(user_id, state)
        await save_call_transcript(user_id, transcript)

        price_words = f"{total_price:,}".replace(",", ",")
        return (
            f"SUCCESS. booking_id={booking_id}. The booking is written and the confirmation SMS has fired. "
            f"NOW speak the confirmation line ('Perfect — your booking is confirmed! You'll get the confirmation "
            f"SMS in just a moment.'), then read back once: name {name}, date {state.appointment_date}, slot "
            f"{state.appointment_time}, package {state.package_name}, {headcount} guest(s), total about Rs. {price_words}. "
            f"Do NOT mention any payment or link. Then ask if there's anything else." +
            _caller_language_reply_hint(state)
        )
    except Exception as e:
        logger.error("confirm_booking failed: %s", e)
        state.booking_status = "FAILED"
        state.follow_up_needed = True
        state.follow_up_note = f"confirm_booking error: {e}"
        await _puno_persist(ctx)
        return (
            "BOOKING WRITE FAILED — do NOT say the booking is confirmed. Say 'Give me one moment, I'm just "
            "having trouble locking that in — let me get a team member to finish this for you', then call "
            "transfer_to_human(reason='booking write failed')." + _caller_language_reply_hint(state)
        )


# --- §6 System action: create_lead -----------------------------------------
@function_tool
async def create_lead(
    ctx: RunContext,
    intent: Annotated[str, "Short intent/category, e.g. 'birthday party', 'pricing enquiry', 'callback'."] = "",
    notes: Annotated[str, "Anything that has no field of its own: special requests, objections, context."] = "",
) -> str:
    """Note the caller's intent when they want info but are not ready to book, or as the
    Step-6 fallback when no human is available for a handover.

    NOTE: this no longer writes the lead itself. A lead is now written for EVERY call by
    finalize_lead_for_call() at end of call, which extracts the structured fields from the
    full transcript and classifies the lead HOT/WARM/COLD. This tool just records the
    intent/notes the model observed, so the finalizer can use them. Writing here as well
    would produce two competing lead rows for one call.
    """
    state = _puno_state(ctx)
    if intent.strip():
        state.call_category = intent.strip()
    if notes.strip():
        prev = (state.follow_up_note or "").strip()
        state.follow_up_note = (prev + " | " + notes.strip()).strip(" |") if prev else notes.strip()
    state.call_outcome = state.call_outcome or "Lead created"
    state.follow_up_needed = True
    await _puno_persist(ctx)
    logger.info("[Lead] intent noted in-call (intent=%r) — record is written at end of call.", intent)
    return (
        "Noted. Tell the caller a team member may follow up and they can call back anytime — "
        "in your own warm words." + _caller_language_reply_hint(state)
    )


# --- §6 System action: transfer_to_human -----------------------------------
@function_tool
async def transfer_to_human(
    ctx: RunContext,
    reason: Annotated[str, "Why: 'existing booking', 'payment enquiry', 'party quote', 'complaint', 'booking write failed', etc."] = "",
) -> str:
    """Warm-transfer the caller to a human (spec §6). Use for anything about an
    EXISTING booking (reschedule/cancel/refund/'SMS not received'), complaints,
    billing, custom/large-group or party quotes, membership terms beyond the card
    table, media/career/vendor calls, payment/receipt disputes, a failed
    confirm_booking, or anything you cannot answer from the knowledge base."""
    state = _puno_state(ctx)
    state.transferred_to_human = True
    state.call_outcome = "Transferred to human"
    state.follow_up_needed = True
    state.follow_up_note = (reason.strip() or state.follow_up_note or "human handover")
    state.call_category = state.call_category or reason.strip() or "handover"
    await _puno_persist(ctx)
    logger.info("PUNO transfer_to_human requested (reason=%s, user=%s)", reason, ctx.userdata.get("user_id"))
    return (
        "Handover flagged with the call context attached. Say the handover line — 'Let me get you to one of our "
        "team members who can sort this out properly — one moment.' — in your own words. If no one is available, "
        "offer to take a callback via create_lead. Do NOT attempt to resolve the underlying issue yourself." +
        _caller_language_reply_hint(state)
    )


# --- §6 System action: log_call_summary ------------------------------------
@function_tool
async def log_call_summary(
    ctx: RunContext,
    category: Annotated[str, "Call intent/category (general enquiry, booking, complaint, group/party, career/vendor/media, other)."] = "",
    outcome: Annotated[str, "One of: Booked, Lead created, Transferred to human, Info only, Not interested."] = "",
    discussed: Annotated[str, "Package or activity discussed."] = "",
    follow_up_needed: Annotated[bool, "Whether a human follow-up is needed."] = False,
    follow_up_note: Annotated[str, "Short follow-up note."] = "",
) -> str:
    """Write the end-of-call CRM record (spec §7). Call ONCE at the end of every
    call, every outcome. payment_status is always 'n/a' in this build."""
    state = _puno_state(ctx)
    if category.strip():
        state.call_category = category.strip()
    if outcome.strip():
        state.call_outcome = outcome.strip()
    if follow_up_needed:
        state.follow_up_needed = True
    if follow_up_note.strip():
        state.follow_up_note = follow_up_note.strip()

    record = {
        "name": state.patient_name or state.caller_name,
        "phone": state.patient_phone,
        "category": state.call_category or category.strip(),
        "discussed": discussed.strip() or state.package_name or state.reason,
        "visit_date": state.appointment_date,
        "time_slot": state.appointment_time,
        "adults_7plus": state.adults_7plus,
        "kids_under7": state.kids_under7,
        "booking_id": state.booking_id,
        "lead_id": state.lead_id,
        "outcome": state.call_outcome or outcome.strip() or "Info only",
        "payment_status": "n/a",
        "follow_up_needed": bool(state.follow_up_needed),
        "follow_up_note": state.follow_up_note or "",
    }
    try:
        await save_puno_call_log(record)
    except Exception as e:
        logger.error("log_call_summary failed: %s", e)
    # Internal — the model need not say anything to the caller about this.
    return "Call summary logged."


# ---- SPECULATIVE PREP (reason while the caller is still talking) ----------
# Gemini native-audio commits a user turn only after server-side VAD
# (END_SENSITIVITY_LOW + ~800ms silence). Tool calls therefore used to start
# only in on_user_turn_completed(), so every lookup paid the full VAD wait
# PLUS the tool round-trip. Interim `user_input_transcribed` events already
# arrive while they are speaking — we use those to warm hospital data, doctor
# details, slots, and language *without speaking*. Results are injected into
# chat_ctx when the turn actually commits. Never call generate_reply() from
# this path.
_SPEC_MIN_CHARS = 12
_SPEC_DEBOUNCE_S = 0.28
# Debounce before pushing language-switch instructions from interim ASR so a
# partial transcript does not trigger a websocket restart on every syllable.
_LANG_INSTR_INTERIM_DEBOUNCE_S = 0.2
_SPEC_TURN_WAIT_S = 0.25

_HOSPITAL_INFO_RE = re.compile(
    r"(address|location|where are you|kahan|kahaan|kaha hai|timing|timings|"
    r"\bopd\b|khula|opening hours?|consultation fee|visit fee|\bfees?\b|"
    r"insurance|mediclaim|\btpa\b|cashless|hospital (?:phone|number|contact)|"
    r"पता|कहाँ|कहां|टाइमिंग|फीस|इंश्योरेंस|मेडिक्लेम)",
    re.I,
)
_BOOK_RE = re.compile(
    r"(appoint(?:ment)?|\bbook\b|\bslot\b|consult|paediatrician|pediatrician|"
    r"\bdoctor\b|check[- ]?up|vaccin|अपॉइंटमेंट|बुकिंग|डॉक्टर|टीका)",
    re.I,
)
_LOOKUP_RE = re.compile(
    r"(cancel|reschedule|postpone|change (?:the )?appoint|already book|"
    r"existing appoint|pehle se|कैंसल|रद्द|रिशिड्यूल|पहले से)",
    re.I,
)
_DR_CUE_RE = re.compile(r"\b(dr\.?|doctor|डॉक्टर|डॉ)\b", re.I)
_HOSPITAL_NAME_RE = re.compile(
    r"sethi children|sethi hospital|सेठी (?:चिल्ड्रन|हॉस्पिटल|अस्पताल)",
    re.I,
)
_DOCTOR_ALIAS_RE = (
    (re.compile(r"\b(dileep|dilip|dilep|दलीप|दिलीप)\b", re.I), "dileep"),
    (re.compile(r"\b(chirag|चिराग)\b", re.I), "chirag"),
)

# Caller affirmatives after a booking summary ("haan kar do", "book kar dijiye", etc.).
# The realtime model often says "main book kar rahi hoon" but omits
# confirm_appointment(user_confirmed=True), leaving booking_confirmed_by_user False
# and trapping the call in a confirmation loop (debug H3).
_BOOKING_AFFIRM_RE = re.compile(
    r"(?:"
    r"haa?n+\s*(?:ji\s*)?(?:kar\s*(?:do|dijiye|dijie|de|dena)|book|confirm|theek|ok)|"
    r"ji\s*haa?n+|"
    r"(?<![a-z])ha+\s+kar\s*(?:do|dijiye|dijie|de|dena)|"
    r"kar\s*(?:do|dijiye|dijie|de|dena)\s*(?:book|appointment|confirm)?|"
    r"book\s*(?:kar|kr)\s*(?:do|dijiye|de)?|"
    r"confirm\s*(?:kar|kr|karo|kijiye)?|"
    r"(?:yes|ok|okay|please\s+book|go\s+ahead|bilkul|theek\s+hai|"
    r"appointment\s+book|meri\s+appointment|abhi\s+ke\s+liye|aaj\s+hi)"
    r")",
    re.I,
)


def _hydrate_booking_draft(state: AppointmentState) -> None:
    """PUNO build: no-op.

    The hospital version filled doctor/department/reason defaults (Dr. Dileep or
    Chirag Sethi, Pediatrics, "... consultation") whenever the model spoke details
    without persisting them. PUNO has no doctors or departments, so there is
    nothing to hydrate and defaulting any of it would only inject hospital data
    into a PUNO call. The five PUNO fields are captured explicitly via
    update_booking_details / confirm_booking.
    """
    return


def _booking_ready_for_confirm(state: AppointmentState) -> bool:
    return bool(
        state.patient_name
        and (state.patient_phone or state.phone_number)
        and state.department
        and (state.doctor_id or state.doctor_name)
        and state.appointment_date
        and state.appointment_time
        and state.reason
        and state.booking_status not in ("BOOKED", "confirmed", "BOOKING_IN_PROGRESS")
    )


class _ToolRunContext:
    """Minimal RunContext stand-in for server-side tool calls from the agent."""

    def __init__(self, userdata: dict):
        self.userdata = userdata
        self.room = userdata.get("room")


async def _resolve_turn_utterance(agent, new_message: ChatMessage) -> str:
    utterance = getattr(new_message, "text_content", "") or ""
    if callable(utterance):
        try:
            utterance = utterance() or ""
        except Exception:
            utterance = ""
    utterance = str(utterance).strip()
    if not utterance and isinstance(getattr(agent, "userdata", None), dict):
        utterance = (agent.userdata.get("last_final_transcript") or "").strip()
    if not utterance:
        utterance = (getattr(agent, "_last_user_transcript", "") or "").strip()
    return utterance


def _schedule_asr_auto_confirm(agent, utterance: str) -> None:
    """Book from the FINAL ASR transcript — runs after on_user_turn on native-audio."""
    if not utterance:
        return

    async def _run() -> None:
        try:
            chat_ctx = getattr(agent, "chat_ctx", None)
            booked = await _maybe_auto_confirm_booking(agent, chat_ctx, utterance)
            if booked:
                await _announce_auto_booking(agent)
        except Exception as exc:
            logger.warning("ASR auto-confirm task failed: %s", exc)

    try:
        task = asyncio.create_task(_run())
        bg = getattr(agent, "_bg_tasks", None)
        if isinstance(bg, set):
            bg.add(task)
            task.add_done_callback(bg.discard)
        _TOOL_BG_TASKS.add(task)
        task.add_done_callback(_TOOL_BG_TASKS.discard)
    except Exception as exc:
        logger.warning("Could not schedule ASR auto-confirm: %s", exc)


async def _announce_auto_booking(agent) -> None:
    """Speak booking confirmation when auto-book ran after the model turn started."""
    result = agent.userdata.pop("auto_booking_result", None)
    if not result or agent.state.booking_status not in ("BOOKED", "confirmed"):
        return
    session = agent.userdata.get("session")
    if session is None:
        return
    try:
        model = (agent.userdata.get("realtime_model") or "").lower()
        if "3.1" in model:
            return
        session.generate_reply(
            instructions=(
                "[AUTO-BOOKING COMPLETE — internal] "
                + result
                + " Tell the caller warmly, in the ACTIVE CONVERSATION LANGUAGE, that their "
                "appointment is now confirmed. One or two short sentences. Do NOT ask them "
                "to confirm again."
            ),
            allow_interruptions=True,
        )
    except Exception as exc:
        logger.warning("Post auto-book announcement failed: %s", exc)


async def _maybe_auto_confirm_booking(agent, chat_ctx: ChatContext | None, utterance: str) -> bool:
    """No-op. PUNO confirms bookings only through the model calling confirm_booking().

    The hospital build booked server-side the moment it heard a 'yes', by calling the
    old confirm_appointment tool from here. That tool and its whole flow have been
    removed; booking now goes through confirm_booking() after the agent reads the
    summary back (see the booking flow in puno_prompt.py). Kept as a stub because
    on_user_turn_completed still calls it.
    """
    return False

def _guess_spoken_date(text: str) -> Optional[str]:
    """Best-effort YYYY-MM-DD from a partial utterance. None if unclear."""
    t = (text or "").strip().lower()
    if not t:
        return None
    today = _today_ist()
    if re.search(r"\b(day after tomorrow|parso|parson|परसों)\b", t):
        return (today + datetime.timedelta(days=2)).strftime("%Y-%m-%d")
    if re.search(r"\b(tomorrow|कल)\b", t) or re.search(r"(?<![a-z])kal(?![a-z])", t):
        return (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    if t in ("aaj", "today") or "आज" in t or re.search(r"\b(today|aaj)\b", t):
        return today.strftime("%Y-%m-%d")

    m = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", t)
    if m:
        try:
            dt = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if dt >= today:
                return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    m = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", t)
    if m:
        try:
            dt = datetime.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            if dt >= today:
                return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # "18 तारीख" / "18 tareekh" with no month — current month, or next if that day passed.
    m = re.search(r"(?<!\d)(\d{1,2})\s*(?:तारी[खख़]|tareekh|tariikh)\b", t)
    if m:
        day = int(m.group(1))
        year, month = today.year, today.month
        for _ in range(2):
            try:
                dt = datetime.date(year, month, day)
                if dt >= today:
                    return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
            if month == 12:
                year, month = year + 1, 1
            else:
                month += 1


    months_map = {
        "january": 1, "jan": 1, "जनवरी": 1,
        "february": 2, "feb": 2, "फ़रवरी": 2, "फरवरी": 2,
        "march": 3, "mar": 3, "मार्च": 3,
        "april": 4, "apr": 4, "अप्रैल": 4,
        "may": 5, "मई": 5,
        "june": 6, "jun": 6, "जून": 6,
        "july": 7, "jul": 7, "जुलाई": 7,
        "august": 8, "aug": 8, "अगस्त": 8,
        "september": 9, "sep": 9, "सितंबर": 9, "सितम्बर": 9,
        "october": 10, "oct": 10, "अक्टूबर": 10,
        "november": 11, "nov": 11, "नवंबर": 11, "नवम्बर": 11,
        "december": 12, "dec": 12, "दिसंबर": 12, "दिसम्बर": 12,
    }
    month_alt = "|".join(sorted((re.escape(k) for k in months_map), key=len, reverse=True))
    m = re.search(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_alt})\b"
        rf"|\b({month_alt})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b",
        t,
    )
    if m:
        if m.group(1) and m.group(2):
            day, month = int(m.group(1)), months_map[m.group(2)]
        else:
            month, day = months_map[m.group(3)], int(m.group(4))
        year = today.year
        try:
            dt = datetime.date(year, month, day)
            if dt < today:
                dt = datetime.date(year + 1, month, day)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def _guess_spoken_time(text: str) -> Optional[str]:
    """Best-effort 'h:mm AM/PM' from a caller utterance. None if no clock time."""
    t = normalize_unicode_digits(text or "").strip()
    if not t:
        return None
    lower = t.lower()
    force_am = bool(re.search(r"सुबह|subah|\bmorning\b", lower))
    force_pm = bool(re.search(r"शाम|दोपहर|shaam|dopahar|\bevening\b|\bafternoon\b", lower))
    force_night = bool(re.search(r"रात|raat|\bnight\b|\bmidnight\b", lower))

    def _ampm_for_hour(hour: int, explicit: Optional[str]) -> Optional[str]:
        if explicit:
            return "PM" if explicit.upper().startswith("P") else "AM"
        if force_night:
            # "raat ke 12" = midnight; "raat ke 8" = 8 PM in Indian usage.
            if hour == 12:
                return "AM"
            return "PM"
        if force_pm:
            return "PM"
        if force_am:
            return "AM"
        # OPD is 9 AM–7 PM. Bare "11:00" is morning; "5:00" is afternoon.
        if 8 <= hour <= 11:
            return "AM"
        if hour == 12 or 1 <= hour <= 7:
            return "PM"
        return "AM"

    def _resolve(hour: int, minute: int, explicit: Optional[str]) -> Optional[str]:
        if not (1 <= hour <= 12 and 0 <= minute <= 59):
            return None
        ampm = _ampm_for_hour(hour, explicit)
        return normalize_time_slot(f"{hour}:{minute:02d} {ampm}")

    m = re.search(
        r"\b(\d{1,2})\s*[:\.]\s*(\d{2})\s*(a\.?m\.?|p\.?m\.?)?\b",
        t,
        re.I,
    )
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        resolved = _resolve(hour, minute, m.group(3))
        if resolved:
            return resolved
        if 13 <= hour <= 19 and 0 <= minute <= 59:
            return normalize_time_slot(f"{hour}:{minute:02d}")

    m = re.search(r"\b(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)\b", t, re.I)
    if m:
        hour = int(m.group(1))
        if 1 <= hour <= 12:
            return _resolve(hour, 0, m.group(2))

    m = re.search(r"\b(\d{1,2})\s*(?:बजे|baje)\b", t, re.I)
    if m:
        hour = int(m.group(1))
        if 1 <= hour <= 12:
            return _resolve(hour, 0, None)
    return None


_JUST_BOOKED_GUARD_S = 5.0


def _remember_user_utterance(userdata: dict, text: str) -> None:
    t = (text or "").strip()
    if not t or not isinstance(userdata, dict):
        return
    hist = userdata.get("_recent_user_utterances")
    if not isinstance(hist, list):
        hist = []
        userdata["_recent_user_utterances"] = hist
    if hist and hist[-1] == t:
        return
    hist.append(t)
    if len(hist) > 20:
        del hist[:-20]


def _apply_spoken_datetime(
    state: AppointmentState,
    text: str,
    *,
    overwrite: bool = True,
) -> bool:
    """Persist a date/time the caller just said. Returns True if state changed."""
    if not text or state is None:
        return False
    if getattr(state, "booking_status", "") in ("BOOKED", "confirmed"):
        return False
    changed = False
    spoken_date = _guess_spoken_date(text)
    spoken_time = _guess_spoken_time(text)
    if spoken_date and spoken_date != (state.appointment_date or ""):
        if overwrite or not state.appointment_date:
            state.appointment_date = spoken_date
            if isinstance(getattr(state, "step_attempts", None), dict):
                state.step_attempts["appointment_date"] = 0
            changed = True
            logger.info("Captured appointment_date=%s from utterance", spoken_date)
    if spoken_time:
        date_for_check = spoken_date or state.appointment_date or _today_ist().strftime("%Y-%m-%d")
        _rd, _rt, err = validate_and_normalize_appointment_datetime(date_for_check, spoken_time)
        if err and "OUTSIDE_OPD_HOURS" in err:
            logger.info(
                "Heard time %s on %s is outside OPD hours — not storing",
                spoken_time, date_for_check,
            )
        elif err and "PAST_TIME" in err:
            # Same-day "8:00" at 12:57 PM often means 8 PM, not 8 AM.
            alt_pm = None
            m = re.match(r"^(\d{1,2}):(\d{2})\s*(AM|PM)$", spoken_time.strip(), re.I)
            if m and m.group(3).upper() == "AM":
                h = int(m.group(1))
                if 6 <= h <= 11:
                    alt_pm = normalize_time_slot(f"{h}:{m.group(2)} PM")
            if alt_pm:
                _rd2, _rt2, err2 = validate_and_normalize_appointment_datetime(date_for_check, alt_pm)
                if not err2:
                    resolved = _rt2 or alt_pm
                    prev = normalize_time_slot(state.appointment_time or "")
                    if resolved and resolved != prev and (overwrite or not prev):
                        state.appointment_time = resolved
                        if isinstance(getattr(state, "step_attempts", None), dict):
                            state.step_attempts["appointment_time"] = 0
                        changed = True
                        logger.info(
                            "Captured appointment_time=%s from utterance (PM reinterpretation of %s)",
                            resolved, spoken_time,
                        )
                    if not state.appointment_date:
                        state.appointment_date = _rd2 or date_for_check
                        changed = True
                else:
                    if not state.appointment_time:
                        logger.info(
                            "Heard time %s on %s is in the past — storing date only",
                            spoken_time, date_for_check,
                        )
                    if not state.appointment_date:
                        state.appointment_date = date_for_check
                        changed = True
            else:
                if not state.appointment_time:
                    logger.info(
                        "Heard time %s on %s is in the past — storing date only",
                        spoken_time, date_for_check,
                    )
                if not state.appointment_date:
                    state.appointment_date = date_for_check
                    changed = True
        else:
            resolved = _rt or normalize_time_slot(spoken_time)
            prev = normalize_time_slot(state.appointment_time or "")
            if resolved and resolved != prev and (overwrite or not prev):
                state.appointment_time = resolved
                if isinstance(getattr(state, "step_attempts", None), dict):
                    state.step_attempts["appointment_time"] = 0
                changed = True
                logger.info("Captured appointment_time=%s from utterance", resolved)
            if not state.appointment_date:
                state.appointment_date = _rd or date_for_check
                changed = True
    return changed


def _gemini31_needs_lang_reinforce(agent) -> bool:
    """Gemini 3.1 Live ignores chat_ctx injection — push language via instructions."""
    try:
        ud = getattr(agent, "userdata", None) or {}
        model = (ud.get("realtime_model") or "") if isinstance(ud, dict) else ""
        if not model:
            model = getattr(agent, "_model_name", "") or getattr(agent, "model", "")
        if hasattr(agent, "_opts") and hasattr(agent._opts, "model"):
            model = agent._opts.model or model
        m = str(model).lower()
        return "gemini-3" in m or "3.1" in m
    except Exception:
        return False


class PunoVoiceAgent(Agent):
    def __init__(self, chat_ctx: ChatContext, state: AppointmentState, userdata: dict, instructions: str):
        self.state = state
        self.userdata = userdata
        # Strong references to fire-and-forget background tasks. asyncio only keeps
        # a WEAK reference to a task, so without this a per-turn task could be
        # garbage-collected mid-flight ("Task was destroyed but it is pending").
        self._bg_tasks: set[asyncio.Task] = set()
        # Latest caller transcript (interim OR final), fed by the session's
        # `user_input_transcribed` event (wired in entrypoint). The native-audio
        # realtime model often leaves ChatMessage.text_content empty on a user
        # turn, so this is the only reliable text to run language detection on.
        # Interims are stored too: they are what lets speculative prep start
        # before Gemini VAD commits the turn.
        self._last_user_transcript: str = ""
        self._spec_gen: int = 0
        self._spec_task: Optional[asyncio.Task] = None
        self._spec_source: str = ""
        self._spec_result: str = ""
        self._spec_last_kick: float = 0.0
        self._lang_instr_task: Optional[asyncio.Task] = None
        self._lang_reinforce_seq: int = 0
        # Seed the no-op guard so the first update_instructions() after pickup
        # does not restart the websocket (that restart used to swallow the greeting).
        self._last_applied_instructions = instructions
        super().__init__(
            instructions=instructions,
            chat_ctx=chat_ctx,
            tools=[
                # PUNO Jaipur (v2.3) — info tools + the four §6 system actions.
                get_puno_info,
                get_packages,
                get_activities,
                get_offers,
                get_membership,
                get_birthday_info,
                check_activity,
                update_booking_details,
                confirm_booking,
                create_lead,
                transfer_to_human,
                log_call_summary,
                end_call,
            ]
        )

    async def on_enter(self) -> None:
        """Speak the opener the instant this agent becomes active.

        LiveKit calls this after the realtime session is up. The cached greeting
        WAV is played immediately so the agent starts the conversation instead of
        waiting for the caller to speak first.
        """
        session = self.userdata.get("session") if isinstance(self.userdata, dict) else None
        if session is None:
            try:
                session = self.session
            except Exception:
                logger.warning("on_enter: no session yet, opener will come from _greet_asap")
                return
        if await _schedule_opening_greeting(session, self.userdata):
            logger.info("Opening greeting scheduled from on_enter")

    async def _background_update_instructions(self) -> None:
        """Update agent instructions in the background — never blocks the LLM response.

        CAUTION: update_instructions() triggers _mark_restart_needed() in the
        Gemini plugin, which RESETS the outgoing message channel and discards
        anything queued on it. Fired at the wrong moment it will throw away the
        reply the model is about to generate for the turn that just ended — the
        same failure that was swallowing the opening greeting.

        The small delay lets the reply for this turn get queued first, so the
        restart lands between turns rather than on top of one.
        """
        if _gemini31_needs_lang_reinforce(self):
            return
        try:
            await asyncio.sleep(0.5)
            await update_agent_instructions(self, self.state)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Error updating agent instructions (background): %s", e)

    def _cancel_language_instruction_task(self) -> None:
        task = self._lang_instr_task
        if task is not None and not task.done():
            task.cancel()
        self._lang_instr_task = None

    def _schedule_language_instruction_update(
        self, prev_lang: str | None, new_lang: str, *, reason: str = ""
    ) -> None:
        """Push updated instructions early when interim ASR detects a switch."""
        if _gemini31_needs_lang_reinforce(self):
            return
        self._cancel_language_instruction_task()

        async def _run() -> None:
            try:
                await asyncio.sleep(_LANG_INSTR_INTERIM_DEBOUNCE_S)
                applied = await update_agent_instructions(
                    self,
                    self.state,
                    force=True,
                    language_switch_from=prev_lang,
                )
                if applied:
                    logger.info(
                        "Interim language instructions applied: %s -> %s (%s)",
                        prev_lang or "unset",
                        new_lang,
                        reason or "interim",
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("Interim language instruction update failed: %s", e)

        task = asyncio.create_task(_run())
        self._lang_instr_task = task
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    def _inject_turn_message(self, chat_ctx: ChatContext, text: str) -> bool:
        """Append a system note to the context the model is about to read.

        VERIFY THE WRITE. An earlier version broke out of the loop on the first
        add_message() that did not raise. On the Gemini realtime plugin that
        call does not raise but also does not reach the live session, so the
        message silently went nowhere and the caller-visible behaviour never
        changed. Confirm the item count actually grew before believing it.
        """
        if getattr(chat_ctx, "_readonly", False) or getattr(chat_ctx, "readonly", False):
            return False
        for _kwargs in (
            {"role": "system", "content": [text]},
            {"role": "system", "content": text},
            {"role": "user", "content": [text]},
        ):
            try:
                _before = len(getattr(chat_ctx, "items", []) or [])
                chat_ctx.add_message(**_kwargs)
                _after = len(getattr(chat_ctx, "items", []) or [])
                if _after > _before:
                    return True
            except Exception:
                continue
        return False

    def note_partial_transcript(self, text: str, is_final: bool = False) -> None:
        """Start (or refresh) speculative lookups from live ASR.

        Safe to call on every `user_input_transcribed` event. Never speaks.
        Never mutates booking fields — only language lock and cached lookups.
        """
        text = (text or "").strip()
        if not text:
            return
        self._last_user_transcript = text

        try:
            _prev_lk = getattr(self.state, "caller_language", None)
            _lk, _why = resolve_caller_language(
                text, _prev_lk
            )
            if _lk and _lk != _prev_lk:
                logger.info(
                    "Caller language -> %s (speculative/%s: %r)",
                    _lk, "final" if is_final else "interim", text[:60],
                )
                self.state.caller_language = _lk
                self.state.caller_language_locked = True
                if _why in ("detected", "explicit_request"):
                    self._schedule_language_instruction_update(
                        _prev_lk, _lk, reason=_why
                    )
        except Exception:
            pass

        if len(text) < _SPEC_MIN_CHARS and not is_final:
            return
        now = time.monotonic()
        if not is_final:
            if text == self._spec_source:
                return
            if now - self._spec_last_kick < _SPEC_DEBOUNCE_S:
                return
        elif text == self._spec_source and self._spec_result:
            return

        self._spec_last_kick = now
        self._spec_source = text
        self._spec_gen += 1
        gen = self._spec_gen
        prev = self._spec_task
        if prev is not None and not prev.done():
            prev.cancel()
        task = asyncio.create_task(self._run_speculative_prep(text, gen))
        self._spec_task = task
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _run_speculative_prep(self, text: str, gen: int) -> None:
        try:
            notes = await self._build_speculative_notes(text)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("Speculative prep failed: %s", e)
            return
        if gen != self._spec_gen:
            return
        if notes:
            self._spec_result = notes
            logger.info("Speculative prep ready (%d chars) from: %r", len(notes), text[:80])

    async def _build_speculative_notes(self, text: str) -> str:
        """PUNO build: speculative prefetch is disabled.

        The hospital build pre-fetched doctor/department/slot lookups here and
        injected them as [PREFETCHED LOOKUPS]. PUNO's knowledge base is small and
        static and is baked directly into the system prompt (see puno_prompt.py),
        so there is nothing to prefetch — and returning nothing guarantees no
        hospital data can leak into a PUNO conversation.
        """
        return ""

    async def _speculative_context_for_turn(self) -> str:
        """Wait briefly for an in-flight prep, then return whatever is ready."""
        task = self._spec_task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=_SPEC_TURN_WAIT_S)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
        return self._spec_result or ""

    async def on_user_turn_completed(self, chat_ctx: ChatContext, new_message: ChatMessage) -> None:
        logger.info("on_user_turn_completed triggered. User message: %s", getattr(new_message, "text_content", ""))

        _turn_utterance = await _resolve_turn_utterance(self, new_message)

        # Persist date/time from this turn BEFORE auto-confirm, so "haan, book
        # kar do" can succeed after the caller already said "aaj 11:00" earlier.
        try:
            _remember_user_utterance(self.userdata, _turn_utterance)
            if _utterance_requests_caller_phone(_turn_utterance):
                if _apply_sip_caller_phone(self.state, self.userdata):
                    _hydrate_booking_draft(self.state)
            if not self.userdata.get("_auto_confirm_in_flight"):
                if _apply_spoken_datetime(self.state, _turn_utterance):
                    _hydrate_booking_draft(self.state)
        except Exception as _dt_err:
            logger.debug("Spoken datetime capture failed: %s", _dt_err)

        # ---- BOOKING AFFIRMATION (server-side — runs even if language block fails)
        try:
            await _maybe_auto_confirm_booking(self, chat_ctx, _turn_utterance)
            _pending = self.userdata.pop("auto_booking_result", None)
            if _pending:
                self._inject_turn_message(
                    chat_ctx,
                    "[AUTO-BOOKING RESULT — internal, do NOT read verbatim] "
                    + _pending
                    + " Tell the caller — in the ACTIVE CONVERSATION LANGUAGE — that the "
                    "appointment is confirmed. Do NOT ask them to confirm again.",
                )
        except Exception as _auto_err:
            logger.debug("Auto booking confirm failed: %s", _auto_err)

        _is_gemini31 = _gemini31_needs_lang_reinforce(self)

        # ---- LANGUAGE LOCK ------------------------------------------------
        # Classify THIS caller utterance and record it on the state, so the next
        # instruction rebuild pins the model to it.
        #
        # Two rules make this stable rather than jittery:
        #   1. Only the caller's transcript is ever inspected. The agent's own
        #      replies and tool results are never used as evidence — that
        #      feedback loop is what made the agent drift into Hindi.
        #   2. detect_language() returns None for utterances with no linguistic
        #      signal (a name, a phone number, "ok"). On None we KEEP the
        #      previous language instead of guessing, so spelling out a name
        #      digit by digit cannot silently flip the call.
        try:
            _utterance = getattr(new_message, "text_content", "") or ""
            if callable(_utterance):          # some plugin versions expose it as a method
                try:
                    _utterance = _utterance() or ""
                except Exception:
                    _utterance = ""
            _utterance = str(_utterance).strip()

            # NATIVE-AUDIO FALLBACK. gemini-live-*-native-audio consumes audio
            # directly and this session has no `stt=`, so new_message.text_content
            # is frequently EMPTY on a user turn. With an empty string
            # resolve_caller_language() can only ever return the previous value,
            # which is why the language never moved on its own and appeared to
            # switch only when the caller asked outright.
            #
            # The finalized transcript does arrive, just on the session's
            # `user_input_transcribed` event — which call_recording.py already
            # listens to. self._last_user_transcript is fed from that same event
            # (wired in entrypoint) and is used here when text_content is blank.
            if not _utterance:
                _utterance = (getattr(self, "_last_user_transcript", "") or "").strip()

            _prev_lang = getattr(self.state, "caller_language", None)
            _lang, _why = resolve_caller_language(_utterance, _prev_lang)

            logger.info(
                "LANG utterance=%r resolved=%r (%s) prev=%r",
                _utterance[:80], _lang, _why, _prev_lang,
            )

            _lang_applied = False
            if _lang:
                _switched = _lang != _prev_lang
                self.state.caller_language = _lang
                self.state.caller_language_locked = True

                if _is_gemini31:
                    if _switched:
                        logger.info(
                            "Caller language %s -> %s (%s) | utterance: %r",
                            _prev_lang or "unset", _lang, _why, _utterance[:60],
                        )
                        self._cancel_language_instruction_task()
                    # State only — update_instructions() restarts the websocket and
                    # kills the reply that activity_end is about to trigger.
                elif _switched:
                    logger.info(
                        "Caller language %s -> %s (%s) | utterance: %r",
                        _prev_lang or "unset", _lang, _why, _utterance[:60],
                    )
                    self._cancel_language_instruction_task()
                    _lang_applied = await update_agent_instructions(
                        self,
                        self.state,
                        force=True,
                        language_switch_from=_prev_lang,
                    )
                    if not _lang_applied:
                        logger.warning(
                            "Language switch %s -> %s: update_instructions did not apply",
                            _prev_lang or "unset",
                            _lang,
                        )

            # ---- APPLY IT TO *THIS* TURN, NOT THE NEXT ONE ----------------
            #
            # Gemini 3.1 Live has mutable_chat_context=False, so chat_ctx injection
            # often never reaches the live session. The synchronous
            # update_agent_instructions(force=True) above is the real fix on 3.1;
            # injection remains a best-effort fallback on models where it works.
            if _lang and not _is_gemini31:
                _directive = (
                    f"[LANGUAGE — applies to the reply you are about to give] "
                    f"The caller is speaking {language_label(_lang)}. Reply in "
                    f"{language_label(_lang)} now. This overrides the language of any "
                    f"earlier turn in this conversation and the language of any tool "
                    f"result you receive."
                )
                if _lang != _prev_lang:
                    _directive += (
                        " The caller JUST switched language — do not use the language "
                        "of your previous reply."
                    )
                if not self._inject_turn_message(chat_ctx, _directive):
                    if not _lang_applied and _gemini31_needs_lang_reinforce(self):
                        logger.warning(
                            "Same-turn language directive did not land in chat_ctx; "
                            "Gemini 3.1 relies on update_instructions for language."
                        )
                    elif _lang != _prev_lang and not _lang_applied:
                        logger.warning(
                            "Same-turn language directive did not land in chat_ctx and "
                            "update_instructions was not applied — reply may use wrong language."
                        )

            # ---- SAME-TURN STATE SNAPSHOT (anti re-asking) ----------------
            # BUG 2 FIX. This snapshot ("you already have name X, phone Y — do
            # NOT ask again") is injected into the chat_ctx that
            # on_user_turn_completed hands to the model for THIS turn's reply —
            # the supported pre-generation injection point. It used to be gated
            # behind `if not _is_gemini31`, so on the live 3.1 model (where
            # update_instructions is also disabled to avoid a websocket restart)
            # the model received NO per-turn reminder of captured fields and
            # re-asked for the name/phone it had just been given. Inject on all
            # models now; _inject_turn_message verifies the write and never raises.
            try:
                _snapshot = build_turn_snapshot(self.state)
                if _snapshot:
                    _ok = self._inject_turn_message(chat_ctx, _snapshot)
                    logger.info(
                        "[Bug2] turn snapshot injected=%s (gemini31=%s): %.90s",
                        _ok, _is_gemini31, _snapshot,
                    )
            except Exception as _se:
                logger.debug("Turn snapshot injection failed: %s", _se)
        except Exception as e:
            logger.warning("Language detection failed (continuing): %s", e)

        # ---- SPECULATIVE LOOKUPS (started on interim ASR, consumed here) --
        # Gemini VAD has only just committed the turn. Lookups that began while
        # the caller was still talking are injected so the model can answer
        # without waiting on a fresh tool round-trip. Never speaks on its own.
        try:
            _utt = ""
            try:
                _raw_utt = getattr(new_message, "text_content", "") or ""
                if callable(_raw_utt):
                    _raw_utt = _raw_utt() or ""
                _utt = str(_raw_utt).strip()
            except Exception:
                _utt = ""
            if not _utt:
                _utt = (getattr(self, "_last_user_transcript", "") or "").strip()
            if _utt:
                self.note_partial_transcript(_utt, is_final=True)
            _prefetch = await self._speculative_context_for_turn()
            if _prefetch:
                if not self._inject_turn_message(chat_ctx, _prefetch):
                    logger.warning("Prefetched lookups did not land in chat_ctx")
                self._spec_result = ""
        except Exception as _pe:
            logger.debug("Speculative context inject failed: %s", _pe)

        # Fire-and-forget instruction refresh so the LLM response starts IMMEDIATELY.
        # Keep a strong reference in self._bg_tasks (discarded on completion) so the
        # event loop doesn't GC a still-running task; also cancelled on disconnect.
        # (The hospital-data refresh that used to run here was removed with the rest
        # of the hospital build — PUNO's knowledge base is static, in puno_data.py.)
        for _coro in (self._background_update_instructions(),):
            _t = asyncio.create_task(_coro)
            self._bg_tasks.add(_t)
            _t.add_done_callback(self._bg_tasks.discard)

        await super().on_user_turn_completed(chat_ctx, new_message)


def _load_local_vad(*, min_silence_ms: int = 800):
    """Load Silero VAD for local turn-taking (Gemini server AAD is disabled).

    Prefers the Silero model bundled with livekit-agents (`inference.VAD`).
    Falls back to `livekit-plugins-silero` if the bundled model is missing.
    """
    min_silence = max(0.2, float(min_silence_ms) / 1000.0)
    try:
        from livekit.agents.inference import VAD as InferenceVAD

        vad = InferenceVAD(
            model="silero",
            min_silence_duration=min_silence,
            prefix_padding_duration=0.15,
            min_speech_duration=0.05,
        )
        logger.info("Loaded bundled Silero VAD, min_silence=%.2fs", min_silence)
        return vad
    except Exception as e:
        logger.info("inference.VAD unavailable (%s); trying livekit-plugins-silero", e)
    try:
        from livekit.plugins import silero

        try:
            vad = silero.VAD.load(
                min_silence_duration=min_silence,
                prefix_padding_duration=0.15,
                min_speech_duration=0.05,
            )
        except TypeError:
            vad = silero.VAD.load()
        logger.info("Loaded silero plugin VAD, min_silence=%.2fs", min_silence)
        return vad
    except Exception as e:
        logger.warning(
            "Could not load local VAD: %s — barge-in/turn-taking may be degraded", e
        )
        return None


def prewarm(proc: agents.JobProcess):
    from storage import _reset_redis_state
    _reset_redis_state()

    # GREETING-LATENCY: prewarm runs while the worker process is idle, BEFORE a
    # call is routed to it. Loading and normalising hospital_cache.json here
    # means the very first caller doesn't pay for it mid-call. Without this the
    # first call on a fresh worker is measurably slower than the rest, since
    # _DATA_CACHE is per-process and starts empty.

    try:
        silence_ms = int(os.environ.get("VAD_SILENCE_DURATION_MS", "800"))
        proc.userdata["vad"] = _load_local_vad(min_silence_ms=silence_ms)
        if proc.userdata.get("vad") is not None:
            logger.info("Prewarm: local VAD ready.")
    except Exception as e:
        logger.warning("Prewarm VAD failed: %s", e)

async def entrypoint(ctx: agents.JobContext):
    # GREETING-LATENCY INSTRUMENTATION. Every stage below logs how long it took
    # from job start, so "why did the agent take N seconds to speak" is
    # answerable from the logs of a real call instead of by guesswork.
    _t0 = time.monotonic()

    def _lap(stage: str) -> None:
        logger.info("[startup] %-28s %6.2fs", stage, time.monotonic() - _t0)

    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
    _lap("ctx.connect done")

    # The SIP/PSTN call is "picked up" once the remote caller is in the room.
    # Greet only after that — greeting audio generated against an empty room
    # never reaches the caller. If they are already present (typical inbound
    # SIP: the job is dispatched after they join) this returns immediately.
    caller_participant = None
    for p in ctx.room.remote_participants.values():
        if p.identity:
            caller_participant = p
            break
    if caller_participant is None:
        try:
            caller_participant = await asyncio.wait_for(
                ctx.wait_for_participant(), timeout=10.0
            )
            _lap("caller on the line")
        except Exception as e:
            logger.warning("No caller participant yet (%s); starting anyway.", e)

    user_id = caller_participant.identity if caller_participant else "anonymous"
    import uuid
    session_id = f"session_{uuid.uuid4().hex[:12]}"

    caller_phone = ""
    if caller_participant:
        identity = caller_participant.identity or ""
        if identity.startswith("sip:"):
            caller_phone = identity[4:]
        else:
            caller_phone = identity
        if "@" in caller_phone:
            caller_phone = caller_phone.split("@")[0]
        caller_phone = "".join(c for c in caller_phone if c.isdigit() or c == "+")
        if not caller_phone and hasattr(caller_participant, "attributes"):
            caller_phone = caller_participant.attributes.get("sip.phoneNumber", "")
            
    # Fallback to room name extraction
    if not caller_phone and ctx.room and ctx.room.name:
        matches = re.findall(r"\d{10,12}", ctx.room.name)
        if matches:
            caller_phone = matches[0]

    # Every new call/session is a fresh conversation. Start with a clean state.
    state = AppointmentState()

    # caller_phone is stored in userdata for logging/routing only.
    # The agent must always explicitly ask the patient for their contact number.
    # Do NOT pre-fill state.patient_phone from the caller ID.

    skip_queue = SkipQueue()
    from tools import CallTranscript
    transcript = CallTranscript(user_id=user_id)

    gcp_cred_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if gcp_cred_file and os.path.exists(gcp_cred_file):
        try:
            with open(gcp_cred_file, "r", encoding="utf-8") as _f:
                _data = json.load(_f)
                _file_proj = _data.get("project_id")
                if _file_proj:
                    os.environ["GOOGLE_CLOUD_PROJECT"] = _file_proj
                    os.environ["GCP_PROJECT_ID"] = _file_proj
        except Exception as _e:
            logger.warning("Could not read project_id from GOOGLE_APPLICATION_CREDENTIALS: %s", _e)

    gcp_project = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    gcp_location = os.environ.get("GCP_LOCATION") or os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1"
    google_api_key = _prefer_google_api_key()

    if gcp_project and "GOOGLE_CLOUD_PROJECT" not in os.environ:
        os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_project
    
    use_vertexai = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").strip().lower() == "true"
    
    # Using global _human_voice_instructions definition

    # ---- VOICE CONFIG (robotic-voice fix) --------------------------------
    # `voice` picks one of Gemini Live's 30 prebuilt HD voices. There is no
    # custom/cloned voice option on this model, so the levers that actually
    # move the needle are: (1) which prebuilt voice, (2) affective dialog,
    # (3) temperature, and (4) the delivery instructions in the prompt.
    #
    # Autonoe is characterised as "Bright" — energetic and announcer-like,
    # which reads as synthetic on a phone line. "Sulafat" (Warm) is the
    # closest match for a calm, human hospital receptionist speaking to a
    # worried parent. Alternatives worth A/B-ing on a real call:
    #   Sulafat     — Warm      (default here; best for reassurance)
    #   Achernar    — Soft      (gentler, quieter)
    #   Vindemiatrix— Gentle    (slower, very calm)
    #   Leda        — Youthful  (younger-sounding)
    # Override without a code change via GEMINI_VOICE in .env.
    voice_name = _configured_voice()
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-live-preview").strip() or "gemini-3.1-flash-live-preview"
    fallback_model = (
        os.environ.get("GEMINI_FALLBACK_MODEL", _GEMINI_FALLBACK_MODEL).strip()
        or _GEMINI_FALLBACK_MODEL
    )
    if fallback_model.lower() == model_name.lower():
        fallback_model = ""
    logger.info(
        "Gemini Live models: primary=%s fallback=%s",
        model_name,
        fallback_model or "(none)",
    )

    # _model_uses_gemini_api now lives at module scope (single source of truth).

    def _affective_for(name: str) -> bool:
        n = (name or "").lower()
        if "gemini-3" in n:
            return False
        env = os.environ.get("GEMINI_AFFECTIVE_DIALOG", "").strip().lower()
        if env == "true":
            return True
        if env == "false":
            return False
        return True

    # ---- TURN-TAKING / VAD ------------------------------------------------
    # Gemini 3.1 Live does not support generate_reply() and manual activity_end
    # does not reliably create LiveKit SpeechHandles — use server-side AAD with
    # turn_detection="realtime_llm" so _on_generation_created drives replies.
    # Fallback / 2.5 models keep local Silero VAD with Gemini AAD disabled.
    _silence_ms = int(os.environ.get("VAD_SILENCE_DURATION_MS", "800"))

    _aad_disabled_config = genai_types.RealtimeInputConfig(
        automatic_activity_detection=genai_types.AutomaticActivityDetection(
            disabled=True,
        ),
    )

    # ---- BUILD FINAL INSTRUCTIONS ONCE, BEFORE THE MODEL EXISTS -----------
    # GREETING-LATENCY FIX (the remaining ~3-4s of dead air).
    #
    # Previously the model was created with the BASE prompt, and then
    # _greet_asap() called update_agent_instructions() to fold in the live
    # state before greeting. That second call goes through the Gemini plugin's
    # _mark_restart_needed(), which tears down the realtime websocket and
    # re-sends the whole ~7.5k-token system prompt. So every single caller paid
    # for TWO full session setups back to back before hearing anything — the
    # no-op guard could never help, because _last_applied_instructions is unset
    # on a fresh call and the strings therefore always differ.
    #
    # Now the composed instructions are built up front and handed straight to
    # RealtimeModel, so the session is correct on its first and only connect.
    # _last_applied_instructions is seeded to the same string below, which makes
    # the pre-greeting update a genuine no-op that returns instantly.
    _initial_instructions = build_agent_instructions(state)

    def _build_realtime_model(name: str) -> RealtimeModel:
        use_v = False if _model_uses_gemini_api(name) else use_vertexai
        kwargs = dict(
            model=name,
            voice=voice_name,
            # Lowered from 0.7. On a native-audio model temperature drives delivery
            # as well as wording: higher values produce erratic pacing and odd
            # stress patterns that land as "off". 0.6 keeps her natural but steady,
            # and also makes tool-calling more reliable.
            temperature=0.6,
            instructions=_initial_instructions,
            enable_affective_dialog=_affective_for(name),
            api_key=google_api_key if not use_v else None,
            vertexai=use_v,
            project=gcp_project if use_v else None,
            location=gcp_location if use_v else None,
        )
        if not _is_gemini31_model(name):
            kwargs["realtime_input_config"] = _aad_disabled_config
        return RealtimeModel(**kwargs)

    def _turn_handling_for(name: str) -> dict:
        if _is_gemini31_model(name):
            return {
                "turn_detection": "realtime_llm",
                "interruption": {
                    "enabled": True,
                    # BUG 5 FIX — the old 0.45s / 1-word threshold let a brief
                    # echo of the agent's own voice or a background noise count as
                    # a barge-in and cut the agent off mid-sentence. Raise the
                    # floor (still env-overridable) so only real speech interrupts.
                    "min_duration": float(os.environ.get("INTERRUPT_MIN_DURATION", "0.6")),
                    "min_words": int(os.environ.get("INTERRUPT_MIN_WORDS", "2")),
                    # If an interruption turns out to be false (no real caller
                    # turn follows), resume the agent's sentence instead of
                    # leaving it stopped dead ("abruptly stops mid-response").
                    "resume_false_interruption": (
                        os.environ.get("RESUME_FALSE_INTERRUPTION", "true").strip().lower() != "false"
                    ),
                    "false_interruption_timeout": float(
                        os.environ.get("FALSE_INTERRUPTION_TIMEOUT_S", "2.0")
                    ),
                },
            }
        return {
            "turn_detection": "vad",
            "endpointing": {
                "min_delay": 1.2,
                "max_delay": 4.0,
            },
            "interruption": {
                "enabled": True,
                "mode": "vad",
                "min_duration": float(os.environ.get("INTERRUPT_MIN_DURATION", "0.45")),
                "min_words": int(os.environ.get("INTERRUPT_MIN_WORDS", "1")),
            },
        }

    def _build_agent_session(llm, vad=None, *, model_name_for_turn: str = "") -> AgentSession:
        mname = model_name_for_turn or model_name
        turn_handling = _turn_handling_for(mname)
        kwargs = dict(
            llm=llm,
            # NOTE: NO `tts=` here — and that is deliberate.
            #
            # ROBOTIC-VOICE FIX: this used to pass tts=google.TTS(hi-IN-Chirp3-HD-Aoede),
            # which is a *separate* Cloud Text-to-Speech engine stacked on top of a
            # native-audio realtime model. gemini-live-2.5-flash-native-audio already
            # generates speech itself, with prosody, emotion and breath baked into the
            # same pass as the words. Bolting a classic TTS on top threw all of that
            # away and re-synthesised flat text — which is exactly what made her sound
            # robotic and monotone. It also forced every reply through a second
            # synthesis hop (extra latency) and pinned output to hi-IN, fighting the
            # model's own natural Hindi/English/Hinglish switching.
            #
            # Removing it lets the native audio through untouched. Do NOT re-add a
            # `tts=` parameter to this session.
            userdata=_userdata,
            turn_handling=turn_handling,
        )
        if vad is not None and not _is_gemini31_model(mname):
            kwargs["vad"] = vad
        return AgentSession(**kwargs)

    realtime_model = _build_realtime_model(model_name)

    storage_user_id = user_id if (user_id and user_id != "anonymous") else session_id
    _userdata: dict = {
        "user_id": storage_user_id,
        "raw_user_id": user_id,
        "session_id": session_id,
        "state": state,
        "skip_queue": skip_queue,
        "transcript": transcript,
        "caller_phone": caller_phone,
        "room": ctx.room,
        "realtime_model": model_name,
        "is_agent_speaking": False,
        # GROUND-TRUTH agent-audio signals (Bug 1). Stamped by the speech_created
        # event and the recorder's agent-audio frame loop — independent of the
        # agent_state_changed event, which lags on Gemini 3.1 native audio.
        "last_agent_audio_monotonic": 0.0,
        "last_agent_speech_monotonic": 0.0,
    }

    _local_vad = None
    try:
        _local_vad = ctx.proc.userdata.get("vad")
    except Exception:
        _local_vad = None
    if _local_vad is None:
        _local_vad = _load_local_vad(min_silence_ms=_silence_ms)
    if _local_vad is None and not _is_gemini31_model(model_name):
        logger.warning(
            "No local VAD loaded — Gemini AAD is disabled, so turn-taking may not work"
        )

    session = _build_agent_session(
        realtime_model, vad=_local_vad, model_name_for_turn=model_name
    )
    _userdata["session"] = session
    _lap("session constructed")

    from call_recording import CallSessionRecorder, derive_call_id
    call_id = derive_call_id(ctx.room, user_id)
    _userdata["call_id"] = call_id  # lead finalizer reads the transcript store by this id

    # BACKSTOP: guarantees a lead for endings that never reach the
    # participant_disconnected handler (silence hangup, end_call, session error).
    # finalize_lead_for_call is idempotent, so whichever path runs first wins.
    #
    # REGISTRATION ORDER MATTERS. ctx shutdown callbacks run in registration order,
    # and the recorder's own _on_shutdown ends by calling _store_clear(), which wipes
    # the transcript this finalizer reads. So this MUST be registered before
    # recorder.attach() — otherwise, on any ending that reaches ctx.shutdown()
    # without passing through one of the explicit pre-teardown calls, the backstop
    # would run against an already-cleared store and save an empty lead.
    async def _lead_shutdown_backstop(*_a, **_kw):
        try:
            await finalize_lead_for_call(_userdata, reason="shutdown")
        except Exception as _e:
            logger.error("[Lead] shutdown backstop failed: %s", _e)

    try:
        ctx.add_shutdown_callback(_lead_shutdown_backstop)
    except Exception as _e:
        logger.warning("Could not register lead shutdown backstop: %s", _e)

    recorder = CallSessionRecorder(ctx, session, call_id, user_id, transcript)
    recorder.attach()

    # Local Silero VAD is attached so `user_state_changed -> "speaking"` fires.
    # `user_input_transcribed` remains the backup caller-speech signal for the
    # silence monitor (native-audio transcripts can lag VAD start/stop).
    _last_user_txt_at = 0.0      # monotonic: last caller transcript of any kind
    _last_agent_done_at = 0.0    # monotonic: when the agent last STOPPED speaking
    _last_checkin_at = 0.0       # monotonic: last silence check-in we fired
    _checkin_in_flight = False   # True while a silence presence-check is being spoken
    last_speech_time = time.monotonic()   # last time someone actually SPOKE (audio)
    last_activity_time = time.monotonic()  # last time any state change happened
    silence_count = 0
    agent_asked_this_turn = False
    thinking_since = 0.0                   # when agent entered "thinking" state (0 = not thinking)

    # Local copies of user/agent state — updated by events.
    # The silence monitor reads these instead of calling session.user_state /
    # session.agent_state directly, which are NOT properties on AgentSession and
    # throw AttributeError, causing the broad except-break to silently kill the
    # monitor on the very first loop iteration.
    _current_u_state = "listening"
    _current_a_state = "listening"

    @session.on("user_state_changed")
    def on_user_state(event):
        nonlocal last_speech_time, last_activity_time, silence_count, agent_asked_this_turn, thinking_since, _current_u_state
        logger.info(f"User state changed: {event.old_state} -> {event.new_state}")
        last_activity_time = time.monotonic()
        _current_u_state = event.new_state
        if event.new_state == "speaking":
            last_speech_time = time.monotonic()
            silence_count = 0
            agent_asked_this_turn = False
            thinking_since = 0.0
        elif event.old_state == "speaking":
            # User just STOPPED speaking — silence starts NOW, not earlier.
            last_speech_time = time.monotonic()

    @session.on("agent_state_changed")
    def on_agent_state(event):
        nonlocal last_speech_time, last_activity_time, silence_count, agent_asked_this_turn, thinking_since, _current_a_state
        nonlocal _last_agent_done_at, _checkin_in_flight
        logger.info(f"Agent state changed: {event.old_state} -> {event.new_state}")
        last_activity_time = time.monotonic()
        _current_a_state = event.new_state
        if event.new_state == "speaking":
            _userdata["is_agent_speaking"] = True
            # Agent is producing audio — real speech, reset the silence timer.
            # Do NOT zero silence_count here: a presence-check is agent speech,
            # and wiping the count on every check-in made hangup-after-N-ignored
            # check-ins unreachable.
            last_speech_time = time.monotonic()
            thinking_since = 0.0
            # The cached opener is the first speaking turn and is not a
            # booking question. Counting it as appointment_date ask #1 made
            # the agent re-ask the date immediately after greeting.
            if not _userdata.get("opener_counted"):
                _userdata["opener_counted"] = True
                agent_asked_this_turn = True
                return
            if not _checkin_in_flight and not agent_asked_this_turn:
                if getattr(state, "doctor_name", None) and not getattr(state, "department", None):
                    _hydrate_booking_draft(state)
                current_slot = state.current_step()
                # BUGFIX: this used to count an "ask" on every single agent
                # speaking turn, including turns where the agent was answering
                # an unrelated question. Two such turns pushed a still-missing
                # mandatory field into skipped_fields and current_step() then
                # skipped it forever — the agent could reach booking with no
                # patient name. Only count when the field is genuinely still
                # unset, and never let a mandatory field be auto-skipped.
                if current_slot not in ("complete", "confirmation"):
                    _age_val = getattr(state, "age", None)
                    _age_num = _extract_age_years(_age_val) if _age_val else None
                    _is_ineligible = _age_num is not None and _age_num > 10
                    _booking_active = getattr(state, "booking_intent_detected", False)
                    if _booking_active and not _is_ineligible and not getattr(state, current_slot, None):
                        state.increment_ask(current_slot)
                        logger.info(
                            "Ask count for %s: %d",
                            current_slot, state.get_ask_count(current_slot),
                        )
                agent_asked_this_turn = True
        elif event.old_state == "speaking":
            _userdata["is_agent_speaking"] = False
            # Agent just STOPPED speaking — silence starts NOW.
            # Also MUST reset thinking_since here so that the monitor's
            # thinking check (thinking_since > 0) doesn't permanently
            # block the silence check in subsequent idle periods.
            last_speech_time = time.monotonic()
            _last_agent_done_at = time.monotonic()
            thinking_since = 0.0
            _checkin_in_flight = False

            # If wrapup was signalled by caller and agent finished speaking, ensure clean end_call disconnect
            if getattr(state, "call_wrapup_ok", False) and not _userdata.get("_drop_in_progress"):
                logger.info("Wrap-up completed by agent speaking turn — scheduling clean end_call disconnect.")
                asyncio.create_task(_execute_end_call_drop(_userdata))
        elif event.new_state == "thinking":
            _userdata["is_agent_speaking"] = False
            # Agent is processing but NOT producing audio.
            if thinking_since == 0.0:
                thinking_since = time.monotonic()
        else:
            _userdata["is_agent_speaking"] = False
            # Agent went to idle/listening — no longer thinking
            thinking_since = 0.0

    # ---- BUG 1: ground-truth "agent is speaking" signal --------------------
    # agent_state_changed lags on Gemini 3.1 native audio, so the silence monitor
    # could fire a presence-check over the agent's own voice. speech_created fires
    # the instant the agent produces a speech handle — a reliable, event-order-
    # independent stamp the silence gate checks before ever speaking into a gap.
    def on_speech_created(event) -> None:
        nonlocal last_speech_time
        try:
            _t_sp = time.monotonic()
            _userdata["last_agent_speech_monotonic"] = _t_sp
            # last_agent_audio_monotonic was initialised to 0.0 and then never
            # written by anything, so `agent_audio_age` was permanently
            # (now - 0.0) — an enormous number that could never fall below the
            # guard. The "ground truth agent audio" check that Bug 1 added was
            # therefore inert from the day it was written. speech_created is a
            # genuine audio-path signal (it fires for session.say() too, and is
            # independent of the agent_state_changed event Gemini 3.1 drops),
            # so stamping it here is what makes that guard real.
            _userdata["last_agent_audio_monotonic"] = _t_sp
            last_speech_time = _t_sp
        except Exception as _e:
            logger.debug("on_speech_created stamp failed: %s", _e)

    # ---- BUG 5: surface session errors that used to fail silently ----------
    def on_session_error(event) -> None:
        try:
            err = getattr(event, "error", event)
            recoverable = getattr(err, "recoverable", None)
            logger.error(
                "[Bug5] AgentSession error (recoverable=%s): %s", recoverable, err
            )
        except Exception as _e:
            logger.error("[Bug5] AgentSession error (unparseable): %s (%s)", event, _e)

    def on_session_close(event) -> None:
        try:
            err = getattr(event, "error", None)
            reason = getattr(event, "reason", None)
            if err is not None:
                logger.error("[Bug5] AgentSession closed with error (reason=%s): %s", reason, err)
            else:
                logger.info("AgentSession closed (reason=%s)", reason)
        except Exception as _e:
            logger.warning("[Bug5] session close handler failed: %s", _e)

    def on_agent_false_interruption(event) -> None:
        # The agent was interrupted but no real caller turn followed (echo/noise).
        # With resume_false_interruption enabled the SDK auto-resumes; log both
        # cases so a genuine mid-response cut-off ("abruptly stops") is diagnosable.
        nonlocal last_speech_time
        try:
            resumed = getattr(event, "resumed", None)
            if resumed:
                logger.info("[Bug5] false interruption auto-resumed")
            else:
                logger.warning(
                    "[Bug5] false interruption NOT resumed — agent may have stopped "
                    "mid-response; caller silence check-in will re-engage if needed."
                )
                last_speech_time = time.monotonic()
        except Exception as _e:
            logger.debug("false-interruption handler failed: %s", _e)

    # ---- BUG 4: runtime backstop against a hallucinated payment-link offer --
    # There is NO payment channel in this build and no code path produces such a
    # line — the prompt forbids it — but a realtime LLM can still hallucinate one.
    # Watch the agent's own transcript; if it ever offers online payment / a link,
    # log it loudly and immediately speak a correction so the caller is not left
    # with wrong information.
    _payment_guard_fired = {"last": 0.0}
    # Cancellation promises get their own debounce so a payment correction and a
    # cancellation correction can never suppress one another.
    _cancel_guard_fired = {"last": 0.0}
    # Last substantive agent turn, for detecting the model repeating itself.
    _last_agent_turn = {"text": ""}

    def on_conversation_item_output_guard(event) -> None:
        try:
            item = getattr(event, "item", None) or event
            role = getattr(item, "role", None)
            if role not in ("assistant", "agent"):
                return
            text = getattr(item, "text_content", None)
            if callable(text):
                text = text()
            if not text:
                content = getattr(item, "content", None)
                if isinstance(content, (list, tuple)):
                    text = " ".join(str(c) for c in content if isinstance(c, str))
                elif isinstance(content, str):
                    text = content
            text = (text or "").strip()
            if not text:
                return

            # ISSUE 1 — Hindi first-person gender drift. The TTS voice is fixed
            # (AGENT_GENDER, default female) but Hindi inflects the first person,
            # so masculine self-reference in a female voice reads as a different
            # persona. The prompt mandates the correct forms; this makes any
            # remaining drift visible instead of silent.
            _expected = (os.environ.get("AGENT_GENDER") or "female").strip().lower()
            if gender_drifted(text, _expected):
                logger.warning(
                    "[Issue1] Hindi self-reference gender drift: found=%s expected=%s "
                    "voice=%s | text=%r",
                    hindi_self_gender(text), _expected,
                    _configured_voice(), text[:160],
                )

            # ISSUE 2 — the model itself must never ask about audibility.
            if mentions_audibility_question(text):
                logger.warning(
                    "[Issue2] Agent generated an audibility/presence question: %r", text[:160]
                )

            # ISSUE 3 — instruction leakage. Observed live: the agent recited its
            # own call-flow scaffolding, step number included. The prompt no longer
            # contains numbered steps to read out; this reports any recurrence and
            # names the exact fragment so the next prompt fix can be targeted.
            _leak = leaked_instruction(text)
            if _leak:
                logger.error(
                    "[Issue3] Agent recited its own instructions: fragment=%r | text=%r",
                    _leak, text[:200],
                )

            # ISSUE 4 — the model repeating its own previous turn verbatim.
            # FixedLineDedup only guards speak_fixed_line(), so model-generated
            # repeats were invisible until a caller complained about them.
            _prev = _last_agent_turn["text"]
            if _prev and is_near_duplicate(_prev, text):
                logger.warning(
                    "[Issue4] Agent repeated its own previous turn: %r", text[:160]
                )
            _last_agent_turn["text"] = text

            # ISSUE 5 — promising to cancel / reschedule / refund a booking.
            # Out of scope in this build and backed by no tool, so the promise is
            # always false. Observed live on a booking that did not even exist.
            # Correct it out loud: a caller who believes their booking is cancelled
            # acts on that, which is the most damaging thing this agent can get wrong.
            if promises_booking_modification(text):
                logger.error(
                    "[Issue5] Agent promised to cancel/reschedule/refund a booking "
                    "(booking_id=%r) — speaking correction: %r",
                    getattr(state, "booking_id", None), text[:160],
                )
                _now_c = time.monotonic()
                if _now_c - _cancel_guard_fired["last"] >= 8.0:
                    _cancel_guard_fired["last"] = _now_c
                    speak_fixed_line(
                        session,
                        booking_modification_correction_line(
                            getattr(state, "caller_language", None)
                        ),
                        allow_interruptions=True,
                        add_to_chat_ctx=False,
                    )
                return

            if not mentions_payment_link(text):
                return
            logger.error(
                "[Bug4] Agent produced a forbidden payment-link/online-payment line: %r "
                "— speaking correction.", text[:120],
            )
            # Debounce corrections so we never stack them.
            _now = time.monotonic()
            if _now - _payment_guard_fired["last"] < 8.0:
                return
            _payment_guard_fired["last"] = _now
            _lang = getattr(state, "caller_language", None)
            speak_fixed_line(
                session,
                payment_correction_line(_lang),
                allow_interruptions=True,
                add_to_chat_ctx=False,
            )
        except Exception as _e:
            logger.debug("output guard failed: %s", _e)

    for _evt, _fn in (
        ("speech_created", on_speech_created),
        ("error", on_session_error),
        ("close", on_session_close),
        ("agent_false_interruption", on_agent_false_interruption),
        ("conversation_item_added", on_conversation_item_output_guard),
    ):
        try:
            session.on(_evt, _fn)
        except Exception as _e:
            logger.debug("could not bind %s handler: %s", _evt, _e)

    async def silence_monitor():
        nonlocal last_speech_time, last_activity_time, silence_count, thinking_since, _current_u_state, _current_a_state
        nonlocal _last_checkin_at, _checkin_in_flight

        # Silence intervals: 12.0s first check-in, 10.0s followup (gives caller room to think)
        _SILENCE_TIMEOUT = float(os.environ.get("SILENCE_FIRST_CHECKIN_S", "12.0"))
        _FOLLOWUP_TIMEOUT = float(os.environ.get("SILENCE_FOLLOWUP_S", "10.0"))
        _THINKING_FILLER_TIMEOUT = float(os.environ.get("THINKING_FILLER_S", "3.5"))
        # Hard cap on thinking fillers per call. "एक सेकंड" appeared many times in
        # the reported transcript; after a couple it reads as a stutter.
        _MAX_THINKING_FILLERS = int(os.environ.get("MAX_THINKING_FILLERS", "2"))
        # BUG 1: never fire a presence-check within this many seconds of the agent
        # last producing audio (ground-truth), regardless of the lagging state event.
        _AGENT_AUDIO_GUARD_S = float(os.environ.get("SILENCE_AGENT_AUDIO_GUARD_S", "2.5"))
        # BUG 5: if the agent sits in "thinking" (generating) this long with no
        # audio, the turn is almost certainly dead — log it loudly and recover.
        _THINKING_STUCK_S = float(os.environ.get("THINKING_STUCK_S", "20.0"))
        _stuck_thinking_logged = False
        _stuck_recovery_pending = False

        # TURN-OWED WATCHDOG. The caller answered a question and the agent never
        # replied — 55s of dead air on the reported call while the caller said
        # "hello?" four times. _THINKING_STUCK_S above cannot catch this: it is
        # nested under `a == "thinking"`, and Gemini 3.1 drops that event, so the
        # recovery written for exactly this failure never armed. This one reads
        # ONLY ground truth (caller transcript vs agent audio), never session state.
        _TURN_OWED_S = float(os.environ.get("TURN_OWED_TIMEOUT_S", "7.0"))
        _MAX_STALL_RECOVERIES = int(os.environ.get("MAX_STALL_RECOVERIES", "3"))
        _stall_recoveries = 0
        _last_stall_recovery_at = 0.0

        # R6 (NEVER END THE CALL AUTOMATICALLY). This is the ONE remaining path
        # by which the agent can hang up without the caller asking, and it exists
        # only for a genuinely dead line (caller walked away, one-way audio,
        # carrier fault) — otherwise the room stays open and billable forever.
        # It now takes 3 ignored check-ins instead of 2, and can be switched off
        # entirely with SILENCE_HANGUP_ENABLED=false, in which case the agent
        # keeps checking in and waits for the carrier to tear the call down.
        _MAX_CHECKINS = int(os.environ.get("SILENCE_MAX_CHECKINS", "3"))
        _HANGUP_ON_SILENCE = (
            os.environ.get("SILENCE_HANGUP_ENABLED", "true").strip().lower() != "false"
        )

        # LANGUAGE FIX: these were hardcoded Hinglish. Spoken into an English
        # call they both sounded wrong AND re-seeded Hindi into the model's own
        # context, which then dragged the rest of the call into Hindi. They are
        # now selected from the live detected caller language.
        _FILLERS_BY_LANG = {
            "en": ["One second...", "Yes, let me check...", "Just checking..."],
            "hi": ["Ek second...", "Haan ji, check kar rahi hoon...", "Dekhti hoon..."],
            "hinglish": ["Ek second...", "Haan ji, check kar rahi hoon...", "Dekhti hoon..."],
        }
        _GOODBYE_BY_LANG = {
            "en": (
                "There is no response from your side, so I am cutting the call. "
                "Thanks for calling PUNO Jaipur. Have a great day!"
            ),
            "hi": (
                "Aapki taraf se koi response nahi mil raha hai, isliye main call cut kar rahi hoon. "
                "PUNO Jaipur mein call karne ke liye dhanyavaad, aapka din shubh ho."
            ),
            "hinglish": (
                "Aapki taraf se koi response nahi mil raha hai, isliye main call cut kar rahi hoon. "
                "Thanks for calling PUNO Jaipur. Have a great day!"
            ),
        }
        # ISSUE 2 FIX — the hardcoded presence-check table used to live here and
        # asked the caller whether they could hear the agent (an audibility
        # question, in EN/HI). That is now forbidden outright. Silence
        # re-engagement lines come from call_guardrails.reengagement_line(), whose
        # first attempt is an EMPTY string (stay silent and wait) and whose later
        # attempts are natural, non-audibility invitations. Every line is screened
        # through mentions_audibility_question() before it can be spoken.

        def _lang_now() -> str:
            lang = getattr(state, "caller_language", None)
            if lang in ("en", "hi", "hinglish"):
                return lang
            # ISSUE 2: this used to default to "hi", so an English caller whose
            # language had not been detected yet was spoken to in HINDI (and the
            # "en" default in every .get() below was unreachable). The fixed
            # opener is English, so English is the correct undetected default.
            return "en"

        def _thinking_filler(idx: int) -> str:
            # BUGD: default to the ENGLISH list, not the Hindi one. `_lang_now()`
            # already falls back to "en", so a Hindi default here only ever
            # applied when caller_language held something unexpected — and a
            # stray Hindi filler spoken to an English caller drags the model's
            # own context back to Hindi, which is exactly what happened on the
            # reported call ("एक सेकंड मैं चेक कर रही हूं" to an English caller).
            opts = _FILLERS_BY_LANG.get(_lang_now(), _FILLERS_BY_LANG["en"])
            return opts[idx % len(opts)]

        _filler_idx = 0

        # Guard: don't count silence until the agent has completed its FIRST
        # speaking turn (i.e., the greeting). Before that, the model may take
        # several seconds to generate the opening line, which is NOT user silence.
        _greeting_delivered = False

        async def _hangup_due_to_silence():
            logger.info("Caller unresponsive after silence check-ins — ending call immediately.")
            _disconnect_event.set()
            # Same ordering rule as end_call: the lead must be built while the
            # transcript store still exists (session.aclose() below clears it).
            try:
                await finalize_lead_for_call(_userdata, reason="silence_hangup")
            except Exception as _le:
                logger.error("[Lead] finalize on silence hangup failed: %s", _le)
            for task in list(_background_tasks):
                if task is not asyncio.current_task() and not task.done():
                    task.cancel()
            try:
                await session.aclose()
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
            try:
                _maybe = ctx.room.disconnect()
                if inspect.isawaitable(_maybe):
                    await _maybe
            except Exception as e:
                logger.warning(f"Error disconnecting room: {e}")
            try:
                from livekit import api
                api_client = api.LiveKitAPI()
                try:
                    await api_client.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
                finally:
                    await api_client.aclose()
            except Exception as e:
                logger.warning(f"Error deleting room on silence hangup: {e}")
            try:
                ctx.shutdown("caller_unresponsive")
            except Exception as e:
                logger.warning(f"Error shutting down: {e}")

        _diag_tick = 0

        try:
            while True:
                await asyncio.sleep(0.5)

                if _disconnect_event.is_set():
                    logger.info("Silence monitor: stopping (disconnect).")
                    break

                u = _current_u_state
                a = _current_a_state
                elapsed = time.monotonic() - last_speech_time

                # --- Diagnostic log every 5 seconds ---
                _diag_tick += 1
                if _diag_tick % 10 == 0:
                    logger.info(
                        f"[SilenceMonitor] u={u} a={a} elapsed={elapsed:.1f}s "
                        f"silence_count={silence_count} thinking_since={thinking_since:.1f}"
                    )

                # ---- 1. User speaking — reset timer --------------------------
                if u == "speaking":
                    last_speech_time = time.monotonic()
                    silence_count = 0
                    continue

                # ---- 1b. TURN-OWED WATCHDOG — the agent stalled ---------------
                # The caller spoke, the agent owes a reply, and none arrived.
                #
                # Why this cannot reuse the machinery below: should_fire_checkin()
                # returns "unanswered_user_turn" precisely BECAUSE a reply is owed,
                # and only force_recovery clears that — which only the thinking
                # watchdog sets, which only arms on an `a == "thinking"` event that
                # Gemini 3.1 drops. On the reported call the gate stayed blocked
                # indefinitely: replaying its real inputs, it still returns
                # "unanswered_user_turn" an hour in. And because `user_txt_age`
                # also gates it, every "hello?" the caller tried re-armed the block.
                # The caller's own attempts to get attention kept the agent mute.
                #
                # So this reads ONLY ground truth — the caller's last transcript
                # against the agent's last actual AUDIO. It deliberately consults
                # neither agent_state nor _last_agent_done_at, both of which are
                # derived from the event that goes missing.
                _now_w = time.monotonic()
                # Take the NEWEST of every "the agent produced output" signal.
                # max() is the safe direction here: a missing signal can only make
                # the agent look MORE recently active, so a dropped event yields a
                # late recovery, never a spurious one spoken over a healthy call.
                _agent_audio_at = max(
                    float(_userdata.get("last_agent_audio_monotonic") or 0.0),
                    float(_userdata.get("last_agent_speech_monotonic") or 0.0),
                    _last_agent_done_at,
                )
                if (
                    _greeting_delivered
                    and _last_user_txt_at > 0.0
                    and _last_user_txt_at > _agent_audio_at        # a reply is owed
                    and (_now_w - _last_user_txt_at) >= _TURN_OWED_S
                    and not _userdata.get("is_agent_speaking")     # ground-truth audio flag
                    and (_now_w - _agent_audio_at) >= _AGENT_AUDIO_GUARD_S
                    and (_now_w - _last_stall_recovery_at) >= _TURN_OWED_S
                    and not _userdata.get("_drop_in_progress")
                ):
                    _stall_recoveries += 1
                    _last_stall_recovery_at = _now_w
                    logger.critical(
                        "[Stall] Agent owed the caller a reply for %.1fs and produced no audio "
                        "— recovering (%d/%d). u_state=%s a_state=%s thinking_since=%.1f "
                        "agent_audio_age=%.1fs model=%s",
                        _now_w - _last_user_txt_at, _stall_recoveries, _MAX_STALL_RECOVERIES,
                        u, a, thinking_since, _now_w - _agent_audio_at,
                        _userdata.get("realtime_model"),
                    )
                    # Unblock the normal path too, so if this recovery is itself
                    # swallowed the check-in ladder below is no longer dead-locked.
                    _stuck_recovery_pending = True
                    thinking_since = 0.0

                    if _stall_recoveries > _MAX_STALL_RECOVERIES:
                        # Past the budget, stop TALKING but keep the block armed.
                        # Repeating "let me get you to a team member" every few
                        # seconds is worse for the caller than the silence it is
                        # trying to fix. Do NOT invent a second hang-up path
                        # (see R6 below) — the check-in ladder owns the outcome.
                        logger.critical(
                            "[Stall] %d recoveries and the session is still producing "
                            "no turns — staying quiet and leaving the outcome to the "
                            "check-in ladder.", _stall_recoveries,
                        )
                        continue

                    _line = stall_recovery_line(
                        _lang_now(),
                        0 if _stall_recoveries < _MAX_STALL_RECOVERIES else 1,
                    )
                    # Same backstop the check-in path uses: an audibility question
                    # must never reach the caller, whatever the source.
                    if mentions_audibility_question(_line):
                        logger.error("[Stall] BLOCKED audibility question: %r", _line)
                    else:
                        try:
                            speak_fixed_line(
                                session, _line,
                                allow_interruptions=True,
                                add_to_chat_ctx=False,
                            )
                        except Exception as _we:
                            logger.warning("[Stall] recovery line failed: %s", _we)
                    continue

                # ---- 2. Agent speaking — reset timer, mark greeting done -----
                # Do not zero silence_count: this speech may itself be a
                # presence-check, and the ignored-check-in tally must survive.
                if a == "speaking":
                    _greeting_delivered = True
                    last_speech_time = time.monotonic()
                    continue

                # ---- 3. Agent THINKING — model is generating a response ------
                # This is NOT user silence: the agent received the user's speech
                # and is now processing it. NEVER fire the silence check here —
                # doing so injects a check-in phrase right as the model is
                # generating its real reply, corrupting the output.
                # If thinking takes too long (>3s), inject a brief filler so
                # the caller hears something — but still don't count as silence.
                if a == "thinking":
                    if thinking_since > 0:
                        thinking_elapsed = time.monotonic() - thinking_since
                        # BUG 5: stuck-turn watchdog. If generation runs far past
                        # the filler budget with no audio, the realtime turn is
                        # almost certainly dead (dropped websocket / aborted turn).
                        # Log CRITICAL for diagnosis and clear the thinking state so
                        # the monitor's normal path can re-engage the caller once
                        # the state settles, instead of sitting mute forever.
                        _total_stuck = time.monotonic() - _last_agent_done_at if _last_agent_done_at else thinking_elapsed
                        if thinking_elapsed >= _THINKING_STUCK_S:
                            if not _stuck_thinking_logged:
                                logger.critical(
                                    "[Bug5] Agent stuck in 'thinking' for %.1fs with no audio — "
                                    "treating the turn as dead and recovering (model=%s).",
                                    thinking_elapsed, _userdata.get("realtime_model"),
                                )
                                _stuck_thinking_logged = True
                            # Force recovery: drop the stale thinking state locally so
                            # the silence check below can fire a check-in to re-engage.
                            thinking_since = 0.0
                            _current_a_state = "listening"
                            a = "listening"  # update this iteration's snapshot too
                            # Let the recovery actually re-engage: the caller's turn
                            # IS unanswered (that is the failure), so the normal
                            # unanswered_user_turn guard must be bypassed once.
                            _stuck_recovery_pending = True
                        elif (
                            thinking_elapsed >= _THINKING_FILLER_TIMEOUT
                            and _filler_idx < _MAX_THINKING_FILLERS
                        ):
                            logger.info(
                                f"Agent stuck thinking {thinking_elapsed:.1f}s — injecting filler "
                                f"({_filler_idx + 1}/{_MAX_THINKING_FILLERS})"
                            )
                            try:
                                filler = _thinking_filler(_filler_idx)
                                _filler_idx += 1
                                speak_fixed_line(session, filler, allow_interruptions=True, add_to_chat_ctx=False)
                            except Exception as e:
                                logger.warning(f"Thinking filler error: {e}")
                            thinking_since = time.monotonic()
                    if _current_a_state == "thinking":
                        # ALWAYS skip silence check while model is genuinely thinking.
                        continue
                else:
                    _stuck_thinking_logged = False

                # ---- 4. Before greeting — don't count yet --------------------
                # The model start-up / first-generation delay is not user silence.
                if not _greeting_delivered:
                    last_speech_time = time.monotonic()
                    continue

                # ---- 5. True silence check ------------------------------------
                # Agent is idle, greeting delivered. EVERY condition below has to
                # hold before we are allowed to speak into the gap. Each one
                # corresponds to a way the old single-condition check produced the
                # reported "क्या आप मुझे सुन सकते हैं?" over the caller's voice.
                required = _SILENCE_TIMEOUT if silence_count == 0 else _FOLLOWUP_TIMEOUT
                _t = time.monotonic()

                # BUG 1 FIX — single, testable gate. The critical additions over
                # the old block-reason logic are the GROUND-TRUTH agent-audio
                # guards (is_agent_speaking + last_agent_audio + last_agent_speech):
                # on Gemini 3.1 native audio the agent_state event can lag or never
                # report "speaking", so a check-in could be spoken over the agent's
                # own voice. These stamps are updated the instant the agent produces
                # audio, independent of that event, so we never talk over it.
                _fire, _reason = should_fire_checkin(
                    greeting_delivered=_greeting_delivered,
                    user_state=u,
                    agent_state=a,
                    is_agent_speaking=bool(_userdata.get("is_agent_speaking")),
                    elapsed_since_speech=elapsed,
                    required_silence=required,
                    agent_audio_age=_t - float(_userdata.get("last_agent_audio_monotonic") or 0.0),
                    agent_speech_age=_t - float(_userdata.get("last_agent_speech_monotonic") or 0.0),
                    user_txt_age=_t - _last_user_txt_at,
                    last_user_after_agent=_last_user_txt_at > _last_agent_done_at,
                    checkin_age=_t - _last_checkin_at,
                    agent_audio_guard=_AGENT_AUDIO_GUARD_S,
                    force_recovery=_stuck_recovery_pending,
                )
                if not _fire:
                    if _reason in ("recent_agent_audio", "recent_agent_speech", "agent_speaking_flag"):
                        logger.info(
                            "[Bug1] Suppressed check-in — agent still producing audio (%s).",
                            _reason,
                        )
                    continue

                if _disconnect_event.is_set():
                    break

                # Silence threshold reached — presence-check, do not keep talking
                # through the gap and do not hang up on the first pause.
                if silence_count < _MAX_CHECKINS or not _HANGUP_ON_SILENCE:
                    _lang_sel = _lang_now()
                    checkin_line = reengagement_line(_lang_sel, silence_count)

                    # ISSUE 2 — full trigger-state snapshot, logged for EVERY silence
                    # re-engagement. If a filler ever fires at the wrong moment again,
                    # this line captures the exact condition that produced it.
                    logger.info(
                        "[Issue2][SilenceReengage] attempt=%d/%d lang=%s line=%r | "
                        "elapsed=%.1fs required=%.1fs u_state=%s a_state=%s "
                        "is_agent_speaking=%s agent_audio_age=%.1fs agent_speech_age=%.1fs "
                        "user_txt_age=%.1fs checkin_age=%.1fs thinking_since=%.1f",
                        silence_count + 1, _MAX_CHECKINS, _lang_sel, checkin_line,
                        elapsed, required, u, a,
                        bool(_userdata.get("is_agent_speaking")),
                        _t - float(_userdata.get("last_agent_audio_monotonic") or 0.0),
                        _t - float(_userdata.get("last_agent_speech_monotonic") or 0.0),
                        _t - _last_user_txt_at,
                        _t - _last_checkin_at,
                        thinking_since,
                    )

                    if not checkin_line:
                        # First silence: say NOTHING. Just keep listening and let the
                        # caller take their time — this is the default behaviour now.
                        logger.info(
                            "[Issue2] Silence attempt %d -> staying silent (no utterance).",
                            silence_count + 1,
                        )
                    elif mentions_audibility_question(checkin_line):
                        # Hard backstop: an audibility question must NEVER be spoken,
                        # even if someone reintroduces one into the table later.
                        logger.error(
                            "[Issue2] BLOCKED audibility question before speaking: %r",
                            checkin_line,
                        )
                    else:
                        _checkin_in_flight = True
                        try:
                            speak_fixed_line(
                                session,
                                checkin_line,
                                allow_interruptions=True,
                                add_to_chat_ctx=False,
                            )
                        except Exception as e:
                            _checkin_in_flight = False
                            logger.warning(f"[SilenceMonitor] re-engagement failed: {e}")

                    silence_count += 1
                    last_speech_time = time.monotonic()
                    _last_checkin_at = time.monotonic()
                    _stuck_recovery_pending = False

                else:
                    # Every check-in ignored — the line is dead. Announce cutting call and end immediately.
                    logger.info(
                        "[SilenceMonitor] No response after %d check-ins — saying cutting call message in %s and ending call.",
                        _MAX_CHECKINS,
                        _lang_now(),
                    )
                    goodbye_line = _GOODBYE_BY_LANG.get(_lang_now(), _GOODBYE_BY_LANG["hi"])
                    try:
                        speak_fixed_line(
                            session,
                            goodbye_line,
                            allow_interruptions=False,
                        )
                    except Exception as e:
                        logger.warning(f"Cutting call announcement failed: {e}")

                    # Wait for greeting to finish speaking + 1.0 second before cutting call
                    for _ in range(6):
                        if _userdata.get("is_agent_speaking"):
                            break
                        await asyncio.sleep(0.1)
                    while _userdata.get("is_agent_speaking"):
                        await asyncio.sleep(0.05)
                    await asyncio.sleep(1.0)
                    await _hangup_due_to_silence()
                    break

        except asyncio.CancelledError:
            logger.info("Silence monitor cancelled.")


    # Warm the hospital data cache WITHOUT blocking the greeting.
    #
    # GREETING-LATENCY FIX: this used to be `await fetch_live_hospital_data()`
    # sitting directly in front of session.start(), so every caller waited for
    # it (disk read + JSON parse + normalise) before the agent could say a
    # single word. Nothing in the greeting needs doctor data — it is a fixed
    # hello — and the first tool call re-fetches anyway (cache-aware, 120s TTL).
    # Fired as a background task instead: it almost always finishes long before
    # the caller has finished saying what they want.
    # GREETING-LATENCY (third cause): this was fired immediately, so its
    # synchronous portions — the httpx/Playwright import, JSON parse and
    # normalise — ran on the SAME event loop that was trying to establish the
    # realtime session and emit the greeting. asyncio.create_task() does not
    # make CPU-bound work concurrent; it just interleaves it, and every
    # non-awaiting stretch inside the fetch stalls audio setup.
    #
    # prewarm() has already populated _DATA_CACHE for this worker, and the
    # greeting is a fixed line that needs no doctor data, so there is nothing
    # to gain from racing it. Delay the live refresh until just after the
    # greeting has been handed off — the caller is still listening to it, and
    # the first tool call is cache-aware (120s TTL) anyway.
    # R1 (FIXED OPENER) + R3 (LANGUAGE MATCHING).
    #
    # This is the ONE mandated opening line. It is spoken verbatim on every call,
    # before the caller has said anything, and it must never be replaced by the
    # hospital address, by an apology ("sorry, I can't hear you"), or by anything
    # the model generates on its own.
    #
    # Its being Hindi carries NO weight for the rest of the call. The language
    # lock is driven only by the caller's own first utterance — see the LANGUAGE
    # section of _human_voice_instructions and _capture_user_transcript below —
    # so a caller who answers this Hindi opener in English is answered in English
    # from turn two onward.
    #
    # Override with GREETING_TEXT in .env if the hospital changes the wording.
    _greeting_text = os.environ.get("GREETING_TEXT", "").strip() or _DEFAULT_GREETING_TEXT
    # allow_interruptions must be True for session.say() / generate_reply()
    # with a RealtimeModel — False is rejected and the opener can play with
    # no audible TTS. Set GREETING_INTERRUPTIBLE=false to force uninterruptible.
    _greet_interruptible = (
        os.environ.get("GREETING_INTERRUPTIBLE", "true").strip().lower()
        != "false"
    )
    _userdata["greeting_text"] = _greeting_text
    _userdata["greeting_interruptible"] = _greet_interruptible
    _userdata["greeting_scheduled"] = False
    _userdata["greeting_wav"] = _ensure_greeting_cache_wav(
        _greeting_text, voice_name, model_name
    )
    if _userdata["greeting_wav"]:
        logger.info("Opening greeting will play from cache: %s", _userdata["greeting_wav"])
    else:
        logger.warning(
            "No matching greeting cache WAV — first turn will use generate_reply"
        )

    # ---- START THE SESSION WITHOUT BLOCKING THE GREETING -------------------
    #
    # THE 10-SECOND BUG. `await session.start(...)` does not return as soon as
    # the agent is ready to talk. Internally it runs:
    #
    #     session.start()
    #       -> RoomIO.start()
    #            -> _ParticipantAudioOutput.start()
    #                 -> await publish_track(...)
    #                 -> await publication.wait_for_subscription()   <-- BLOCKS
    #
    # That last await waits for the REMOTE side to subscribe to the agent's
    # audio track. On a SIP/PSTN call the LiveKit SIP bridge only subscribes
    # once the carrier has fully set up the media path, which routinely takes
    # several seconds after the caller has already been connected. So the
    # caller is on the line, hears nothing, and the agent is still stuck inside
    # session.start(). Any greeting placed after `await session.start()` — no
    # matter how well ordered — inherits that entire delay.
    #
    # Fix: don't await start() on the critical path. Run it as a task, and greet
    # as soon as the session's activity exists (session.current_agent is not
    # None), which happens early inside start(), well before the subscription
    # wait. Audio frames generated before subscription are buffered by the
    # output and flushed when the track goes live, so nothing is lost — the
    # greeting simply plays the instant the media path is up instead of
    # several seconds later.
    _agent = PunoVoiceAgent(
        chat_ctx=ChatContext(),
        state=state,
        userdata=_userdata,
        # Same composed string given to RealtimeModel above — keeping
        # these identical is what lets the pre-greeting
        # update_agent_instructions() short-circuit as a no-op.
        instructions=_initial_instructions,
    )

    # ---- FEED CALLER TRANSCRIPTS TO LANGUAGE DETECTION + SPECULATIVE PREP
    # This session has no `stt=`; the native-audio realtime model transcribes
    # server-side and delivers INTERIM and FINAL text on this event. Without
    # capturing finals, on_user_turn_completed() sees an empty text_content
    # and language detection never fires. Interims are also used: they kick
    # doctor/slot/hospital lookups while Gemini VAD is still waiting for
    # ~800ms of silence, so reasoning starts before the caller finishes.
    @session.on("user_input_transcribed")
    def _capture_user_transcript(ev) -> None:
        nonlocal _last_user_txt_at, last_speech_time, silence_count
        try:
            # BUG1: stamp the caller-speech clock on EVERY transcript, interim or
            # final. This is the only reliable "the caller is talking" signal on a
            # session with local VAD, and the silence monitor also gates on it. It
            # must be updated before the is_final early-return below, because a
            # long caller turn produces many interim transcripts and no final one
            # until they stop — which is precisely the window in which check-ins
            # were firing over the caller.
            _last_user_txt_at = time.monotonic()
            last_speech_time = time.monotonic()
            silence_count = 0

            _text = (getattr(ev, "transcript", "") or "").strip()
            _is_final = bool(getattr(ev, "is_final", False))
            # SPECULATIVE PREP: kick doctor/slot/hospital lookups on INTERIM
            # transcripts, not only on the final. Local VAD still owns when
            # the agent may SPEAK; this only warms tools so the reply after
            # ~800ms silence does not also wait on a lookup round-trip.
            if _text:
                _agent.note_partial_transcript(_text, is_final=_is_final)
            if not _is_final:
                _userdata["last_interim_transcript"] = _text
                try:
                    _st = _userdata.get("state")
                    if (
                        _st is not None
                        and _STRONG_CONFIRM_RE.search(_text)
                        and _booking_ready_for_confirm(_st)
                    ):
                        _schedule_asr_auto_confirm(_agent, _text)
                except Exception:
                    pass
                return
            if _text:
                _userdata["last_final_transcript"] = _text
                _userdata["last_interim_transcript"] = ""
                _remember_user_utterance(_userdata, _text)
                try:
                    _st = _userdata.get("state")
                    if _st is not None:
                        if _utterance_requests_caller_phone(_text):
                            if _apply_sip_caller_phone(_st, _userdata):
                                _hydrate_booking_draft(_st)
                        if not _userdata.get("_auto_confirm_in_flight"):
                            _apply_spoken_datetime(_st, _text)
                except Exception:
                    pass
                # BUGE: log the RAW ASR text. This is the single most useful line
                # for debugging language behaviour, because Gemini transcribes
                # Indian-accented English into Devanagari and the classifier can
                # only be judged against what it actually received. Without it
                # every language report is guesswork.
                logger.info("ASR final: %r", _text)
                _schedule_asr_auto_confirm(_agent, _text)
        except Exception as _e:
            logger.debug("Could not capture user transcript for language: %s", _e)

    _room_input_options = RoomInputOptions(
        close_on_disconnect=False,
        # pre_connect_audio buffers audio a browser client sends before
        # it has finished connecting. SIP/PSTN callers never do this, so
        # on this telephony agent the feature only adds a handler and a
        # 3s timeout window to startup with nothing to show for it.
        pre_connect_audio=False,
    )

    def _rebind_session_handlers(sess) -> None:
        sess.on("user_state_changed", on_user_state)
        sess.on("agent_state_changed", on_agent_state)
        sess.on("user_input_transcribed", _capture_user_transcript)

    _greet_task = None
    _background_setup_tasks: set[asyncio.Task] = set()

    async def _start_with_fallback() -> None:
        """Start the realtime session; on failure rebuild with GEMINI_FALLBACK_MODEL."""
        nonlocal session, _agent, _greet_task, _rebind_session_handlers
        models = [model_name]
        if fallback_model:
            models.append(fallback_model)
        last_err: Optional[BaseException] = None
        for i, mname in enumerate(models):
            try:
                if i > 0:
                    logger.warning(
                        "Realtime model %s failed (%s) — falling back to %s",
                        models[i - 1], last_err, mname,
                    )
                    try:
                        await session.aclose()
                    except Exception:
                        pass
                    session = _build_agent_session(
                        _build_realtime_model(mname),
                        vad=_local_vad,
                        model_name_for_turn=mname,
                    )
                    _userdata["session"] = session
                    _userdata["realtime_model"] = mname
                    _agent = PunoVoiceAgent(
                        chat_ctx=ChatContext(),
                        state=state,
                        userdata=_userdata,
                        instructions=_initial_instructions,
                    )
                    recorder._session = session
                    try:
                        session.on(
                            "user_input_transcribed",
                            recorder._on_user_input_transcribed,
                        )
                        session.on(
                            "conversation_item_added",
                            recorder._on_conversation_item_added,
                        )
                    except Exception as e:
                        logger.debug("recorder session rebind: %s", e)
                    if _rebind_session_handlers is not None:
                        _rebind_session_handlers(session)
                    _userdata["greeting_scheduled"] = False
                    if _greet_task is not None and not _greet_task.done():
                        _greet_task.cancel()
                    try:
                        _greet_task = asyncio.create_task(_greet_asap())
                        _background_setup_tasks.add(_greet_task)
                        _greet_task.add_done_callback(_background_setup_tasks.discard)
                    except NameError:
                        # _greet_asap not defined yet — the later greet task will run.
                        pass
                logger.info("Starting realtime session with model=%s", mname)
                await session.start(
                    room=ctx.room,
                    agent=_agent,
                    # close_on_disconnect=False: take manual control of the shutdown
                    # sequence. The default tears down the RTC session immediately when
                    # the SIP caller disconnects, racing the recording/transcript
                    # cleanup and triggering the webrtc-sys "malformed serialized
                    # RtcError" panic.
                    room_input_options=_room_input_options,
                )
                logger.info("Realtime session started with model=%s", mname)
                _userdata["realtime_model"] = mname
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_err = e
                logger.warning("session.start() failed for model %s: %s", mname, e)
        raise last_err or RuntimeError("Realtime session failed to start")

    _start_task = asyncio.create_task(_start_with_fallback())
    _background_setup_tasks.add(_start_task)

    async def _greet_asap() -> None:
        """Backup opener if on_enter did not already schedule one.

        Primary path is PunoVoiceAgent.on_enter, which fires the
        moment the realtime activity exists — that is when session.say() /
        generate_reply() actually work. This loop only covers the case where
        on_enter was skipped (plugin version, fallback rebuild, etc.).

        Do NOT read session.current_agent before start() has assigned _agent:
        that property raises RuntimeError rather than returning None, which
        used to kill this task and leave the line silent until the caller spoke.
        """
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if _userdata.get("greeting_scheduled"):
                _lap("greeting already scheduled")
                return
            if _start_task.done() and _start_task.exception():
                logger.error(
                    "session.start() failed, cannot greet: %s", _start_task.exception()
                )
                return
            # generate_reply() needs _activity, not merely _agent.
            activity = getattr(session, "_activity", None)
            if activity is not None:
                _lap("session ready")
                _lap("greeting now")
                if await _schedule_opening_greeting(session, _userdata):
                    _lap("greeting scheduled")
                return
            await asyncio.sleep(0.01)
        if not _userdata.get("greeting_scheduled"):
            logger.error("Session never became ready within 15s — no greeting sent.")

    # ── R6: detect a genuine caller sign-off ────────────────────────────────
    # `end_call` is now gated on state.call_wrapup_ok (see the guard inside the
    # tool). That flag is set here, from the caller's OWN words only — never from
    # a booking completing, and never from anything the agent said. Until the
    # caller says some version of "that's all", the agent physically cannot hang
    # up, which is the enforcement behind requirement 6.
    _WRAPUP_PHRASES = (
        # English
        "bye", "goodbye", "good bye", "that's all", "thats all", "that is all",
        "nothing else", "no thanks", "no thank you", "i'm done", "im done",
        "we're done", "that's it", "thats it", "all good", "you can hang up",
        "end the call", "end call", "hang up the call", "hang up now",
        "hang up", "cut the call", "cut call", "disconnect the call",
        "you can disconnect", "please disconnect",
        "nothing thank you", "nothing thanks", "ok thank you", "okay thank you", "no nothing",
        # Hindi / Hinglish (romanised)
        "bas itna hi", "bas itna", "itna hi", "bas ho gaya", "bas yahi",
        "kuch nahi chahiye", "kuch nahin chahiye", "aur kuch nahi",
        "aur kuch nahin", "nahi bas", "nahin bas", "theek hai bas", "kuch nahi",
        "rakh deti hoon", "rakh deta hoon", "phone rakhti hoon", "phone rakhta hoon",
        "call kaat do", "call kaat dijiye", "phone kaat do", "kaat dijiye",
        "call end kar do", "call end kardo",
        # Devanagari — native Hindi
        "बस इतना", "कुछ नहीं चाहिए", "और कुछ नहीं", "अलविदा",
        "कॉल काट दो", "फोन काट दो", "कॉल एंड कर दो",
        # Devanagari — TRANSLITERATED ENGLISH
        "दैट्स ऑल", "दैट्स इट", "नथिंग एल्स", "नो थैंक यू",
        "बाय", "गुड बाय", "आई एम डन", "दैट इज ऑल", "ऑल गुड",
        "एंड द कॉल", "एंड कॉल", "हैंग अप", "कॉल काट दीजिए",
    )

    def _normalize_for_wrapup_match(text: str) -> str:
        """Lowercase + strip punctuation before matching.

        BUGFIX: exact-substring matching against raw ASR text missed plain
        punctuation variants — "No. No, thanks." never matched "no thanks"
        because of the comma, so a caller who twice said no to "anything
        else?" was refused a hangup and left looping check-ins until they
        gave up and disconnected themselves. Stripping punctuation (but
        keeping Devanagari/Latin letters, digits, and spaces) before the
        substring check fixes this class of miss without needing every
        punctuation permutation spelled out in the phrase list above.
        """
        if not text:
            return ""
        # Keep letters (incl. Devanagari), digits, spaces; drop everything else
        # (periods, commas, question marks, apostrophes, etc.), then collapse
        # whitespace so "No.  No,  thanks." -> "no no thanks".
        cleaned = re.sub(r"[^\w\u0900-\u097F\s]", " ", text, flags=re.UNICODE)
        return re.sub(r"\s+", " ", cleaned).strip().lower()

    _WRAPUP_PHRASES_NORMALIZED = tuple(
        _normalize_for_wrapup_match(p) for p in _WRAPUP_PHRASES
    )

    @session.on("user_input_transcribed")
    def _detect_wrapup(ev) -> None:
        try:
            if not getattr(ev, "is_final", False):
                return
            _raw = (getattr(ev, "transcript", "") or "").strip()
            if not _raw:
                return
            _t = _normalize_for_wrapup_match(_raw)
            if not _t:
                return
            if any(p in _t for p in _WRAPUP_PHRASES_NORMALIZED):
                if not getattr(state, "call_wrapup_ok", False):
                    logger.info("R6: caller signalled wrap-up (%r) — end_call unlocked.", _raw[:60])
                state.call_wrapup_ok = True
        except Exception as _e:
            logger.debug("wrap-up detection failed: %s", _e)

    def _rebind_session_handlers(sess) -> None:
        sess.on("user_state_changed", on_user_state)
        sess.on("agent_state_changed", on_agent_state)
        sess.on("user_input_transcribed", _capture_user_transcript)
        sess.on("user_input_transcribed", _detect_wrapup)
        # Bug 1 / Bug 4 / Bug 5 handlers must survive a fallback-model rebuild too.
        for _evt, _fn in (
            ("speech_created", on_speech_created),
            ("error", on_session_error),
            ("close", on_session_close),
            ("agent_false_interruption", on_agent_false_interruption),
            ("conversation_item_added", on_conversation_item_output_guard),
        ):
            try:
                sess.on(_evt, _fn)
            except Exception:
                pass

    _greet_task = asyncio.create_task(_greet_asap())
    _background_setup_tasks.add(_greet_task)
    _greet_task.add_done_callback(_background_setup_tasks.discard)

    # Wait for start() to finish before the rest of the setup, which genuinely
    # does need a fully-started session. The greeting is already in flight by
    # now, so this wait costs the caller nothing.
    await _start_task
    _lap("session.start() returned")

    # NOTE: update_agent_instructions() is deliberately NOT called here. 
    #
    # It used to run at this exact point, and it is what killed the greeting.
    # It calls agent.update_instructions() -> the Gemini plugin's
    # _mark_restart_needed(), which resets the outgoing message channel and
    # discards whatever was queued on it. In the failing logs the greeting was
    # queued at 23.022 and this call landed at 23.026 — four milliseconds later
    # — so the greeting was thrown away and its future hung until it timed out
    # ("generate_reply timed out waiting for generation_created event"). The
    # caller heard nothing until they spoke first.
    #
    # The instructions are now applied inside _greet_asap(), BEFORE the greeting
    # is requested, so the restart happens first and the greeting survives.
    # Do not reintroduce a call here.
    # ── Graceful disconnect handling ──────────────────────────────────────────
    # We manually orchestrate shutdown so that cleanup tasks (recording upload,
    # transcript saving) finish BEFORE the RTC peer connection is torn down.

    _disconnect_event = asyncio.Event()
    _background_tasks: list[asyncio.Task] = []

    async def _graceful_disconnect(participant: rtc.RemoteParticipant) -> None:
        """Orchestrate the correct shutdown sequence on caller disconnect.
        
        Order matters critically:
          1. Block outgoing audio       (prevents frames reaching closing RTC)
          2. Cancel background tasks     (silence_monitor, fetch)
          3. Close agent session         (triggers recorder _on_shutdown → saves files)
          4. Disconnect from room        (cleanly closes peer connection from OUR side)
          5. Signal framework shutdown   (lets the agents framework exit cleanly)
        
        Without step 4, the LiveKit server force-closes the data channels ~20s later,
        which triggers the Rust panic in webrtc-sys ("malformed serialized RtcError").
        """
        identity = getattr(participant, "identity", "unknown")
        logger.info("Graceful disconnect handler started for participant: %s", identity)
        _disconnect_event.set()

        # EVERY call produces a lead — do this FIRST, while the transcript store,
        # state and userdata are all still intact (the recorder clears the store at
        # the end of its own shutdown). Idempotent, so the shutdown-callback
        # backstop below cannot double-write.
        try:
            await finalize_lead_for_call(_userdata, reason="participant_disconnected")
        except Exception as _le:
            logger.error("[Lead] finalize on disconnect failed: %s", _le)

        # 1. Cancel background tasks that interact with the session — including the
        #    per-turn tasks the agent spawns (hospital refresh / instruction update),
        #    which were previously untracked and could keep running against a closed
        #    session.
        agent_bg = list(getattr(session.current_agent, "_bg_tasks", ()) or ())
        tool_bg = [t for t in list(_TOOL_BG_TASKS) if not t.done()]
        for task in list(_background_tasks) + agent_bg + tool_bg:
            if not task.done():
                task.cancel()
        _all_bg = list(_background_tasks) + agent_bg + tool_bg
        if _all_bg:
            await asyncio.gather(*_all_bg, return_exceptions=True)
            logger.info("Background tasks cancelled during disconnect.")

        # 2. Small buffer for in-flight capture_frame calls to observe the
        #    per-recorder shutdown flag set by the recorder's disconnect handler.
        await asyncio.sleep(0.3)

        # 3. Close the agent session. This triggers the shutdown callbacks
        #    (including the recorder's _on_shutdown) which persist audio/transcript.
        try:
            await session.aclose()
            logger.info("Agent session closed gracefully.")
        except Exception as e:
            logger.warning("Error closing agent session: %s", e)

        # 4. Disconnect from the room IMMEDIATELY. This cleanly closes the
        #    WebRTC peer connection from our side, preventing the LiveKit server
        #    from forcibly tearing down the data channels ~20s later (which is
        #    what causes the webrtc-sys "malformed serialized RtcError" panic).
        try:
            # rtc.Room.disconnect() is a coroutine and MUST be awaited — without
            # this, step 4 was a no-op and the whole "disconnect before the server
            # tears us down" panic-mitigation this handler is built around never
            # actually ran.
            _maybe = ctx.room.disconnect()
            if inspect.isawaitable(_maybe):
                await _maybe
            logger.info("Disconnected from room cleanly.")
        except Exception as e:
            logger.warning("Error disconnecting from room: %s", e)

        # 5. Signal the agents framework to shut down this job.
        try:
            ctx.shutdown("participant_disconnected")
            logger.info("Job context shutdown signalled.")
        except Exception as e:
            logger.warning("Error signalling job shutdown: %s", e)

        logger.info("Graceful disconnect handler completed for participant: %s", identity)

    @ctx.room.on("participant_disconnected")
    def _on_room_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
        """Room-level disconnect handler that kicks off graceful shutdown."""
        asyncio.ensure_future(_graceful_disconnect(participant))

    # Start the proactive silence monitor task (tracked for cancellation)
    _silence_task = asyncio.create_task(silence_monitor())
    _lap("setup complete")
    _background_tasks.append(_silence_task)

    # Opening greeting: PunoVoiceAgent.on_enter (primary) and
    # _greet_asap (backup). Do not add another generate_reply here — it would
    # be a second greeting.


if __name__ == "__main__":
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="puno-voice-agent",
            load_threshold=0.70,
            # Dev mode defaults to 0 idle processes; prod can set NUM_IDLE_PROCESSES=4.
            num_idle_processes=int(os.environ.get("NUM_IDLE_PROCESSES", "0")),
            # agent.py is large — cold import + prewarm can exceed the SDK default 10s.
            initialize_process_timeout=float(os.environ.get("INIT_PROCESS_TIMEOUT", "30")),
            port=8989,
        )
    )