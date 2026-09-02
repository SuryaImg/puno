"""
call_guardrails.py — pure, unit-testable guardrails for call-quality bugs.

These helpers hold NO livekit/session state, so they can be imported and tested
directly. agent.py wires them into the live session. Each addresses one of the
reported call-quality bugs:

  - mentions_payment_link()      → Bug 4 (agent must never offer online payment)
  - FixedLineDedup               → Bug 3 (agent repeats its own fixed line)
  - should_fire_checkin()        → Bug 1 (agent check-in fires over its own audio)
  - payment_correction_line()    → Bug 4 corrective override

Added after the live Hindi call in which the agent recited its own prompt,
promised to cancel a booking that never existed, and repeated a turn verbatim:

  - promises_booking_modification() → agent must never promise cancel/reschedule
  - leaked_instruction()            → agent must never recite these instructions
  - is_near_duplicate()             → agent must never repeat its own turn
"""

from __future__ import annotations

import re
import time
from difflib import SequenceMatcher
from typing import Optional, Tuple


# ===========================================================================
# Bug 4 — payment-link / online-payment phrasing detector
#
# This build takes NO payment and there is no payment link anywhere. The line
# "I'll send a payment link so you can pay online as well" is not in any code
# path — it can only reach the caller as an LLM hallucination. This detector is
# the runtime backstop behind the prompt prohibition: if the model ever emits
# such phrasing we detect it, log it, and speak a correction.
# ===========================================================================

# Matches an OFFER/PROMISE to pay online or via a link — English, Hindi and
# Hinglish. Deliberately narrow: it must pair a "pay / payment / link / UPI / QR"
# token with an "online / link / send" cue, so a caller-safe sentence like
# "there is no online payment" is NOT flagged (handled by the negation guard).
_PAYMENT_OFFER_RE = re.compile(
    r"""(
        payment\s*link                                   # "payment link"
        | pay(?:ment)?\s*(?:link|gateway)                # "pay link" / "payment gateway"
        | pay\s*online | online\s*pay(?:ment)?           # "pay online" / "online payment"
        | (?:net\s*banking|card|debit|credit)\s+pay(?:ment)?
        | (?:send|sending|share|sharing|bhej\w*|share\s+kar\w*)\s+.{0,25}\blink\b   # "send ... link"
        | \blink\b.{0,20}(?:\bpay|\bbhej|\bsend|\bshare)                            # "link ... pay/send"
        | pay\s+(?:via|through|by|using|se|karne)\s+\w+                             # "pay via UPI/link/online"
        | \bupi\b.{0,20}(?:pay|link|id|scan|send|number|qr)                         # UPI + pay/link/…
        | (?:pay|link|send|share|scan|qr|id)\b.{0,20}\bupi\b                        # … + UPI
        | (?:scan|qr)\s*(?:code)?\b.{0,20}\bpay | pay\b.{0,20}(?:scan|qr)\s*code
        | pay(?:ment)?\s*(?:karne?\s*ke\s*liye|ke\s*liye)\s*(?:link|online)         # Hindi: "pay karne ke liye link"
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# If the sentence is explicitly saying there is NO online payment / nothing to
# pay now, it is a CORRECT, allowed statement — never flag it.
_PAYMENT_NEGATION_RE = re.compile(
    r"(no\s+online\s+payment|nothing('?s)?\s+(?:been\s+)?charged|no\s+payment"
    r"|nothing\s+to\s+pay|no\s+.{0,25}\blink\b|no\s+online\s+payment\s+or\s+link"
    r"|there\s+is\s+no\s+.{0,25}(?:pay|link)|settle\s+at\s+the\s+front\s+desk"
    r"|don'?t\s+.{0,15}pay"
    # Hindi/Hinglish negations: "koi ... nahi", "... nahi hai", "front desk par settle"
    r"|koi\s+[^.]{0,40}?nahi|payment\s+ya\s+link\s+nahi|front\s+desk\s+par\s+(?:hi\s+)?settle)",
    re.IGNORECASE,
)


def mentions_payment_link(text: str) -> bool:
    """True if `text` OFFERS online payment / a payment link (a forbidden line).

    Returns False for allowed negations ("there's no online payment") and for
    empty/short text.
    """
    if not text:
        return False
    t = str(text).strip()
    if len(t) < 3:
        return False
    if _PAYMENT_NEGATION_RE.search(t):
        # A sentence can both negate and offer; treat a clear negation as safe
        # UNLESS it also contains an explicit "payment link" / "pay online" offer.
        if not re.search(r"payment\s*link|pay\s+online|link\s+(?:bhej|send)", t, re.IGNORECASE):
            return False
    return bool(_PAYMENT_OFFER_RE.search(t))


def payment_correction_line(lang: Optional[str]) -> str:
    """A short corrective line to override a hallucinated payment offer."""
    if lang == "hi":
        return (
            "Ek chhoti si baat — is booking par koi online payment ya link nahi hai. "
            "Aap aane par front desk par hi settle karte hain."
        )
    if lang == "hinglish":
        return (
            "Just to clarify — is booking par koi online payment ya link nahi hai, "
            "aap front desk par hi settle karte hain."
        )
    return (
        "Just to clarify — there's no online payment or link for this booking. "
        "You simply settle at the front desk when you arrive."
    )


# ===========================================================================
# Bug 3 — fixed-line dedup (agent repeats its own line verbatim)
#
# Check-ins, thinking fillers and the greeting are spoken via speak_fixed_line().
# A scheduling race (silence loop firing twice before the in-flight flag flips,
# a filler overlapping a check-in, a greeting re-scheduled) could speak the SAME
# line twice back-to-back. This tracks recently-spoken lines and suppresses an
# exact repeat inside a short window.
# ===========================================================================

class FixedLineDedup:
    """Suppress the same fixed line being spoken again within `window_s`."""

    def __init__(self, window_s: float = 8.0):
        self._window_s = window_s
        self._recent: dict[str, float] = {}

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def should_speak(self, text: str, now: Optional[float] = None) -> bool:
        """True if this line may be spoken now (not a recent duplicate)."""
        key = self._norm(text)
        if not key:
            return False
        _now = now if now is not None else time.monotonic()
        # prune old entries so the dict cannot grow unbounded on a long call
        for k, ts in list(self._recent.items()):
            if _now - ts > self._window_s:
                self._recent.pop(k, None)
        last = self._recent.get(key)
        if last is not None and (_now - last) < self._window_s:
            return False
        self._recent[key] = _now
        return True

    def note_spoken(self, text: str, now: Optional[float] = None) -> None:
        key = self._norm(text)
        if key:
            self._recent[key] = now if now is not None else time.monotonic()


# ===========================================================================
# Bug 1 — should a silence check-in fire right now?
#
# The single decision the silence monitor makes before speaking a presence-check
# ("are you still there?"). Extracted as a pure function so every guard can be
# unit-tested. The bug was firing a check-in over the agent's own audio because
# the only signal was agent_state_changed, which lags on Gemini 3.1 native audio.
# The `agent_audio_age` / `agent_speech_age` inputs are GROUND TRUTH (last time
# the agent actually produced audio / started speech), independent of that event.
# ===========================================================================

def should_fire_checkin(
    *,
    greeting_delivered: bool,
    user_state: str,
    agent_state: str,
    is_agent_speaking: bool,
    elapsed_since_speech: float,
    required_silence: float,
    agent_audio_age: float,
    agent_speech_age: float,
    user_txt_age: float,
    last_user_after_agent: bool,
    checkin_age: float,
    agent_audio_guard: float = 2.5,
    force_recovery: bool = False,
) -> Tuple[bool, str]:
    """Return (fire, reason). `fire=False` with a reason string when blocked.

    reason == "fire" means a check-in should be spoken.
    """
    if not greeting_delivered:
        return (False, "pre_greeting")
    if user_state == "speaking":
        return (False, "user_speaking")
    if agent_state == "speaking":
        return (False, "agent_speaking")
    if agent_state == "thinking":
        return (False, "agent_thinking")
    # GROUND-TRUTH agent-audio guard (Bug 1): never talk over the agent's own
    # voice even if agent_state_changed lagged and never reported "speaking".
    if is_agent_speaking:
        return (False, "agent_speaking_flag")
    if agent_audio_age < agent_audio_guard:
        return (False, "recent_agent_audio")
    if agent_speech_age < agent_audio_guard:
        return (False, "recent_agent_speech")
    if elapsed_since_speech < required_silence:
        return (False, "not_enough_silence")
    if user_txt_age < required_silence:
        return (False, "recent_caller_transcript")
    if last_user_after_agent and not force_recovery:
        # Normally an unanswered caller turn means the agent is about to reply, so
        # speaking would talk over it. But when the stuck-turn watchdog has decided
        # the reply is never coming, this guard is exactly what used to block the
        # recovery it was written to enable — hence force_recovery.
        return (False, "unanswered_user_turn")
    if checkin_age < required_silence:
        return (False, "checkin_cooldown")
    return (True, "fire")


# ===========================================================================
# Issue 2 — audibility-question detector.
#
# The agent must NEVER ask the caller whether they can hear it / are still on
# the line. The hardcoded silence "check-in" lines used to do exactly that
# ("Hello, kya aap mujhe sun pa rahe hain?"). This detector is the backstop:
# any fixed line the silence monitor is about to speak is screened through it,
# so the phrase cannot be reintroduced by a future edit, and it also flags the
# model if it generates one itself.
# ===========================================================================

_AUDIBILITY_RE = re.compile(
    r"""(
        can\s+you\s+hear\s+me
      | are\s+you\s+able\s+to\s+hear
      | do\s+you\s+hear\s+me
      | (?:are|is)\s+(?:you|anyone|someone)\s+(?:still\s+)?(?:there|on\s+the\s+line)
      | still\s+with\s+me
      | am\s+i\s+audible
      | sun\s*(?:pa|paa)?\s*rahe?\s*h(?:ain|ai|o)          # "sun pa rahe hain"
      | sun\s*sakte\s*h(?:ain|ai)                           # "sun sakte hain"
      | line\s*(?:pe|par)\s*h(?:ain|ai|o)                   # "line pe hain"
      | mujhe\s+sun
      | आप\s*मुझे\s*सुन
      | सुन\s*(?:पा\s*)?रहे\s*ह
      | लाइन\s*पर\s*ह
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def mentions_audibility_question(text: str) -> bool:
    """True if `text` asks whether the listener can hear / is still on the line."""
    if not text:
        return False
    return bool(_AUDIBILITY_RE.search(str(text)))


# Natural silence re-engagement lines. NONE of these ask about audibility or the
# line — they simply give the caller room and invite them to continue. Index 0 is
# intentionally empty: the FIRST silence is handled by staying silent and waiting.
REENGAGE_BY_LANG = {
    "en": [
        "",  # first silence: say nothing, just wait
        "Take your time — I'm here whenever you're ready.",
        "Whenever you're ready, just let me know how I can help.",
    ],
    "hi": [
        "",
        "Koi jaldi nahi, main yahin hoon — aaram se bataiye.",
        "Jab aap taiyaar hon, bata dijiye. Main aapki madad ke liye yahin hoon.",
    ],
    "hinglish": [
        "",
        "Koi jaldi nahi, main yahin hoon — take your time.",
        "Jab ready hon, bata dijiye — main yahin hoon.",
    ],
}


def reengagement_line(lang: str, attempt: int) -> str:
    """Line for silence attempt `attempt` (0-based). Empty string = stay silent."""
    opts = REENGAGE_BY_LANG.get(lang) or REENGAGE_BY_LANG["en"]
    line = opts[attempt % len(opts)] if opts else ""
    # Hard backstop: never return an audibility question, whatever the table says.
    if line and mentions_audibility_question(line):
        return ""
    return line


# ===========================================================================
# Issue 1 — Hindi first-person gender drift detector.
#
# The TTS voice is fixed (female by default), but Hindi inflects the first
# person for gender. If the model emits masculine self-reference ("main kar
# sakta hoon") in a female voice, the caller hears a persona mismatch. The
# prompt now mandates the correct forms; this detector is the runtime backstop
# that makes any remaining drift visible in the logs.
# ===========================================================================

# First-person masculine markers. Anchored on "main"/"maine" nearby, or on the
# unambiguous "<verb>-a hoon/tha" endings, to avoid flagging the agent correctly
# describing a male CALLER ("aap aa rahe hain" is 2nd person and never matches).
_MASC_SELF_RE = re.compile(
    r"\b(?:main|maine|mai)\b[^.?!]{0,40}?\b(?:sakta|karta|raha|rahaa|deta|leta|"
    r"jaata|jata|aata|ata|bolta|dekhta|samajhta|gaya|doonga|dunga|karunga|"
    r"karoonga|batata|bheja|bhejunga)\b"
    r"|\b(?:sakta|karta|raha|deta|bolta|dekhta)\s+(?:hoon|hun|tha)\b",
    re.IGNORECASE,
)

_FEM_SELF_RE = re.compile(
    r"\b(?:main|maine|mai)\b[^.?!]{0,40}?\b(?:sakti|karti|rahi|deti|leti|"
    r"jaati|jati|aati|ati|bolti|dekhti|samajhti|gayi|doongi|dungi|karungi|"
    r"karoongi|batati|bheji|bhejungi)\b"
    r"|\b(?:sakti|karti|rahi|deti|bolti|dekhti)\s+(?:hoon|hun|thi)\b",
    re.IGNORECASE,
)


def hindi_self_gender(text: str) -> Optional[str]:
    """Return 'masculine' / 'feminine' / None for the agent's own first-person forms."""
    if not text:
        return None
    t = str(text)
    masc = bool(_MASC_SELF_RE.search(t))
    fem = bool(_FEM_SELF_RE.search(t))
    if masc and not fem:
        return "masculine"
    if fem and not masc:
        return "feminine"
    if masc and fem:
        return "mixed"
    return None


def gender_drifted(text: str, expected: str = "female") -> bool:
    """True if `text` uses first-person forms that contradict the configured voice."""
    found = hindi_self_gender(text)
    if found is None:
        return False
    want = "feminine" if str(expected).lower().startswith("f") else "masculine"
    return found != want


# ===========================================================================
# Promising to cancel / reschedule / refund an EXISTING booking
#
# Observed live: the caller said something that was misheard, and the agent
# replied "मैं अभी आपकी बुकिंग कैंसिल कर देती हूं" ("I'll cancel your booking
# right now") — for a booking that did not exist, in a flow that is explicitly
# out of scope in this build. The agent has no such tool, so the promise is
# always a lie, and cancellation is the one mistake a caller acts on.
#
# Same shape as the payment detector: pair a modification verb with a
# first-person completive cue, then let an explicit negation clear it, so the
# CORRECT line ("I can't cancel that myself — let me get you to the team")
# is never flagged.
# ===========================================================================

_BOOKING_MOD_RE = re.compile(
    r"""(
        # English: "I'll cancel it", "I am cancelling", "let me cancel", "I can reschedule"
        (?:i\s*(?:'|’)?\s*(?:ll|will|am|m)\b | let\s+me\b | i\s+can\b
           | i\s+have\s+ | going\s+to\s+)
            [^.?!]{0,40}?\b(?:cancel|cancell|reschedul|refund|postpon)
        # "...cancel that for you" / "reschedule it for you"
      | \b(?:cancel|cancell\w*|reschedul\w*|refund\w*)\b[^.?!]{0,20}?\bfor\s+you\b
        # Hinglish: "cancel kar deti hoon", "cancel kar dungi", "reschedule kar rahi hoon"
      | \b(?:cancel|cancell|reschedule|refund)\w*\s+kar\s*(?:de|d[iu]|rah|du)\w*
        # Devanagari: "कैंसिल कर देती हूं", "रद्द कर रही हूं", "रिफंड कर देते हैं"
      | (?:कैंसिल | कैन्सिल
           | रद्द | रीशेड्यूल
           | रिशेड्यूल
           | रिफंड | रिफ़ंड)
          \s*(?:कर)?\s*
          (?:देती | देता | दूंगी
             | दूंगा | रही | रहा
             | देते | दे)
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# An explicit disclaimer is the CORRECT response — never flag it. Note that
# "I can't cancel that" trips _BOOKING_MOD_RE ("i can" ... "cancel"), so this
# check is what keeps the right answer from being treated as the wrong one.
_BOOKING_MOD_NEGATION_RE = re.compile(
    r"(can'?t\s+(?:cancel|reschedul|refund)"
    r"|cannot\s+(?:cancel|reschedul|refund)"
    r"|(?:not\s+able|unable)\s+to\s+(?:cancel|reschedul|refund)"
    r"|(?:don'?t|do\s+not)\s+have\s+(?:the\s+)?(?:ability|access|option)"
    r"|nahi\s+kar\s+sak|nahi\s+kar\s+paa"
    r"|नहीं\s*कर\s*(?:सकती|सकता"
    r"|पाऊंगी|पाती)"
    r")",
    re.IGNORECASE,
)


def promises_booking_modification(text: str) -> bool:
    """True if `text` PROMISES to cancel / reschedule / refund a booking.

    This build cannot do any of those — every such request must go to a human —
    so an affirmative promise here is always a hallucination worth interrupting.
    Explicit disclaimers ("I can't cancel that myself") return False.
    """
    if not text:
        return False
    t = str(text).strip()
    if len(t) < 3:
        return False
    if _BOOKING_MOD_NEGATION_RE.search(t):
        return False
    return bool(_BOOKING_MOD_RE.search(t))


def booking_modification_correction_line(lang: Optional[str]) -> str:
    """Corrective line to override a hallucinated cancel/reschedule promise."""
    if lang == "hi":
        return (
            "Maaf kijiye, maine galat keh diya — main khud booking cancel ya "
            "reschedule nahi kar sakti. Main aapko team member se connect karti hoon."
        )
    if lang == "hinglish":
        return (
            "Sorry, correction — main khud booking cancel ya reschedule nahi kar sakti. "
            "Let me connect you to one of our team members."
        )
    return (
        "Sorry, let me correct that — I'm not able to cancel or change a booking "
        "myself. Let me get you to one of our team members who can sort this out."
    )


# ===========================================================================
# Instruction leakage
#
# Observed live: the agent spoke "एट मोस्ट टू डिटेल्स एट अ टाइम ... समराइज एट
# स्टेप 5.1" — reciting its own system prompt, step number included. The real
# fix is the prompt rewrite (the numbered call-flow scaffolding it was reading
# out is gone); this is the backstop that tells us if it ever happens again.
# ===========================================================================

_LEAKED_INSTRUCTION_RE = re.compile(
    r"""(
        \bstep\s*[0-9]
      | स्टेप\s*[0-9०-९]
      | \bat\s+most\s+(?:two|2)\s+details
      | एट\s*मोस्ट
      | \bsummari[sz]e\s+at\b
      | समराइज\s*एट
      | \bguardrail
      | \bstored\s+fields?\b | \bmissing\s+fields?\b
      | \bknowledge\s+base\b | \bcall\s+flow\b | \bsystem\s+prompt\b
      | \bmy\s+instructions?\b | \bas\s+per\s+(?:my\s+)?instructions?\b
      | \bi\s+(?:have\s+been|was|am)\s+(?:told|instructed|supposed)\b
      | \b(?:confirm_booking|create_lead|log_call_summary|transfer_to_human
           |end_call|check_activity|update_booking_details|get_packages)\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def leaked_instruction(text: str) -> Optional[str]:
    """Return the leaked fragment if `text` recites the system prompt, else None.

    Returns the matched text (not just a bool) so the log line names exactly
    which phrasing escaped — that is what makes the next prompt fix targeted.
    """
    if not text:
        return None
    m = _LEAKED_INSTRUCTION_RE.search(str(text))
    return m.group(0) if m else None


# ===========================================================================
# Self-repetition across MODEL-GENERATED turns
#
# FixedLineDedup covers speak_fixed_line() only — it never sees what the model
# generates, so the observed "agent repeats its own turn verbatim after the
# caller asks it to slow down" went undetected. This compares consecutive
# assistant turns so agent.py can log it and nudge.
# ===========================================================================


def _norm_for_compare(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (Devanagari preserved)."""
    t = (text or "").strip().lower()
    t = re.sub(r"[^\w\sऀ-ॿ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ===========================================================================
# Agent stall recovery — the caller is waiting on US
#
# reengagement_line() answers the opposite situation: the CALLER has gone quiet
# and is entitled to think, so attempt 0 is deliberately an empty string and the
# later lines say "take your time". Spoken into an AGENT stall those lines are
# actively wrong — the caller is not taking their time, they are saying "hello?
# are you there?" while the agent owes them a reply.
#
# Observed live: 55 seconds of dead air after the caller answered a question.
# These lines break that silence, own the pause, and hand the turn back.
# ===========================================================================

_STALL_RECOVERY_BY_LANG = {
    "en": [
        "Sorry, I went quiet for a second there — I'm back with you. "
        "Could you say that once more?",
        "Sorry about that again — I'm having a bit of trouble on my end. "
        "Let me get you to one of our team members.",
    ],
    "hi": [
        "Maaf kijiye, main ek pal ke liye ruk gayi thi — ab main yahin hoon. "
        "Aap phir se bata dijiye?",
        "Maaf kijiye, mujhe thodi dikkat aa rahi hai. "
        "Main aapko team member se connect karti hoon.",
    ],
    "hinglish": [
        "Sorry, main ek pal ke liye ruk gayi thi — ab main yahin hoon. "
        "Aap phir se bata dijiye?",
        "Sorry, mujhe thodi dikkat aa rahi hai — "
        "main aapko team member se connect karti hoon.",
    ],
}


def stall_recovery_line(lang: Optional[str], attempt: int = 0) -> str:
    """Line to speak when the agent owes the caller a reply that never came.

    Unlike reengagement_line(), attempt 0 is NEVER empty: staying silent is the
    correct answer to a caller who is thinking, and the worst possible answer to
    a caller who is waiting on the agent. Feminine Hindi self-reference matches
    the configured voice (see gender_drifted).
    """
    opts = _STALL_RECOVERY_BY_LANG.get(
        lang if lang in _STALL_RECOVERY_BY_LANG else "en",
        _STALL_RECOVERY_BY_LANG["en"],
    )
    return opts[min(max(int(attempt), 0), len(opts) - 1)]


def is_near_duplicate(a: str, b: str, threshold: float = 0.90) -> bool:
    """True if two agent turns are the same sentence said twice.

    Short turns are exempt: "okay", "haan ji", "got it" legitimately recur and
    flagging them would be noise. Only substantive turns are compared.
    """
    na, nb = _norm_for_compare(a), _norm_for_compare(b)
    if not na or not nb:
        return False
    if len(na) < 15 or len(nb) < 15:
        return False
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold
