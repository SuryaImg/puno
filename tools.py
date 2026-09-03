"""
tools.py — AppointmentState + SkipQueue + CallTranscript for Hospital Receptionist Agent
"""

import logging
import time
import os
import json
import re
import datetime
from typing import Optional, List, ClassVar
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

import unicodedata


def normalize_unicode_digits(text: str) -> str:
    """Convert any Unicode decimal-digit glyph (Arabic-Indic ٠-٩, Extended
    Arabic-Indic ۰-۹, Devanagari ०-९, Gujarati ૦-૯, Bengali ০-৯, Fullwidth,
    etc.) to plain ASCII 0-9. Speech-to-text for Hindi/Urdu callers frequently
    returns numbers in these scripts, which the old parser left untouched — so a
    perfectly valid phone number like "٩٨..." came out as an unusable non-ASCII
    string and every downstream booking validation failed. Non-digit characters
    are left as-is."""
    if not text:
        return text
    out = []
    for ch in text:
        if ch.isdigit():
            try:
                out.append(str(unicodedata.digit(ch)))
                continue
            except (TypeError, ValueError):
                pass
        out.append(ch)
    return "".join(out)


class SkipQueue:
    """
    Tracks which checklist fields have been skipped and how many times
    each field has been asked in total.
    """
    ACTIVE = "active"
    DEFERRED = "deferred"
    PERMANENT = "permanent"

    def __init__(self) -> None:
        self._state: dict[str, str] = {}
        self._ask_count: dict[str, int] = {}
        self._deferred_queue: list[str] = []

    def increment_ask(self, field: str) -> int:
        self._ask_count[field] = self._ask_count.get(field, 0) + 1
        return self._ask_count[field]

    def ask_count(self, field: str) -> int:
        return self._ask_count.get(field, 0)

    def max_asks_reached(self, field: str, max_asks: int = 2) -> bool:
        return self.ask_count(field) >= max_asks

    def skip(self, field: str) -> str:
        current = self._state.get(field, self.ACTIVE)
        if current == self.PERMANENT:
            return self.PERMANENT
        if current == self.DEFERRED:
            self._state[field] = self.PERMANENT
            if field in self._deferred_queue:
                self._deferred_queue.remove(field)
            return self.PERMANENT
        self._state[field] = self.DEFERRED
        if field not in self._deferred_queue:
            self._deferred_queue.append(field)
        return self.DEFERRED

    def mark_complete(self, field: str) -> None:
        self._state[field] = self.ACTIVE
        if field in self._deferred_queue:
            self._deferred_queue.remove(field)

    def is_permanently_skipped(self, field: str) -> bool:
        return self._state.get(field) == self.PERMANENT

    def is_deferred(self, field: str) -> bool:
        return self._state.get(field) == self.DEFERRED

    def deferred_fields(self) -> list[str]:
        return list(self._deferred_queue)

    def pop_next_deferred(self) -> Optional[str]:
        while self._deferred_queue:
            f = self._deferred_queue[0]
            if self._state.get(f) == self.PERMANENT:
                self._deferred_queue.pop(0)
                continue
            self._deferred_queue.pop(0)
            return f
        return None

    def summary(self) -> str:
        deferred = [f for f in self._deferred_queue if not self.is_permanently_skipped(f)]
        permanent = [f for f, s in self._state.items() if s == self.PERMANENT]
        counts = {f: c for f, c in self._ask_count.items() if c > 0}
        return f"deferred={deferred}, permanent_skip={permanent}, ask_counts={counts}"


class BookingStatusStr(str):
    def __eq__(self, other):
        if not isinstance(other, str):
            return NotImplemented
        s1 = self.upper()
        s2 = other.upper()
        if s1 == s2:
            return True
        if s1 == "BOOKED" and s2 == "CONFIRMED":
            return True
        if s1 == "CONFIRMED" and s2 == "BOOKED":
            return True
        non_booked = {"PENDING", "DRAFT", "READY_FOR_CONFIRMATION", "BOOKING_IN_PROGRESS", "FAILED"}
        if s1 in non_booked and s2 in ("PENDING", "DRAFT"):
            return True
        if s2 in non_booked and s1 in ("PENDING", "DRAFT"):
            return True
        return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.upper())



class AppointmentState(BaseModel):
    """Tracks progress through the hospital appointment booking flow."""

    appointment_id: str | None = Field(None, description="Unique appointment ID")
    caller_name: str | None = Field(None, description="Caller's full name")
    patient_name: str | None = Field(None, description="Patient's full name")
    patient_phone: str | None = Field(None, description="Patient's contact number")
    phone_number: str | None = Field(None, description="Patient's contact number")
    department: str | None = Field(None, description="Department for consultation")
    doctor_name: str | None = Field(None, description="Name of the doctor")
    doctor_id: int | None = Field(None, description="Doctor ID from the backend")
    doctor_preference: str | None = Field(None, description="Doctor preference (e.g. any, specific)")
    appointment_date: str | None = Field(None, description="Date of the appointment")
    appointment_time: str | None = Field(None, description="Time of the appointment")
    reason: str | None = Field(None, description="Reason for visit / activities of interest")
    time_preference: str | None = Field(None, description="Time preference (e.g. FLEXIBLE)")

    # --- PUNO Jaipur booking fields (v2.3) ---------------------------------
    # The five Step-4 fields map onto existing fields where one already fits
    # (name -> patient_name, phone -> patient_phone, date -> appointment_date,
    # slot -> appointment_time); these add the PUNO-specific ones. Headcount is
    # split under-7 / 7+ because it drives package eligibility.
    kids_under7: int | None = Field(None, description="Number of visitors under 7 years old")
    adults_7plus: int | None = Field(None, description="Number of visitors aged 7 and over")
    package_id: str | None = Field(None, description="Selected PUNO package id (e.g. tourist_weekday)")
    package_name: str | None = Field(None, description="Selected PUNO package display name")
    total_price: int | None = Field(None, description="Computed total booking price in rupees (GST inclusive)")
    is_peak_day: bool | None = Field(None, description="Whether the chosen date is flagged as a peak day")
    # System-action results / CRM fields
    booking_id: str | None = Field(None, description="Booking id returned by confirm_booking()")
    lead_id: str | None = Field(None, description="Lead id returned by create_lead()")
    call_category: str | None = Field(None, description="Call intent/category for the CRM log")
    call_outcome: str | None = Field(None, description="Booked | Lead created | Transferred to human | Info only | Not interested")
    payment_status: str = Field("n/a", description="Always 'n/a' in this build — no payment channel exists")
    follow_up_needed: bool = Field(False, description="Whether a human follow-up is required")
    follow_up_note: str | None = Field(None, description="Free-text note for follow-up / handover context")
    transferred_to_human: bool = Field(False, description="Whether the call was handed to a human")

    booking_status: str = Field("DRAFT", description="Status of booking (DRAFT, READY_FOR_CONFIRMATION, BOOKING_IN_PROGRESS, BOOKED, FAILED)")
    booking_intent_detected: bool = Field(False, description="Whether the user has expressed an intent to book an appointment")
    callback_scheduled_at: str | None = Field(None, description="Scheduled callback datetime string")
    callback_reason: str | None = Field(None, description="Reason for callback scheduling")
    patient_name_locked: bool = Field(False, description="Whether the patient name has been confirmed and locked.")
    patient_name_spelled: str | None = Field(None, description="The spelled out patient name, if provided.")
    mismatch_acknowledged: bool = Field(False, description="Whether the symptom mismatch has been acknowledged.")
    availability_verified: bool = Field(False, description="Whether slot availability is verified")
    patient_phone_confirmed: bool = Field(False, description="Whether the patient phone number is confirmed")
    phone_confirmation_attempts: int = Field(0, description="Number of confirmation attempts for phone number")
    # R5: digits heard so far when the caller's number came through incomplete.
    # Lets the agent ask only for the MISSING digits instead of making the caller
    # recite all ten again. Cleared the moment a full valid number is stored.
    phone_partial: str | None = Field(None, description="Partial phone digits captured so far (<10), used to ask only for the missing digits")
    # R6: True only once the CALLER has signalled they are finished with the call
    # (goodbye / "bas itna hi" / "nothing else"). The end_call tool refuses to run
    # while this is False, so a confirmed booking can never hang up on its own.
    call_wrapup_ok: bool = Field(False, description="True once the caller has explicitly signalled the call can end")
    # The caller must hear "your booking is confirmed" EXACTLY ONCE. Set the first
    # time the agent actually states it; every later confirmation-style turn is a
    # duplicate and is cut off at the code level (prompt rules alone did not hold —
    # the agent stated it three times on the reported call, once after the goodbye).
    booking_stated: bool = Field(False, description="True once the booking confirmation has been SPOKEN to the caller")
    name_confirmation_attempts: int = Field(0, description="Number of confirmation attempts for patient name")

    relation: str | None = Field(None, description="Patient relationship (e.g. self, mother, father, child, etc.)")
    gender: str | None = Field(None, description="Patient gender (e.g. male, female, etc.)")
    age: str | None = Field(None, description="Patient age")
    additional_notes: str | None = Field(None, description="Any additional notes shared by the user")
    patient_profiles: dict[str, dict] = Field(default_factory=dict, description="All patient profiles in this conversation")
    active_profile_key: str | None = Field(None, description="Key of the active patient profile")

    # --- LANGUAGE LOCK -------------------------------------------------
    # The language the CALLER is actually speaking. Set only from the caller's
    # own transcribed speech (never from the agent's own replies, never from a
    # tool result). "en" | "hi" | "hinglish". None = not yet determined, in
    # which case only the greeting language applies.
    caller_language: str | None = Field(None, description="Detected language of the caller: en | hi | hinglish")
    caller_language_locked: bool = Field(False, description="True once the caller has spoken at least one clear, classifiable utterance")

    booking_state: str | None = Field(None, description="High-level booking state")
    total_appointments_requested: int = Field(0, description="Total number of appointments requested")
    completed_appointments_count: int = Field(0, description="Number of completed appointments")

    # --- Emergency case tracking -------------------------------------------
    # An emergency is NOT an appointment booking: no doctor/slot/date is
    # collected. It is persisted separately (register_emergency_case) so the
    # ER desk has a record of who is on their way. These fields exist so the
    # emergency survives state save/reload and is visible in the summary.
    is_emergency: bool = Field(False, description="Whether an emergency has been detected on this call")
    emergency_registered: bool = Field(False, description="Whether the emergency case was persisted to the ER registry")
    emergency_case_id: str | None = Field(None, description="ID of the persisted emergency case record")
    emergency_description: str | None = Field(None, description="What the caller described (symptoms/incident)")
    emergency_age_verified: bool = Field(False, description="Whether the child's age was confirmed as under 10 for this emergency")

    ask_counts: dict[str, int] = Field(default_factory=dict, description="Number of times each slot was asked")
    skipped_fields: list[str] = Field(default_factory=list, description="Fields that should be skipped because they were asked twice")
    step_attempts: dict[str, int] = Field(default_factory=dict, description="Number of times each step/state was executed/attempted")

    @model_validator(mode="before")
    @classmethod
    def pre_sync_and_normalize(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            return data
        
        # 1. Normalize booking_status
        status = data.get("booking_status")
        if isinstance(status, str):
            status_upper = status.upper()
            if status_upper == "CONFIRMED":
                data["booking_status"] = "BOOKED"
            elif status_upper == "PENDING":
                data["booking_status"] = "DRAFT"
            else:
                data["booking_status"] = status_upper

        # 2. Sync phone number fields
        phone = data.get("phone_number")
        pat_phone = data.get("patient_phone")
        if pat_phone is not None and phone is None:
            data["phone_number"] = pat_phone
        elif phone is not None and pat_phone is None:
            data["patient_phone"] = phone
            
        return data

    def __getattribute__(self, name):
        val = super().__getattribute__(name)
        if name == "booking_status" and isinstance(val, str):
            return BookingStatusStr(val)
        return val

    def __setattr__(self, name, value):
        if name == "booking_status" and isinstance(value, str):
            val_upper = value.upper()
            if val_upper == "CONFIRMED":
                value = "BOOKED"
            elif val_upper == "PENDING":
                value = "DRAFT"
            elif val_upper in ("DRAFT", "READY_FOR_CONFIRMATION", "BOOKING_IN_PROGRESS", "BOOKED", "FAILED"):
                value = val_upper
        elif name in ("patient_phone", "phone_number"):
            super().__setattr__("patient_phone", value)
            super().__setattr__("phone_number", value)
            return
        super().__setattr__(name, value)

    # Fields a booking cannot legally be completed without. These may NEVER be
    # auto-skipped: the previous code moved any field into skipped_fields after
    # two asks, and current_step() then skipped it permanently — so a call where
    # the agent happened to speak twice while the name was still blank could
    # walk all the way to confirm_appointment with no patient name at all.
    MANDATORY_FIELDS: ClassVar[tuple[str, ...]] = (
        "patient_name", "patient_phone", "phone_number",
        "department", "doctor_id", "appointment_date", "appointment_time",
    )

    def increment_ask(self, field: str) -> None:
        self.ask_counts[field] = self.ask_counts.get(field, 0) + 1
        if (
            self.ask_counts[field] >= 2
            and field not in self.skipped_fields
            and field not in self.MANDATORY_FIELDS
        ):
            self.skipped_fields.append(field)

    def repeatedly_asked_fields(self, threshold: int = 2) -> list[str]:
        """Fields asked about `threshold`+ times — surfaced to the model so it
        knows not to ask them again in the same words."""
        return [f for f, c in self.ask_counts.items() if c >= threshold]

    def get_ask_count(self, field: str) -> int:
        return self.ask_counts.get(field, 0)

    def increment_step_attempt(self, step: str) -> None:
        self.step_attempts[step] = self.step_attempts.get(step, 0) + 1
        logger.info(f"STATE STEP ATTEMPT: Step '{step}' attempt count incremented to {self.step_attempts[step]}")
        
    def is_step_limit_exceeded(self, step: str, limit: int = 3) -> bool:
        return self.step_attempts.get(step, 0) > limit

    def current_step(self) -> str:
        if "department" not in self.skipped_fields and not self.department: return "department"
        if "appointment_date" not in self.skipped_fields and not self.appointment_date: return "appointment_date"
        if "appointment_time" not in self.skipped_fields and not self.appointment_time: return "appointment_time"
        if "patient_name" not in self.skipped_fields and not self.patient_name: return "patient_name"
        if "phone_number" not in self.skipped_fields and "patient_phone" not in self.skipped_fields and (not self.phone_number or (self.phone_number and not self.patient_phone_confirmed and self.phone_confirmation_attempts < 2)): return "patient_phone"
        
        if self.booking_status != "BOOKED" and self.booking_status != "confirmed":
            return "confirmation"
        return "complete"

    def completed_steps_summary(self) -> str:
        steps = []
        if self.patient_name: steps.append(f"✓ Name: {self.patient_name}")
        if self.phone_number: steps.append(f"✓ Phone: {self.phone_number}")
        if self.department: steps.append(f"✓ Dept: {self.department}")
        if self.doctor_name: steps.append(f"✓ Doctor: {self.doctor_name}")
        if self.doctor_id: steps.append(f"✓ Doctor ID: {self.doctor_id}")
        if self.appointment_date: steps.append(f"✓ Date: {self.appointment_date}")
        if self.appointment_time: steps.append(f"✓ Time: {self.appointment_time}")
        if self.reason: steps.append(f"✓ Reason: {self.reason}")
        if self.booking_status == "BOOKED" or self.booking_status == "confirmed": steps.append("✓ Booking Confirmed")
        return "\n".join(steps) if steps else "No steps completed yet."


# IST, not the host clock — see storage._now_ist_str(). Duplicated here rather
# than imported because storage.py imports THIS module (importing back would be
# circular). Same fixed +5:30 offset, same output format.
_IST_TZ = datetime.timezone(datetime.timedelta(hours=5, minutes=30))


def _now_ist_str() -> str:
    """Current IST wall-clock as 'YYYY-MM-DD HH:MM:SS'."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .astimezone(_IST_TZ)
        .strftime("%Y-%m-%d %H:%M:%S")
    )


class TranscriptEntry(BaseModel):
    """A single event/entry in the call transcript."""
    timestamp: str = Field(default_factory=_now_ist_str)
    event_type: str = Field(
        description="Type: appointment_booked | callback_scheduled | step_completed | "
                    "error_reported | user_response | call_ended | reschedule_requested | "
                    "escalation_requested"
    )
    details: str = Field(description="Human-readable description of what happened")
    data: dict = Field(default_factory=dict, description="Structured data for this event")


class CallTranscript(BaseModel):
    """
    Complete call transcript saved continuously during the call.
    Stored in Redis as: transcript:{user_id}:{call_start_time}
    """
    user_id: str
    call_start_time: str = Field(default_factory=lambda: str(int(time.time())))
    call_start_human: str = Field(default_factory=_now_ist_str)
    call_end_time: str | None = None
    phone_number: str | None = None

    # Snapshot of the booking
    patient_name: str | None = None
    doctor_name: str | None = None
    appointment_date: str | None = None
    appointment_time: str | None = None
    booking_status: str | None = None

    callback_scheduled: bool = False
    callback_datetime: str | None = None
    callback_reason: str | None = None

    # Escalation tracking
    escalations: List[dict] = Field(default_factory=list, description="List of escalation events")

    pending_steps: List[str] = Field(default_factory=list)
    events: List[TranscriptEntry] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    def add_event(self, event_type: str, details: str, data: dict | None = None) -> None:
        entry = TranscriptEntry(event_type=event_type, details=details, data=data or {})
        self.events.append(entry)
        # Track escalation events separately for easy lookup
        if event_type == "escalation_requested" and data:
            self.escalations.append({
                "timestamp": entry.timestamp,
                "department": data.get("department"),
                "reason": data.get("reason"),
            })
        logger.debug("Transcript[%s]: %s — %s", event_type, details, data)

    def summary_text(self) -> str:
        lines = [
            f"Call: {self.call_start_human} | User: {self.user_id}",
            f"Patient: {self.patient_name or 'Unknown'} | Doctor: {self.doctor_name or 'Unknown'}",
            f"Date/Time: {self.appointment_date or '?'} {self.appointment_time or '?'} | Status: {self.booking_status or 'Pending'}",
        ]
        if self.callback_scheduled:
            lines.append(f"Callback scheduled: {self.callback_datetime} — {self.callback_reason}")
        if self.escalations:
            esc_summary = "; ".join(
                f"{e['department']} ({e['reason']})" for e in self.escalations
            )
            lines.append(f"Escalations: {esc_summary}")
        if self.pending_steps:
            lines.append(f"Pending: {', '.join(self.pending_steps)}")
        if self.errors:
            lines.append(f"Errors: {'; '.join(self.errors)}")
        return "\n".join(lines)


_SYMPTOM_RULES: list[tuple[list[str], str]] = [
    # --- Dr. Chirag Sethi: Pediatric Critical Care, PICU, NICU, Emergencies ---
    (["emergency", "picu", "nicu", "breathing", "breathless", "difficulty breathing",
      "convulsion", "convulsions", "seizure", "seizures", "fits", "critical illness",
      "icu", "icu admission", "serious infection", "serious infections", "premature",
      "preterm", "ventilator", "ventilator support", "trauma", "severe", "life-threatening",
      "saans phoolna", "saans lene me takleef", "daura", "daure", "jhatka", "behosh",
      "behoshi", "gambhira", "saans me takleef", "गंभीर", "सांस लेने में तकलीफ",
      "दौरा", "दौरे", "बेहोशी", "झटका", "वेंटिलेटर", "आईसीयू"],
     "Pediatric Critical Care"),

    # --- Dr. Dileep Sethi: Pediatrics & Neonatology (General Pediatrics, Newborn, Growth, Routine, Default) ---
    (["fever", "mild fever", "high fever", "cold", "cough", "vomiting", "vomit", "diarrhea", "diarrhoea",
      "loose motion", "loose motions", "nutrition", "growth", "growth issues",
      "vaccine", "vaccination", "immunization", "tika", "teeka", "well baby", "well-baby",
      "newborn", "newborn consultation", "jaundice", "breastfeeding", "feeding",
      "development", "child development", "general pediatric", "general pediatrics",
      "consultation", "follow up", "follow-up", "routine", "child health",
      "bukhar", "halka bukhar", "mild bukhar", "khansi", "jukam", "ultii", "dast", "vajan", "vikas", "piliya",
      "doodh", "chhaati me dard", "pet dard", "pet me dard", "headache", "head ache", "sar dard", "sir dard",
      "sar me dard", "sir me dard", "halka sar dard", "halka sir dard", "badan dard", "body pain", "body ache",
      "rash", "rashes", "skin", "skin rash", "skin problem", "dermatology", "itching", "khaj khujli", "khujli", "twacha",
      "pulmonary", "respiratory", "asthma", "lung", "phaphda", "wheezing",
      "stomach", "digestive", "digestion", "abdominal", "constipation", "kabj", "gas", "acidity",
      "ent", "ear", "ear pain", "throat", "throat pain", "gala", "gala dard", "tonsil", "nose", "nak", "kan",
      "neurology", "neurological", "nerve", "dizziness", "chakkar",
      "allergy", "allergies", "allergic",
      "bone", "joint", "joint pain", "bone pain", "eye", "eye pain", "aankh", "weakness", "kamzori",
      "injury", "checkup", "gala dard", "throat pain",
      "बुखार", "हल्का बुखार", "खांसी", "जुकाम", "उल्टी", "दस्त", "वजन", "विकास", "पीलिया",
      "सर दर्द", "सिर दर्द", "पेट दर्द", "माथा दर्द", "त्वचा", "खुजली", "सांस", "दमा", "एलर्जी", "कान", "गला",
      "टीका", "टीकाकरण", "आंख", "हड्डी", "जांच"],
     "Pediatrics & Neonatology"),
]

# Exclusive pediatric hospital catch-all: Pediatrics & Neonatology (Dr. Dileep Sethi)
_FALLBACK_SPECIALTY = "Pediatrics & Neonatology"

# No pediatric case is blocked from our two doctors.
_NO_GP_FALLBACK_SPECIALTIES = set()
_HIGH_RISK_SPECIALTIES = set()


_SYMPTOM_PRIORITY = {
    # High-risk pediatric specialists.
    "Pediatric Neurology": 0, "Neurologist": 0,
    "Pediatric Cardiology": 0, "Cardiology": 0,
    "Neonatology": 0,
    "Oncology": 0,
    # Mid specialists.
    "Pediatric Surgery": 1, "Surgeon": 1,
    "Pediatric Orthopedics": 1, "Orthopedics": 1,
    "Pediatric Dermatology": 1, "Dermatology": 1,
    "Pediatric ENT": 1, "ENT": 1,
    "Pediatric Pulmonology": 1, "Pulmonology": 1,
    "Pediatric Dentistry": 1, "Dentistry": 1,
    "Child & Adolescent Psychiatry": 1, "Psychiatry": 1,
    "Vaccination & Immunization": 1,
    "Ophthalmology": 1, "Urology": 1, "Nephrology": 1,
    "Gynecology": 1, "Gastroenterology": 1,
    # Catch-all: only wins if nothing else scores.
    "Pediatrics": 9,
    "General Physician": 9,
}




def normalize_hindi_phone_number(phone_str: str) -> str:
    if not phone_str:
        return ""

    # Convert any non-ASCII digit glyphs (Arabic-Indic, Devanagari, etc.) to
    # ASCII first, so numbers dictated in any script parse correctly.
    phone_str = normalize_unicode_digits(phone_str)

    WORD_VALUES = {
        "शून्य": 0, "सुन्ना": 0, "जीरो": 0, "जीरों": 0, "zero": 0,
        # Urdu-script number words (STT often returns these for Hindi/Urdu callers)
        "صفر": 0, "زیرو": 0, "ایک": 1, "دو": 2, "تین": 3, "چار": 4,
        "پانچ": 5, "پانج": 5, "چھ": 6, "چھے": 6, "سات": 7, "آٹھ": 8,
        "اٹھ": 8, "نو": 9, "نائن": 9,
        # Romanized/Hinglish Hindi number words — Hindi STT very often transliterates
        # spoken digits to Latin ("nau aath do") instead of Devanagari, which the
        # map previously missed entirely (parsed to an empty string).
        "shunya": 0, "sunna": 0, "sifar": 0,
        "ek": 1, "eak": 1,
        "do": 2, "doe": 2,
        "teen": 3, "tin": 3,
        "char": 4, "chaar": 4,
        "paanch": 5, "panch": 5, "paach": 5,
        "chhe": 6, "che": 6, "chha": 6, "chah": 6, "cheh": 6,
        "saat": 7, "sat": 7,
        "aath": 8, "ath": 8, "aat": 8,
        "nau": 9, "nao": 9,
        "एक": 1, "one": 1,
        "दो": 2, "two": 2,
        "तीन": 3, "three": 3,
        "चार": 4, "four": 4,
        "पांच": 5, "पाँच": 5, "five": 5,
        "छह": 6, "छः": 6, "छ": 6, "six": 6,
        "सात": 7, "seven": 7,
        "आठ": 8, "eight": 8,
        "नौ": 9, "नो": 9, "nine": 9,
        "दस": 10, "ten": 10,
        "ग्यारह": 11, "eleven": 11,
        "बारह": 12, "twelve": 12,
        "तेरह": 13, "thirteen": 13,
        "चौदह": 14, "fourteen": 14,
        "पंद्रह": 15, "fifteen": 15,
        "सोलह": 16, "sixteen": 16,
        "सत्रह": 17, "seventeen": 17,
        "अठारह": 18, "eighteen": 18,
        "उन्नीस": 19, "nineteen": 19,
        "बीस": 20, "twenty": 20,
        "इक्कीस": 21, "twentyone": 21, "twenty-one": 21,
        "बाईस": 22, "twentytwo": 22, "twenty-two": 22,
        "तेईस": 23, "तेइस": 23, "twentythree": 23, "twenty-three": 23,
        "चौबीस": 24, "twentyfour": 24, "twenty-four": 24,
        "पच्चीस": 25, "twentyfive": 25, "twenty-five": 25,
        "छब्बीस": 26, "twentysix": 26, "twenty-six": 26,
        "सत्ताईस": 27, "twentyseven": 27, "twenty-seven": 27,
        "अठ्ठाईस": 28, "twentyeight": 28, "twenty-eight": 28,
        "उनतीस": 29, "twentynine": 29, "twenty-nine": 29,
        "तीस": 30, "thirty": 30,
        "इकतीस": 31, "thirtyone": 31, "thirty-one": 31,
        "बत्तीस": 32, "thirtytwo": 32, "thirty-two": 32,
        "तेतीस": 33, "thirtythree": 33, "thirty-three": 33,
        "चौंतीस": 34, "thirtyfour": 34, "thirty-four": 34,
        "पैंतीस": 35, "पैतिस": 35, "thirtyfive": 35, "thirty-five": 35,
        "छत्तीस": 36, "thirtysix": 36, "thirty-six": 36,
        "सैंतीस": 37, "सेंतीस": 37, "thirtyseven": 37, "thirty-seven": 37,
        "अड़तीस": 38, "thirtyeight": 38, "thirty-eight": 38,
        "उनतालीस": 39, "thirtynine": 39, "thirty-nine": 39,
        "चालीस": 40, "forty": 40,
        "इकतालीस": 41, "fortyone": 41, "forty-one": 41,
        "बयालीस": 42, "fortytwo": 42, "forty-two": 42,
        "तैंतालीस": 43, "तैतिस": 43, "fortythree": 43, "forty-three": 43,
        "चवालीस": 44, "fortyfour": 44, "forty-four": 44,
        "पैंतालीस": 45, "पैतिस": 45, "fortyfive": 45, "forty-five": 45,
        "छियालीस": 46, "fortysix": 46, "forty-six": 46,
        "सैंतालीस": 47, "fortyseven": 47, "forty-seven": 47,
        "अड़तालीस": 48, "fortyeight": 48, "forty-eight": 48,
        "उनचास": 49, "fortynine": 49, "forty-nine": 49,
        "पचास": 50, "fifty": 50,
        "इक्यावन": 51, "fiftyone": 51, "fifty-one": 51,
        "बावन": 52, "fiftytwo": 52, "fifty-two": 52,
        "तिरेपन": 53, "fiftythree": 53, "fifty-three": 53,
        "चौवन": 54, "fiftyfour": 54, "fifty-four": 54,
        "पचपन": 55, "fiftyfive": 55, "fifty-five": 55,
        "छ्प्पन": 56, "छप्पन": 56, "fiftysix": 56, "fifty-six": 56,
        "सत्तावन": 57, "fiftyseven": 57, "fifty-seven": 57,
        "अठ्ठावन": 58, "fiftyeight": 58, "fifty-eight": 58,
        "उनसठ": 59, "fiftynine": 59, "fifty-nine": 59,
        "साठ": 60, "sixty": 60,
        "इकसठ": 61, "sixtyone": 61, "sixty-one": 61,
        "बासठ": 62, "sixtytwo": 62, "sixty-two": 62,
        "तिरेसठ": 63, "sixtythree": 63, "sixty-three": 63,
        "चौसठ": 64, "sixtyfour": 64, "sixty-four": 64,
        "पैंसठ": 65, "sixtyfive": 65, "sixty-five": 65,
        "छियासठ": 66, "sixtysix": 66, "sixty-six": 66,
        "सरसठ": 67, "sixtyseven": 67, "sixty-seven": 67,
        "अड़सठ": 68, "sixtyeight": 68, "sixty-eight": 68,
        "उनहत्तर": 69, "sixtynine": 69, "sixty-nine": 69,
        "सत्तर": 70, "seventy": 70,
        "इकहत्तर": 71, "seventyone": 71, "seventy-one": 71,
        "बहत्तर": 72, "seventytwo": 72, "seventy-two": 72,
        "तिहत्तर": 73, "seventythree": 73, "seventy-three": 73,
        "चौहत्तर": 74, "seventyfour": 74, "seventy-four": 74,
        "पचहत्तर": 75, "seventyfive": 75, "seventy-five": 75,
        "छिहत्तर": 76, "seventysix": 76, "seventy-six": 76,
        "सतहत्तर": 77, "seventyseven": 77, "seventy-seven": 77,
        "अठहत्तर": 78, "seventyeight": 78, "seventy-eight": 78,
        "उन्यासी": 79, "seventynine": 79, "seventy-nine": 79,
        "अस्सी": 80, "eighty": 80,
        "इक्यासी": 81, "eightyone": 81, "eighty-one": 81,
        "बयासी": 82, "eightytwo": 82, "eighty-two": 82,
        "तिरासी": 83, "eightythree": 83, "eighty-three": 83,
        "चौरासी": 84, "eightyfour": 84, "eighty-four": 84,
        "पचासी": 85, "eightyfive": 85, "eighty-five": 85,
        "छियासी": 86, "eightysix": 86, "eighty-six": 86,
        "सत्तासी": 87, "eightyseven": 87, "eighty-seven": 87,
        "अठासी": 88, "eightyeight": 88, "eighty-eight": 88,
        "नवासी": 89, "eightynine": 89, "eighty-nine": 89,
        "नब्बे": 90, "ninety": 90,
        "इक्यानवे": 91, "ninetyone": 91, "ninety-one": 91,
        "बानवे": 92, "बयानवे": 92, "बांवे": 92, "ninetytwo": 92, "ninety-two": 92,
        "तिरानवे": 93, "ninetythree": 93, "ninety-three": 93,
        "चौरानवे": 94, "ninetyfour": 94, "ninety-four": 94,
        "पचानवे": 95, "ninetyfive": 95, "ninety-five": 95,
        "छियानवे": 96, "ninetysix": 96, "ninety-six": 96,
        "सत्तानवे": 97, "ninetyseven": 97, "ninety-seven": 97,
        "अट्ठानवे": 98, "ninetyeight": 98, "ninety-eight": 98,
        "निन्यानवे": 99, "ninetynine": 99, "ninety-nine": 99
    }

    SCALE_VALUES = {
        "सौ": 100, "sau": 100, "hundred": 100, "सैकड़ा": 100,
        "हजार": 1000, "hazar": 1000, "thousand": 1000
    }

    tokens = re.split(r"[\s\-\(\)\+\,\.]+", phone_str.strip())
    result_digits = []
    multiplier = 1
    
    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i].lower().strip()
        if not token:
            i += 1
            continue
            
        if token in ("डबल", "double", "ڈبل"):
            multiplier = 2
            i += 1
            continue
        if token in ("ट्रिपल", "triple", "ٹرپل"):
            multiplier = 3
            i += 1
            continue
            
        if token in SCALE_VALUES:
            scale = SCALE_VALUES[token]
            if result_digits:
                prev = result_digits[-1]
                if len(prev) == 1:
                    result_digits.pop()
                    scaled_val = str(int(prev) * scale)
                    result_digits.extend([scaled_val] * multiplier)
                else:
                    result_digits.extend([str(scale)] * multiplier)
            else:
                result_digits.extend([str(scale)] * multiplier)
            multiplier = 1
            i += 1
            continue

        if token in WORD_VALUES:
            val = WORD_VALUES[token]
            if i + 1 < n and tokens[i+1].lower().strip() in SCALE_VALUES:
                scale = SCALE_VALUES[tokens[i+1].lower().strip()]
                combined_val = val * scale
                i += 2
                if i < n and tokens[i].lower().strip() in WORD_VALUES:
                    next_val = WORD_VALUES[tokens[i].lower().strip()]
                    combined_val += next_val
                    i += 1
                result_digits.extend([str(combined_val)] * multiplier)
            else:
                result_digits.extend([str(val)] * multiplier)
                i += 1
            multiplier = 1
            continue
            
        cleaned = re.sub(r"\D", "", token)
        if cleaned:
            result_digits.extend([cleaned] * multiplier)
            multiplier = 1
        i += 1
        
    return "".join(result_digits)


def normalize_time_slot(time_str: str) -> str:
    if not time_str:
        return ""
    # Try parsing standard formats
    for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M", "%H:%M %p"):
        try:
            dt = datetime.datetime.strptime(time_str.strip(), fmt)
            return dt.strftime("%I:%M %p").lstrip('0')
        except ValueError:
            continue
    # Regex fallback
    m = re.match(r"^0?(\d+):(\d+)\s*(AM|PM|am|pm)?", time_str.strip())
    if m:
        hr = int(m.group(1))
        min_str = m.group(2)
        am_pm = (m.group(3) or "AM").upper()
        return f"{hr}:{min_str} {am_pm}"
    return time_str.strip()

# ---------------------------------------------------------------------------
# LANGUAGE DETECTION
# ---------------------------------------------------------------------------
# Decides, from ONE caller utterance, which language they are speaking. This is
# deliberately transcript-only: it must never look at the agent's own replies or
# at tool output, because that is exactly how the old behaviour drifted — the
# agent read its own Hindi greeting / Hindi tool strings back as "context" and
# concluded the call was in Hindi.

import re as _lang_re

_DEVANAGARI = _lang_re.compile(r"[\u0900-\u097F]")

# Romanised-Hindi function words. Content words are excluded on purpose: an
# English speaker will say "doctor", "appointment", "fever" — those are not
# evidence of Hindi. Only grammar/function words are, because they only appear
# when the sentence is actually built in Hindi.
# ---------------------------------------------------------------------------
# BUGFIX (language never switched to English): the previous version had "the"
# and "me" in BOTH marker sets. "the" is the single most common English word,
# so every ordinary English sentence scored a Hindi hit as well as an English
# one, the hi/en counts came out near-equal, and the classifier returned
# "hinglish" for plainly English turns like "Tell me the timings" or "Show me
# the slots". The agent then answered an English-speaking caller in Hinglish and
# only ever "switched" when the caller asked outright — the reported symptom.
#
# Markers are now split into STRONG (decisive on their own) and WEAK (only
# meaningful in aggregate), and NEUTRAL tokens are excluded from both counts so
# that "ok" / "haan" / "yes" carry no language weight at all.

# Romanised Hindi function words that CANNOT be English. One of these is enough
# to call a short turn Hindi ("Emergency hai", "Theek hai ji").
_HINDI_STRONG = {
    "hai", "hain", "nahi", "nahin", "kya", "kyun", "kyon", "kaise", "kaun", "kab",
    "kahan", "mujhe", "mera", "meri", "mere", "aap", "aapka", "aapki", "aapke",
    "hum", "humein", "tum", "yeh", "woh", "chahiye", "karna", "karni", "karo",
    "kariye", "karenge", "dijiye", "dijiyega", "bataiye", "batao", "bolo",
    "boliye", "raha", "rahi", "rahe", "tha", "thi", "hoga", "hogi", "mein",
    "liye", "saath", "abhi", "phir", "bhi", "milega", "milegi", "chalega",
    "dekhiye", "suniye", "namaste", "namaskar", "dhanyavaad", "shukriya",
    "kripya", "zaroor", "jaroor", "mareez", "beta", "beti", "bete", "bacche",
    "baccha", "bachcha", "bachche", "bachchi", "naam", "umar", "bukhar",
    "tareekh", "subah", "shaam", "dopahar", "aaj", "parso", "jaldi", "sahi",
    "galat", "koi", "kuch", "apna", "apne", "apni", "wala", "wali",
}

# Hindi particles that are real evidence but individually weak (short, and easy
# for a speech-to-text engine to hallucinate out of noise).
_HINDI_WEAK = {
    "kar", "ke", "ka", "ki", "ko", "se", "par", "ye", "vo", "sab", "bata",
    "bol", "din", "samay", "kal",
}

_HINDI_MARKERS = _HINDI_STRONG | _HINDI_WEAK

# Unambiguous English function words — same logic in reverse.
# NOTE: "am" is deliberately ABSENT — "10 AM" is a time, not English grammar,
# and this flow is full of them.
_ENGLISH_STRONG = {
    "the", "is", "are", "was", "were", "i", "you", "your", "my", "we", "he",
    "she", "it", "they", "this", "that", "these", "those", "and", "but", "with",
    "want", "need", "would", "could", "should", "can", "will", "do", "does",
    "did", "have", "has", "had", "what", "when", "where", "who", "how", "why",
    "there", "been", "being", "just", "also", "because", "which", "them",
}

_ENGLISH_WEAK = {
    "or", "for", "an", "a", "of", "to", "in", "on", "at", "from", "about",
    "here", "book", "me", "good", "morning", "evening", "afternoon", "night",
}

_ENGLISH_MARKERS = _ENGLISH_STRONG | _ENGLISH_WEAK

# Bilingual fillers. A caller says these interchangeably in both languages, so
# they must never be counted as evidence for either side.
_NEUTRAL_TOKENS = {
    "ok", "okay", "okey", "yes", "yeah", "yep", "no", "haan", "han", "ha", "ji",
    "hello", "helo", "hi", "hey", "thanks", "thank", "please", "sure", "sir",
    "madam", "mam", "acha", "achha", "accha", "theek", "thik", "theeg", "hmm",
    "hm", "arey", "are", "bye", "namaskaar",
}

# DEFENSIVE: whatever the literals above say, a token may never live in both
# sets. Any overlap is resolved in favour of English (Roman-script "the"/"me"
# are English far more often than they are a romanisation of Hindi), and the
# assertion below makes a future edit that reintroduces an overlap fail loudly
# at import time instead of silently degrading detection months later.
_HINDI_STRONG -= _ENGLISH_MARKERS
_HINDI_WEAK -= _ENGLISH_MARKERS
_HINDI_MARKERS = _HINDI_STRONG | _HINDI_WEAK
assert not (_HINDI_MARKERS & _ENGLISH_MARKERS), (
    "Hindi/English marker overlap: %s" % sorted(_HINDI_MARKERS & _ENGLISH_MARKERS)
)

# ── Devanagari-transliterated ENGLISH function words ─────────────────────
# The realtime ASR frequently renders Indian-accented English in Devanagari:
# "एंड व्हाट अबाउट डॉक्टर निखिल वर्मा?" is an English sentence in Hindi script.
# Without this set, every marker above is Latin-only, so ANY Devanagari in the
# transcript classified the turn as Hindi — and an English-speaking caller got
# Hindi replies for the entire call. These are function words only, on the same
# principle as the Latin sets: "डॉक्टर"/"अपॉइंटमेंट" are loanwords a Hindi
# speaker uses too and carry no signal, so they are deliberately absent.
_ENGLISH_STRONG_DEVA = {
    "द", "इज", "आर", "वाज", "वेयर", "आई", "यू", "योर", "माय", "मी", "वी",
    "ही", "शी", "इट", "दे", "देम", "देयर", "दिस", "दैट", "दीज", "दोज",
    "एंड", "बट", "विद", "विदाउट", "वांट", "नीड", "वुड", "कुड", "शुड",
    "कैन", "कैन्ट", "विल", "वोंट", "डू", "डज", "डिड", "डोंट", "हैव", "हैज",
    "हैड", "व्हाट", "वॉट", "व्हेन", "वेन", "व्हेयर", "व्हो", "हू", "हाउ",
    "व्हाई", "वाय", "व्हिच", "बीन", "बीइंग", "बी", "बिकॉज", "जस्ट", "ऑल्सो",
    "मच", "मेनी", "मोर", "मोस्ट", "एनी", "सम", "नॉट", "नो", "ओनली", "अबाउट",
    "प्लीज", "टेल", "गिव", "शो", "एक्सप्लेन", "लेट", "लाइक", "अदर", "अनदर",
    "एवरी", "बोथ", "ईच", "इफ", "देन", "दैन", "आल्सो", "ऑल", "आल",
    # Added after a second live call. All observed in real ASR output where the
    # caller was speaking English and the turn was still scored as Hindi.
    "व्हाटएवर", "वॉटएवर", "कम", "अगेन", "मेक", "इट", "प्रेफर", "वर्क्स",
    "वेलकम", "एनीथिंग", "एल्स", "मोर", "अदर", "स्पेसिफिक", "लुकिंग",
    "टेल", "गिव", "वुड", "लाइक", "हैव", "देयर", "इज", "आर", "वी", "अस",
    "सो", "दैट्स", "इट्स", "आइल", "यूअर", "माइन", "हिम", "हर", "हिज",
    "व्हिच", "व्होज", "व्होम", "अनटिल", "व्हाइल", "आफ्टर", "बिफोर",
    "बिटवीन", "अगेंस्ट", "अमंग", "ड्यूरिंग", "थ्रू", "एक्रॉस",
}
_ENGLISH_WEAK_DEVA = {
    "ऑफ", "ऑफ़", "टू", "इन", "ऑन", "एट", "फॉर", "फ्रॉम", "ऑर", "एन", "अ",
    "हियर", "बाय", "आउट", "अप", "विल", "सो", "वेल", "नाउ", "बैक",
}
_ENGLISH_MARKERS_DEVA = _ENGLISH_STRONG_DEVA | _ENGLISH_WEAK_DEVA

# Hindi function words in their native script. Previously the Devanagari branch
# never counted these either — it simply ASSUMED Hindi whenever the script
# appeared, which is the bug above. Now both sides are actually measured.
_HINDI_STRONG_DEVA = {
    "है", "हैं", "हूं", "हूँ", "नहीं", "नही", "क्या", "क्यों", "कैसे", "कौन",
    "कब", "कहां", "कहाँ", "मुझे", "मेरा", "मेरी", "मेरे", "आप", "आपका",
    "आपकी", "आपके", "हम", "हमें", "तुम", "यह", "ये", "वह", "वो", "चाहिए",
    "करना", "करनी", "करो", "कीजिए", "करेंगे", "दीजिए", "बताइए", "बताओ",
    "बोलो", "बोलिए", "रहा", "रही", "रहे", "था", "थी", "थे", "होगा", "होगी",
    "में", "मैं", "लिए", "साथ", "अभी", "फिर", "भी", "मिलेगा", "मिलेगी",
    "देखिए", "सुनिए", "नमस्ते", "धन्यवाद", "कृपया", "ज़रूर", "जरूर",
    "बच्चे", "बच्चा", "बच्ची", "नाम", "उम्र", "बुखार", "तारीख", "सुबह",
    "शाम", "दोपहर", "आज", "कल", "परसों", "जल्दी", "सही", "गलत", "ग़लत",
    "कोई", "कुछ", "अपना", "अपने", "अपनी", "वाला", "वाली", "को", "का", "की",
    "के", "से", "पर", "और",
}

# "और" is Hindi "and"; "एंड" is the transliteration of English "and". They are
# distinct tokens so there is no clash. Enforce no-overlap the same way the
# Latin sets do, so a future edit fails loudly instead of degrading silently.
_HINDI_STRONG_DEVA -= _ENGLISH_MARKERS_DEVA
assert not (_HINDI_STRONG_DEVA & _ENGLISH_MARKERS_DEVA), (
    "Devanagari Hindi/English marker overlap: %s"
    % sorted(_HINDI_STRONG_DEVA & _ENGLISH_MARKERS_DEVA)
)

_TOKEN = _lang_re.compile(r"[a-zA-Z\u0900-\u097F]+")


def detect_language(text: str | None) -> str | None:
    """Classify ONE caller utterance as 'hi', 'en' or 'hinglish'.

    Returns None when the utterance carries no usable signal (empty, or pure
    digits/proper nouns like "Rahul Sharma" or "nine eight seven..."). Returning
    None is important: a name or a phone number must NOT be allowed to flip the
    conversation language, which is what a naive detector would do.
    """
    if not text:
        return None
    raw = text.strip()
    if not raw:
        return None

    has_devanagari = bool(_DEVANAGARI.search(raw))

    tokens = [t.lower() for t in _TOKEN.findall(raw)]
    if not tokens:
        return None

    # Bilingual fillers carry no language information — drop them before
    # counting so "haan"/"ok"/"yes" can never tip the balance.
    tokens = [t for t in tokens if t not in _NEUTRAL_TOKENS]
    if not tokens and not has_devanagari:
        return None

    strong_hi = sum(1 for t in tokens if t in _HINDI_STRONG)
    weak_hi = sum(1 for t in tokens if t in _HINDI_WEAK)
    strong_en = sum(1 for t in tokens if t in _ENGLISH_STRONG)
    weak_en = sum(1 for t in tokens if t in _ENGLISH_WEAK)

    hi_hits = strong_hi + weak_hi
    en_hits = strong_en + weak_en

    if has_devanagari:
        # BUG3 FIX. This branch used to read:
        #     return "hinglish" if strong_en >= 2 else "hi"
        # and strong_en counts LATIN-script markers only. The realtime ASR writes
        # Indian-accented English in Devanagari ("एंड व्हाट अबाउट डॉक्टर निखिल
        # वर्मा?"), so strong_en was always 0 and any Devanagari at all was read
        # as Hindi. An English-speaking caller was locked into Hindi for the whole
        # call. Now the Devanagari tokens are scored on both sides.
        deva_en_strong = sum(1 for t in tokens if t in _ENGLISH_STRONG_DEVA)
        deva_en_weak = sum(1 for t in tokens if t in _ENGLISH_WEAK_DEVA)
        deva_hi = sum(1 for t in tokens if t in _HINDI_STRONG_DEVA)

        # Fold in any Latin-script markers too: mixed-script transcripts are
        # common ("appointment चाहिए", "व्हाट अबाउट the fees").
        en_total = deva_en_strong + deva_en_weak + strong_en + weak_en
        hi_total = deva_hi + hi_hits

        if en_total == 0 and hi_total == 0:
            # Devanagari, but only content words / names / numbers. No grammar to
            # judge by, so change nothing rather than guessing.
            return None
        if hi_total == 0:
            return "en"
        if en_total == 0:
            return "hi"
        # Mixed-script turns: Hindi clause + English clause in one utterance.
        # Do not let a trailing English detail block override a Hindi symptom
        # clause into pure ENGLISH unless English dominates heavily.
        _has_latin = bool(re.search(r"[A-Za-z]", raw))
        _deva_tokens = sum(1 for t in tokens if _DEVANAGARI.search(t))
        if _has_latin and has_devanagari and _deva_tokens >= 3 and en_total >= 2:
            if en_total >= hi_total * 3 and hi_total <= 1:
                return "en"
            if hi_total >= en_total * 3:
                return "hi"
            return "hinglish"
        if en_total >= hi_total * 2:
            return "en"
        if hi_total >= en_total * 2:
            return "hi"
        return "hinglish"

    if hi_hits == 0 and en_hits == 0:
        # No function words at all — a name, a number, "Sethi", "10 AM".
        # No signal. Do not guess.
        return None

    # --- one side only --------------------------------------------------
    # A single STRONG marker is decisive even on a very short turn, because a
    # strong marker cannot occur in the other language: "Emergency hai" is
    # Hindi, "the doctor" is English. The old code required 2+ markers OR 6+
    # tokens here, so both of those returned None and the caller's actual
    # language was never picked up on short turns — which is most turns on a
    # phone call.
    if hi_hits and not en_hits:
        if strong_hi:
            return "hi"
        return "hi" if (weak_hi >= 2 or len(tokens) >= 4) else None

    if en_hits and not hi_hits:
        if strong_en:
            return "en"
        return "en" if (weak_en >= 2 or len(tokens) >= 4) else None

    # --- both sides present ---------------------------------------------
    # Decide on STRONG markers first: they reflect which language the sentence
    # is grammatically built in, which is what Hinglish is actually about
    # (Hindi grammar carrying English nouns is Hindi-leaning, not 50/50).
    if strong_hi and not strong_en:
        return "hinglish" if en_hits >= 2 else "hi"
    if strong_en and not strong_hi:
        return "hinglish" if hi_hits >= 2 else "en"

    if hi_hits >= en_hits * 2:
        return "hi"
    if en_hits >= hi_hits * 2:
        return "en"
    return "hinglish"


_LANG_LABEL = {
    "en": "ENGLISH",
    "hi": "HINDI",
    "hinglish": "HINGLISH (natural Hindi-English mix)",
}


def language_label(code: str | None) -> str:
    return _LANG_LABEL.get(code or "", "NOT YET KNOWN")


# ---------------------------------------------------------------------------
# EXPLICIT LANGUAGE REQUESTS
# ---------------------------------------------------------------------------
# The detector classifies the language a sentence is WRITTEN IN. That is the
# right signal 99% of the time, but it inverts on one important case: a caller
# who asks, in English, "can you speak in Hindi?" is detected as English while
# actually requesting Hindi. This runs first and overrides the detector.

_LANG_WORDS = {
    "hi": r"(?:hindi|हिंदी|हिन्दी)",
    "en": r"(?:english|angrezi|angreji|इंग्लिश|अंग्रेज़ी|अंग्रेजी)",
}

# Speech verbs, in both scripts. \b boundaries are essential — without them 
# "me" matched inside "medium" and turned "Hindi medium school" into a request.
_SPEECH_VERB = (
    r"(?:speak|speaking|talk|reply|respond|answer|say|continue|switch|use|change|"
    r"bol|bolo|boliye|bolie|baat|bataiye|batao|bata|karo|kijiye|kariye|karein|"
    r"बोल|बोलो|बोलिए|बोलिये|बताइए|बताओ|कीजिए|कीजिये|करो|करिए)"
)

# The language word used as an ADJECTIVE on a noun is never a request:
# "a Hindi speaking doctor", "Hindi medium school", "English news channel".
_LANG_AS_ADJECTIVE = (
    r"(?:speaking|speaker|speakers|medium|school|class|classes|teacher|tuition|"
    r"movie|movies|film|channel|news|paper|newspaper|book|books|subject|"
    r"translation|translator|version|doctor|staff|report)"
)


# Python's \b is unusable here: Devanagari matras ("ी" in "हिंदी", "े" in
# "बोलिये") are combining marks, which re treats as NON-word characters, so a
# trailing \b never fires and every Devanagari request silently failed to match.
# These explicit boundaries treat the whole Devanagari block as letters, which is
# what we actually mean, and still stop "use" matching inside "house".
_LB = r"(?<![A-Za-z\u0900-\u097F])"   # left boundary
_RB = r"(?![A-Za-z\u0900-\u097F])"    # right boundary


def _build_request_pattern(lang_word: str) -> "_lang_re.Pattern[str]":
    verb = _LB + _SPEECH_VERB + _RB
    lang = _LB + lang_word + _RB
    return _lang_re.compile(
        # verb ... language   ("speak in Hindi", "switch to English")
        rf"{verb}[^.?!]{{0,30}}?{lang}"
        r"|"
        # language ... verb   ("Hindi me bolo", "अंग्रेजी में बोलिए")
        # ...but NOT when the language is just describing the next noun.
        rf"{lang}(?!\s+{_LANG_AS_ADJECTIVE}{_RB})[^.?!]{{0,30}}?{verb}"
        r"|"
        # bare polite form: "Hindi please", "English plz"
        rf"{lang}[,\s]+(?:please|plz|pls|kindly){_RB}",
        _lang_re.I,
    )


_EXPLICIT_HI = _build_request_pattern(_LANG_WORDS["hi"])
_EXPLICIT_EN = _build_request_pattern(_LANG_WORDS["en"])


def detect_explicit_language_request(text: str | None) -> str | None:
    """Return 'hi'/'en' if the caller explicitly ASKED for a language, else None.

    Deliberately narrow. It requires a speech verb next to the language name, so
    "I need a Hindi-speaking doctor" or "the report is in English" do not trip
    it — only an actual request aimed at the agent.
    """
    if not text:
        return None
    if _EXPLICIT_HI.search(text):
        return "hi"
    if _EXPLICIT_EN.search(text):
        return "en"
    return None


def resolve_caller_language(text: str | None, current: str | None) -> tuple[str | None, str]:
    """Decide the conversation language after one caller utterance.

    Returns (language, reason). `language` is None only when there is no signal
    at all AND nothing was previously set. Precedence:
      1. an explicit request from the caller  ("bol" / "speak in X")
      2. the language the utterance is actually in
      3. whatever was already in force (short/neutral turns change nothing)
    """
    explicit = detect_explicit_language_request(text)
    if explicit:
        return explicit, "explicit_request"

    detected = detect_language(text)
    if detected:
        return detected, "detected"

    return current, "unchanged"