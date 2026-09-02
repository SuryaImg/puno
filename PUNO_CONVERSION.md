# PUNO Jaipur — Voice Agent Conversion (spec v2.3)

This codebase was originally the **Sethi Children Hospital** appointment-booking
voice agent (LiveKit + Gemini Realtime). It has been re-configured into the
**PUNO Jaipur** entertainment-centre agent per the v2.3 system-prompt spec
(Demo / POC, **no payment of any kind**, dummy confirmation SMS).

The voice/telephony plumbing (LiveKit session, Gemini realtime model, greeting
synthesis + caching, silence monitor, call recording, SIP) is domain-agnostic
and was kept. Everything domain-specific (persona, knowledge base, tools, state,
storage) was swapped to PUNO.

---

## New files

| File | Purpose |
|---|---|
| `puno_data.py` | **Single source of truth** for every PUNO fact (brand, location, timings, contact, attractions, packages, offers, membership, birthday/party, groups, safety, peak days, FAQs). Edit facts here only. |
| `puno_prompt.py` | The full system prompt (persona, tone, call flow, guardrails, system-action usage). Imports the KB from `puno_data.py` and bakes it in. Exposes `PUNO_VOICE_INSTRUCTIONS`. |
| `puno_cache.json` | Lightweight PUNO reference data (replaces `hospital_cache.json`, which was deleted). Only the legacy background-refresh path reads it; the live KB is `puno_data.py`. |

## Changed files

- **`agent.py`**
  - `_human_voice_instructions` now imports `PUNO_VOICE_INSTRUCTIONS`.
  - `build_agent_instructions()` rewritten: fills only `[Stored Fields]` / `[Missing Fields]` from the **five PUNO fields** + the language-lock directive (unchanged behaviour). All hospital data-rendering (doctors/departments/emergency/TPA) removed.
  - New PUNO tools added (info tools + the four §6 system actions) and registered on the agent; the 13 hospital tools were **unregistered** (left in the file as inert dead code — never reachable by the model).
  - `_build_speculative_notes()` → no-op (was pre-fetching hospital doctor/slot lookups).
  - `_maybe_auto_confirm_booking()` and `_hydrate_booking_draft()` → no-ops (were hospital booking/doctor-defaulting paths).
  - `build_turn_snapshot()` and `get_booking_state_label()` → PUNO fields/labels.
  - Greeting text + greeting-synthesis persona → PUNO. Closing/silence goodbye lines → PUNO.
  - Agent class `HospitalReceptionistAgent` → `PunoVoiceAgent`; `agent_name` → `puno-voice-agent`.
- **`tools.py`** — `AppointmentState` extended with PUNO fields (`kids_under7`, `adults_7plus`, `package_id`, `package_name`, `total_price`, `is_peak_day`, `booking_id`, `lead_id`, `call_category`, `call_outcome`, `payment_status`, `follow_up_needed`, `follow_up_note`, `transferred_to_human`). `CACHE_FILE` → `puno_cache.json`.
- **`storage.py`** — added PUNO persistence: `generate_booking_id`, `generate_lead_id`, `save_puno_booking`, `save_puno_lead`, `save_puno_call_log`, `save_dummy_sms`.
- **`scrapper.py`** — `CACHE_FILE` → `puno_cache.json`.
- **`sip_setup.py`, `token_server.py`** — `AGENT_NAME` → `puno-voice-agent` (kept in sync with `agent.py`).

Field reuse: the booking name reuses `patient_name`, phone reuses `patient_phone`,
date reuses `appointment_date`, and time slot reuses `appointment_time`, so the
existing persistence/turn machinery works unchanged.

---

## Tools the agent now exposes

**Info** (read verified facts from `puno_data.py`): `get_puno_info`,
`get_packages`, `get_activities`, `get_offers`, `get_membership`,
`get_birthday_info`.

**Booking / §6 system actions**: `update_booking_details` (collect the 5 fields),
`confirm_booking`, `create_lead`, `transfer_to_human`, `log_call_summary`,
`end_call`.

### `confirm_booking` behaviour (spec §5A / §6)
1. Requires name, phone, package, date, slot and ≥1 visitor — otherwise it asks
   for what's missing and does **not** book.
2. On success: generates a `booking_id` (`PUNO######`), writes a `Booked` record
   with `payment_status: n/a`, **fires the dummy confirmation SMS immediately**,
   commits state, and returns the `booking_id`. The model only speaks the
   confirmation line after a `booking_id` comes back.
3. On failure: returns a handover instruction (do **not** confirm) → `transfer_to_human`.
4. Idempotent — a second call returns the same `booking_id`.

**There is no payment channel anywhere.** No link, QR, UPI or gateway is offered
or referenced. `payment_status` is locked to `n/a`.

### Redis keys written
`puno_booking:<id>`, `puno_lead:<id>`, `puno_call_log:<id|CALL...>`,
`puno_sms:<booking_id>` (the dummy SMS record). All fail soft to process memory
if Redis is briefly unavailable.

---

## Deployment notes

1. **Re-run SIP dispatch setup.** `agent_name` changed to `puno-voice-agent`, so
   the LiveKit dispatch rule must be recreated: `python sip_setup.py`
   (delete the old `hospital-receptionist` rule first, per that script's prompt).
2. **Greeting cache.** The stale Hindi greeting WAV was deleted; the PUNO English
   greeting re-synthesizes and re-caches on the first call.
3. **Python env.** The bundled `env/` venv is from another machine and is broken
   here; run with a local interpreter that has the deps in `requirements.txt`.
4. Optional `GREETING_TEXT` env var overrides the opener; leave it unset to use
   the spec greeting.

---

## Known scope boundaries (Demo / POC)

- `transfer_to_human` attaches call context and cues the handover line but does
  **not** perform a real SIP transfer — that is a platform/telephony hook.
- `log_call_summary` is called by the model at end of call (per prompt Step 7);
  there is no on-disconnect guarantee if the caller hangs up mid-sentence.
- Peak-day dates are **flagged and confirmed** conversationally — the agent has
  no dated calendar (spec open item 1).
- The 13 legacy hospital tools remain in `agent.py` as inert, unregistered dead
  code; they can be pruned later.

See spec §7 "Open items" for the full list of items to confirm with the client
before go-live (peak-day calendar, per-ride limits, group pricing, production
payment model, membership tier mapping, WhatsApp confirmation channel, etc.).
