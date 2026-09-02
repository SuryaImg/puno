# End-of-call lead capture

## The two bugs

**1. Structured fields came back null even when the caller said them.**
`create_lead()` built its record ONLY from `AppointmentState`:

```python
record = {"name": state.patient_name or state.caller_name,
          "phone": state.patient_phone,
          "package_interest": state.package_name or state.reason, ...}
```

`AppointmentState` is populated **only** when the model separately calls
`update_booking_details()`. In a normal conversation the model just talks, so every
typed field stayed null while the same information went into the free-text `notes`
argument the model passed to `create_lead`. **Nothing ever read the transcript, and
nothing ever looked at the SIP caller ID** — which is why "Sam" and "same number as
this call" ended up as prose in `notes` with `name` and `phone` null.

That also explains the asymmetry in the reported Example 1: `name` and `preferred_date`
were filled because the model happened to call `update_booking_details` for those,
while `package_interest` was not.

**2. Leads were only saved when the model chose to call `create_lead`.**
There was no unconditional end-of-call save, so hangups, transfers, confirmed bookings,
silence timeouts and errors wrote no lead at all.

## The fix

New module **`lead_capture.py`** — pure and injectable, so it unit-tests without a call:

- `render_transcript()` / `count_turns()` — the FULL conversation, tail-preserved if long.
- `build_extraction_prompt()` + `EXTRACTION_PROMPT` — field-by-field rules, explicit null
  semantics ("null ONLY if never mentioned"), relative-date resolution, and an explicit
  instruction never to describe in `notes` anything that has its own field.
- `resolve_phone()` — precedence: **spoken digits → extracted digits → SIP caller ID**.
- `classify_rule_based()` — deterministic HOT/WARM/COLD used when the LLM is unavailable.
- `build_lead_record()` — orchestrates and **always returns a record**.

New **`finalize_lead_for_call(userdata, reason)`** in `agent.py` runs on **every** call,
exactly once (claims an `_lead_finalized` flag before any await). Trigger points:

| path | when |
|---|---|
| `participant_disconnected` handler | caller hangs up (primary) |
| `_execute_end_call_drop` | agent ends the call, **before** teardown |
| `_hangup_due_to_silence` | silence timeout, **before** `session.aclose()` |
| `ctx.add_shutdown_callback` | backstop for errors / any other ending |

Ordering matters: `session.aclose()` → recorder `_on_shutdown` → `_store_clear()` wipes the
in-memory transcript, so the lead is always built *before* teardown.

`create_lead()` no longer writes a row — it records intent/notes into state so the
finalizer can use them. Otherwise one call would produce two competing lead rows.

`call_recording.get_conversation(call_id)` is the new public accessor for the coalesced turns.

## Classification rubric

- **HOT** — booking confirmed, OR clear intent + specific date/package + reachable contact
  + readiness to book. A confirmed `booking_id` is always HOT (the model cannot downgrade it).
- **WARM** — genuine interest: asked real questions or gave partial details (name, date,
  package or headcount), but no commitment.
- **COLD** — no real intent: vague/generic with no follow-through, wrong number, spam, or
  the caller ended almost immediately (≤1 caller turn short-circuits to COLD).

The LLM assigns the label with reasoning; the deterministic scorer is the fallback and the
tie-break when the label is missing or invalid. Engagement counts: ≥3 caller turns is itself
a WARM signal, so an interested caller is never dumped to COLD just because extraction failed.

## Schema

See `LEAD_RECORD_SCHEMA` in `lead_capture.py`. Every lead now carries `lead_quality`,
`reasoning`, `phone_source`, `extraction_source`, `turn_count`/`caller_turn_count` and
`end_reason` alongside the extracted fields.

## Tests

```bash
python test_lead_capture.py   # 35 checks: HOT / WARM / COLD, each also with the LLM offline
```

New env knobs: `LEAD_EXTRACTION_MODEL` (default `gemini-2.5-flash`),
`LEAD_EXTRACTION_TIMEOUT_S` (20).

---

## Post-implementation audit corrections

A 33-agent adversarial audit of the implementation above found a real race **in the fix
itself**, plus a transcript-corruption bug. Both are now fixed.

### 1. Zero grace period — the caller's last turn was lost

`finalize_lead_for_call` read the transcript store **synchronously at t=0** of the
disconnect handler (no `await` between claiming `_lead_finalized` and `get_conversation`).
But a caller turn only enters the store once the realtime provider emits a FINAL input
transcription, and the SDK explicitly documents that a provider *may withhold the final
transcript until its reply has finished generating*
(`livekit/agents/voice/agent_activity.py`).

So the exact high-value case — *"same number, book me for Saturday"* then hang up — lost
that turn, leaving the fields null and classifying a HOT call as COLD. Same symptom class
as the original bug.

**Fix:** a bounded **transcript drain** before snapshotting — poll the turn count every
100ms until it stops growing for `LEAD_TRANSCRIPT_DRAIN_QUIET_S` (0.35s) or
`LEAD_TRANSCRIPT_DRAIN_MAX_S` (1.5s) elapses. Measured cost when nothing is pending: 0.45s.
`_lead_finalized` is claimed *before* the wait, so the backstop still cannot double-fire.

### 2. The shutdown backstop ran after the store was already cleared

ctx shutdown callbacks run in registration order, and the recorder's `_on_shutdown` ends by
calling `_store_clear()`. The lead backstop was registered *after* `recorder.attach()`, so on
any ending that reached `ctx.shutdown()` without passing an explicit pre-teardown call, the
backstop would have saved an empty lead.

**Fix:** the backstop is now registered **before** `recorder.attach()`.

### 3. `_merge_text` silently deleted caller turns

Coalescing merged *any* consecutive same-role entries, and `_merge_text` returns `prev`
whenever `cur in prev`. A caller saying "yes, book it for Saturday" and later just "yes" lost
the second turn entirely — real turns vanishing from the transcript the extractor reads.

**Fix:** merging is now time-gated (`_should_merge`) to entries under
`TRANSCRIPT_MERGE_GAP_MS` (1500ms) apart — i.e. genuine streaming chunks of one utterance.
Verified both ways: streaming chunks still coalesce; a later substring turn is preserved.

### Flagged, NOT fixed (out of scope)

`_store_append` does a **blocking full-file rewrite of two files on the event loop for every
transcript append** (`call_recording.py`). During a live realtime call this is blocking I/O on
the audio path. Suggested fix: throttle to one write every ~2s, or move it off-thread — the
shutdown path already flushes unconditionally, so nothing would be lost.

### Tests
```bash
python test_lead_race.py      # late-final drain, no-late-turn latency, idempotency
```
