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
`LEAD_EXTRACTION_TIMEOUT_S` (10, now actually applied as an HTTP timeout),
`LEAD_EXTRACTION_THINKING_BUDGET` (0), `LEAD_SAVE_DRAIN_MAX_S` (15),
`SHUTDOWN_PROCESS_TIMEOUT` (30).

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

---

## Correction 2 — `puno_lead` was empty while `puno_call_log` filled up

Reported symptom: call logs were landing in Redis, leads were not.

### Cause A — teardown cancelled the lead save

`_execute_end_call_drop` gave the finalizer a 4s budget
(`END_CALL_LEAD_MAX_S`), then disconnected, on the stated assumption that the
shielded task "continues in the background". It did not. The task was registered
in `_TOOL_BG_TASKS`, and `_graceful_disconnect` cancels that entire set moments
later. `asyncio.shield` only stops the *waiter's* cancellation propagating —
cancelling the task itself still kills it. `_lead_finalized` had already been
claimed, so the shutdown backstop then skipped the retry and no lead was written
at all. The call log was unaffected because it is written mid-call.

### Cause B — the extraction could never fit in 4s

`gemini-2.5-flash` thinks by default. Measured on a real 30-turn PUNO call:
**20.5s with thinking on, 4.0s with `thinking_budget=0`** for the same extracted
fields. So the 4s budget expired on essentially every call.
`LEAD_EXTRACTION_TIMEOUT_S` was declared but never passed to the SDK, so a
stalled request had no ceiling either.

### Fix

`finalize_lead_for_call` is split into two phases:

| phase | work | must finish before teardown? |
|---|---|---|
| `_collect_lead_inputs` | transcript drain + state snapshot (~0.4s, no network) | **yes** — reads the live store |
| `_build_and_save_lead` | LLM extraction + Redis write (~5s) | no — works off the snapshot |

`background=True` (used by `end_call` and the silence hangup) awaits phase 1 only
and parks phase 2 in **`_LEAD_TASKS`** — a set teardown *awaits* rather than
cancels (`_await_pending_lead_saves`, ≤`LEAD_SAVE_DRAIN_MAX_S`). Cancelling a
lead save now also releases the `_lead_finalized` claim so a later path can retry,
and the minimal-record fallback records its `lead_id` on userdata.

Extraction runs with `thinking_budget=0` and a real HTTP timeout. A timeout is no
longer a lost lead: `build_lead_record` falls back to the deterministic classifier.

`shutdown_process_timeout` is raised from the SDK default 10s to 30s — the job's
shutdown callbacks (lead drain + the recorder's audio/transcript persistence) did
not fit in 10s, and the lead save was what got cut off.

### Also corrected

Correction 2 above claimed shutdown callbacks "run in registration order". In the
installed SDK they are **gathered concurrently**
(`ipc/job_proc_lazy_main.py`), so registering the backstop before
`recorder.attach()` does not by itself protect the read: `_store_clear()` can land
mid-drain. The drain now keeps a running best snapshot and bails out the moment
the store empties, which makes it monotonic regardless of ordering.

### Verified

Old code, replayed end_call sequence: 4s timeout → task cancelled → **no lead**.
New code, same sequence: caller released in 0.42s, lead written 5s later with the
full 30-turn transcript, `extraction_source=llm`, exactly one row. Also verified
with `_store_clear()` fired at 0.4s, mid-drain: 30 turns still saved.

### Timestamps are IST

`created_at` (and `puno_booking.created_at`, `puno_call_log.timestamp`,
`puno_sms.sent_at`, the `transcript:*` stamps, and the `call_id` suffix) were
produced with `time.strftime()` / `datetime.now()` — the **host's** clock. On the
UTC deployment VM that stored every record 5h30m behind the real call time (a lead
saved at 12:40 IST was written as `"2026-09-03 07:10:31"`), while the same code on
an IST machine looked correct — so the store held two different clocks at once.

All of them now go through a fixed +5:30 helper (`storage._now_ist_str()`, mirrored
in `tools.py` and `call_recording.py` to avoid a circular import), matching
`agent._now_ist()`. Format is unchanged — `YYYY-MM-DD HH:MM:SS` — only the value
moves, so existing consumers parse it exactly as before. **Rows written before this
change keep their original values.**
