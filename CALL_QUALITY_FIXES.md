# Call-quality bug fixes

Five reported call-quality bugs, their root causes (traced through the code and
the live model's behaviour on `gemini-3.1-flash-live-preview`), and the fixes.
All fixes are root-cause oriented with logging + guardrails and are covered by
`test_call_guardrails.py` (pure units) and `test_bug_fixes.py` (wiring).

New module: **`call_guardrails.py`** — pure, unit-testable helpers
(`mentions_payment_link`, `FixedLineDedup`, `should_fire_checkin`,
`payment_correction_line`).

---

## Bug 1 — Interrupts itself mid-sentence with "kya aap sun pa rahe hain"

**Root cause.** The silence monitor's presence-check ("are you still there?")
decided whether to speak using only `agent_state_changed` events
(`_current_a_state` / `last_speech_time`). On the Gemini 3.1 native-audio model
that event **lags or is missed** while the agent is still streaming audio, so the
12-second silence timer could mature and the monitor spoke a check-in **over the
agent's own voice**, cutting it off.

**Fix.** Added a **ground-truth "agent is producing audio" signal** independent of
that event:
- `call_recording.py` stamps `userdata["last_agent_audio_monotonic"]` on every
  non-silent agent audio frame (the agent track only emits during turns).
- A new `speech_created` handler stamps `userdata["last_agent_speech_monotonic"]`.
- The whole check-in decision moved into `should_fire_checkin(...)`, which
  **refuses to fire** while `is_agent_speaking`, or within `SILENCE_AGENT_AUDIO_GUARD_S`
  (2.5s) of the last agent audio/speech — and logs `[Bug1] Suppressed check-in`.

`agent.py`: `should_fire_checkin` gate in `silence_monitor`; `on_speech_created`;
recorder stamp. Tunable via `SILENCE_AGENT_AUDIO_GUARD_S`.

## Bug 2 — Re-asks for a detail already given (e.g., name)

**Root cause.** The per-turn "you already have name X, phone Y — do NOT ask again"
snapshot (`build_turn_snapshot` → `_inject_turn_message` into the
`on_user_turn_completed` chat_ctx) was gated behind `if not _is_gemini31:`. On the
live 3.1 model — where `update_instructions` is also disabled to avoid a websocket
restart — that meant the model got **no per-turn reminder of captured fields** and
re-asked for what it had just been told.

**Fix.** Inject the snapshot on **all** models (the `on_user_turn_completed`
chat_ctx is the supported pre-generation injection point). `_inject_turn_message`
already verifies the write and never raises. Added a `[Bug2] turn snapshot
injected=…` log. `build_turn_snapshot` was also updated (during the PUNO
conversion) to the five PUNO fields. Backed by the prompt's strong
`[Stored Fields] … DO NOT ASK AGAIN` rules.

## Bug 3 — Repeats its own previous line verbatim

**Root cause.** Check-ins, thinking fillers and the greeting all flow through
`speak_fixed_line()`, which had **no dedup**. A scheduling race (silence loop
firing twice before the in-flight flag flips, a filler overlapping a check-in)
could speak the **same fixed line twice** back-to-back.

**Fix.** A per-session `FixedLineDedup` (in `userdata`) suppresses an exact repeat
of a fixed line within `FIXED_LINE_DEDUP_WINDOW_S` (8s) and logs
`[Bug3] Suppressed duplicate fixed line`. (The model's own free-text repetition is
separately constrained by the prompt's NO-REPETITION rules.)

## Bug 4 — Says "I'll send a payment link so you can pay online as well"

**Root cause.** This line is **not in any code path** — not in the prompt, tools,
data, or even the pre-conversion hospital backup. It can only reach the caller as
an **LLM hallucination** (booking assistants commonly offer payment links). This
build takes no payment at all.

**Fix (defense-in-depth).**
1. Prompt: an explicit, razor-specific prohibition of this exact phrasing (and
   variants) in `puno_prompt.py`.
2. Runtime backstop: a `conversation_item_added` handler runs the agent's own
   transcript through `mentions_payment_link()` (English/Hindi/Hinglish, negation-
   aware so "there's no online payment" is *not* flagged). On a hit it logs
   `[Bug4] Agent produced a forbidden payment-link…` and immediately speaks a
   correction (`payment_correction_line`) so wrong info is overridden.

## Bug 5 — Abruptly stops mid-response and doesn't continue

**Root causes.** (a) The interruption threshold was very aggressive
(`min_duration 0.45s`, `min_words 1`) so a brief echo of the agent's own voice or
background noise counted as a barge-in and cut the agent off. (b) False
interruptions weren't explicitly set to auto-resume. (c) There were **no handlers**
for the session's `error` / `close` events, so failures were silent. (d) No
recovery if a turn got stuck generating.

**Fix.**
- Interruption tuning in `_turn_handling_for` (3.1): `min_duration 0.6`,
  `min_words 2`, explicit `resume_false_interruption: true`,
  `false_interruption_timeout: 2.0` — all env-overridable.
- New `error` and `close` handlers log the failure (`[Bug5]`) instead of failing
  silently; an `agent_false_interruption` handler logs whether it auto-resumed.
- A **stuck-turn watchdog** in the silence monitor: if the agent sits in
  `thinking` for `THINKING_STUCK_S` (20s) with no audio, it logs `CRITICAL` and
  recovers by clearing the stale thinking state so a check-in can re-engage the
  caller instead of sitting mute.

---

## Tests

```bash
python test_call_guardrails.py   # payment detector, fixed-line dedup, check-in gate
python test_bug_fixes.py         # wiring: snapshot on 3.1, dedup in speak_fixed_line, prompt + handlers
```

## New / changed env knobs

`INTERRUPT_MIN_DURATION` (0.6), `INTERRUPT_MIN_WORDS` (2),
`RESUME_FALSE_INTERRUPTION` (true), `FALSE_INTERRUPTION_TIMEOUT_S` (2.0),
`SILENCE_AGENT_AUDIO_GUARD_S` (2.5), `THINKING_STUCK_S` (20.0),
`FIXED_LINE_DEDUP_WINDOW_S` (8.0).

---

# Round 2 — persona/voice mismatch and the audibility filler

## Issue 1 — "kya main aapki madad kar **sakta** hoon" in a female voice

**Cause: MODEL BEHAVIOR (prompt gap). Not TTS, not config, not a platform default.**

Evidence from the five real calls recorded on 2026-09-02:

| call | first-person forms |
|---|---|
| `vuUCQAbXcuJ9` | **masculine** — "madad kar sakta hoon" ×3, "main yeh keh raha tha" |
| `EHBZ5UXP4fjW`, `5GSbj4F94Xqz`, `zK89KnADUfRc`, `xroi9fy2etxc` | feminine — "kar sakti hoon", "bata deti hoon", "bhej doongi" |

Hindi inflects the **first person** for gender. The TTS voice is a single fixed
female voice, but the PUNO prompt said **nothing** about which grammatical gender
the agent should use for itself, so the model picked one per call. On the one call
where it picked masculine, the caller heard a female voice speaking masculine
grammar — which reads as "the wrong voice/persona".

This was a regression introduced in the PUNO rewrite: the original hospital prompt
contained *"You are female — use feminine verb forms ('bol rahi hoon', 'karti hoon')"*,
and that instruction was not carried over.

**Voice config was audited and is clean:**
- Exactly one `RealtimeModel(...)` construction, and it always passes `voice=voice_name`.
- `GEMINI_VOICE` defaults to `Vindemiatrix`; every synthesis site uses the same value.
- The greeting cache validates text **and** voice **and** model before reuse.
- No second TTS provider — `tts=` is deliberately absent from `AgentSession`.
- The Gemini plugin's default voice is `Puck` (male-sounding) **if `voice` were omitted** — it never is.

**Fixes**
1. `puno_prompt.py` — new `=== YOUR VOICE & GRAMMATICAL GENDER ===` block with explicit
   ALWAYS/NEVER form lists, injected via a `[PERSONA GENDER]` placeholder and driven by
   `AGENT_GENDER` (default `female`) so it stays in sync with `GEMINI_VOICE`.
2. `call_guardrails.py` — `hindi_self_gender()` / `gender_drifted()` detectors.
3. `agent.py` — the output guard now logs `[Issue1] Hindi self-reference gender drift…`
   with the found/expected gender and the configured voice.
4. `agent.py` — the one `session.say(text, ...)` that omitted `audio=` (a last-resort
   fallback, unreachable on 3.1) now routes through `_speak_fixed_line_audio`, so **no**
   audio path can render in an unpinned voice.

## Issue 2 — "kya aap mujhe sun pa rahe hain" during a call

**Cause: CODE (hardcoded string).** The previous round gated *when* the check-in
could fire but left the phrases in place. `_CHECKIN_BY_LANG` in `agent.py` still held
"Hello, kya aap mujhe sun pa rahe hain?" / "Hello, are you able to hear me?", and the
silence monitor spoke one after `SILENCE_FIRST_CHECKIN_S` (12s) of silence — which a
caller pausing to think experiences as "during active conversation". It never appeared
in `transcripts/` because it is spoken with `add_to_chat_ctx=False`.

**Fixes**
1. `_CHECKIN_BY_LANG` **deleted**. Silence re-engagement now comes from
   `call_guardrails.reengagement_line()`, whose **first attempt is an empty string —
   the agent stays silent and simply waits.** Later attempts are natural invitations
   ("Take your time — I'm here whenever you're ready.") that never mention hearing.
2. `mentions_audibility_question()` screens every line before it is spoken; a match is
   logged at ERROR and **not** spoken, so the phrase cannot be reintroduced by an edit.
3. `puno_prompt.py` — absolute negative instruction ("NEVER ask the caller whether they
   can hear you… under ANY circumstance", listing EN/HI/Devanagari variants), the
   "I can't hear you" loophole closed, and the lines that *told* the model a background
   check-in exists removed.
4. `agent.py` — a full trigger-state log before every re-engagement:
   `[Issue2][SilenceReengage] attempt=… elapsed=… u_state=… a_state=… is_agent_speaking=… agent_audio_age=…`
5. Dead code hardened: `silence_monitor.py` (not imported by `agent.py`) and
   `agent_silence_integration_example.py` had "Are you there?" prompts and an unvoiced
   `say()`; both now carry banners, safe defaults, and runtime screening.

## Tests
```bash
python test_issue_fixes.py     # 57 checks: normal flow, mid-call pause, network/stuck turn, real-call fixtures
```
New env knob: `AGENT_GENDER` (`female` default) — keep in sync with `GEMINI_VOICE`.

---

## Round 2b — correction: fixed lines really WERE in a different voice

A 52-agent adversarial audit overturned part of the Round 2 conclusion above.
The earlier statement "every audio path uses the same voice" was **wrong**: they
all pass the same voice *name*, but not the same *engine*.

**`_synthesize_fixed_line_pcm` rendered every fixed line through batch TTS**
(`client.models.generate_content` against `gemini-2.5-flash-preview-tts`), while the
conversation itself is voiced by the `gemini-3.1-flash-live-preview` **native-audio
live model**. Same `voice_name=Vindemiatrix`, different synthesis engine — which does
not sound like the same person. The codebase already documented this hazard for the
greeting (`agent.py`: *"TTS shares the voice name but does not sound like the live
agent"*) and the greeting path guards against it by trying the live model first — but
the fixed-line path had **no live branch at all**.

So every silence re-engagement, thinking filler, goodbye and payment correction was
spoken in a different voice from the rest of the call, several times per call.

**Fix.** `_synthesize_fixed_line_pcm` now tries `_synthesize_greeting_pcm_live`
(the same engine as the call) first, falling back to batch TTS only with a logged
warning, and caches PCM per `(text, voice, live-model)` — which also removes a live
API round-trip from every filler.

### Other confirmed defects fixed in this pass

| # | Defect | Fix |
|---|---|---|
| 1 | `_lang_now()` defaulted to **`"hi"`**, so an English caller whose language wasn't yet detected got the **Hindi** line (the `"en"` default in every `.get()` was unreachable) | defaults to `"en"`, matching the English opener |
| 2 | A **missing `.greeting_cache.meta` skipped ALL cache validation** — any stale WAV on disk was played to callers | fails closed and refuses to play an unvalidated WAV |
| 3 | Voice validation was skipped whenever either side was blank | both required, else reject |
| 4 | The cache stamped `live:{model}` **unconditionally**, so a TTS-rendered greeting was recorded as live-rendered and never rebuilt | real provenance recorded via `_LAST_GREETING_ENGINE` (`live:` / `tts:`) |
| 5 | The Round-1 stuck-turn recovery **could never re-engage** — the `unanswered_user_turn` guard blocked the very check-in it was written to enable | `force_recovery=True` bypasses that one guard; the agent-audio guards still hold |
| 6 | Five separate hardcoded `"Vindemiatrix"` literals could drift apart; `GEMINI_VOICE` wasn't in `.env` | single `_configured_voice()`; `GEMINI_VOICE` + `AGENT_GENDER` now explicit in `.env` |

**Confirmed NOT a cause (no provider-side change needed):** neither livekit-agents
1.7.1 nor livekit-plugins-google contains any built-in spoken utterance that bypasses
the app's prompt/voice. The Gemini plugin's default voice is `Puck`, used only if
`voice` were omitted — the app always passes it.

New env knobs: `GEMINI_VOICE`, `AGENT_GENDER`, `FIXED_LINE_LIVE_TIMEOUT_S` (12.0).

---

# Round 3 — greeting/name, availability, no-guessing, and hang-up

## Issue 1 — new greeting + agent name "Rahi" — CONFIG/PROMPT

The greeting lived in **four** places, all updated so the old wording cannot resurface:

| where | was | now |
|---|---|---|
| `agent.py` `_DEFAULT_GREETING_TEXT` | "Hi, thanks for calling PUNO Jaipur! I'm Puno, your virtual assistant…" | "Hi! This is Rahi from PUNO Jaipur. How may I assist you?" |
| `agent.py` greeting-synthesis `system_instruction` | "You are Puno, a warm, upbeat front-desk host…" | "You are Rahi, …" |
| `puno_prompt.py` IDENTITY | "You are Puno, the AI voice assistant…" | "You are Rahi, the voice assistant…" |
| `puno_prompt.py` OPENING LINE | quoted the old line | quotes the new line |

Also added an explicit self-identification rule: *"Your name is Rahi… Never call yourself
'Puno' — PUNO is the venue's name, not yours."* The `GREETING_TEXT` env override is unset
(so the new default applies) and the cached greeting WAV was cleared so it re-synthesises.
Brand references (PUNO Jaipur, "Let's get you Punofied!") are deliberately unchanged.

## Issue 2 — decline unavailable games — PROMPT + missing data check

There was no notion of "not offered" anywhere: the KB listed what exists, and the model was
left to infer the rest, so it hedged instead of declining.

- `puno_data.py`: `find_activity()`, `activity_availability()`, `available_activity_names()`
  — matching on attraction names, descriptions and spoken aliases ("jumping", "kids area").
  Returns `available` / `not_offered` / `uncertain`.
- New **`check_activity(activity)` tool** returns a directive answer: for `not_offered` it
  states the decline is definitive and supplies alternatives; it explicitly forbids "maybe".
- Prompt gained a **closed list** section rendered from `ATTRACTIONS`, so the list can never
  drift from the KB, plus the rule to decline clearly and offer 1-2 alternatives.

A small `AMBIGUOUS_ACTIVITIES` set (go-karting, bumper cars, simulators) deliberately returns
`uncertain` rather than a wrong denial — those route into the Issue-3 behaviour instead.

## Issue 3 — stop guessing — PROMPT

The only existing rule was one soft line ("If you are unsure… NEVER guess"), which lost to the
much louder *"NEVER GO SILENT / keep the call moving / every turn ends in a question"* rules —
so the model filled gaps to stay helpful.

New **`=== WHEN YOU DO NOT KNOW ===`** section that explicitly **OUTRANKS** those rules, bans
the hedge vocabulary by name ("probably", "roughly", "I think", "it should be around"), forbids
reasoning a plausible number out of known packages, and prescribes the exact reply: *"I don't
have that exact detail with me — let me note it down and have our team confirm it for you."*

## Issue 4 — the call never hung up — CODE (two bugs)

**Cause A (pre-existing, primary).** `_execute_end_call_drop` waited on an **unbounded** loop:

```python
while userdata.get("is_agent_speaking"):
    await asyncio.sleep(0.05)
```

`is_agent_speaking` is set from `agent_state_changed` — the same event we already proved
lags/misses the "stopped speaking" transition on Gemini 3.1 native audio (Round 1, Bug 1).
Whenever that event was missed the loop spun **forever** and the call was never disconnected:
the agent said goodbye and stayed on a live line. Now bounded by two independent escapes —
the ground-truth `last_agent_audio_monotonic` stamp (no agent audio for
`END_CALL_AUDIO_QUIET_S`=1.2s means speech finished, whatever the flag says) and a hard
`END_CALL_SPEECH_MAX_WAIT_S`=15s ceiling, both logged.

**Cause B (introduced by the lead-capture work).** The unconditional lead save was awaited
between that wait and `room.disconnect()`, and it performs a transcript drain plus an LLM
extraction — holding the caller on the line for seconds after goodbye. Now time-boxed to
`END_CALL_LEAD_MAX_S`=4s and `asyncio.shield`ed, so the save completes in the background
while the call drops on time.

Prompt side: the wrap-up now spells out that **saying goodbye does not hang up** — the
`end_call` tool must be called in the same turn, and is the only thing that ends the call.
The silence-hangup path (3 ignored check-ins) remains as the fallback.

## Tests
```bash
python test_four_issues.py    # 58 checks across all four issues
```
Issue 4's regression test drives `_execute_end_call_drop` with `is_agent_speaking` **stuck
True** — the real failure mode — and asserts the room still disconnects, both via the
audio-quiet escape and via the hard ceiling.

New env knobs: `END_CALL_SPEECH_MAX_WAIT_S` (15), `END_CALL_AUDIO_QUIET_S` (1.2),
`END_CALL_LEAD_MAX_S` (4).

---

# Round 4 — the genai routing root fix

## The defect

`agent.py` stated the rule, at `:7537`, **nested inside `entrypoint()`**:

> *"Native-audio / Gemini 3 Live models must use the Gemini API, not Vertex."*

`_build_realtime_model` applied it, so the live **session** worked. But the module-level
synthesis helpers are defined thousands of lines earlier and could not reach it — so each
of them hand-rolled the decision. Seven `genai.Client()` constructions across four call
sites, one correct.

`_synthesize_greeting_pcm_live` read `GOOGLE_GENAI_USE_VERTEXAI` directly and asked Vertex
for `gemini-3.1-flash-live-preview`, which is not a Vertex publisher model → websocket
**1008** on every attempt → 4 retries → fallback to 2.5 → cache stamped
`live:gemini-live-2.5-flash-native-audio` → the cache check demanded an exact match to 3.1
→ rejected → **re-synthesised on every prewarm and every call**. 32-second startups;
callers hung up before the agent spoke.

## The fix

**New module `genai_routing.py`** — the single decision point:

- `_model_uses_gemini_api(model)` — lifted to module scope, extended to cover the preview
  **TTS** families (which were also being pointed at Vertex).
- `_genai_client_for(model)` — every Gemini client in the project is now built here.
- `_is_permanent_model_error(exc)` — distinguishes never-retry failures from transient ones.

**Call sites rerouted:** `_synthesize_greeting_pcm_live` (the 1008 source),
`_synthesize_fixed_line_pcm_tts`, the greeting TTS fallback, and
`lead_capture._gemini_generate`. Two dead `from google import genai` imports removed. The
duplicate nested rule deleted. `grep genai.Client(` now returns **one file**.

**Cache check** (`_resolve_greeting_cache_wav`) compares **engine class**, not exact model:
any `live:` marker is reused, `tts:` is still rejected. The text / voice / rate / channel
guards are untouched. This is what makes the system self-healing — a one-off fallback render
no longer causes permanent regeneration.

**Retry loop** aborts a model on a permanent error instead of burning 4 attempts, while still
falling through to the fallback model.

### The subtlety that would have made this a no-op

`genai.Client(api_key=...)` **still returns a Vertex client** when
`GOOGLE_GENAI_USE_VERTEXAI=true`, because the SDK reads that env var at construction time.
Empirically confirmed: all four model families still came back `vertexai=True` after the
first pass. `vertexai=False` must be passed **explicitly** — which is exactly what
`_build_realtime_model` does, and why the session path worked all along.

`gemini-2.5-flash` (lead extraction) deliberately still honours the env var — no evidence it
is broken on Vertex, so that path is unchanged.

## Tests
```bash
python test_genai_routing.py    # 39 checks
```
Covers family routing, the explicit `vertexai=False`, permanent-vs-transient classification,
cache class-matching (incl. that TTS/unmarked/wrong-text/wrong-voice are still rejected), and
that exactly one file constructs a client.

## Verifying on the next real call
`Live greeting attempt 1` succeeds on **3.1** with no 1008s → meta reads
`live:gemini-3.1-flash-live-preview` → the *second* call logs
`Opening greeting will play from cache` with no synthesis → `[startup] setup complete` under
~1s instead of 32s. The stale 2.5-rendered cache was cleared so the first call re-renders.

## Round 4b — correction: the first predicate broke the fallback model

A 33-agent audit found a **blocker in the fix above**, reproduced and confirmed by
executing the SDK's own validator.

The first predicate used `"native-audio" in name` as a discriminator. But the plugin's
own tables say:

```python
KNOWN_VERTEXAI_MODELS   = {"gemini-live-2.5-flash-native-audio"}          # Vertex ONLY
KNOWN_GEMINI_API_MODELS = {"gemini-3.1-flash-live-preview",
                           "gemini-2.5-flash-native-audio-preview-12-2025"}
```

`gemini-live-2.5-flash-native-audio` contains "native-audio" but is **Vertex-only** — and
it is this project's `_GEMINI_FALLBACK_MODEL`. `RealtimeModel.__init__` calls
`_validate_model_api_match()` and **raises ValueError** when the family and the `vertexai`
flag disagree, so misclassifying it would have:

1. made `_build_realtime_model(fallback)` raise → `_start_with_fallback`'s second attempt
   dies before `session.start()` → **the call has no fallback at all**; and
2. pointed the greeting's live fallback at `generativelanguage` for a Vertex-only model —
   the exact mirror of the original 1008 bug.

Worse, the first version of `test_genai_routing.py` **asserted the wrong classification**,
so the suite went green over the defect.

**Correction.** The predicate now seeds from the SDK's own
`KNOWN_VERTEXAI_MODELS` / `KNOWN_GEMINI_API_MODELS` (imported lazily, with the known values
as a static fallback), and only falls back to the family rule the SDK documents —
`gemini-live-*` is Vertex, `gemini-<N.N>-*` is Gemini API — for models the plugin does not
list. Net effect: primary `gemini-3.1-flash-live-preview` → Gemini API (fixes the 1008),
fallback `gemini-live-2.5-flash-native-audio` → Vertex (restores the working path).

**New permanent guard:** `test_agrees_with_sdk_validator()` runs the plugin's real
`_validate_model_api_match()` against our routing for **every** model the SDK knows. If the
project ever disagrees with the plugin again, that test fails instead of a live call.

Tests: 42 checks in `test_genai_routing.py`.

---

# Round 5 — hospital-era and test code removed

The project was converted from a hospital appointment agent, and the hospital code was
left in place (unregistered) rather than deleted. It has now been removed, along with the
regression test files.

| file | before | after |
|---|---|---|
| `agent.py` | ~8,900 lines | **4,715** |
| `tools.py` | ~1,900 | **1,065** |
| `storage.py` | ~1,300 | **520** |

**Deleted outright:** `scrapper.py`, `puno_cache.json`, `silence_monitor.py`,
`agent_silence_integration_example.py`, `IMPLEMENTATION_SUMMARY.md`,
`SILENCE_MONITOR_INTEGRATION_GUIDE.md`, `SILENCE_MONITOR_QUICKSTART.md`,
`debug_recording.txt` (a runtime log — it is recreated), and the seven `test_*.py` files.

**Removed from `agent.py`:** the 13 unregistered hospital tools (booking, slots, doctors,
emergency, reschedule/search — 2,557 lines) plus ~40 helper functions that became orphaned,
removed iteratively until the graph was stable.

**Removed from `storage.py`:** slot locking, confirmed-appointment and emergency-case
persistence, doctor-id resolution, seller-url storage, and their transitive helpers.

**Removed from `tools.py`:** doctor/department/symptom classification, `DoctorNameStr`,
appointment validation and the cache loader.

## Three things that were EXECUTING, not just dead

These ran on every call and were removed deliberately — each is a small behaviour change:

1. **`confirmed_appointment:*` mirror.** `save_appointment_state` wrote a hospital-shaped
   record for every `BOOKED` state. Nothing read that namespace (its only reader,
   `search_confirmed_appointments`, was itself dead). A PUNO booking is written by
   `confirm_booking()` to `puno_booking:*`.
2. **`patient_profiles` map.** `save_current_profile_to_map` ran on every state save and
   was **write-only** — nothing ever read it back.
3. **Per-turn hospital-data refresh.** `_background_refresh_hospital_data` reloaded a JSON
   cache on every user turn that nothing read; PUNO's knowledge base is static in
   `puno_data.py`.

## Deliberately KEPT

- **`AppointmentState`'s hospital fields** (`doctor_name`, `department`, `age`,
  `appointment_id`, …). Removing them risks breaking pydantic deserialisation of state
  already persisted in Redis. They are unused and default to `None`.
- Field names reused with PUNO meaning: `patient_name` = customer, `appointment_date` =
  visit date, `appointment_time` = slot. Renaming is a separate, riskier refactor.
- Historical comments explaining why code is shaped the way it is.

## Verification

Every removal step was verified with compile + import + the full test suite, and the tests
were deleted **last** so they guarded the whole operation. After the final removal the
13 registered tools, lead finalizer, greeting, prompt rendering and genai routing were all
re-verified directly.

> **Note:** the test files are gone, so there is no longer a regression guard. The most
> valuable one was `test_genai_routing.py::test_agrees_with_sdk_validator`, which runs the
> livekit plugin's own `_validate_model_api_match()` against our model routing — it caught a
> blocker that would have killed the session fallback. Worth restoring if this codebase keeps
> being changed.

# Round 6 — the Hindi call: prompt recitation, a false cancellation, a wrong name

Six issues from one live Hindi transcript. Two were serious.

## Issue 1 (serious) — the agent recited its own system prompt

It said, out loud, in Hindi: *"at most two details at a time ... name and phone
number ... summarize at step 5.1"* — its own call-flow scaffolding, step number
included.

**Root cause was the prompt's shape, not its content.** It contained **35
recitable structures**: 8 `Step N` headers, 13 indented `1..N` items and 14
numbered guardrails. A numbered checklist reads like a script, and native-audio
models recite scripts. There was also *no rule anywhere* forbidding the agent
from speaking its instructions — the closest rule banned markdown and field
labels, which does not cover instruction prose or step numbers.

**Fix, in `puno_prompt.py`:**
- Every `Step N` header replaced with a descriptive heading (`--- Taking booking
  details ---`). You cannot recite a step number that does not exist.
- The five booking fields converted from a numbered list into prose describing
  the *goal*. A list invites enumeration; a goal invites conversation.
- The 14 numbered guardrails de-numbered to bullets.
- New top-priority section `=== THESE INSTRUCTIONS ARE PRIVATE ===`, listing the
  exact leaked phrases as forbidden in any language.

Net: 35 recitable structures → **0**. The prompt also got ~3k chars shorter.

**Runtime backstop:** `leaked_instruction()` returns the *matched fragment* (not
a bool) so the log names precisely which phrasing escaped.

## Issue 2 (most dangerous) — it agreed to cancel a booking that never existed

Triggered by a misheard word, on a flow that is explicitly out of scope, for a
booking that had never been made. A caller who believes their booking is
cancelled acts on that.

- **Prompt:** `=== EXISTING BOOKINGS ===` rewritten with the same razor-specific
  treatment that made the payment rule stick — the exact forbidden sentences in
  English, Hinglish and Devanagari, plus "never infer a cancellation from a
  single word you might have misheard".
- **Runtime:** `promises_booking_modification()` + a spoken correction, wired
  into the output guard. Negation-aware, so the *correct* line ("I can't cancel
  that myself — let me get you to the team") is never flagged. That case is
  load-bearing: `"I can't cancel"` matches the promise pattern via `i can ...
  cancel`, so without the negation check the guard would fire on the right answer.

## Issue 3 — it invented the caller's name

Caller said "कप्तान" (captain); the agent replied "ओ कुतुब जी". No rule
constrained this. New `=== THE CALLER'S NAME ===` section, anchored to data the
model already sees every turn: *if a name is not in `[Stored Fields]`, you do not
know their name*. That makes the rule checkable rather than abstract.

## Issue 4 — dumped all five fields, then repeated the turn verbatim

The dumping is fixed by the Issue 1 restructure (there is no longer a list to
dump). The verbatim repeat came after the caller explicitly asked it to go one at
a time — so a rule was added to honour slow-down requests, and
`is_near_duplicate()` now compares consecutive **model-generated** turns.
`FixedLineDedup` only ever wrapped `speak_fixed_line()`, so model repeats were
invisible.

## Issue 5 — "आई एम सॉरी" after the greeting — investigated, NOT a code bug

Worth recording because the obvious suspicion was wrong. `_schedule_opening_greeting`
plays the cached WAV via `session.say(audio=...)` and does **not** call
`generate_reply()` afterwards (agent.py has an explicit comment guarding against
adding one). So nothing forces a turn with no caller input. This is the model
reacting to unclear first input, which makes it prompt-side and inherently
best-effort — the existing "never open with an apology" rule is the only lever.

## Issue 6 — truncated final sentence

Already-tuned barge-in thresholds (`INTERRUPT_MIN_DURATION`, `INTERRUPT_MIN_WORDS`)
are the lever; env-only, no code change.

## Handler rename

`on_conversation_item_payment_guard` → `on_conversation_item_output_guard`. It now
runs five checks on every agent turn (gender drift, audibility, instruction leak,
self-repetition, cancellation promise, payment link), so the old name misdescribed it.

## Verification

All six modules compile; `build_agent_instructions()` renders with every
placeholder filled and zero leftover numbering. Guard regexes were checked against
both must-flag and must-NOT-flag corpora — the false-positive set (correct handover
lines, confirmations, opening hours) is clean. The transcript's own offending lines
were replayed through the full chain: leakage, cancellation promise and repetition
all now trip a guard.

# Round 7 — the agent went silent for 55 seconds mid-call

Reported: "agent stopped in the middle of the conversation". Recording:
`recordings/hospital-call-_07013008161_KCFHwLEqV74A-20260902-182110.wav`.

## What happened

    12:51:38  agent  "...ya fir aapko packages ke baare mein janna hai?"
    12:51:48  CALLER "जानना है।"                        <- answered the question
              ...agent produces nothing...
    12:52:05  CALLER "silent hai... Hello, jaanna hai batao. Tum log bolegi kaise? Hello, hello."
              ...agent STILL produces nothing...
    12:52:44  CALLER "Hello."
    12:52:44  agent  finally replies (14 ms later)

**55.5 seconds of dead air**, during which the caller spoke three times and could not
wake the agent.

Confirmed independently from the audio, not just the transcript: an RMS scan of the
149-second WAV finds **every quiet stretch >= 3s falls inside the stall window
(37s-93s)** — none before it, none after. The last one ends at 93.0s; the agent's
audio resumes at 93.28s. The call was fluent on both sides of a dead zone.

## Root cause — the recovery was unreachable exactly when it was needed

`should_fire_checkin()` in `call_guardrails.py` contains:

```python
if last_user_after_agent and not force_recovery:
    return (False, "unanswered_user_turn")
```

The check-in is blocked **because** the agent owes a reply. The only key that
unlocks it is `force_recovery`, which is set in exactly one place: the stuck-turn
watchdog, nested under `if a == "thinking":` and requiring `thinking_since > 0`.
Both derive from `agent_state_changed` — the event Gemini 3.1 native-audio is
documented to lag and drop.

So when the model silently fails to generate, no "thinking" event arrives,
`force_recovery` stays False, and the guard blocks **permanently**. Replaying the
real function with the observed call's inputs:

    t=70s    (False, 'unanswered_user_turn')
    t=300s   (False, 'unanswered_user_turn')
    t=3600s  (False, 'unanswered_user_turn')

An hour in, still blocked. The agent would have sat mute until the caller hung up.

**And it was self-reinforcing.** `user_txt_age < required_silence` gates the same
function, so every "hello?" the caller tried re-armed the block. The caller's own
attempts to get attention are what kept the agent silent.

## Second defect — the recovery said the wrong thing

`reengagement_line()` answers the OPPOSITE situation: the caller has gone quiet and
is entitled to think. Attempt 0 is deliberately an empty string, and the later lines
say *"Take your time — I'm here whenever you're ready."* Spoken into an agent stall
that is actively wrong: the caller is not taking their time, they are asking "are you
there?" while waiting on us. One monitor was serving two opposite situations.

## Third defect (pre-existing, found while fixing) — a guard that never worked

`last_agent_audio_monotonic` was initialised to `0.0` and **written by nothing**,
while being read in four places. `agent_audio_age` was therefore always `now - 0.0`
— a number that can never fall below the guard. The "GROUND-TRUTH agent-audio guard"
that Bug 1 added, and whose comments claim it is what stops the agent talking over
its own voice, has been **inert since the day it was written**.

This also nearly broke the new watchdog: reading `0.0` would make "a reply is owed"
permanently true and fire spurious recovery lines ~7s after every agent turn.
`speech_created` is the genuine signal — the LiveKit plugin emits it from `say()`,
`_generate_reply()` and `_on_generation_created`, so it covers fixed lines *and*
model output, and is independent of `agent_state_changed`. It is now stamped there.

## The fix

**A turn-owed watchdog** (`agent.py`, silence_monitor step 1b) that reads ONLY ground
truth — the caller's last transcript against the agent's last actual audio — and
consults nothing derived from `agent_state_changed`. If a reply has been owed for
`TURN_OWED_TIMEOUT_S` (default 7.0) and the agent has produced no audio, it logs
CRITICAL, speaks a stall-recovery line, and sets `force_recovery` so the ladder below
is no longer dead-locked.

`_agent_audio_at` takes `max()` of every available "agent produced output" stamp.
max() is the safe direction: a missing signal can only make the agent look *more*
recently active, so a dropped event yields a late recovery, never a spurious one
spoken over a healthy call.

**`stall_recovery_line()`** (`call_guardrails.py`) — never empty at attempt 0, owns
the pause, hands the turn back, and offers a human on the last attempt. Feminine
Hindi self-reference to match the voice; screened through
`mentions_audibility_question()` like every other fixed line.

**Bounded**: after `MAX_STALL_RECOVERIES` (default 3) it stops speaking but keeps the
block armed — repeating "let me get you to a team member" every 7 seconds is worse
for the caller than the silence it is fixing. No second hang-up path was added (R6).

## Verification

Simulated against the real timeline and against healthy calls:

    healthy call            -> no recoveries          (correct)
    caller thinks 25s       -> no recoveries          (correct - that is caller silence)
    slow-but-ok reply (5s)  -> no recoveries          (correct - under budget)
    THE OBSERVED STALL      -> recovery at 45.0s, 62.0s
    wedged session          -> speech capped at 3, then quiet

Dead air on the reported call: **55.5s -> 7.0s (87% reduction)**. All six modules
compile, the live prompt still renders with every placeholder filled, the Round 6
guards are unchanged, and the AST scan reports no dangling references.

## Tunables

    TURN_OWED_TIMEOUT_S   default 7.0   how long a reply may be owed before recovery
    MAX_STALL_RECOVERIES  default 3     spoken recovery attempts before going quiet

## Round 7 addendum — two collateral defects (found by audit, NOT yet fixed)

A 33-agent adversarial audit of the stall independently reproduced the
`last_agent_audio_monotonic` dead-guard above, and surfaced two further defects.
Both were verified by hand against the shipped SDK source. Neither causes the
stall; both are recorded here because they are live correctness bugs.

**1. `INTERRUPT_MIN_DURATION` / `INTERRUPT_MIN_WORDS` are inert on Gemini 3.1.**
`agent.py:3487` sets `turn_detection: "realtime_llm"`, and the SDK's
`_interrupt_by_audio_activity()` (agent_activity.py:2122-2125) returns
`if self._rt_turn_detection_enabled` — *before* it ever reads `interruption_options`.
`min_words` is additionally gated on `self.stt is not None` (agent_activity.py:2128),
and this session has no `stt=` at all; `min_duration` is read only in
`on_vad_inference_done`, which returns early for `realtime_llm`, and no VAD is
attached to 3.1 anyway (`agent.py:3543`). So barge-in is 100% Gemini server-side AAD
at default sensitivity, and **tuning those env vars changes nothing**. The "BUG 5 FIX"
comment above them is wrong for this model. Line echo re-triggering server AAD is the
plausible cause of the mid-sentence truncations, and the real lever would be
`RealtimeInputConfig` start/end sensitivity, which 3.1 is currently passed none of.

**2. `PunoVoiceAgent.on_user_turn_completed()` never runs on Gemini 3.1.**
It is reached only via `AgentActivity.on_end_of_turn` -> `AudioRecognition._run_eou_detection`,
whose call sites all require an STT or VAD event (audio_recognition.py:1078/1235/1324/1414).
This build has neither. Everything inside it — the language lock, `build_turn_snapshot`
injection, `_maybe_auto_confirm_booking`, speculative-context injection — is dead code
on the production model, and the log line at `agent.py:3041` never appears.

# Round 8 — "your booking is confirmed", said three times

Reported: the agent states the booking confirmation three times in one call — once
on confirming, again as a full summary in the same breath, and a third time AFTER
the goodbye line.

## Root cause — the TOOL RESULTS were scripting the duplicates, not the prompt

The prompt was a contributor, but not the driver. `confirm_booking()`'s success
return literally instructed two utterances:

```
"NOW speak the confirmation line ('Perfect — your booking is confirmed! You'll get
 the confirmation SMS in just a moment.'), then read back once: name {name}, date
 {date}, slot {slot}, package {pkg}, {headcount} guest(s), total about Rs. {price}."
```

A tool result arriving at the moment of confirmation outweighs the system prompt, so
**this string, not the prompt, is what the caller actually heard** — confirmation
line, then the whole summary again. It also left a quotable verbatim line sitting in
context for the model to re-emit later.

The prompt said the same thing (`"...tell the caller plainly: '...' Then read back
the essentials once — name, date, time slot, package, headcount"`), so both layers
agreed on the wrong behaviour. That is the user's hypothesis 1, confirmed — with the
correction that the tool return is the proximate cause.

The third statement, after goodbye, is the user's hypothesis 3, confirmed:
`end_call()` returns a **tool result** to the model, and every tool result provokes a
fresh model turn. Its old wording ("The call will disconnect cleanly 1 second after
you finish speaking the closing greeting") even gave that turn something to narrate.
`_execute_end_call_drop()` then waits for speech to end, sleeps 1.0s, and allows up
to 4s for the lead save — a **~5 second window** in which that extra turn is spoken
before the line drops. The idempotency branch of `confirm_booking` made it worse by
instructing "Reassure the caller their booking stands" — i.e. say it again.

Hypothesis 2 (a separate summary/logging LLM call being spoken aloud) is **ruled
out**: `log_call_summary` writes to storage and returns an internal note; nothing
routes a second generation to TTS.

## Fixes

**Tool returns became data instead of scripts** (`agent.py`):
- `confirm_booking` success: facts only, no verbatim line to quote, no read-back
  list, and an explicit "say this ONCE as a SINGLE short sentence... do NOT state
  the booking is confirmed again later unless the caller asks".
- `confirm_booking` idempotency: "do NOT state that the booking is confirmed again".
- `end_call`: terse `"Call ending. Say nothing further."`, and deliberately **no**
  language hint — "speak in Hindi" next to "say nothing" is a contradiction that
  invites the very turn being suppressed.

**Prompt** (`puno_prompt.py`): the confirmation is now one sentence, one time, with
an explicit ban on a following summary or any later restatement; and the closing
line is declared the last thing said on the call.

**Code-level hard block** (the part that does not rely on the model obeying) — in
the existing output guard, two independent rules, either of which calls
`session.interrupt(force=True)`:
- a confirmation restated when `state.booking_stated` is already True
- any second turn generated after `end_call` (the first is the goodbye itself,
  which commits after the tool runs and must be allowed)

`states_booking_confirmed()` deliberately distinguishes **stating** from **asking**:
"क्या मैं इसे कंफर्म कर दूं?" / "shall I confirm?" is the correct permission-seeking
turn and must never burn the one allowed statement.

`asks_to_repeat()` re-arms one restatement when the caller genuinely asks to hear it
again — without it, the once-only rule would gag the agent exactly when the caller
missed it, which is a worse failure than the repetition.

## Test

Two simulated bookings with different headcounts, replayed through the real guard:

    BOOKING 1 (7 people, Tourist Pass, Hindi — the reported call)
      confirmations spoken: 1   turns after goodbye: 0   PASS
    BOOKING 2 (3 people, Junior pass, English)
      confirmations spoken: 1   turns after goodbye: 0   PASS
    CONTROL (caller asks "can you repeat that?")
      requested repeat allowed; the next unprompted one still blocked   PASS

## Limitation

`session.interrupt(force=True)` cuts the turn, it does not prevent it — a syllable or
two may reach the caller before the cut. Prevention is the tool/prompt change; the
interrupt is the backstop for when the model ignores both.

# Round 9 — the agent accepted a booking for a date already past

Reported: "agent should never take past date and time". Visible in the previous
transcript, which booked **12 August 2026** on a call made **2 September 2026** —
three weeks in the past.

## Root cause — three failures stacked

**1. The model was never told what day it is.** `build_agent_instructions()` injected
the captured fields and the active language, but not the date. Nothing in the 33k-char
prompt said what "today" was, so the agent had no way to know August was behind it.
This is the root enabler: everything else is a missing backstop.

**2. The booking tools wrote the date with zero validation.** Both
`update_booking_details()` and `confirm_booking()` did:

```python
if date.strip():
    state.appointment_date = date.strip()
```

Straight from the model's argument to state. No parse, no check, no normalisation.

**3. The one validator that exists could not fire on this path.**
`validate_and_normalize_appointment_datetime()` bails out immediately unless the date
is already `YYYY-MM-DD` (`if not date_match: return date_str, time_str, None`), so a
spoken "12 August 2026" or "12 अगस्त" was returned untouched. Its only live call site
is `_apply_spoken_datetime()`, which inspects **raw ASR text**, never tool arguments —
and its other call site is inside `on_user_turn_completed()`, which Round 7's audit
established is **unreachable on Gemini 3.1**. It also enforces hospital OPD hours
(9:00 AM–7:00 PM) when PUNO runs 11:30 AM–10:00 PM, so wiring it into the booking
tools would have imported a second, unrelated bug.

## Fix

**`check_booking_not_past(date, time, *, now=None)`** — a focused new guard, separate
from the hospital-era validator on purpose. It parses ISO, D/M/Y, "12 August 2026",
"12 अगस्त", and relative words (today/tomorrow/parso, आज/कल/परसों), plus clock times
in English and Hindi day-parts (सुबह/शाम/रात/दोपहर).

Two design decisions worth stating:

- **It fails OPEN.** Anything it cannot confidently resolve is allowed through.
  Refusing a booking we merely failed to parse would be a worse bug than the one being
  fixed, so only a slot positively known to be behind us is rejected.
- **A year-less date rolls FORWARD.** On 2 September, "12 August" means next August.
  Rejecting it would be wrong, and silently reading it as *this* year is exactly how
  the reported call booked into the past.

Wired in at both choke points: `update_booking_details()` refuses to STORE a past
slot, and `confirm_booking()` re-checks the **effective** state (not just its own
arguments, since a date can also arrive via `_apply_spoken_datetime`) and clears the
bad value so the agent cannot re-confirm it unchanged.

**Today's date is now injected into the live state every turn**, at the top of the
stored-fields block, with the current IST time — plus a prompt rule that a month named
without a year means its next occurrence.

## A Devanagari trap worth recording

`\b` does not work at the end of Devanagari words that end in a combining mark.
`परसों` ends in U+0902 (anusvara) and the months `जनवरी` / `फरवरी` end in U+0940 —
neither is a `\w` character, so a trailing `\b` never matches and the word silently
fails to parse. Caught by testing; the first draft dropped `परसों`, `जनवरी` and
`फरवरी` on the floor. Latin alternatives keep `\b`; Devanagari uses substring matching
or the script-agnostic `_MONTH_END` lookahead.

## Test

Driven end-to-end through the real tools (today = 2 September 2026, 7:37 PM IST):

    update_booking_details()          confirm_booking()
      12 August 2026   BLOCKED          BLOCKED, booking_id=none
      12 अगस्त 2026    BLOCKED          BLOCKED, booking_id=none
      2026-08-12       BLOCKED          BLOCKED, booking_id=none
      1/9/2026         BLOCKED          BLOCKED, booking_id=none
      12 August        stored 2027-08-12   booked (rolled forward)
      2026-12-25       stored 2026-12-25   booked
      kal              stored 2026-09-03   booked
    same-day boundary: 4:37 PM BLOCKED / 9:37 PM accepted

## Adjacent defect, NOT fixed (out of scope)

`validate_and_normalize_appointment_datetime()` still enforces `_OPD_OPEN_MINUTES`
9:00 AM and `_OPD_CLOSE_MINUTES` 7:00 PM — Arora Hospital's OPD hours. PUNO Jaipur
opens 11:30 AM and closes 10:00 PM, so on the spoken-date path that function will
reject a valid 8:00 PM booking and accept an impossible 9:30 AM one. Left alone
because this round was scoped to past dates.
