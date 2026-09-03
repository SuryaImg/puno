"""
storage.py — Redis-backed onboarding state + call transcript + seller URL lookup.

Key schema:
  conversation:{session_id}     → JSON blob of OnboardingState/AppointmentState fields
  booking_status:{session_id}   → booking status string
  booking_lock:{doctor_id}:{date}:{time} → lock to prevent duplicate slot bookings
  confirmed_appointment:{appointment_id} → JSON blob of confirmed appointments
"""

import asyncio
import datetime
import json
import logging
import os
import re
import time
import random
from typing import Optional

from dotenv import load_dotenv
from tools import AppointmentState, CallTranscript, normalize_time_slot

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
REDIS_URL                = os.environ.get("REDIS_URL", "redis://localhost:6379")


def _masked_redis_url(url: str) -> str:
    """Redact userinfo (user:password@) before a Redis URL is ever logged.

    SECURITY FIX: REDIS_URL was being logged verbatim on every connection
    across every call ("Redis connected: redis://default:<password>@host:port"),
    writing a live credential into application/syslog on every single call.
    Everything downstream of this function's callers must use the masked form;
    the raw REDIS_URL constant should never be passed to logger.* directly.
    """
    try:
        scheme, _, rest = url.partition("://")
        if "@" in rest:
            userinfo, _, hostpart = rest.rpartition("@")
            user = userinfo.split(":", 1)[0] if userinfo else ""
            return f"{scheme}://{user}:***@{hostpart}"
        return url
    except Exception:
        # Fail closed: if parsing goes wrong for any reason, never risk
        # leaking the original string — return a fixed placeholder instead.
        return "<redis-url-redacted>"
# When REDIS_REQUIRED=true, booking-critical operations (slot locking, confirmed
# appointment writes) will RAISE instead of silently falling back to per-process
# in-memory storage. In a multi-process deployment (num_idle_processes > 1) the
# in-memory fallback is NOT shared across workers, so silently using it lets two
# concurrent callers on different processes double-book the same doctor + slot.
# Set this to "true" in production so a misconfigured/unreachable Redis fails
# LOUDLY rather than silently corrupting bookings. Defaults to "false" so local
# dev without Redis keeps working.
REDIS_REQUIRED           = os.environ.get("REDIS_REQUIRED", "false").strip().lower() == "true"
REDIS_STATE_TTL_SEC      = int(os.environ.get("REDIS_STATE_TTL_DAYS", "30")) * 86400
REDIS_TRANSCRIPT_TTL_SEC = 90 * 86400   # 90 days
REDIS_SELLER_URL_TTL_SEC = 7 * 86400    # 7 days
KEY_PREFIX               = "conversation:"
SELLER_URL_PREFIX        = "seller_url:"
TRANSCRIPT_PREFIX        = "transcript:"

# ── Connection singleton ──────────────────────────────────────────────────────
_redis = None
_use_memory: bool = False
_memory_store: dict[str, AppointmentState] = {}
_memory_transcripts: dict[str, dict] = {}
_memory_seller_urls: dict[str, str] = {}

_connect_lock: Optional[asyncio.Lock] = None
_connect_lock_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_connect_lock() -> asyncio.Lock:
    global _connect_lock, _connect_lock_loop

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        if _connect_lock is None:
            _connect_lock = asyncio.Lock()
        return _connect_lock

    if _connect_lock is None or _connect_lock_loop is not current_loop:
        _connect_lock = asyncio.Lock()
        _connect_lock_loop = current_loop

    return _connect_lock


def _reset_redis_state() -> None:
    global _redis, _use_memory, _connect_lock, _connect_lock_loop
    _redis = None
    _use_memory = False
    _connect_lock = None
    _connect_lock_loop = None
    _invalidate_confirmed_cache()
    logger.debug("Redis state reset for new worker process")


async def _execute_with_retry(func, *args, timeout: float = 3.0, **kwargs):
    max_attempts = 3
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            logger.debug(
                f"Executing Redis operation attempt {attempt}/{max_attempts} "
                f"for function '{getattr(func, '__name__', str(func))}'"
            )
            res = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            return res
        except Exception as e:
            last_err = e
            # Capacity errors get worse if we hammer the server. Fail over
            # immediately so the caller can use memory/cache instead of
            # burning three round-trips in a millisecond.
            if _redis_error_is_capacity(e):
                logger.warning(
                    "Redis at connection capacity (%s) — not retrying",
                    e,
                )
                break
            if _redis_error_is_dead_connection(e):
                await _discard_redis_client()
            if attempt < max_attempts:
                logger.warning(
                    f"Redis operation attempt {attempt}/{max_attempts} failed with: {e}. Retrying..."
                )
                await asyncio.sleep(min(0.05 * (2 ** (attempt - 1)), 0.4))
            else:
                logger.warning(
                    f"Redis operation attempt {attempt}/{max_attempts} failed with: {e}. No more retries."
                )
    raise last_err


def _redis_error_is_capacity(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "too many connections",
            "max number of clients",
            "maxclients",
            "err max number of clients reached",
        )
    )


def _redis_error_is_dead_connection(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "connection reset",
            "broken pipe",
            "connection closed",
            "not connected",
            "eof",
            "server closed",
        )
    )


async def _discard_redis_client() -> None:
    """Close and drop the process Redis client without leaking connections."""
    global _redis
    client = _redis
    _redis = None
    if client is None:
        return
    try:
        aclose = getattr(client, "aclose", None)
        if aclose is not None:
            await aclose()
        else:
            close = getattr(client, "close", None)
            if close is not None:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
    except Exception:
        pass


async def _get_redis():
    global _redis, _use_memory

    if _use_memory:
        return None
    if _redis is not None:
        return _redis

    async with _get_connect_lock():
        if _redis is not None:
            return _redis

        try:
            import redis.asyncio as aioredis

            if REDIS_URL.startswith("redis://") and ":6380" in REDIS_URL:
                logger.warning(
                    "REDIS_URL uses plaintext 'redis://' on port 6380 — that port "
                    "is usually TLS-only. If connection fails, switch to 'rediss://'."
                )

            client = aioredis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
                retry_on_timeout=True,
                max_connections=int(os.environ.get("REDIS_MAX_CONNECTIONS", "8")),
                health_check_interval=30,
            )
            await asyncio.wait_for(client.ping(), timeout=3.0)
            _redis = client
            logger.info("Redis connected: %s", _masked_redis_url(REDIS_URL))
            return _redis

        except Exception as e:
            # Stick to memory for the rest of this process. Retrying a 3s
            # connect on every save/lock stalled the live call (agent stuck
            # "thinking", no audio) while Redis was unreachable.
            _use_memory = True
            logger.warning(
                "Redis unavailable (%s) — falling back to in-memory storage for this process",
                e,
            )
            return None


def _key(user_id: str) -> str:
    return f"conversation:{user_id}"


def _status_key(user_id: str) -> str:
    return f"booking_status:{user_id}"


def _transcript_key(user_id: str, call_ts: str) -> str:
    return f"{TRANSCRIPT_PREFIX}{user_id}:{call_ts}"


def _state_to_dict(state: AppointmentState) -> dict:
    return state.model_dump()


def _clean_data_for_appointment_state(data: dict) -> dict:
    if not isinstance(data, dict):
        return data
    d = dict(data)
    if "status" in d and "booking_status" not in d:
        d["booking_status"] = d["status"]
    if "doctor_id" in d and d["doctor_id"] is not None:
        try:
            d["doctor_id"] = int(d["doctor_id"])
        except (ValueError, TypeError):
            d["doctor_id"] = None
    return d


def _dict_to_state(data: dict) -> AppointmentState:
    cleaned = _clean_data_for_appointment_state(data)
    return AppointmentState(**cleaned)


_confirmed_cache: Optional[list] = None
_confirmed_cache_at: float = 0.0
_confirmed_fail_until: float = 0.0
_CONFIRMED_CACHE_TTL_SEC = 3.0
_CONFIRMED_FAIL_COOLDOWN_SEC = 0.5
_slot_redis_skip_until: float = 0.0


def _invalidate_confirmed_cache() -> None:
    global _confirmed_cache, _confirmed_cache_at, _confirmed_fail_until, _slot_redis_skip_until
    _confirmed_cache = None
    _confirmed_cache_at = 0.0
    _confirmed_fail_until = 0.0
    _slot_redis_skip_until = 0.0


async def get_appointment_state(user_id: str) -> Optional[AppointmentState]:
    """Fetch patient appointment state. Returns None if no prior state exists."""
    r = await _get_redis()
    if r is None:
        state = _memory_store.get(user_id)
        if state:
            status = _memory_store.get(_status_key(user_id))
            if status:
                state.booking_status = status
        return state

    try:
        async def do_read():
            raw = await r.get(_key(user_id))
            status = await r.get(_status_key(user_id))
            return raw, status

        raw, status = await _execute_with_retry(do_read)
        if raw is None:
            return None
        data = json.loads(raw)
        state = _dict_to_state(data)
        if status:
            state.booking_status = status
        else:
            state.booking_status = "DRAFT"
        logger.debug(
            "Loaded state from Redis for user=%s step=%s, status=%s",
            user_id,
            state.current_step(),
            state.booking_status,
        )
        return state
    except Exception as e:
        logger.warning("Redis GET failed (%s) — memory fallback", e)
        state = _memory_store.get(user_id)
        if state:
            status = _memory_store.get(_status_key(user_id))
            if status:
                state.booking_status = status
        return state


async def save_appointment_state(user_id: str, state: AppointmentState) -> None:
    """Upsert appointment state."""
    # The hospital build also mirrored every BOOKED state into a
    # `confirmed_appointment:*` record. Nothing reads that namespace any more —
    # a PUNO booking is written by confirm_booking() to `puno_booking:*` — so the
    # mirror was write-only duplication and has been removed with the rest of the
    # hospital code.
    r = await _get_redis()
    if r is None:
        _memory_store[user_id] = state
        _memory_store[_status_key(user_id)] = state.booking_status
        logger.info(
            "Saved state in memory for user=%s step=%s",
            user_id,
            state.current_step(),
        )
        return

    async def do_write_and_verify():
        payload = json.dumps(_state_to_dict(state))
        await r.set(_key(user_id), payload, ex=REDIS_STATE_TTL_SEC)
        await r.set(_status_key(user_id), str(state.booking_status), ex=REDIS_STATE_TTL_SEC)
        
        # Verify both
        verify_raw = await r.get(_key(user_id))
        verify_status = await r.get(_status_key(user_id))
        if verify_raw is None:
            raise ValueError(f"Write verification failed: key {_key(user_id)} not found after write")
        if verify_status is None:
            raise ValueError(f"Write verification failed: key {_status_key(user_id)} not found after write")
        
        if str(verify_status).upper() != str(state.booking_status).upper():
            raise ValueError(f"Write verification failed: booking_status mismatch (expected {state.booking_status}, got {verify_status})")

    try:
        await _execute_with_retry(do_write_and_verify)
        logger.info(
            "Saved state in Redis for user=%s step=%s",
            user_id,
            state.current_step(),
        )
    except Exception as e:
        logger.warning("Redis SET failed after retries (%s) — falling back to memory", e)
        _memory_store[user_id] = state
        _memory_store[_status_key(user_id)] = state.booking_status


# ── Seller URL Lookup ─────────────────────────────────────────────────────────

async def save_call_transcript(user_id: str, transcript: "CallTranscript") -> str:
    r = await _get_redis()
    call_ts = transcript.call_start_time or str(int(time.time()))
    key = _transcript_key(user_id, call_ts)

    payload = json.dumps(transcript.model_dump(), default=str)

    if r is None:
        _memory_transcripts[key] = json.loads(payload)
        logger.info("Transcript saved in memory for user=%s key=%s", user_id, key)
        return key

    try:
        await _execute_with_retry(r.set, key, payload, ex=REDIS_TRANSCRIPT_TTL_SEC)
        logger.info("Transcript saved in Redis for user=%s key=%s", user_id, key)
    except Exception as e:
        logger.warning("Redis transcript SET failed (%s) — memory fallback", e)
        _memory_transcripts[key] = json.loads(payload)

    return key


async def close_pool() -> None:
    global _redis
    if _redis is not None:
        try:
            await _redis.aclose()
        except Exception:
            pass
        _redis = None


async def _generate_prefixed_id(prefix: str, key_ns: str) -> str:
    """Collision-checked '<PREFIX><6 digits>' id, unique within `key_ns:*`."""
    r = await _get_redis()
    for _ in range(20):
        candidate = f"{prefix}{random.randint(100000, 999999)}"
        key = f"{key_ns}:{candidate}"
        if r is not None:
            try:
                if not await _execute_with_retry(r.exists, key):
                    return candidate
                continue
            except Exception:
                break
        else:
            if key not in _memory_store:
                return candidate
    return f"{prefix}{int(time.time() * 1000) % 900000 + 100000}"


async def generate_booking_id() -> str:
    """Local reference id for a confirmed PUNO booking (e.g. PUNO482913)."""
    return await _generate_prefixed_id("PUNO", "puno_booking")


async def generate_lead_id() -> str:
    """Local reference id for a PUNO lead (e.g. LEAD771204)."""
    return await _generate_prefixed_id("LEAD", "puno_lead")


# ── Record timestamps ─────────────────────────────────────────────────────────
# PUNO Jaipur is in India, so every timestamp a human reads out of Redis must be
# IST. These were `time.strftime(...)`, i.e. the HOST's local clock — correct on
# a dev laptop set to IST, but the agent runs on a UTC VM, so every stored
# created_at/timestamp/sent_at was 5h30m behind the real call time (a lead saved
# at 12:40 IST was written as "07:10:31").
#
# Mirrors agent._now_ist(): a fixed +5:30 offset rather than an IANA zone lookup
# (IST has no DST, and this cannot then depend on the tz database being present
# and correct on the host). Kept as a naive "%Y-%m-%d %H:%M:%S" string so the
# stored format is byte-for-byte what every existing consumer already parses —
# only the value moves.
_IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))


def _now_ist_str() -> str:
    """Current IST wall-clock as 'YYYY-MM-DD HH:MM:SS'. Use for every timestamp
    written into a PUNO record — never time.strftime()/datetime.now(), which
    silently follow the server's timezone."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .astimezone(_IST)
        .strftime("%Y-%m-%d %H:%M:%S")
    )


async def _save_puno_record(key: str, record: dict) -> None:
    """Write a JSON record to Redis, falling back to process memory on failure."""
    payload = json.dumps(record, default=str)
    r = await _get_redis()
    if r is None:
        _memory_store[key] = record
        logger.warning("Redis unavailable — %s saved to PROCESS MEMORY ONLY.", key)
        return
    try:
        async def _do_write():
            await r.set(key, payload, ex=REDIS_STATE_TTL_SEC)
        await _execute_with_retry(_do_write)
        logger.info("PUNO record saved in Redis: %s", key)
    except Exception as e:
        logger.error("Redis save FAILED (%s) — falling back to memory for %s.", e, key)
        _memory_store[key] = record


async def save_puno_booking(record: dict) -> str:
    """Persist a confirmed PUNO booking to `puno_booking:*`.

    Booking record carries payment_status='n/a' — this build has no payment
    channel; the field exists only so the live-build schema stays stable.
    Returns the booking_id used as the key.
    """
    booking_id = record.get("booking_id") or await generate_booking_id()
    record["booking_id"] = booking_id
    record.setdefault("type", "PUNO_BOOKING")
    record.setdefault("payment_status", "n/a")
    record.setdefault("created_at", _now_ist_str())
    await _save_puno_record(f"puno_booking:{booking_id}", record)
    return booking_id


async def save_puno_lead(record: dict) -> str:
    """Persist a PUNO lead to `puno_lead:*`. Returns the lead_id."""
    lead_id = record.get("lead_id") or await generate_lead_id()
    record["lead_id"] = lead_id
    record.setdefault("type", "PUNO_LEAD")
    record.setdefault("created_at", _now_ist_str())
    await _save_puno_record(f"puno_lead:{lead_id}", record)
    return lead_id


async def save_puno_call_log(record: dict) -> str:
    """Persist the end-of-call CRM record to `puno_call_log:*`.

    Keyed by booking_id when present (so the log joins the booking and SMS on
    the same reference), otherwise by a timestamp-derived id.
    """
    log_id = (
        record.get("booking_id")
        or record.get("lead_id")
        or f"CALL{int(time.time() * 1000) % 900000 + 100000}"
    )
    record.setdefault("type", "PUNO_CALL_LOG")
    record.setdefault("payment_status", "n/a")
    record.setdefault("timestamp", _now_ist_str())
    await _save_puno_record(f"puno_call_log:{log_id}", record)
    return log_id


async def save_dummy_sms(booking_id: str, phone: str, body: str) -> None:
    """Record the dummy confirmation SMS for the demo.

    NO real SMS gateway is wired in this build (see spec §5A / §6). This writes
    a `puno_sms:*` record and logs the send so the demo can show that the
    confirmation 'fired' immediately from confirm_booking(), with no payment
    dependency and no link of any kind.
    """
    record = {
        "type": "PUNO_DUMMY_SMS",
        "booking_id": booking_id,
        "to": phone,
        "body": body,
        "channel": "SMS",
        "status": "SENT (dummy)",
        "sent_at": _now_ist_str(),
    }
    logger.info("DUMMY confirmation SMS to %s for booking %s: %s", phone, booking_id, body)
    await _save_puno_record(f"puno_sms:{booking_id}", record)