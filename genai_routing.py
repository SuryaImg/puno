"""
genai_routing.py — the ONE place that decides how a Gemini model is reached.

WHY THIS MODULE EXISTS
----------------------
The project had SEVEN google-genai client constructions across four call sites, and
each hand-rolled the "Vertex AI or Gemini API?" decision. Only the realtime-session
path got it right. The rule itself was written down — but *inside* entrypoint(), so
the module-level synthesis helpers defined thousands of lines earlier could not reach
it and each invented its own version.

The result: greeting synthesis asked Vertex for `gemini-3.1-flash-live-preview`, which
is not a Vertex publisher model, got websocket close 1008 on every attempt, fell back
to a different model, and then re-synthesised on every single call — 32-second
startups and callers hanging up before the agent ever spoke.

Everything that talks to Gemini now routes through _genai_client_for() here, so a new
call site cannot reintroduce the bug.
"""

from __future__ import annotations

import os
import re


def _sdk_model_tables():
    """The livekit-google plugin's own model tables, if importable.

    Seeding from the SDK is what stops this project from disagreeing with the plugin:
    RealtimeModel.__init__ calls _validate_model_api_match() and RAISES when the model
    family and the vertexai flag disagree, so a predicate that differs from these sets
    does not merely route badly — it takes the session down.
    Imported lazily so lead_capture.py does not pull in livekit just to extract a lead.
    """
    try:
        from livekit.plugins.google.realtime.realtime_api import (
            KNOWN_VERTEXAI_MODELS,
            KNOWN_GEMINI_API_MODELS,
        )
        return set(KNOWN_VERTEXAI_MODELS), set(KNOWN_GEMINI_API_MODELS)
    except Exception:
        # Keep the known values as a fallback so behaviour is stable if the plugin
        # moves these symbols in a future release.
        return (
            {"gemini-live-2.5-flash-native-audio"},
            {"gemini-3.1-flash-live-preview", "gemini-2.5-flash-native-audio-preview-12-2025"},
        )


def _model_uses_gemini_api(name: str) -> bool:
    """True when `name` must be reached via the Gemini API rather than Vertex AI.

    THIS IS THE SINGLE SOURCE OF TRUTH for that decision. It previously lived *inside*
    entrypoint(), so the module-level synthesis helpers defined thousands of lines
    earlier could not reach it and each hand-rolled its own version — which is how one
    code path ended up correct and four ended up wrong (asking Vertex for
    `gemini-3.1-flash-live-preview` returns websocket close 1008 "Publisher model ...
    not found", which is what made the greeting re-synthesise on every call).

    IMPORTANT — "native-audio" is NOT the discriminator. `gemini-live-2.5-flash-native-audio`
    contains it but is a VERTEX-ONLY model; classifying it as Gemini API makes
    RealtimeModel.__init__ raise ValueError and kills the session fallback entirely.
    The real family split is:
        gemini-live-*   -> Vertex AI
        gemini-<N.N>-*  -> Gemini API
    We consult the SDK's own tables first, and only fall back to that family rule for
    model names the installed plugin does not know about.
    """
    n = (name or "").strip().lower()
    if not n:
        return False

    vertex_models, gemini_models = _sdk_model_tables()
    if n in {m.lower() for m in vertex_models}:
        return False
    if n in {m.lower() for m in gemini_models}:
        return True

    # Unknown model — fall back to the family naming convention the SDK documents
    # ("VertexAI models typically start with 'gemini-live-'").
    if n.startswith("gemini-live-"):
        return False
    if "gemini-3" in n:
        return True
    # Preview TTS models are Gemini-API-served.
    if "-tts" in n or "tts-" in n:
        return True
    if "native-audio" in n:
        return True
    return False


def _genai_client_for(model: str):
    """Build a google-genai Client correctly routed for `model`.

    The ONE place the Vertex-vs-Gemini-API decision is made. Every genai client in
    this project is constructed here so that adding a new call site cannot
    reintroduce the routing bug.
    """
    from google import genai

    def _api_key_client():
        # vertexai=False MUST be explicit. google-genai reads GOOGLE_GENAI_USE_VERTEXAI
        # from the environment at construction time, so passing api_key= alone still
        # yields a Vertex client when that env var is "true" — which would make this
        # whole fix a silent no-op. _build_realtime_model gets this right by passing
        # vertexai=use_v explicitly; that is why the live SESSION works today.
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key:
            return genai.Client(vertexai=False, api_key=api_key)
        return genai.Client(vertexai=False)

    if _model_uses_gemini_api(model):
        return _api_key_client()

    use_v = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() == "true"
    proj = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    loc = (
        os.environ.get("GCP_LOCATION")
        or os.environ.get("GOOGLE_CLOUD_LOCATION")
        or "us-central1"
    )
    if use_v and proj:
        return genai.Client(vertexai=True, project=proj, location=loc)
    return _api_key_client()


# A model that is missing, denied or unsupported fails the same way every time, so
# retrying it only burns the caller's patience — the observed 32s startup was four
# retries against a permanently-absent publisher model, twice over. Deliberately
# phrase-based rather than matching bare status numbers, which show up in unrelated
# messages (byte counts, ids) and would misclassify transient failures.
_PERMANENT_MODEL_ERROR_RE = re.compile(
    # Separators vary: the API returns PERMISSION_DENIED / INVALID_ARGUMENT with
    # underscores, while prose messages use spaces — accept either.
    r"(\b1008\b|publisher[_\s]model|not[_\s]found|permission[_\s]denied"
    r"|invalid[_\s]argument|unsupported[_\s]model|does[_\s]not[_\s]exist"
    r"|is[_\s]not[_\s]supported|failed[_\s]precondition|unauthenticated)",
    re.IGNORECASE,
)


def _is_permanent_model_error(exc: BaseException) -> bool:
    """True when retrying `exc` against the SAME model can never succeed.

    Callers should stop retrying that model but still fall through to their
    fallback model — the point is to skip useless retries, not to give up.
    """
    try:
        return bool(_PERMANENT_MODEL_ERROR_RE.search(str(exc) or ""))
    except Exception:
        return False
