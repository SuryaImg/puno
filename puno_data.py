"""
puno_data.py — Verified knowledge base for PUNO Jaipur (v2.3).

Single source of truth for every PUNO fact the voice agent is allowed to state.
Both the system prompt (via `puno_knowledge_block()`) and the runtime info tools
read from the structures here, so a fact is edited in exactly one place.

Nothing in this file is scraped or fetched — PUNO's data is small and static
(unlike the hospital build's dynamic doctor list), so it lives inline. Guardrail:
the agent must NEVER state a price, timing, address, offer or safety rule that is
not represented here. Anything beyond the three packages is quoted as
"starting from / let me confirm the live price".
"""

from __future__ import annotations

from typing import Optional

# --- 1. Brand & identity ----------------------------------------------------
BRAND = {
    "name": "PUNO Jaipur",
    "outlet": "PUNO Jaipur / PUNO Advance Jaipur",
    "full_form": "Play, Unlimited, Notch-up, Optimism",
    "philosophy": "Fun, Family, Fitness, Food — Together",
    "mascot": "Quto",
    "operator": "Shri Poornima Entertainments Private Limited",
    "tagline": "India's first experiential entertainment centre — all under one roof.",
}

# --- 2. Location & timings --------------------------------------------------
LOCATION = {
    "address": (
        "Fantasy Square, 192-196, Ajmer Road, near 200 Feet Bypass Road, "
        "Dr. Rajendra Prasad Nagar-A, Vardhman Nagar, Karni Vihar, Jaipur, "
        "Rajasthan 302019"
    ),
    # Spoken form — never read the full address aloud; offer WhatsApp for the pin.
    "spoken_address": "We're at Fantasy Square on Ajmer Road, near the 200 Feet Bypass",
    "timings": "11:30 AM to 10:00 PM, Monday to Sunday (may extend past 10 PM on select days)",
    "parking": "Available at the venue",
}

# --- 3. Contact -------------------------------------------------------------
CONTACT = {
    "phone": "+91 82982 94449",
    "whatsapp": "+91 82982 94449",
    "email": "enquiry@funatpuno.com",
    "website": "funatpuno.com",
    "instagram_jaipur": "funatpuno.jaipur",
    "socials": "funatpuno on Instagram, Facebook, LinkedIn, YouTube, X and Threads",
}

# --- 4. Attractions ---------------------------------------------------------
# Each: name, one-line description, age/notes.
ATTRACTIONS = [
    {
        "name": "Adventure & Trampoline",
        "desc": "Rope course, ninja course, sky wall, sky stepper, donut slide, human "
                "gyro, spider tower, jump arena, battle beam, swing bridge, slam dunk, "
                "wipe out, roller bridge and interactive trampolines",
        "notes": "Typically 90 min. All ages. Weight limit generally up to 120 kg. Grip socks mandatory.",
    },
    {
        "name": "Bowling",
        "desc": "Augmented, interactive, glow-in-the-dark bowling — lanes light up and themes change with your game",
        "notes": "All ages. Socks required.",
    },
    {
        "name": "VR Rides",
        "desc": "Immersive VR games and rides",
        "notes": "Generally 7 years and above.",
    },
    {
        "name": "Arcade",
        "desc": "60+ games — retro, racing, shooting, sports and ticket redemption; several launched first-time in India",
        "notes": "All ages.",
    },
    {
        "name": "Mad Wheels",
        "desc": "Real-time racing zone with cars and bikes",
        "notes": "",
    },
    {
        "name": "VEX",
        "desc": "Immersive interactive play — technology plus movement plus live challenges",
        "notes": "",
    },
    {
        "name": "PUNO Junior / Kids Zone",
        "desc": "Foam pits, mini cars, sand pool and supervised age-appropriate play",
        "notes": "For children below 7 years. Typically 90-120 min.",
    },
    {
        "name": "Others",
        "desc": "Cube Chase, Cricket Game and Social Zone",
        "notes": "",
    },
    {
        "name": "Pots & Plates",
        "desc": "On-site multi-cuisine sports cafe / restaurant",
        "notes": "",
    },
]

# --- 5. Packages (prices inclusive of GST) ----------------------------------
# package_id is the stable key passed to confirm_booking().
PACKAGES = [
    {
        "id": "tourist_weekday",
        "name": "Tourist Pass — Weekdays",
        "duration": "90 min",
        "price": 1887,
        "price_str": "Rs. 1,887",
        "includes": "Adventure & Trampoline, Bowling, Mad Wheel, VR Ride and 5 Arcade Games",
        "validity": "Weekdays",
        "age": "All ages",
    },
    {
        "id": "tourist_weekend",
        "name": "Tourist Pass — Weekends",
        "duration": "90 min",
        "price": 2241,
        "price_str": "Rs. 2,241",
        "includes": "Adventure & Trampoline, Bowling, Mad Wheel, VR Ride and 5 Arcade Games",
        "validity": "Age 7+ only. Saturday & Sunday only. Not valid on peak days.",
        "age": "7+",
    },
    {
        "id": "junior_fun_play",
        "name": "Junior Fun n' Play",
        "duration": "120 min",
        "price": 825,
        "price_str": "Rs. 825",
        "includes": "Junior Zone access",
        "validity": "Below age 7. Monday to Friday only. Not valid on peak days.",
        "age": "under 7",
    },
]

# --- 6. Offers (always add: "offers change, let me confirm the live price") --
OFFERS = [
    {
        "name": "Happy Hours / Funtastic Hours",
        "desc": "Up to 50% off on games & activities, typically Mon-Thu 11 AM-4 PM. "
                "Walk-ins and IVR bookings only; cannot be clubbed with other offers; not valid on peak days.",
    },
    {
        "name": "Bowling Fantastic",
        "desc": "Bowling from Rs. 199 per person (weekday daytime).",
    },
    {
        "name": "Junior Arena Pass",
        "desc": "Rs. 499 to Rs. 1,000 for 120 minutes (under 7 only).",
    },
    {
        "name": "Adventure & Trampoline Thrill Pass",
        "desc": "90 min including rope course, trampoline arena and ninja course.",
    },
    {
        "name": "Card Bonus",
        "desc": "Recharge card system; amount deducted per game. Bonus games on higher recharges "
                "(e.g. Rs. 3,000 recharge gives free games usable same day; larger bonuses at Rs. 5,000+).",
    },
    {
        "name": "Grip / normal socks",
        "desc": "Mandatory in trampoline, bowling and play zones. Extra charge Rs. 30 to Rs. 80, available on-site.",
    },
]

# --- 7. Membership — PUNO VIP "Infinity Club" -------------------------------
# Tiers Golden and Purple. How they map to the three cards is NOT defined here
# → capture interest and hand off to sales (guardrail 9).
MEMBERSHIP_TIERS = "Golden and Purple"
MEMBERSHIP_CARDS = [
    {
        "name": "Sapphire",
        "price": "Rs. 11,000",
        "value": "Rs. 12,500 (save Rs. 1,500)",
        "benefits": "3 free games, birthday perks, limited discounts",
    },
    {
        "name": "Elite",
        "price": "Rs. 15,000",
        "value": "Rs. 18,750 (save Rs. 3,750)",
        "benefits": "5 free games, F&B discount, birthday perks, VIP pricing",
    },
    {
        "name": "Primia",
        "price": "Rs. 25,000",
        "value": "Rs. 32,500+ (save Rs. 7,500+)",
        "benefits": "10 free games, premium benefits, higher F&B discount, birthday perks",
    },
]

# --- 8. Birthday & party packages -------------------------------------------
# Party bookings are NEVER completed on-call — collect interest → Party Specialist.
BIRTHDAY = {
    "included": "Dedicated party host, mascot Quto, personalised invitation video, "
                "party area, and food options (Rock Bottom / Knock Out / Jumper packages)",
    "minimum": "Minimum booking is often 20 members. Party hall charges and food may be extra depending on package.",
    "options": [
        "Junior weekday party: 120 min Junior Arena, kids below 7 — from around Rs. 1,000 per person",
        "Advance Knockout: 60/90 min Advance access plus food (any 6 items), age 7+ — from around Rs. 2,399 per person",
        "Other options: Power Pack, Advance Combo, Bowling packages",
    ],
}

# --- 9. Groups / corporate / school -----------------------------------------
# Never quote bulk/group pricing on-call → dedicated specialist calls back.
GROUPS = (
    "Suitable for birthday parties, school trips and picnics, corporate outings and "
    "team building, kitty parties, social gatherings and bowling tournaments. Process: "
    "collect details, assign a dedicated specialist, and the specialist calls back with a custom quote."
)

# --- 10. Safety & guidelines ------------------------------------------------
SAFETY = [
    "Staff brief every guest on safety before activities.",
    "A waiver form is generally mandatory.",
    "Weight limits apply on certain attractions (typically up to 120 kg on adventure/trampoline).",
    "Socks are mandatory in trampoline, bowling and play zones — for hygiene and grip.",
    "Staff assistance is available throughout.",
    "Ride-specific age/height limits vary by activity — if unknown, say restrictions vary and offer WhatsApp confirmation or transfer.",
]

# --- 11. Peak days ----------------------------------------------------------
# Exact dated calendar is NOT available to the agent → flag & confirm, never invent.
PEAK_DAYS = (
    "Peak days follow the PUNO Advance Calendar — festivals, public holidays and school "
    "vacations. On peak days, weekend pricing applies and several offers are not valid. "
    "Known categories: New Year, Republic Day, Valentine's Day, Holi, Independence Day, "
    "Diwali period and winter holidays (late December). The exact dated calendar is not "
    "available; if a caller names a date near any of these, say it may fall on a peak day so "
    "weekend pricing would apply, and confirm before locking the slot."
)

# --- 12. Approved FAQ answers ----------------------------------------------
FAQS = [
    ("What is PUNO?",
     "India's first experiential entertainment centre — adventure, trampoline, bowling, VR, "
     "arcade, junior zone and food, all under one roof, for all ages."),
    ("How much does it cost?",
     "It depends on the package — there are three: Tourist Pass Weekdays (Rs. 1,887), "
     "Tourist Pass Weekends (Rs. 2,241) and Junior Fun n' Play (Rs. 825)."),
    ("How do I book?",
     "On this call, on the website, on WhatsApp, or walk-in subject to availability."),
    ("Age limits?",
     "Suitable for all ages. Junior Arena is for under 7. Weight limits apply on adventure rides."),
    ("Do I need to book in advance?",
     "Recommended — especially weekends, peak days, groups and parties."),
    ("Socks needed?",
     "Yes, for trampoline, bowling and play areas. Grip socks recommended, available on-site at extra charge."),
    ("Parking?",
     "Yes, available at the venue."),
    ("Birthday party?",
     "Yes — dedicated host, mascot Quto, party area, food and decoration options. Our Party "
     "Specialist will call you with the exact quote."),
    ("Food on-site?",
     "Yes — Pots & Plates, a multi-cuisine cafe/restaurant."),
    ("Do I have to pay now to book?",
     "No. Nothing is charged on this call and there's no online payment. Your slot is confirmed "
     "on the call — you settle at the front desk when you arrive."),
]


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------
def get_package(package_id_or_name: str) -> dict | None:
    """Resolve a package by id or (loose) name. Returns the package dict or None."""
    if not package_id_or_name:
        return None
    key = str(package_id_or_name).strip().lower()
    for p in PACKAGES:
        if p["id"] == key or p["name"].lower() == key:
            return p
    # Loose contains-match on name / id keywords.
    for p in PACKAGES:
        if key in p["name"].lower() or key in p["id"]:
            return p
    # Keyword heuristics.
    if "junior" in key or "kid" in key or "child" in key:
        return get_package("junior_fun_play")
    if "weekend" in key or "saturday" in key or "sunday" in key:
        return get_package("tourist_weekend")
    if "week" in key or "tourist" in key:
        return get_package("tourist_weekday")
    return None


def package_choices_line() -> str:
    """Short spoken summary of the three packages (for the agent to paraphrase)."""
    return (
        "Tourist Pass Weekdays is Rs. 1,887 for 90 minutes; Tourist Pass Weekends is "
        "Rs. 2,241 for 90 minutes, age 7+; and Junior Fun n' Play is Rs. 825 for 120 "
        "minutes, for kids below 7."
    )


# ---------------------------------------------------------------------------
# Prompt block — a compact, spoken-safe rendering of the whole KB, baked into
# the system prompt so the model always has verified facts in-context.
# ---------------------------------------------------------------------------
def puno_knowledge_block() -> str:
    lines: list[str] = []
    lines.append("--- LOCATION & TIMINGS ---")
    lines.append(f"- Outlet: {BRAND['outlet']}.")
    lines.append(f"- Spoken address (do NOT read the full address aloud): \"{LOCATION['spoken_address']}\" — offer the full address + Google Maps pin on WhatsApp.")
    lines.append(f"- Timings: {LOCATION['timings']}.")
    lines.append(f"- Parking: {LOCATION['parking']}.")
    lines.append("")
    lines.append("--- CONTACT (share only if asked) ---")
    lines.append(f"- Phone / WhatsApp: {CONTACT['phone']}. Email: {CONTACT['email']}. Website: {CONTACT['website']}. Instagram (Jaipur): {CONTACT['instagram_jaipur']}.")
    lines.append("")
    lines.append("--- ATTRACTIONS ---")
    for a in ATTRACTIONS:
        note = f" [{a['notes']}]" if a["notes"] else ""
        lines.append(f"- {a['name']}: {a['desc']}.{note}")
    lines.append("")
    lines.append("--- PACKAGES (all prices inclusive of GST) ---")
    for p in PACKAGES:
        lines.append(f"- {p['name']}: {p['price_str']}, {p['duration']} — includes {p['includes']}. Validity: {p['validity']}")
    lines.append("")
    lines.append("--- OFFERS (always add: \"offers change, let me confirm the live price for you\") ---")
    for o in OFFERS:
        lines.append(f"- {o['name']}: {o['desc']}")
    lines.append("")
    lines.append("--- MEMBERSHIP — PUNO VIP \"Infinity Club\" ---")
    lines.append(f"- Tiers: {MEMBERSHIP_TIERS}.")
    for c in MEMBERSHIP_CARDS:
        lines.append(f"- {c['name']}: {c['price']} (value {c['value']}) — {c['benefits']}.")
    lines.append("- If asked how Golden/Purple map to these cards, or for terms beyond this table → capture interest and hand off to sales. Do NOT improvise.")
    lines.append("")
    lines.append("--- BIRTHDAY & PARTY (never completed on this call — route to Party Specialist) ---")
    lines.append(f"- Included: {BIRTHDAY['included']}.")
    lines.append(f"- {BIRTHDAY['minimum']}")
    for opt in BIRTHDAY["options"]:
        lines.append(f"- {opt}.")
    lines.append("- Always frame party prices as \"starting from, depending on inclusions\" and confirm the exact quote with the Party Specialist.")
    lines.append("")
    lines.append("--- GROUPS / CORPORATE / SCHOOL (never quote bulk pricing — route to specialist) ---")
    lines.append(f"- {GROUPS}")
    lines.append("")
    lines.append("--- SAFETY & GUIDELINES ---")
    for s in SAFETY:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("--- PEAK DAYS ---")
    lines.append(f"- {PEAK_DAYS}")
    return "\n".join(lines)


# ===========================================================================
# Activity availability — the CLOSED list of what PUNO Jaipur actually offers.
#
# ATTRACTIONS above is the complete offering, so "not in this list" is a fact,
# not a guess: the agent can decline confidently. A small AMBIGUOUS set is kept
# for things that might or might not map to an existing zone — those must NOT be
# answered either way; the agent says it will confirm.
# ===========================================================================

# Extra spoken aliases -> canonical attraction name. Matching on names and
# descriptions alone misses how people actually ask ("jumping", "cricket net").
_ACTIVITY_ALIASES = {
    "trampoline": "Adventure & Trampoline",
    "trampolining": "Adventure & Trampoline",
    "jumping": "Adventure & Trampoline",
    "rope course": "Adventure & Trampoline",
    "ninja": "Adventure & Trampoline",
    "obstacle": "Adventure & Trampoline",
    "adventure": "Adventure & Trampoline",
    "sky wall": "Adventure & Trampoline",
    "climbing wall": "Adventure & Trampoline",
    "foam pit": "PUNO Junior / Kids Zone",
    "sand pool": "PUNO Junior / Kids Zone",
    "kids zone": "PUNO Junior / Kids Zone",
    "kids area": "PUNO Junior / Kids Zone",
    "junior": "PUNO Junior / Kids Zone",
    "soft play": "PUNO Junior / Kids Zone",
    "bowling": "Bowling",
    "bowling alley": "Bowling",
    "vr": "VR Rides",
    "virtual reality": "VR Rides",
    "arcade": "Arcade",
    "video game": "Arcade",
    "claw machine": "Arcade",
    "shooting game": "Arcade",
    "racing": "Mad Wheels",
    "mad wheels": "Mad Wheels",
    "cricket": "Others (Cube Chase, Cricket Game, Social Zone)",
    "cube chase": "Others (Cube Chase, Cricket Game, Social Zone)",
    "social zone": "Others (Cube Chase, Cricket Game, Social Zone)",
    "vex": "VEX",
    "cafe": "Pots & Plates",
    "restaurant": "Pots & Plates",
    "food": "Pots & Plates",
}

# Things that plausibly map onto an existing zone but are NOT confirmed. The
# agent must not confirm OR deny these — it says it will check.
AMBIGUOUS_ACTIVITIES = (
    "go kart", "go-kart", "gokart", "karting", "go carting", "go cart",
    "bumper car", "simulator", "motion ride", "4d",
)


def _norm_activity(q: str) -> str:
    import re as _re
    return _re.sub(r"[^a-z0-9 ]+", " ", str(q or "").lower()).strip()


def find_activity(query: str) -> Optional[str]:
    """Canonical attraction name if `query` names something PUNO offers, else None."""
    q = _norm_activity(query)
    if not q:
        return None
    for alias, canon in _ACTIVITY_ALIASES.items():
        if alias in q:
            return canon
    for a in ATTRACTIONS:
        name_n = _norm_activity(a["name"])
        if name_n and (name_n in q or q in name_n):
            return a["name"]
        # match individual sub-attractions named inside the description
        for piece in _norm_activity(a["desc"]).split(","):
            piece = piece.strip()
            if len(piece) >= 4 and piece in q:
                return a["name"]
    return None


def activity_availability(query: str) -> dict:
    """Definitive availability answer for a requested game/activity.

    status: "available" | "not_offered" | "uncertain"
    Returns alternatives to offer when declining.
    """
    q = _norm_activity(query)
    if not q:
        return {"status": "uncertain", "matched": None, "alternatives": [], "query": query}
    for amb in AMBIGUOUS_ACTIVITIES:
        if _norm_activity(amb) in q:
            return {"status": "uncertain", "matched": None, "alternatives": [], "query": query}
    matched = find_activity(query)
    if matched:
        return {"status": "available", "matched": matched, "alternatives": [], "query": query}
    return {
        "status": "not_offered",
        "matched": None,
        "alternatives": ["Adventure & Trampoline", "Bowling", "VR Rides"],
        "query": query,
    }


def available_activity_names() -> list:
    """The closed list of attraction names, for the prompt and for tools."""
    return [a["name"] for a in ATTRACTIONS]
