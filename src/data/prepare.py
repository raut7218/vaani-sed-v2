"""Parquet row -> manifest record: tier resolution and event parsing.

Used by `scripts/download_data.py`, which materialises the corpus.

Manifest record:
{
  "uid": "...", "path": "audio/x.wav", "duration": 2.17, "tier": "gold|silver|bronze",
  "state": "...", "district": "...", "language": "...",
  "events":  [{"cls": "animal_sound", "start": 0.5, "end": 1.7}],   # [] for bronze
  "clip_labels": ["animal_sound"]                                    # always present
}
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.labels import canon_category, resolve_event_class  # noqa: E402

# Columns that carry the annotation tier. Matched case- and separator-insensitively,
# so `annotationQuality`, `annotation_quality` and `AnnotationQuality` all hit.
TIER_COLUMNS = ["annotationquality", "tier", "quality", "splittier", "labelquality",
                "annotationtier"]
VERIFIED_COLUMNS = ["verified", "isverified", "agreement", "numannotators", "nannotators"]

# Observed / plausible spellings of the three tiers. The dataset is gated, so this
# is deliberately generous; anything unmatched is reported rather than guessed.
QUALITY_ALIASES = {
    "gold": "gold", "goldstandard": "gold", "verified": "gold", "high": "gold",
    "tier1": "tier1_gold", "1": "gold", "multiannotator": "gold",
    "multipleannotator": "gold", "doubleannotated": "gold", "highquality": "gold",
    "silver": "silver", "unverified": "silver", "medium": "silver", "2": "silver",
    "singleannotator": "silver", "singleannotated": "silver", "mediumquality": "silver",
    "bronze": "bronze", "weak": "bronze", "low": "bronze", "3": "bronze",
    "tagonly": "bronze", "notimestamp": "bronze", "cliplevel": "bronze",
    "lowquality": "bronze", "weaklabel": "bronze",
}
QUALITY_ALIASES["tier1"] = "gold"
QUALITY_ALIASES["tier2"] = "silver"
QUALITY_ALIASES["tier3"] = "bronze"

# Values of the tier column that could not be mapped; surfaced at the end of a run.
UNMAPPED_QUALITY: Counter = Counter()


def _to_float(v) -> float | None:
    """Parse a timestamp that may arrive as float, int or string."""
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        f = float(v)
        return f if f == f else None            # reject NaN
    s = str(v).strip().replace(",", ".")
    if not s or s.lower() in ("na", "nan", "none", "null", "-"):
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return f if f == f else None


def _norm_key(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _lookup(row: dict, names: list) -> tuple:
    """Fetch a column by normalised name; returns (found, value)."""
    norm = {_norm_key(k): k for k in row}
    for n in names:
        k = norm.get(n)
        if k is not None:
            return True, row[k]
    return False, None


def quality_to_tier(value) -> str | None:
    """Map an annotationQuality value onto gold/silver/bronze, or None."""
    if value is None:
        return None
    t = _norm_key(value)
    if not t:
        return None
    if t in QUALITY_ALIASES:
        return QUALITY_ALIASES[t]
    # Substring fallback, longest key first. "verified" (-> gold) is itself a
    # substring of "unverified" (-> silver), so the real dataset's
    # "unverified_timestamps" would otherwise match "verified" first - in
    # insertion order that key comes before "unverified" - and every
    # unverified-timestamp clip would silently collapse into gold. Checking
    # the most specific (longest) alias first means "unverified" wins over the
    # "verified" it happens to contain, regardless of dict insertion order.
    for key in sorted(QUALITY_ALIASES, key=len, reverse=True):
        if key in t:
            return QUALITY_ALIASES[key]
    return None


def _resolve_tier(row: dict, has_ts: bool, gold_ids: set, default_ts_tier: str) -> str:
    """Assign gold / silver / bronze.

    Priority: annotation-quality column -> verification column -> gold-id list ->
    configured default. A clip with no usable timestamps is always bronze
    regardless of what the quality column claims: bronze is defined by having no
    timestamps, and the frame-level loss has nothing to consume without them.
    """
    found, raw = _lookup(row, TIER_COLUMNS)
    if found:
        tier = quality_to_tier(raw)
        if tier is not None:
            # A gold/silver label without timestamps still cannot supply frame
            # supervision, so it is demoted to bronze.
            return tier if has_ts else "bronze"
        if raw not in (None, ""):
            UNMAPPED_QUALITY[str(raw)] += 1

    if not has_ts:
        return "bronze"

    found, v = _lookup(row, VERIFIED_COLUMNS)
    if found:
        if isinstance(v, bool):
            return "gold" if v else "silver"
        if isinstance(v, (int, float)):
            return "gold" if v >= 2 else "silver"
    if gold_ids:
        for key in ("uid", "id", "segment_id", "audio_id", "imageFileName"):
            if str(row.get(key, "")) in gold_ids:
                return "gold"
        return "silver"
    return default_ts_tier


def build_record(row: dict, uid: str, duration: float, gold_ids: set,
                 default_ts_tier: str, expand_vehicle: bool,
                 unknown: Counter | None = None) -> dict:
    """Turn one raw dataset row into a manifest record.

    Shared by `prepare.py` (HF `load_dataset` path) and
    `scripts/download_data.py` (direct parquet path) so the two cannot drift.
    """
    unknown = unknown if unknown is not None else Counter()
    ts = _lookup(row, ["noisesubcategorytimestamp"])[1] or []
    tier = _resolve_tier(row, len(ts) > 0, gold_ids, default_ts_tier)

    events = []
    for ev in ts:
        cls = resolve_event_class(ev.get("category", ""), ev.get("tag", ""), expand_vehicle)
        if cls is None:
            unknown[str(ev.get("category"))] += 1
            continue
        # In the full corpus `start`/`end` are typed as *string*, not float32 as
        # in the earlier sample, so parse defensively: a bad value must skip one
        # event, never abort a multi-hour download.
        s, e = _to_float(ev.get("start")), _to_float(ev.get("end"))
        if s is None or e is None:
            unknown["<unparsable timestamp>"] += 1
            continue
        if e <= s:
            continue
        s = max(0.0, min(s, duration))
        e = max(0.0, min(e, duration))
        if e - s <= 0:
            continue
        events.append({"cls": cls, "start": round(s, 4), "end": round(e, 4),
                       "tag": ev.get("tag", "")})

    clip_labels = []
    for c in (row.get("NoiseCategory") or []):
        cc = canon_category(c)
        if cc is None:
            unknown[str(c)] += 1
        else:
            clip_labels.append(cc)
    for ev in events:
        b = ev["cls"]
        clip_labels.append("vehicle_traffic" if b.startswith("vehicle_") else b)

    if tier != "bronze" and not events:
        tier = "bronze"  # timestamps existed but none survived validation

    return {
        "uid": uid, "duration": round(duration, 4), "tier": tier,
        "state": row.get("state", ""), "district": row.get("district", ""),
        "language": row.get("language", ""), "events": events,
        "clip_labels": sorted(set(clip_labels)),
    }
