"""Materialise the HF dataset into local audio files + a JSONL manifest.

Run once (it is resumable). Produces:
    <out>/audio/<uid>.wav        16 kHz mono
    <out>/manifest.jsonl         one record per clip
    <out>/stats.json             tier / class / duration report

Manifest record:
{
  "uid": "...", "path": "audio/x.wav", "duration": 2.17, "tier": "gold|silver|bronze",
  "state": "...", "district": "...", "language": "...",
  "events":  [{"cls": "animal_sound", "start": 0.5, "end": 1.7}],   # [] for bronze
  "clip_labels": ["animal_sound"]                                    # always present
}
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.labels import LabelEncoder, canon_category, resolve_event_class  # noqa: E402

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


def _uid(row: dict, i: int) -> str:
    a = row.get("audio")
    p = ""
    if isinstance(a, dict):
        p = a.get("path") or ""
    p = os.path.basename(str(p))
    stem = os.path.splitext(p)[0]
    return stem if stem else "clip_%07d" % i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="PavanKumarJ-ARTPARK/Vaani_Noise_Event_TimeStamp")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N clips")
    ap.add_argument("--streaming", action="store_true",
                    help="stream instead of downloading the whole parquet set first")
    ap.add_argument("--gold-ids", default="", help="optional txt file, one gold uid per line")
    ap.add_argument("--default-ts-tier", default="silver", choices=["gold", "silver"],
                    help="tier for timestamped clips when the dataset exposes no tier field")
    ap.add_argument("--expand-vehicle", type=int, default=1)
    args = ap.parse_args()

    import soundfile as sf
    from datasets import load_dataset

    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)

    gold_ids = set()
    if args.gold_ids and Path(args.gold_ids).exists():
        gold_ids = {ln.strip() for ln in Path(args.gold_ids).read_text().splitlines() if ln.strip()}
        print("[prepare] loaded %d gold ids" % len(gold_ids))

    le = LabelEncoder(bool(args.expand_vehicle))
    ds = load_dataset(args.repo, split=args.split, streaming=args.streaming)
    # Keep raw bytes: decoding through HF's Audio feature is slower and we
    # re-encode to a fixed 16 kHz mono wav anyway.
    try:
        import datasets as _hfds
        ds = ds.cast_column("audio", _hfds.Audio(decode=False))
    except Exception as e:  # noqa: BLE001
        print("[prepare] cast_column(decode=False) unavailable (%s); decoding normally" % e)

    man_path = out / "manifest.jsonl"
    done = set()
    if man_path.exists():
        with man_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["uid"])
                except Exception:  # noqa: BLE001
                    pass
        print("[prepare] resuming, %d clips already written" % len(done))

    tiers, cls_count, unknown = Counter(), Counter(), Counter()
    dur_by_tier = defaultdict(float)
    n_written = 0

    with man_path.open("a", encoding="utf-8") as fout:
        for i, row in enumerate(ds):
            if args.limit and i >= args.limit:
                break
            uid = _uid(row, i)
            if uid in done:
                continue

            ts = row.get("NoiseSubCategoryTimeStamp") or []
            has_ts = len(ts) > 0
            tier = _resolve_tier(row, has_ts, gold_ids, args.default_ts_tier)

            # --- audio ---
            wav_path = out / "audio" / (uid + ".wav")
            if not wav_path.exists():
                a = row.get("audio")
                try:
                    if isinstance(a, dict) and a.get("bytes"):
                        import librosa
                        y, _ = librosa.load(io.BytesIO(a["bytes"]), sr=args.sr, mono=True)
                    elif isinstance(a, dict) and a.get("array") is not None:
                        import numpy as np
                        y = np.asarray(a["array"], dtype="float32")
                        if a.get("sampling_rate") and a["sampling_rate"] != args.sr:
                            import librosa
                            y = librosa.resample(y, orig_sr=a["sampling_rate"], target_sr=args.sr)
                    else:
                        print("[prepare] no audio for %s, skipping" % uid)
                        continue
                    sf.write(str(wav_path), y, args.sr)
                except Exception as e:  # noqa: BLE001
                    print("[prepare] failed to decode %s: %s" % (uid, e))
                    continue
                duration = float(len(y)) / args.sr
            else:
                duration = float(sf.info(str(wav_path)).duration)

            # --- events ---
            events = []
            for ev in ts:
                cls = resolve_event_class(ev.get("category", ""), ev.get("tag", ""),
                                          bool(args.expand_vehicle))
                if cls is None:
                    unknown[str(ev.get("category"))] += 1
                    continue
                s, e = float(ev.get("start", 0.0)), float(ev.get("end", 0.0))
                if e <= s:
                    continue
                s = max(0.0, min(s, duration))
                e = max(0.0, min(e, duration))
                if e - s <= 0:
                    continue
                events.append({"cls": cls, "start": round(s, 4), "end": round(e, 4),
                               "tag": ev.get("tag", "")})
                cls_count[cls] += 1

            clip_labels = []
            for c in (row.get("NoiseCategory") or []):
                cc = canon_category(c)
                if cc is None:
                    unknown[str(c)] += 1
                else:
                    clip_labels.append(cc)
            # Events imply clip labels even when NoiseCategory is missing.
            for ev in events:
                b = ev["cls"]
                clip_labels.append("vehicle_traffic" if b.startswith("vehicle_") else b)
            clip_labels = sorted(set(clip_labels))

            if tier != "bronze" and not events:
                tier = "bronze"  # timestamps existed but none survived validation

            rec = {
                "uid": uid, "path": "audio/%s.wav" % uid, "duration": round(duration, 4),
                "tier": tier, "state": row.get("state", ""), "district": row.get("district", ""),
                "language": row.get("language", ""), "events": events, "clip_labels": clip_labels,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            tiers[tier] += 1
            dur_by_tier[tier] += duration
            if n_written % 500 == 0:
                fout.flush()
                print("[prepare] %d clips  tiers=%s" % (n_written, dict(tiers)))

    stats = {
        "written": n_written,
        "tiers": dict(tiers),
        "hours_by_tier": {k: round(v / 3600, 3) for k, v in dur_by_tier.items()},
        "events_per_class": dict(cls_count),
        "unknown_category_strings": dict(unknown),
        "classes": le.classes,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    if unknown:
        print("\n[prepare] WARNING: unmapped category strings above. "
              "Add them to ALIASES in src/data/labels.py so they are not dropped.")


if __name__ == "__main__":
    main()
