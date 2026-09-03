"""Download the Vaani noise-event dataset from Hugging Face and materialise it.

    python scripts/download_data.py --out /content/work/data

The dataset is **gated**: you must accept the terms on the dataset page and supply
a token. In Colab, store it as the secret `HF_TOKEN` and this script finds it
automatically.

Fetches one parquet shard at a time, decodes each row to an audio file, appends to
`manifest.jsonl`, then frees that shard's cache blob. Peak disk stays at roughly
the decoded audio size instead of decoded + 17.7 GB of parquet.

Everything is resumable: clips already in the manifest are skipped, so re-running
after a disconnect costs only the shards you had not reached.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.labels import LabelEncoder  # noqa: E402
from src.data.prepare import (  # noqa: E402
    TIER_COLUMNS, UNMAPPED_QUALITY, _lookup, build_record, quality_to_tier)

REPO = "ARTPARK-IISc/Vaani-Noise-Event-Dataset"
# The earlier 9-clip sample repo, kept for reference:
SAMPLE_REPO = "PavanKumarJ-ARTPARK/Vaani_Noise_Event_TimeStamp"

# Challenge-page figures, for the coverage report.
TARGET_HOURS = 154.6
TARGET_CLIPS = 90637

GATED_HELP = """
This dataset is GATED - every request needs a Hugging Face token from an account
that has been granted access.

  1. Open https://huggingface.co/datasets/%s and click "Agree and access".
  2. Create a token: https://huggingface.co/settings/tokens (read scope is enough).
  3. In Colab: Secrets (key icon in the left sidebar) -> add HF_TOKEN, and turn on
     "Notebook access" for it. This script picks it up automatically.
     On Kaggle: Add-ons -> Secrets -> add HF_TOKEN, and attach it to the notebook.
     This script picks that up automatically too.
     Elsewhere: export HF_TOKEN=hf_...   or pass --token hf_...
"""


def resolve_token(explicit: str = "") -> str | None:
    """Find an HF token: --token, then a Colab/Kaggle secret, then env, then hf CLI login."""
    if explicit:
        return explicit
    try:
        from google.colab import userdata  # type: ignore
        for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
            try:
                v = userdata.get(key)
                if v:
                    print("[auth] using Colab secret %s" % key)
                    return v
            except Exception:  # noqa: BLE001 - secret absent or access not granted
                continue
    except ImportError:
        pass
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        client = UserSecretsClient()
        for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
            try:
                v = client.get_secret(key)
                if v:
                    print("[auth] using Kaggle secret %s" % key)
                    return v
            except Exception:  # noqa: BLE001 - secret absent or access not granted
                continue
    except ImportError:
        pass
    import os
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(key)
        if v:
            print("[auth] using environment variable %s" % key)
            return v
    try:
        from huggingface_hub import HfFolder
        v = HfFolder.get_token()
        if v:
            print("[auth] using cached huggingface-cli login")
            return v
    except Exception:  # noqa: BLE001
        pass
    return None


def list_remote(repo: str, token: str | None) -> list:
    """What the server actually holds, before downloading anything."""
    from huggingface_hub import HfApi
    api = HfApi(token=token or None)
    info = api.repo_info(repo_id=repo, repo_type="dataset", files_metadata=True)
    out = []
    for s in info.siblings:
        size = getattr(s, "size", None)
        if not size and getattr(s, "lfs", None):
            size = s.lfs.size
        out.append({"path": s.rfilename, "size": size})
    return out


def human(n) -> str:
    if not n:
        return "?"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit)
        n /= 1024.0
    return "%.1f GB" % n


def select_shards(shards: list, max_shards: int, start: int) -> list:
    """Optionally take a contiguous slice of the shard list.

    17.7 GB is a long download. Pulling the first few shards gets a real training
    run going in minutes, and the manifest is additive, so later shards can be
    added afterwards without redoing any work.
    """
    shards = sorted(shards, key=lambda f: f["path"])
    if start:
        shards = shards[start:]
    if max_shards:
        shards = shards[:max_shards]
    return shards


def _iter_repair_rows(rows: list, rel_name: str, gold_uids: set):
    """Yield (uid, new_tier) for already-gold uids that resolve to a different
    tier under the current `quality_to_tier`.

    Pure function over plain row dicts (no HF/pyarrow calls) so the uid
    derivation and re-classification logic can be unit tested without network
    access - it must match `main()`'s uid derivation exactly, or a repair run
    would silently mismatch rows to the wrong clips.
    """
    row_i = 0
    for row in rows:
        audio = row.get("audio") or {}
        stem = Path(str(audio.get("path") or "")).stem
        uid = stem if stem else "%s_%07d" % (Path(rel_name).stem, row_i)
        row_i += 1
        if uid not in gold_uids:
            continue
        found, q = _lookup(row, TIER_COLUMNS)
        if not found:
            continue
        new_tier = quality_to_tier(q)
        if new_tier and new_tier != "gold":
            yield uid, new_tier


def repair_tiers(args, token: str | None, shards: list) -> None:
    """Patch `tier` in an already-materialised manifest.jsonl, no re-download of
    audio and no re-decoding - only the (small) parquet metadata is re-fetched.

    The substring-matching bug in `quality_to_tier` could only ever misclassify
    a lower tier's clips as "gold" (never the reverse - "gold" clips could not
    have been silently demoted), so only clips currently tagged gold need
    re-checking against the raw annotationQuality column. Everything else
    (path, events, duration, clip_labels, ...) is left untouched.
    """
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    out = Path(args.out)
    man_path = out / "manifest.jsonl"
    if not man_path.exists():
        raise SystemExit("[repair] no manifest.jsonl at %s - nothing to repair" % man_path)

    recs = []
    with man_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    gold_uids = {r["uid"] for r in recs if r.get("tier") == "gold"}
    print("[repair] %d clips in manifest, %d currently tagged gold to re-check"
          % (len(recs), len(gold_uids)))
    if not gold_uids:
        print("[repair] nothing tagged gold - nothing to do")
        return

    fixed = {}
    for fi, meta in enumerate(shards):
        rel = meta["path"]
        print("[repair] shard %d/%d  %s" % (fi + 1, len(shards), Path(rel).name))
        try:
            pf_path = hf_hub_download(repo_id=args.repo, filename=rel,
                                      repo_type="dataset", token=token,
                                      cache_dir=args.cache or None)
        except Exception as e:  # noqa: BLE001
            print("[repair]   FAILED to fetch %s: %s" % (rel, e))
            continue

        pf = pq.ParquetFile(pf_path)
        for batch in pf.iter_batches(batch_size=args.batch_rows):
            for uid, new_tier in _iter_repair_rows(batch.to_pylist(), rel, gold_uids):
                fixed[uid] = new_tier

        if not args.keep_parquet:
            try:
                Path(pf_path).unlink()
            except OSError:
                pass

    print("[repair] %d/%d previously-gold clips reclassified: %s"
          % (len(fixed), len(gold_uids), dict(Counter(fixed.values()))))
    if not fixed:
        print("[repair] nothing to change - manifest already correct")
        return

    for r in recs:
        nt = fixed.get(r["uid"])
        if nt:
            r["tier"] = nt
    with man_path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[repair] rewrote %s" % man_path)

    stats_path = out / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats["tiers_after_repair"] = dict(Counter(r["tier"] for r in recs))
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print("[repair] updated %s -> tiers_after_repair=%s"
              % (stats_path, stats["tiers_after_repair"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--out", required=True, help="where to write audio/ + manifest.jsonl")
    ap.add_argument("--cache", default="", help="HF cache dir (default: HF's own)")
    ap.add_argument("--token", default="", help="HF token; falls back to Colab secret / env")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--format", default="flac", choices=["wav", "flac"],
                    help="flac is lossless and about half the size of wav (default)")
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N clips")
    ap.add_argument("--max-shards", type=int, default=0,
                    help="download only the first N shards (0 = all). Start small.")
    ap.add_argument("--shard-start", type=int, default=0,
                    help="skip the first N shards (with --max-shards, pages through)")
    ap.add_argument("--gold-ids", default="")
    ap.add_argument("--default-ts-tier", default="silver", choices=["gold", "silver"])
    ap.add_argument("--expand-vehicle", type=int, default=1)
    ap.add_argument("--list-only", action="store_true",
                    help="show what is on the server and exit, downloading nothing")
    ap.add_argument("--repair-tiers", action="store_true",
                    help="patch tier in an existing manifest.jsonl after a "
                         "quality_to_tier fix, without re-downloading or "
                         "re-decoding any audio")
    ap.add_argument("--keep-parquet", action="store_true",
                    help="keep the HF cache; by default each shard's blob is deleted "
                         "once materialised, which halves peak disk use")
    ap.add_argument("--batch-rows", type=int, default=256,
                    help="parquet rows held in memory at once")
    args = ap.parse_args()

    token = resolve_token(args.token)
    if not token:
        print("[auth] no token found - trying anonymously (will fail if gated)")

    # ---- 1. what is on the server ------------------------------------------
    print("[download] repo: %s" % args.repo)
    try:
        remote = list_remote(args.repo, token)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if any(k in msg for k in ("401", "403", "gated", "Gated", "restricted", "Access")):
            raise SystemExit("[download] cannot access %s\n%s\nOriginal error: %s"
                             % (args.repo, GATED_HELP % args.repo, msg))
        raise

    all_shards = [f for f in remote if f["path"].endswith(".parquet")]
    total_bytes = sum(f["size"] or 0 for f in all_shards)
    print("[download] %d parquet shard(s), %s total on the server"
          % (len(all_shards), human(total_bytes)))
    if not all_shards:
        raise SystemExit("no parquet shards found in %s" % args.repo)

    shards = select_shards(all_shards, args.max_shards, args.shard_start)
    sel_bytes = sum(f["size"] or 0 for f in shards)
    if len(shards) != len(all_shards):
        print("[download] selected %d shard(s), %s: %s .. %s"
              % (len(shards), human(sel_bytes),
                 Path(shards[0]["path"]).name, Path(shards[-1]["path"]).name))

    if args.list_only:
        for f in remote[:8]:
            print("   %-45s %s" % (f["path"], human(f["size"])))
        if len(remote) > 8:
            print("   ... and %d more files" % (len(remote) - 8))
        frac = sel_bytes / max(1.0, float(total_bytes))
        est = TARGET_HOURS * 3600 * args.sr * 2 * (0.55 if args.format == "flac" else 1.0)
        print("[download] estimated decoded %s for this selection: ~%s"
              % (args.format, human(est * frac)))
        return

    if args.repair_tiers:
        repair_tiers(args, token, shards)
        return

    # ---- 2. materialise, one shard at a time --------------------------------
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    import soundfile as sf

    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.jsonl"

    done = set()
    if man_path.exists():
        with man_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["uid"])
                except Exception:  # noqa: BLE001
                    pass
        print("[download] resuming: %d clips already in the manifest" % len(done))

    gold_ids = set()
    if args.gold_ids and Path(args.gold_ids).exists():
        gold_ids = {ln.strip() for ln in Path(args.gold_ids).read_text().splitlines()
                    if ln.strip()}
        print("[download] %d gold ids loaded" % len(gold_ids))

    le = LabelEncoder(bool(args.expand_vehicle))
    ext = args.format
    tiers, cls_count, unknown, quality_raw = Counter(), Counter(), Counter(), Counter()
    dur_by_tier = defaultdict(float)
    n_written = n_skipped = n_failed = 0
    stop = False

    with man_path.open("a", encoding="utf-8") as fout:
        for fi, meta in enumerate(shards):
            rel = meta["path"]
            print("[download] shard %d/%d  %s (%s)"
                  % (fi + 1, len(shards), Path(rel).name, human(meta["size"])))
            try:
                pf_path = hf_hub_download(repo_id=args.repo, filename=rel,
                                          repo_type="dataset", token=token,
                                          cache_dir=args.cache or None)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                # Repo *metadata* is public while the files are gated, so a
                # missing/unapproved token only surfaces here. Abort loudly
                # rather than repeating the same 401 for all 182 shards.
                if any(k in msg for k in ("401", "403", "gated", "Gated",
                                          "restricted", "Access", "awaiting")):
                    raise SystemExit(
                        "[download] shard fetch denied - the repo metadata is public "
                        "but its files are gated.\n%s\nOriginal error: %s"
                        % (GATED_HELP % args.repo, msg))
                print("[download]   FAILED to fetch %s: %s" % (rel, e))
                n_failed += 1
                continue

            pf = pq.ParquetFile(pf_path)
            row_i = 0
            for batch in pf.iter_batches(batch_size=args.batch_rows):
                for row in batch.to_pylist():
                    if args.limit and n_written >= args.limit:
                        stop = True
                        break
                    audio = row.get("audio") or {}
                    stem = Path(str(audio.get("path") or "")).stem
                    uid = stem if stem else "%s_%07d" % (Path(rel).stem, row_i)
                    row_i += 1
                    if uid in done:
                        n_skipped += 1
                        continue

                    found, q = _lookup(row, TIER_COLUMNS)
                    if found:
                        quality_raw[str(q)] += 1

                    try:
                        if audio.get("bytes"):
                            import librosa
                            y, _ = librosa.load(io.BytesIO(audio["bytes"]), sr=args.sr,
                                                mono=True)
                        elif audio.get("array") is not None:
                            import numpy as np
                            y = np.asarray(audio["array"], dtype="float32")
                        else:
                            n_failed += 1
                            continue
                        sf.write(str(out / "audio" / ("%s.%s" % (uid, ext))), y, args.sr)
                    except Exception as e:  # noqa: BLE001
                        print("[download]   failed %s: %s" % (uid, e))
                        n_failed += 1
                        continue

                    duration = float(len(y)) / args.sr
                    rec = build_record(row, uid, duration, gold_ids,
                                       args.default_ts_tier, bool(args.expand_vehicle),
                                       unknown)
                    rec["path"] = "audio/%s.%s" % (uid, ext)
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done.add(uid)

                    n_written += 1
                    tiers[rec["tier"]] += 1
                    dur_by_tier[rec["tier"]] += duration
                    for ev in rec["events"]:
                        cls_count[ev["cls"]] += 1
                    if n_written % 1000 == 0:
                        fout.flush()
                        print("[download]   %d clips, %.2f h  tiers=%s"
                              % (n_written, sum(dur_by_tier.values()) / 3600, dict(tiers)))
                if stop:
                    break
            fout.flush()

            if not args.keep_parquet:
                # Free the ~100 MB blob now the shard is materialised. Across 182
                # shards this is the difference between +17.7 GB of disk and none.
                try:
                    Path(pf_path).unlink()
                except OSError:
                    pass
            if stop:
                break

    # ---- 3. report ----------------------------------------------------------
    total_clips = len(done)
    total_h = 0.0
    with man_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total_h += json.loads(line).get("duration", 0.0)
    total_h /= 3600

    stats = {
        "new_clips": n_written, "skipped_existing": n_skipped, "failed": n_failed,
        "total_clips_in_manifest": total_clips,
        "hours_total": round(total_h, 3),
        "shards_processed": len(shards), "shards_on_server": len(all_shards),
        "tiers_this_run": dict(tiers),
        "hours_by_tier_this_run": {k: round(v / 3600, 3) for k, v in dur_by_tier.items()},
        "annotationQuality_values_seen": dict(quality_raw),
        "unmapped_annotationQuality": dict(UNMAPPED_QUALITY),
        "events_per_class_this_run": dict(cls_count),
        "unknown_category_strings": dict(unknown),
        "classes": le.classes,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\n" + json.dumps(stats, indent=2))

    print("\n[download] %d clips, %.2f h  (challenge lists %d clips / %.1f h)"
          % (total_clips, total_h, TARGET_CLIPS, TARGET_HOURS))
    if quality_raw:
        print("[download] annotationQuality values seen: %s"
              % ", ".join("%s x%d" % (k, v) for k, v in quality_raw.most_common(10)))
    if UNMAPPED_QUALITY:
        print("\n[download] *** %d clips had an annotationQuality value that does not map"
              "\n    to gold/silver/bronze: %s"
              "\n    They fell back to timestamp presence. Add them to QUALITY_ALIASES in"
              "\n    src/data/prepare.py and re-run - materialised clips are skipped, so"
              "\n    the re-run only rewrites the manifest and is fast."
              % (sum(UNMAPPED_QUALITY.values()), dict(UNMAPPED_QUALITY)))
    if unknown:
        print("[download] unmapped category strings: %s" % dict(unknown))


if __name__ == "__main__":
    main()
