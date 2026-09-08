"""Dry run the whole pipeline on a couple of shards before committing to all 182.

    python scripts/preflight.py --data /kaggle/working/data \
        --config configs/kaggle.yaml --shards 2

Downloads N real shards, runs VAD + synthesis + one distributed training epoch +
a checkpoint save on them, then extrapolates disk use to the full corpus and
refuses to continue if it will not fit.

Two failures this is here to catch, both from a real session:

  * a crash in a stage *below* the download, found 35 minutes in, after the
    download has already been paid for. Every such stage runs here on ~1% of the
    data, in a couple of minutes.
  * `No space left on device` at the first checkpoint save - i.e. after two
    epochs of GPU time - because corpus + VAD + synthetic audio came to more
    than the working quota. Nothing about that is visible until it is far too
    late, so it is measured here rather than guessed.

The downloaded shards are the real thing, written into the real `--data`
directory, and `download_data.py` skips them afterwards - so the dry run costs
no duplicate download.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GB = float(1 << 30)

# /kaggle/working is capped at 20 GB whatever the filesystem reports free - and
# it is the cap, not the filesystem, that ends the session.
KAGGLE_QUOTA_GB = 20.0


def du(path: Path) -> int:
    """Bytes under `path` (0 if absent)."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def budget(path: Path) -> float:
    """Bytes still writable at `path`, honouring Kaggle's working-dir quota."""
    free = float(shutil.disk_usage(path).free)
    work = Path("/kaggle/working")
    if work.exists() and str(path).startswith(str(work)):
        free = min(free, KAGGLE_QUOTA_GB * GB - du(work))
    return max(0.0, free)


def run(stage: str, cmd: list) -> None:
    print("\n[preflight] %s\n$ %s" % (stage, " ".join(str(c) for c in cmd)), flush=True)
    t0 = time.time()
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        raise SystemExit(
            "\n[preflight] FAILED at '%s'. This is the dry run on a few shards - "
            "the same error would otherwise have hit after the full download, an "
            "hour from now. Fix it and re-run; the shards already fetched are "
            "kept, so the retry is cheap." % stage)
    print("[preflight] %s ok (%.0fs)" % (stage, time.time() - t0), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--shards", type=int, default=2, help="shards to dry-run on")
    ap.add_argument("--target-shards", type=int, default=0,
                    help="MAX_SHARDS the real run will fetch, for the projection "
                         "(0 = everything on the server)")
    ap.add_argument("--synth-n", type=int, default=200,
                    help="synthetic clips to make here, to measure their size")
    ap.add_argument("--target-synth", type=int, default=20000,
                    help="N_SYNTHETIC the real run will make, for the projection")
    ap.add_argument("--nproc", type=int, default=2, help="GPUs, as the real run uses")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--margin-gb", type=float, default=2.0,
                    help="headroom for pip, the torch hub cache and the saved output")
    ap.add_argument("--no-vad", action="store_true")
    ap.add_argument("--no-synth", action="store_true")
    ap.add_argument("--no-train", action="store_true",
                    help="skip the 1-epoch train (disk projection only)")
    args = ap.parse_args()

    data = Path(args.data)
    data.mkdir(parents=True, exist_ok=True)
    # All the dry run's own output lives here and is deleted before the verdict,
    # so it is not counted against the space the real run needs.
    tmp = data.parent / "preflight"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    run_dir = tmp / "run"

    print("[preflight] %.1f GB writable at %s" % (budget(data) / GB, data))

    # ---- 1. a couple of real shards ----------------------------------------
    run("download %d shards" % args.shards,
        [sys.executable, "scripts/download_data.py", "--out", str(data),
         "--max-shards", str(args.shards)])

    stats = json.loads((data / "stats.json").read_text(encoding="utf-8"))
    done, on_server = int(stats["shards_processed"]), int(stats["shards_on_server"])
    if done < 1:
        raise SystemExit("[preflight] no shard materialised - see the download log above.")
    # Project to what the real run will actually fetch, not always to all 182:
    # with MAX_SHARDS set, projecting the whole corpus would refuse a run that
    # fits comfortably.
    target = min(on_server, args.target_shards) if args.target_shards > 0 else on_server
    scale = max(1.0, target / float(done))

    # ---- 2. every stage that comes after the download -----------------------
    if not args.no_vad:
        run("vad", [sys.executable, "scripts/make_vad.py", "--data", str(data)])
    if not args.no_synth:
        run("synthetic",
            [sys.executable, "scripts/make_synthetic.py", "--data", str(data),
             "--out", str(tmp / "synth"), "-n", str(args.synth_n)])
    if not args.no_train:
        # The real launcher, the real config, the real encoders, and - the whole
        # point - a real checkpoint write at the end of it.
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node=%d" % args.nproc, "-m", "src.train.train",
               "--config", args.config, "--data", str(data), "--out", str(run_dir),
               "--epochs", "1", "--batch-size", str(args.batch_size)]
        if not args.no_synth:
            cmd += ["--extra-data", str(tmp / "synth")]
        run("train 1 epoch + eval + checkpoint save (%d GPUs)" % args.nproc, cmd)
        if not (run_dir / "best.pt").exists():
            raise SystemExit("[preflight] training finished but wrote no best.pt")

    # ---- 3. what the full run will actually cost ----------------------------
    audio_now, vad_now = du(data / "audio"), du(data / "vad")
    n_synth = len(list((tmp / "synth" / "audio").glob("*"))) or 1
    synth_full = du(tmp / "synth") / n_synth * args.target_synth
    audio_full, vad_full = audio_now * scale, vad_now * scale
    ckpts = du(run_dir)

    shutil.rmtree(tmp, ignore_errors=True)   # before measuring what is left
    free = budget(data)
    todo = ((audio_full - audio_now) + (vad_full - vad_now) + synth_full
            + ckpts + args.margin_gb * GB)

    print("\n[preflight] projected disk for the full run (%d of %d shards):"
          % (target, on_server))
    print("  corpus audio      %6.1f GB  (%.2f GB here x %.1f)"
          % (audio_full / GB, audio_now / GB, scale))
    print("  vad labels        %6.1f GB" % (vad_full / GB))
    print("  synthetic x%-6d %6.1f GB" % (args.target_synth, synth_full / GB))
    print("  checkpoints       %6.1f GB  (measured; does not grow with the corpus)"
          % (ckpts / GB))
    print("  margin            %6.1f GB" % args.margin_gb)
    print("  -----------------------------")
    print("  still to write    %6.1f GB" % (todo / GB))
    print("  writable now      %6.1f GB" % (free / GB))

    if todo > free:
        per_shard = (audio_full + vad_full) / target
        fits = done + int(max(0.0, free - synth_full - ckpts - args.margin_gb * GB)
                          / max(per_shard, 1.0))
        raise SystemExit(
            "\n[preflight] STOP: the full run needs %.1f GB and only %.1f GB is "
            "writable here.\n"
            "Left alone this surfaces as `No space left on device` at the first "
            "checkpoint save - after two epochs of GPU time, with nothing saved.\n"
            "Pick one in the CONFIG cell and re-run:\n"
            "  * MAX_SHARDS = %-6d (fits: ~%.0f%% of the corpus)\n"
            "  * N_SYNTHETIC = 0      (frees %.1f GB)\n"
            "  * USE_VAD = False      (frees %.1f GB, drops the speech head)\n"
            % (todo / GB, free / GB, fits, 100.0 * fits / on_server,
               synth_full / GB, vad_full / GB))

    print("\n[preflight] PASS - every stage runs, and %d shards fit." % target)


if __name__ == "__main__":
    main()
