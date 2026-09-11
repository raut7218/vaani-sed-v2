"""Pack the prepared corpus into a few large files a Kaggle notebook can save.

    python scripts/pack_data.py --data /tmp/data --synth /tmp/synth --out /kaggle/working/vaani

A Kaggle notebook's saved output is capped at 500 files, and the corpus is
~110k (90k clips + 20k synthetic + one VAD array per clip). So the one-off data
notebook packs everything and the training notebook attaches that output as an
input, instead of every 12 h session spending its first hour re-downloading
16.5 GB and re-running VAD and synthesis.

Layout written to --out:

    audio_000.bin ...     original FLAC bytes, concatenated, <= --pack-gb each
    vad.f16               every clip's 25 fps speech probabilities, float16
    manifest.jsonl        the corpus manifest + pack/off/nbytes (+ vad_off/vad_n)
    stats.json            copied from the download
    synth/audio_000.bin   the synthetic clips, packed the same way
    synth/manifest.jsonl  -> usable as `--extra-data <out>/synth`

The audio is not re-encoded: each clip's file bytes are copied verbatim, and
`VaaniSpanDataset` decodes them from memory exactly as it would from disk.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def load_manifest(p: Path) -> list:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def pack_audio(recs: list, src: Path, out: Path, pack_bytes: int,
               move: bool = False) -> int:
    """Append every record's file to rolling packs; rewrite `path` into a ref."""
    out.mkdir(parents=True, exist_ok=True)
    idx, off, f = 0, 0, None
    missing = 0
    kept = []
    for i, r in enumerate(recs):
        p = src / r["path"]
        try:
            blob = p.read_bytes()
        except OSError:
            missing += 1
            continue
        if f is None or off + len(blob) > pack_bytes:
            if f is not None:
                f.close()
                idx += 1
            f = (out / ("audio_%03d.bin" % idx)).open("wb")
            off = 0
        f.write(blob)
        if move:
            # Peak disk stays at one copy of the corpus rather than two, which
            # matters when the scratch dir shares a volume with the output.
            p.unlink()
        r = dict(r)
        r.pop("path", None)
        r.update(pack="audio_%03d.bin" % idx, off=off, nbytes=len(blob))
        kept.append(r)
        off += len(blob)
        if (i + 1) % 10000 == 0:
            print("[pack] %s: %d/%d clips" % (out.name, i + 1, len(recs)), flush=True)
    if f is not None:
        f.close()
    recs[:] = kept
    return missing


def pack_vad(recs: list, vad_dir: Path, out: Path) -> int:
    """Concatenate per-clip VAD arrays into one float16 file; returns clips with VAD."""
    have = 0
    off = 0
    with (out / "vad.f16").open("wb") as f:
        for r in recs:
            p = vad_dir / (r["uid"] + ".npy")
            if not p.exists():
                continue
            v = np.load(p).astype("float16")
            f.write(v.tobytes())
            r["vad_off"], r["vad_n"] = off, int(v.size)
            off += int(v.size)
            have += 1
    return have


def write_manifest(recs: list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="download_data.py output (+ vad/)")
    ap.add_argument("--synth", default="", help="make_synthetic.py output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pack-gb", type=float, default=4.0)
    ap.add_argument("--move", action="store_true",
                    help="delete each source file once it is packed")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pack_bytes = int(args.pack_gb * (1 << 30))

    recs = load_manifest(data / "manifest.jsonl")
    n0 = len(recs)
    missing = pack_audio(recs, data, out, pack_bytes, args.move)
    vad_dir = data / "vad"
    n_vad = pack_vad(recs, vad_dir, out) if vad_dir.exists() else 0
    write_manifest(recs, out / "manifest.jsonl")
    if (data / "stats.json").exists():
        shutil.copy(data / "stats.json", out / "stats.json")
    print("[pack] corpus: %d/%d clips packed (%d missing audio), %d with VAD"
          % (len(recs), n0, missing, n_vad))
    if missing > 0.01 * n0:
        sys.exit("[pack] more than 1%% of the audio is missing - the download did "
                 "not finish; refusing to save a partial corpus")

    if args.synth:
        srecs = load_manifest(Path(args.synth) / "manifest.jsonl")
        m = pack_audio(srecs, Path(args.synth), out / "synth", pack_bytes, args.move)
        write_manifest(srecs, out / "synth" / "manifest.jsonl")
        print("[pack] synthetic: %d clips packed (%d missing)" % (len(srecs), m))

    files = sorted(p for p in out.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print("[pack] %d files, %.2f GB in %s" % (len(files), total / (1 << 30), out))
    for p in files:
        print("   %-28s %8.1f MB" % (p.relative_to(out), p.stat().st_size / (1 << 20)))


if __name__ == "__main__":
    main()
