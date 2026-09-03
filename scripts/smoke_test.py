"""End-to-end test on synthetic audio: no download, no GPU, ~1-2 minutes.

Builds a tiny corpus with events at known times, trains a few epochs, runs
inference, and validates the submission archive against every rule the scorer
enforces. If this passes, the wiring is sound.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_corpus(root: Path, n: int = 96, sr: int = 16000, dur: float = 6.0):
    import soundfile as sf
    rng = np.random.default_rng(0)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    classes = ["animal_sound", "vehicle_horn", "human_non_speech", "singing_music"]
    states = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    lines = []
    for i in range(n):
        n_s = int(dur * sr)
        y = rng.normal(0, 0.01, n_s).astype("float32")
        events = []
        for _ in range(rng.integers(1, 3)):
            t0 = float(rng.uniform(0.2, dur - 1.5))
            ln = float(rng.uniform(0.25, 1.2))
            f = float(rng.uniform(400, 3000))
            a, b = int(t0 * sr), int((t0 + ln) * sr)
            t = np.arange(b - a) / sr
            env = np.hanning(len(t)).astype("float32")
            y[a:b] += (0.3 * np.sin(2 * np.pi * f * t) * env).astype("float32")
            events.append({"cls": classes[int(rng.integers(len(classes)))],
                           "start": round(t0, 3), "end": round(t0 + ln, 3)})
        uid = "IISc_VaaniProject_K_%s_D1_smoke%03d" % (states[i % len(states)], i)
        sf.write(str(root / "audio" / (uid + ".wav")), y, sr)
        tier = "gold" if i % 3 else "silver"
        if i % 11 == 0:                       # a few tag-only bronze clips
            lines.append(json.dumps({"uid": uid, "path": "audio/%s.wav" % uid,
                                     "duration": dur, "tier": "bronze",
                                     "state": states[i % len(states)], "district": "D1",
                                     "events": [],
                                     "clip_labels": [events[0]["cls"]]}))
            continue
        lines.append(json.dumps({"uid": uid, "path": "audio/%s.wav" % uid,
                                 "duration": dur, "tier": tier,
                                 "state": states[i % len(states)], "district": "D1",
                                 "events": sorted(events, key=lambda e: e["start"]),
                                 "clip_labels": sorted({e["cls"] for e in events})}))
    (root / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cmd, cwd=None):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd or ROOT)
    if r.returncode != 0:
        raise SystemExit("FAILED: %s" % " ".join(str(c) for c in cmd))


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="vaani_smoke_"))
    try:
        data, run_dir = tmp / "data", tmp / "run"
        print("[smoke] building synthetic corpus in %s" % data)
        make_corpus(data)

        cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
        cfg["model"]["encoders"] = []          # no downloads in the smoke test
        cfg["model"]["d_model"] = 96
        cfg["model"]["n_levels"] = 4
        cfg["model"]["n_base_layers"] = 1
        cfg["data"]["clip_len"] = 6.0
        cfg["data"]["n_folds"] = 5
        cfg["train"].update(epochs=3, batch_size=8, num_workers=0, warmup_steps=10,
                            unfreeze_epoch=999, lr=0.001)
        cfg_path = tmp / "smoke.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        run([sys.executable, "-m", "src.train.train", "--config", str(cfg_path),
             "--data", str(data), "--out", str(run_dir)])
        assert (run_dir / "best.pt").exists(), "no checkpoint written"

        run([sys.executable, "-m", "src.infer.predict", "--ckpt",
             str(run_dir / "best.pt"), "--audio-dir", str(data / "audio"),
             "--out", str(tmp / "submission.zip"), "--num-workers", "0",
             "--batch-size", "8"])

        # --- validate the archive exactly as the scorer would read it ---
        zp = tmp / "submission.zip"
        with zipfile.ZipFile(zp) as z:
            names = z.namelist()
            assert names == ["predictions.jsonl"], \
                "archive must hold exactly predictions.jsonl at the root, got %s" % names
            rows = [json.loads(l) for l in
                    z.read("predictions.jsonl").decode("utf-8").splitlines() if l.strip()]

        uids = {p.stem for p in (data / "audio").glob("*.wav")}
        got = [r["clip_id"] for r in rows]
        assert len(got) == len(set(got)), "duplicate clip_id in submission"
        assert set(got) == uids, "every eval clip must appear exactly once"
        for r in rows:
            last = -1.0
            for e in r["events"]:
                assert e["onset"] >= 0, "negative onset in %s" % r["clip_id"]
                assert e["offset"] >= e["onset"], "offset before onset"
                assert e["onset"] >= last - 1e-9, "events must be non-decreasing"
                last = e["onset"]
        n_ev = float(np.mean([len(r["events"]) for r in rows]))
        print("\n[smoke] archive valid: %d clips, %.2f events/clip" % (len(rows), n_ev))

        hist = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        print("[smoke] best val score %.4f" % max(h["score"] for h in hist))
        print("\n[smoke] PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
