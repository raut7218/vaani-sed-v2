"""Proves the span head is time-aligned to the audio. `python tests/test_overfit.py`

This is the test that catches the nastiest class of bug in this codebase: a
silent time offset between the waveform and the regression targets. A shift like
that does not show up in the loss - the model simply learns the offset - and it
costs the entire event-F1 term, because every boundary is then systematically
wrong by exactly the amount the metric refuses to forgive.

If you change the front-end, the CNN pooling, `level_point_coords`, or
`decode_spans`, run this. It overfits a handful of clips with tones at known
times and asserts the decoded events land on the tones.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import evaluate                            # noqa: E402
from src.infer.runner import candidates_to_events, spans_from_output   # noqa: E402
from src.models.span_model import build_model                          # noqa: E402
from src.train.losses import SpanLoss                                  # noqa: E402
from src.data.dataset import MAX_EVENTS                                # noqa: E402

SR, FPS, CLIP = 16000, 25.0, 6.0
N_FRAMES = int(CLIP * FPS)


def make_batch(n=8, seed=0):
    rng = np.random.default_rng(seed)
    wav = rng.normal(0, 0.005, (n, int(CLIP * SR))).astype("float32")
    spans = np.full((n, MAX_EVENTS, 2), -1.0, "float32")
    cls = np.full((n, MAX_EVENTS), -1, "int64")
    frame = np.zeros((n, N_FRAMES, 4), "float32")
    refs = {}
    for i in range(n):
        # Deliberately off-grid: 0.03 s steps against a 0.04 s grid, so a model
        # that can only emit grid points cannot pass.
        t0 = 0.5 + 0.31 * i
        ln = 0.9 + 0.03 * i
        a, b = int(t0 * SR), int((t0 + ln) * SR)
        t = np.arange(b - a) / SR
        wav[i, a:b] += (0.5 * np.sin(2 * np.pi * 900 * t) * np.hanning(len(t))).astype("float32")
        spans[i, 0] = (t0 * FPS, (t0 + ln) * FPS)
        cls[i, 0] = 0
        frame[i, int(t0 * FPS):int((t0 + ln) * FPS), 0] = 1.0
        refs["c%d" % i] = [[t0, t0 + ln]]
    return {
        "wav": torch.from_numpy(wav),
        "spans": torch.from_numpy(spans),
        "span_cls": torch.from_numpy(cls),
        "frame_target": torch.from_numpy(frame),
        "clip_target": torch.from_numpy(frame.max(1)),
        "frame_valid": torch.ones(n, N_FRAMES),
        "speech_target": torch.zeros(n, N_FRAMES),
        "has_vad": torch.zeros(n),
        "n_events": torch.ones(n, dtype=torch.long),
        "tier": torch.zeros(n, dtype=torch.long),
        "uid": ["c%d" % i for i in range(n)],
    }, refs


def main() -> None:
    torch.manual_seed(0)
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs"
                          / "default.yaml").read_text(encoding="utf-8"))
    cfg["model"].update(encoders=[], d_model=128, n_levels=4, n_base_layers=1,
                        dropout=0.0, specaug=False)
    cfg["data"]["clip_len"] = CLIP
    model = build_model(cfg, 4, None)
    crit = SpanLoss(cfg, 4, int(cfg["model"]["n_bins"]))
    batch, refs = make_batch()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for step in range(400):
        out = model(batch["wav"], batch["frame_valid"])
        loss, logs = crit(out, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 100 == 0:
            print("step %3d  loss %.4f  reg %.4f  dfl %.4f"
                  % (step, float(logs["loss"]), float(logs["reg"]), float(logs["dfl"])))

    model.eval()
    with torch.no_grad():
        out = model(batch["wav"], batch["frame_valid"])
    dur = np.full((len(refs),), CLIP)
    cands = spans_from_output(out, FPS, dur, cfg["postproc"])
    preds = {u: candidates_to_events(c, cfg["postproc"])
             for u, c in zip(batch["uid"], cands)}
    r = evaluate(preds, refs)

    print("\nrefs :", {k: [[round(x, 3) for x in e] for e in v] for k, v in list(refs.items())[:3]})
    print("preds:", {k: preds[k] for k in list(preds)[:3]})
    print("\nF1 %.4f  Dice %.4f  score %.4f" % (r["event_f1"], r["segment_dice"], r["score"]))

    errs = []
    for u, ref in refs.items():
        if preds[u]:
            errs.append(abs(preds[u][0][0] - ref[0][0]))
            errs.append(abs(preds[u][0][1] - ref[0][1]))
    if errs:
        print("mean absolute boundary error: %.1f ms  (metric tolerance here: %.0f ms)"
              % (1000 * float(np.mean(errs)), 1000 * max(0.2 * 0.9, 0.05)))

    ok = r["event_f1"] >= 0.99 and r["segment_dice"] >= 0.9
    print("\n%s: overfit F1 %.3f, Dice %.3f" % ("PASS" if ok else "FAIL",
                                                r["event_f1"], r["segment_dice"]))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
