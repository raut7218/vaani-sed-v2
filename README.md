# Vaani Noise Event Detection — Track 1, v2

Sound event detection on Indic speech recordings: find the noise events and their
onset/offset timestamps.

**Competition:** [Datathon@IndoML 2026 — Track 1](https://www.codabench.org/competitions/17825/)
on Codabench. Scored on **Event F1 + Segment Dice** (max 2.0) over an 11 h
withheld test set; submit a ZIP holding one `predictions.jsonl`.

> **This is a rewrite of [`vaani-sed-track1`](https://github.com/raut7218/vaani-sed-track1),
> not a tuning pass.** That system scored 0.89 (rank 27). A constant heuristic —
> *"the whole clip is one event"* — scores **0.919** on the same validation data.
> The v1 stack was worth less than one line of code, and the reason turned out to
> be structural rather than a matter of hyperparameters.

---

## The diagnosis, in five measurements

All measured on the v1 checkpoint over 8,187 held-out clips.

| # | Measurement | Value | What it means |
|---|---|---|---|
| 1 | Constant `[[0, clip_len]]` baseline | **0.919** vs v1's 0.89 LB | The model added nothing. |
| 2 | Loose recall (>50% overlap) vs **strict** recall | **0.877** vs **0.343** | It *found* 88% of events and *localised* 34%. A boundary problem, not a detection problem. |
| 3 | Posterior 10–90% rise time at a true onset | **760 ms** | Boundaries needing 50 ms precision were represented by a three-quarter-second ramp. |
| 4 | Optimal per-clip threshold: mean / std | **0.357 / 0.290**, near-uniform on [0.05, 0.95] | Slope 0.54/s ⇒ 50 ms precision needs the threshold correct to **±0.027**. The observed spread is **11× wider**. No fixed threshold can work. |
| 5 | Submitted coverage vs reference coverage | **0.29** vs **0.52** | The operating point was badly miscalibrated on unseen states — worth ~0.12–0.15 alone. |

And two ceilings that set the strategy:

* An oracle picking the best **per-clip threshold** on v1's own posteriors scores **1.414**
  (the leaderboard leader is at 1.52). The information was already in the features;
  the decoder could not get it out.
* A perfect model quantised to the 25 fps grid scores **1.988**. Frame rate is *not*
  the bottleneck — v1's README suggested raising `fps` to 50, which is worth **+0.006**.
  *(Refined by the v2 measurements below: this is true of what a distribution can
  **represent**, and false of what it can be trained to **resolve**. DFL resolution
  scales with bin width, and bin width is the pyramid level's stride — which the
  level-assignment table, not `fps`, decides.)*

Reproduce any of these on your own checkpoint with `scripts/diagnose.py`.

---

## What v2 measured, and what it ruled out

The v2 design above was built from the v1 diagnosis. Running it produced a
checkpoint at **1.1881** (fold 0) and a second round of measurements, several of
which contradict assumptions in this file. They are recorded here so the dead
ends are not re-explored.

**The head is not the limit.** Overfit deliberately to a handful of clips, the
span head places boundaries to **4.2 ms** against a 180 ms tolerance
(`tests/test_overfit.py`). Sub-tolerance boundaries are representable and
learnable; the 0.19 s the trained model carries is a *generalisation* gap, not a
ceiling of the architecture. That is worth knowing before redesigning the head -
one such redesign (the refinement branch below) was built and did not pay.

**The score is bounded by boundary jitter, and the amount is known.** Push the
real references through the real scorer with synthetic noise on every boundary:

| jitter σ | 0.05 s | 0.10 s | 0.15 s | 0.20 s |
|---|---|---|---|---|
| score | 1.777 | 1.475 | 1.295 | **1.171** |

The checkpoint scores 1.188. It behaves as a near-perfect detector carrying
~0.19 s of jitter, and 1.6 would need that under 0.08 s.

**Gold and silver are different annotation regimes.** Not the same process held
to different standards — different processes:

| | median event | coverage | starts at 0.000 s | spans whole clip |
|---|---|---|---|---|
| gold | 0.45 s | 0.287 | 5.4% | 0.9% |
| silver | 0.98 s | 0.585 | **29.6%** | **11.7%** |

Timestamps themselves are *not* quantised — 99.7% land off any 10 ms grid — so
there is no hard label-noise ceiling. But training at a 50% gold quota while
validating on a split that is 87% silver is a distribution mismatch, and a
boundary head trained on silver's clip-edge "events" learns to fire on silence.

**Ruled out by measurement:**

| Tried | Result |
|---|---|
| Tuning the decoder (130 configs) | **+0.013**. The selection oracle is real but no *fixed rule* reaches it. |
| A 20 ms onset/offset refinement branch | **Costs score**, monotonically in how much you apply it. |
| Boundary agreement as a ranking term | A wash (1.1723 → 1.1704). |

The branch failure is instructive rather than embarrassing. Probed directly, it
*had* learned: its peak sits a median 0.072 s from a true onset and 0.044 s from
an offset, against the 0.125 s a random peak in the same window would give. It is
simply outclassed — the regression head's median error on those same endpoints is
**+0.010 s**. Refinement was replacing a 10 ms error with a 50 ms one, and the
errors that actually cost score (p10 −2.31 s, p90 +2.16 s) sit outside any window
the metric's tolerance permits it to search.

**Where the score actually goes.** Over one fixed candidate pool:

| | score |
|---|---|
| as decoded | 1.1753 |
| oracle **count**, own ranking | 1.2123 (+0.037) |
| oracle **subset** | **1.4569 (+0.28)** |

Choosing the right *number* of spans is worth almost nothing beside choosing the
right *ones*. The candidates are good; the ranking cannot find them. That is what
Quality Focal Loss addresses — a separate IoU head cannot, because it only ever
sees positives and is never shown a bad span to calibrate against.

---

## What changed

| | v1 | v2 | Why |
|---|---|---|---|
| **Primary encoder** | frozen BEATs, **160 ms** patches | **ATST-Frame, 40 ms**, fine-tuned; BEATs demoted to a semantic side-channel | The metric's 10th-percentile tolerance is **59 ms**. BEATs is ~3× too coarse, and v1 then linearly interpolated those tokens up to a 40 ms grid. BEATs is good at *what*; ATST is good at *when*. |
| **Encoder training** | frozen forever | staged: frozen → top-4 blocks at 5% LR | v1 justified freezing by "22 h gold, Colab time". The corpus is 154 h and Kaggle gives 2×T4. Staged fine-tuning is the ATST-SED recipe. |
| **Output** | per-frame posterior | **anchor-free span regression** with distributional boundaries | Measurement 4. Boundaries are *regressed*, never read off a level set. |
| **Boundary representation** | implicit in the threshold | 16-bin distribution per boundary, expectation = the value | Continuous output on a discrete grid. Verified: the head places boundaries to **5.2 ms** on deliberately off-grid targets (`tests/test_overfit.py`). |
| **Post-processing** | cSEBBs, 8 classes × 4 tuned params, unioned | SoftNMS + a learned **event-count head** | The only thing left to decide is *which* spans and *how many* — never *where*. |
| **Losses** | frame BCE + clip BCE + consistency | focal + **1D DIoU** + **DFL** + **soft-Dice on the class-agnostic mask** | The leaderboard is F1 + Dice. Soft-Dice is literally half the metric, differentiably. |
| **Mean teacher** | `lambda_cons: 2.0` | **removed**; EMA of *weights* only | Consistency is a smoothness prior and smoothness is the enemy here. v1's own history shows student ≈ teacher (Δ<0.005) from epoch 11 — past that it was contributing blur and nothing else. |
| **MixStyle** | `p: 0.5` | removed; per-clip normalisation + per-district calibration | Same blur objection, and there are better domain-shift levers. |
| **Silver tier** | 0.5 weight on the whole frame loss | full weight on tags, **0.25 on boundaries** | Silver's *tags* are as trustworthy as gold's. Only its *timestamps* are unverified. Weighting the whole loss confuses those two things. *(The v2 label statistics below show this premise is only half right: silver is a different annotation **regime**, not the same one held to a lower standard.)* |
| **Bronze tier** | attention pooling only | pooling **+ `make_synthetic.py`** | Cut a segment from a clip tagged `animal_sound`, paste it at a known position, and you have a strong label that is correct *by construction*. No confidence threshold, so no error to accumulate. |
| **Validation** | one 5-state holdout | **state-grouped k-fold** | v1 tuned 8 classes × 4 params against one narrow slice. Val→LB dropped 0.16. |
| **Test-time** | none | **transductive per-district calibration** | Filenames encode `State_District`; the test set spans ~150 of them. Uses only unlabelled test audio. |
| **Window** | 10 s (~40% padding) | 8 s | Corpus mean is 5.7 s. |
| **Speech** | not modelled | **auxiliary VAD head** | This is Project Vaani — *speech* recordings. Every DCASE-derived system treats it as a generic soundscape. |
| **Ensembling** | posterior averaging | **1D weighted box fusion** | Averaging two posteriors that localise an onset 80 ms apart widens the ramp by 80 ms. Fuse spans, never posteriors. |

---

## Quick start

```bash
git clone https://github.com/raut7218/vaani-sed-v2.git && cd vaani-sed-v2
pip install -r requirements.txt

python scripts/smoke_test.py          # end-to-end on synthetic audio, ~2 min, no GPU
python tests/test_components.py       # 67 checks
python tests/test_overfit.py          # proves the head is time-aligned to the audio
```

Then the real thing:

```bash
python scripts/fetch_encoders.py --all                    # BEATs + ATST-Frame

python scripts/download_data.py --out data/vaani          # 182 shards, 154.6 h, gated
python scripts/make_vad.py --data data/vaani              # speech pseudo-labels
python scripts/make_synthetic.py --data data/vaani --out data/vaani_synth -n 20000

python -m src.train.train --config configs/default.yaml \
    --data data/vaani --extra-data data/vaani_synth --out runs/f0 --fold 0

python scripts/diagnose.py --ckpt runs/f0/best.pt --data data/vaani --fold 0
python -m src.infer.predict --ckpt runs/f0/best.pt --audio-dir data/test --out submission.zip
```

### On Kaggle

Two notebooks, both on **GPU T4 x2** with Internet on:

1. [`notebooks/Vaani_Data_Kaggle.ipynb`](notebooks/Vaani_Data_Kaggle.ipynb) — **once**.
   Downloads the corpus (gated: paste an `HF_TOKEN` or attach the secret), runs VAD and
   synthesis, and packs everything with `scripts/pack_data.py` into ~10 large files.
   Kaggle saves at most 500 output files and the prepared corpus is ~110k, so without
   packing every session spent its first hour re-downloading 16.5 GB.
2. [`notebooks/Vaani_Track1_v2_Kaggle.ipynb`](notebooks/Vaani_Track1_v2_Kaggle.ipynb) —
   training and submission. *Add Data → Your Work → vaani-data*; the CONFIG cell finds
   the packed corpus on its own.

`--batch-size` is **per GPU**, the standard DDP convention:

```bash
torchrun --standalone --nproc_per_node=2 -m src.train.train \
    --config configs/default.yaml --data data/vaani --out runs/f0 --batch-size 48
```

Training writes `state.pt` every epoch and resumes losslessly if a session runs out:

```bash
torchrun --standalone --nproc_per_node=2 -m src.train.train \
    --config configs/default.yaml --data data/vaani --out runs/f0 \
    --batch-size 48 --resume auto
```

`--resume auto` restores the model, optimiser, GradScaler, EMA shadow, history and —
critically — the **step counter**. Without the step counter the cosine schedule restarts
and the learning rate jumps back up, which quietly undoes the previous session's
progress. `state.pt` is written to a temp name and renamed, so a session killed
mid-write leaves the previous state intact rather than a truncated file.

### The dataset

**[ARTPARK-IISc/Vaani-Noise-Event-Dataset](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset)**
— 182 shards, 16.5 GB, 90,637 clips / ~154.6 h, and **gated** (accept the terms,
then expose a token as `HF_TOKEN`). `PavanKumarJ-ARTPARK/Vaani_Noise_Event_TimeStamp`
is a 9-clip *sample*; only the full repo trains anything.

### The ATST-Frame checkpoint

`fetch_encoders.py --atst` clones the upstream
[ATST-SED](https://github.com/Audio-WestlakeU/ATST-SED) source and pulls the weights.
The authors publish them on Google Drive, not on the Hub, so the script tries a Hub
mirror of the ATST-SED stage-2 checkpoint first (the ATST-Frame encoder sits inside it
under `atst_frame.atst.*`) and falls back to the authors' own `atst_as2M.ckpt` via
`gdown`. Both layouts, and the C2F one, are normalised to bare `FrameAST` keys by
`_atst_frame_state`, and the download is checked for ATST-Frame marker tensors before
it is kept. If both routes fail, download `atst_as2M.ckpt` manually to
`checkpoints/atst_frame.ckpt`.

ATST-Frame consumes **its own** mel — 64 bands, 60–7800 Hz, 10 ms hop, dB, min-max
scaled to [-1, 1] — not a waveform and not our 128-band mel; `ATSTFrameEncoder.features`
reproduces the upstream transform exactly, and `tests/test_components.py` asserts a
burst at a known time lands on the right token whenever the checkpoint is present.

The loader **refuses to run** if under 90% of the checkpoint's tensors land in the
model — a half-loaded encoder trains a partly-random network and still produces a
perfectly plausible loss curve, which is the single most expensive silent failure
available here. Everything still trains without it (BEATs-only, or mel-only), just
less accurately.

---

## Throughput on 2×T4

Measured on Kaggle, batch 48 per GPU, the real packed corpus (607 steps per epoch per
GPU):

| | Before | Now |
|---|---|---|
| Frozen epoch | ~12.5 min | **~9 min** |
| Unfrozen epoch | ~32 min | **~13.7 min** |
| One validation pass (14.5k clips) | ~10 min, every epoch | **~1.8 min**, every 5th epoch |
| Setup per session (download, VAD, synthesis) | ~1 h | **0** (packed once, attached) |
| 50 epochs end to end | ~30 h | **~11 h**, inside one 12 h session |

Where it came from, in order of size:

* **BEATs' shared position embedding.** BEATs ties one relative-position embedding
  across all 12 layers. Unfreezing the top blocks made it trainable, so every frozen
  layer's attention depended on a trainable tensor and backward ran through the whole
  stack. It stays frozen now; the frozen stack builds no autograd graph at all.
* **No whole-encoder activation checkpoint.** The frozen blocks were being recomputed
  every step for nothing; the top blocks now keep their activations (10–12 GB/GPU).
* **Validation on both GPUs, EMA weights only, every 5th epoch**, with CPU
  post-processing pipelined behind the next batch. It used to run twice (EMA + raw)
  over the whole held-out set on rank 0 every epoch while rank 1 waited.
* **BEATs' positional conv** (k=128, 16 groups) ran on cuDNN's direct grouped kernel —
  a quarter of a whole step. It is an FFT correlation now.
* **Fused attention** (`scaled_dot_product_attention`) in both encoders; BEATs' bias
  built in fp16; BEATs' GELU in fp16 instead of an fp32 round trip; batched Kaldi
  fbank instead of a per-clip loop; channels-last mel CNN.
* EMA over trainable tensors only, gain/noise augmentation on the GPU, fp16 gradient
  all-reduce, checkpoints written from a background thread, frozen layers in eval mode.

Every numerical change is tested against upstream (`tests/test_components.py`):
encoder outputs to ~1e-6, gradients to ~1e-7. `--time-limit-h` stops training after
the last epoch that fits and leaves `state.pt` for `--resume auto`, because a Kaggle
commit that hits the 12 h wall keeps no output.

Two fixes rode along. BEATs' LayerDrop (0.05 in the checkpoint's config) is off: it
draws per rank, so once the top blocks train it skips a *different* trainable layer on
each GPU, which DDP cannot reduce. And BEATs' 392 tokens are 49 time steps × 8
frequency patches, time-major — they were being linearly interpolated to 200 frames as
if the whole sequence were time, so consecutive output frames each saw a different
frequency band. The patches are now averaged per time step first.

---

## Architecture

```
waveform 16 kHz
  ├── log-mel @100 fps  +  spectral-flux onset channels ──► CNN ──► 25 fps, 512-d
  │      (time pooled by exactly 4, and only in the first two blocks)
  └── ATST-Frame (40 ms) ‖ BEATs (160 ms, freq-pooled) ──► proj 256 each
                                    │
                              concat, 25 fps
                                    │
                       TemporalFPN — 5 levels, 40…640 ms
                                    │
        ┌───────────────────────────┴────────────────────────────┐
   TridentHead                                          auxiliary heads
   • actionness (C + 1 agnostic)                        • frame class logits
   • P(start bins), P(end bins) ─► E[·] = continuous     • class-agnostic frame
   • quality / centerness                               • speech presence
                                                        • clip tags (attn pool)
                                                        • event count
                                    │
                     SoftNMS ─► count-head selection ─► spans
```

Events are assigned to exactly one pyramid level by length (0.32 s / 0.64 s /
1.28 s / 2.56 s boundaries). Assigning one span to several levels produces
duplicate detections, and under 1-to-1 matching a duplicate is a pure false
positive — the overfit test caught exactly that and it is why `count_slack`
defaults to 0.

---

## Why each stage

| Stage | Why |
|---|---|
| **ATST-Frame primary** | 40 ms tokens against a 59 ms p10 tolerance. The single largest structural fix. |
| **Trident boundary distributions** | Sub-frame boundaries on a coarse grid; removes the threshold entirely. Measured: 5.2 ms error on off-grid targets. |
| **1D DIoU** | Keeps a gradient when prediction and target do not overlap — plain IoU is flat at zero there, and half the events are under a second. |
| **Soft-Dice on the agnostic channel** | Half the leaderboard score, differentiable, applied directly. |
| **Spectral-flux channels** | Classical onset detectors were engineered for exactly this precision. Two subtractions. |
| **Count head** | 83% of clips hold exactly one event. That prior is strong, free, and replaces a tuned score floor. |
| **Speech head** | Vaani is speech. Factoring speech out makes noise boundaries far easier to place. Pseudo-labels are free. |
| **Synthetic bronze splicing** | Converts 32 h of tag-only audio into strong labels that are correct by construction. |
| **Per-district calibration** | ~150 districts in the test set, grouping key sitting in the filename, uses no labels. |
| **1D WBF ensembling** | Preserves boundary sharpness that posterior averaging destroys. |

---

## Roadmap

| # | Step | Status | Expected |
|---|---|---|---|
| 1 | Span regression + trident head + count selection | ✅ | — |
| 2 | ATST-Frame primary, staged unfreeze | ✅ (needs the checkpoint) | ~1.15–1.25 |
| 3 | Synthetic bronze + speech head | ✅ | ~1.25–1.35 |
| 4 | 5-fold state CV, select on the mean | ✅ `--fold` | ~1.35–1.45 |
| 5 | Seed/fold ensemble via 1D WBF + per-district calibration | ✅ | **1.50–1.60** |
| 6 | Boundary-contrastive pretraining on the unlabelled hours | 🔜 | — |

The confidence in steps 4–5 comes from measurement: an oracle over v1's *already
blurred* posteriors reached 1.414. A model that regresses boundaries starts from
sharper features and needs no oracle.

---

## Ablations

Each is one flag or one config line.

| Change | Tests |
|---|---|
| `--no-encoders` | what the pretrained stack is worth at all |
| `model.encoders: [beats]` | v1's encoder choice, everything else v2 |
| `model.encoders: [atst_frame]` | ATST alone vs the fusion |
| `model.flux: false` | value of the spectral-flux channels |
| `loss.dice: 0` | training the metric vs training BCE |
| `loss.dfl: 0` | distributional boundaries vs scalar regression |
| `loss.speech: 0` | value of the speech head |
| `postproc.count_weight: 0` | count head vs a plain score floor |
| `postproc.merge_gap: 0.12` | v1-style span merging (expect it to hurt) |
| `--no-calibrate` | value of per-district calibration |
| `--no-tta` | value of shift TTA |

---

## Metrics

`src/evaluation/metrics.py` is carried over from v1 unchanged — it was a faithful
reimplementation of the published scorer and it is the reason the diagnosis above
was possible.

| | |
|---|---|
| **Event F1** | Greedy closest-first 1-to-1 matching; a prediction matches when onset *and* offset are within `max(0.20 × ref_duration, 0.05)` s. Micro-averaged. |
| **Segment Dice** | Events rasterised to a 10 ms grid with an inclusive offset frame; `2·\|P ∩ G\| / (\|P\| + \|G\|)`, **macro**-averaged. Empty-vs-empty scores 1.0. |
| **Combined** | `F1 + Dice`, max **2.0**. A run scoring 1.42 is not 142%. |

Three details that each cost real score: the tolerance floor is **50 ms, not zero**;
Dice is **macro**, so a short clip counts as much as a long one; and the score is a
**sum, not a mean**.

---

## Submission format

`predict.py` writes a ZIP holding a single `predictions.jsonl` at its **root**:

```jsonl
{"clip_id": "vaani_eval_001", "events": [{"onset": 1.24, "offset": 3.81}]}
{"clip_id": "vaani_eval_002", "events": []}
```

Enforced for you: every evaluation clip appears exactly once (a missing line is not
the same as an empty one); times are millisecond-rounded, non-negative and
non-decreasing (the scorer does `int(onset / 0.01)`, so a negative onset indexes the
mask from the wrong end); clips outside the evaluation set are never emitted, since
their events would count as false positives. `scripts/smoke_test.py` validates a
real archive against every one of these.

---

## Layout

```
configs/default.yaml          all hyperparameters, annotated with the reasoning
scripts/fetch_encoders.py     vendor BEATs + ATST-Frame
scripts/download_data.py      HF -> wavs + manifest + tier assignment
scripts/make_vad.py           speech pseudo-labels for the auxiliary head
scripts/make_synthetic.py     bronze tags -> strong labels by construction
scripts/pack_data.py          corpus + VAD + synthetic -> a few large files (Kaggle's 500-file cap)
scripts/diagnose.py           where the score is going, not just what it is
scripts/smoke_test.py         end-to-end on synthetic audio
src/models/encoders.py        ATST-Frame / BEATs (fused attention) + fusion
src/models/frontend.py        log-mel, per-clip norm, spectral flux
src/models/trident.py         temporal FPN + distributional boundary head
src/models/span_model.py      the assembled model and its auxiliary heads
src/train/losses.py           assignment, focal, 1D DIoU, DFL, soft-Dice, tier masks
src/train/train.py            training loop (DDP, AMP, staged unfreeze, sharded validation)
src/infer/decode.py           SoftNMS, 1D WBF, count selection
src/infer/predict.py          -> submission.zip
src/postproc/calibrate.py     transductive per-district calibration
src/evaluation/metrics.py     event F1 + segment Dice (unchanged from v1)
third_party/beats/            BEATs, vendored from microsoft/unilm (MIT)
```

## Credits

BEATs © Microsoft, MIT-licensed, vendored in `third_party/beats/`.
ATST / ATST-SED © Audio-WestlakeU, fetched at setup.
Dataset: [Vaani Noise Event Timestamps](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset)
(CC-BY-4.0), derived from Project Vaani (IISc Bangalore / ARTPARK).

Method references: TriDet (relative boundary modelling), ActionFormer (anchor-free
temporal localisation), Generalized Focal Loss (distribution focal loss), ATST-SED
(frame-level pretraining and the staged fine-tuning schedule), and the
complementary-SSL-fusion result that puts ATST-Frame and BEATs in the same model.
