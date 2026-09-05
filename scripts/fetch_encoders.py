"""Vendor the pretrained encoders this repo can use.

BEATs downloads itself on first use. ATST-Frame does not: it needs its upstream
model definition as well as a checkpoint, and reimplementing a checkpointed
architecture from memory is how you end up training a half-random encoder that
looks fine in the loss curve. So we clone the upstream source and write a small
adapter that exposes one factory function.

    python scripts/fetch_encoders.py --atst
    python scripts/fetch_encoders.py --beats
    python scripts/fetch_encoders.py --all
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD = ROOT / "third_party"
CKPT = ROOT / "checkpoints"

ATST_REPO = "https://github.com/Audio-WestlakeU/ATST-SED.git"

# Where the ATST-Frame weights actually live. The authors publish them on Google
# Drive, not on the Hub - there is no `Audio-WestlakeU` HF org, so anything
# pointing there 401s. Community mirrors on the Hub are tried first because they
# need no extra dependency and resolve in seconds; the authors' own Drive link is
# the fallback for when a mirror disappears.
ATST_HF_CANDIDATES = [
    # ATST-SED stage-2 weights: carries the full ATST-Frame encoder under
    # `atst_frame.atst.*`, which `_atst_frame_state` unwraps.
    ("igonzf/ATST-SED", "model.pth"),
]
# atst_as2M.ckpt, linked from the ATST-SED README. Needs `gdown`.
ATST_GDRIVE_ID = "1_xb0_n3UNbUG_pH1vLHTviLfsaSfCzxz"

ADAPTER = '''"""Adapter: expose one `build_atst_frame()` over the vendored upstream source.

`src/models/encoders.py` imports this. It is intentionally thin - all it does is
find the upstream ATST-Frame factory wherever the repo layout happens to put it
and return an instance with `embed_dim` set, so the loader can verify how much of
the checkpoint actually landed.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for sub in ("", "src", "ATST-SED", "ATST-SED/train", "audiossl"):
    p = _HERE / sub
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _find_factory():
    # (module, attribute). The first entry is where ATST-SED actually keeps it;
    # the rest cover the audiossl layout and older checkouts.
    candidates = [
        ("desed_task.nnet.atst.audio_transformer", "FrameASTModel"),
        ("desed_task.nnet.atst.audio_transformer", "FrameAST"),
        ("audiossl.models.atst.frame.atst", "FrameAST"),
        ("atst.frame_atst", "FrameAST"),
        ("frame_atst", "FrameAST"),
    ]
    for mod_name, attr in candidates:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            return getattr(mod, attr)
        except Exception:
            continue
    raise ImportError(
        "Could not locate the ATST-Frame factory inside third_party/atst. "
        "The upstream layout has changed - point _find_factory() at the right "
        "module and attribute name."
    )


def build_atst_frame(**kw):
    # Dropout defaults to 0.1 upstream; this encoder starts out frozen and is
    # regularised by the head, so default it off and let the caller override.
    kw.setdefault("atst_dropout", 0.0)
    factory = _find_factory()
    try:
        model = factory(**kw)
    except TypeError:
        kw.pop("atst_dropout", None)
        model = factory(**kw)
    if not hasattr(model, "embed_dim"):
        model.embed_dim = getattr(model, "out_dim", 768)
    return model
'''


def run(cmd, **kw):
    print("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kw)


def _is_frame_ckpt(path: Path) -> bool:
    """Does this file actually contain ATST-Frame weights?

    The primary source is a third-party mirror, so verify rather than trust:
    a wrong-but-loadable checkpoint would otherwise only surface at train time,
    as a `min_load_frac` failure with no hint of where it came from.
    """
    try:
        import torch
        sys.path.insert(0, str(ROOT))
        from src.models.encoders import _atst_frame_state
        sd = _atst_frame_state(torch.load(str(path), map_location="cpu",
                                          weights_only=False))
    except Exception as e:                                            # noqa: BLE001
        print("[atst] could not inspect the download: %s" % e)
        return False
    need = ("pos_embed", "patch_embed.patch_embed.weight", "blocks.11.mlp.fc2.weight")
    have = [k for k in need if k in sd]
    if len(have) != len(need):
        print("[atst] download is not an ATST-Frame checkpoint (%d/%d marker "
              "tensors, %d keys)" % (len(have), len(need), len(sd)))
        return False
    return True


def _download_atst_ckpt(target: Path) -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None
        print("[atst] huggingface_hub not installed, skipping the Hub mirrors")

    tmp = target.with_suffix(".part")
    for repo, fname in ATST_HF_CANDIDATES if hf_hub_download else []:
        try:
            p = hf_hub_download(repo_id=repo, filename=fname)
            shutil.copy(p, tmp)
        except Exception as e:                                        # noqa: BLE001
            print("[atst] %s/%s unavailable (%s)" % (repo, fname, e))
            continue
        if _is_frame_ckpt(tmp):
            tmp.replace(target)
            print("[atst] checkpoint <- %s/%s -> %s" % (repo, fname, target))
            return True
        tmp.unlink(missing_ok=True)

    try:
        import gdown
    except ImportError:
        print("[atst] pip install gdown to try the authors' Drive link")
        return False
    try:
        print("[atst] trying the authors' Google Drive copy of atst_as2M.ckpt")
        gdown.download(id=ATST_GDRIVE_ID, output=str(tmp), quiet=False)
    except Exception as e:                                            # noqa: BLE001
        print("[atst] Drive download failed (%s)" % e)
        tmp.unlink(missing_ok=True)
        return False
    if tmp.exists() and _is_frame_ckpt(tmp):
        tmp.replace(target)
        print("[atst] checkpoint <- Google Drive -> %s" % target)
        return True
    tmp.unlink(missing_ok=True)
    return False


def fetch_atst() -> None:
    dest = THIRD / "atst"
    dest.mkdir(parents=True, exist_ok=True)
    src = dest / "ATST-SED"
    if not src.exists():
        try:
            run(["git", "clone", "--depth", "1", ATST_REPO, str(src)])
        except Exception as e:                                        # noqa: BLE001
            print("[atst] clone failed: %s" % e)
            print("[atst] clone it manually into %s and re-run." % src)
            return
    (dest / "atst_adapter.py").write_text(ADAPTER, encoding="utf-8")
    print("[atst] source at %s, adapter written" % src)

    CKPT.mkdir(parents=True, exist_ok=True)
    target = CKPT / "atst_frame.ckpt"
    if target.exists():
        print("[atst] checkpoint already at %s" % target)
        return
    if _download_atst_ckpt(target):
        return
    print("[atst] Could not auto-download the ATST-Frame weights.\n"
          "       Download atst_as2M.ckpt from the link in the ATST-SED README\n"
          "         https://github.com/Audio-WestlakeU/ATST-SED\n"
          "       and save it as:\n"
          "         %s\n"
          "       Training runs without it (BEATs-only), just less accurately."
          % target)


def fetch_beats() -> None:
    sys.path.insert(0, str(ROOT))
    from src.models.encoders import download_beats
    p = download_beats(CKPT)
    print("[beats] %s" % (p or "download failed"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atst", action="store_true")
    ap.add_argument("--beats", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all or a.beats:
        fetch_beats()
    if a.all or a.atst:
        fetch_atst()
    if not (a.all or a.atst or a.beats):
        ap.print_help()


if __name__ == "__main__":
    main()
