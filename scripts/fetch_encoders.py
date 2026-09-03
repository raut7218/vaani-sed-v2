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
# Mirrors that host the ATST-Frame weights. Tried in order.
ATST_CANDIDATES = [
    ("Audio-WestlakeU/ATST-SED", "atstframe_base.ckpt"),
    ("Audio-WestlakeU/ATST-SED", "atst_as2M.ckpt"),
]

ADAPTER = '''"""Adapter: expose one `build_atst_frame()` over the vendored upstream source.

`src/models/encoders.py` imports this. It is intentionally thin - all it does is
find the upstream ATST-Frame class wherever the repo layout happens to put it and
return an instance with `embed_dim` set, so the loader can verify how much of the
checkpoint actually landed.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for sub in ("", "src", "ATST-SED", "ATST-SED/train", "audiossl"):
    p = _HERE / sub
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _find_class():
    candidates = [
        ("atst.frame_atst", "FrameAST"),
        ("frame_atst", "FrameAST"),
        ("models.atst_frame", "ATSTFrame"),
        ("audiossl.models.atst.frame.atst", "FrameAST"),
    ]
    for mod_name, cls_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            return getattr(mod, cls_name)
        except Exception:
            continue
    raise ImportError(
        "Could not locate the ATST-Frame class inside third_party/atst. "
        "The upstream layout has changed - point _find_class() at the right "
        "module and class name."
    )


def build_atst_frame(**kw):
    cls = _find_class()
    model = cls(**kw)
    if not hasattr(model, "embed_dim"):
        model.embed_dim = getattr(model, "out_dim", 768)
    return model
'''


def run(cmd, **kw):
    print("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kw)


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
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[atst] pip install huggingface_hub, then re-run")
        return
    for repo, fname in ATST_CANDIDATES:
        try:
            p = hf_hub_download(repo_id=repo, filename=fname)
            shutil.copy(p, target)
            print("[atst] checkpoint -> %s" % target)
            return
        except Exception as e:                                        # noqa: BLE001
            print("[atst] %s/%s unavailable (%s)" % (repo, fname, e))
    print("[atst] Could not auto-download the ATST-Frame weights.\n"
          "       Download them from the ATST-SED release page and save as:\n"
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
