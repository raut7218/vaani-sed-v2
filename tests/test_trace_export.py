"""A stripped span checkpoint reloads to exactly the model that was exported.

Needs the pretrained encoders under checkpoints/ (scripts/fetch_encoders.py --all).
"""
from pathlib import Path

import pytest
import torch
import yaml

from src.data.labels import LabelEncoder
from src.models.encoders import build_encoder
from src.models.span_model import build_model
from tracesed.predict import ENCODER_PREFIX, export_span_checkpoint, load_span_model

CK = Path("checkpoints")


@pytest.mark.skipif(not (CK / "atst_frame.ckpt").exists() or not (CK / "BEATs_iter3_plus_AS2M.pt").exists(),
                    reason="pretrained encoders not fetched")
def test_stripped_span_checkpoint_round_trips(tmp_path):
    cfg = yaml.safe_load(open("configs/default.yaml"))
    le = LabelEncoder(expand_vehicle=True)
    torch.manual_seed(0)
    model = build_model(cfg, len(le), build_encoder(cfg["model"], ckpt_dir=str(CK))).eval()
    full = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg, "which": "ema", "epoch": 3}, full)

    small = tmp_path / "span_best.pt"
    export_span_checkpoint(str(full), str(small))
    sd = torch.load(small, map_location="cpu", weights_only=False)
    assert sd["encoders_stripped"] and not any(k.startswith(ENCODER_PREFIX) for k in sd["model"])
    assert small.stat().st_size < 0.25 * full.stat().st_size

    loaded, _ = load_span_model(str(small), str(CK), torch.device("cpu"))
    wav = torch.randn(1, int(cfg["data"]["clip_len"] * cfg["data"]["sr"])) * 0.05
    fv = torch.ones(1, int(cfg["data"]["clip_len"] * cfg["data"]["fps"]))
    with torch.no_grad():
        a, b = model(wav, fv), loaded(wav, fv)
    for k in ("frame_logits", "count_logits", "onset_logits"):
        torch.testing.assert_close(a[k], b[k])
