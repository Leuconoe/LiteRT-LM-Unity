"""Turn a FunASR training checkpoint into a self-contained SenseVoice model dir.

  python sv_ckpt_dir.py <train_out>/model.pt.ep2 <dir> [--base External/sensevoice-work/base/SenseVoiceSmall]

<dir> gets model.pt (weights only — the training file also carries optimizer state,
2.8 GB vs 0.94 GB), config.yaml with the training-time absolute paths and init_param
dropped, am.mvn, the bpe model and configuration.json from the base. That is the
layout sv_eval.py (funasr/onnx side files), the ONNX export and an HF release expect.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--base", default="External/sensevoice-work/base/SenseVoiceSmall")
    a = ap.parse_args()
    ckpt, out, base = Path(a.ckpt), Path(a.out), Path(a.base)
    out.mkdir(parents=True, exist_ok=True)

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    ref = torch.load(base / "model.pt", map_location="cpu", weights_only=False)
    ref = ref.get("state_dict", ref)
    assert state.keys() == ref.keys(), "checkpoint keys differ from the base model"
    torch.save(state, out / "model.pt")

    for name in ("am.mvn", "configuration.json", *[p.name for p in base.glob("*.bpe.model")]):
        shutil.copy2(base / name, out / name)
    conf = yaml.safe_load((base / "config.yaml").read_text(encoding="utf-8"))
    if "init_param" in conf:  # rewrite only when needed; otherwise ship the upstream file byte-identical
        conf.pop("init_param")
        (out / "config.yaml").write_text(yaml.safe_dump(conf, allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        shutil.copy2(base / "config.yaml", out / "config.yaml")

    changed = sum(not torch.equal(state[k], ref[k]) for k in state)
    print(f"{out}: {len(state)} tensors, {changed} differ from base")
    assert changed, "weights identical to the base — wrong checkpoint?"


if __name__ == "__main__":
    main()
