"""Exports a fine-tuned checkpoint as fp16 ONNX with fp32 input and output, as the addon loads it."""
import argparse
import hashlib
import os
from pathlib import Path

# The fp16 converter produces mistyped Cast nodes from timm's fused attention.
os.environ["TIMM_FUSED_ATTN"] = "0"

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxconverter_common import float16

from train import create_model

DEFAULT_TARGET = Path(__file__).parents[2] / "vendor" / "models" / "detectra-openfake" / "model.onnx"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("target", type=Path, nargs="?", default=DEFAULT_TARGET)
    args = ap.parse_args()

    model = create_model(args.checkpoint).eval()
    x = torch.randn(2, 3, 384, 384)
    args.target.parent.mkdir(parents=True, exist_ok=True)
    fp32 = args.target.with_suffix(".fp32.onnx")
    torch.onnx.export(model, x, fp32, input_names=["pixel_values"], output_names=["logits"], opset_version=17,
                      dynamo=False, dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}})
    onnx.save(float16.convert_float_to_float16(onnx.load(fp32), keep_io_types=True), args.target)
    fp32.unlink()

    expected = model(x).detach().numpy()
    got = ort.InferenceSession(args.target, providers=["CPUExecutionProvider"]).run(None, {"pixel_values": x.numpy()})[0]
    print(f"max deviation from PyTorch: {np.abs(got - expected).max():.4f}")
    print(f"sha256 {hashlib.sha256(args.target.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
