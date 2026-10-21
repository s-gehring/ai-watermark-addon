"""Exports a fine-tuned checkpoint as 8-bit ONNX (MatMulNBits, block 16, asymmetric) with fp32 input and output, as the addon loads it."""
import argparse
import hashlib
import os
import tempfile
from pathlib import Path

os.environ["TIMM_FUSED_ATTN"] = "0"

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

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
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "model.fp32.onnx"
        torch.onnx.export(model, x, fp32, input_names=["pixel_values"], output_names=["logits"], opset_version=17,
                          dynamo=False, dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}})
        quantizer = MatMulNBitsQuantizer(onnx.load(fp32), bits=8, block_size=16, is_symmetric=False)
        quantizer.process()
        quantizer.model.save_model_to_file(str(args.target), use_external_data_format=False)

    expected = model(x).detach().numpy()
    got = ort.InferenceSession(args.target, providers=["CPUExecutionProvider"]).run(None, {"pixel_values": x.numpy()})[0]
    print(f"max deviation from PyTorch: {np.abs(got - expected).max():.4f}")
    print(f"sha256 {hashlib.sha256(args.target.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
