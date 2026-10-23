"""Zero-shot probability that a training image is art, illustration or graphics (CLIP ViT-B/16), keyed by the SHA-1 of the image bytes.
The model folder holds onnx/text_model.onnx, onnx/vision_model.onnx and tokenizer.json from Xenova/clip-vit-base-patch16."""
import argparse
import glob
import hashlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
from PIL import Image
from tokenizers import Tokenizer

from fetch_data import DEFAULT_DATA

Image.MAX_IMAGE_PIXELS = None
PHOTO = ["a photo", "a photograph taken with a phone camera", "a snapshot of a real scene", "a photo of a person", "a portrait photo",
         "a photo of food", "a landscape photo", "a product photo", "a studio photo of an object", "a photo of an animal",
         "a street photo", "a photo of a building interior"]
ART = ["a painting", "an oil painting", "a watercolor painting", "a drawing", "a pencil sketch", "an illustration", "a cartoon",
       "an anime image", "digital art", "concept art", "a 3D render", "a CGI image", "a vector graphic", "a logo", "a poster design",
       "an engraving", "a comic", "pixel art", "a fantasy artwork", "a chart or diagram", "a screenshot of a website"]
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def key(data):
    return hashlib.sha1(data).hexdigest()[:16]


def prep(data):
    image = Image.open(io.BytesIO(data))
    image.draft("RGB", (448, 448))
    image = image.convert("RGB")
    w, h = image.size
    s = 224 / min(w, h)
    image = image.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = image.size
    left, top = (w - 224) // 2, (h - 224) // 2
    a = np.asarray(image.crop((left, top, left + 224, top + 224)), np.float32) / 255
    return ((a - MEAN) / STD).transpose(2, 0, 1)


def load_group(job):
    path, group = job
    f = pq.ParquetFile(path)
    columns = ["image"] + (["type"] if "type" in f.schema_arrow.names else [])
    out = []
    for r in f.read_row_group(group, columns=columns).to_pylist():
        data = r["image"]["bytes"] if isinstance(r["image"], dict) else r["image"]
        if r.get("type") == "video" or not data:
            continue
        try:
            out.append((key(data), prep(data)))
        except Exception:
            pass
    return out


def text_features(clip):
    tok = Tokenizer.from_file(str(clip / "tokenizer.json"))
    session = ort.InferenceSession(str(clip / "onnx" / "text_model.onnx"), providers=["CPUExecutionProvider"])
    t = np.array([session.run(None, {"input_ids": np.array([tok.encode(p).ids], np.int64)})[0][0] for p in PHOTO + ART], np.float32)
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or args.data / "router.json"

    text = text_features(args.clip)
    vision = ort.InferenceSession(str(args.clip / "onnx" / "vision_model.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    files = sorted(p for p in glob.glob(str(args.data / "**" / "*.parquet"), recursive=True) if "/runs/" not in p)
    jobs = [(p, g) for p in files for g in range(pq.ParquetFile(p).metadata.num_row_groups)]
    result = {}

    def run(batch):
        v = vision.run(None, {"pixel_values": np.stack([a for _, a in batch])})[0]
        s = 100 * (v / np.linalg.norm(v, axis=1, keepdims=True)) @ text.T
        s = np.exp(s - s.max(1, keepdims=True))
        p = s[:, len(PHOTO):].sum(1) / s.sum(1)
        result.update({k: round(float(x), 4) for (k, _), x in zip(batch, p)})

    batch = []
    with ProcessPoolExecutor(12) as pool:
        for rows in pool.map(load_group, jobs):
            for row in rows:
                batch.append(row)
                if len(batch) == 128:
                    run(batch)
                    batch = []
        if batch:
            run(batch)
    out.write_text(json.dumps(result))
    print(f"{len(result)} images -> {out}")


if __name__ == "__main__":
    main()
