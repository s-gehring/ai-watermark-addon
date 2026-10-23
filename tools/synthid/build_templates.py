import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image

import synthid
from eval import TRANSFORMS

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).parent / ".cache"
META = ROOT / "synthid.json"
GOOGLE = [
    {
        "dataset": "aoxo/reverse-synthid",
        "revision": "7a4fbc2ba0277b0a73a9d6e9a1ebcfa36deb15bf",
        "folders": ["nano-banana-pro-preview/diverse", "gemini-3.1-flash-image-preview/diverse", "gemini_random"],
    },
    {
        "dataset": "bigdatamark/synthid-research",
        "revision": "af707a1b8823b277f260a1ebfbae3c9877d3d897",
        "folders": ["gemini-3.1-flash-image"],
    },
]
GOOGLE_SIZE = 512
GOOGLE_THRESHOLD = 10
ROBUST_BAND = 32
ROBUST_CONDITIONS = ("WebP 80", "web 1280 px, JPEG 80", "web 800 px, JPEG 75")
OPENAI = json.loads((Path(__file__).parent / "openai_sources.json").read_text())
OPENAI_TEMPLATES = [
    {"file": "openai-512.bin", "size": 512},
    {"file": "openai-1024.bin", "size": 1024, "prescreen": True},
]
OPENAI_CHANNELS = (0,)
OPENAI_MAX_FREQUENCY = 384
OPENAI_BORDER = 0.02
WATERMARK_ACTION = b"c2pa.watermarked"
AMAZON = json.loads((Path(__file__).parent / "amazon_sources.json").read_text())
AMAZON_SIZE = 1024
AMAZON_MIN_FREQUENCY = 128
AMAZON_BORDER = 0.02
AMAZON_MIN_CHANNEL_STD = 0.5
CR = 2
NOVA_MANIFEST = (b"Amazon_Bedrock", b"Nova Canvas")


def download(source, suffixes=(".png",)):
    local = CACHE / source["dataset"].split("/")[1]
    patterns = source.get("files") or [f"{folder}/*" for folder in source["folders"]]
    snapshot_download(
        source["dataset"],
        repo_type="dataset",
        revision=source["revision"],
        local_dir=local,
        allow_patterns=patterns,
        max_workers=8,
    )
    if "files" in source:
        return sorted(local / file for file in source["files"])
    return sorted(p for folder in source["folders"] for p in (local / folder).rglob("*") if p.suffix.lower() in suffixes)


def google_images():
    return {source["dataset"]: download(source, (".png", ".jpg")) for source in GOOGLE}


def bands(size):
    ky, kx = synthid.frequencies(size)
    n = size // 2 // ROBUST_BAND
    index = np.minimum(np.abs(ky) // ROBUST_BAND, n - 1) * n + np.minimum(kx // ROBUST_BAND, n - 1)
    return np.broadcast_to(index, (size, size // 2 + 1)).astype(int), n * n


def degraded_band_sums(args):
    path, weights, phases, size = args
    index, count = bands(size)
    signal, energy = np.zeros((3, count)), np.zeros((3, count))
    for name in ROBUST_CONDITIONS:
        rgb = TRANSFORMS[name](synthid.load_rgb(path))
        height, width, _ = rgb.shape
        w = weights * synthid.usable(height, width, size)[None]
        projection = np.real(synthid.unit_spectrum(rgb, size) * np.exp(-1j * phases))
        for c in range(3):
            signal[c] += np.bincount(index.ravel(), (w[c] * projection[c]).ravel(), count)
            energy[c] += np.bincount(index.ravel(), (w[c] ** 2).ravel(), count)
    return signal, energy


def survival(paths, size):
    index, count = bands(size)
    signal, energy = np.zeros((3, count)), np.zeros((3, count))
    paths = sorted(paths)
    halves = paths[0::2], paths[1::2]
    with Pool() as pool:
        for train, test in (halves, halves[::-1]):
            weights, phases, _ = synthid.build(train, size)
            for s, e in pool.imap(degraded_band_sums, [(p, weights, phases, size) for p in test]):
                signal += s
                energy += e
    rho = np.maximum(signal / np.maximum(energy, 1e-12), 0)
    return np.stack([rho[c][index] for c in range(3)])


def openai_images():
    return {
        source["dataset"]: [path for path in download(source) if WATERMARK_ACTION in path.read_bytes()]
        for source in OPENAI
    }


def amazon_images():
    groups, seen = {}, set()
    for source in AMAZON:
        repository, revision = source["repository"], source["revision"]
        for file in source["files"]:
            local = CACHE / "amazon-nova" / repository / revision / file
            if not local.exists():
                local.parent.mkdir(parents=True, exist_ok=True)
                url = f"https://raw.githubusercontent.com/{repository}/{revision}/{urllib.parse.quote(file)}"
                with urllib.request.urlopen(url, timeout=60) as response:
                    local.write_bytes(response.read())
            data = local.read_bytes()
            if not all(marker in data for marker in NOVA_MANIFEST):
                continue
            pixels = hashlib.sha1(synthid.load_rgb(local).tobytes()).hexdigest()
            if pixels not in seen:
                seen.add(pixels)
                groups.setdefault(repository, []).append(local)
    return groups


def build(entry, sources, paths_by_dataset, existing, weighting=None, **options):
    paths = [p for paths in paths_by_dataset.values() for p in paths]
    weights, phases, count = synthid.build(paths, entry["size"], **options)
    if weighting is not None:
        weights = weights * weighting(paths, entry["size"])
    w, p, scale = synthid.quantize(weights, phases)
    target = ROOT / synthid.TEMPLATE_DIR / entry["file"]
    synthid.save(target, w, p)
    datasets = [
        {**{k: v for k, v in s.items() if k != "files"}, "images": len(paths_by_dataset[s["dataset"]])}
        for s in sources
        if s.get("dataset") in paths_by_dataset
    ]
    repositories = [
        {"repository": s["repository"], "revision": s["revision"], "images": len(paths_by_dataset[s["repository"]])}
        for s in sources
        if s.get("repository") in paths_by_dataset
    ]
    entry.update(
        weightScale=scale,
        threshold=existing.get(entry["file"], {}).get("threshold", entry.pop("threshold")),
        source={**({"datasets": datasets} if datasets else {}), **({"repositories": repositories} if repositories else {}), "images": count},
    )
    print(f"{entry['file']}: {count} images -> {target.relative_to(ROOT)} ({target.stat().st_size // 1024} KB)", flush=True)
    return entry


def main():
    parser = argparse.ArgumentParser(description="Build the SynthID phase templates from pinned Hugging Face datasets.")
    parser.add_argument("providers", nargs="*", choices=["google", "openai", "amazon"], default=["google", "openai", "amazon"])
    args = parser.parse_args()

    existing = {t["file"]: t for t in json.loads(META.read_text())["templates"]} if META.exists() else {}
    templates = dict(existing)
    if "google" in args.providers:
        images = google_images()
        entry = {"provider": "google", "file": "google.bin", "size": GOOGLE_SIZE, "threshold": GOOGLE_THRESHOLD}
        templates[entry["file"]] = build(entry, GOOGLE, images, existing)
        entry = {"provider": "google", "file": "google-robust.bin", "size": GOOGLE_SIZE, "threshold": GOOGLE_THRESHOLD}
        templates[entry["file"]] = build(entry, GOOGLE, images, existing, weighting=survival)
    if "openai" in args.providers:
        templates = {file: t for file, t in templates.items() if t["provider"] != "openai"}
        paths_by_dataset = openai_images()
        for template in OPENAI_TEMPLATES:
            entry = {"provider": "openai", **template, "threshold": 8}
            templates[entry["file"]] = build(
                entry,
                OPENAI,
                paths_by_dataset,
                existing,
                max_frequency=OPENAI_MAX_FREQUENCY,
                border=OPENAI_BORDER,
                channels=OPENAI_CHANNELS,
                masked=True,
            )
    if "amazon" in args.providers:
        entry = {
            "provider": "amazon",
            "file": "amazon-nova-canvas.bin",
            "size": AMAZON_SIZE,
            "minChannelStd": AMAZON_MIN_CHANNEL_STD,
            "threshold": 8,
        }
        templates[entry["file"]] = build(
            entry, AMAZON, amazon_images(), existing, min_frequency=AMAZON_MIN_FREQUENCY, border=AMAZON_BORDER, channels=(CR,)
        )
    order = ["google.bin", "google-robust.bin", *(t["file"] for t in OPENAI_TEMPLATES), "amazon-nova-canvas.bin"]
    ordered = sorted(templates.values(), key=lambda t: order.index(t["file"]) if t["file"] in order else len(order))
    META.write_text(json.dumps({"templates": ordered}, indent=2) + "\n")


if __name__ == "__main__":
    main()
