import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image

import synthid

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).parent / ".cache"
META = ROOT / "synthid.json"
GOOGLE = [
    {
        "dataset": "aoxo/reverse-synthid",
        "revision": "7a4fbc2ba0277b0a73a9d6e9a1ebcfa36deb15bf",
        "folders": ["nano-banana-pro-preview/diverse", "gemini-3.1-flash-image-preview/diverse", "gemini_random"],
    }
]
OPENAI = [
    {
        "dataset": "bigdatamark/synthid-research",
        "revision": "af707a1b8823b277f260a1ebfbae3c9877d3d897",
        "folders": ["gpt-image-2", "gpt-image-2.5-flare", "gpt-image-2.5-sunburst"],
    },
    {
        "dataset": "winrisef/gpt-generated-image-collection",
        "revision": "b0837d79e6bdd203ef7f4f3af819a8f0105f0adf",
        "folders": ["images/batch-0001", "images/batch-0002", "images/batch-0003"],
    },
]
OPENAI_SIZES = {(1254, 1254): 512, (1448, 1086): 1024, (1672, 941): 1024, (1122, 1402): 1024}
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


def download(source):
    local = CACHE / source["dataset"].split("/")[1]
    snapshot_download(
        source["dataset"],
        repo_type="dataset",
        revision=source["revision"],
        local_dir=local,
        allow_patterns=[f"{folder}/*" for folder in source["folders"]],
        max_workers=8,
    )
    return sorted(p for folder in source["folders"] for p in (local / folder).rglob("*.png"))


def google_images():
    return [path for source in GOOGLE for path in download(source)]


def openai_images():
    groups = {}
    for source in OPENAI:
        for path in download(source):
            with Image.open(path) as image:
                size = image.size
            if size in OPENAI_SIZES and WATERMARK_ACTION in path.read_bytes():
                groups.setdefault(size, {}).setdefault(source["dataset"], []).append(path)
    return groups


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


def build(entry, sources, paths_by_dataset, existing, **options):
    paths = [p for paths in paths_by_dataset.values() for p in paths]
    weights, phases, count = synthid.build(paths, entry["size"], **options)
    w, p, scale = synthid.quantize(weights, phases)
    target = ROOT / synthid.TEMPLATE_DIR / entry["file"]
    synthid.save(target, w, p)
    datasets = [{**s, "images": len(paths_by_dataset[s["dataset"]])} for s in sources if s.get("dataset") in paths_by_dataset]
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
        entry = {"provider": "google", "file": "google.bin", "size": 512, "threshold": 12}
        templates[entry["file"]] = build(entry, GOOGLE, {GOOGLE[0]["dataset"]: google_images()}, existing)
    if "openai" in args.providers:
        for (width, height), paths_by_dataset in openai_images().items():
            entry = {
                "provider": "openai",
                "file": f"openai-{width}x{height}.bin",
                "size": OPENAI_SIZES[(width, height)],
                "outputSize": [width, height],
                "threshold": 8,
            }
            templates[entry["file"]] = build(
                entry, OPENAI, paths_by_dataset, existing, max_frequency=OPENAI_MAX_FREQUENCY, border=OPENAI_BORDER
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
    order = ["google.bin", *(f"openai-{w}x{h}.bin" for w, h in OPENAI_SIZES), "amazon-nova-canvas.bin"]
    ordered = sorted(templates.values(), key=lambda t: order.index(t["file"]) if t["file"] in order else len(order))
    META.write_text(json.dumps({"templates": ordered}, indent=2) + "\n")


if __name__ == "__main__":
    main()
