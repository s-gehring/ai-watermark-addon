import json
from pathlib import Path

from huggingface_hub import snapshot_download

import synthid

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).parent / ".cache" / "reverse-synthid"
META = ROOT / "synthid.json"
TEMPLATE = ROOT / "src" / "background" / "synthid-template.bin"
DATASET = "aoxo/reverse-synthid"
REVISION = "7a4fbc2ba0277b0a73a9d6e9a1ebcfa36deb15bf"
FOLDERS = [
    "nano-banana-pro-preview/diverse",
    "gemini-3.1-flash-image-preview/diverse",
    "gemini_random",
]


def main():
    snapshot_download(
        DATASET,
        repo_type="dataset",
        revision=REVISION,
        local_dir=CACHE,
        allow_patterns=[f"{folder}/**" for folder in FOLDERS],
    )
    paths = sorted(p for folder in FOLDERS for p in (CACHE / folder).rglob("*.png"))
    weights, phases, count = synthid.build(paths)
    w, p, scale = synthid.quantize(weights, phases)
    synthid.save(TEMPLATE, w, p)

    meta = json.loads(META.read_text()) if META.exists() else {}
    meta.update(
        size=synthid.SIZE,
        weightScale=scale,
        source={"dataset": DATASET, "revision": REVISION, "folders": FOLDERS, "images": count},
    )
    meta.setdefault("threshold", 12)
    META.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{count} images -> {TEMPLATE.relative_to(ROOT)} ({TEMPLATE.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
