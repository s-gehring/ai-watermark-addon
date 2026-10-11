from pathlib import Path

from huggingface_hub import snapshot_download

OUT = Path(__file__).parent / ".cache" / "synthid-research"
DATASET = "bigdatamark/synthid-research"
REVISION = "af707a1b8823b277f260a1ebfbae3c9877d3d897"
FOLDERS = [
    "gpt-image-2",
    "gpt-image-2.5-flare",
    "gpt-image-2.5-sunburst",
    "gemini-3.1-flash-image",
    "flux.2",
    "qwen-image-2.1",
]


def main():
    snapshot_download(
        DATASET,
        repo_type="dataset",
        revision=REVISION,
        local_dir=OUT,
        allow_patterns=["metadata.csv", *(f"{folder}/*" for folder in FOLDERS)],
        max_workers=8,
    )
    for folder in FOLDERS:
        print(folder, len(list((OUT / folder).iterdir())))


if __name__ == "__main__":
    main()
