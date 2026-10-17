# AI Watermark

[![CI](https://github.com/s-gehring/ai-watermark-addon/actions/workflows/ci.yml/badge.svg)](https://github.com/s-gehring/ai-watermark-addon/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Firefox 140+](https://img.shields.io/badge/Firefox-140%2B-orange.svg)](https://www.mozilla.org/firefox/)

A Firefox add-on that checks images on web pages for signs of AI generation and marks hits with a small badge. Everything runs locally in the browser: no image leaves your computer, and there is no API and no server.

![Three images on a web page: a Gemini image marked "AI · SynthID" with the details panel open, a FLUX.2 image marked "AI 99 %" and a real photo with a green check mark](docs/images/screenshot.jpg)

<sub>Seal (Gemini 3.1 Flash Image) and swan (FLUX.2) from [bigdatamark/synthid-research](https://huggingface.co/datasets/bigdatamark/synthid-research), metadata removed. River photo from [DOCCI](https://huggingface.co/datasets/google/docci) by Google, CC BY 4.0.</sub>

## What it checks

Every image whose shorter side is at least 200 px goes through up to three checks. The first one that finds something decides the badge.

| Check | Looks for | Badge |
|---|---|---|
| **Metadata** | AI labels in C2PA manifests and IPTC/XMP, generation parameters from Stable Diffusion, ComfyUI, NovelAI, InvokeAI and the Stability AI API, the Chinese AIGC label (GB 45438-2025) | AI · Metadata |
| **SynthID watermark** | the invisible watermark that Google (Gemini, Imagen) and, since May 2026, OpenAI (ChatGPT, API) embed in generated images, measured in the pixels; also the watermark of Amazon Nova Canvas, the Stable Diffusion, SDXL and FLUX `invisible-watermark` in unmodified PNGs and the visible Gemini and Kling logos in the bottom right corner | AI · SynthID, AI · Watermark |
| **Model** | a ViT classifier ([Detectra fine-tuned on OpenFake](https://huggingface.co/s-gehring/detectra-openfake)) run with ONNX Runtime Web | AI 97 % |

There is no public SynthID detector. The add-on reconstructs a fixed part of the watermark from public sample images and compares image spectra against it. [docs/how-it-works.md](docs/how-it-works.md) explains all three checks.

## How well it works

- **Model:** at the default threshold it flags about 1 % of real photos and detects about 60 % of all AI images under web conditions (downscaled and JPEG-compressed): 71–78 % from OpenFake and AI subreddits, 49–55 % from the newest generators (FLUX.2, GPT-Image 2, Qwen-Image 2.1, Gemini 3.1). It flagged none of 156 museum paintings, 5–7 % of WikiArt paintings and no charts.
- **SynthID:** detects 95 % of Gemini 3.1 images as delivered and 75–94 % of ChatGPT images in the four supported output sizes. False positives are rare: the Google template triggered on 1 of about 1,700 real photos, the OpenAI templates on none of about 1,900 images without the watermark. Cropping destroys the signal and recompression weakens it. Amazon Nova Canvas images are found as unmodified PNGs (all 75 test images), rarely after downscaling or JPEG compression.
- **Visible Gemini logo:** finds 29 of 34 real images that still carry the logo (SynthID finds 3 of them), also after downscaling and recompression, and none of about 8,900 images without it. Logos on bright or busy backgrounds and cropped images are missed.
- **Visible Kling logo:** finds 98 % of 1,005 Kling images from 2024 and all 68 current free-tier images, also after downscaling and recompression, and none of about 10,500 images without it, among them 1,617 Kling images without the logo. Logos on white backgrounds, cropped images and the Chinese and versioned variants of the logo are missed.
- **Metadata:** reliable when present, but many platforms strip metadata on upload.

The badge is a hint, not proof. Full numbers and limitations are in [docs/accuracy.md](docs/accuracy.md).

## Install

Open the `.xpi` file of the [latest release](https://github.com/s-gehring/ai-watermark-addon/releases/latest) in Firefox and confirm the installation. If Firefox only downloads the file, open `about:addons`, click the gear icon and choose "Install Add-on From File…". The add-on is signed by Mozilla and updates itself, but is not listed on addons.mozilla.org.

On Firefox for Android, enable the debug menu (Settings → About Firefox, tap the logo five times), then use "Install extension from file" in the settings.

Requires Firefox 140 or later, or Firefox for Android 142 or later.

## Privacy and permissions

- All checks run locally in a Web Worker. The add-on sends no data to any third party and has no telemetry.
- To read the original bytes of images from other domains, the add-on fetches each image a second time from the same URL, usually served from the browser cache.
- `<all_urls>` is needed to find and fetch images on any page, `storage` to keep your settings.

## Build from source

Requires Node 22.

```bash
npm ci
npm run fetch-model   # downloads the model weights (44 MB) from Hugging Face and verifies the checksum
npm run build         # builds to dist/
npm start             # starts Firefox with the add-on
```

Tests, linting, releases, model evaluation and training, and rebuilding the SynthID templates are described in [docs/development.md](docs/development.md).

## Written with AI

Large parts of the code, tools and documentation in this repository were written by AI (Claude Code with Claude Opus 5.5).

## Contributing

Bug reports, false positives and missed AI images are welcome as [issues](https://github.com/s-gehring/ai-watermark-addon/issues). See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License

The code is licensed under the [MIT License](LICENSE).

The bundled model weights are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) because they are trained on OpenFake. **A built add-on may therefore only be used and redistributed non-commercially.** [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists all third-party components, their licenses and the required attributions.

This project is not affiliated with Google, OpenAI, Amazon, Kuaishou, Stability AI or the C2PA.

## Acknowledgements

- [Detectra](https://huggingface.co/TensorFold/detectra-v1) by TensorFold and [Community Forensics](https://huggingface.co/OwensLab/commfor-model-384) by Jeongsoo Park and Andrew Owens, the base models
- [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) by the Complex Data Lab, for fine-tuning and evaluation
- [bigdatamark/synthid-research](https://huggingface.co/datasets/bigdatamark/synthid-research), [winrisef/gpt-generated-image-collection](https://huggingface.co/datasets/winrisef/gpt-generated-image-collection) and [aoxo/reverse-synthid](https://huggingface.co/datasets/aoxo/reverse-synthid), the sources of the SynthID templates, and the GitHub repositories listed in [`tools/synthid/amazon_sources.json`](tools/synthid/amazon_sources.json) for the Nova Canvas template
- [GeminiWatermarkTool](https://github.com/allenk/GeminiWatermarkTool) by AllenK, the alpha map and positions of the visible Gemini logo
- [nyuuzyou/klingai](https://huggingface.co/datasets/nyuuzyou/klingai) by nyuuzyou, the Kling images from 2024 that the template of the older Kling logo is averaged from
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime), which runs the model in the browser
