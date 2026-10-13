# How it works

## Which images are checked

- `<img>` elements and CSS background images whose shorter rendered side is at least 200 px (configurable)
- images loaded later as well (lazy loading, infinite scroll, `srcset` changes); an image is checked only once it gets close to the visible area
- SVGs are skipped

## Pipeline

The background script fetches each image again (usually from the HTTP cache), so images from other domains can be checked without a CORS header. Metadata scan, decoding, the SynthID check and inference run in a Web Worker to keep the popup and background script responsive. Results are cached per URL. Images are checked one after another, visible ones first; one image takes about 0.8 s in Firefox.

Each image goes through up to three checks. The first one that finds something decides the badge.

## 1. Metadata

Badge "AI · Metadata". A metadata hit is treated as certain. Detected labels:

- IPTC Digital Source Type `trainedAlgorithmicMedia` or `compositeWithTrainedAlgorithmicMedia` in XMP or in C2PA manifests (OpenAI, Google, Adobe Firefly, Midjourney and others)
- Stable Diffusion generation parameters (A1111/Forge), ComfyUI workflows, NovelAI, InvokeAI
- metadata from the Stability AI API: the EXIF fields `Model: stability-diffusion` and `Software: stability.ai` (SDXL on Amazon Bedrock, v1 API) and the C2PA manifest of Stability's Content Credentials service, which carries no IPTC source type
- the AIGC label required by the Chinese labeling standard GB 45438-2025 (JSON with `Label`, `ContentProducer`, `ProduceID` and so on, as PNG text chunk `AIGC`, in the XMP namespace `http://www.tc260.org.cn/ns/AIGC/1.0/` or in the EXIF comment; mandatory for providers in China since September 2025, for example Kling)

If a C2PA manifest declares the SynthID watermark (Google writes "Applied imperceptible SynthID watermark"), the badge reads "AI · SynthID".

Many platforms strip metadata on upload. Then only the next two checks remain.

## 2. SynthID watermark

Badge "AI · SynthID". Google embeds the invisible [SynthID](https://deepmind.google/models/synthid/) watermark in images from Gemini and Imagen, and since May 2026 OpenAI does the same for all images from ChatGPT and its API. There is no public detector.

The add-on therefore checks for a fixed component of the watermark that can be measured in watermarked images: at certain spatial frequencies, counted in cycles per image width and height, the spectrum of every watermarked image has the same phase, regardless of the image content. The image is scaled to 512 × 512 or 1024 × 1024, each YCbCr channel is transformed with an FFT, and the phases are compared with templates averaged over watermarked images ([`src/background/synthid-templates/`](../src/background/synthid-templates/), [`synthid.json`](../synthid.json)). The result is a z-score per template, roughly standard normal for images without the watermark. From 12 (Google) or 8 (OpenAI) on, the watermark counts as found, the model is skipped, and the details panel names the provider.

- **Google:** one template from 288 Gemini images. Because frequencies are counted relative to the image size, this works at any resolution and aspect ratio.
- **OpenAI:** here the fixed component depends on the output size. Images of the same size from different sources share it, images of different sizes do not. The add-on has templates for the four output sizes with enough images carrying a verified watermark: 1254 × 1254, 1448 × 1086, 1672 × 941 and 1122 × 1402 (ChatGPT picks them for square, 4:3, 16:9 and 4:5 images). Each template is applied only to images with its aspect ratio (±2 %), which includes downscaled copies. The pattern sits at high frequencies, so the three non-square templates work at 1024 × 1024 (up to 384 cycles per image). All OpenAI templates also mask a 2 % border, because otherwise edge effects instead of the watermark would end up in the template.

Measured in Node, the SynthID check takes about 0.13 s for a 4-megapixel image, about 0.35 s at 4:3, 16:9 and 4:5 because of the finer OpenAI templates.

[`tools/synthid/synthid.py`](../tools/synthid/synthid.py) is the reference implementation of [`src/background/synthid.ts`](../src/background/synthid.ts); a unit test checks that both produce the same z-scores. Detection rates and false positives are in [accuracy.md](accuracy.md#synthid-watermark).

### Stable Diffusion watermark

After the SynthID templates, the same step decodes the classic [invisible-watermark](https://pypi.org/project/invisible-watermark) (`dwtDct`) that the Stable Diffusion reference scripts (`StableDiffusionV1`, `SDV2`) and the SDXL pipelines (a fixed 46-bit pattern) write into the pixels. It is a fixed algorithm without a secret key: a Haar transform of the U channel, in each 4 × 4 block the largest coefficient modulo 36 gives one bit, and the bits are voted per payload position. A hit needs the decoded bits to match a payload at z ≥ 5 (about 1 false positive in 500,000 images with random bits; none in 977 real images tested). Badge "AI · Watermark".

This only works on images that are still byte-identical to the generator output: the scheme does not survive JPEG compression (not even at quality 90), scaling or cropping, and many front ends (A1111, ComfyUI, Forge) switch it off. In practice it only matches PNGs straight from a local pipeline.

## 3. Model

Badge "AI 99 %". A classifier run with ONNX Runtime Web (WASM) in the worker: Detectra fine-tuned on OpenFake (44 MB, fp16).

The starting point is [Detectra v3](https://huggingface.co/TensorFold/detectra-v1) (MIT), a fine-tune of the Community Forensics ViT-S on newer generators (DALL·E 3, Midjourney, FLUX, SD 3.5, Recraft, HiDream), on images from social networks ([WildRF](https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld)), and with real paintings, anime drawings, memes and screenshots as negatives.

With [`tools/train/`](../tools/train/) it is trained for three more epochs on 15 files from the train split of [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (about 60,000 images, half real and half from many generators), plus about 3,500 paintings, 500 drawings and 500 prints from [The Met's Open Access collection](https://huggingface.co/datasets/metmuseum/openaccess) (CC0), 5,900 photos from COCO and charts from ChartQA as negatives. Every training image is randomly downscaled and recompressed as JPEG so the model learns the conditions on web pages. See [development.md](development.md#train-the-model) for the training procedure.

Preprocessing: bilinear resize to 440 px, center crop to 384 px. [`src/background/preprocess.ts`](../src/background/preprocess.ts) reimplements it and matches Pillow bit for bit. The raw output is calibrated with Platt scaling ([`model.json`](../model.json)); the raw value would be unsuitable as a percentage because it is often below 0.1 % for real photos.

Under web conditions, at the same false positive rate on photos, the model finds about twice as many AI images as the previously bundled Community Forensics model (46–47 % instead of 23 %). It flags no paintings from the Cleveland Museum of Art and 5 to 7 % of WikiArt paintings (Community Forensics: 8 to 15 % and 11 %). Details are in [accuracy.md](accuracy.md).
