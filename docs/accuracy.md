# Accuracy

## Model

Measured with `npm run eval` on 4,906 images. One half is used for calibration, the numbers below come from the other half.

- **AI images:** 300 from the test split of [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (18 generators up to 2026), 188 from OpenFake's Reddit test set (AI subreddits) and 600 from newer generators in [bigdatamark/synthid-research](https://huggingface.co/datasets/bigdatamark/synthid-research) (150 each from Gemini 3.1 Flash Image, Qwen-Image 2.1, FLUX.2 and GPT-Image 2)
- **real photos:** 943 from OpenFake (ImageNet, DOCCI), 375 from photography subreddits (OpenFake Reddit test set), 1,000 from [COCO](https://huggingface.co/datasets/detection-datasets/coco) and 600 YouTube video thumbnails from the [AudioSet](https://research.google.com/audioset/) eval set (uploaded until early 2017, 300 with and 300 without music), cropped to 16:9 the way YouTube shows them in lists
- **graphics and art:** 300 charts from [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA), 300 paintings from [WikiArt](https://huggingface.co/datasets/huggan/wikiart) and 300 paintings from the [Cleveland Museum of Art Open Access](https://openaccess-api.clevelandart.org/) collection (CC0)

Every image is scored three times: as the original file and the way images typically arrive on web pages, downscaled to at most 1280 px or 800 px and saved as JPEG with quality 80 or 75. Calibration and default threshold are chosen so that about 1 % of real photos are flagged under web conditions. Fitted on the calibration half alone, the threshold would be 91 %, which flags 1.8 % of the photos and 9 % of the WikiArt paintings in the other half at 1280 px. The default of 93 % is the threshold for 1 % flagged photos on both halves together.

At the default threshold of 93 %, with the previous version of the model (add-on 0.3.0, default threshold 92 %) in parentheses:

| | Original file | Web 1280 px | Web 800 px |
|---|---|---|---|
| AI images detected (OpenFake) | 77 % (67 %) | 78 % (65 %) | 71 % (62 %) |
| AI images detected (Reddit) | 81 % (65 %) | 76 % (65 %) | 72 % (54 %) |
| AI images detected (newer generators) | 71 % (57 %) | 49 % (33 %) | 55 % (40 %) |
| real photos flagged (all) | 1.4 % (1.1 %) | 1.2 % (1.3 %) | 1.0 % (1.1 %) |
| real photos flagged (photography subreddits) | 3.7 % (2.7 %) | 3.7 % (2.7 %) | 3.7 % (2.7 %) |
| real photos flagged (DOCCI camera photos) | 6.3 % (5.0 %) | 5.0 % (6.3 %) | 1.9 % (3.8 %) |
| YouTube thumbnails flagged | 0.7 % (0.7 %) | 0.3 % (0.7 %) | 0.7 % (1.4 %) |
| paintings flagged (Cleveland Museum of Art) | 0 % (0 %) | 0 % (0 %) | 0 % (0 %) |
| paintings flagged (WikiArt) | 6.0 % (6.6 %) | 6.6 % (5.3 %) | 5.3 % (4.6 %) |
| charts flagged | 0 % (0 %) | 0 % (0 %) | 0 % (0 %) |

AI images detected per generator at the default threshold, previous version in parentheses:

| Generator (images) | Original file | Web 1280 px | Web 800 px |
|---|---|---|---|
| FLUX.2 (78) | 75 (71) | 59 (47) | 64 (53) |
| GPT-Image 2 (70) | 44 (41) | 30 (21) | 35 (30) |
| Qwen-Image 2.1 (73) | 27 (23) | 14 (11) | 15 (12) |
| Gemini 3.1 Flash Image (85) | 72 (38) | 48 (23) | 55 (26) |
| OpenFake: Z-Image Turbo (29) | 24 (22) | 23 (21) | 22 (22) |
| OpenFake: FLUX.2 klein 9B (25) | 21 (9) | 21 (10) | 20 (9) |
| OpenFake: Midjourney 7 (20) | 10 (11) | 12 (10) | 8 (9) |
| OpenFake: Illustrious (19) | 19 (19) | 19 (19) | 19 (19) |
| OpenFake: GPT-Image 1.5 (17) | 14 (11) | 15 (12) | 11 (8) |
| OpenFake: 12 other generators (43) | 30 (31) | 29 (28) | 28 (28) |
| Reddit (85) | 69 (55) | 65 (55) | 61 (46) |

Independent check on 2,320 images that are neither in the training data nor in this measurement: 1,678 AI images (816 GPT-Image 2 images from three Hugging Face datasets, among them [winrisef/gpt-generated-image-collection](https://huggingface.co/datasets/winrisef/gpt-generated-image-collection) and [Sarim-Hash/liars-dividend-GPT-image-2-photoreal](https://huggingface.co/datasets/Sarim-Hash/liars-dividend-GPT-image-2-photoreal), 300 other OpenAI images from three more datasets, 35 Gemini and OpenAI images from Wikimedia Commons, and 527 further images from the OpenFake test split: Z-Image Turbo, FLUX.2 klein 9B, GPT-Image 1.5, Midjourney 7, GPT-Image 2 and Nano Banana Pro) and 642 further real photos from the OpenFake test split (ImageNet, DOCCI). At the default threshold, under web conditions (1280 px and 800 px):

| | this version | previous version |
|---|---|---|
| AI images detected (all) | 63 % and 60 % | 47 % and 44 % |
| GPT-Image 2 from the three datasets | 45 % and 41 % | 27 % and 25 % |
| real photos flagged | 1.4 % and 0.9 % | 1.4 % and 0.8 % |

Other thresholds (web 1280 px, all AI images against real photos):

| Threshold | AI images detected | real photos flagged |
|---|---|---|
| 50 % | 84 % | 10 % |
| 70 % | 78 % | 6.0 % |
| 80 % | 72 % | 3.9 % |
| 90 % | 65 % | 2.1 % |
| **93 % (default)** | **62 %** | **1.2 %** |
| 97 % | 53 % | 0.5 % |

Comparison with other local models under web conditions (1280 px and 800 px), AI images detected at exactly 1 % flagged photos:

| Model | AUC | AI images detected |
|---|---|---|
| **Detectra fine-tuned on OpenFake (bundled)** | **0.94–0.95** | **57–62 %** |
| Detectra fine-tuned on OpenFake, previous version (add-on 0.3.0) | 0.91–0.93 | 46–47 % |
| [Detectra v3](https://huggingface.co/TensorFold/detectra-v1) without fine-tuning (fp16) | 0.89–0.90 | 33–34 % |
| [fev v13](https://huggingface.co/angelhd25/ull-ai-image-detector) (fp32, 88 MB), see below | 0.90 | 37–39 % |
| Community Forensics (int8, previously bundled) | 0.85 | 23 % |
| Community Forensics without quantization (fp32, four times the size) | 0.89 | 25 % |
| [SigLIP detector](https://huggingface.co/Ateeqq/ai-vs-human-image-detector) (int8) | 0.71 | 8–9 % |
| SMOGY, ViT deepfake classifier | 0.55–0.62 | 0 % |
| [capcheck/ai-image-detection](https://huggingface.co/onnx-community/ai-image-detection-ONNX), [ai-image-detect-distilled](https://huggingface.co/onnx-community/ai-image-detect-distilled-ONNX) | 0.46–0.48 | 0–1 % |

The SigLIP detector often considers YouTube video thumbnails AI-generated, including those of well-known old music videos: the original PyTorch model outputs more than 99 % for 23 of 26 tested thumbnails, so it is not caused by quantization or preprocessing. At 1 % false positives on photos overall, it flags 2–4 % of YouTube thumbnails.

### Limitations

- The badge is a hint, not proof. The default threshold is tuned for few false positives; in exchange, about 40 % of the AI images without metadata or watermark stay unmarked under web conditions. To catch more, lower the threshold in the settings and accept more false positives.
- The model is weaker on the newest generators. Under web conditions (1280 px) it finds 59 of 78 FLUX.2 images, 30 of 70 GPT-Image 2 images, 14 of 73 Qwen-Image 2.1 images and 48 of 85 Gemini images; Community Forensics found 14 to 18 % of these, but only 1 to 3 of the Gemini images. As original files it detects 71 %, so compression costs particularly much here. Gemini images are additionally found by the SynthID check.
- The model flags high-quality camera photos like those from DOCCI more often than Community Forensics, 2 to 5 % instead of 1 to 2 % under web conditions, and heavily edited photos from photography subreddits slightly more often (4 % instead of 2 to 3 %). YouTube thumbnails are flagged at most 0.7 %. 3D renderings are not in the test set.
- The fine-tuning uses public domain artworks from the Met as negatives, mostly from before 1900. The paintings from the Cleveland Museum of Art are similar and are not flagged at all, WikiArt contains more modern and abstract art and is flagged at 5 to 7 %. Detectra, the base model, was itself trained with WikiArt paintings.
- The fine-tuning uses the OpenFake train split, COCO train and Met artworks, none of which appear in the measurement. The train split contains some of the newest generators measured here or their siblings (FLUX.2 dev and klein 4B, Qwen-Image, and 1,095 images from Nano Banana 2, which is Gemini 3.1 Flash Image), but other images and prompts. The measurement uses the test split with other generators and other photo sources (ImageNet and DOCCI instead of LAION and Pexels). The checkpoint was, however, selected based on this measurement (out of a few variants), so the numbers are slightly optimistic. The independent check above shows a similar gain.
- Compression and downscaling cost detection (AUC 0.97 as original files, 0.94 to 0.95 under web conditions).
- Calibration and measurement use separate images but the same sources, and the default threshold also takes the measured photos into account. Sites with a different image mix or their own compression may differ.

## SynthID watermark

### Google

Measured with `tools/synthid/eval.py` on a sample from the OpenFake test split (`tools/synthid/fetch_openfake.py`). None of these images is part of the template.

| Gemini images (Nano Banana Pro) | detected |
|---|---|
| originals as PNG (32 images, 848 × 1264 up to 4K) | 88 % |
| the same, downscaled to 50 % or 33 % | 88 % or 84 % |
| the same as JPEG, quality 90, without or with chroma subsampling (4:4:4 or 4:2:0) | 69 % or 47 % |
| the same as WebP, quality 80 | 3 % |
| the same, cropped to 95 % on all sides | 0 % |
| already re-encoded as JPEG/WebP in the dataset (28 images) | 11 % |

Second measurement on [bigdatamark/synthid-research](https://huggingface.co/datasets/bigdatamark/synthid-research) (CC BY 4.0, images from September 2026, `tools/synthid/fetch_research.py`), same prompts for all generators:

| Images | flagged by the Google template |
|---|---|
| Gemini 3.1 Flash Image, JPEG 2048 × 2048 as delivered by Gemini (700) | 95 % |
| gpt-image-2 (400), FLUX.2 (400), Qwen-Image-2.1 (700) | 0 % |

False positives: 0 of 1,067 real photos (ImageNet, DOCCI) and 0 of 417 images from other generators (DALL·E 3, GPT-Image 1.5 and 2, Sora 2, Seedream 5, FLUX.2, Z-Image, ERNIE-Image, Illustrious). The highest z-score of a real photo was 9.4. Three more images that OpenFake attributes to other generators triggered. One carries a C2PA manifest from Google, the other two have exactly the resolution of Nano Banana Pro outputs (2528 × 1696 and 3392 × 5056); they are probably Gemini images. In a second sample from OpenFake (642 real photos), the Google template triggered on one DOCCI photo (z 15.4).

### OpenAI

Measured with `tools/synthid/crossval.py`: each template is built from two thirds of the images of its size and tested on the remaining third (z ≥ 8).

| Output size (test images) | Original (PNG) | downscaled to 50 % / 33 % | JPEG 90, 4:4:4 / 4:2:0 | WebP 80 |
|---|---|---|---|---|
| 1254 × 1254 (141) | 94 % | 94 % / 87 % | 93 % / 94 % | 84 % |
| 1448 × 1086 (73) | 93 % | 88 % / 30 % | 81 % / 78 % | 59 % |
| 1672 × 941 (41) | 88 % | 85 % / 41 % | 71 % / 63 % | 61 % |
| 1122 × 1402 (64) | 75 % | 67 % / 11 % | 56 % / 36 % | 25 % |

Cropped to 95 % on all sides: 0 %. The shipped templates are built from all images (424, 219, 123 and 193).

Two probes show that the component depends on the size and not on the account: a template built only from the 420 images in bigdatamark detects three of the four 1254 × 1254 images from the second source (z 8 to 18), but none of the other sizes from the same source (at most z 3.1). And in a third, independent dataset ([Sarim-Hash/liars-dividend-GPT-image-2-photoreal](https://huggingface.co/datasets/Sarim-Hash/liars-dividend-GPT-image-2-photoreal), 80 images at 1024 × 1024 without C2PA), one image reaches z 69 with the 1254 template, probably a downscaled copy. The other 79 have the uncovered size 1024 × 1024 and do not trigger.

False positives (measured like `eval.py --ignore-aspect`, so every template on every image): none of 1,933 images without an OpenAI watermark (642 real photos from ImageNet and DOCCI, 1,291 images from Gemini, FLUX.2, Qwen-Image, Z-Image, Midjourney 7 and GPT-Image 1.5), not even when every image is additionally cropped to each template's aspect ratio and scaled to its exact size. Likewise none of 456 further OpenAI images from Hugging Face datasets without C2PA evidence, mostly in uncovered sizes. The highest z-score was 6.6, the standard deviation 1.1 (ideal would be 1).

### Limitations

- This is a reconstruction from public sample images, not Google's decoder. Only images from Gemini 3 Pro Image (Nano Banana Pro), Gemini 3.1 Flash Image and GPT-Image 2 (ChatGPT and API) are tested, other models are unverified.
- OpenAI images are only detected in the four template sizes and as downscaled copies of them. Fixed API sizes like 1024 × 1024 or 1536 × 1024 and rare sizes are missing because there are no or too few images with a verified watermark. The 1122 × 1402 template detects less reliably, and every template is based on at most two sources. OpenAI images from before May 2026 carry no SynthID.
- Cropping destroys the signal completely. Lossy re-encoding weakens it, strongly for Gemini images. Images from social networks are therefore often not detected; then the model takes over.
- Real photos edited with Gemini carry SynthID as well and are flagged, those edited with ChatGPT probably too.
- The Google template is computed from the Hugging Face dataset [aoxo/reverse-synthid](https://huggingface.co/datasets/aoxo/reverse-synthid) (no license stated there), the OpenAI templates from [bigdatamark/synthid-research](https://huggingface.co/datasets/bigdatamark/synthid-research) (CC BY 4.0) and [winrisef/gpt-generated-image-collection](https://huggingface.co/datasets/winrisef/gpt-generated-image-collection) (MIT). For OpenAI, only images whose C2PA manifest records the watermark (`c2pa.watermarked.unbound`) count. All revisions are pinned; only averaged phases and weights are shipped, no images.

## Amazon Nova Canvas watermark

Measured with `tools/synthid/crossval.py amazon` on 75 Nova Canvas PNGs whose C2PA manifest names the model, from 10 GitHub repositories (512 × 512 to 1280 × 720): the template is built from nine repositories and tested on the images of the tenth, in turn for each repository (z ≥ 8).

| Nova Canvas images (75) | detected |
|---|---|
| originals (PNG) | 100 % |
| downscaled to 50 % or 33 % | 23 % or 0 % |
| JPEG 90 without or with chroma subsampling (4:4:4 or 4:2:0) | 81 % or 7 % |
| WebP 80 | 0 % |
| cropped to 95 % on all sides | 0 % |

The shipped template, applied to images from AWS blog posts that are not part of it: 52 of 58 PNGs from the Nova Canvas visual guide (all with seed 0, partly downscaled by the blog), all 13 PNGs from five other posts, none of 9 JPEGs with chroma subsampling.

False positives: none of 7,795 images without the Nova Canvas watermark (real photos from ImageNet, DOCCI, COCO and Reddit, YouTube thumbnails, paintings, charts, and images from Gemini, GPT-Image 1.5 and 2, FLUX.2, Qwen-Image, Z-Image, Midjourney 7 and other generators), nor any of 37 Titan images. The highest z-score was 4.5, the standard deviation 1.05. Without skipping images whose Cr channel hardly varies, a black-and-white GPT-Image 1.5 image reached z 12.9.

### Limitations

- The signal sits close to the resolution limit of the original. Downscaling to half the size, JPEG with chroma subsampling, WebP and cropping erase it, so in practice only unmodified PNGs from Nova Canvas are found. Unmodified Nova Canvas and Titan images also carry a C2PA manifest with `trainedAlgorithmicMedia`, which the metadata check reads first.
- 46 of the 75 template images come from one repository. Output sizes other than 512 × 512, 1024 × 576, 1024 × 768, 1024 × 1024 and 1280 × 720 are untested, Nova 2 Omni is unverified.
- The template images come from public GitHub repositories, pinned to commits in [`tools/synthid/amazon_sources.json`](../tools/synthid/amazon_sources.json); only images whose C2PA manifest from Amazon Bedrock names Nova Canvas count. Only averaged phases and weights are shipped, no images.

## Stable Diffusion, SDXL and FLUX watermark

Measured on 12 AI images (6 FLUX.2 Pro PNGs, 6 SDXL JPEGs) that the Python library `invisible-watermark` marked in the same way as the reference code:

| marked images (12 per variant) | detected |
|---|---|
| SDXL and FLUX payload, PNG | 10 of 12 each |
| SDXL and FLUX payload with red and blue swapped (diffusers before April 2024), PNG | 9 of 12 each |
| `StableDiffusionV1` or `SDV2`, PNG | 10 or 9 of 12 |
| any payload, JPEG 90 | 0 of 12 |

In the wild: 10 SDXL outputs among the 279 PNGs of the diffusers documentation images ([huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images/tree/df6c7e6873499cfe63686cb4b6edb05d6f8a5755/diffusers), for example [`sdxl-double-prompt.png`](https://huggingface.co/datasets/huggingface/documentation-images/blob/df6c7e6873499cfe63686cb4b6edb05d6f8a5755/diffusers/sdxl-double-prompt.png) and [`with_cfg_callback.png`](https://huggingface.co/datasets/huggingface/documentation-images/blob/df6c7e6873499cfe63686cb4b6edb05d6f8a5755/diffusers/with_cfg_callback.png)), 46 to 48 of 48 bits matching; eight of them have red and blue swapped. Most other images there are grids, which break the bit order. SDXL on Amazon Bedrock and the v1 Stability API writes the same watermark: 26 of 36 distinct SDXL images with the Stability EXIF fields from 27 public GitHub repositories carry it (43 to 48 of 48 bits), plus 5 images whose metadata was removed. The misses are mostly 512 × 512 and 768 × 512 images from 2023. No hit in Wikimedia Commons' Stable Diffusion category (52 PNGs), the example images of the Stability, Black Forest Labs, CompVis and diffusers repositories, or seven datasets (FLUX.1.1 Pro and FLUX.2 Pro via API, SDXL, SD 1.4/2.1, FLUX.2 klein).

False positives: none of 9,090 images without this watermark (real photos from ImageNet, DOCCI, COCO and Reddit, YouTube thumbnails, paintings, charts, and images from FLUX.1.1 Pro, FLUX.2, Gemini, GPT-Image 1.5 and 2, Qwen-Image, Z-Image, Midjourney 7 and other generators).

### Limitations

- Only images that are still byte-identical to the generator output: JPEG compression, scaling, cropping and grids erase the watermark. Many front ends (A1111, ComfyUI, Forge) and the FLUX API do not write it.

## Visible Gemini logo

Measured with `tools/visible/eval.py`; a hit needs a correlation of at least 0.88.

Real images that carry the logo, labeled by eye: 12 of the 33 images in Wikimedia Commons' Nano Banana and Gemini categories (682 × 372 to 1264 × 843, mostly PNG) and 22 of the 188 AI images in OpenFake's Reddit test set (posts from AI subreddits, JPEG as downloaded, 572 × 1024 to 2816 × 1536).

| Real images with the logo | Commons (12) | Reddit (22) |
|---|---|---|
| as downloaded | 10 | 19 |
| JPEG 85 or WebP 80 | 10 | 19 |
| downscaled to 800 px width, JPEG 75 | 9 | 19 |
| downscaled to 50 %, JPEG 80 | 8 | 19 |
| downscaled to 33 %, JPEG 80 | 8 | 18 |
| cropped to 95 % on all sides | 0 | 0 |

The misses as downloaded: two logos on graffiti and across the edge of a sheet of paper (0.82 and 0.77), one on wet asphalt (0.82), and two Reddit images that were cropped or letterboxed after generation, so the logo is no longer where the layout puts it. The Google SynthID template finds only 3 of these 34 images.

Synthetic test: the logo overlaid with the alpha maps and positions of GeminiWatermarkTool in five layouts on 200 AI images without it (100 each from the OpenFake test split and bigdatamark/synthid-research):

| Layout (logo, margin) | Original | JPEG 85 | WebP 80 | 800 px, JPEG 75 | 50 %, JPEG 80 | 33 %, JPEG 80 | Crop 95 % |
|---|---|---|---|---|---|---|---|
| older, 1024 × 1024 (48 px, 32 px) | 79 % | 78 % | 76 % | 76 % | 78 % | 75 % | 0 % |
| older, 2816 × 1536 (96 px, 64 px) | 74 % | 74 % | 74 % | 70 % | 71 % | 71 % | 0 % |
| newer, 1024 × 1024 (36 px, 71 px) | 58 % | 57 % | 56 % | 56 % | 52 % | 0 % | 0 % |
| newer, 1376 × 768 (48 px, 96 px) | 50 % | 50 % | 49 % | 50 % | 51 % | 46 % | 0 % |
| newer, 2816 × 1536 (96 px, 192 px) | 56 % | 55 % | 56 % | 52 % | 51 % | 52 % | 0 % |

The newer logo is fainter (α at most 0.37 instead of 0.51) and disappears more often in bright or busy corners. At 33 % the 36 px logo shrinks to 12 px, below the smallest size the check tries. Plain grayscale correlation without the blending model found 56 to 66 % of the older and 30 to 40 % of the newer layout (thresholds 0.85 to 0.9, 120 of these images) and 8 of the 12 Commons images.

False positives: none of 8,883 images without the logo, namely 4,458 real photos, paintings, charts and YouTube thumbnails (the model's test set and 642 photos from OpenFake) and 4,425 AI images, among them 732 Gemini images without the logo and images from GPT-Image 1.5, 2 and 2.5, FLUX.2, Qwen-Image, Z-Image, Midjourney 7, Sora 2 and Nova Canvas. The highest correlation was 0.84 for a round highlight 16 px wide in a Sora 2 image and 0.81 for a reflection in a COCO photo; small round highlights cause most high scores, which is why logos below 16 px are not searched.

The check takes about 3 ms per image in Node, independent of the resolution.

### Limitations

- The logo has to be where Gemini put it. Cropping, added borders and mirroring move it away from the corner. Downscaled copies are found down to a logo of 16 px, which is a 1024 × 1024 image downscaled to about 340 px (older layout) or 460 px (newer layout).
- On white or very bright corners the white logo leaves almost no trace, and on busy textures it drowns. Users can switch the logo off since August 2026, and removal tools such as GeminiWatermarkTool reverse the blending exactly.
- Every image with a white four-pointed star of this shape in this place is flagged, for example a pasted Gemini icon.
- The real test images come from two sources and are few (34); most carry the older layout.

## Visible Kling logo

Measured with `tools/visible/eval.py --logo kling`; a hit needs a correlation of at least 0.6.

Real images with the logo, labeled by eye and downloaded unchanged from Kling's servers:

- **2024, nyuuzyou (1,005):** text-to-image outputs from [nyuuzyou/klingai](https://huggingface.co/datasets/nyuuzyou/klingai) (CC0), one per user, July to December 2024, in seven output sizes of about one megapixel (1024 × 1024, 768 × 1344, 1344 × 768 and others).
- **2024, GitHub (18):** Kling images linked from seven other GitHub repositories, among them the newer model version 1.5 and output sizes the template was not built from (832 × 1248, 1280 × 720, 1080 × 1964).
- **current, GitHub (68):** free-tier outputs from 2026 linked from andreathar/Movies_Projects (65) and newmemomedo2018-afk/cgi-generator-mongo (3), eight output sizes from 896 × 992 to 4128 × 1024.

The templates are averaged from the first and the third set. With templates averaged from one half of each set, the other half scores the same (983 of 1,005 and 68 of 68).

| Real images with the logo | 2024, nyuuzyou (1,005) | 2024, GitHub (18) | current, GitHub (68) |
|---|---|---|---|
| as downloaded | 983 (97.8 %) | 17 | 68 |
| JPEG 85 | 970 (96.5 %) | 17 | 68 |
| WebP 80 | 971 (96.6 %) | 17 | 68 |
| downscaled to 800 px width, JPEG 75 | 964 (95.9 %) | 17 | 68 |
| downscaled to 50 %, JPEG 80 | 941 (93.6 %) | 17 | 68 |
| downscaled to 33 %, JPEG 80 | 893 (88.9 %) | 17 | 68 |
| cropped to 95 % on all sides | 0 | 0 | 0 |

The median correlation as downloaded is 0.97 for both layouts; current logos score 0.87 to 0.98. Most of the 22 missed 2024 images have the logo on a white or very bright background, where the white letters leave only a thin gray outline, the rest on busy or strongly colored ones. The missed GitHub image has a second watermark painted over the logo. The AIGC label in the metadata is present in 16 of the 68 current images and in none of the 1,023 with the 2024 layout.

Kling images without the logo: 1,627 images from the public explore pages of kling.ai (Kolors 1.5 to 3.0, 720 × 1280 to 3072 × 5440). 10 of them still show the current Kling logo, several of them visibly painted by the model itself, with a different symbol or misspelled as "KiingAI" or "KingAi". The check finds 9 of these 10 (0.64 to 0.94); the tenth, a logo away from the usual position, scores 0.52. The other 1,617 score at most 0.39.

False positives: none of 10,534 images without the logo, namely the 8,917 images of the Gemini logo test (the 8,883 images without the Gemini logo and the 34 with it) and the 1,617 Kling images without a logo. The highest correlation was 0.40 for a GPT-Image 1.5 image, and 0.42 after downscaling all of them to 33 % with JPEG 80. White text of 20 other names written into the same place in 60 photos and AI images ("Hailuo AI", "Jimeng AI", "Midjourney", "Getty Images", "豆包AI生成" and others, in four fonts) scores at most 0.43 in either layout; only the word "KlingAI" without the symbol reaches 0.61.

The check takes 2.5 to 5.3 ms per image in Node, depending on the size of the logo.

### Limitations

- The logo has to be where Kling put it, in the proportions of the original output. Cropping, added borders and mirroring move it away. Logos narrower than 24 px are not searched, which for the 2024 layout means images whose long side is below about 270 px.
- On white or very bright corners the white logo leaves almost no trace. Images from paid plans carry no logo.
- Other variants are not covered: the logo of the Chinese site ("可灵AI") and logos with a model version such as "KLING AI 1.6", "可灵AI 2.1" or "KlingAI 3.0 Omni". The GitHub sets contain examples of them, but too few per variant to measure positions.
- Every image with this logo in this place is flagged, also a copy that a model painted into an image.
- The current layout is measured on 68 images from two GitHub repositories, most of them from a single project.

## Tested and not included

- **Ensembles with other models** (Detectra v3, [CommunityForensics-DeepfakeDet-ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT), [ai-vs-human-SigLIP-ONNX](https://huggingface.co/neilxxx/ai-vs-human-SigLIP-ONNX)): on top of the fine-tuned model, these models, alone or together, add no hits at the same false positive rate, but cost 0.7 s per image each (155 MB together).
- **WikiArt paintings as negatives:** a version fine-tuned with 4,500 WikiArt paintings instead of the Met artworks detected 2 to 3 percentage points more AI images under web conditions and flagged only 0.7 % of WikiArt paintings, but 9 to 14 % of the paintings from the Cleveland Museum of Art. It had learned WikiArt rather than paintings in general, and WikiArt's images are not licensed for training. Met images sampled from the whole collection instead of contiguous parts of it, and an average of the weights of both runs, detected no more AI images at 1 % flagged photos.
- **synthid-research images in training:** a pilot run (3 OpenFake files, 2 epochs) added 813 images from [bigdatamark/synthid-research](https://huggingface.co/datasets/bigdatamark/synthid-research) (CC BY 4.0) at four times their weight: every image whose prompt (the number in the file name) is not among the 415 prompts of the 600 images in this measurement, from Gemini 3.1 Flash Image (285 and 35 downscaled copies), Qwen-Image 2.1 (285), FLUX.2 (99), GPT-Image 2 (99) and GPT-Image 2.5 (10). Under web conditions it detected 85 to 87 % instead of 29 to 33 % of the synthid-research images in the measurement, but not more of the other images of these generators (independent check 36 % instead of 34 % at 1280 px, 30 % instead of 34 % at 800 px), and 8 percentage points fewer from OpenFake and Reddit. It had learned the style of the dataset (the same prompts and settings for every generator, mostly square object images), not the generators.
- **Only the newest generators' images in the 15 OpenFake files** (2,916, shown three more times per epoch, without the 6,768 from further files and the Pexels photos): 53 to 54 % of AI images detected at 1 % flagged photos under web conditions instead of 57 to 62 %, on the independent check 54 and 51 % instead of 63 and 60 %.
- **Five crops per image** (measured with the WikiArt version): center and four corners of the image downscaled to 640 px on the short side, taking the highest score. Under web conditions this detects 1 to 5 percentage points more AI images, but takes five times the compute and flags more camera photos (DOCCI up to 7 %) and edited subreddit photos (up to 5 %).
- **fev v13** ([angelhd25/ull-ai-image-detector](https://huggingface.co/angelhd25/ull-ai-image-detector), MIT), like Detectra a fine-tune of the Community Forensics ViT-S for a browser add-on: alone it detects 37–39 % of AI images under web conditions, but flags 7 % of paintings. In an ensemble with Detectra v3 it added no hits at no more than 5 % flagged paintings, but costs 88 MB and 0.7 s per image. It was also partly trained on the OpenFake validation split (CC BY-NC 4.0), whose generators and photo sources do not appear in the test split used here.
- **Other detectors from Hugging Face** (capcheck/ai-image-detection, ai-image-detect-distilled, SMOGY and others): many are trained on small datasets like CIFAKE and separate AI images under web conditions hardly or not at all better than chance (AUC 0.46 to 0.62). Larger models like SwinV2-Large detectors or SigLIP-DINOv2 ensembles would be too big for the add-on at 200 MB to 2 GB.
- **Amazon Titan Image Generator watermark:** Titan images (G1 and G1 v2, 35 images with C2PA from 12 GitHub repositories and 37 from an AWS blog post) share a phase pattern only when they were generated with the same seed (0 and 42 are common defaults). Images with random seeds from other repositories show none (z ≈ 0), so no fixed template is possible. Unmodified Titan images are recognized by their C2PA manifest.
- **Other visible logos:** [remove-ai-watermarks](https://github.com/wiltodelta/remove-ai-watermarks) (Apache 2.0) has alpha maps for the visible marks of Doubao, Jimeng, Qwen, Samsung Galaxy AI and others, captured over flat backgrounds (Kling's logo is checked, see [above](#visible-kling-logo)). Most of them are text labels such as "豆包AI生成" whose size and position depend on the output, and since One UI 6.1.1 Samsung's star comes with localized text ("AI-generated content", in Italian "Contenuti generati…"). None of the local test sets contains real examples, so positions and false positives could not be checked without collecting new images. A Grok and a Meta image from that project's captures show no visible mark.
- **Stability AI on Amazon Bedrock:** the newer models (SD3 Large, SD3.5 Large, Stable Image Ultra) write no `invisible-watermark` and show no fixed phase pattern in 434 PNGs from 61 GitHub repositories and 23 PNGs from AWS blog posts. Only the SDXL outputs among them carry the SDXL watermark and the EXIF fields of the Stability API.
