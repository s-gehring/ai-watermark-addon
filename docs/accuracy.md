# Accuracy

## Model

Measured with `npm run eval` on 4,906 images. One half is used for calibration, the numbers below come from the other half.

- **AI images:** 300 from the test split of [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (18 generators up to 2026), 188 from OpenFake's Reddit test set (AI subreddits) and 600 from newer generators in [bigdatamark/synthid-research](https://huggingface.co/datasets/bigdatamark/synthid-research) (150 each from Gemini 3.1 Flash Image, Qwen-Image 2.1, FLUX.2 and GPT-Image 2)
- **real photos:** 943 from OpenFake (ImageNet, DOCCI), 375 from photography subreddits (OpenFake Reddit test set), 1,000 from [COCO](https://huggingface.co/datasets/detection-datasets/coco) and 600 YouTube video thumbnails from the [AudioSet](https://research.google.com/audioset/) eval set (uploaded until early 2017, 300 with and 300 without music), cropped to 16:9 the way YouTube shows them in lists
- **graphics and art:** 300 charts from [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA), 300 paintings from [WikiArt](https://huggingface.co/datasets/huggan/wikiart) and 300 paintings from the [Cleveland Museum of Art Open Access](https://openaccess-api.clevelandart.org/) collection (CC0)

Every image is scored three times: as the original file and the way images typically arrive on web pages, downscaled to at most 1280 px or 800 px and saved as JPEG with quality 80 or 75. The calibration is fitted on one half, and the default threshold of 95 % flags at most 1 % of that half's real photos under web conditions.

At the default threshold of 95 %:

| | Original file | Web 1280 px | Web 800 px |
|---|---|---|---|
| AI images detected (OpenFake) | 83 % | 77 % | 76 % |
| AI images detected (Reddit) | 78 % | 76 % | 66 % |
| AI images detected (newer generators) | 83 % | 68 % | 68 % |
| real photos flagged (all) | 0.6 % | 0.4 % | 0.3 % |
| real photos flagged (photography subreddits) | 2.1 % | 1.6 % | 0 % |
| real photos flagged (DOCCI camera photos) | 0 % | 0.6 % | 0.6 % |
| YouTube thumbnails flagged | 0.3 % | 0.3 % | 1.0 % |
| paintings flagged (Cleveland Museum of Art) | 0.6 % | 0 % | 0 % |
| paintings flagged (WikiArt) | 7.3 % | 6.6 % | 4.6 % |
| charts flagged | 0 % | 0 % | 0 % |

On 2,320 further images that are neither in the training data nor in this measurement (GPT-Image 2 and other OpenAI images from six Hugging Face datasets, Gemini and OpenAI images from Wikimedia Commons, further OpenFake test images, ImageNet and DOCCI photos), it detects 68 % and 60 % of the AI images under web conditions (1280 px and 800 px) and flags 0.5 % and 0.3 % of the photos.

Other thresholds (web 1280 px, all AI images against real photos):

| Threshold | AI images detected | real photos flagged |
|---|---|---|
| 50 % | 88 % | 7.2 % |
| 70 % | 86 % | 4.2 % |
| 80 % | 83 % | 2.5 % |
| 90 % | 78 % | 1.2 % |
| **95 % (default)** | **72 %** | **0.4 %** |
| 97 % | 66 % | 0.1 % |

Comparison with other local models under web conditions (1280 px and 800 px), AI images detected at exactly 1 % flagged photos:

| Model | AUC | AI images detected |
|---|---|---|
| **[Detectra fine-tuned on OpenFake](https://huggingface.co/s-gehring/detectra-openfake) (bundled)** | **0.97** | **77 %** |
| [Detectra v3](https://huggingface.co/TensorFold/detectra-v1) without fine-tuning (fp16) | 0.89–0.90 | 33–34 % |
| [fev v13](https://huggingface.co/angelhd25/ull-ai-image-detector) (fp32, 88 MB), see below | 0.90 | 37–39 % |
| Community Forensics (int8) | 0.85 | 23 % |
| Community Forensics without quantization (fp32, four times the size) | 0.89 | 25 % |
| [SigLIP detector](https://huggingface.co/Ateeqq/ai-vs-human-image-detector) (int8) | 0.71 | 8–9 % |
| SMOGY, ViT deepfake classifier | 0.55–0.62 | 0 % |
| [capcheck/ai-image-detection](https://huggingface.co/onnx-community/ai-image-detection-ONNX), [ai-image-detect-distilled](https://huggingface.co/onnx-community/ai-image-detect-distilled-ONNX) | 0.46–0.48 | 0–1 % |

The SigLIP detector often considers YouTube video thumbnails AI-generated, including those of well-known old music videos: the original PyTorch model outputs more than 99 % for 23 of 26 tested thumbnails, so it is not caused by quantization or preprocessing. At 1 % false positives on photos overall, it flags 2–4 % of YouTube thumbnails.

### Limitations

- The badge is a hint, not proof. The default threshold is tuned for few false positives; in exchange, about 30 % of the AI images without metadata or watermark stay unmarked under web conditions. To catch more, lower the threshold in the settings and accept more false positives.
- The model is weaker on the newest generators. Under web conditions (1280 px) it finds 67 of 78 FLUX.2 images, 47 of 70 GPT-Image 2 images, 26 of 73 Qwen-Image 2.1 images and 67 of 85 Gemini images; Community Forensics found 14 to 18 % of these, but only 1 to 3 of the Gemini images. As original files it detects 83 %, so compression costs particularly much here. Gemini images are additionally found by the SynthID check.
- Heavily edited photos from photography subreddits are flagged at up to 2 %, camera photos from DOCCI and YouTube thumbnails at up to 1 %. 3D renderings are not in the test set.
- The fine-tuning uses public domain artworks from the Met as negatives, mostly from before 1900. The paintings from the Cleveland Museum of Art are similar and are hardly flagged (1 of 156 as original file), WikiArt contains more modern and abstract art and is flagged at 5 to 7 %. Detectra, the base model, was itself trained with WikiArt paintings.
- The fine-tuning uses the OpenFake train split, COCO train, Met artworks, photos from LAION-Mobile and Unsplash and self-generated images, none of which appear in the measurement. The train split contains some of the newest generators measured here or their siblings (FLUX.2 dev and klein 4B, Qwen-Image, and 1,095 images from Nano Banana 2, which is Gemini 3.1 Flash Image), but other images and prompts. The self-generated images come from FLUX.2 klein, Qwen-Image 2.1 and Z-Image, also with other prompts. The measurement uses the test split with other generators and other photo sources (ImageNet and DOCCI instead of LAION and Pexels). The checkpoint was, however, selected based on this measurement (out of a few variants), so the numbers are slightly optimistic. On the independent images above it detects about as many.
- Compression and downscaling cost detection (AUC 0.98 as original files, 0.97 under web conditions).
- Calibration and measurement use separate images but the same sources; the threshold also uses the measured photos. Sites with a different image mix or their own compression may differ.

## SynthID watermark

### Google

Measured with `tools/synthid/eval.py` on [bigdatamark/synthid-research](https://huggingface.co/datasets/bigdatamark/synthid-research) (CC BY 4.0, images from September 2026, `tools/synthid/fetch_research.py`), same prompts for all generators. None of these images is part of the template.

| Gemini 3.1 Flash Image (200 JPEGs, 2048 × 2048 as delivered by Gemini) | detected |
|---|---|
| as delivered | 100 % |
| downscaled to 50 % or 33 % | 100 % or 100 % |
| re-encoded as JPEG 90, without or with chroma subsampling (4:4:4 or 4:2:0) | 100 % or 99 % |
| re-encoded as JPEG 75 with chroma subsampling | 91 % |
| re-encoded as WebP 80 | 0 % |
| cropped to 95 % on all sides | 0 % |

| Images | flagged by the Google template |
|---|---|
| Gemini 3.1 Flash Image, all 700 JPEGs as delivered | 699 of 700 |
| gpt-image-2 (400), FLUX.2 (400), Qwen-Image-2.1 (700) | 0 % |

False positives (measured like `eval.py --ignore-aspect`): 1 of 4,457 real photos, paintings, charts and YouTube thumbnails (ImageNet, DOCCI, COCO, OpenFake's Reddit photos, Cleveland Museum of Art, WikiArt), a 3302 × 4922 photo from a photography subreddit (z 37), possibly edited with Gemini. Otherwise the highest z-score of a real image was 3.6, the standard deviation 1.0. None of 1,711 images from other generators (GPT-Image 1.5 and 2, FLUX.2, Qwen-Image, Z-Image, Midjourney 7, Sora 2, Seedream 5, ERNIE-Image, Illustrious and others) reached more than z 5.7. Three more images that OpenFake attributes to GPT-Image 1.5 triggered; they have exactly the resolution of Nano Banana Pro outputs (2752 × 1536 and 1696 × 2528) and are probably Gemini images.

### OpenAI

Measured with `tools/synthid/crossval.py`: each template is built from two thirds of the images of its size and tested on the remaining third (z ≥ 8).

| Output size (test images) | Original (PNG) | downscaled to 50 % / 33 % | JPEG 90, 4:4:4 / 4:2:0 | WebP 80 |
|---|---|---|---|---|
| 1254 × 1254 (141) | 100 % | 100 % / 100 % | 100 % / 100 % | 97 % |
| 1448 × 1086 (73) | 100 % | 100 % / 100 % | 100 % / 100 % | 99 % |
| 1672 × 941 (41) | 100 % | 100 % / 100 % | 100 % / 100 % | 98 % |
| 1122 × 1402 (64) | 100 % | 100 % / 100 % | 100 % / 100 % | 95 % |

Cropped to 95 % on all sides: 0 %. The shipped templates are built from all images (424, 219, 123 and 193).

Other output sizes: ChatGPT picks the size from the aspect ratio of the input image, so many images have no template of their own. Of 2,165 GPT-Image 2 PNGs with a C2PA watermark assertion in 67 other sizes (1086 × 1448, 1559 × 1009, 1537 × 1023, 1024 × 1536, 1536 × 1024 and others; from the MLLM dataset [zr-zhang/MLLM-Generated-Image-Detection-Dataset](https://huggingface.co/datasets/zr-zhang/MLLM-Generated-Image-Detection-Dataset), wafflefan/felix-chatgpt-generated-images, mramazan/Synthetic-Character-Dataset-for-FLUX-LORA and winrisef), 99 % trigger.

In an independent dataset ([Sarim-Hash/liars-dividend-GPT-image-2-photoreal](https://huggingface.co/datasets/Sarim-Hash/liars-dividend-GPT-image-2-photoreal), 80 images at 1024 × 1024 without C2PA), one image reaches z 195 with the 1254 template, probably a downscaled copy. The other 79 have the uncovered size 1024 × 1024 and do not trigger.

False positives (measured like `eval.py --ignore-aspect`, so every template on every image): none of 8,830 images without an OpenAI watermark (the 4,457 real images above, 1,873 images from Gemini, GPT-Image 1.5, FLUX.2, Qwen-Image, Z-Image, Midjourney 7 and other generators, and 1,500 real photos and 1,000 FLUX.2, Qwen-Image 2.1 and Nano Banana 2 images from the MLLM dataset). The highest z-score was 4.4, the standard deviation 1.0. Of 420 further OpenAI images from four Hugging Face datasets without C2PA evidence, 110 trigger, all GPT-Image 2 images from one dataset (110 of 120, mostly JPEGs at 900 × 1200); the 300 images of the other three datasets do not. In OpenFake, 14 of 36 GPT-Image 2 images (mostly 1024 × 1536) and none of 80 GPT-Image 1.5 images trigger.

### Limitations

- This is a reconstruction from public sample images, not Google's decoder. Only images from Gemini 3 Pro Image (Nano Banana Pro), Gemini 3.1 Flash Image and GPT-Image 2 (ChatGPT and API) are tested, other models are unverified.
- OpenAI images in sizes without their own template are found less reliably once they are downscaled and saved as JPEG (102 of 109 JPEGs at 900 × 1200, 14 of 30 OpenFake images at 1024 × 1536 and 1536 × 1024). Images at 1024 × 1024 without C2PA are almost never found, and there are no such images with a verified watermark. Every template is based on at most two sources. OpenAI images from before May 2026 carry no SynthID.
- Cropping destroys the signal completely, and WebP erases it in Gemini images. Gemini images that are downscaled and then saved as JPEG, as under the model's web conditions (1280 or 800 px, JPEG 80 or 75), are no longer detected, ChatGPT images in the template sizes still are. Images from social networks are therefore often not detected; then the model takes over.
- Real photos edited with Gemini carry SynthID as well and are flagged, those edited with ChatGPT probably too.
- The Google template is computed from the Hugging Face dataset [aoxo/reverse-synthid](https://huggingface.co/datasets/aoxo/reverse-synthid) (no license stated there), the OpenAI templates from [bigdatamark/synthid-research](https://huggingface.co/datasets/bigdatamark/synthid-research) (CC BY 4.0) and [winrisef/gpt-generated-image-collection](https://huggingface.co/datasets/winrisef/gpt-generated-image-collection) (MIT). For OpenAI, only images whose C2PA manifest records the watermark (`c2pa.watermarked.unbound`) count. All revisions are pinned; only averaged phases and weights are shipped, no images.

## Amazon Nova Canvas watermark

Measured with `tools/synthid/crossval.py amazon` on 75 Nova Canvas PNGs whose C2PA manifest names the model, from 10 GitHub repositories (512 × 512 to 1280 × 720): the template is built from nine repositories and tested on the images of the tenth, in turn for each repository (z ≥ 8).

| Nova Canvas images (75) | detected |
|---|---|
| originals (PNG) | 100 % |
| downscaled to 50 % or 33 % | 45 % or 0 % |
| JPEG 90 without or with chroma subsampling (4:4:4 or 4:2:0) | 93 % or 11 % |
| WebP 80 | 0 % |
| cropped to 95 % on all sides | 0 % |

The shipped template, applied to images from AWS blog posts that are not part of it: 57 of 58 PNGs from the Nova Canvas visual guide (all with seed 0, partly downscaled by the blog), all 13 PNGs from five other posts plus a virtual try-on result (z 11), none of 9 JPEGs with chroma subsampling.

False positives: none of 6,672 images without the Nova Canvas watermark (real photos from ImageNet, DOCCI, COCO and Reddit, YouTube thumbnails, paintings, charts, and images from Gemini, GPT-Image 1.5 and 2, FLUX.2, Qwen-Image, Z-Image, Midjourney 7 and other generators), nor any of 348 images from AWS posts and guides on Titan, among them 37 Titan images. The highest z-score was 4.9, the standard deviation 1.07.

### Limitations

- The signal sits close to the resolution limit of the original. Downscaling to a third, JPEG with chroma subsampling, WebP and cropping erase it, downscaling to half the size leaves about half detectable. In practice mostly unmodified PNGs from Nova Canvas are found. Unmodified Nova Canvas and Titan images also carry a C2PA manifest with `trainedAlgorithmicMedia`, which the metadata check reads first.
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

## ByteDance tile pattern

Measured on 8 images with the pattern at the original size: 7 Jimeng PNGs (936 × 1664) and one Doubao PNG (2048 × 2048) from [pagedMark](https://github.com/doofzoff/pagedMark).

| condition | detected | correlation |
|---|---|---|
| as delivered | 8 of 8 | 0.86–0.89 |
| JPEG 90 or 80 | 8 of 8 | 0.70–0.74 |
| JPEG 70 | 4 of 8 | 0.41–0.51 |

False positives: none of 9,093 images large enough for the check (real photos, Kling, Amazon Nova, OpenFake and Reddit AI images and others); the highest correlation was 0.16. Seedream 3, 4.5 and 5.0 images from APIs and datasets and Doubao images downloaded without the visible label carry no pattern (0 of 67).

### Limitations

- Only images at the original size and position: scaling and cropping erase the pattern. Images need at least 24 whole tiles of 112 × 96 pixels.

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

The misses as downloaded: two logos on graffiti and across the edge of a sheet of paper (0.82 and 0.77), one on wet asphalt (0.82), and two Reddit images that were cropped or letterboxed after generation, so the logo is no longer where the layout puts it. The Google SynthID template finds 8 of these 34 images.

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
- Logos with a model version such as "KLING AI 1.6", "可灵AI 2.1" or "KlingAI 3.0 Omni" are not covered; the GitHub sets contain too few examples per variant to measure positions. The label "可灵AI 3.0" of the Chinese site is checked with the [further visible marks](#further-visible-marks).
- Every image with this logo in this place is flagged, also a copy that a model painted into an image.
- The current layout is measured on 68 images from two GitHub repositories, most of them from a single project.

## Further visible marks

Measured with `tools/visible/eval.py --logo marks`; a hit needs a correlation of at least 0.6 for the Chinese labels, 0.65 for the current Grok marks and 0.8 for the 2024 Grok mark.

Real images with the mark, labeled by eye:

- **Doubao (13), Jimeng (7), Qwen (21), Kling "可灵AI 3.0" (4):** sample outputs in [NanmiCoder/OpenNoMark](https://github.com/NanmiCoder/OpenNoMark), in two output sizes each for Doubao and Qwen.
- **Grok "GROK xAI" (247):** Aurora outputs from January 2025 in [Rapidata/xAI_Aurora_t2i_human_preferences](https://huggingface.co/datasets/Rapidata/xAI_Aurora_t2i_human_preferences), 1024 × 768.
- **Grok "⊘ Grok" (5) and "GROK" with a slash (2):** public domain images on Wikimedia Commons and viral images from [WildAIGC-Bench](https://github.com/yuzj1997yzj/WildAIGC-Bench).

The Doubao and Jimeng templates come from alpha maps and are independent of these images. The others are averaged from them; left out of the average, each image still scores 0.97 or more for Kling, 0.83 or more for "⊘ Grok", 0.88 for "GROK" and a median of 0.96 for Qwen (minimum 0.57).

| Real images with the mark | Doubao (13) | Jimeng (7) | Qwen (21) | Kling (4) | GROK xAI (247) | ⊘ Grok (5) | GROK (2) |
|---|---|---|---|---|---|---|---|
| as downloaded | 12 | 7 | 21 | 4 | 221 (89.5 %) | 5 | 2 |
| JPEG 85 | 12 | 7 | 21 | 4 | 216 (87.4 %) | 5 | 2 |
| WebP 80 | 12 | 7 | 21 | 4 | 214 (86.6 %) | 5 | 2 |
| downscaled to 800 px width, JPEG 75 | 12 | 7 | 21 | 4 | 209 (84.6 %) | 5 | 2 |
| downscaled to 50 %, JPEG 80 | 12 | 6 | 21 | 4 | 194 (78.5 %) | 5 | 2 |
| downscaled to 33 %, JPEG 80 | 12 | 6 | 20 | 4 | 169 (68.4 %) | 4 | 2 |
| cropped to 95 % on all sides | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Missed marks sit on bright backgrounds, where the white text leaves little contrast; this hits Grok's small mark on Aurora's often bright scenes most. Of the 406 WildAIGC images, the check finds 5 Grok images (3 of them in the templates) and nothing else; it misses a screenshot of a Grok image and a very faint mark. Images with one provider's mark score at most 0.44 for the templates of the other providers, and the 1,091 Kling images of the Kling logo test at most 0.56.

False positives: none of 10,432 images without these marks (the images of the Kling logo test without duplicates and without 8 Aurora images, all of which carry the Grok mark and are found), as downloaded and after downscaling to 50 % with JPEG 80. The highest correlations are 0.52 for the Chinese labels, 0.53 for the current Grok marks and 0.70 for the 2024 Grok mark.

All seven marks together take about 9 times as long as the Kling check, mostly for the Grok scan.

### Limitations

- The test sets are small: 4 to 21 images per Chinese label, most from a single source, and 7 images of the two older Grok marks.
- The Chinese labels have to be in the size and place that GB 45438-2025 makes the providers use; cropping and added borders move them away. Other providers (Yuanbao, Baidu and others) and Kling's labels with other versions are not covered.
- Grok's marks are searched only between 4 % and 10 % of the long side, which covers originals with a long side of about 660 to 1,650 px (890 to 2,200 px for the 2024 mark). Marks narrower than 20 px are not searched.
- Every image with such a mark in this place is flagged, also a copy that a model painted into an image.

## Tested and not included

- **Ensembles with other models** (Detectra v3, [CommunityForensics-DeepfakeDet-ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT), [ai-vs-human-SigLIP-ONNX](https://huggingface.co/neilxxx/ai-vs-human-SigLIP-ONNX)): on top of the fine-tuned model, these models, alone or together, add no hits at the same false positive rate, but cost 0.7 s per image each (155 MB together).
- **WikiArt paintings as negatives:** a version fine-tuned with 4,500 WikiArt paintings instead of the Met artworks detected 2 to 3 percentage points more AI images under web conditions and flagged only 0.7 % of WikiArt paintings, but 9 to 14 % of the paintings from the Cleveland Museum of Art. It had learned WikiArt rather than paintings in general, and WikiArt's images are not licensed for training. Met images sampled from the whole collection instead of contiguous parts of it, and an average of the weights of both runs, detected no more AI images at 1 % flagged photos.
- **synthid-research images in training:** 813 images whose prompt number (in the file name) is not among the 415 prompts of the measured images. The model learned the style of the dataset, which uses the same prompts and settings for every generator: it detected 85 % instead of about 30 % of the measured synthid-research images, but not more of other images from these generators, and 8 percentage points fewer from OpenFake and Reddit.
- **Only the newest generators' images in the 15 OpenFake files**, without the rows from further files: 4 to 8 percentage points fewer AI images detected at 1 % flagged photos under web conditions.
- **Other weight formats:** 4-bit weights (block size 16, 19 MB) detect 3 to 4 percentage points fewer AI images under web conditions at 1 % flagged photos. fp16 (44 MB) detects as many as the shipped 8-bit weights (30 MB).
- **Five crops per image** (measured with the WikiArt version): center and four corners of the image downscaled to 640 px on the short side, taking the highest score. Under web conditions this detects 1 to 5 percentage points more AI images, but takes five times the compute and flags more camera photos (DOCCI up to 7 %) and edited subreddit photos (up to 5 %).
- **fev v13** ([angelhd25/ull-ai-image-detector](https://huggingface.co/angelhd25/ull-ai-image-detector), MIT), like Detectra a fine-tune of the Community Forensics ViT-S for a browser add-on: alone it detects 37–39 % of AI images under web conditions, but flags 7 % of paintings. In an ensemble with Detectra v3 it added no hits at no more than 5 % flagged paintings, but costs 88 MB and 0.7 s per image. It was also partly trained on the OpenFake validation split (CC BY-NC 4.0), whose generators and photo sources do not appear in the test split used here.
- **Other detectors from Hugging Face** (capcheck/ai-image-detection, ai-image-detect-distilled, SMOGY and others): many are trained on small datasets like CIFAKE and separate AI images under web conditions hardly or not at all better than chance (AUC 0.46 to 0.62). Larger models like SwinV2-Large detectors or SigLIP-DINOv2 ensembles would be too big for the add-on at 200 MB to 2 GB.
- **Amazon Titan Image Generator watermark:** Titan images (G1 and G1 v2, 35 images with C2PA from 12 GitHub repositories and 37 from an AWS blog post) share a phase pattern only when they were generated with the same seed (0 and 42 are common defaults). Images with random seeds from other repositories show none (z ≈ 0), so no fixed template is possible. Unmodified Titan images are recognized by their C2PA manifest.
- **Other visible marks:** Meta AI has shown no visible mark since Muse Image (July 2026), and there are no public examples of its older "Imagined with AI" label. Samsung Galaxy AI marks come with localized text, Hailuo (MiniMax) marks only videos, and Grok's paid plans and API add no mark. For Samsung and the older Meta label no real examples were found to measure positions and false positives.
- **Stability AI on Amazon Bedrock:** the newer models (SD3 Large, SD3.5 Large, Stable Image Ultra) write no `invisible-watermark` and show no fixed phase pattern in 434 PNGs from 61 GitHub repositories and 23 PNGs from AWS blog posts. Only the SDXL outputs among them carry the SDXL watermark and the EXIF fields of the Stability API.
