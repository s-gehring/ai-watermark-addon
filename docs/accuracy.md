# Accuracy

## Model

Measured with `npm run eval` on 4,906 images. One half is used for calibration, the numbers below come from the other half.

- **AI images:** 300 from the test split of [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (18 generators up to 2026), 188 from OpenFake's Reddit test set (AI subreddits) and 600 from newer generators in [bigdatamark/synthid-research](https://huggingface.co/datasets/bigdatamark/synthid-research) (150 each from Gemini 3.1 Flash Image, Qwen-Image 2.1, FLUX.2 and GPT-Image 2)
- **real photos:** 943 from OpenFake (ImageNet, DOCCI), 375 from photography subreddits (OpenFake Reddit test set), 1,000 from [COCO](https://huggingface.co/datasets/detection-datasets/coco) and 600 YouTube video thumbnails from the [AudioSet](https://research.google.com/audioset/) eval set (uploaded until early 2017, 300 with and 300 without music), cropped to 16:9 the way YouTube shows them in lists
- **graphics and art:** 300 charts from [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA), 300 paintings from [WikiArt](https://huggingface.co/datasets/huggan/wikiart) and 300 paintings from the [Cleveland Museum of Art Open Access](https://openaccess-api.clevelandart.org/) collection (CC0)

Every image is scored three times: as the original file and the way images typically arrive on web pages, downscaled to at most 1280 px or 800 px and saved as JPEG with quality 80 or 75. Calibration and default threshold are chosen so that about 1 % of real photos are flagged under web conditions.

At the default threshold of 92 %, with the previously bundled Community Forensics model at its default threshold of 95 % in parentheses:

| | Original file | Web 1280 px | Web 800 px |
|---|---|---|---|
| AI images detected (OpenFake) | 67 % (40 %) | 65 % (30 %) | 62 % (27 %) |
| AI images detected (Reddit) | 65 % (31 %) | 65 % (28 %) | 54 % (29 %) |
| AI images detected (newer generators) | 57 % (43 %) | 33 % (14 %) | 40 % (18 %) |
| real photos flagged (all) | 1.1 % (0.8 %) | 1.3 % (0.8 %) | 1.1 % (0.8 %) |
| real photos flagged (photography subreddits) | 2.7 % (3.2 %) | 2.7 % (1.6 %) | 2.7 % (2.7 %) |
| real photos flagged (DOCCI camera photos) | 5.0 % (2.5 %) | 6.3 % (1.9 %) | 3.8 % (0.6 %) |
| YouTube thumbnails flagged | 0.7 % (0 %) | 0.7 % (0 %) | 1.4 % (0.3 %) |
| paintings flagged (Cleveland Museum of Art) | 0 % (15 %) | 0 % (13 %) | 0 % (8 %) |
| paintings flagged (WikiArt) | 6.6 % (11 %) | 5.3 % (11 %) | 4.6 % (11 %) |
| charts flagged | 0 % (0 %) | 0 % (0 %) | 0 % (0 %) |

Other thresholds (web 1280 px, all AI images against real photos):

| Threshold | AI images detected | real photos flagged |
|---|---|---|
| 50 % | 78 % | 13 % |
| 70 % | 65 % | 5.6 % |
| 80 % | 59 % | 3.5 % |
| 90 % | 50 % | 1.6 % |
| **92 % (default)** | **47 %** | **1.3 %** |
| 97 % | 37 % | 0.1 % |

Comparison with other local models under web conditions (1280 px and 800 px), AI images detected at exactly 1 % flagged photos:

| Model | AUC | AI images detected |
|---|---|---|
| **Detectra fine-tuned on OpenFake (bundled)** | **0.91–0.93** | **46–47 %** |
| [Detectra v3](https://huggingface.co/TensorFold/detectra-v1) without fine-tuning (fp16) | 0.89–0.90 | 33–34 % |
| [fev v13](https://huggingface.co/angelhd25/ull-ai-image-detector) (fp32, 88 MB), see below | 0.90 | 37–39 % |
| Community Forensics (int8, previously bundled) | 0.85 | 23 % |
| Community Forensics without quantization (fp32, four times the size) | 0.89 | 25 % |
| [SigLIP detector](https://huggingface.co/Ateeqq/ai-vs-human-image-detector) (int8) | 0.71 | 8–9 % |
| SMOGY, ViT deepfake classifier | 0.55–0.62 | 0 % |
| [capcheck/ai-image-detection](https://huggingface.co/onnx-community/ai-image-detection-ONNX), [ai-image-detect-distilled](https://huggingface.co/onnx-community/ai-image-detect-distilled-ONNX) | 0.46–0.48 | 0–1 % |

The SigLIP detector often considers YouTube video thumbnails AI-generated, including those of well-known old music videos: the original PyTorch model outputs more than 99 % for 23 of 26 tested thumbnails, so it is not caused by quantization or preprocessing. At 1 % false positives on photos overall, it flags 2–4 % of YouTube thumbnails.

### Limitations

- The badge is a hint, not proof. The default threshold is tuned for few false positives; in exchange, about half of the AI images without metadata or watermark stay unmarked. To catch more, lower the threshold in the settings and accept more false positives.
- The model is weaker on the newest generators. Under web conditions (1280 px) it finds 47 of 78 FLUX.2 images, 21 of 70 GPT-Image 2 images, 11 of 73 Qwen-Image 2.1 images and 23 of 85 Gemini images; Community Forensics found 14 to 18 % of these, but only 1 to 3 of the Gemini images. As original files it detects 57 %, so compression costs particularly much here. Gemini images are additionally found by the SynthID check.
- The model flags high-quality camera photos like those from DOCCI more often than Community Forensics, 4 to 6 % instead of 1 to 2 % under web conditions, and YouTube thumbnails at about 1 % instead of almost never. Heavily edited photos from photography subreddits are flagged about as often (3 % instead of 2 to 3 %). 3D renderings are not in the test set.
- The fine-tuning uses public domain artworks from the Met as negatives, mostly from before 1900. The paintings from the Cleveland Museum of Art are similar and are not flagged at all, WikiArt contains more modern and abstract art and is flagged at 5 to 7 %. Detectra, the base model, was itself trained with WikiArt paintings.
- The fine-tuning uses the OpenFake train split, COCO train and Met artworks, none of which appear in the measurement. The measurement uses the test split with other generators and other photo sources (ImageNet and DOCCI instead of LAION and Pexels). The checkpoint was, however, selected based on this measurement (out of a few variants), so the numbers are slightly optimistic.
- Compression and downscaling cost detection (AUC 0.95 as original files, 0.91 to 0.93 under web conditions).
- Calibration and measurement use separate images but the same sources. Sites with a different image mix or their own compression may differ.

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

## Tested and not included

- **Ensembles with other models** (Detectra v3, [CommunityForensics-DeepfakeDet-ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT), [ai-vs-human-SigLIP-ONNX](https://huggingface.co/neilxxx/ai-vs-human-SigLIP-ONNX)): on top of the fine-tuned model, these models, alone or together, add no hits at the same false positive rate, but cost 0.7 s per image each (155 MB together).
- **WikiArt paintings as negatives:** a version fine-tuned with 4,500 WikiArt paintings instead of the Met artworks detected 2 to 3 percentage points more AI images under web conditions and flagged only 0.7 % of WikiArt paintings, but 9 to 14 % of the paintings from the Cleveland Museum of Art. It had learned WikiArt rather than paintings in general, and WikiArt's images are not licensed for training. Met images sampled from the whole collection instead of contiguous parts of it, and an average of the weights of both runs, detected no more AI images at 1 % flagged photos.
- **Five crops per image** (measured with the WikiArt version): center and four corners of the image downscaled to 640 px on the short side, taking the highest score. Under web conditions this detects 1 to 5 percentage points more AI images, but takes five times the compute and flags more camera photos (DOCCI up to 7 %) and edited subreddit photos (up to 5 %).
- **fev v13** ([angelhd25/ull-ai-image-detector](https://huggingface.co/angelhd25/ull-ai-image-detector), MIT), like Detectra a fine-tune of the Community Forensics ViT-S for a browser add-on: alone it detects 37–39 % of AI images under web conditions, but flags 7 % of paintings. In an ensemble with Detectra v3 it added no hits at no more than 5 % flagged paintings, but costs 88 MB and 0.7 s per image. It was also partly trained on the OpenFake validation split (CC BY-NC 4.0), whose generators and photo sources do not appear in the test split used here.
- **Other detectors from Hugging Face** (capcheck/ai-image-detection, ai-image-detect-distilled, SMOGY and others): many are trained on small datasets like CIFAKE and separate AI images under web conditions hardly or not at all better than chance (AUC 0.46 to 0.62). Larger models like SwinV2-Large detectors or SigLIP-DINOv2 ensembles would be too big for the add-on at 200 MB to 2 GB.
- **Amazon Titan Image Generator watermark:** Titan images (G1 and G1 v2, 35 images with C2PA from 12 GitHub repositories and 37 from an AWS blog post) share a phase pattern only when they were generated with the same seed (0 and 42 are common defaults). Images with random seeds from other repositories show none (z ≈ 0), so no fixed template is possible. Unmodified Titan images are recognized by their C2PA manifest.
- **Stability AI on Amazon Bedrock:** the newer models (SD3 Large, SD3.5 Large, Stable Image Ultra) write no `invisible-watermark` and show no fixed phase pattern in 434 PNGs from 61 GitHub repositories and 23 PNGs from AWS blog posts. Only the SDXL outputs among them carry the SDXL watermark and the EXIF fields of the Stability API.
- **Kakao (Kanana):** Kakao [announced in May 2026](https://www.kakaocorp.com/page/detail/12040?lang=ENG) that it would add SynthID to its image model Kanana-Kollage and video model Kanana-Kinema from the second half of 2026, starting with KakaoTalk's video feature Kanana Template. No Kakao image with verifiable provenance could be found to test: Kanana-Kollage was only offered in the standalone Kanana app, which shut down on 15 October 2026, and none of its images are on Hugging Face, Wikimedia Commons or GitHub. KakaoTalk's own image generation uses ChatGPT, so those images carry OpenAI's watermark. The add-on does not check videos.
