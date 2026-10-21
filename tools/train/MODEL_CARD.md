---
license: cc-by-nc-4.0
library_name: onnx
pipeline_tag: image-classification
base_model: TensorFold/detectra-v1
datasets:
  - ComplexDataLab/OpenFake
  - laionmobile/laion-mobile
tags:
  - ai-image-detection
  - deepfake-detection
  - onnx
---

# Detectra fine-tuned on OpenFake

AI-generated image detector bundled with the Firefox add-on [AI Watermark](https://github.com/s-gehring/ai-watermark-addon). A ViT-S/16 at 384 px binary classifier, exported to ONNX for ONNX Runtime Web with the weights quantized to 8 bits (MatMulNBits, block size 16, asymmetric).

## Model

- **Base model:** [Detectra v3](https://huggingface.co/TensorFold/detectra-v1) (MIT), itself a fine-tune of the [Community Forensics ViT-S](https://huggingface.co/OwensLab/commfor-model-384) (MIT)
- **Fine-tuning:** 3 epochs on 15 files of the `core/train` split of [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (about 60,000 images, half real, half from many generators), plus about 3,500 paintings, 500 drawings and 500 prints from [The Met's Open Access collection](https://huggingface.co/datasets/metmuseum/openaccess) (CC0), 5,900 COCO photos and ChartQA charts as negatives. Images from the newest generators (FLUX.2, Qwen-Image, Nano Banana 2, Seedream 4 and others) are weighted higher, among them 6,768 from 319 further files of the train split, together with 1,034 Pexels photos from these files as negatives. Every training image is randomly downscaled and recompressed as JPEG to match conditions on web pages. Batch 48, learning rate 3e-5 with layer decay 0.75, drop path 0.1.
- **Self-generated images:** 10,348 images generated locally with the captions of real OpenFake training photos as prompts: 3,624 from [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) and 245 from [FLUX.2 klein base 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) (Apache 2.0), 6,200 from [Qwen-Image 2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) and 133 from [Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (Qwen Research License, non-commercial), 146 from [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) (Apache 2.0), in the resolutions and file formats (PNG, JPEG, WebP) of the online services.
- **Modern photos:** 19,463 smartphone and recent camera photos as negatives, 11,497 from [LAION-Mobile](https://huggingface.co/datasets/laionmobile/laion-mobile) (metadata CC BY 4.0, images as URLs from re-LAION-5B) and 7,966 from [Unsplash Lite](https://unsplash.com/data) (Unsplash License). The 5,314 photos that the previous version of this model scored at 0.5 or more are weighted four times as high as the rest.
- **Training code:** [`tools/train/`](https://github.com/s-gehring/ai-watermark-addon/tree/main/tools/train)

## Usage

- **Input:** `pixel_values`, float32, 1 × 3 × 384 × 384. Resize the shortest edge to 440 px (bilinear), center-crop 384 px, scale to [0, 1], normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- **Output:** `logits`, float32, 1 × 1. `sigmoid(logit)` is the raw probability that the image is AI-generated.

The raw probability is poorly calibrated for real photos. The add-on applies Platt scaling (`sigmoid(0.5488 * logit + 1.0703)`), scores images between 0.5 and 0.95 again on center crops with a shorter edge of 576 and 768 px, keeps the highest score and flags images from 0.95, which marks less than 1 % of real photos under web conditions.

## Evaluation

Measured on held-out test data (OpenFake test split with generators and photo sources not seen in training, OpenFake Reddit test set, newer generators, COCO, YouTube thumbnails, paintings, charts), downscaled to 1280 px or 800 px and recompressed as JPEG:

- AUC 0.97
- 71–74 % of AI images detected at 1 % flagged real photos (Detectra v3 without fine-tuning: 33–34 %), 77 % with the add-on's second pass on larger crops
- 68 % of the newest generators' images detected at the add-on's threshold (FLUX.2, GPT-Image 2, Qwen-Image 2.1, Gemini 3.1 Flash Image)
- at the add-on's threshold: 0 % of paintings from the Cleveland Museum of Art, 5–7 % of WikiArt paintings and 0 % of charts flagged

Full results and limitations: [docs/accuracy.md](https://github.com/s-gehring/ai-watermark-addon/blob/main/docs/accuracy.md).

The output is a hint, not proof. Do not use it as the sole basis for decisions about people or their work.

## License

The weights are trained on OpenFake, which is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), and are released under the same license: non-commercial use only.

Attribution: OpenFake by Victor Livernoche, Akshatha Arodi, Andreea Musulan, Zachary Yang, Adam Salvail, Gaétan Marceau Caron, Jean-François Godbout and Reihaneh Rabbany ([arXiv:2509.09495](https://arxiv.org/abs/2509.09495)). Detectra v3 by TensorFold. Community Forensics by Jeongsoo Park and Andrew Owens.

Improved using Qwen: part of the self-generated training images come from Qwen-Image 2.1 and Qwen-Image 2.1 Turbo under the Qwen Research License.

The other fine-tuning data come with their own terms: The Met's Open Access images are CC0, COCO images are under their individual Flickr licenses, ChartQA is licensed under GPL-3.0, the LAION-Mobile metadata under CC BY 4.0 (the photos are scraped web images without individual licenses, like the LAION photos in OpenFake), and Unsplash photos under the Unsplash License. FLUX.2 klein and Z-Image are licensed under Apache 2.0.
