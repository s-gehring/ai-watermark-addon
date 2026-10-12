---
license: cc-by-nc-4.0
library_name: onnx
pipeline_tag: image-classification
base_model: TensorFold/detectra-v1
datasets:
  - ComplexDataLab/OpenFake
tags:
  - ai-image-detection
  - deepfake-detection
  - onnx
---

# Detectra fine-tuned on OpenFake

AI-generated image detector bundled with the Firefox add-on [AI Watermark](https://github.com/s-gehring/ai-watermark-addon). A ViT-S/16 at 384 px binary classifier, exported to fp16 ONNX for ONNX Runtime Web.

## Model

- **Base model:** [Detectra v3](https://huggingface.co/TensorFold/detectra-v1) (MIT), itself a fine-tune of the [Community Forensics ViT-S](https://huggingface.co/OwensLab/commfor-model-384) (MIT)
- **Fine-tuning:** 3 epochs on 15 files of the `core/train` split of [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (about 60,000 images, half real, half from many generators), plus about 3,500 paintings, 500 drawings and 500 prints from [The Met's Open Access collection](https://huggingface.co/datasets/metmuseum/openaccess) (CC0), 5,900 COCO photos and ChartQA charts as negatives. Every training image is randomly downscaled and recompressed as JPEG to match conditions on web pages. Batch 48, learning rate 3e-5 with layer decay 0.75, drop path 0.1, EMA 0.999.
- **Training code:** [`tools/train/`](https://github.com/s-gehring/ai-watermark-addon/tree/main/tools/train)

## Usage

- **Input:** `pixel_values`, float32, 1 × 3 × 384 × 384. Resize the shortest edge to 440 px (bilinear), center-crop 384 px, scale to [0, 1], normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- **Output:** `logits`, float32, 1 × 1. `sigmoid(logit)` is the raw probability that the image is AI-generated.

The raw probability is poorly calibrated for real photos. The add-on applies Platt scaling (`sigmoid(0.4371 * logit + 1.4406)`) and flags images from 0.92, which marks about 1 % of real photos under web conditions.

## Evaluation

Measured on held-out test data (OpenFake test split with generators and photo sources not seen in training, OpenFake Reddit test set, newer generators, COCO, YouTube thumbnails, paintings, charts), downscaled to 1280 px or 800 px and recompressed as JPEG:

- AUC 0.91–0.93
- 46–47 % of AI images detected at 1 % flagged real photos (Detectra v3 without fine-tuning: 33–34 %)
- at the add-on's threshold: 0 % of paintings from the Cleveland Museum of Art, 5–7 % of WikiArt paintings and 0 % of charts flagged

Full results and limitations: [docs/accuracy.md](https://github.com/s-gehring/ai-watermark-addon/blob/main/docs/accuracy.md).

The output is a hint, not proof. Do not use it as the sole basis for decisions about people or their work.

## License

The weights are trained on OpenFake, which is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), and are released under the same license: non-commercial use only.

Attribution: OpenFake by Victor Livernoche, Akshatha Arodi, Andreea Musulan, Zachary Yang, Adam Salvail, Gaétan Marceau Caron, Jean-François Godbout and Reihaneh Rabbany ([arXiv:2509.09495](https://arxiv.org/abs/2509.09495)). Detectra v3 by TensorFold. Community Forensics by Jeongsoo Park and Andrew Owens.

The other fine-tuning data come with their own terms: The Met's Open Access images are CC0, COCO images are under their individual Flickr licenses, and ChartQA is licensed under GPL-3.0.
