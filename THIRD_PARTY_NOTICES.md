# Third-party notices

The source code in this repository is licensed under the [MIT License](LICENSE). A built add-on also contains the following third-party components, which keep their own licenses.

## Summary

| Component | Shipped in the add-on | License |
|---|---|---|
| ONNX Runtime Web | yes (`worker.js`, `ort/`) | MIT |
| Model weights `detectra-openfake` | yes (`models/`), downloaded by `npm run fetch-model` | CC BY-NC 4.0 |
| SynthID templates | yes (`synthid-templates/`) | derived data, see below |

Because the bundled model weights are licensed for non-commercial use only, **a built add-on as a whole may only be used and redistributed non-commercially**. To use the code commercially, replace the model with one whose license allows it (see [docs/development.md](docs/development.md#evaluate-or-swap-the-model)).

## ONNX Runtime Web

Copyright (c) Microsoft Corporation. Licensed under the MIT License.
<https://github.com/microsoft/onnxruntime>

## Model weights

`detectra-openfake` is a fine-tune of Detectra v3, produced with [`tools/train/`](tools/train/). The weights are published at <https://huggingface.co/s-gehring/detectra-openfake> under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

Base models:

- [Detectra v3](https://huggingface.co/TensorFold/detectra-v1) by TensorFold, MIT License
- [Community Forensics ViT-S](https://huggingface.co/OwensLab/commfor-model-384) by Jeongsoo Park and Andrew Owens, MIT License

Fine-tuning data (not shipped):

- [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) by Victor Livernoche, Akshatha Arodi, Andreea Musulan, Zachary Yang, Adam Salvail, Gaétan Marceau Caron, Jean-François Godbout and Reihaneh Rabbany (Complex Data Lab), [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Used: 15 files of the `core/train` split. Changes: images were randomly downscaled and recompressed as JPEG during training; the dataset itself is not redistributed.
- [WikiArt](https://huggingface.co/datasets/huggan/wikiart) (huggan/wikiart): images © their original authors, no license stated; the dataset card restricts use to non-commercial research
- [COCO](https://huggingface.co/datasets/detection-datasets/coco) train images: Flickr images under their individual licenses
- [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) train split, GPL-3.0

## SynthID templates

The files in [`src/background/synthid-templates/`](src/background/synthid-templates/) contain spectral phases and weights averaged over many watermarked images. They contain no images and no image can be reconstructed from them. Sources, revisions and image counts are recorded in [`synthid.json`](synthid.json).

- [bigdatamark/synthid-research](https://huggingface.co/datasets/bigdatamark/synthid-research) by Mark Adams. Curation licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the dataset states that the images themselves carry no copyright. Changes: images reduced to averaged FFT phases.
- [winrisef/gpt-generated-image-collection](https://huggingface.co/datasets/winrisef/gpt-generated-image-collection), MIT License
- [aoxo/reverse-synthid](https://huggingface.co/datasets/aoxo/reverse-synthid), no license stated

## Trademarks

SynthID, Gemini and Imagen are trademarks of Google LLC. OpenAI, ChatGPT and GPT-Image are trademarks of OpenAI. C2PA is a trademark of the Coalition for Content Provenance and Authenticity. They are used here only to describe what the add-on detects. This project is not affiliated with or endorsed by any of them.
