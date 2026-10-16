# Third-party notices

The source code in this repository is licensed under the [MIT License](LICENSE). A built add-on also contains the following third-party components, which keep their own licenses.

## Summary

| Component | Shipped in the add-on | License |
|---|---|---|
| ONNX Runtime Web | yes (`worker.js`, `ort/`) | MIT |
| Model weights `detectra-openfake` | yes (`models/`), downloaded by `npm run fetch-model` | CC BY-NC 4.0 |
| SynthID templates | yes (`synthid-templates/`) | derived data, see below |
| Gemini logo alpha map | yes (in `worker.js`) | MIT |

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
- [The Metropolitan Museum of Art Open Access](https://huggingface.co/datasets/metmuseum/openaccess): public domain paintings, drawings and prints, [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Changes: images larger than 2048 px were downscaled.
- [COCO](https://huggingface.co/datasets/detection-datasets/coco) train images: Flickr images under their individual licenses
- [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) train split, GPL-3.0

## SynthID templates

The files in [`src/background/synthid-templates/`](src/background/synthid-templates/) contain spectral phases and weights averaged over many watermarked images. They contain no images and no image can be reconstructed from them. Sources, revisions and image counts are recorded in [`synthid.json`](synthid.json).

- [bigdatamark/synthid-research](https://huggingface.co/datasets/bigdatamark/synthid-research) by Mark Adams. Curation licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the dataset states that the images themselves carry no copyright. Changes: images reduced to averaged FFT phases.
- [winrisef/gpt-generated-image-collection](https://huggingface.co/datasets/winrisef/gpt-generated-image-collection), MIT License
- [aoxo/reverse-synthid](https://huggingface.co/datasets/aoxo/reverse-synthid), no license stated
- Amazon Nova Canvas images from 10 GitHub repositories, pinned to the commits in [`tools/synthid/amazon_sources.json`](tools/synthid/amazon_sources.json): aws-samples/amazon-nova-samples, aws-samples/sample-amazon-nova-prompt-examples, aws-samples/sample-devops-agent-demo, camila-hinojosa-anez/practice-speaking-with-sonic and niravdd/ArtSmoker (MIT-0), baochen10luo/stagenthand (MIT), asha1-alt/AITraining-1, ExamProCo/aws-storage-genai-workshop, leegilmorecode/llm-image-assertions and stellarshenson/stellars-jupyterhub-ds (no license stated). Changes: images reduced to averaged FFT phases.

## Gemini logo alpha map

[`src/background/gemini-logo.json`](src/background/gemini-logo.json) contains the alpha map of the visible Gemini logo from [GeminiWatermarkTool](https://github.com/allenk/GeminiWatermarkTool) (`assets/embedded_assets.hpp`, `bg_96_png`, commit `7c6a99f`), downscaled from 96 × 96 to 24 × 24 px by [`tools/visible/gemini_logo.py`](tools/visible/gemini_logo.py). The unit test fixtures in `test/unit/fixtures/visible-watermark/` are overlaid with the 48 px map of the older layout and the 96 px map of the newer one (downscaled to 48 px) from the same file.

```
MIT License

Copyright (c) 2024 AllenK (Kwyshell)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Trademarks

SynthID, Gemini and Imagen are trademarks of Google LLC. OpenAI, ChatGPT and GPT-Image are trademarks of OpenAI. Amazon, Amazon Bedrock, Amazon Nova and Amazon Titan are trademarks of Amazon.com, Inc. or its affiliates. Stability AI is a trademark of Stability AI Ltd. C2PA is a trademark of the Coalition for Content Provenance and Authenticity. They are used here only to describe what the add-on detects. This project is not affiliated with or endorsed by any of them.
