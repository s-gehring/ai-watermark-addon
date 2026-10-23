# Third-party notices

The source code in this repository is licensed under the [MIT License](LICENSE). A built add-on also contains the following third-party components, which keep their own licenses.

## Summary

| Component | Shipped in the add-on | License |
|---|---|---|
| ONNX Runtime Web | yes (`worker.js`, `ort/`) | MIT |
| Model weights `detectra-openfake` | yes (`models/`), downloaded by `npm run fetch-model` | CC BY-NC 4.0 |
| SynthID templates | yes (`synthid-templates/`) | derived data, see below |
| Gemini logo alpha map | yes (in `worker.js`) | MIT |
| Kling logo templates | yes (in `worker.js`) | derived data, see below |

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
- 10,348 images generated locally with [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), [FLUX.2 klein base 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) and [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) (Apache 2.0) and with [Qwen-Image 2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) and [Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (Qwen Research License, non-commercial). Improved using Qwen.

## SynthID templates

The files in [`src/background/synthid-templates/`](src/background/synthid-templates/) contain spectral phases and weights averaged over many watermarked images. They contain no images and no image can be reconstructed from them. Sources, revisions and image counts are recorded in [`synthid.json`](synthid.json).

- [bigdatamark/synthid-research](https://huggingface.co/datasets/bigdatamark/synthid-research) by Mark Adams. Curation licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the dataset states that the images themselves carry no copyright. Changes: images reduced to averaged FFT phases.
- [winrisef/gpt-generated-image-collection](https://huggingface.co/datasets/winrisef/gpt-generated-image-collection), MIT License, files pinned in [`tools/synthid/openai_sources.json`](tools/synthid/openai_sources.json)
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

## Kling logo templates

[`src/background/kling-logo.json`](src/background/kling-logo.json) contains the average contrast of the two visible Kling logos (48 × 12 and 64 × 18 values), computed by [`tools/visible/kling_logo.py`](tools/visible/kling_logo.py) from the bottom right corners of real Kling images. Averaged over many backgrounds, only the logo remains. Sources:

- older logo: 1,005 images of [nyuuzyou/klingai](https://huggingface.co/datasets/nyuuzyou/klingai) (revision `4ae5e43`), CC0 1.0
- current logo: 68 images on Kling's servers that are linked from [andreathar/Movies_Projects](https://github.com/andreathar/Movies_Projects) (commit `35f8447`, 65 images) and [newmemomedo2018-afk/cgi-generator-mongo](https://github.com/newmemomedo2018-afk/cgi-generator-mongo) (commit `09dcddd`, 3 images), no license stated

The unit test fixture `test/unit/fixtures/visible-watermark/kling-2024.jpg` is image 130211779 of nyuuzyou/klingai, downscaled to 512 × 512 px. `kling-2026.jpg` is a synthetic background overlaid with the template of the current logo.

## Further visible mark templates

[`src/background/visible-marks.json`](src/background/visible-marks.json) contains the templates of the visible Doubao, Jimeng, Qwen, Kling (Chinese) and Grok marks (48 values wide each), computed by [`tools/visible/marks.py`](tools/visible/marks.py). Sources:

- Doubao and Jimeng: `doubao_alpha.png` and `jimeng_alpha.png` from [remove-ai-watermarks](https://github.com/wiltodelta/remove-ai-watermarks) by wiltodelta (`src/remove_ai_watermarks/assets/`, commit `5677c22`), [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), downscaled to 48 values wide and converted to contrast
- Qwen and Kling (Chinese): average contrast of 21 Qwen and 4 Kling images in [NanmiCoder/OpenNoMark](https://github.com/NanmiCoder/OpenNoMark) (commit `9455d2d`), CC BY-NC-SA 4.0
- Grok "GROK xAI": average contrast of 247 images of [Rapidata/xAI_Aurora_t2i_human_preferences](https://huggingface.co/datasets/Rapidata/xAI_Aurora_t2i_human_preferences) (revision `8f32b40`), CDLA-Permissive-2.0
- Grok "⊘ Grok" and "GROK" with a slash: average contrast of 4 public domain images on Wikimedia Commons ("Great Sea Serpent Andersen literary fairy tale.jpg", "Man using laptop during sunset (Grok Aurora 2025).webp", "Tesla Cybertruck on a Snowy Landscape Under the Aurora Borealis (Grok Aurora 2025).webp", "Astronaut Riding a Horse (Grok Aurora).webp") and 3 images of [WildAIGC-Bench](https://github.com/yuzj1997yzj/WildAIGC-Bench) (commit `96bb4e8`), no license stated for the images

The unit test fixture `test/unit/fixtures/visible-watermark/grok-aurora.jpg` is image `21eb1bb54253987e` (shard `train-00010`) of the Rapidata dataset, downscaled to 512 × 384 px. `doubao.jpg`, `jimeng.jpg`, `qwen.jpg`, `kling-cjk.jpg`, `grok.jpg` and `grok-2024.jpg` are synthetic backgrounds overlaid with these templates.

## Trademarks

SynthID, Gemini and Imagen are trademarks of Google LLC. OpenAI, ChatGPT and GPT-Image are trademarks of OpenAI. Amazon, Amazon Bedrock, Amazon Nova and Amazon Titan are trademarks of Amazon.com, Inc. or its affiliates. Kling is a trademark of Kuaishou Technology. Doubao and Jimeng are trademarks of ByteDance, Qwen of Alibaba and Grok of xAI. Stability AI is a trademark of Stability AI Ltd. C2PA is a trademark of the Coalition for Content Provenance and Authenticity. They are used here only to describe what the add-on detects. This project is not affiliated with or endorsed by any of them.
