# Development

Requirements: Node 22 and Firefox. Python 3 only for the tools below.

The evaluation, training and template tools download third-party datasets and YouTube thumbnails for local use. They are not part of this repository; their own licenses and terms apply to you when you run the tools.

```bash
npm ci
npm run fetch-model   # downloads the model to vendor/models/ and verifies its SHA-256
npm run build         # builds to dist/
npm start             # starts Firefox with the add-on (web-ext run)
```

| Command | Purpose |
|---|---|
| `npm run watch` | rebuild on changes |
| `npm test` | unit tests (Vitest) |
| `npm run test:e2e` | builds and tests the add-on in headless Firefox (Selenium/geckodriver) |
| `npm run typecheck` | TypeScript check |
| `npm run lint` | `web-ext lint` on `dist/`, fails on warnings |
| `npm run package` | creates the ZIP in `web-ext-artifacts/` |

## Release

Release Firefox only installs add-ons signed by Mozilla. Without a signature, `dist/` can be loaded temporarily via `about:debugging` → "This Firefox" → "Load Temporary Add-on…" (pick `dist/manifest.json`), or permanently in Firefox Developer Edition or Nightly with `xpinstall.signatures.required = false`.

Releases are listed on addons.mozilla.org (AMO) and also published as GitHub releases by [`.github/workflows/release.yml`](../.github/workflows/release.yml) and [`.github/workflows/publish.yml`](../.github/workflows/publish.yml). To release:

1. Raise `version` in `package.json`, write the release notes into [`RELEASE_NOTES.md`](../RELEASE_NOTES.md) and merge the change into `main`.
2. Tag the merge commit and push the tag:

   ```bash
   git tag v0.2.0 origin/main
   ```

   ```bash
   git push origin v0.2.0
   ```

The release workflow tests and lints the build, uploads it together with the source code and [`amo-metadata.json`](../amo-metadata.json) (categories, support URL, license) to AMO. The release notes on AMO come from `RELEASE_NOTES.md` if it changed since the previous tag, otherwise they list the subjects of all commits since then, without PR numbers and without the `Release x.y.z` commit. The workflow then creates a draft GitHub release with the same notes. AMO reviews the version, which can take a few days. Every night the publish workflow asks AMO about each draft release. Once the version is approved, it attaches the signed `.xpi` to the release, publishes it and marks the highest published release as latest. To check right away, start the publish workflow by hand under Actions → Publish → "Run workflow". If AMO disables a version and a newer release exists, for example because the newer version was uploaded during the review, the workflow deletes the draft release and keeps the tag. A disabled version without a newer release, for example after a rejection, makes the workflow fail until its draft release is deleted.

Firefox updates the add-on through AMO, also when it was installed from the `.xpi` of a GitHub release.

The package is about 38 MB (model, WASM runtime and SynthID templates), below the AMO limit of 200 MB.

## Evaluate or swap the model

```bash
python3 -m venv .venv && .venv/bin/pip install pyarrow fsspec aiohttp huggingface_hub pillow
.venv/bin/python tools/eval/fetch_samples.py
npm run eval -- bundled --calibrate --per-generator
npm run eval -- all
npm run eval -- members --members=detectra-openfake,angelhd25/ull-ai-image-detector   # fit and measure any combination
```

`fetch_samples.py` reads only single Parquet row groups from OpenFake, COCO, ChartQA and WikiArt and 600 files from bigdatamark/synthid-research, plus 600 YouTube thumbnails from `i.ytimg.com` and 300 paintings from the Cleveland Museum of Art API (about 5 GB download instead of the full datasets, 3.3 GB on disk). Test data and raw scores always live in the main checkout under `tools/eval/.cache/`, even when the scripts run in a git worktree, so they survive deleting the worktree.

`npm run eval` stores the raw scores per model in `tools/eval/.cache/scores/`; later runs only score new images. For the bundled model it also scores the extra views of the `cascade` in `model.json` for the images whose score falls in its band. `--calibrate` uses the calibration half to choose the weighting of several bundled models (as many detected AI images as possible at no more than 5 % flagged paintings), fits Platt scaling and picks the threshold for 1 % flagged photos under web conditions. The resulting values go into [`model.json`](../model.json) (`calibration`, `threshold`). When the calibration changes, the add-on resets a user-chosen threshold to the new default, because the percentages mean something else afterwards.

## Train the model

Needs a CUDA GPU with 12 GB (measured on an RTX 4070 Ti) and about 130 GB of disk space.

```bash
.venv/bin/pip install torch timm safetensors pyarrow pillow numpy huggingface_hub onnx onnxruntime tokenizers
.venv/bin/python tools/train/fetch_data.py   # Detectra weights, 16 OpenFake files, single rows from 319 more and negatives to tools/train/.cache/
.venv/bin/python tools/train/train.py --name photo --modern-photo-weight 0.5 --hard-negative-weight 2   # 3 epochs, about 1 h
.venv/bin/python tools/train/train.py --name art   # 3 epochs, about 1 h
.venv/bin/python tools/train/route_clip.py path/to/clip-vit-base-patch16   # p(art) per training image -> tools/train/.cache/router.json
.venv/bin/python tools/train/distill.py tools/train/.cache/runs/{photo,art}/epoch3.safetensors --photo-platt W,B --art-platt W,B --name distill   # 2 epochs, about 30 min
.venv/bin/python tools/train/export_onnx.py tools/train/.cache/runs/distill/epoch2.safetensors   # -> vendor/models/detectra-openfake/model.onnx
```

`route_clip.py` reads the ONNX files and the tokenizer of [Xenova/clip-vit-base-patch16](https://huggingface.co/Xenova/clip-vit-base-patch16). `--photo-platt` and `--art-platt` are the Platt parameters of the two models, from `npm run eval -- bundled --calibrate` with each of them exported and bundled.

The datasets are pinned to fixed revisions. The rows taken from the other OpenFake files are listed in [`openfake_rows.json`](../tools/train/openfake_rows.json); `fetch_data.py` reads only the pages of the image column that hold them (about 16 GB). Self-generated images, modern photos, mined hard negatives and the artworks with their AI counterparts are not downloaded; the scripts add Parquet files from `selfgen/`, `modern-photos/` and `hard-negatives/` in the data folder when they exist, `distill.py` also from `art-real/` and `art-fake/`. The last of the 16 OpenFake files is held out (`distill.py` uses the first three and holds out the fourth); both scripts print AUC and the detection rate at 1 % false positives on it after every epoch.

Training is not bit-for-bit reproducible. After a new run, put the SHA-256 printed by `export_onnx.py` into `model.json`, upload the model together with [`tools/train/MODEL_CARD.md`](../tools/train/MODEL_CARD.md) as `README.md` to the Hugging Face repository, update `revision` in `model.json`, and run `npm run eval -- bundled --calibrate` for the new calibration.

The weights are trained on OpenFake, which is licensed under CC BY-NC 4.0, so they are published under the same license (see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)).

## Rebuild or check the SynthID templates

```bash
.venv/bin/pip install numpy pillow pyarrow fsspec aiohttp huggingface_hub
.venv/bin/python tools/synthid/build_templates.py   # 988 Gemini, 2,034 OpenAI and 75 Nova Canvas images (about 8 GB) -> src/background/synthid-templates/, synthid.json
.venv/bin/python tools/synthid/build_templates.py openai   # only the OpenAI templates, from the files pinned in tools/synthid/openai_sources.json
.venv/bin/python tools/synthid/build_templates.py amazon   # only the Nova Canvas template, from the GitHub commits in tools/synthid/amazon_sources.json
.venv/bin/python tools/synthid/crossval.py --transforms   # Google: templates from two thirds of the images, test on the remaining third; OpenAI: templates from two thirds of the images of every output size, test on the remaining third (--leave-size-out also without the tested size); Amazon: leave one repository out
.venv/bin/python tools/synthid/fetch_openfake.py    # test images to tools/synthid/.cache/openfake (about 9 GB download)
.venv/bin/python tools/synthid/eval.py --transforms tools/synthid/.cache/openfake/fake/nano-banana-pro
.venv/bin/python tools/synthid/eval.py tools/synthid/.cache/openfake/real/* tools/synthid/.cache/openfake/fake/*
.venv/bin/python tools/synthid/fetch_research.py   # second test dataset to tools/synthid/.cache/synthid-research (about 5 GB)
.venv/bin/python tools/synthid/eval.py tools/synthid/.cache/synthid-research/*/
.venv/bin/python tools/synthid/eval.py tools/synthid/.cache/openfake/real/*   # false positives on real photos
```

The thresholds are in [`synthid.json`](../synthid.json) (`threshold` per template).

## Rebuild or check the visible logo checks

Gemini:

```bash
.venv/bin/python tools/visible/gemini_logo.py build   # alpha map from GeminiWatermarkTool (pinned commit) -> src/background/gemini-logo.json
.venv/bin/python tools/visible/gemini_logo.py score path/to/image.png   # score and verdict per image
.venv/bin/python tools/visible/eval.py --transforms path/to/images-with-logo   # detection, also after re-encoding, downscaling and cropping
.venv/bin/python tools/visible/eval.py --apply --transforms --sample 100 tools/eval/.cache/samples/research/fake tools/eval/.cache/samples/core/fake   # logo overlaid in five Gemini layouts
.venv/bin/python tools/visible/eval.py tools/eval/.cache/samples/*/real   # false positives
```

Kling:

```bash
.venv/bin/python tools/visible/kling_logo.py build   # templates averaged from tools/visible/.cache/real/kling-2024 and kling-2026 -> src/background/kling-logo.json
.venv/bin/python tools/visible/kling_logo.py score path/to/image.png   # score per layout and verdict per image
.venv/bin/python tools/visible/eval.py --logo kling --transforms path/to/images-with-logo   # detection, also after re-encoding, downscaling and cropping
.venv/bin/python tools/visible/eval.py --logo kling tools/eval/.cache/samples/*/real   # false positives
```

Doubao, Jimeng, Qwen, Kling (Chinese) and Grok:

```bash
.venv/bin/python tools/visible/marks.py build   # alpha maps from remove-ai-watermarks (pinned commit) and averages of tools/visible/.cache/real/{qwen,kling-cjk,grok,grok-2024,grok-aurora} -> src/background/visible-marks.json
.venv/bin/python tools/visible/marks.py score path/to/image.png   # score per mark and verdict per image
.venv/bin/python tools/visible/eval.py --logo marks --transforms path/to/images-with-mark   # detection, also after re-encoding, downscaling and cropping
.venv/bin/python tools/visible/eval.py --logo marks tools/eval/.cache/samples/*/real   # false positives
```

With `--logo marks`, `eval.py` reports the highest ratio of score to threshold over all marks, so a hit is a value of at least 1.

`build` needs the labeled images with the logo: `kling-2024` holds the images of [nyuuzyou/klingai](https://huggingface.co/datasets/nyuuzyou/klingai), `kling-2026` the current free-tier images listed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md#kling-logo-templates). The labeled images of the further marks are listed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md#further-visible-mark-templates). The thresholds are `THRESHOLD` in [`tools/visible/gemini_logo.py`](../tools/visible/gemini_logo.py) and [`tools/visible/kling_logo.py`](../tools/visible/kling_logo.py), `threshold` per mark in [`tools/visible/marks.py`](../tools/visible/marks.py) and `GEMINI_THRESHOLD`, `KLING_THRESHOLD` and `MARK_THRESHOLDS` in [`src/background/visible-watermark.ts`](../src/background/visible-watermark.ts).
