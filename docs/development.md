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
| `npm run lint` | `web-ext lint` on `dist/` |
| `npm run package` | creates the ZIP in `web-ext-artifacts/` |

## Release

Release Firefox only installs add-ons signed by Mozilla. Without a signature, `dist/` can be loaded temporarily via `about:debugging` → "This Firefox" → "Load Temporary Add-on…" (pick `dist/manifest.json`), or permanently in Firefox Developer Edition or Nightly with `xpinstall.signatures.required = false`.

Releases are listed on addons.mozilla.org (AMO) and also published as GitHub releases by [`.github/workflows/release.yml`](../.github/workflows/release.yml) and [`.github/workflows/publish.yml`](../.github/workflows/publish.yml). To release:

1. Raise `version` in `package.json` and merge the change into `main`.
2. Tag the merge commit and push the tag:

   ```bash
   git tag v0.2.0 origin/main
   ```

   ```bash
   git push origin v0.2.0
   ```

The release workflow tests and lints the build, uploads it together with the source code and [`amo-metadata.json`](../amo-metadata.json) (categories, support URL, license) to AMO and creates a draft GitHub release. AMO reviews the version, which can take a few days. Every night the publish workflow asks AMO about each draft release. Once the version is approved, it attaches the signed `.xpi` to the release and publishes it. To check right away, start the publish workflow by hand under Actions → Publish → "Run workflow". A rejected version makes the publish workflow fail until the draft release is deleted.

Firefox updates the add-on through AMO, also when it was installed from the `.xpi` of a GitHub release.

The package is about 49 MB (model, WASM runtime and SynthID templates), below the AMO limit of 200 MB.

## Evaluate or swap the model

```bash
python3 -m venv .venv && .venv/bin/pip install pyarrow fsspec aiohttp huggingface_hub pillow
.venv/bin/python tools/eval/fetch_samples.py
npm run eval -- bundled --calibrate --per-generator
npm run eval -- all
npm run eval -- members --members=detectra-openfake,angelhd25/ull-ai-image-detector   # fit and measure any combination
```

`fetch_samples.py` reads only single Parquet row groups from OpenFake, COCO, ChartQA and WikiArt and 600 files from bigdatamark/synthid-research, plus 600 YouTube thumbnails from `i.ytimg.com` and 300 paintings from the Cleveland Museum of Art API (about 5 GB download instead of the full datasets, 3.3 GB on disk). Test data and raw scores always live in the main checkout under `tools/eval/.cache/`, even when the scripts run in a git worktree, so they survive deleting the worktree.

`npm run eval` stores the raw scores per model in `tools/eval/.cache/scores/`; later runs only score new images. `--calibrate` uses the calibration half to choose the weighting of several bundled models (as many detected AI images as possible at no more than 5 % flagged paintings), fits Platt scaling and picks the threshold for 1 % flagged photos under web conditions. The resulting values go into [`model.json`](../model.json) (`calibration`, `threshold`). When the calibration changes, the add-on resets a user-chosen threshold to the new default, because the percentages mean something else afterwards.

## Train the model

Needs a CUDA GPU with 12 GB (measured on an RTX 4070 Ti) and about 95 GB of disk space.

```bash
.venv/bin/pip install torch timm safetensors pyarrow pillow numpy huggingface_hub onnx onnxruntime onnxconverter-common
.venv/bin/python tools/train/fetch_data.py   # Detectra weights, 16 OpenFake files and negatives to tools/train/.cache/
.venv/bin/python tools/train/train.py --name full   # 3 epochs, about 1 h
.venv/bin/python tools/train/export_onnx.py tools/train/.cache/runs/full/epoch3.safetensors   # -> vendor/models/detectra-openfake/model.onnx
```

The datasets are pinned to fixed revisions. The last of the 16 OpenFake files is held out; `train.py` prints AUC and the detection rate at 1 % false positives on it after every epoch.

Training is not bit-for-bit reproducible. After a new run, put the SHA-256 printed by `export_onnx.py` into `model.json`, upload the model together with [`tools/train/MODEL_CARD.md`](../tools/train/MODEL_CARD.md) as `README.md` to the Hugging Face repository, update `revision` in `model.json`, and run `npm run eval -- bundled --calibrate` for the new calibration.

The weights are trained on OpenFake, which is licensed under CC BY-NC 4.0, so they are published under the same license (see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)).

## Rebuild or check the SynthID templates

```bash
.venv/bin/pip install numpy pillow pyarrow fsspec aiohttp huggingface_hub
.venv/bin/python tools/synthid/build_templates.py   # 288 Gemini and 959 OpenAI images (about 3.3 GB) -> src/background/synthid-templates/, synthid.json
.venv/bin/python tools/synthid/build_templates.py openai   # only the OpenAI templates
.venv/bin/python tools/synthid/crossval.py --transforms   # OpenAI: template from two thirds, test on the remaining third
.venv/bin/python tools/synthid/fetch_openfake.py    # test images to tools/synthid/.cache/openfake (about 9 GB download)
.venv/bin/python tools/synthid/eval.py --transforms tools/synthid/.cache/openfake/fake/nano-banana-pro
.venv/bin/python tools/synthid/eval.py tools/synthid/.cache/openfake/real/* tools/synthid/.cache/openfake/fake/*
.venv/bin/python tools/synthid/fetch_research.py   # second test dataset to tools/synthid/.cache/synthid-research (about 5 GB)
.venv/bin/python tools/synthid/eval.py tools/synthid/.cache/synthid-research/*/
.venv/bin/python tools/synthid/eval.py --ignore-aspect tools/synthid/.cache/openfake/real/*   # every template on every image
```

The thresholds are in [`synthid.json`](../synthid.json) (`threshold` per template).
