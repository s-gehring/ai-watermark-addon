# Contributing

Thanks for your interest in improving AI Watermark.

## Reporting detection errors

False positives and missed AI images are the most useful reports. Please use the [detection issue template](https://github.com/s-gehring/ai-watermark-addon/issues/new?template=detection.yml) and include:

- a link to the image (not just the page), if it is public
- what the badge showed and what you expected
- the details panel from hovering the badge (a screenshot is fine)
- if you know it: the generator that made the image, or where a real photo comes from

Do not upload images of private people.

## Development setup

See [docs/development.md](docs/development.md). In short:

```bash
npm ci
npm run fetch-model
npm run build
npm test
npm run test:e2e
```

## Pull requests

- Keep a pull request focused on one change and describe what it changes and why.
- Run `npm run typecheck`, `npm test` and `npm run test:e2e` before opening it; CI runs them as well.
- Add or update tests for behavior changes.
- Changes to detection (model, calibration, thresholds, SynthID templates) need measurements: run the evaluation described in [docs/development.md](docs/development.md) and update [docs/accuracy.md](docs/accuracy.md).
- User-facing strings live in `src/_locales/`. Add every new string in English and German.
- Code, comments, commit messages and documentation are in English.

By contributing, you agree that your contribution is licensed under the [MIT License](LICENSE).
