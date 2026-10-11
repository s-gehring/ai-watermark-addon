# AI Watermark (Firefox-Addon)

Prüft größere Bilder auf Webseiten auf AI-Generierung und markiert Treffer mit einem „AI“-Badge oben rechts im Bild. Alles läuft lokal im Browser: Kein Bild verlässt den Rechner, es gibt keine API und keinen Server.

## Was geprüft wird

- `<img>`-Elemente und CSS-Hintergrundbilder, deren kürzere gerenderte Seite mindestens 200 px misst (einstellbar)
- auch nachgeladene Bilder (Lazy Loading, Infinite Scroll, `srcset`-Wechsel); geprüft wird erst, wenn ein Bild in die Nähe des sichtbaren Bereichs kommt
- SVGs werden übersprungen

## Wie erkannt wird

1. **Metadaten** (sicherer Treffer, Badge „AI · Metadaten“): IPTC Digital Source Type `trainedAlgorithmicMedia` bzw. `compositeWithTrainedAlgorithmicMedia` in XMP oder C2PA-Manifesten (OpenAI, Google, Adobe Firefly, Midjourney u. a.), Stable-Diffusion-Parameter (A1111/Forge), ComfyUI-Workflows, NovelAI, InvokeAI. Viele Plattformen entfernen Metadaten beim Upload, dann greift nur Schritt 2.
2. **Modell** (Badge „AI 87 %“): [CommunityForensics-DeepfakeDet-ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT) (ViT-S, CVPR 2025, MIT-Lizenz), int8-quantisiert (23 MB), ausgeführt mit ONNX Runtime Web (WASM) im Background-Script. Die Vorverarbeitung (Bicubic-Resize auf 440 px, Center-Crop 384 px) ist in `src/background/preprocess.ts` nachgebaut und rechnet bitgenau wie Pillow in der Referenz-Pipeline des Modells. Der Rohwert wird per Platt-Scaling kalibriert (`model.json`).

Das Background-Script lädt das Bild selbst noch einmal (meist aus dem HTTP-Cache), damit auch Bilder fremder Domains ohne CORS-Freigabe geprüft werden können. Ergebnisse werden pro URL zwischengespeichert. Ein Bild braucht in Firefox ca. 0,7 s; Bilder werden nacheinander geprüft, sichtbare zuerst.

## Genauigkeit

Gemessen mit `npm run eval` auf 660 Bildern aus dem Testsplit von [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (246 AI-Bilder von 18 Generatoren bis 2026, 400 echte Fotos aus ImageNet und DOCCI):

| Schwelle | erkannte AI-Bilder | echte Fotos fälschlich markiert |
|---|---|---|
| 50 % | 81 % | 11,5 % |
| 70 % | 74 % | 6,0 % |
| **80 % (Default)** | **70 %** | **3,3 %** |
| 90 % | 57 % | 1,3 % |

AUC 0,94. Zum Vergleich lagen die anderen geprüften lokalen Modelle (SMOGY, SigLIP-basierte Detektoren, ViT-Deepfake-Klassifikatoren) bei AUC 0,81–0,86.

Einschränkungen:

- Das Badge ist ein Hinweis, kein Beweis. Rund jedes dritte AI-Bild ohne Metadaten bleibt unmarkiert, und einzelne echte Bilder werden markiert.
- Computergrafik (3D-Renderings, Diagramme, stark bearbeitete Grafiken) wirkt auf das Modell „synthetisch“ und wird überdurchschnittlich oft markiert.
- Starke Kompression und kleine Vorschaubilder verschlechtern die Erkennung.
- Kalibrierung und Schwelle stammen aus demselben Datensatz wie die Messung. Auf echten Webseiten sind die Raten realistisch etwas schlechter.

## Bedienung

- Toolbar-Button: Anzahl markierter Bilder auf der Seite, Addon global oder pro Seite an- und ausschalten
- Klick auf ein Badge blendet es aus
- Einstellungen (`about:addons` → AI Watermark → Einstellungen): Mindestgröße, Schwelle, Metadaten-Prüfung, CSS-Hintergründe, Prozentanzeige, deaktivierte Seiten

## Entwicklung

Voraussetzungen: Node 22, Firefox.

```bash
npm ci
npm run fetch-model   # lädt das Modell (gepinnte Revision, SHA-256-geprüft) nach vendor/models/
npm run build         # baut nach dist/
npm start             # startet Firefox mit dem Addon (web-ext run)
```

| Befehl | Zweck |
|---|---|
| `npm run watch` | Rebuild bei Änderungen |
| `npm test` | Unit-Tests (Vitest) |
| `npm run test:e2e` | baut und testet das Addon in headless Firefox (Selenium/geckodriver) |
| `npm run typecheck` | TypeScript-Prüfung |
| `npm run lint` | `web-ext lint` auf `dist/` |
| `npm run package` | erzeugt das ZIP in `web-ext-artifacts/` |

### Dauerhaft installieren

Release-Firefox installiert nur signierte Addons. Ohne Signatur geht es nur temporär über `about:debugging` → „Temporäres Add-on laden“ (`dist/manifest.json` wählen) oder dauerhaft in Firefox Developer Edition/Nightly mit `xpinstall.signatures.required = false`.

Für eine signierte, nicht öffentlich gelistete Version mit [AMO-API-Schlüsseln](https://addons.mozilla.org/developers/addon/api/key/):

```bash
npm run build
npx web-ext sign --source-dir dist --channel unlisted --api-key "$AMO_JWT_ISSUER" --api-secret "$AMO_JWT_SECRET"
```

Das Paket ist ca. 23 MB groß (Modell und WASM-Runtime), unter dem AMO-Limit von 200 MB.

### Modell evaluieren oder tauschen

```bash
python3 -m venv .venv && .venv/bin/pip install pyarrow fsspec aiohttp huggingface_hub
.venv/bin/python tools/eval/fetch_samples.py --shards 13 --per-class 400 --per-generator 25 --wild 150
npm run eval -- bundled --calibrate --per-generator
npm run eval -- all
```

`fetch_samples.py` liest nur einzelne Parquet-Row-Groups aus OpenFake (ca. 1,5 GB Download statt der kompletten Shards). `--calibrate` fittet Platt-Scaling neu; die Werte gehören dann nach `model.json` (`calibration`, `threshold`).

## Aufbau

```
src/background/   Modell, Vorverarbeitung, Metadaten-Scan, Queue und Cache
src/content/      Bildsuche (IntersectionObserver, MutationObserver) und Badge-Overlay im Shadow DOM
src/popup/        Toolbar-Popup
src/options/      Einstellungen
model.json        gepinntes Modell, Eingabeformat, Kalibrierung, Default-Schwelle
tools/eval/       Testdaten-Download und Modellvergleich
```
