# AI Watermark (Firefox-Addon)

Prüft größere Bilder auf Webseiten auf AI-Generierung und markiert Treffer mit einem „AI“-Badge oben rechts im Bild. Alles läuft lokal im Browser: Kein Bild verlässt den Rechner, es gibt keine API und keinen Server.

## Was geprüft wird

- `<img>`-Elemente und CSS-Hintergrundbilder, deren kürzere gerenderte Seite mindestens 200 px misst (einstellbar)
- auch nachgeladene Bilder (Lazy Loading, Infinite Scroll, `srcset`-Wechsel); geprüft wird erst, wenn ein Bild in die Nähe des sichtbaren Bereichs kommt
- SVGs werden übersprungen

## Wie erkannt wird

1. **Metadaten** (sicherer Treffer, Badge „AI · Metadaten“): IPTC Digital Source Type `trainedAlgorithmicMedia` bzw. `compositeWithTrainedAlgorithmicMedia` in XMP oder C2PA-Manifesten (OpenAI, Google, Adobe Firefly, Midjourney u. a.), Stable-Diffusion-Parameter (A1111/Forge), ComfyUI-Workflows, NovelAI, InvokeAI. Viele Plattformen entfernen Metadaten beim Upload, dann greift nur Schritt 2.
2. **Modell** (Badge „AI 97 %“): [CommunityForensics-DeepfakeDet-ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT) (ViT-S, CVPR 2025, MIT-Lizenz), int8-quantisiert (23 MB), ausgeführt mit ONNX Runtime Web (WASM) im Background-Script. Die Vorverarbeitung (Bicubic-Resize auf 440 px, Center-Crop 384 px) ist in `src/background/preprocess.ts` nachgebaut und rechnet bitgenau wie Pillow in der Referenz-Pipeline des Modells. Der Rohwert wird per Platt-Scaling kalibriert (`model.json`).

Das Background-Script lädt das Bild selbst noch einmal (meist aus dem HTTP-Cache), damit auch Bilder fremder Domains ohne CORS-Freigabe geprüft werden können. Metadaten-Scan, Dekodierung und Inferenz laufen in einem Web Worker, damit Popup und Background-Script reaktionsfähig bleiben. Ergebnisse werden pro URL zwischengespeichert. Ein Bild braucht in Firefox ca. 0,7 s; Bilder werden nacheinander geprüft, sichtbare zuerst.

## Genauigkeit

Gemessen mit `npm run eval` auf 3406 Bildern. Die eine Hälfte dient der Kalibrierung, die Zahlen unten stammen aus der anderen:

- AI-Bilder: 300 aus dem Testsplit von [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (18 Generatoren bis 2026) und 188 aus OpenFakes Reddit-Testset (AI-Subreddits)
- echte Fotos: 943 aus OpenFake (ImageNet, DOCCI), 375 aus Fotografie-Subreddits (OpenFake-Reddit-Testset) und 1000 aus [COCO](https://huggingface.co/datasets/detection-datasets/coco)
- Grafik und Kunst: 300 Diagramme aus [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) und 300 Gemälde aus [WikiArt](https://huggingface.co/datasets/huggan/wikiart)

Jedes Bild wird dreimal bewertet: als Originaldatei und so, wie Bilder auf Webseiten typischerweise ankommen, also auf höchstens 1280 px bzw. 800 px verkleinert und als JPEG mit Qualität 80 bzw. 75 gespeichert. Kalibrierung und Default-Schwelle sind so gewählt, dass unter Web-Bedingungen rund 1 % der echten Fotos markiert werden.

Bei der Default-Schwelle von 94 %:

| | Originaldatei | Web 1280 px | Web 800 px |
|---|---|---|---|
| AI-Bilder erkannt (OpenFake) | 42 % | 32 % | 29 % |
| AI-Bilder erkannt (Reddit) | 33 % | 28 % | 32 % |
| echte Fotos markiert (alle) | 1,0 % | 1,0 % | 1,2 % |
| echte Fotos markiert (Fotografie-Subreddits) | 3,2 % | 1,6 % | 3,2 % |
| Gemälde markiert | 12 % | 11 % | 12 % |
| Diagramme markiert | 0 % | 0 % | 0 % |

Andere Schwellen (Web 1280 px, nur Fotos):

| Schwelle | AI-Bilder erkannt | echte Fotos markiert |
|---|---|---|
| 50 % | 75 % | 18 % |
| 70 % | 59 % | 7,3 % |
| 80 % | 50 % | 4,5 % |
| 90 % | 38 % | 2,1 % |
| **94 % (Default)** | **31 %** | **1,0 %** |
| 97 % | 22 % | 0,4 % |

AUC unter Web-Bedingungen im Vergleich: Das gebündelte Modell erreicht 0,85–0,87, dasselbe Modell ohne Quantisierung (fp32, viermal so groß) 0,88–0,89, die anderen geprüften lokalen Modelle (SMOGY, SigLIP-Detektor, ViT-Deepfake-Klassifikator) 0,71–0,85.

Einschränkungen:

- Das Badge ist ein Hinweis, kein Beweis. Die Default-Schwelle ist auf wenige Fehlalarme ausgelegt, dafür bleiben rund zwei von drei AI-Bildern ohne Metadaten unmarkiert. Wer mehr erkennen will, senkt die Schwelle in den Einstellungen und nimmt mehr Fehlalarme in Kauf.
- Gemälde werden deutlich öfter markiert als Fotos (rund jedes achte). Bei stark nachbearbeiteten Fotos wie aus Fotografie-Subreddits sind es bis zu 3 %. 3D-Renderings sind nicht im Testset.
- Kompression und Verkleinerung kosten deutlich Erkennung (AUC 0,93 als Originaldatei, 0,85–0,87 unter Web-Bedingungen).
- Kalibrierung und Messung nutzen getrennte Bilder, aber dieselben Quellen. Seiten mit anderer Bildauswahl oder eigener Kompression können abweichen.

## Bedienung

Jedes geprüfte Bild bekommt oben rechts ein Symbol:

| Symbol | Bedeutung |
|---|---|
| dunkler Kreis mit Spinner | wird geprüft |
| violettes „AI 97 %“ bzw. „AI · Metadaten“ | wahrscheinlich bzw. laut Metadaten AI-generiert |
| grüner Haken | geprüft, unauffällig |
| graues „?“ | konnte nicht geprüft werden (Download, Format oder Seitenverhältnis) |

- Toolbar-Button: Zahl der AI-Bilder auf der Seite; solange noch geprüft wird, ist das Icon gedimmt und zeigt einen Fortschrittsbalken
- Popup: Fortschritt („Prüfe Bilder: 12 von 30“), Zähler für unauffällige und nicht prüfbare Bilder, Addon global oder pro Seite an- und ausschalten
- Klick auf ein Symbol blendet es aus
- Einstellungen (`about:addons` → AI Watermark → Einstellungen): Mindestgröße, Schwelle, Metadaten-Prüfung, CSS-Hintergründe, Prozentanzeige, Symbole für unauffällige Bilder und laufende Prüfungen, deaktivierte Seiten

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
.venv/bin/python tools/eval/fetch_samples.py
npm run eval -- bundled --calibrate --per-generator
npm run eval -- all
```

`fetch_samples.py` liest nur einzelne Parquet-Row-Groups aus OpenFake, COCO, ChartQA und WikiArt (ca. 4 GB Download statt der kompletten Datensätze, 1,9 GB auf der Platte). `npm run eval` speichert die Rohwerte pro Modell in `tools/eval/.cache/scores/`, weitere Läufe rechnen nur neue Bilder. `--calibrate` fittet Platt-Scaling auf der Kalibrierhälfte neu und wählt die Schwelle für 1 % markierte Fotos unter Web-Bedingungen; die Werte gehören dann nach `model.json` (`calibration`, `threshold`). Ändert sich die Kalibrierung, setzt das Addon eine selbst gewählte Schwelle auf den neuen Default zurück, weil die Prozentwerte danach etwas anderes bedeuten.

## Aufbau

```
src/background/   Bild-Download, Cache, Tab-Statistik und Toolbar-Icon; Worker mit Metadaten-Scan, Vorverarbeitung und Modell
src/content/      Bildsuche (IntersectionObserver, MutationObserver) und Badge-Overlay im Shadow DOM
src/popup/        Toolbar-Popup
src/options/      Einstellungen
model.json        gepinntes Modell, Eingabeformat, Kalibrierung, Default-Schwelle
tools/eval/       Testdaten-Download und Modellvergleich
```
