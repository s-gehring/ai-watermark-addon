# AI Watermark (Firefox-Addon)

Prüft größere Bilder auf Webseiten auf AI-Generierung und markiert Treffer mit einem „AI“-Badge oben rechts im Bild. Alles läuft lokal im Browser: Kein Bild verlässt den Rechner, es gibt keine API und keinen Server.

## Was geprüft wird

- `<img>`-Elemente und CSS-Hintergrundbilder, deren kürzere gerenderte Seite mindestens 200 px misst (einstellbar)
- auch nachgeladene Bilder (Lazy Loading, Infinite Scroll, `srcset`-Wechsel); geprüft wird erst, wenn ein Bild in die Nähe des sichtbaren Bereichs kommt
- SVGs werden übersprungen

## Wie erkannt wird

1. **Metadaten** (sicherer Treffer, Badge „AI · Metadaten“): IPTC Digital Source Type `trainedAlgorithmicMedia` bzw. `compositeWithTrainedAlgorithmicMedia` in XMP oder C2PA-Manifesten (OpenAI, Google, Adobe Firefly, Midjourney u. a.), Stable-Diffusion-Parameter (A1111/Forge), ComfyUI-Workflows, NovelAI, InvokeAI. Deklariert ein C2PA-Manifest das SynthID-Wasserzeichen (Google schreibt dort „Applied imperceptible SynthID watermark“), heißt das Badge „AI · SynthID“. Viele Plattformen entfernen Metadaten beim Upload, dann greifen nur die Schritte 2 und 3.
2. **SynthID-Wasserzeichen** (Badge „AI · SynthID“): Google bettet in Bilder aus Gemini und Imagen das unsichtbare Wasserzeichen [SynthID](https://deepmind.google/models/synthid/) ein. Googles Detektor ist nicht öffentlich. Das Addon prüft deshalb auf einen festen Anteil des Wasserzeichens, der sich in Gemini-Bildern messen lässt: Bei bestimmten Ortsfrequenzen, gezählt in Schwingungen pro Bildbreite und -höhe, hat das Spektrum vor allem der Farbkanäle (Cb, Cr) in jedem Gemini-Bild dieselbe Phase, unabhängig vom Bildinhalt. Das Bild wird auf 512 × 512 skaliert, je Kanal per FFT zerlegt, und die Phasen werden mit einer Vorlage verglichen, die aus 288 Gemini-Bildern gemittelt ist (`src/background/synthid-template.bin`, `synthid.json`). Ergebnis ist ein z-Wert (für Bilder ohne Wasserzeichen etwa standardnormalverteilt), ab 12 gilt das Wasserzeichen als gefunden und das Modell wird übersprungen. Weil die Frequenzen relativ zur Bildgröße gezählt werden, klappt das bei jeder Auflösung und jedem Seitenverhältnis. Kostet in Node gemessen ca. 0,15 s für ein 4-Megapixel-Bild.
3. **Modell** (Badge „AI 97 %“): [CommunityForensics-DeepfakeDet-ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT) (ViT-S, CVPR 2025, MIT-Lizenz), int8-quantisiert (23 MB), ausgeführt mit ONNX Runtime Web (WASM) im Background-Script. Die Vorverarbeitung (Bicubic-Resize auf 440 px, Center-Crop 384 px) ist in `src/background/preprocess.ts` nachgebaut und rechnet bitgenau wie Pillow in der Referenz-Pipeline des Modells. Der Rohwert wird per Platt-Scaling kalibriert (`model.json`).

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

- Das Badge ist ein Hinweis, kein Beweis. Die Default-Schwelle ist auf wenige Fehlalarme ausgelegt, dafür bleiben rund zwei von drei AI-Bildern ohne Metadaten oder Wasserzeichen unmarkiert. Wer mehr erkennen will, senkt die Schwelle in den Einstellungen und nimmt mehr Fehlalarme in Kauf.
- Gemälde werden deutlich öfter markiert als Fotos (rund jedes achte). Bei stark nachbearbeiteten Fotos wie aus Fotografie-Subreddits sind es bis zu 3 %. 3D-Renderings sind nicht im Testset.
- Kompression und Verkleinerung kosten deutlich Erkennung (AUC 0,93 als Originaldatei, 0,85–0,87 unter Web-Bedingungen).
- Kalibrierung und Messung nutzen getrennte Bilder, aber dieselben Quellen. Seiten mit anderer Bildauswahl oder eigener Kompression können abweichen.

### SynthID-Prüfung

Gemessen mit `tools/synthid/eval.py` auf einer Stichprobe aus dem Testsplit von OpenFake (`tools/synthid/fetch_openfake.py`). Keines dieser Bilder steckt in der Vorlage.

| Gemini-Bilder (Nano Banana Pro) | erkannt |
|---|---|
| Originale als PNG (32 Bilder, 848 × 1264 bis 4K) | 88 % |
| dieselben, auf 50 % bzw. 33 % verkleinert | 88 % bzw. 84 % |
| dieselben als JPEG, Qualität 90, ohne bzw. mit Chroma-Subsampling (4:4:4 bzw. 4:2:0) | 69 % bzw. 47 % |
| dieselben als WebP, Qualität 80 | 3 % |
| dieselben, ringsum auf 95 % zugeschnitten | 0 % |
| im Datensatz bereits als JPEG/WebP neu kodiert (28 Bilder) | 11 % |

Fälschlich markiert: 0 von 1.067 echten Fotos (ImageNet, DOCCI) und 0 von 417 Bildern anderer Generatoren (DALL·E 3, GPT-Image 1.5 und 2, Sora 2, Seedream 5, FLUX.2, Z-Image, ERNIE-Image, Illustrious). Der höchste z-Wert eines echten Fotos lag bei 9,4. Drei weitere Bilder, die OpenFake anderen Generatoren zuordnet, schlugen an. Eines trägt ein C2PA-Manifest von Google, die beiden anderen haben exakt die Auflösung von Nano-Banana-Pro-Ausgaben (2528 × 1696 und 3392 × 5056); vermutlich sind es Gemini-Bilder.

Einschränkungen:

- Das ist ein Nachbau aus öffentlichen Beispielbildern, nicht Googles Decoder. Getestet sind nur Bilder von Gemini 3 Pro Image (Nano Banana Pro) und Gemini 3.1 Flash Image. Andere Google-Modelle und andere Anbieter, die SynthID einsetzen, sind ungeprüft.
- Zuschneiden zerstört das Signal vollständig, verlustbehaftete Neukodierung (WebP, JPEG mit Chroma-Subsampling) weitgehend. Bilder aus sozialen Netzwerken werden deshalb meist nicht erkannt; dann greift wie bisher das Modell.
- Mit Gemini bearbeitete echte Fotos tragen ebenfalls SynthID und werden markiert.
- Die Vorlage ist aus dem Hugging-Face-Datensatz [aoxo/reverse-synthid](https://huggingface.co/datasets/aoxo/reverse-synthid) berechnet (gepinnte Revision, Lizenz dort nicht angegeben); ausgeliefert werden nur die gemittelten Phasen und Gewichte, keine Bilder.

## Bedienung

Jedes geprüfte Bild bekommt oben rechts ein Symbol:

| Symbol | Bedeutung |
|---|---|
| dunkler Kreis mit Spinner | wird geprüft |
| violettes „AI 97 %“ bzw. „AI · Metadaten“ | wahrscheinlich bzw. laut Metadaten AI-generiert |
| violettes „AI · SynthID“ | SynthID-Wasserzeichen von Google in den Pixeln oder im C2PA-Manifest gefunden |
| grüner Haken | geprüft, unauffällig |
| graues „?“ | konnte nicht geprüft werden (Download, Format oder Seitenverhältnis) |

- Toolbar-Button: Zahl der AI-Bilder auf der Seite; solange noch geprüft wird, ist das Icon gedimmt und zeigt einen Fortschrittsbalken
- Popup: Fortschritt („Prüfe Bilder: 12 von 30“), Zähler für unauffällige und nicht prüfbare Bilder, Addon global oder pro Seite an- und ausschalten
- Klick auf ein Symbol blendet es aus
- Einstellungen (`about:addons` → AI Watermark → Einstellungen): Mindestgröße, Schwelle, Metadaten-Prüfung, SynthID-Prüfung der Pixel, CSS-Hintergründe, Prozentanzeige, Symbole für unauffällige Bilder und laufende Prüfungen, deaktivierte Seiten

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

### SynthID-Vorlage neu berechnen oder prüfen

```bash
.venv/bin/pip install numpy pillow pyarrow fsspec aiohttp huggingface_hub
.venv/bin/python tools/synthid/build_template.py    # 288 Gemini-Bilder (ca. 1 GB) -> src/background/synthid-template.bin, synthid.json
.venv/bin/python tools/synthid/fetch_openfake.py    # Testbilder nach tools/synthid/.cache/openfake (ca. 9 GB Download)
.venv/bin/python tools/synthid/eval.py --transforms tools/synthid/.cache/openfake/fake/nano-banana-pro
.venv/bin/python tools/synthid/eval.py tools/synthid/.cache/openfake/real/* tools/synthid/.cache/openfake/fake/*
```

`tools/synthid/synthid.py` ist die Referenzimplementierung von `src/background/synthid.ts`; ein Unit-Test prüft, dass beide dieselben z-Werte liefern. Die Schwelle steht in `synthid.json` (`threshold`).

## Aufbau

```
src/background/   Bild-Download, Cache, Tab-Statistik und Toolbar-Icon; Worker mit Metadaten-Scan, SynthID-Prüfung, Vorverarbeitung und Modell
src/content/      Bildsuche (IntersectionObserver, MutationObserver) und Badge-Overlay im Shadow DOM
src/popup/        Toolbar-Popup
src/options/      Einstellungen
model.json        gepinntes Modell, Eingabeformat, Kalibrierung, Default-Schwelle
synthid.json      Herkunft, Skalierung und Schwelle der SynthID-Vorlage (src/background/synthid-template.bin)
tools/eval/       Testdaten-Download und Modellvergleich
tools/synthid/    Referenzimplementierung, Vorlagen-Bau und Messung der SynthID-Prüfung
```
