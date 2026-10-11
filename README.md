# AI Watermark (Firefox-Addon)

Prüft größere Bilder auf Webseiten auf AI-Generierung und markiert Treffer mit einem „AI“-Badge oben rechts im Bild. Alles läuft lokal im Browser: Kein Bild verlässt den Rechner, es gibt keine API und keinen Server.

## Was geprüft wird

- `<img>`-Elemente und CSS-Hintergrundbilder, deren kürzere gerenderte Seite mindestens 200 px misst (einstellbar)
- auch nachgeladene Bilder (Lazy Loading, Infinite Scroll, `srcset`-Wechsel); geprüft wird erst, wenn ein Bild in die Nähe des sichtbaren Bereichs kommt
- SVGs werden übersprungen

## Wie erkannt wird

1. **Metadaten** (sicherer Treffer, Badge „AI · Metadaten“): IPTC Digital Source Type `trainedAlgorithmicMedia` bzw. `compositeWithTrainedAlgorithmicMedia` in XMP oder C2PA-Manifesten (OpenAI, Google, Adobe Firefly, Midjourney u. a.), Stable-Diffusion-Parameter (A1111/Forge), ComfyUI-Workflows, NovelAI, InvokeAI sowie das AIGC-Label nach der chinesischen Kennzeichnungspflicht GB 45438-2025 (JSON mit `Label`, `ContentProducer`, `ProduceID` … als PNG-Textchunk `AIGC`, im XMP-Namespace `http://www.tc260.org.cn/ns/AIGC/1.0/` oder im EXIF-Kommentar; seit September 2025 Pflicht für Anbieter in China, z. B. Kling). Deklariert ein C2PA-Manifest das SynthID-Wasserzeichen (Google schreibt dort „Applied imperceptible SynthID watermark“), heißt das Badge „AI · SynthID“. Viele Plattformen entfernen Metadaten beim Upload, dann greifen nur die Schritte 2 und 3.
2. **SynthID-Wasserzeichen** (Badge „AI · SynthID“): Google bettet in Bilder aus Gemini und Imagen das unsichtbare Wasserzeichen [SynthID](https://deepmind.google/models/synthid/) ein, OpenAI seit Mai 2026 ebenfalls in alle Bilder aus ChatGPT und der API. Einen öffentlichen Detektor gibt es nicht. Das Addon prüft deshalb auf einen festen Anteil des Wasserzeichens, der sich in markierten Bildern messen lässt: Bei bestimmten Ortsfrequenzen, gezählt in Schwingungen pro Bildbreite und -höhe, hat das Spektrum in jedem markierten Bild dieselbe Phase, unabhängig vom Bildinhalt. Das Bild wird auf 512 × 512 bzw. 1024 × 1024 skaliert, je YCbCr-Kanal per FFT zerlegt, und die Phasen werden mit Vorlagen verglichen, die aus markierten Bildern gemittelt sind (`src/background/synthid-templates/`, `synthid.json`). Ergebnis ist je Vorlage ein z-Wert (für Bilder ohne Wasserzeichen etwa standardnormalverteilt). Ab 12 (Google) bzw. 8 (OpenAI) gilt das Wasserzeichen als gefunden, das Modell wird übersprungen, und der Tooltip nennt den Anbieter.
   - **Google:** Eine Vorlage aus 288 Gemini-Bildern. Weil die Frequenzen relativ zur Bildgröße gezählt werden, klappt das bei jeder Auflösung und jedem Seitenverhältnis.
   - **OpenAI:** Hier hängt der feste Anteil von der Ausgabegröße ab. Bilder gleicher Größe aus verschiedenen Quellen teilen ihn, Bilder verschiedener Größen nicht. Das Addon hat Vorlagen für die vier Ausgabegrößen, von denen genug Bilder mit nachgewiesenem Wasserzeichen vorliegen: 1254 × 1254, 1448 × 1086, 1672 × 941 und 1122 × 1402 (ChatGPT wählt sie für quadratische, 4:3-, 16:9- und 4:5-Bilder). Jede Vorlage wird nur auf Bilder mit ihrem Seitenverhältnis (±2 %) angewandt, also auch auf verkleinerte Kopien. Das Muster liegt bei hohen Frequenzen, die drei nicht quadratischen Vorlagen rechnen deshalb mit 1024 × 1024 (bis 384 Schwingungen pro Bild). Alle OpenAI-Vorlagen blenden außerdem einen Rand von 2 % aus, weil dort sonst Effekte der Bildränder statt des Wasserzeichens in die Vorlage geraten.

   Kostet in Node gemessen ca. 0,13 s für ein 4-Megapixel-Bild, bei den Seitenverhältnissen 4:3, 16:9 und 4:5 wegen der feineren OpenAI-Vorlagen ca. 0,35 s.
3. **Modell** (Badge „AI 87 %“): [CommunityForensics-DeepfakeDet-ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT) (ViT-S, CVPR 2025, MIT-Lizenz), int8-quantisiert (23 MB), ausgeführt mit ONNX Runtime Web (WASM) im Background-Script. Die Vorverarbeitung (Bicubic-Resize auf 440 px, Center-Crop 384 px) ist in `src/background/preprocess.ts` nachgebaut und rechnet bitgenau wie Pillow in der Referenz-Pipeline des Modells. Der Rohwert wird per Platt-Scaling kalibriert (`model.json`).

Das Background-Script lädt das Bild selbst noch einmal (meist aus dem HTTP-Cache), damit auch Bilder fremder Domains ohne CORS-Freigabe geprüft werden können. Metadaten-Scan, Dekodierung und Inferenz laufen in einem Web Worker, damit Popup und Background-Script reaktionsfähig bleiben. Ergebnisse werden pro URL zwischengespeichert. Ein Bild braucht in Firefox ca. 0,7 s; Bilder werden nacheinander geprüft, sichtbare zuerst.

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

### SynthID-Prüfung

#### Google

Gemessen mit `tools/synthid/eval.py` auf einer Stichprobe aus dem Testsplit von OpenFake (`tools/synthid/fetch_openfake.py`). Keines dieser Bilder steckt in der Vorlage.

| Gemini-Bilder (Nano Banana Pro) | erkannt |
|---|---|
| Originale als PNG (32 Bilder, 848 × 1264 bis 4K) | 88 % |
| dieselben, auf 50 % bzw. 33 % verkleinert | 88 % bzw. 84 % |
| dieselben als JPEG, Qualität 90, ohne bzw. mit Chroma-Subsampling (4:4:4 bzw. 4:2:0) | 69 % bzw. 47 % |
| dieselben als WebP, Qualität 80 | 3 % |
| dieselben, ringsum auf 95 % zugeschnitten | 0 % |
| im Datensatz bereits als JPEG/WebP neu kodiert (28 Bilder) | 11 % |

Zweite Messung auf [bigdatamark/synthid-research](https://huggingface.co/datasets/bigdatamark/synthid-research) (CC BY 4.0, Bilder vom September 2026, `tools/synthid/fetch_research.py`), dieselben Prompts für alle Generatoren:

| Bilder | von der Google-Vorlage markiert |
|---|---|
| Gemini 3.1 Flash Image, JPEG 2048 × 2048 wie von Gemini ausgeliefert (700) | 95 % |
| gpt-image-2 (400), FLUX.2 (400), Qwen-Image-2.1 (700) | 0 % |

Fälschlich markiert: 0 von 1.067 echten Fotos (ImageNet, DOCCI) und 0 von 417 Bildern anderer Generatoren (DALL·E 3, GPT-Image 1.5 und 2, Sora 2, Seedream 5, FLUX.2, Z-Image, ERNIE-Image, Illustrious). Der höchste z-Wert eines echten Fotos lag bei 9,4. Drei weitere Bilder, die OpenFake anderen Generatoren zuordnet, schlugen an. Eines trägt ein C2PA-Manifest von Google, die beiden anderen haben exakt die Auflösung von Nano-Banana-Pro-Ausgaben (2528 × 1696 und 3392 × 5056); vermutlich sind es Gemini-Bilder. In einer zweiten Stichprobe aus OpenFake (642 echte Fotos) schlug die Google-Vorlage bei einem DOCCI-Foto an (z 15,4).

#### OpenAI

Gemessen mit `tools/synthid/crossval.py`: Jede Vorlage wird aus zwei Dritteln der Bilder ihrer Größe gebaut und am übrigen Drittel geprüft (z ≥ 8).

| Ausgabegröße (Testbilder) | Original (PNG) | auf 50 % / 33 % verkleinert | JPEG 90, 4:4:4 / 4:2:0 | WebP 80 |
|---|---|---|---|---|
| 1254 × 1254 (141) | 94 % | 94 % / 87 % | 93 % / 94 % | 84 % |
| 1448 × 1086 (73) | 93 % | 88 % / 30 % | 81 % / 78 % | 59 % |
| 1672 × 941 (41) | 88 % | 85 % / 41 % | 71 % / 63 % | 61 % |
| 1122 × 1402 (64) | 75 % | 67 % / 11 % | 56 % / 36 % | 25 % |

Ringsum auf 95 % zugeschnitten: 0 %. Ausgeliefert werden Vorlagen aus allen Bildern (424, 219, 123 bzw. 193).

Dass der Anteil an der Größe hängt und nicht am Konto, zeigen zwei Proben: Eine Vorlage nur aus den 420 Bildern von bigdatamark erkennt drei der vier 1254 × 1254-Bilder aus der zweiten Quelle (z 8 bis 18), aber keines der anderen Größen aus derselben Quelle (höchstens z 3,1). Und in einem dritten, unabhängigen Datensatz ([Sarim-Hash/liars-dividend-GPT-image-2-photoreal](https://huggingface.co/datasets/Sarim-Hash/liars-dividend-GPT-image-2-photoreal), 80 Bilder in 1024 × 1024 ohne C2PA) erreicht ein Bild mit der 1254er-Vorlage z 69, vermutlich eine verkleinerte Kopie. Die übrigen 79 haben die nicht abgedeckte Größe 1024 × 1024 und schlagen nicht an.

Fälschlich markiert (gemessen wie mit `eval.py --ignore-aspect`, also jede Vorlage auf jedes Bild): keines von 1.933 Bildern ohne OpenAI-Wasserzeichen (642 echte Fotos aus ImageNet und DOCCI, 1.291 Bilder von Gemini, FLUX.2, Qwen-Image, Z-Image, Midjourney 7 und GPT-Image 1.5), auch dann nicht, wenn jedes Bild zusätzlich auf das Seitenverhältnis jeder Vorlage zugeschnitten und auf ihre genaue Größe skaliert wird. Ebenso keines von 456 weiteren OpenAI-Bildern aus Hugging-Face-Datensätzen ohne C2PA-Nachweis, meist in nicht abgedeckten Größen. Der höchste z-Wert lag bei 6,6, die Streuung bei 1,1 (ideal wäre 1).

Einschränkungen:

- Das ist ein Nachbau aus öffentlichen Beispielbildern, nicht Googles Decoder. Getestet sind nur Bilder von Gemini 3 Pro Image (Nano Banana Pro), Gemini 3.1 Flash Image und GPT-Image 2 (ChatGPT und API), andere Modelle sind ungeprüft.
- OpenAI-Bilder werden nur in den vier Größen der Vorlagen und als deren verkleinerte Kopien erkannt. Feste API-Größen wie 1024 × 1024 oder 1536 × 1024 und seltene Größen fehlen, weil es dafür keine oder zu wenige Bilder mit nachgewiesenem Wasserzeichen gibt. Die Vorlage für 1122 × 1402 erkennt schwächer, und jede Vorlage beruht auf höchstens zwei Quellen. OpenAI-Bilder von vor Mai 2026 tragen kein SynthID.
- Zuschneiden zerstört das Signal vollständig. Verlustbehaftete Neukodierung schwächt es, bei Gemini-Bildern stark. Bilder aus sozialen Netzwerken werden deshalb oft nicht erkannt; dann greift wie bisher das Modell.
- Mit Gemini bearbeitete echte Fotos tragen ebenfalls SynthID und werden markiert, mit ChatGPT bearbeitete vermutlich auch.
- Die Google-Vorlage ist aus dem Hugging-Face-Datensatz [aoxo/reverse-synthid](https://huggingface.co/datasets/aoxo/reverse-synthid) berechnet (Lizenz dort nicht angegeben), die OpenAI-Vorlagen aus [bigdatamark/synthid-research](https://huggingface.co/datasets/bigdatamark/synthid-research) (CC BY 4.0) und [winrisef/gpt-generated-image-collection](https://huggingface.co/datasets/winrisef/gpt-generated-image-collection) (MIT). Für OpenAI zählen nur Bilder, deren C2PA-Manifest das Wasserzeichen vermerkt (`c2pa.watermarked.unbound`). Alle Revisionen sind gepinnt; ausgeliefert werden nur die gemittelten Phasen und Gewichte, keine Bilder.

### Geprüft und nicht eingebaut

- **DWT-DCT-Wasserzeichen aus `invisible-watermark`:** Der Referenzcode von SDXL und Stable Video Diffusion (Stability, diffusers) sowie von FLUX.1 und FLUX.2 (Black Forest Labs) bettet je eine feste 48-Bit-Nachricht ein. Ein Nachbau des Decoders fand sie in selbst markierten Bildern (PNG und JPEG 4:4:4), aber in keinem echten Bild aus sieben Datensätzen (FLUX.1.1 Pro und FLUX.2 Pro über die API, SDXL, SD 1.4/2.1, FLUX.2 klein). JPEG mit Chroma-Subsampling, wie im Web üblich, löscht das Wasserzeichen ganz.

## Bedienung

Jedes geprüfte Bild bekommt oben rechts ein Symbol:

| Symbol | Bedeutung |
|---|---|
| dunkler Kreis mit Spinner | wird geprüft |
| violettes „AI 87 %“ bzw. „AI · Metadaten“ | wahrscheinlich bzw. laut Metadaten AI-generiert |
| violettes „AI · SynthID“ | SynthID-Wasserzeichen (Google oder OpenAI) in den Pixeln oder im C2PA-Manifest gefunden |
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

Das Paket ist ca. 28 MB groß (Modell, WASM-Runtime und SynthID-Vorlagen), unter dem AMO-Limit von 200 MB.

### Modell evaluieren oder tauschen

```bash
python3 -m venv .venv && .venv/bin/pip install pyarrow fsspec aiohttp huggingface_hub
.venv/bin/python tools/eval/fetch_samples.py --shards 13 --per-class 400 --per-generator 25 --wild 150
npm run eval -- bundled --calibrate --per-generator
npm run eval -- all
```

`fetch_samples.py` liest nur einzelne Parquet-Row-Groups aus OpenFake (ca. 1,5 GB Download statt der kompletten Shards). `--calibrate` fittet Platt-Scaling neu; die Werte gehören dann nach `model.json` (`calibration`, `threshold`).

### SynthID-Vorlagen neu berechnen oder prüfen

```bash
.venv/bin/pip install numpy pillow pyarrow fsspec aiohttp huggingface_hub
.venv/bin/python tools/synthid/build_templates.py   # 288 Gemini- und 959 OpenAI-Bilder (ca. 3,3 GB) -> src/background/synthid-templates/, synthid.json
.venv/bin/python tools/synthid/build_templates.py openai   # nur die OpenAI-Vorlagen
.venv/bin/python tools/synthid/crossval.py --transforms   # OpenAI: Vorlage aus zwei Dritteln, Test am übrigen Drittel
.venv/bin/python tools/synthid/fetch_openfake.py    # Testbilder nach tools/synthid/.cache/openfake (ca. 9 GB Download)
.venv/bin/python tools/synthid/eval.py --transforms tools/synthid/.cache/openfake/fake/nano-banana-pro
.venv/bin/python tools/synthid/eval.py tools/synthid/.cache/openfake/real/* tools/synthid/.cache/openfake/fake/*
.venv/bin/python tools/synthid/fetch_research.py   # zweiter Testdatensatz nach tools/synthid/.cache/synthid-research (ca. 5 GB)
.venv/bin/python tools/synthid/eval.py tools/synthid/.cache/synthid-research/*/
.venv/bin/python tools/synthid/eval.py --ignore-aspect tools/synthid/.cache/openfake/real/*   # jede Vorlage auf jedes Bild
```

`tools/synthid/synthid.py` ist die Referenzimplementierung von `src/background/synthid.ts`; ein Unit-Test prüft, dass beide dieselben z-Werte liefern. Die Schwellen stehen in `synthid.json` (`threshold` je Vorlage).

## Aufbau

```
src/background/   Bild-Download, Cache, Tab-Statistik und Toolbar-Icon; Worker mit Metadaten-Scan, SynthID-Prüfung, Vorverarbeitung und Modell
src/content/      Bildsuche (IntersectionObserver, MutationObserver) und Badge-Overlay im Shadow DOM
src/popup/        Toolbar-Popup
src/options/      Einstellungen
model.json        gepinntes Modell, Eingabeformat, Kalibrierung, Default-Schwelle
synthid.json      Herkunft, Größe, Skalierung und Schwelle der SynthID-Vorlagen (src/background/synthid-templates/)
tools/eval/       Testdaten-Download und Modellvergleich
tools/synthid/    Referenzimplementierung, Vorlagen-Bau und Messung der SynthID-Prüfung
```
