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
3. **Modelle** (Badge „AI 99 %“): Zwei Detektoren, beide int8-quantisiert und mit ONNX Runtime Web (WASM) im Background-Script ausgeführt:
   - [CommunityForensics-DeepfakeDet-ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT) (ViT-S, CVPR 2025, MIT-Lizenz, 23 MB), Vorverarbeitung: Bicubic-Resize auf 440 px, Center-Crop 384 px
   - [ai-vs-human-SigLIP-ONNX](https://huggingface.co/neilxxx/ai-vs-human-SigLIP-ONNX) (88 MB), laut `config.json` eine ONNX-Konvertierung von [Ateeqq/ai-vs-human-image-detector](https://huggingface.co/Ateeqq/ai-vs-human-image-detector) (SigLIP2, Apache-2.0), Vorverarbeitung: bilineares Resize auf 224 × 224 px

   Beide Vorverarbeitungen sind in `src/background/preprocess.ts` nachgebaut und rechnen bitgenau wie Pillow in den Referenz-Pipelines der Modelle. Die Rohwerte werden gewichtet addiert und per Platt-Scaling kalibriert (`model.json`). Zusammen erkennen die Modelle unter Web-Bedingungen bei gleich vielen Fehlalarmen auf Fotos mehr AI-Bilder und markieren nur rund halb so viele Gemälde wie das erste Modell allein. Das Detail-Panel eines Symbols zeigt pro Modell den Wert, den das Ensemble hätte, wenn alle Modelle diesem einen zustimmten (gleiche Kalibrierung, nur dieser Rohwert). Der Gesamtwert ist das mit den Modellgewichten gewichtete Mittel dieser Werte in Log-Odds und liegt deshalb immer zwischen ihnen. Die rohen Modellausgaben selbst wären dafür ungeeignet: Bei echten Fotos liegen sie oft unter 0,1 %, auch wenn der kalibrierte Gesamtwert 30 % beträgt.

Das Background-Script lädt das Bild selbst noch einmal (meist aus dem HTTP-Cache), damit auch Bilder fremder Domains ohne CORS-Freigabe geprüft werden können. Metadaten-Scan, Dekodierung und Inferenz laufen in einem Web Worker, damit Popup und Background-Script reaktionsfähig bleiben. Ergebnisse werden pro URL zwischengespeichert. Ein Bild braucht in Firefox ca. 1,5 s, je Modell etwa die Hälfte; Bilder werden nacheinander geprüft, sichtbare zuerst.

## Genauigkeit

Gemessen mit `npm run eval` auf 4006 Bildern. Die eine Hälfte dient der Kalibrierung, die Zahlen unten stammen aus der anderen:

- AI-Bilder: 300 aus dem Testsplit von [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (18 Generatoren bis 2026), 188 aus OpenFakes Reddit-Testset (AI-Subreddits) und 600 von neueren Generatoren aus [bigdatamark/synthid-research](https://huggingface.co/datasets/bigdatamark/synthid-research) (je 150 von Gemini 3.1 Flash Image, Qwen-Image 2.1, FLUX.2 und GPT-Image 2)
- echte Fotos: 943 aus OpenFake (ImageNet, DOCCI), 375 aus Fotografie-Subreddits (OpenFake-Reddit-Testset) und 1000 aus [COCO](https://huggingface.co/datasets/detection-datasets/coco)
- Grafik und Kunst: 300 Diagramme aus [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) und 300 Gemälde aus [WikiArt](https://huggingface.co/datasets/huggan/wikiart)

Jedes Bild wird dreimal bewertet: als Originaldatei und so, wie Bilder auf Webseiten typischerweise ankommen, also auf höchstens 1280 px bzw. 800 px verkleinert und als JPEG mit Qualität 80 bzw. 75 gespeichert. Die Gewichtung der beiden Modelle ist so gewählt, dass auf der Kalibrierhälfte möglichst viele AI-Bilder erkannt und höchstens 5 % der Gemälde markiert werden. Kalibrierung und Default-Schwelle sind so gewählt, dass unter Web-Bedingungen rund 1 % der echten Fotos markiert werden.

Bei der Default-Schwelle von 98 %, in Klammern das CommunityForensics-Modell allein, ebenfalls auf 1 % markierte Fotos kalibriert:

| | Originaldatei | Web 1280 px | Web 800 px |
|---|---|---|---|
| AI-Bilder erkannt (OpenFake) | 44 % (42 %) | 43 % (32 %) | 40 % (30 %) |
| AI-Bilder erkannt (Reddit) | 44 % (33 %) | 45 % (28 %) | 42 % (32 %) |
| AI-Bilder erkannt (neuere Generatoren) | 21 % (47 %) | 13 % (16 %) | 15 % (19 %) |
| echte Fotos markiert (alle) | 0,9 % (1,0 %) | 1,0 % (1,0 %) | 1,1 % (1,2 %) |
| echte Fotos markiert (Fotografie-Subreddits) | 2,7 % (3,2 %) | 2,7 % (1,6 %) | 3,7 % (3,2 %) |
| Gemälde markiert | 7 % (12 %) | 7 % (11 %) | 7 % (13 %) |
| Diagramme markiert | 0 % (0 %) | 0 % (0 %) | 0 % (0 %) |

Andere Schwellen (Web 1280 px, alle AI-Bilder gegen echte Fotos):

| Schwelle | AI-Bilder erkannt | echte Fotos markiert |
|---|---|---|
| 50 % | 70 % | 15 % |
| 70 % | 54 % | 6,9 % |
| 80 % | 47 % | 5,7 % |
| 90 % | 37 % | 3,6 % |
| 97 % | 30 % | 1,5 % |
| **98 % (Default)** | **26 %** | **1,0 %** |

Vergleich mit anderen lokalen Modellen unter Web-Bedingungen (1280 px und 800 px), erkannte AI-Bilder bei genau 1 % markierten Fotos:

| Modell | AUC | AI-Bilder erkannt |
|---|---|---|
| **Ensemble (gebündelt)** | **0,87** | **24–26 %** |
| CommunityForensics allein (int8) | 0,86–0,87 | 20–22 % |
| CommunityForensics ohne Quantisierung (fp32, viermal so groß) | 0,89–0,90 | 22–24 % |
| SigLIP-Detektor allein (int8) | 0,77 | 22–23 % |
| SMOGY, ViT-Deepfake-Klassifikator | 0,57–0,67 | 0–8 % |

Einschränkungen:

- Das Badge ist ein Hinweis, kein Beweis. Die Default-Schwelle ist auf wenige Fehlalarme ausgelegt, dafür bleibt ohne Metadaten oder Wasserzeichen mehr als die Hälfte der AI-Bilder unmarkiert. Wer mehr erkennen will, senkt die Schwelle in den Einstellungen und nimmt mehr Fehlalarme in Kauf.
- Die neuesten Generatoren erkennen die Modelle schlecht, das Ensemble als Originaldatei sogar schlechter als das erste Modell allein. Gemini-Bilder (3 von 85) findet fast nur die SynthID-Prüfung.
- Gemälde werden öfter markiert als Fotos (rund jedes fünfzehnte), stark nachbearbeitete Fotos wie aus Fotografie-Subreddits zu bis zu 4 %. 3D-Renderings sind nicht im Testset.
- Kompression und Verkleinerung kosten Erkennung (AUC 0,94 als Originaldatei, 0,87 unter Web-Bedingungen).
- Kalibrierung und Messung nutzen getrennte Bilder, aber dieselben Quellen. Seiten mit anderer Bildauswahl oder eigener Kompression können abweichen.

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
| violettes „AI 99 %“ bzw. „AI · Metadaten“ | wahrscheinlich bzw. laut Metadaten AI-generiert |
| violettes „AI · SynthID“ | SynthID-Wasserzeichen (Google oder OpenAI) in den Pixeln oder im C2PA-Manifest gefunden |
| grüner Haken | geprüft, unauffällig |
| graues „?“ | konnte nicht geprüft werden (Download, Format oder Seitenverhältnis) |

- Toolbar-Button: Zahl der AI-Bilder auf der Seite; solange noch geprüft wird, ist das Icon gedimmt und zeigt einen Fortschrittsbalken
- Popup: Fortschritt („Prüfe Bilder: 12 von 30“), Zähler für unauffällige und nicht prüfbare Bilder, Addon global oder pro Seite an- und ausschalten
- Mauszeiger auf einem Symbol: Ein Panel zeigt das Ergebnis jeder Prüfung (Metadaten, SynthID-Wasserzeichen und die Modelle mit Einzelwerten, Gesamtwert, Schwelle und Gewichtung)
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

Das Paket ist ca. 90 MB groß (zwei Modelle, WASM-Runtime und SynthID-Vorlagen), unter dem AMO-Limit von 200 MB.

### Modell evaluieren oder tauschen

```bash
python3 -m venv .venv && .venv/bin/pip install pyarrow fsspec aiohttp huggingface_hub
.venv/bin/python tools/eval/fetch_samples.py
npm run eval -- bundled --calibrate --per-generator
npm run eval -- all
```

`fetch_samples.py` liest nur einzelne Parquet-Row-Groups aus OpenFake, COCO, ChartQA und WikiArt und 600 Dateien aus bigdatamark/synthid-research (ca. 5 GB Download statt der kompletten Datensätze, 3,3 GB auf der Platte). Testdaten und Rohwerte liegen immer im Haupt-Checkout unter `tools/eval/.cache/`, auch wenn die Skripte in einem Git-Worktree laufen, damit sie das Löschen des Worktrees überstehen. `npm run eval` speichert die Rohwerte pro Modell in `tools/eval/.cache/scores/`, weitere Läufe rechnen nur neue Bilder. `--calibrate` wählt auf der Kalibrierhälfte die Gewichtung der Modelle (möglichst viele erkannte AI-Bilder bei höchstens 5 % markierten Gemälden), fittet Platt-Scaling und wählt die Schwelle für 1 % markierte Fotos unter Web-Bedingungen; die Werte gehören dann nach `model.json` (`calibration`, `threshold`). Ändert sich die Kalibrierung, setzt das Addon eine selbst gewählte Schwelle auf den neuen Default zurück, weil die Prozentwerte danach etwas anderes bedeuten.

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
