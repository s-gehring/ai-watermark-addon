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
3. **Modell** (Badge „AI 99 %“): Ein Detektor, mit ONNX Runtime Web (WASM) im Worker ausgeführt: Detectra, auf OpenFake feinjustiert (44 MB, fp16). Ausgangspunkt ist [Detectra v3](https://huggingface.co/TensorFold/detectra-v1) (MIT-Lizenz), ein Feintuning des CommunityForensics-ViT-S auf neueren Generatoren (DALL·E 3, Midjourney, FLUX, SD 3.5, Recraft, HiDream), auf Bildern aus sozialen Netzwerken ([WildRF](https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld)) und mit echten Gemälden, Anime-Zeichnungen, Memes und Screenshots als Gegenbeispielen. Mit `tools/train/` ist es drei Epochen lang auf 15 Dateien aus dem Trainingssplit von [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) weitertrainiert (ca. 60.000 Bilder, je zur Hälfte echt und von vielen Generatoren), dazu 4.500 Gemälde aus WikiArt, 5.900 Fotos aus COCO und Diagramme aus ChartQA als Gegenbeispiele. Jedes Trainingsbild wird zufällig verkleinert und als JPEG neu komprimiert, damit das Modell die Bedingungen auf Webseiten kennt. Die Gewichte liegen nur lokal vor, weil OpenFake unter CC BY-NC 4.0 steht (siehe [Modell trainieren](#modell-trainieren)). Vorverarbeitung: bilineares Resize auf 440 px, Center-Crop 384 px.

   Die Vorverarbeitung ist in `src/background/preprocess.ts` nachgebaut und rechnet bitgenau wie Pillow. Der Rohwert wird per Platt-Scaling kalibriert (`model.json`); die Rohausgabe selbst wäre als Prozentwert ungeeignet, weil sie bei echten Fotos oft unter 0,1 % liegt. Unter Web-Bedingungen erkennt das Modell bei gleich vielen Fehlalarmen auf Fotos rund doppelt so viele AI-Bilder wie das bisher gebündelte CommunityForensics (48–49 % statt 23 %) und markiert kaum Gemälde (0,7 % statt 11 %, siehe [Genauigkeit](#genauigkeit)).

Das Background-Script lädt das Bild selbst noch einmal (meist aus dem HTTP-Cache), damit auch Bilder fremder Domains ohne CORS-Freigabe geprüft werden können. Metadaten-Scan, Dekodierung und Inferenz laufen in einem Web Worker, damit Popup und Background-Script reaktionsfähig bleiben. Ergebnisse werden pro URL zwischengespeichert. Ein Bild braucht in Firefox ca. 0,8 s; Bilder werden nacheinander geprüft, sichtbare zuerst.

## Genauigkeit

Gemessen mit `npm run eval` auf 4606 Bildern. Die eine Hälfte dient der Kalibrierung, die Zahlen unten stammen aus der anderen:

- AI-Bilder: 300 aus dem Testsplit von [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (18 Generatoren bis 2026), 188 aus OpenFakes Reddit-Testset (AI-Subreddits) und 600 von neueren Generatoren aus [bigdatamark/synthid-research](https://huggingface.co/datasets/bigdatamark/synthid-research) (je 150 von Gemini 3.1 Flash Image, Qwen-Image 2.1, FLUX.2 und GPT-Image 2)
- echte Fotos: 943 aus OpenFake (ImageNet, DOCCI), 375 aus Fotografie-Subreddits (OpenFake-Reddit-Testset), 1000 aus [COCO](https://huggingface.co/datasets/detection-datasets/coco) und 600 Vorschaubilder von YouTube-Videos aus dem Evalset von [AudioSet](https://research.google.com/audioset/) (hochgeladen bis Anfang 2017, je 300 mit und ohne Musik), so auf 16:9 zugeschnitten, wie YouTube sie in Listen zeigt
- Grafik und Kunst: 300 Diagramme aus [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) und 300 Gemälde aus [WikiArt](https://huggingface.co/datasets/huggan/wikiart)

Jedes Bild wird dreimal bewertet: als Originaldatei und so, wie Bilder auf Webseiten typischerweise ankommen, also auf höchstens 1280 px bzw. 800 px verkleinert und als JPEG mit Qualität 80 bzw. 75 gespeichert. Kalibrierung und Default-Schwelle sind so gewählt, dass unter Web-Bedingungen rund 1 % der echten Fotos markiert werden.

Bei der Default-Schwelle von 92 %, in Klammern das bisher gebündelte CommunityForensics bei seiner Default-Schwelle von 95 %:

| | Originaldatei | Web 1280 px | Web 800 px |
|---|---|---|---|
| AI-Bilder erkannt (OpenFake) | 70 % (40 %) | 66 % (30 %) | 65 % (27 %) |
| AI-Bilder erkannt (Reddit) | 72 % (31 %) | 66 % (28 %) | 64 % (29 %) |
| AI-Bilder erkannt (neuere Generatoren) | 64 % (43 %) | 33 % (14 %) | 38 % (18 %) |
| echte Fotos markiert (alle) | 1,1 % (0,8 %) | 1,0 % (0,8 %) | 1,0 % (0,8 %) |
| echte Fotos markiert (Fotografie-Subreddits) | 2,7 % (3,2 %) | 1,1 % (1,6 %) | 1,1 % (2,7 %) |
| echte Fotos markiert (DOCCI-Kamerafotos) | 4,4 % (2,5 %) | 5,0 % (1,9 %) | 3,8 % (0,6 %) |
| YouTube-Vorschaubilder markiert | 1,0 % (0 %) | 0,7 % (0 %) | 1,4 % (0,3 %) |
| Gemälde markiert | 0,7 % (11 %) | 0,7 % (11 %) | 0,7 % (11 %) |
| Diagramme markiert | 0 % (0 %) | 0 % (0 %) | 0 % (0 %) |

Andere Schwellen (Web 1280 px, alle AI-Bilder gegen echte Fotos):

| Schwelle | AI-Bilder erkannt | echte Fotos markiert |
|---|---|---|
| 50 % | 80 % | 12 % |
| 70 % | 66 % | 5,5 % |
| 80 % | 61 % | 3,1 % |
| 90 % | 51 % | 1,4 % |
| **92 % (Default)** | **47 %** | **1,0 %** |
| 97 % | 39 % | 0,7 % |

Vergleich mit anderen lokalen Modellen unter Web-Bedingungen (1280 px und 800 px), erkannte AI-Bilder bei genau 1 % markierten Fotos:

| Modell | AUC | AI-Bilder erkannt |
|---|---|---|
| **Detectra, auf OpenFake feinjustiert (gebündelt)** | **0,92–0,93** | **48–49 %** |
| dasselbe mit fünf Ausschnitten pro Bild, siehe unten | 0,94 | 51–53 % |
| [Detectra v3](https://huggingface.co/TensorFold/detectra-v1) ohne Feintuning (fp16) | 0,89–0,90 | 33–34 % |
| [fev v13](https://huggingface.co/angelhd25/ull-ai-image-detector) (fp32, 88 MB), siehe unten | 0,90 | 37–39 % |
| CommunityForensics (int8, bisher gebündelt) | 0,85 | 23 % |
| CommunityForensics ohne Quantisierung (fp32, viermal so groß) | 0,89 | 25 % |
| [SigLIP-Detektor](https://huggingface.co/Ateeqq/ai-vs-human-image-detector) (int8) | 0,71 | 8–9 % |
| SMOGY, ViT-Deepfake-Klassifikator | 0,55–0,62 | 0 % |
| [capcheck/ai-image-detection](https://huggingface.co/onnx-community/ai-image-detection-ONNX), [ai-image-detect-distilled](https://huggingface.co/onnx-community/ai-image-detect-distilled-ONNX) | 0,46–0,48 | 0–1 % |

Der SigLIP-Detektor hält Vorschaubilder von YouTube-Videos oft für AI-generiert, auch die bekannter alter Musikvideos: Das Originalmodell in PyTorch gibt bei 23 von 26 getesteten über 99 % aus, es liegt also nicht an Quantisierung oder Vorverarbeitung. Bei 1 % Fehlalarmen auf Fotos insgesamt markiert er 2–4 % der YouTube-Vorschaubilder.

Einschränkungen:

- Das Badge ist ein Hinweis, kein Beweis. Die Default-Schwelle ist auf wenige Fehlalarme ausgelegt, dafür bleibt ohne Metadaten oder Wasserzeichen rund die Hälfte der AI-Bilder unmarkiert. Wer mehr erkennen will, senkt die Schwelle in den Einstellungen und nimmt mehr Fehlalarme in Kauf.
- Die neuesten Generatoren erkennt das Modell schlechter. Unter Web-Bedingungen (1280 px) findet es 44 von 78 FLUX.2-Bildern, 19 von 70 GPT-Image-2-Bildern, 14 von 73 Qwen-Image-2.1-Bildern und 24 von 85 Gemini-Bildern; CommunityForensics fand 14 bis 18 % dieser Bilder, von Gemini 1 bis 3. Als Originaldatei erkennt es 64 %, Kompression kostet hier also besonders viel. Gemini-Bilder findet zusätzlich die SynthID-Prüfung.
- Hochwertige Kamerafotos wie die aus DOCCI markiert das Modell häufiger als CommunityForensics, unter Web-Bedingungen 4 bis 5 % statt 1 bis 2 %, YouTube-Vorschaubilder zu rund 1 % statt fast nie. Stark nachbearbeitete Fotos aus Fotografie-Subreddits markiert es etwas seltener (1 % statt 2 bis 3 %). 3D-Renderings sind nicht im Testset.
- Detectra und das Feintuning nutzen WikiArt-Gemälde als Gegenbeispiele, und die Gemälde dieser Messung stammen ebenfalls aus WikiArt (aus anderen Dateien als beim Feintuning). Der Wert für markierte Gemälde ist deshalb vermutlich zu niedrig.
- Das Feintuning nutzt den Trainingssplit von OpenFake, COCO-Train und WikiArt-Dateien, die in der Messung nicht vorkommen. Die Messung nutzt den Testsplit mit anderen Generatoren und anderen Fotoquellen (ImageNet und DOCCI statt LAION und Pexels). Ausgewählt wurde der Checkpoint aber anhand dieser Messung (aus wenigen Varianten), die Zahlen sind deshalb leicht optimistisch.
- Kompression und Verkleinerung kosten Erkennung (AUC 0,96 als Originaldatei, 0,92 bis 0,93 unter Web-Bedingungen).
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

- **Ensembles mit anderen Modellen** (Detectra v3, [CommunityForensics-DeepfakeDet-ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT), [ai-vs-human-SigLIP-ONNX](https://huggingface.co/neilxxx/ai-vs-human-SigLIP-ONNX)): Zusätzlich zum feinjustierten Modell bringen diese Modelle, einzeln oder zusammen, bei gleich vielen Fehlalarmen keine zusätzlichen Treffer, kosten aber je 0,7 s pro Bild (zusammen 155 MB).
- **Fünf Ausschnitte pro Bild:** Mitte und vier Ecken aus dem auf 640 px kurze Kante verkleinerten Bild, davon der höchste Wert. Unter Web-Bedingungen erkennt das 1 bis 5 Prozentpunkte mehr AI-Bilder, braucht aber die fünffache Rechenzeit und markiert mehr Kamerafotos (DOCCI bis 7 %) und bearbeitete Subreddit-Fotos (bis 5 %).
- **fev v13** ([angelhd25/ull-ai-image-detector](https://huggingface.co/angelhd25/ull-ai-image-detector), MIT-Lizenz), wie Detectra ein Feintuning des CommunityForensics-ViT-S für ein Browser-Addon: Allein erkennt es unter Web-Bedingungen 37–39 % der AI-Bilder, markiert aber 7 % der Gemälde. Im Ensemble mit Detectra v3 brachte es bei höchstens 5 % markierten Gemälden keine zusätzlichen Treffer, kostet aber 88 MB und 0,7 s pro Bild. Es wurde zudem teils auf dem Validierungssplit von OpenFake trainiert, der unter CC BY-NC 4.0 steht. Dessen Generatoren und Fotoquellen kommen im Testsplit dieser Messung nicht vor.
- **Weitere Detektoren von Hugging Face** (capcheck/ai-image-detection, ai-image-detect-distilled, SMOGY u. a.): Viele sind auf kleinen Datensätzen wie CIFAKE trainiert und unterscheiden AI-Bilder unter Web-Bedingungen kaum oder gar nicht besser als Zufall (AUC 0,46 bis 0,62). Größere Modelle wie SwinV2-Large-Detektoren oder SigLIP-DINOv2-Ensembles wären mit 200 MB bis 2 GB zu groß für das Addon.

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
- Mauszeiger auf einem Symbol: Ein Panel zeigt das Ergebnis jeder Prüfung (Metadaten, SynthID-Wasserzeichen und AI-Score des Modells mit Schwelle)
- Klick auf ein Symbol blendet es aus
- Einstellungen (`about:addons` → AI Watermark → Einstellungen): Mindestgröße, Schwelle, Metadaten-Prüfung, SynthID-Prüfung der Pixel, CSS-Hintergründe, Prozentanzeige, Symbole für unauffällige Bilder und laufende Prüfungen, deaktivierte Seiten

## Entwicklung

Voraussetzungen: Node 22, Firefox.

```bash
npm ci
npm run fetch-model   # prüft das Modell in vendor/models/ (SHA-256), siehe Modell trainieren
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

Das Paket ist ca. 49 MB groß (Modell, WASM-Runtime und SynthID-Vorlagen), unter dem AMO-Limit von 200 MB.

### Modell evaluieren oder tauschen

```bash
python3 -m venv .venv && .venv/bin/pip install pyarrow fsspec aiohttp huggingface_hub pillow
.venv/bin/python tools/eval/fetch_samples.py
npm run eval -- bundled --calibrate --per-generator
npm run eval -- all
npm run eval -- members --members=detectra-openfake,angelhd25/ull-ai-image-detector   # beliebige Kombination fitten und messen
```

`fetch_samples.py` liest nur einzelne Parquet-Row-Groups aus OpenFake, COCO, ChartQA und WikiArt und 600 Dateien aus bigdatamark/synthid-research, dazu 600 YouTube-Vorschaubilder von `i.ytimg.com` (ca. 5 GB Download statt der kompletten Datensätze, 3,3 GB auf der Platte). Testdaten und Rohwerte liegen immer im Haupt-Checkout unter `tools/eval/.cache/`, auch wenn die Skripte in einem Git-Worktree laufen, damit sie das Löschen des Worktrees überstehen. `npm run eval` speichert die Rohwerte pro Modell in `tools/eval/.cache/scores/`, weitere Läufe rechnen nur neue Bilder. `--calibrate` wählt auf der Kalibrierhälfte bei mehreren gebündelten Modellen deren Gewichtung (möglichst viele erkannte AI-Bilder bei höchstens 5 % markierten Gemälden), fittet Platt-Scaling und wählt die Schwelle für 1 % markierte Fotos unter Web-Bedingungen; die Werte gehören dann nach `model.json` (`calibration`, `threshold`). Ändert sich die Kalibrierung, setzt das Addon eine selbst gewählte Schwelle auf den neuen Default zurück, weil die Prozentwerte danach etwas anderes bedeuten.

### Modell trainieren

Braucht eine CUDA-GPU mit 12 GB (gemessen auf einer RTX 4070 Ti) und ca. 95 GB Platz.

```bash
.venv/bin/pip install torch timm safetensors pyarrow pillow numpy huggingface_hub onnx onnxruntime onnxconverter-common
.venv/bin/python tools/train/fetch_data.py   # Detectra-Gewichte, 16 OpenFake-Dateien und Gegenbeispiele nach tools/train/.cache/
.venv/bin/python tools/train/train.py --name full   # 3 Epochen, ca. 1 h
.venv/bin/python tools/train/export_onnx.py tools/train/.cache/runs/full/epoch3.safetensors   # -> vendor/models/detectra-openfake/model.onnx
```

Die Datensätze sind auf feste Revisionen gepinnt. Die letzte der 16 OpenFake-Dateien dient als Holdout; `train.py` gibt nach jeder Epoche AUC und Erkennungsrate bei 1 % Fehlalarmen darauf aus. Das Training ist nicht bitgenau reproduzierbar: Nach einem neuen Lauf gehört der von `export_onnx.py` ausgegebene SHA-256 nach `model.json`, und `npm run eval -- bundled --calibrate` liefert die neue Kalibrierung. Weil OpenFake unter CC BY-NC 4.0 steht, sind die Gewichte nicht veröffentlicht und `npm run fetch-model` kann sie nicht laden. CI prüft deshalb nur Typen und Unit-Tests.

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
model.json        Modell, Eingabeformat, Kalibrierung, Default-Schwelle
synthid.json      Herkunft, Größe, Skalierung und Schwelle der SynthID-Vorlagen (src/background/synthid-templates/)
tools/eval/       Testdaten-Download und Modellvergleich
tools/train/      Feintuning des Modells (Daten-Download, Training, ONNX-Export)
tools/synthid/    Referenzimplementierung, Vorlagen-Bau und Messung der SynthID-Prüfung
```
