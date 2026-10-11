import { describe, expect, it } from 'vitest';
import { detectAiMetadata } from '../../src/background/metadata';

const IPTC = 'http://cv.iptc.org/newscodes/digitalsourcetype/';
const latin1 = (s: string) => new Uint8Array(Buffer.from(s, 'latin1'));
const jpegWith = (segment: string) => latin1(`\xff\xd8\xff\xe1${segment}\xff\xdb...image data...\xff\xd9`);
const pngWith = (keyword: string, text: string) => latin1(`\x89PNG\r\n\x1a\n....tEXt${keyword}\x00${text}....IDAT....IEND`);

describe('detectAiMetadata', () => {
  it('detects the IPTC digital source type for generated media in XMP or C2PA', () => {
    const xmp = `<rdf:Description Iptc4xmpExt:DigitalSourceType="${IPTC}trainedAlgorithmicMedia"/>`;
    expect(detectAiMetadata(jpegWith(xmp))).toBe('ai-generated');
  });

  it('reports AI edits separately from fully generated images', () => {
    expect(detectAiMetadata(jpegWith(`${IPTC}compositeWithTrainedAlgorithmicMedia`))).toBe('ai-edited');
  });

  it('reports SynthID when a C2PA action declares the watermark', () => {
    const action = `factionkc2pa.editedkdescriptionx(Applied imperceptible SynthID watermark.qdigitalSourceTypexF${IPTC}trainedAlgorithmicMedia`;
    expect(detectAiMetadata(pngWith('caBX', `jumdc2pa${action}`))).toBe('synthid');
  });

  it('does not treat captions that mention SynthID as a watermark declaration', () => {
    expect(detectAiMetadata(jpegWith('<dc:description>How SynthID watermarks work</dc:description>'))).toBeNull();
  });

  it('ignores camera captures and images without metadata', () => {
    expect(detectAiMetadata(jpegWith(`${IPTC}digitalCapture`))).toBeNull();
    expect(detectAiMetadata(latin1('\xff\xd8\xff\xdb\x00\x43plain\xff\xd9'))).toBeNull();
  });

  it('detects Stable Diffusion parameters in PNG text chunks', () => {
    const params = 'a cat in a hat\nNegative prompt: blurry\nSteps: 30, Sampler: DPM++ 2M, CFG scale: 7';
    expect(detectAiMetadata(pngWith('parameters', params))).toBe('stable-diffusion');
  });

  it('detects Stable Diffusion parameters in UTF-16 EXIF user comments', () => {
    const comment = Buffer.from('Steps: 20, Sampler: Euler a, CFG scale: 7', 'utf16le').toString('latin1');
    expect(detectAiMetadata(jpegWith(`Exif\x00\x00UNICODE\x00${comment}`))).toBe('stable-diffusion');
  });

  it('detects ComfyUI, NovelAI and InvokeAI metadata', () => {
    expect(detectAiMetadata(pngWith('prompt', '{"3": {"class_type": "KSampler", "inputs": {}}}'))).toBe('comfyui');
    expect(detectAiMetadata(pngWith('Software', 'NovelAI'))).toBe('novelai');
    expect(detectAiMetadata(pngWith('invokeai_metadata', '{"model": "sdxl"}'))).toBe('invokeai');
  });

  it('detects the GB 45438 AIGC label in PNG text chunks, XMP and EXIF user comments', () => {
    const label = '{"Label":"1","ContentProducer":"kling","ProduceID":"SGP_PROD_ai_web_300000000123456"}';
    expect(detectAiMetadata(pngWith('AIGC', label))).toBe('aigc-label');
    const xmp = `<rdf:Description xmlns:TC260="http://www.tc260.org.cn/ns/AIGC/1.0/"><TC260:AIGC>${label.replaceAll('"', '&quot;')}</TC260:AIGC>`;
    expect(detectAiMetadata(jpegWith(xmp))).toBe('aigc-label');
    const comment = Buffer.from(`{"AIGC":${label}}`, 'utf16le').toString('latin1');
    expect(detectAiMetadata(jpegWith(`Exif\x00\x00UNICODE\x00${comment}`))).toBe('aigc-label');
  });

  it('does not treat captions that mention AIGC as a label', () => {
    expect(detectAiMetadata(jpegWith('<dc:description>AIGC art, label: "1st place"</dc:description>'))).toBeNull();
  });

  it('does not treat ordinary prose about sampling as generator parameters', () => {
    expect(detectAiMetadata(pngWith('Description', 'Steps to take when sampling soil'))).toBeNull();
  });
});
