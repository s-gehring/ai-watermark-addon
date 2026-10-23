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

  it('detects the EXIF fields and the C2PA manifest written by the Stability AI API', () => {
    const exif = 'MM\x00*\x00\x00\x00\x08\x00\x06a cat in a hat\x00stability-diffusion\x00stability.ai\x00\x007Y+?H0sw#B004Kl8\x00';
    expect(detectAiMetadata(latin1(`\x89PNG\r\n\x1a\n....eXIf${exif}....IDAT....IEND`))).toBe('stability-ai');
    const manifest = 'jumdc2pa....claim_generatorx8Stability_ContentCredentials_Service/v0.1 c2pa-rs/0.31.3';
    expect(detectAiMetadata(pngWith('caBX', manifest))).toBe('stability-ai');
  });

  it('detects the EXIF signature written by Grok and the EXIF make written by Ideogram', () => {
    const signature = 'R4e3GBwF6cxj8ZhUQFpJt1MRd67IC/+'.repeat(10);
    expect(detectAiMetadata(jpegWith(`Exif\x00\x00II*\x00ASCII\x00\x00\x00Signature: ${signature}==\x00`))).toBe('grok');
    expect(detectAiMetadata(jpegWith('Exif\x00\x00II*\x00\x00\x00\x00\x00Ideogram AI\x002025:11:16 11:28:42\x00'))).toBe('ideogram');
  });

  it('does not treat short signatures or captions naming Ideogram as generator metadata', () => {
    expect(detectAiMetadata(jpegWith('Exif\x00\x00II*\x00Signature: John Doe\x00'))).toBeNull();
    expect(detectAiMetadata(jpegWith('<dc:description>Logo made with Ideogram AI tools</dc:description>'))).toBeNull();
  });

  it('does not treat a mention of Stability AI as its API metadata', () => {
    expect(detectAiMetadata(pngWith('Comment', 'Upscaled with stability.ai, model stability-diffusion'))).toBeNull();
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

  it('detects ComfyUI graphs with custom samplers and workflow-only chunks', () => {
    const prompt = '{"7": {"inputs": {"seed": 1}, "class_type": "ClownsharKSampler_Beta", "_meta": {"title": "Sampler"}}}';
    expect(detectAiMetadata(pngWith('prompt', prompt))).toBe('comfyui');
    expect(detectAiMetadata(pngWith('prompt', '{"4": {"inputs": {}, "class_type": "UNETLoader"}}'))).toBe('comfyui');
    const workflow = '{"id": "b8437c87", "revision": 0, "last_node_id": 558, "last_link_id": 1069, "nodes": []}';
    expect(detectAiMetadata(pngWith('workflow', workflow))).toBe('comfyui');
  });

  it('detects ComfyUI graphs in UTF-16 EXIF user comments', () => {
    const graph = Buffer.from('{"23": {"class_type": "KSampler", "inputs": {}}}', 'utf16le').toString('latin1');
    expect(detectAiMetadata(jpegWith(`Exif\x00\x00UNICODE\x00${graph}`))).toBe('comfyui');
  });

  it('does not treat other JSON with a class_type key as a ComfyUI graph', () => {
    expect(detectAiMetadata(pngWith('Comment', '{"class_type": "Invoice", "inputs": {}}'))).toBeNull();
    expect(detectAiMetadata(pngWith('Comment', '{"last_node_id": 3}'))).toBeNull();
  });

  it('detects the EXIF artist written by the Civitai generator', () => {
    const uuid = '2d569de9-2123-456b-832c-83d04103880a';
    const exif = `Exif\x00\x00II*\x00\x08\x00\x00\x00\x02\x00\x31\x01\x02\x00\x25\x00\x00\x00\x26\x00\x00\x00\x3b\x01\x02\x00\x03\x00\x00\x00ai\x00\x00\x00\x00\x00\x00${uuid}\x00`;
    expect(detectAiMetadata(jpegWith(exif))).toBe('civitai');
  });

  it('does not treat other artists or the word "ai" in captions as the Civitai marker', () => {
    const artist = (name: string) => `Exif\x00\x00II*\x00\x08\x00\x00\x00\x01\x00\x3b\x01\x02\x00${String.fromCharCode(name.length + 1)}\x00\x00\x00${name}\x00\x00\x00\x00\x00`;
    expect(detectAiMetadata(jpegWith(artist('aiden')))).toBeNull();
    expect(detectAiMetadata(jpegWith(artist('Maist')))).toBeNull();
    expect(detectAiMetadata(jpegWith('<dc:creator>ai</dc:creator>'))).toBeNull();
  });

  it('detects the EXIF software written by Reve', () => {
    const exif = 'Exif\x00\x00II*\x00\x08\x00\x00\x00\x01\x00\x31\x01\x02\x00\x09\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00Reve.art\x00';
    expect(detectAiMetadata(jpegWith(exif))).toBe('reve');
    expect(detectAiMetadata(jpegWith('<dc:description>Made after reading Reve.art</dc:description>'))).toBeNull();
  });

  it('does not treat ordinary prose about sampling as generator parameters', () => {
    expect(detectAiMetadata(pngWith('Description', 'Steps to take when sampling soil'))).toBeNull();
  });
});
