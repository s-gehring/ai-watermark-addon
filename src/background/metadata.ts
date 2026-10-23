import type { MetadataReason } from '../shared/messages';

const FULL_SCAN_LIMIT = 8 * 1024 * 1024;
const HEAD_BYTES = 4 * 1024 * 1024;
const TAIL_BYTES = 1024 * 1024;
const EXIF_REGION = 512 * 1024;

const QUOTE = '(?:"|&quot;)';
const AIGC_LABEL = new RegExp(`AIGC[\\s\\S]{0,256}?${QUOTE}Label${QUOTE}\\s*:\\s*${QUOTE}?[123]\\b`);

const COMFYUI = /"class_type"\s*:\s*"(?:[^"]*Sampler[^"]*|CLIPTextEncode|CheckpointLoader\w*|UNETLoader|VAEDecode)"/;
const COMFYUI_WORKFLOW = /"last_node_id"\s*:\s*\d+\s*,\s*"last_link_id"\s*:\s*\d+/;

const PATTERNS: ReadonlyArray<[MetadataReason, RegExp]> = [
  ['synthid', /c2pa\.\w+[\s\S]{0,80}?\bSynthID\b/],
  ['ai-edited', /compositeWithTrainedAlgorithmicMedia/],
  ['ai-generated', /\btrainedAlgorithmicMedia/],
  ['stable-diffusion', /parameters\x00[\s\S]{0,16384}?Steps: \d+,\s*Sampler: /],
  ['comfyui', COMFYUI],
  ['comfyui', COMFYUI_WORKFLOW],
  ['novelai', /Software\x00NovelAI/],
  ['invokeai', /(?:invokeai_metadata|invokeai_graph|sd-metadata)\x00/],
  ['stability-ai', /stability-diffusion\x00stability\.ai\x00|Stability_ContentCredentials_Service/],
  ['grok', /Signature: [A-Za-z0-9+/]{200,}={0,2}\x00/],
  ['ideogram', /Ideogram AI\x00/],
  ['aigc-label', AIGC_LABEL],
  ['civitai', /\x3b\x01\x02\x00\x03\x00\x00\x00ai\x00\x00/],
  ['reve', /Reve\.art\x00/],
];

function latin1(bytes: Uint8Array): string {
  return new TextDecoder('latin1').decode(bytes);
}

function scannedText(bytes: Uint8Array): string {
  if (bytes.length <= FULL_SCAN_LIMIT) return latin1(bytes);
  return latin1(bytes.subarray(0, HEAD_BYTES)) + latin1(bytes.subarray(bytes.length - TAIL_BYTES));
}

export function detectAiMetadata(bytes: Uint8Array): MetadataReason | null {
  const text = scannedText(bytes);
  for (const [reason, pattern] of PATTERNS) {
    if (pattern.test(text)) return reason;
  }
  const exifText = text.slice(0, EXIF_REGION).replaceAll('\x00', '');
  if (/Steps: \d+,\s*Sampler: /.test(exifText)) return 'stable-diffusion';
  if (AIGC_LABEL.test(exifText)) return 'aigc-label';
  if (COMFYUI.test(exifText)) return 'comfyui';
  return null;
}
