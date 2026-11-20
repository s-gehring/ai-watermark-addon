import type { MetadataReason } from '../shared/messages';

const FULL_SCAN_LIMIT = 8 * 1024 * 1024;
const HEAD_BYTES = 4 * 1024 * 1024;
const TAIL_BYTES = 1024 * 1024;
const EXIF_REGION = 512 * 1024;

const QUOTE = '(?:"|&quot;)';
const AIGC_LABEL = new RegExp(`AIGC[\\s\\S]{0,256}?${QUOTE}Label${QUOTE}\\s*:\\s*${QUOTE}?[123]\\b`);

interface Matcher {
  test(text: string): boolean;
}

const COMFYUI_NODE = /"class_type"\s*:\s*(?="([^"]*)")/g;
const COMFYUI_NODE_NAME = /Sampler|^(?:CLIPTextEncode|CheckpointLoader\w*|UNETLoader|VAEDecode)$/;
const COMFYUI: Matcher = {
  test(text) {
    for (const [, name] of text.matchAll(COMFYUI_NODE)) {
      if (COMFYUI_NODE_NAME.test(name)) return true;
    }
    return false;
  },
};
const COMFYUI_WORKFLOW = /"last_node_id"\s*:\s*\d+\s*,\s*"last_link_id"\s*:\s*\d+/;

const SD_SETTINGS = /Steps: \d+,\s*Sampler: /;
const SD_SETTINGS_ALL = new RegExp(SD_SETTINGS, 'g');
const SD_PNG_KEYWORD = 'parameters\x00';
const SD_PNG_WINDOW = 16384;
const SD_PNG: Matcher = {
  test(text) {
    let keyword = -1;
    let next = text.indexOf(SD_PNG_KEYWORD);
    if (next < 0) return false;
    for (const { index } of text.matchAll(SD_SETTINGS_ALL)) {
      while (next >= 0 && next + SD_PNG_KEYWORD.length <= index) {
        keyword = next;
        next = text.indexOf(SD_PNG_KEYWORD, next + SD_PNG_KEYWORD.length);
      }
      if (keyword >= 0 && index - keyword - SD_PNG_KEYWORD.length <= SD_PNG_WINDOW) return true;
    }
    return false;
  },
};

const PATTERNS: ReadonlyArray<[MetadataReason, Matcher]> = [
  ['synthid', /c2pa\.\w+[\s\S]{0,80}?\bSynthID\b/],
  ['ai-edited', /compositeWithTrainedAlgorithmicMedia/],
  ['ai-generated', /\btrainedAlgorithmicMedia/],
  ['stable-diffusion', SD_PNG],
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
  if (SD_SETTINGS.test(exifText)) return 'stable-diffusion';
  if (AIGC_LABEL.test(exifText)) return 'aigc-label';
  if (COMFYUI.test(exifText)) return 'comfyui';
  return null;
}
