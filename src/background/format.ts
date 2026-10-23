const ascii = (bytes: Uint8Array, offset: number, text: string) =>
  offset + text.length <= bytes.length && [...text].every((c, i) => bytes[offset + i] === c.charCodeAt(0));

export function isLossyWebp(bytes: Uint8Array): boolean {
  if (!ascii(bytes, 0, 'RIFF') || !ascii(bytes, 8, 'WEBP')) return false;
  for (let offset = 12; offset + 8 <= bytes.length; ) {
    if (ascii(bytes, offset, 'VP8 ')) return true;
    if (ascii(bytes, offset, 'VP8L') || ascii(bytes, offset, 'ALPH')) return false;
    const size = (bytes[offset + 4] | (bytes[offset + 5] << 8) | (bytes[offset + 6] << 16) | (bytes[offset + 7] << 24)) >>> 0;
    offset += 8 + size + (size & 1);
  }
  return false;
}
