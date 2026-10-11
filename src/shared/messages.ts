export type MetadataReason =
  | 'ai-generated'
  | 'ai-edited'
  | 'stable-diffusion'
  | 'comfyui'
  | 'novelai'
  | 'invokeai';

export type Verdict =
  | { ok: true; score: number; source: 'model' }
  | { ok: true; score: 1; source: 'metadata'; reason: MetadataReason }
  | { ok: false; error: 'fetch' | 'decode' | 'model' | 'unsupported' };

export type BackgroundRequest =
  | { type: 'classify'; url: string; bytes?: ArrayBuffer }
  | { type: 'page-stats'; flagged: number; initial: boolean };

export type ContentRequest = { type: 'get-stats' };

export interface PageStats {
  flagged: number;
  checked: number;
  active: boolean;
}

export function isFlagged(verdict: Verdict, threshold: number): boolean {
  return verdict.ok && (verdict.source === 'metadata' || verdict.score >= threshold);
}
