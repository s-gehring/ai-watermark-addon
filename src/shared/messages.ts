export type MetadataReason =
  | 'synthid'
  | 'ai-generated'
  | 'ai-edited'
  | 'stable-diffusion'
  | 'comfyui'
  | 'novelai'
  | 'invokeai'
  | 'aigc-label';

export type SynthIdProvider = 'google' | 'openai';

export type Verdict =
  | { ok: true; score: number; source: 'model' }
  | { ok: true; score: 1; source: 'metadata'; reason: MetadataReason }
  | { ok: true; score: 1; source: 'synthid'; provider: SynthIdProvider }
  | { ok: false; error: 'fetch' | 'decode' | 'model' | 'unsupported' };

export interface FrameStats {
  flagged: number;
  clean: number;
  pending: number;
  failed: number;
}

export type BackgroundRequest =
  | { type: 'classify'; url: string; bytes?: ArrayBuffer }
  | { type: 'page-stats'; stats: FrameStats; initial: boolean };

export type ContentRequest = { type: 'get-status' };

export interface PageStatus {
  active: boolean;
}

export const EMPTY_STATS: Readonly<FrameStats> = { flagged: 0, clean: 0, pending: 0, failed: 0 };

export function isFlagged(verdict: Verdict, threshold: number): boolean {
  return verdict.ok && (verdict.source !== 'model' || verdict.score >= threshold);
}

export function sumStats(all: Iterable<FrameStats>): FrameStats {
  const total = { ...EMPTY_STATS };
  for (const s of all) {
    total.flagged += s.flagged;
    total.clean += s.clean;
    total.pending += s.pending;
    total.failed += s.failed;
  }
  return total;
}

export function checkedCount(stats: FrameStats): number {
  return stats.flagged + stats.clean + stats.failed;
}

export function progressOf(stats: FrameStats): number | null {
  if (stats.pending === 0) return null;
  return checkedCount(stats) / (checkedCount(stats) + stats.pending);
}

export const tabStatsKey = (tabId: number) => `tab:${tabId}`;
