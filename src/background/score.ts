export type OutputSpec = 'sigmoid' | { softmaxLabels: string[] };

export interface Calibration {
  weights: number[];
  bias: number;
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export function aiLogit(logits: ArrayLike<number>, spec: OutputSpec, id2label: Record<string, string> = {}): number {
  const values = Array.from(logits, Number);
  if (spec === 'sigmoid') return values[0];
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const p = exps.reduce((acc, e, i) => (spec.softmaxLabels.includes(id2label[i]) ? acc + e / sum : acc), 0);
  const clamped = Math.min(Math.max(p, 1e-7), 1 - 1e-7);
  return Math.log(clamped / (1 - clamped));
}

export function calibratedProbability(logits: number[], calibration: Calibration): number {
  return sigmoid(logits.reduce((sum, z, i) => sum + calibration.weights[i] * z, calibration.bias));
}

export function modelProbabilities(logits: number[], calibration: Calibration): number[] {
  const total = calibration.weights.reduce((sum, w) => sum + w, 0);
  return logits.map((z) => sigmoid(calibration.bias + total * z));
}
