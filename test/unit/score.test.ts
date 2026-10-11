import { describe, expect, it } from 'vitest';
import { aiLogit, calibratedProbability } from '../../src/background/score';

describe('aiLogit', () => {
  it('passes a single sigmoid logit through', () => {
    expect(aiLogit([1.5], 'sigmoid')).toBe(1.5);
  });

  it('sums the probability of all AI labels for softmax outputs', () => {
    const logit = aiLogit([0, 0, 0, 0], { softmaxLabels: ['a', 'b'] }, { 0: 'a', 1: 'b', 2: 'real', 3: 'other' });
    expect(logit).toBeCloseTo(0);
  });
});

describe('calibratedProbability', () => {
  it('applies Platt scaling to a single model', () => {
    expect(calibratedProbability([0], { weights: [1], bias: 0 })).toBe(0.5);
    expect(calibratedProbability([-2], { weights: [0.5], bias: 1 })).toBe(0.5);
    expect(calibratedProbability([10], { weights: [1], bias: 0 })).toBeGreaterThan(0.9999);
  });

  it('combines several models with one weight each', () => {
    expect(calibratedProbability([2, -4], { weights: [1, 0.5], bias: 0 })).toBe(0.5);
    expect(calibratedProbability([2, 4], { weights: [1, 0.5], bias: -4 })).toBe(0.5);
  });
});
