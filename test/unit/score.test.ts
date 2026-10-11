import { describe, expect, it } from 'vitest';
import { aiLogit, calibratedProbability, modelProbabilities } from '../../src/background/score';

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

describe('modelProbabilities', () => {
  const logOdds = (p: number) => Math.log(p / (1 - p));

  it('puts every model on the ensemble scale so that the ensemble is their weighted mean in log-odds', () => {
    const calibration = { weights: [0.3, 0.1], bias: 2 };
    const logits = [1.5, -6];
    const [a, b] = modelProbabilities(logits, calibration);
    expect(0.75 * logOdds(a) + 0.25 * logOdds(b)).toBeCloseTo(logOdds(calibratedProbability(logits, calibration)), 10);
  });

  it('equals the ensemble when all models agree', () => {
    const calibration = { weights: [0.2561, 0.2561], bias: 3.5385 };
    const ensemble = calibratedProbability([-8, -8], calibration);
    for (const p of modelProbabilities([-8, -8], calibration)) expect(p).toBeCloseTo(ensemble, 12);
  });
});
