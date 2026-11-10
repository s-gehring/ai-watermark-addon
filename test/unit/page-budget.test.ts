import { describe, expect, it, vi } from 'vitest';
import { PageBudget } from '../../src/background/page-budget';

const file = (bytes: number) => vi.fn(async () => ({ bytes: new ArrayBuffer(bytes) }));

describe('PageBudget', () => {
  it('loads files until the bytes of a page view are spent', async () => {
    const budget = new PageBudget(100);
    const load = file(60);
    expect(await budget.load(1, 0, 'https://a.example/', load)).not.toBeNull();
    expect(await budget.load(1, 0, 'https://a.example/', load)).not.toBeNull();
    expect(await budget.load(1, 0, 'https://a.example/', load)).toBeNull();
    expect(load).toHaveBeenCalledTimes(2);
  });

  it('does not count files that failed to load', async () => {
    const budget = new PageBudget(100);
    await budget.load(1, 0, 'https://a.example/', async () => null);
    expect(await budget.load(1, 0, 'https://a.example/', file(100))).not.toBeNull();
  });

  it('starts over when the URL of the frame changes', async () => {
    const budget = new PageBudget(100);
    await budget.load(1, 0, 'https://a.example/', file(100));
    expect(await budget.load(1, 0, 'https://a.example/', file(1))).toBeNull();
    expect(await budget.load(1, 0, 'https://a.example/next', file(1))).not.toBeNull();
  });

  it('starts over when the frame is reset or its tab is forgotten', async () => {
    const budget = new PageBudget(100);
    await budget.load(1, 0, 'https://a.example/', file(100));
    budget.reset(1, 0);
    expect(await budget.load(1, 0, 'https://a.example/', file(100))).not.toBeNull();
    budget.forgetTab(1);
    expect(await budget.load(1, 0, 'https://a.example/', file(1))).not.toBeNull();
  });

  it('counts every frame and tab separately', async () => {
    const budget = new PageBudget(100);
    await budget.load(1, 0, 'https://a.example/', file(100));
    expect(await budget.load(1, 3, 'https://a.example/', file(1))).not.toBeNull();
    expect(await budget.load(2, 0, 'https://a.example/', file(1))).not.toBeNull();
    expect(await budget.load(1, 0, 'https://a.example/', file(1))).toBeNull();
  });

  it('hands out single units until a page view has used them up', () => {
    const budget = new PageBudget(2);
    expect(budget.take(1, 0, 'https://a.example/')).toBe(true);
    expect(budget.take(1, 0, 'https://a.example/')).toBe(true);
    expect(budget.take(1, 0, 'https://a.example/')).toBe(false);
    expect(budget.take(1, 0, 'https://a.example/next')).toBe(true);
  });
});
