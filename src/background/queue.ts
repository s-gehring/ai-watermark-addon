export class TaskQueue {
  private active = 0;
  private readonly waiting: (() => void)[] = [];

  constructor(private readonly concurrency = 1) {}

  async run<T>(task: () => Promise<T>): Promise<T> {
    if (this.active < this.concurrency) this.active++;
    else await new Promise<void>((resolve) => this.waiting.push(resolve));
    try {
      return await task();
    } finally {
      const next = this.waiting.shift();
      if (next) next();
      else this.active--;
    }
  }
}

export async function firstResult<T>(candidates: string[], load: (candidate: string) => Promise<T | null>): Promise<T | null> {
  for (const candidate of candidates) {
    const result = await load(candidate);
    if (result) return result;
  }
  return null;
}
