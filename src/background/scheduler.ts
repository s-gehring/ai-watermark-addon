export interface Asker {
  tabId: number;
  frameId: number;
  id: number;
  page: string;
}

export interface Requester extends Asker {
  inView: boolean;
}

export type ActiveTabs = () => Promise<ReadonlySet<number>>;

export class Job {
  readonly requesters: Requester[] = [];
  private readonly dropping = new AbortController();

  constructor(readonly order: number) {}

  get dropped(): boolean {
    return this.dropping.signal.aborted;
  }

  get signal(): AbortSignal {
    return this.dropping.signal;
  }

  drop(): void {
    this.dropping.abort();
  }

  rank(activeTabs: ReadonlySet<number>): number {
    let best = Infinity;
    for (const { tabId, inView } of this.requesters) {
      best = Math.min(best, (activeTabs.has(tabId) ? 0 : 2) + (inView ? 0 : 1));
    }
    return best;
  }
}

export class JobBoard<T> {
  private readonly jobs = new Map<string, { job: Job; result: Promise<T> }>();
  private readonly outOfView = new Map<number, Map<number, ReadonlySet<number>>>();
  private created = 0;

  join(key: string, asker: Asker, start: (job: Job) => Promise<T>): Promise<T> {
    const requester = { ...asker, inView: !this.outOfView.get(asker.tabId)?.get(asker.frameId)?.has(asker.id) };
    const existing = this.jobs.get(key);
    if (existing) {
      existing.job.requesters.push(requester);
      return existing.result;
    }
    const job = new Job(this.created++);
    job.requesters.push(requester);
    const entry = { job, result: start(job) };
    this.jobs.set(key, entry);
    const forget = () => {
      if (this.jobs.get(key) === entry) this.jobs.delete(key);
    };
    entry.result.then(forget, forget);
    return entry.result;
  }

  leaveFrame(tabId: number, frameId: number): void {
    this.outOfView.get(tabId)?.delete(frameId);
    this.leave((requester) => requester.tabId === tabId && requester.frameId === frameId);
  }

  leaveTab(tabId: number): void {
    this.outOfView.delete(tabId);
    this.leave((requester) => requester.tabId === tabId);
  }

  private leave(gone: (requester: Requester) => boolean): void {
    for (const [key, { job }] of this.jobs) {
      const kept = job.requesters.filter((requester) => !gone(requester));
      if (kept.length === job.requesters.length) continue;
      job.requesters.splice(0, job.requesters.length, ...kept);
      if (kept.length) continue;
      job.drop();
      this.jobs.delete(key);
    }
  }

  setOutOfView(tabId: number, frameId: number, ids: ReadonlySet<number>): void {
    let frames = this.outOfView.get(tabId);
    if (!frames) this.outOfView.set(tabId, (frames = new Map()));
    frames.set(frameId, ids);
    for (const { job } of this.jobs.values()) {
      for (const requester of job.requesters) {
        if (requester.tabId === tabId && requester.frameId === frameId) requester.inView = !ids.has(requester.id);
      }
    }
  }
}

interface Waiting {
  job: Job;
  grant: (granted: boolean) => void;
}

export class Lane {
  private readonly waiting: Waiting[] = [];
  private running = 0;
  private picking = false;

  constructor(
    private readonly activeTabs: ActiveTabs,
    private readonly concurrency = 1,
  ) {}

  async run<T>(job: Job, task: () => Promise<T>): Promise<T | null> {
    const granted = await new Promise<boolean>((grant) => {
      this.waiting.push({ job, grant });
      void this.pick();
    });
    if (!granted) return null;
    try {
      return await task();
    } finally {
      this.running--;
      void this.pick();
    }
  }

  private async pick(): Promise<void> {
    if (this.picking) return;
    this.picking = true;
    try {
      while (this.releaseDropped() && this.running < this.concurrency) {
        const activeTabs = await this.activeTabs();
        if (!this.releaseDropped()) break;
        let next = 0;
        for (let i = 1; i < this.waiting.length; i++) {
          const [candidate, best] = [this.waiting[i].job, this.waiting[next].job];
          const difference = candidate.rank(activeTabs) - best.rank(activeTabs);
          if (difference < 0 || (difference === 0 && candidate.order < best.order)) next = i;
        }
        this.running++;
        this.waiting.splice(next, 1)[0].grant(true);
      }
    } finally {
      this.picking = false;
    }
  }

  private releaseDropped(): boolean {
    for (let i = this.waiting.length - 1; i >= 0; i--) {
      if (this.waiting[i].job.dropped) this.waiting.splice(i, 1)[0].grant(false);
    }
    return this.waiting.length > 0;
  }
}
