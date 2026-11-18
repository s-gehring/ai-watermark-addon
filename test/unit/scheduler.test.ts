import { describe, expect, it } from 'vitest';
import { Job, JobBoard, Lane, type Requester } from '../../src/background/scheduler';

const requester = (tabId: number, id = 0, frameId = 0, inView = true): Requester => ({ tabId, frameId, id, inView });
const tick = () => new Promise((resolve) => setTimeout(resolve));

function jobFor(order: number, ...requesters: Requester[]): Job {
  const job = new Job(order);
  job.requesters.push(...requesters);
  return job;
}

function gate(): { open: () => void; task: () => Promise<void> } {
  let open = () => {};
  const opened = new Promise<void>((resolve) => (open = resolve));
  return { open, task: () => opened };
}

describe('Job', () => {
  it('ranks by its best requester: tab in front and in view first, other tabs last', () => {
    const front = new Set([1]);
    expect(jobFor(0, requester(1)).rank(front)).toBe(0);
    expect(jobFor(0, requester(1, 0, 0, false)).rank(front)).toBe(1);
    expect(jobFor(0, requester(2)).rank(front)).toBe(2);
    expect(jobFor(0, requester(2, 0, 0, false)).rank(front)).toBe(3);
    expect(jobFor(0, requester(2, 0, 0, false), requester(1)).rank(front)).toBe(0);
  });
});

describe('JobBoard', () => {
  it('starts one job per key and shares its result', async () => {
    const board = new JobBoard<string>();
    const started: Job[] = [];
    const start = async (job: Job) => {
      started.push(job);
      await tick();
      return 'verdict';
    };
    const results = [board.join('a', requester(1), start), board.join('a', requester(2), start)];
    expect(await Promise.all(results)).toEqual(['verdict', 'verdict']);
    expect(started).toHaveLength(1);
    expect(started[0].requesters.map((r) => r.tabId)).toEqual([1, 2]);
  });

  it('starts over once a job has finished', async () => {
    const board = new JobBoard<number>();
    let runs = 0;
    const start = async () => ++runs;
    expect(await board.join('a', requester(1), start)).toBe(1);
    expect(await board.join('a', requester(1), start)).toBe(2);
  });

  it('drops a job when its last requester leaves and keeps it while one remains', () => {
    const board = new JobBoard<void>();
    const jobs: Job[] = [];
    const start = (job: Job) => {
      jobs.push(job);
      return new Promise<void>(() => {});
    };
    void board.join('shared', requester(1), start);
    void board.join('shared', requester(2), start);
    void board.join('own', requester(1), start);
    board.leave((r) => r.tabId === 1);
    expect(jobs.map((job) => job.dropped)).toEqual([false, true]);
    expect(jobs[0].requesters.map((r) => r.tabId)).toEqual([2]);
  });

  it('starts a new job for a key whose job was dropped', () => {
    const board = new JobBoard<void>();
    const jobs: Job[] = [];
    const start = (job: Job) => {
      jobs.push(job);
      return new Promise<void>(() => {});
    };
    void board.join('a', requester(1), start);
    board.leave(() => true);
    void board.join('a', requester(1), start);
    expect(jobs.map((job) => job.dropped)).toEqual([true, false]);
  });

  it('marks the requests of one frame as out of view and back in view', () => {
    const board = new JobBoard<void>();
    const jobs: Job[] = [];
    const start = (job: Job) => {
      jobs.push(job);
      return new Promise<void>(() => {});
    };
    void board.join('a', requester(1, 7), start);
    void board.join('b', requester(1, 8), start);
    void board.join('c', requester(1, 7, 3), start);
    board.setOutOfView(1, 0, new Set([7]));
    expect(jobs.map((job) => job.requesters[0].inView)).toEqual([false, true, true]);
    board.setOutOfView(1, 0, new Set());
    expect(jobs.map((job) => job.requesters[0].inView)).toEqual([true, true, true]);
  });
});

describe('Lane', () => {
  const lane = (front: number[], concurrency?: number) => new Lane(async () => new Set(front), concurrency);

  async function order(queue: Lane, jobs: Record<string, Job>): Promise<string[]> {
    const ran: string[] = [];
    const first = gate();
    const blocker = queue.run(jobFor(-1, requester(99)), first.task);
    await tick();
    const all = Object.entries(jobs).map(([name, job]) => queue.run(job, async () => void ran.push(name)));
    first.open();
    await Promise.all([blocker, ...all]);
    return ran;
  }

  it('runs one task at a time in the order of the requests', async () => {
    const ran = await order(lane([]), { a: jobFor(0, requester(1)), b: jobFor(1, requester(1)), c: jobFor(2, requester(1)) });
    expect(ran).toEqual(['a', 'b', 'c']);
  });

  it('prefers the tab in front, then what is in view', async () => {
    const ran = await order(lane([2]), {
      otherTab: jobFor(0, requester(1)),
      scrolledPast: jobFor(1, requester(2, 0, 0, false)),
      inView: jobFor(2, requester(2)),
      alsoInView: jobFor(3, requester(2)),
    });
    expect(ran).toEqual(['inView', 'alsoInView', 'scrolledPast', 'otherTab']);
  });

  it('ranks again before every task, so a tab that comes to the front goes first', async () => {
    const front = new Set([1]);
    const queue = new Lane(async () => front);
    const ran: string[] = [];
    const first = gate();
    const a = queue.run(jobFor(0, requester(1)), first.task);
    await tick();
    const b = queue.run(jobFor(1, requester(1)), async () => void ran.push('first tab'));
    const c = queue.run(jobFor(2, requester(2)), async () => void ran.push('second tab'));
    front.clear();
    front.add(2);
    first.open();
    await Promise.all([a, b, c]);
    expect(ran).toEqual(['second tab', 'first tab']);
  });

  it('skips a waiting task whose job was dropped and answers it with null', async () => {
    const queue = lane([]);
    const first = gate();
    const blocker = queue.run(jobFor(0, requester(1)), first.task);
    await tick();
    const dropped = jobFor(1, requester(1));
    let ran = false;
    const result = queue.run(dropped, async () => (ran = true));
    dropped.dropped = true;
    first.open();
    await blocker;
    expect(await result).toBeNull();
    expect(ran).toBe(false);
  });

  it('does not start a task for a job that was dropped before it arrived', async () => {
    const dropped = jobFor(0);
    dropped.dropped = true;
    let ran = false;
    expect(await lane([]).run(dropped, async () => (ran = true))).toBeNull();
    expect(ran).toBe(false);
  });

  it('keeps going after a task fails', async () => {
    const queue = lane([]);
    const failed = queue.run(jobFor(0, requester(1)), async () => {
      throw new Error('boom');
    });
    await expect(failed).rejects.toThrow('boom');
    expect(await queue.run(jobFor(1, requester(1)), async () => 'next')).toBe('next');
  });

  it('runs as many tasks at once as it has room for', async () => {
    const queue = lane([], 2);
    const gates = [gate(), gate(), gate()];
    const started: number[] = [];
    const runs = gates.map((g, i) =>
      queue.run(jobFor(i, requester(1)), () => {
        started.push(i);
        return g.task();
      }),
    );
    await tick();
    expect(started).toEqual([0, 1]);
    gates[0].open();
    await runs[0];
    await tick();
    expect(started).toEqual([0, 1, 2]);
    gates[1].open();
    gates[2].open();
    await Promise.all(runs);
  });
});
