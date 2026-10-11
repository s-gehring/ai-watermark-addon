const FRAME = '#5b21b6';
const HILL = new Path2D('M10 30l9-11 6 7 4-4 9 8z');
const SPARK = new Path2D('M34 7.5l1.6 5 5 1.6-5 1.6-1.6 5-1.6-5-5-1.6 5-1.6z');
const SIZES = [16, 32] as const;
const STEPS = 20;

const cache = new Map<number, Record<number, ImageData>>();

function draw(size: number, progress: number): ImageData {
  const canvas = new OffscreenCanvas(size, size);
  const context = canvas.getContext('2d')!;
  context.scale(size / 48, size / 48);

  context.globalAlpha = 0.55;
  context.fillStyle = FRAME;
  context.beginPath();
  context.roundRect(4, 4, 40, 30, 6);
  context.fill();
  context.fillStyle = '#c4b5fd';
  context.fill(HILL);
  context.fillStyle = '#ffffff';
  context.fill(SPARK);

  context.globalAlpha = 1;
  context.fillStyle = '#d4d4d8';
  context.beginPath();
  context.roundRect(2, 38, 44, 9, 4.5);
  context.fill();
  context.fillStyle = '#f59e0b';
  context.beginPath();
  context.roundRect(2, 38, Math.max(9, 44 * progress), 9, 4.5);
  context.fill();

  return context.getImageData(0, 0, size, size);
}

export function progressIcon(progress: number): { key: string; imageData: Record<number, ImageData> } {
  const step = Math.round(Math.min(Math.max(progress, 0), 1) * STEPS);
  let imageData = cache.get(step);
  if (!imageData) {
    imageData = Object.fromEntries(SIZES.map((size) => [size, draw(size, step / STEPS)]));
    cache.set(step, imageData);
  }
  return { key: `progress-${step}`, imageData };
}
