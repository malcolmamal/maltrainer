import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

export interface LossPoint {
  step: number;
  loss: number;
}

const W = 680;
const LEFT = 52;
const RIGHT = 672;
const TOP = 12;
const BOTTOM = 208;
const MAX_POINTS = 600;
const SMOOTHING = 0.15;

export interface LossGeometry {
  raw: string;
  smooth: string;
  grid: { y: number; label: string }[];
  ticks: { x: number; label: string }[];
  last: LossPoint;
  lastSmooth: number;
}

/** Keep at most MAX_POINTS evenly spread points, always including the last one. */
function thin(points: LossPoint[]): LossPoint[] {
  if (points.length <= MAX_POINTS) return points;
  const every = Math.ceil(points.length / MAX_POINTS);
  const kept = points.filter((_, i) => i % every === 0);
  if (kept[kept.length - 1] !== points[points.length - 1]) kept.push(points[points.length - 1]);
  return kept;
}

/** SVG polylines, grid and axis labels for a loss series over a run of `total` steps. */
export function lossGeometry(input: LossPoint[], total: number): LossGeometry | null {
  if (input.length < 2) return null;
  const points = thin(input);
  const smooth: number[] = [];
  points.forEach((p, i) =>
    smooth.push(i === 0 ? p.loss : smooth[i - 1] + SMOOTHING * (p.loss - smooth[i - 1]))
  );
  const losses = points.map((p) => p.loss);
  let lo = Math.min(...losses);
  let hi = Math.max(...losses);
  const pad = hi > lo ? (hi - lo) * 0.1 : Math.max(Math.abs(hi) * 0.1, 0.01);
  lo -= pad;
  hi += pad;
  const end = Math.max(total, points[points.length - 1].step, 1);
  const x = (step: number) => Math.round((LEFT + (step / end) * (RIGHT - LEFT)) * 10) / 10;
  const y = (loss: number) =>
    Math.round((BOTTOM - ((loss - lo) / (hi - lo)) * (BOTTOM - TOP)) * 10) / 10;
  const line = (values: number[]) => points.map((p, i) => `${x(p.step)},${y(values[i])}`).join(' ');
  return {
    raw: line(losses),
    smooth: line(smooth),
    grid: [0, 1, 2, 3].map((i) => {
      const value = lo + ((hi - lo) * (i + 0.5)) / 4;
      return { y: y(value), label: value.toPrecision(2) };
    }),
    ticks: [0, 0.25, 0.5, 0.75, 1].map((f) => ({
      x: x(end * f),
      label: Math.round(end * f).toLocaleString('en-US')
    })),
    last: points[points.length - 1],
    lastSmooth: smooth[smooth.length - 1]
  };
}

/** Loss per step (thin line) and its moving average (thick line). */
@Component({
  selector: 'ui-loss-chart',
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './loss-chart.component.html',
  styleUrl: './loss-chart.component.scss'
})
export class LossChartComponent {
  readonly points = input.required<LossPoint[]>();
  readonly total = input.required<number>();
  readonly geometry = computed(() => lossGeometry(this.points(), this.total()));
  readonly width = W;
  readonly left = LEFT;
  readonly right = RIGHT;
}
