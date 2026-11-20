import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

/** More checkpoint ticks than this would blur into a solid bar, so none are drawn. */
export const MAX_TICKS = 60;

export interface TimelineTick {
  step: number;
  left: number;
  saved: boolean;
}

const pct = (step: number, total: number): number =>
  Math.round((Math.min(Math.max(step, 0), total) / total) * 10000) / 100;

/** Checkpoint steps from `every` up to `total`, or none when there would be too many. */
export function checkpointTicks(
  total: number,
  every: number | null,
  current: number | null
): TimelineTick[] {
  if (!(total > 0) || !every || every <= 0 || total / every > MAX_TICKS) return [];
  const ticks: TimelineTick[] = [];
  for (let step = every; step <= total; step += every) {
    ticks.push({ step, left: pct(step, total), saved: current !== null && step <= current });
  }
  return ticks;
}

/**
 * The signature element: a run's step axis with its checkpoints, the template's keep range,
 * where it overcooks and, while training, how far it got.
 */
@Component({
  selector: 'ui-cook-timeline',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { role: 'img', '[attr.aria-label]': 'description()' },
  template: `
    <div class="track-area" [class.live]="current() !== null">
      @if (keep(); as k) {
        <div class="keep" [style.left.%]="k.left" [style.width.%]="k.width"></div>
      }
      @if (overcookLeft() !== null) {
        <div class="over" [style.left.%]="overcookLeft()"></div>
      }
      <div class="track"></div>
      @if (current() !== null) {
        <div class="fill" [style.width.%]="currentLeft()"></div>
        <div class="now" [style.left.%]="currentLeft()"></div>
      }
      @for (tick of ticks(); track tick.step) {
        <div class="tick" [class.saved]="tick.saved" [style.left.%]="tick.left"></div>
      }
    </div>
    <div class="axis mt-mono mt-num">
      <span>0</span><span>{{ format(half()) }}</span
      ><span>{{ format(total()) }}</span>
    </div>
  `,
  styleUrl: './cook-timeline.component.scss'
})
export class CookTimelineComponent {
  readonly total = input.required<number>();
  readonly saveEvery = input<number | null>(null);
  /** The step reached so far; null for a plan that has not started. */
  readonly current = input<number | null>(null);
  readonly keepFrom = input<number | null>(null);
  readonly keepTo = input<number | null>(null);
  readonly overcookAfter = input<number | null>(null);

  readonly safeTotal = computed(() => Math.max(1, this.total()));
  readonly half = computed(() => Math.round(this.total() / 2));
  readonly ticks = computed(() => checkpointTicks(this.total(), this.saveEvery(), this.current()));
  readonly currentLeft = computed(() => pct(this.current() ?? 0, this.safeTotal()));
  readonly keep = computed(() => {
    const from = this.keepFrom();
    const to = this.keepTo();
    if (from === null || to === null || to <= from || from >= this.safeTotal()) return null;
    const left = pct(from, this.safeTotal());
    return { from, to, left, width: pct(to, this.safeTotal()) - left };
  });
  /** The step past which the run overcooks, when it falls inside the run. */
  readonly overcook = computed(() => {
    const after = this.overcookAfter();
    return after === null || after >= this.safeTotal() ? null : after;
  });
  readonly overcookLeft = computed(() => {
    const after = this.overcook();
    return after === null ? null : pct(after, this.safeTotal());
  });

  readonly description = computed(() => {
    const f = (value: number) => this.format(value);
    const current = this.current();
    const steps = `${f(this.total())} steps`;
    const parts = [current === null ? steps : `Step ${f(current)} of ${steps}`];
    const ticks = this.ticks();
    if (ticks.length) parts.push(`a checkpoint every ${f(ticks[0].step)}`);
    const keep = this.keep();
    if (keep) parts.push(`best checkpoints ${f(keep.from)} to ${f(keep.to)}`);
    const over = this.overcook();
    if (over !== null) parts.push(`overcooks past ${f(over)}`);
    return parts.join(', ');
  });

  format(value: number): string {
    return value.toLocaleString('en-US');
  }
}
