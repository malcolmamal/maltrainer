import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

import { StatusTone } from './status-pill.component';

/** Share of `value` in `max`, 0–100, rounded to one decimal; 0 when `max` is not positive. */
export function percentOf(value: number, max: number): number {
  if (!(max > 0)) return 0;
  return Math.round(Math.min(100, Math.max(0, (value / max) * 100)) * 10) / 10;
}

/** A thin bar for a list row. For the main progress of a run use `ui-cook-timeline`. */
@Component({
  selector: 'ui-progress-bar',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: {
    role: 'progressbar',
    'aria-valuemin': '0',
    '[attr.aria-valuemax]': 'max()',
    '[attr.aria-valuenow]': 'value()',
    '[class]': "'ui-progress ui-progress--' + tone()"
  },
  template: `<div class="fill" [style.width.%]="percent()"></div>`,
  styles: `
    :host {
      display: block;
      width: 110px;
      height: 4px;
      border-radius: 2px;
      background: var(--mt-track);
      overflow: hidden;
    }
    .fill {
      height: 100%;
      border-radius: 2px;
      background: var(--mt-tick);
    }
    :host(.ui-progress--live) .fill {
      background: var(--mt-accent);
    }
    :host(.ui-progress--good) .fill {
      background: var(--mt-good);
    }
    :host(.ui-progress--bad) .fill {
      background: var(--mt-danger);
    }
  `
})
export class ProgressBarComponent {
  readonly value = input.required<number>();
  readonly max = input.required<number>();
  readonly tone = input<StatusTone>('live');
  readonly percent = computed(() => percentOf(this.value(), this.max()));
}
