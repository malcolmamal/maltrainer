import { ChangeDetectionStrategy, Component, input, model } from '@angular/core';

export interface SegmentOption<T extends string = string> {
  value: T;
  label: string;
  /** Shown after the label in a quieter tone, e.g. a count. */
  detail?: string | number;
}

/** A small pick-one switch (2–4 options): `<ui-segmented [options]="…" [(value)]="mode" />`. */
@Component({
  selector: 'ui-segmented',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { role: 'radiogroup', '[attr.aria-label]': 'ariaLabel()' },
  template: `
    @for (option of options(); track option.value) {
      <button
        type="button"
        role="radio"
        class="segment"
        [class.on]="option.value === value()"
        [attr.aria-checked]="option.value === value()"
        (click)="value.set(option.value)"
      >
        {{ option.label }}
        @if (option.detail !== undefined) {
          <span class="detail mt-num">{{ option.detail }}</span>
        }
      </button>
    }
  `,
  styles: `
    :host {
      display: inline-flex;
      flex-wrap: wrap;
      align-self: flex-start;
      padding: 3px;
      border-radius: var(--mt-radius-md);
      background: var(--mt-sunken);
      border: 1px solid var(--mt-line-strong);
    }
    .segment {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 36px;
      padding: 0 14px;
      border: 0;
      border-radius: 7px;
      background: transparent;
      color: var(--mt-text-2);
      font-weight: 500;
      cursor: pointer;
    }
    .segment:hover {
      color: var(--mt-text);
    }
    .segment.on {
      background: var(--mt-track);
      color: var(--mt-text);
    }
    .detail {
      font-size: 12px;
      color: var(--mt-text-3);
    }
  `
})
export class SegmentedComponent<T extends string = string> {
  readonly options = input.required<SegmentOption<T>[]>();
  readonly value = model.required<T>();
  readonly ariaLabel = input.required<string>();
}
