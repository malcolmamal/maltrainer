import { ChangeDetectionStrategy, Component, input } from '@angular/core';

export interface Fact {
  label: string;
  value: string | number;
  /** A quieter suffix after the value, e.g. "× 96 steps". */
  detail?: string;
  /** Set the value in the data font (paths, ids). */
  mono?: boolean;
}

/** Label/value pairs in a grid: plan numbers, run settings. Not a hero-metric row. */
@Component({
  selector: 'ui-fact-list',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <dl [style.--cols]="columns()">
      @for (fact of facts(); track fact.label) {
        <div class="fact">
          <dt>{{ fact.label }}</dt>
          <dd class="mt-num" [class.mt-mono]="fact.mono">
            {{ fact.value }}
            @if (fact.detail) {
              <small>{{ fact.detail }}</small>
            }
          </dd>
        </div>
      }
    </dl>
  `,
  styles: `
    dl {
      display: grid;
      grid-template-columns: repeat(var(--cols), minmax(0, 1fr));
      gap: 14px var(--mt-space-5);
    }
    dt {
      font-size: 12px;
      color: var(--mt-text-3);
    }
    dd {
      margin-top: 2px;
      font-size: 17px;
      font-weight: 600;
      overflow-wrap: anywhere;
    }
    dd.mt-mono {
      font-size: 13px;
      font-weight: 500;
    }
    small {
      font-size: 13px;
      font-weight: 400;
      color: var(--mt-text-2);
    }
    @media (max-width: 640px) {
      dl {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
  `
})
export class FactListComponent {
  readonly facts = input.required<Fact[]>();
  readonly columns = input(2);
}
