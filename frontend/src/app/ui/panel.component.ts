import { ChangeDetectionStrategy, Component, input } from '@angular/core';

let nextId = 0;

/**
 * A raised surface with a heading: `<ui-panel heading="Queue">`.
 * Put header buttons in an element marked `panelActions`. Never nest panels.
 */
@Component({
  selector: 'ui-panel',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { role: 'region', '[attr.aria-labelledby]': 'headingId' },
  template: `
    <div class="head">
      <h2 class="mt-display" [id]="headingId">{{ heading() }}</h2>
      <div class="actions"><ng-content select="[panelActions]" /></div>
    </div>
    <ng-content />
  `,
  styles: `
    :host {
      display: flex;
      flex-direction: column;
      gap: var(--mt-space-4);
      min-width: 0;
      padding: 22px;
      border-radius: var(--mt-radius-xl);
      background: var(--mt-surface);
      border: 1px solid var(--mt-line);
    }
    .head {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: var(--mt-space-3);
    }
    h2 {
      font-size: 19px;
      line-height: 1.3;
    }
    .actions {
      margin-left: auto;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: var(--mt-space-2);
    }
    .actions:empty {
      display: none;
    }
  `
})
export class PanelComponent {
  readonly heading = input.required<string>();
  readonly headingId = `ui-panel-${nextId++}`;
}
