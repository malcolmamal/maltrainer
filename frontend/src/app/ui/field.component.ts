import { ChangeDetectionStrategy, Component, input } from '@angular/core';

/**
 * A labelled form control: `<ui-field label="Optimizer"><select uiInput>…</select></ui-field>`.
 * The label wraps the projected control, so no `for`/`id` pair is needed.
 */
@Component({
  selector: 'ui-field',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <label class="field">
      <span class="label">{{ label() }}</span>
      <ng-content />
    </label>
    @if (hint()) {
      <p class="mt-hint">{{ hint() }}</p>
    }
  `,
  styles: `
    :host {
      display: flex;
      flex-direction: column;
      gap: var(--mt-space-2);
      min-width: 0;
    }
    .field {
      display: flex;
      flex-direction: column;
      gap: var(--mt-space-2);
    }
    .label {
      font-size: 13px;
      font-weight: 500;
      color: var(--mt-text-2);
    }
  `
})
export class FieldComponent {
  readonly label = input.required<string>();
  readonly hint = input('');
}
