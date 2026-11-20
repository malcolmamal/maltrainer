import { ChangeDetectionStrategy, Component, input, model } from '@angular/core';

/** An on/off setting with a label: `<ui-switch label="Dry run" [(checked)]="dryRun" />`. */
@Component({
  selector: 'ui-switch',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <label class="row">
      <input
        type="checkbox"
        role="switch"
        [checked]="checked()"
        (change)="checked.set($any($event.target).checked)"
      />
      <span class="label">{{ label() }}</span>
    </label>
    <ng-content />
  `,
  styles: `
    :host {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: var(--mt-space-3) var(--mt-space-4);
    }
    .row {
      display: inline-flex;
      align-items: center;
      gap: var(--mt-space-3);
      min-height: var(--mt-control-h);
      cursor: pointer;
      font-weight: 500;
    }
    input {
      appearance: none;
      position: relative;
      width: 40px;
      height: 24px;
      margin: 0;
      border-radius: var(--mt-radius-pill);
      background: var(--mt-track);
      cursor: pointer;
      transition: background-color 0.2s var(--mt-ease-out);
    }
    input::after {
      content: '';
      position: absolute;
      top: 3px;
      left: 3px;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: var(--mt-text-2);
      transition:
        transform 0.25s var(--mt-ease-out),
        background-color 0.2s;
    }
    input:checked {
      background: var(--mt-accent);
    }
    input:checked::after {
      transform: translateX(16px);
      background: var(--mt-on-accent);
    }
  `
})
export class SwitchComponent {
  readonly label = input.required<string>();
  readonly checked = model(false);
}
