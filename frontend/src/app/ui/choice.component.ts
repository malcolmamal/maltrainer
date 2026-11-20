import { ChangeDetectionStrategy, Component, input } from '@angular/core';

/**
 * One option of a pick-one list drawn as a row or card: `<button uiChoice [selected]="…">`.
 * Put the options in an element with `role="radiogroup"` and an aria-label.
 */
@Component({
  selector: 'button[uiChoice]',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: {
    class: 'ui-choice',
    type: 'button',
    role: 'radio',
    '[attr.aria-checked]': 'selected()',
    '[class.ui-choice--on]': 'selected()',
    '[class.ui-choice--card]': "layout() === 'card'"
  },
  template: `<span class="dot" aria-hidden="true"></span><ng-content />`,
  styles: `
    :host {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px var(--mt-space-4);
      min-height: 56px;
      padding: 10px var(--mt-space-4);
      border: 1px solid var(--mt-line);
      border-radius: var(--mt-radius-lg);
      background: var(--mt-surface-2);
      text-align: left;
      cursor: pointer;
      transition:
        background-color 0.15s var(--mt-ease-out),
        border-color 0.15s var(--mt-ease-out);
    }
    :host(:hover) {
      border-color: var(--mt-line-strong);
    }
    :host(.ui-choice--on) {
      background: var(--mt-accent-soft);
      border-color: var(--mt-accent-border);
    }
    :host(.ui-choice--card) {
      flex-direction: column;
      align-items: flex-start;
      gap: var(--mt-space-1);
      min-height: 84px;
      padding: 14px var(--mt-space-4);
      position: relative;
    }
    :host(.ui-choice--card) .dot {
      position: absolute;
      top: 16px;
      right: 16px;
    }
    .dot {
      width: 16px;
      height: 16px;
      flex: none;
      border-radius: 50%;
      border: 1.5px solid var(--mt-tick);
      transition: border-width 0.2s var(--mt-ease-out);
    }
    :host(.ui-choice--on) .dot {
      border: 5px solid var(--mt-accent);
    }
    :host(:disabled) {
      opacity: 0.55;
      cursor: not-allowed;
    }
  `
})
export class ChoiceComponent {
  readonly selected = input(false);
  /** row: a list line (datasets); card: a grid tile (models). */
  readonly layout = input<'row' | 'card'>('row');
}
