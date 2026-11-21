import { ChangeDetectionStrategy, Component, input } from '@angular/core';

export type ButtonVariant = 'primary' | 'ghost' | 'danger';

/**
 * The only button look in the app: `<button uiButton>` or `<a uiButton routerLink>`.
 * primary = the one main action of a screen, ghost = everything else, danger = stops or deletes.
 */
@Component({
  selector: 'button[uiButton], a[uiButton]',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: {
    class: 'ui-button',
    '[class.ui-button--primary]': "variant() === 'primary'",
    '[class.ui-button--ghost]': "variant() === 'ghost'",
    '[class.ui-button--danger]': "variant() === 'danger'",
    '[class.ui-button--sm]': "size() === 'sm'"
  },
  template: '<ng-content />',
  styles: `
    :host {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: var(--mt-space-2);
      min-height: var(--mt-control-h);
      padding: 0 18px;
      border-radius: var(--mt-radius-md);
      border: 1px solid transparent;
      background: transparent;
      font-weight: 600;
      text-decoration: none;
      white-space: nowrap;
      cursor: pointer;
      transition:
        background-color 0.15s var(--mt-ease-out),
        border-color 0.15s var(--mt-ease-out);
    }
    :host(.ui-button--primary) {
      background: var(--mt-accent);
      color: var(--mt-on-accent);
    }
    :host(.ui-button--primary:hover) {
      background: var(--mt-accent-hover);
    }
    :host(.ui-button--ghost) {
      border-color: var(--mt-line-strong);
      color: var(--mt-text);
    }
    :host(.ui-button--ghost:hover) {
      background: var(--mt-hover);
    }
    :host(.ui-button--danger) {
      border-color: var(--mt-danger-border);
      color: var(--mt-danger-text);
    }
    :host(.ui-button--danger:hover) {
      background: var(--mt-danger-soft);
    }
    :host(.ui-button--sm) {
      min-height: var(--mt-control-h-sm);
      padding: 0 12px;
      font-size: 13px;
      font-weight: 500;
    }
    :host(:disabled),
    :host([aria-disabled='true']) {
      cursor: not-allowed;
      opacity: 0.6;
    }
    :host(.ui-button--primary:disabled) {
      opacity: 1;
      background: var(--mt-accent-disabled);
      color: var(--mt-on-accent-disabled);
    }
  `
})
export class ButtonComponent {
  readonly variant = input<ButtonVariant>('ghost');
  readonly size = input<'md' | 'sm'>('md');
}
