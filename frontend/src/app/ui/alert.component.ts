import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

import { IconComponent, IconName } from './icon.component';

export type AlertTone = 'error' | 'warning' | 'info';

const ICONS: Record<AlertTone, IconName> = { error: 'error', warning: 'alert', info: 'info' };

/**
 * A message box. error and warning are announced (role="alert"); info is a status note.
 * Errors say what went wrong and what to do next.
 */
@Component({
  selector: 'ui-alert',
  imports: [IconComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: {
    '[attr.role]': "tone() === 'info' ? 'status' : 'alert'",
    '[class]': "'ui-alert ui-alert--' + tone()"
  },
  template: `<ui-icon [name]="icon()" [size]="18" />
    <div class="body"><ng-content /></div>`,
  styles: `
    :host {
      display: flex;
      gap: var(--mt-space-3);
      padding: 14px var(--mt-space-4);
      border-radius: var(--mt-radius-lg);
      border: 1px solid;
      font-size: 14px;
      line-height: 1.5;
    }
    ui-icon {
      margin-top: 2px;
    }
    .body {
      min-width: 0;
      overflow-wrap: anywhere;
    }
    :host(.ui-alert--error) {
      background: var(--mt-danger-soft);
      border-color: var(--mt-danger-border);
      color: var(--mt-danger-text);
    }
    :host(.ui-alert--warning) {
      background: var(--mt-warn-soft);
      border-color: var(--mt-warn-border);
      color: var(--mt-warn-text);
    }
    :host(.ui-alert--info) {
      background: var(--mt-good-soft);
      border-color: var(--mt-good-border);
      color: var(--mt-good-text);
    }
  `
})
export class AlertComponent {
  readonly tone = input<AlertTone>('error');
  readonly icon = computed(() => ICONS[this.tone()]);
}
