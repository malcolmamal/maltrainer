import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

export type StatusTone = 'live' | 'neutral' | 'good' | 'bad';

/** How each backend job status reads. Unknown statuses show as they are, neutral. */
const STATUSES: Record<string, { label: string; tone: StatusTone }> = {
  running: { label: 'Training', tone: 'live' },
  queued: { label: 'Queued', tone: 'neutral' },
  exited: { label: 'Finished', tone: 'good' },
  failed: { label: 'Failed', tone: 'bad' }
};

export function statusTone(status: string): StatusTone {
  return STATUSES[status]?.tone ?? 'neutral';
}

/** A job status: `<ui-status-pill status="running" />`. Running jobs get the ember pulse. */
@Component({
  selector: 'ui-status-pill',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { '[class]': "'ui-status ui-status--' + tone()" },
  template: `@if (tone() === 'live') {
      <span class="pulse" aria-hidden="true"></span>
    }
    {{ label() }}`,
  styles: `
    :host {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 3px 10px;
      border-radius: var(--mt-radius-pill);
      font-size: 13px;
      font-weight: 600;
      white-space: nowrap;
    }
    :host(.ui-status--live) {
      background: var(--mt-accent-soft);
      color: var(--mt-accent-text);
    }
    :host(.ui-status--neutral) {
      border: 1px solid var(--mt-line-strong);
      color: var(--mt-text-2);
      font-weight: 500;
    }
    :host(.ui-status--good) {
      background: var(--mt-good-soft);
      color: var(--mt-good-text);
    }
    :host(.ui-status--bad) {
      background: var(--mt-danger-soft);
      color: var(--mt-danger-text);
    }
    .pulse {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--mt-accent);
      animation: pulse 1.8s var(--mt-ease-out) infinite;
    }
    @keyframes pulse {
      60% {
        transform: scale(1.5);
        opacity: 0.45;
      }
    }
  `
})
export class StatusPillComponent {
  readonly status = input.required<string>();
  readonly tone = computed(() => statusTone(this.status()));
  readonly label = computed(() => STATUSES[this.status()]?.label ?? this.status());
}
