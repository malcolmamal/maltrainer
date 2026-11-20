import { ChangeDetectionStrategy, Component, input } from '@angular/core';

/** The top of every page: one h1, an optional sentence, and actions on the right. */
@Component({
  selector: 'ui-page-header',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="text">
      <h1 class="mt-display" [class.mt-mono]="monoHeading()">{{ heading() }}</h1>
      @if (description()) {
        <p class="description">{{ description() }}</p>
      }
      <ng-content select="[headerMeta]" />
    </div>
    <div class="actions"><ng-content /></div>
  `,
  styles: `
    :host {
      display: flex;
      flex-wrap: wrap;
      align-items: flex-end;
      justify-content: space-between;
      gap: var(--mt-space-4) var(--mt-space-6);
    }
    h1 {
      font-size: 34px;
      line-height: 1.1;
    }
    h1.mt-mono {
      font-family: var(--mt-font-mono);
      font-weight: 500;
      letter-spacing: -0.03em;
    }
    .description {
      margin-top: var(--mt-space-2);
      color: var(--mt-text-2);
      max-width: 64ch;
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
    }
    @media (max-width: 640px) {
      h1 {
        font-size: 28px;
      }
    }
  `
})
export class PageHeaderComponent {
  readonly heading = input.required<string>();
  readonly description = input('');
  /** For headings that are data (a dataset name), set in the data font. */
  readonly monoHeading = input(false);
}
