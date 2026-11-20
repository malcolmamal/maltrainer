import {
  afterRenderEffect,
  ChangeDetectionStrategy,
  Component,
  computed,
  ElementRef,
  input,
  signal,
  viewChild
} from '@angular/core';

import { SegmentedComponent, SegmentOption } from './segmented.component';

export interface LogLine {
  stream: 'stdout' | 'stderr';
  line: string;
}

type LogFilter = 'all' | 'stderr';

/** Lines within this many pixels of the bottom count as "at the bottom": new lines follow. */
const FOLLOW_SLACK = 24;

/**
 * A process's output: a filter, the replaced-in-place progress line, and the log that
 * follows new lines while the reader is at the bottom.
 */
@Component({
  selector: 'ui-log-view',
  imports: [SegmentedComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './log-view.component.html',
  styleUrl: './log-view.component.scss'
})
export class LogViewComponent {
  readonly lines = input.required<LogLine[]>();
  /** The latest tqdm-style line, shown above the log instead of piling up in it. */
  readonly liveLine = input<string | null>(null);
  readonly liveStream = input<'stdout' | 'stderr' | null>(null);

  readonly filter = signal<LogFilter>('all');
  readonly following = signal(true);
  readonly visible = computed(() =>
    this.filter() === 'all' ? this.lines() : this.lines().filter((l) => l.stream === 'stderr')
  );
  readonly filterOptions = computed<SegmentOption<LogFilter>[]>(() => [
    { value: 'all', label: 'All', detail: this.lines().length },
    {
      value: 'stderr',
      label: 'stderr',
      detail: this.lines().filter((l) => l.stream === 'stderr').length
    }
  ]);

  private readonly box = viewChild.required<ElementRef<HTMLElement>>('box');

  constructor() {
    afterRenderEffect(() => {
      this.visible();
      if (this.following()) {
        const el = this.box().nativeElement;
        el.scrollTop = el.scrollHeight;
      }
    });
  }

  onScroll(): void {
    const el = this.box().nativeElement;
    this.following.set(el.scrollHeight - el.scrollTop - el.clientHeight <= FOLLOW_SLACK);
  }
}
