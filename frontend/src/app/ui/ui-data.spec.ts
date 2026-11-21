import { Component, signal } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import {
  checkpointTicks,
  Fact,
  LogLine,
  lossGeometry,
  LossPoint,
  MAX_TICKS,
  percentOf,
  statusTone,
  UI
} from '.';

describe('pure helpers', () => {
  it('percentOf clamps and guards a missing maximum', () => {
    expect(percentOf(5, 40)).toBe(12.5);
    expect(percentOf(50, 40)).toBe(100);
    expect(percentOf(-3, 40)).toBe(0);
    expect(percentOf(5, 0)).toBe(0);
    expect(percentOf(1, 3)).toBe(33.3);
  });

  it('statusTone maps backend statuses and falls back to neutral', () => {
    expect(['running', 'queued', 'exited', 'failed', 'stopped'].map(statusTone)).toEqual([
      'live',
      'neutral',
      'good',
      'bad',
      'neutral'
    ]);
  });

  it('checkpointTicks spaces ticks, marks saved ones, and skips impossible or dense input', () => {
    expect(checkpointTicks(1000, 250, 600)).toEqual([
      { step: 250, left: 25, saved: true },
      { step: 500, left: 50, saved: true },
      { step: 750, left: 75, saved: false },
      { step: 1000, left: 100, saved: false }
    ]);
    expect(checkpointTicks(1000, 300, null).map((t) => [t.step, t.saved])).toEqual([
      [300, false],
      [600, false],
      [900, false]
    ]);
    expect(checkpointTicks(1000, null, null)).toEqual([]);
    expect(checkpointTicks(1000, 0, null)).toEqual([]);
    expect(checkpointTicks(0, 100, null)).toEqual([]);
    expect(checkpointTicks(MAX_TICKS * 10, 10, null)).toHaveLength(MAX_TICKS);
    expect(checkpointTicks(MAX_TICKS * 10 + 10, 10, null)).toEqual([]);
  });

  it('lossGeometry needs two points and scales them into the plot', () => {
    expect(lossGeometry([{ step: 1, loss: 0.1 }], 10)).toBeNull();
    const g = lossGeometry(
      [
        { step: 0, loss: 0.2 },
        { step: 50, loss: 0.1 }
      ],
      100
    );
    // 10% padding above and below the range keeps the line off the plot edges
    expect(g?.raw).toBe('52,28.3 362,191.7');
    // the moving average moves 15% of the way: 0.2 → 0.185
    expect(g?.lastSmooth).toBeCloseTo(0.185);
    expect(g?.last).toEqual({ step: 50, loss: 0.1 });
    expect(g?.ticks.map((t) => t.label)).toEqual(['0', '25', '50', '75', '100']);
    expect(g?.grid.map((l) => l.y)).toEqual([183.5, 134.5, 85.5, 36.5]);
  });

  it('lossGeometry pads a flat series and stretches the axis past the planned total', () => {
    const g = lossGeometry(
      [
        { step: 10, loss: 0.5 },
        { step: 30, loss: 0.5 }
      ],
      20
    );
    expect(g?.raw).toBe('258.7,110 672,110');
    expect(g?.ticks[4].label).toBe('30');
    const zero = lossGeometry(
      [
        { step: 1, loss: 0 },
        { step: 2, loss: 0 }
      ],
      0
    );
    expect(zero?.grid.map((l) => l.label)).toEqual(['-0.0075', '-0.0025', '0.0025', '0.0075']);
  });

  it('lossGeometry thins long series but keeps the latest point', () => {
    const many: LossPoint[] = Array.from({ length: 1001 }, (_, i) => ({
      step: i,
      loss: 1 / (i + 1)
    }));
    const g = lossGeometry(many, 1000);
    // every second point: 0, 2, … 1000, which already ends on the last one
    expect(g?.raw.split(' ').length).toBe(501);
    expect(g?.last.step).toBe(1000);
    // 0, 2, … 998, then the last point (999) is added back
    const even = lossGeometry(many.slice(0, 1000), 1000);
    expect(even?.raw.split(' ').length).toBe(501);
    expect(even?.last.step).toBe(999);
  });
});

@Component({
  imports: [...UI],
  template: `
    <ui-status-pill [status]="status()" />
    <ui-progress-bar [value]="3" [max]="4" [tone]="'good'" />
    <ui-cook-timeline
      id="plan"
      [total]="2000"
      [saveEvery]="500"
      [keepFrom]="keepFrom()"
      [keepTo]="1750"
      [overcookAfter]="overcook()"
    />
    <ui-cook-timeline id="live" [total]="1000" [current]="current()" />
    <ui-fact-list [facts]="facts" [columns]="3" />
    <ui-loss-chart [points]="points()" [total]="100" />
    <ui-log-view [lines]="lines()" [liveLine]="live()" liveStream="stderr" />
  `
})
class Host {
  readonly status = signal('running');
  readonly keepFrom = signal<number | null>(1250);
  readonly overcook = signal<number | null>(1500);
  readonly current = signal<number | null>(400);
  readonly points = signal<LossPoint[]>([]);
  readonly lines = signal<LogLine[]>([]);
  readonly live = signal<string | null>(null);
  readonly facts: Fact[] = [
    { label: 'Epochs', value: 20, detail: '× 96 steps' },
    { label: 'Output', value: '/out/x', mono: true }
  ];
}

describe('design system data components', () => {
  let fixture: ComponentFixture<Host>;
  let host: Host;
  const q = (sel: string): HTMLElement => fixture.nativeElement.querySelector(sel);
  const all = (sel: string): HTMLElement[] =>
    Array.from(fixture.nativeElement.querySelectorAll(sel));

  beforeEach(() => {
    fixture = TestBed.createComponent(Host);
    host = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('labels statuses, pulses only live ones, and shows unknown ones as they are', () => {
    expect(q('ui-status-pill').textContent?.trim()).toBe('Training');
    expect(q('ui-status-pill .pulse')).not.toBeNull();
    host.status.set('exited');
    fixture.detectChanges();
    expect(q('ui-status-pill').textContent?.trim()).toBe('Finished');
    expect(q('ui-status-pill').className).toContain('ui-status--good');
    expect(q('ui-status-pill .pulse')).toBeNull();
    host.status.set('stopped');
    fixture.detectChanges();
    expect(q('ui-status-pill').textContent?.trim()).toBe('stopped');
  });

  it('fills the progress bar and exposes its values', () => {
    const bar = q('ui-progress-bar');
    expect(bar.getAttribute('role')).toBe('progressbar');
    expect(bar.getAttribute('aria-valuenow')).toBe('3');
    expect(bar.getAttribute('aria-valuemax')).toBe('4');
    expect(bar.className).toContain('ui-progress--good');
    expect((bar.querySelector('.fill') as HTMLElement).style.width).toBe('75%');
  });

  it('draws a plan timeline with ticks, the keep range and the overcook zone', () => {
    const plan = q('#plan');
    expect(all('#plan .tick').map((t) => t.style.left)).toEqual(['25%', '50%', '75%', '100%']);
    expect((plan.querySelector('.keep') as HTMLElement).style.left).toBe('62.5%');
    expect((plan.querySelector('.keep') as HTMLElement).style.width).toBe('25%');
    expect((plan.querySelector('.over') as HTMLElement).style.left).toBe('75%');
    expect(plan.querySelector('.fill')).toBeNull();
    expect(plan.querySelector('.axis')?.textContent).toBe('01,0002,000');
    expect(plan.getAttribute('aria-label')).toBe(
      '2,000 steps, a checkpoint every 500, best checkpoints 1,250 to 1,750, overcooks past 1,500'
    );
  });

  it('hides keep and overcook marks that are missing or beyond the run', () => {
    host.keepFrom.set(null);
    host.overcook.set(2000);
    fixture.detectChanges();
    expect(q('#plan .keep')).toBeNull();
    expect(q('#plan .over')).toBeNull();
    host.keepFrom.set(2500);
    fixture.detectChanges();
    expect(q('#plan .keep')).toBeNull();
    host.keepFrom.set(1800);
    fixture.detectChanges();
    expect(q('#plan .keep')).toBeNull();
    expect(q('#plan').getAttribute('aria-label')).toBe('2,000 steps, a checkpoint every 500');
  });

  it('shows how far a live run got', () => {
    const live = q('#live');
    expect(live.querySelector('.track-area')?.classList).toContain('live');
    expect((live.querySelector('.fill') as HTMLElement).style.width).toBe('40%');
    expect((live.querySelector('.now') as HTMLElement).style.left).toBe('40%');
    expect(live.getAttribute('aria-label')).toBe('Step 400 of 1,000 steps');
    host.current.set(null);
    fixture.detectChanges();
    expect(live.querySelector('.fill')).toBeNull();
  });

  it('lists facts with details and data-font values', () => {
    const dds = all('ui-fact-list dd');
    expect(all('ui-fact-list dt').map((d) => d.textContent)).toEqual(['Epochs', 'Output']);
    expect(dds[0].textContent?.replace(/\s+/g, ' ').trim()).toBe('20 × 96 steps');
    expect(dds[1].classList).toContain('mt-mono');
    expect(dds[1].querySelector('small')).toBeNull();
    expect((q('ui-fact-list dl') as HTMLElement).style.getPropertyValue('--cols')).toBe('3');
  });

  it('waits for two loss points before drawing the chart', () => {
    expect(q('ui-loss-chart .mt-hint')?.textContent).toContain('once the trainer reports');
    expect(q('ui-loss-chart svg')).toBeNull();
    host.points.set([
      { step: 10, loss: 0.2 },
      { step: 20, loss: 0.1 }
    ]);
    fixture.detectChanges();
    expect(all('ui-loss-chart polyline')).toHaveLength(2);
    expect(q('ui-loss-chart .now')?.textContent?.replace(/\s+/g, ' ').trim()).toBe(
      '0.1000 now · 0.1850 smoothed'
    );
    expect(q('ui-loss-chart svg').getAttribute('aria-label')).toBe(
      'Loss at step 20 is 0.1000, smoothed 0.1850'
    );
  });

  it('filters the log to stderr, counts lines and explains empty states', () => {
    expect(q('ui-log-view .empty').textContent?.trim()).toBe('No output yet.');
    expect(q('ui-log-view .live')).toBeNull();
    host.lines.set([
      { stream: 'stdout', line: 'hello' },
      { stream: 'stdout', line: 'again' }
    ]);
    host.live.set('20%|##');
    fixture.detectChanges();
    expect(all('ui-log-view .logLine')).toHaveLength(2);
    expect(q('ui-log-view .live').classList).toContain('err');
    expect(q('ui-log-view .live .text').textContent).toBe('20%|##');

    const [allBtn, errBtn] = all('ui-log-view ui-segmented button');
    expect(allBtn.querySelector('.detail')?.textContent).toBe('2');
    expect(errBtn.querySelector('.detail')?.textContent).toBe('0');
    errBtn.click();
    fixture.detectChanges();
    expect(q('ui-log-view .empty').textContent?.trim()).toBe('No stderr lines yet.');

    host.lines.update((l) => [...l, { stream: 'stderr', line: 'oops' }]);
    fixture.detectChanges();
    expect(all('ui-log-view .logLine').map((l) => l.textContent)).toEqual(['stderroops']);
    expect(all('ui-log-view .logLine')[0].classList).toContain('err');
  });

  it('follows new lines only while the reader is at the bottom', () => {
    const box = q('ui-log-view .log');
    Object.defineProperty(box, 'scrollHeight', { configurable: true, value: 1000 });
    Object.defineProperty(box, 'clientHeight', { configurable: true, value: 200 });
    host.lines.set([{ stream: 'stdout', line: 'a' }]);
    fixture.detectChanges();
    expect(box.scrollTop).toBe(1000);

    box.scrollTop = 100;
    box.dispatchEvent(new Event('scroll'));
    fixture.detectChanges();
    expect(q('ui-log-view .follow').textContent).toContain('Paused');
    host.lines.update((l) => [...l, { stream: 'stdout', line: 'b' }]);
    fixture.detectChanges();
    expect(box.scrollTop).toBe(100);

    box.scrollTop = 790;
    box.dispatchEvent(new Event('scroll'));
    fixture.detectChanges();
    expect(q('ui-log-view .follow').textContent).toContain('Following new lines');
  });
});
