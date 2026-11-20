import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';

import { JobStatusResponse, ModelInfo } from '../../services/trainer-api.service';
import { formatStarted, RunsPage } from './runs.page';

function job(overrides: Partial<JobStatusResponse>): JobStatusResponse {
  return {
    job_id: 'j1',
    dataset: 'cats',
    output_path: '/out/cats',
    config_path: '/cfg',
    status: 'queued',
    started_at: 0,
    ended_at: null,
    exit_code: null,
    progress: { step: 0, total: 100 },
    model: 'qwen_image_2_1',
    optimizer: 'adamw8bit',
    duration: '100 steps',
    ...overrides
  };
}

const running = job({ job_id: 'r1', status: 'running', progress: { step: 40, total: 100 } });
// The backend sets started_at when a job is queued, not when it starts.
const queued = job({ job_id: 'q1', dataset: 'dogs', model: 'mystery', started_at: 1_700_000_000 });
const done = job({ job_id: 'd1', status: 'exited', exit_code: 0, started_at: 1_700_000_000 });
const failed = job({ job_id: 'f1', status: 'failed', exit_code: 3 });
const crashed = job({ job_id: 'f2', status: 'failed', exit_code: null });
const qwen = { id: 'qwen_image_2_1', label: 'Qwen-Image 2.1' } as ModelInfo;

describe('RunsPage', () => {
  let http: HttpTestingController;
  let fixture: ComponentFixture<RunsPage>;
  let page: RunsPage;
  const root = (): HTMLElement => fixture.nativeElement;
  const all = (sel: string): HTMLElement[] => Array.from(root().querySelectorAll(sel));
  const text = (sel: string): string => root().querySelector(sel)?.textContent?.trim() ?? '';
  const button = (label: string): HTMLButtonElement =>
    all('button').find((b) => b.textContent?.trim() === label) as HTMLButtonElement;

  function start(jobs: JobStatusResponse[] = [running, queued, done, failed]): void {
    fixture = TestBed.createComponent(RunsPage);
    page = fixture.componentInstance;
    http.expectOne('/api/models').flush([qwen]);
    http.expectOne('/api/jobs').flush({ items: jobs });
    fixture.detectChanges();
  }

  beforeEach(() => {
    jest.useFakeTimers();
    TestBed.configureTestingModule({
      imports: [RunsPage],
      providers: [provideHttpClient(), provideHttpClientTesting(), provideRouter([])]
    });
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    fixture?.destroy();
    http.verify();
    jest.useRealTimers();
  });

  it('lists every run with its model label, progress, status and link', () => {
    start();
    const rows = all('.runRow');
    expect(rows.map((r) => r.querySelector('.runName')?.textContent)).toEqual([
      'cats',
      'dogs',
      'cats',
      'cats'
    ]);
    expect(rows[0].querySelector('.runMeta')?.textContent).toBe('Qwen-Image 2.1 · 100 steps');
    expect(rows[1].querySelector('.runMeta')?.textContent).toBe('mystery · 100 steps');
    expect(rows[0].querySelector('.progress')?.textContent?.trim()).toBe('40 / 100');
    expect(rows[0].querySelector('ui-progress-bar')?.className).toContain('ui-progress--live');
    expect(rows.map((r) => r.querySelector('ui-status-pill')?.textContent?.trim())).toEqual([
      'Training',
      'Queued',
      'Finished',
      'Failed'
    ]);
    expect(rows[0].querySelector('.runName')?.getAttribute('href')).toBe('/jobs/r1');
    expect(rows[1].children[4].textContent?.trim()).toBe('—');
    expect(rows[2].children[4].textContent?.trim()).toMatch(/14/);
    expect(text('ui-page-header .description')).toBe(
      '1 training, 1 queued. The queue runs one job at a time.'
    );
  });

  it('says how a failed run ended, when the exit code is known', () => {
    start([failed, crashed]);
    const errors = all('.runRow').map(
      (r) => r.querySelector('.runError')?.textContent?.trim() ?? null
    );
    expect(errors).toEqual(['Exited with code 3. Open the run for its output.', null]);
  });

  it('filters runs and counts each group', () => {
    start();
    const tabs = all('ui-segmented button');
    expect(tabs.map((t) => t.textContent?.replace(/\s+/g, ' ').trim())).toEqual([
      'All 4',
      'Active 2',
      'Finished 1',
      'Failed 1'
    ]);
    tabs[1].click();
    fixture.detectChanges();
    expect(
      all('.runRow').map((r) => r.querySelector('ui-status-pill')?.textContent?.trim())
    ).toEqual(['Training', 'Queued']);
    tabs[2].click();
    fixture.detectChanges();
    expect(all('.runRow')).toHaveLength(1);
    tabs[3].click();
    fixture.detectChanges();
    expect(page.visible().map((j) => j.job_id)).toEqual(['f1']);
  });

  it('explains an empty list and an empty filter differently', () => {
    start([]);
    expect(text('.empty')).toBe('No runs yet. Set up the first one.');
    page.jobs.set([done]);
    page.filter.set('failed');
    fixture.detectChanges();
    expect(text('.empty')).toBe('No runs match this filter.');
  });

  it('shows nothing in place of the list until the first load answers', () => {
    fixture = TestBed.createComponent(RunsPage);
    fixture.detectChanges();
    expect(root().querySelector('.empty')).toBeNull();
    http.expectOne('/api/models').flush('x', { status: 500, statusText: 'x' });
    http.expectOne('/api/jobs').flush('x', { status: 500, statusText: 'x' });
    fixture.detectChanges();
    expect(text('.jobsError')).toBe('Failed to load runs (HTTP 500)');
    expect(fixture.componentInstance.loaded()).toBe(true);
  });

  it('starts a queued run, disabled while another trains, and reloads', () => {
    start([queued]);
    const startBtn = button('Start');
    expect(startBtn.disabled).toBe(false);
    startBtn.click();
    http.expectOne('/api/jobs/q1/start').flush({ status: 'ok' });
    http.expectOne('/api/jobs').flush({ items: [running, queued] });
    fixture.detectChanges();
    expect(button('Start').disabled).toBe(true);
  });

  it('shows why an action failed and clears it on the next try', () => {
    start([queued]);
    page.startJob('q1');
    http
      .expectOne('/api/jobs/q1/start')
      .flush({ detail: 'Another job is running' }, { status: 409, statusText: 'x' });
    fixture.detectChanges();
    expect(text('.actionError')).toBe('Another job is running');
    expect(root().querySelector('.actionError')?.getAttribute('role')).toBe('alert');

    page.startJob('q1');
    expect(page.actionError()).toBeNull();
    http.expectOne('/api/jobs/q1/start').flush({ status: 'ok' });
    http.expectOne('/api/jobs').flush({ items: [] });
  });

  it('starts the next run from the header and re-enables the button either way', () => {
    start();
    button('Start next').click();
    fixture.detectChanges();
    expect(button('Starting…').disabled).toBe(true);
    http.expectOne('/api/jobs/start-next').flush({ status: 'ok' });
    http.expectOne('/api/jobs').flush({ items: [] });
    expect(page.startingQueue()).toBe(false);

    page.startNext();
    http.expectOne('/api/jobs/start-next').flush('x', { status: 500, statusText: 'x' });
    expect(page.startingQueue()).toBe(false);
    expect(page.actionError()).toBe('Failed to start the next job (HTTP 500)');
  });

  it('polls every three seconds, clears a list error on success, and stops on destroy', () => {
    start();
    jest.advanceTimersByTime(3000);
    http.expectOne('/api/jobs').flush('x', { status: 502, statusText: 'x' });
    expect(page.jobsError()).toBe('Failed to load runs (HTTP 502)');
    expect(page.jobs()).toHaveLength(4);
    jest.advanceTimersByTime(3000);
    http.expectOne('/api/jobs').flush({ items: [done] });
    expect(page.jobsError()).toBeNull();
    fixture.destroy();
    jest.advanceTimersByTime(6000);
    http.expectNone('/api/jobs');
  });
});

describe('formatStarted', () => {
  it('shows a dash before a run starts and a short date after', () => {
    expect(formatStarted(0)).toBe('—');
    expect(formatStarted(1_700_000_000)).toMatch(/14/);
  });
});
