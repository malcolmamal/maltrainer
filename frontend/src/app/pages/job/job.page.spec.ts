import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ActivatedRoute, convertToParamMap, provideRouter } from '@angular/router';

import { JobStatusResponse } from '../../services/trainer-api.service';
import { JobPage } from './job.page';

class FakeSocket {
  static instances: FakeSocket[] = [];
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;

  constructor(readonly url: string) {
    FakeSocket.instances.push(this);
  }

  close(): void {
    this.closed = true;
  }

  emit(payload: unknown): void {
    this.onmessage?.({ data: typeof payload === 'string' ? payload : JSON.stringify(payload) });
  }
}

const job: JobStatusResponse = {
  job_id: 'j1',
  dataset: 'cats',
  output_path: '/out/cats',
  config_path: '/cfg/cats.yaml',
  status: 'running',
  started_at: 1,
  ended_at: null,
  exit_code: null,
  progress: { step: 10, total: 40 },
  model: 'qwen_image_2_1',
  optimizer: 'automagic3',
  duration: '40 steps'
};

describe('JobPage', () => {
  let http: HttpTestingController;
  const originalWebSocket = globalThis.WebSocket;

  function setup(id: string | null): ComponentFixture<JobPage> {
    TestBed.configureTestingModule({
      imports: [JobPage],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        {
          provide: ActivatedRoute,
          useValue: { snapshot: { paramMap: convertToParamMap(id ? { id } : {}) } }
        }
      ]
    });
    http = TestBed.inject(HttpTestingController);
    return TestBed.createComponent(JobPage);
  }

  const socket = (): FakeSocket => FakeSocket.instances[0];
  const root = (fixture: ComponentFixture<JobPage>): HTMLElement => fixture.nativeElement;
  const text = (fixture: ComponentFixture<JobPage>, selector: string): string =>
    root(fixture).querySelector(selector)?.textContent?.trim() ?? '';

  beforeEach(() => {
    FakeSocket.instances = [];
    globalThis.WebSocket = FakeSocket as unknown as typeof WebSocket;
  });

  afterEach(() => {
    globalThis.WebSocket = originalWebSocket;
    http.verify();
  });

  it('loads the job and fills the summary and progress', () => {
    const fixture = setup('j1');
    http.expectOne('/api/jobs/j1').flush(job);
    fixture.detectChanges();

    expect(text(fixture, '.status')).toBe('running');
    expect(text(fixture, '.summary')).toContain('cats');
    expect(text(fixture, '.summary')).toContain('qwen_image_2_1 · automagic3');
    expect(text(fixture, '.progressInfo')).toContain('10 / 40');
    expect(fixture.componentInstance.progressPercent()).toBe(25);
    expect(root(fixture).querySelector<HTMLElement>('.progressFill')?.style.width).toBe('25%');
  });

  it('opens a websocket for the job id on the page host', () => {
    const fixture = setup('j1');
    http.expectOne('/api/jobs/j1').flush(job);
    fixture.detectChanges();

    expect(socket().url).toBe(`ws://${window.location.host}/ws/jobs/j1`);
  });

  it('shows an error and opens nothing when the job id is missing', () => {
    const fixture = setup(null);
    fixture.detectChanges();

    expect(text(fixture, '.error')).toBe('Missing job id');
    expect(FakeSocket.instances).toHaveLength(0);
  });

  it('shows the error message when the job cannot be loaded', () => {
    const fixture = setup('j1');
    http.expectOne('/api/jobs/j1').flush('boom', { status: 500, statusText: 'Server Error' });
    fixture.detectChanges();

    expect(text(fixture, '.error')).toContain('500');
  });

  it('appends log lines and marks stderr lines', () => {
    const fixture = setup('j1');
    http.expectOne('/api/jobs/j1').flush(job);
    socket().emit({ type: 'log', stream: 'stdout', line: 'hello' });
    socket().emit({ type: 'log', stream: 'stderr', line: 'oops' });
    fixture.detectChanges();

    const lines = Array.from(root(fixture).querySelectorAll('.logLine'));
    expect(
      lines.map((l) =>
        Array.from(l.querySelectorAll('span'))
          .map((s) => s.textContent)
          .join(' ')
      )
    ).toEqual(['stdout hello', 'stderr oops']);
    expect(lines.map((l) => l.classList.contains('err'))).toEqual([false, true]);
  });

  it('keeps progress-bar lines out of the log and shows the latest one live', () => {
    const fixture = setup('j1');
    http.expectOne('/api/jobs/j1').flush(job);
    socket().emit({
      type: 'log',
      stream: 'stderr',
      line: '10%|#',
      replace: true,
      is_progress: true
    });
    socket().emit({
      type: 'log',
      stream: 'stderr',
      line: '20%|##',
      replace: true,
      is_progress: true
    });
    fixture.detectChanges();

    expect(fixture.componentInstance.logs()).toEqual([]);
    expect(text(fixture, '.liveProgress .text')).toBe('20%|##');
    expect(root(fixture).querySelector('.liveProgress')?.classList.contains('err')).toBe(true);
  });

  it('updates progress, status and job from socket messages', () => {
    const fixture = setup('j1');
    http.expectOne('/api/jobs/j1').flush(job);

    socket().emit({ type: 'progress', step: 30, total: 40 });
    expect(fixture.componentInstance.progressPercent()).toBe(75);

    socket().emit({ type: 'status', status: 'failed' });
    fixture.detectChanges();
    expect(text(fixture, '.status')).toBe('failed');
    expect(root(fixture).querySelector('.status')?.classList.contains('failed')).toBe(true);

    socket().emit({ type: 'status' });
    expect(fixture.componentInstance.status()).toBe('failed');

    socket().emit({
      type: 'job',
      job: { ...job, status: 'exited', progress: { step: 40, total: 40 } }
    });
    fixture.detectChanges();
    expect(text(fixture, '.status')).toBe('exited');
    expect(root(fixture).querySelector('.status')?.classList.contains('exited')).toBe(true);
    expect(fixture.componentInstance.progressStep()).toBe(40);
  });

  it('defaults missing progress values and keeps previous ones on a sparse job message', () => {
    const fixture = setup('j1');
    http.expectOne('/api/jobs/j1').flush(job);
    socket().emit({ type: 'progress' });
    expect(fixture.componentInstance.progressTotal()).toBe(0);
    expect(fixture.componentInstance.progressPercent()).toBe(0);

    socket().emit({ type: 'progress', step: 5, total: 10 });
    socket().emit({ type: 'job', job: { ...job, progress: undefined, status: undefined } });
    expect(fixture.componentInstance.progressStep()).toBe(5);
    expect(fixture.componentInstance.progressTotal()).toBe(10);
  });

  it('caps the progress percent at 100', () => {
    const fixture = setup('j1');
    http.expectOne('/api/jobs/j1').flush({ ...job, progress: { step: 50, total: 40 } });
    expect(fixture.componentInstance.progressPercent()).toBe(100);
  });

  it('defaults progress to zero when the loaded job has none', () => {
    const fixture = setup('j1');
    http.expectOne('/api/jobs/j1').flush({ ...job, progress: undefined });
    expect(fixture.componentInstance.progressStep()).toBe(0);
    expect(fixture.componentInstance.progressTotal()).toBe(0);
  });

  it('ignores malformed socket messages', () => {
    const fixture = setup('j1');
    http.expectOne('/api/jobs/j1').flush(job);
    expect(() => socket().emit('not json')).not.toThrow();
    expect(fixture.componentInstance.logs()).toEqual([]);
    expect(fixture.componentInstance.error()).toBeNull();
  });

  it('reports websocket errors', () => {
    const fixture = setup('j1');
    http.expectOne('/api/jobs/j1').flush(job);
    socket().onerror?.();
    fixture.detectChanges();

    expect(text(fixture, '.error')).toBe('WebSocket connection error');
  });

  it('closes the socket on destroy', () => {
    const fixture = setup('j1');
    http.expectOne('/api/jobs/j1').flush(job);
    fixture.destroy();

    expect(socket().closed).toBe(true);
  });

  it('can be destroyed without a socket', () => {
    const fixture = setup(null);
    expect(() => fixture.destroy()).not.toThrow();
  });
});
