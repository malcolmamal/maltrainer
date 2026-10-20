import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';

import {
  DatasetInfo,
  JobStatusResponse,
  ModelInfo,
  StartTrainRequest,
  TrainPlan
} from '../../services/trainer-api.service';
import { TrainerPage } from './trainer.page';

const datasets: DatasetInfo[] = [
  { name: 'cats', path: '/d/cats', image_count: 12 },
  { name: 'dogs', path: '/d/dogs', image_count: 30 }
];

const qwen: ModelInfo = {
  id: 'qwen_image_2_1',
  label: 'Qwen-Image 2.1',
  optimizers: [
    { name: 'automagic3', label: 'Automagic3', default_lr: 0.0001, note: 'Recommended for Qwen' },
    { name: 'adamw8bit', label: 'AdamW 8-bit', default_lr: 0.0002, note: '' }
  ],
  default_optimizer: 'automagic3',
  default_duration_mode: 'steps',
  default_epochs: 70,
  default_max_steps: 2000,
  native_max_steps: true,
  sec_per_step: 5,
  models_root: '/models',
  missing_files: [],
  auto_steps: true,
  target_epochs: 70,
  min_auto_steps: 500,
  max_auto_steps: 3000,
  resolution_note: 'Trains at 1024px'
};

const zimage: ModelInfo = {
  ...qwen,
  id: 'z_image',
  label: 'Z-Image',
  optimizers: [{ name: 'prodigy', label: 'Prodigy', default_lr: 1, note: '' }],
  default_optimizer: 'prodigy',
  default_duration_mode: 'epochs',
  default_epochs: 30,
  default_max_steps: 900,
  native_max_steps: false,
  auto_steps: false,
  missing_files: ['vae.safetensors'],
  resolution_note: ''
};

const plan: TrainPlan = {
  model: 'qwen_image_2_1',
  dataset: 'cats',
  optimizer: 'automagic3',
  learning_rate: 0.0001,
  duration_mode: 'steps',
  image_count: 12,
  steps_per_epoch: 12,
  epochs: 70,
  total_steps: 840,
  save_every: 100,
  output_path: '/out/cats',
  duration: '840 steps',
  auto_steps: true,
  estimated_minutes: 70,
  notes: ['note one'],
  warnings: ['careful']
};

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
    optimizer: 'automagic3',
    duration: '100 steps',
    ...overrides
  };
}

describe('TrainerPage', () => {
  let http: HttpTestingController;
  let fixture: ComponentFixture<TrainerPage>;
  let page: TrainerPage;

  const root = (): HTMLElement => fixture.nativeElement;
  const text = (selector: string): string =>
    root().querySelector(selector)?.textContent?.trim() ?? '';
  const all = (selector: string): HTMLElement[] => Array.from(root().querySelectorAll(selector));
  const button = (label: string): HTMLButtonElement => {
    const found = all('button').find((b) => b.textContent?.trim() === label);
    if (!found) throw new Error(`No button "${label}"`);
    return found as HTMLButtonElement;
  };
  const field = (label: string): HTMLInputElement | HTMLSelectElement => {
    const found = all('label.field').find(
      (l) => l.querySelector('span')?.textContent?.trim() === label
    );
    const control = found?.querySelector<HTMLInputElement | HTMLSelectElement>('input, select');
    if (!control) throw new Error(`No field "${label}"`);
    return control;
  };
  const type = async (
    control: HTMLInputElement | HTMLSelectElement,
    value: string
  ): Promise<void> => {
    control.value = value;
    control.dispatchEvent(new Event(control instanceof HTMLSelectElement ? 'change' : 'input'));
    await settle();
  };
  /** ngModel writes to the DOM in a microtask; real timers stay faked, so do not use whenStable. */
  const settle = async (): Promise<void> => {
    fixture.detectChanges();
    await Promise.resolve();
    fixture.detectChanges();
  };

  /** Create the page and answer the three startup requests. */
  function start(
    opts: { datasets?: DatasetInfo[]; models?: ModelInfo[]; jobs?: JobStatusResponse[] } = {}
  ): void {
    fixture = TestBed.createComponent(TrainerPage);
    page = fixture.componentInstance;
    http.expectOne('/api/datasets').flush(opts.datasets ?? datasets);
    http.expectOne('/api/models').flush(opts.models ?? [qwen, zimage]);
    http.expectOne('/api/jobs').flush({ items: opts.jobs ?? [] });
    fixture.detectChanges();
  }

  /** Let the debounce run and answer the resulting plan request. */
  function answerPlan(body: TrainPlan = plan): StartTrainRequest {
    jest.advanceTimersByTime(250);
    const req = http.expectOne('/api/plan');
    req.flush(body);
    fixture.detectChanges();
    return req.request.body;
  }

  beforeEach(() => {
    jest.useFakeTimers();
    TestBed.configureTestingModule({
      imports: [TrainerPage],
      providers: [provideHttpClient(), provideHttpClientTesting(), provideRouter([])]
    });
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    fixture?.destroy();
    jest.useRealTimers();
  });

  describe('startup', () => {
    it('lists datasets, selects the first and prefers the Qwen model', () => {
      start();

      expect(all('.datasetItem').map((d) => d.querySelector('h3')?.textContent)).toEqual([
        'cats',
        'dogs'
      ]);
      expect(all('.datasetItem')[0].classList).toContain('active');
      expect(page.selectedModelId()).toBe('qwen_image_2_1');
      expect(page.optimizer()).toBe('automagic3');
      expect(page.learningRate()).toBe(0.0001);
      expect(page.maxSteps()).toBeNull();
      expect(text('.hint')).toBe('Trains at 1024px');
    });

    it('falls back to the first model when Qwen is absent', () => {
      start({ models: [zimage] });
      expect(page.selectedModelId()).toBe('z_image');
      expect(page.durationMode()).toBe('epochs');
    });

    it('shows placeholders when nothing is available', () => {
      start({ datasets: [], models: [] });
      expect(text('.placeholder')).toBe('No datasets found.');
      expect(page.selectedModel()).toBeNull();
      expect(text('.jobList')).toBe('');
      expect(all('.placeholder').map((p) => p.textContent)).toContain('No active jobs yet.');
    });

    it('shows a loading placeholder until datasets arrive', () => {
      fixture = TestBed.createComponent(TrainerPage);
      fixture.detectChanges();
      expect(text('.placeholder')).toBe('Loading datasets…');
      http.expectOne('/api/datasets');
      http.expectOne('/api/models');
      http.expectOne('/api/jobs');
    });

    it('keeps an existing dataset selection when datasets are reloaded', () => {
      start();
      page.selectedDataset.set('dogs');
      page.loadDatasets();
      http.expectOne('/api/datasets').flush(datasets);
      expect(page.selectedDataset()).toBe('dogs');
      expect(page.loading()).toBe(false);
    });

    it('reports a dataset load failure and stops loading', () => {
      fixture = TestBed.createComponent(TrainerPage);
      page = fixture.componentInstance;
      http.expectOne('/api/datasets').flush('x', { status: 500, statusText: 'Server Error' });
      http.expectOne('/api/models').flush([qwen]);
      http.expectOne('/api/jobs').flush({ items: [] });
      fixture.detectChanges();

      expect(page.loading()).toBe(false);
      expect(text('.error')).toContain('500');
    });

    it('reports a model load failure using the backend detail', () => {
      fixture = TestBed.createComponent(TrainerPage);
      page = fixture.componentInstance;
      http.expectOne('/api/datasets').flush(datasets);
      http
        .expectOne('/api/models')
        .flush({ detail: 'models broken' }, { status: 500, statusText: 'x' });
      http.expectOne('/api/jobs').flush({ items: [] });
      fixture.detectChanges();

      expect(text('.error')).toBe('models broken');
    });
  });

  describe('setup form', () => {
    it('selects a dataset by clicking it', async () => {
      start();
      all('.datasetItem')[1].click();
      await settle();

      expect(page.selectedDataset()).toBe('dogs');
      expect((field('Selected dataset') as HTMLSelectElement).value).toBe('dogs');
      answerPlan();
    });

    it('refreshes datasets with the button and disables it while loading', () => {
      start();
      answerPlan();
      button('Refresh').click();
      fixture.detectChanges();
      expect(page.loading()).toBe(true);
      http.expectOne('/api/datasets').flush(datasets);
    });

    it('resets every model-specific field when the model changes', async () => {
      start();
      answerPlan();
      await type(field('Model'), 'z_image');

      expect(page.selectedModelId()).toBe('z_image');
      expect(page.durationMode()).toBe('epochs');
      expect(page.epochs()).toBe(30);
      expect(page.optimizer()).toBe('prodigy');
      expect(page.learningRate()).toBe(1);
      expect(page.saveEvery()).toBeNull();
      expect(text('.error')).toContain('vae.safetensors');
      expect(text('.error')).toContain('under /models');
      answerPlan();
    });

    it('uses the model default step budget when the model has no auto steps', () => {
      start({ models: [{ ...zimage, default_duration_mode: 'steps' }] });
      expect(page.maxSteps()).toBe(900);
      answerPlan();
    });

    it('ignores an unknown model id', () => {
      start();
      answerPlan();
      page.selectModel('nope');
      expect(page.selectedModelId()).toBe('qwen_image_2_1');
    });

    it('pre-fills the learning rate and shows the note when the optimizer changes', async () => {
      start();
      answerPlan();
      await type(field('Optimizer'), 'adamw8bit');

      expect(page.learningRate()).toBe(0.0002);
      expect((field('Learning rate') as HTMLInputElement).value).toBe('0.0002');
      answerPlan();
    });

    it('clears the learning rate for an unknown optimizer', () => {
      start();
      answerPlan();
      page.selectOptimizer('mystery');
      expect(page.learningRate()).toBeNull();
      expect(page.selectedOptimizer()).toBeNull();
      fixture.detectChanges();
      answerPlan();
    });

    it('shows the optimizer note and the auto steps hint', () => {
      start();
      answerPlan();
      const hints = all('.hint').map((h) => h.textContent?.replace(/\s+/g, ' ').trim());
      expect(hints).toContain('Recommended for Qwen');
      expect(hints.some((h) => h?.includes('about 70 epochs') && h.includes('500–3000'))).toBe(
        true
      );
    });

    it('explains rounding for trainers without native step budgets', () => {
      start({ models: [{ ...zimage, default_duration_mode: 'steps' }] });
      answerPlan();
      expect(root().textContent).toContain('trains whole epochs');
    });

    it('toggles between max steps and epochs inputs', () => {
      start();
      answerPlan();
      expect(all('label.field span').map((s) => s.textContent)).toContain('Max steps');

      button('Epochs').click();
      fixture.detectChanges();
      expect(page.durationMode()).toBe('epochs');
      expect(button('Epochs').getAttribute('aria-checked')).toBe('true');
      expect(all('label.field span').map((s) => s.textContent)).toContain('Epochs');
      expect(all('label.field span').map((s) => s.textContent)).not.toContain('Max steps');

      button('Max steps').click();
      fixture.detectChanges();
      expect(page.durationMode()).toBe('steps');
      answerPlan();
    });

    it('shows the trigger-word prompt hint only for Qwen and uses the dataset as placeholder', () => {
      start();
      answerPlan();
      expect((field('Trigger word') as HTMLInputElement).placeholder).toBe('cats (dataset name)');
      expect(text('code')).toBe('cats, a woman, …');
      page.trigger.set('sks');
      fixture.detectChanges();
      expect(text('code')).toBe('sks, a woman, …');
      answerPlan();

      page.selectModel('z_image');
      fixture.detectChanges();
      expect((field('Trigger word') as HTMLInputElement).placeholder).toBe('template default');
      expect(root().textContent).not.toContain('a woman');
      answerPlan();
    });
  });

  describe('buildRequest', () => {
    it('returns null without a dataset or a model', () => {
      start({ datasets: [], models: [] });
      expect(page.buildRequest()).toBeNull();
    });

    it('omits empty or invalid optional values', () => {
      start();
      answerPlan();
      page.learningRate.set(0);
      page.saveEvery.set(-3);
      page.trigger.set('   ');
      page.optimizer.set('');

      expect(page.buildRequest()).toEqual({
        dataset: 'cats',
        model: 'qwen_image_2_1',
        optimizer: undefined,
        learning_rate: undefined,
        duration_mode: 'steps',
        epochs: undefined,
        max_steps: undefined,
        save_every: undefined,
        trigger: undefined,
        dry_run: false,
        dry_run_step_time: 0.025
      });
      fixture.detectChanges();
      answerPlan();
    });

    it('floors and trims numeric and text values in steps mode', () => {
      start();
      answerPlan();
      page.maxSteps.set(1234.9);
      page.saveEvery.set(99.5);
      page.trigger.set('  sks ');
      page.dryRun.set(true);
      page.dryRunStepTime.set(0.5);

      expect(page.buildRequest()).toMatchObject({
        max_steps: 1234,
        save_every: 99,
        trigger: 'sks',
        epochs: undefined,
        dry_run: true,
        dry_run_step_time: 0.5
      });
      fixture.detectChanges();
      answerPlan();
    });

    it('sends at least one epoch in epochs mode and no step budget', () => {
      start();
      answerPlan();
      page.durationMode.set('epochs');
      page.maxSteps.set(500);
      page.epochs.set(0);
      expect(page.buildRequest()).toMatchObject({ epochs: 1, max_steps: undefined });
      page.epochs.set(7.8);
      expect(page.buildRequest()?.epochs).toBe(7);
      fixture.detectChanges();
      answerPlan();
    });
  });

  describe('plan preview', () => {
    it('requests a plan after the debounce and renders it', () => {
      start();
      jest.advanceTimersByTime(249);
      http.expectNone('/api/plan');

      const body = answerPlan();
      expect(body).toMatchObject({
        dataset: 'cats',
        model: 'qwen_image_2_1',
        optimizer: 'automagic3'
      });
      expect(text('.planCard .inlineInfo')).toContain('840');
      expect(text('.planCard .inlineInfo')).toContain('≈ 70 min');
      expect(text('.planCard')).toContain('/out/cats');
      expect(text('.planCard')).toContain('checkpoint every 100 steps');
      expect(text('.note')).toBe('note one');
      expect(text('.warning')).toBe('careful');
    });

    it('shows a dash when the time estimate is unknown', () => {
      start();
      answerPlan({ ...plan, estimated_minutes: null });
      expect(text('.planCard .inlineInfo')).toContain('—');
    });

    it('debounces rapid edits into one plan request', () => {
      start();
      page.maxSteps.set(100);
      fixture.detectChanges();
      jest.advanceTimersByTime(100);
      page.maxSteps.set(200);
      fixture.detectChanges();
      jest.advanceTimersByTime(100);
      http.expectNone('/api/plan');

      const body = answerPlan();
      expect(body.max_steps).toBe(200);
    });

    it('shows the backend detail when planning fails and clears the plan', () => {
      start();
      answerPlan();
      page.maxSteps.set(5);
      fixture.detectChanges();
      jest.advanceTimersByTime(250);
      http
        .expectOne('/api/plan')
        .flush({ detail: 'Output folder exists' }, { status: 400, statusText: 'Bad' });
      fixture.detectChanges();

      expect(page.plan()).toBeNull();
      expect(text('.error')).toBe('Output folder exists');
      expect(root().querySelector('.planCard')).toBeNull();
    });

    it('joins validation messages from a list of details', () => {
      start();
      jest.advanceTimersByTime(250);
      http
        .expectOne('/api/plan')
        .flush(
          { detail: [{ msg: 'bad lr' }, { msg: '' }, { msg: 'bad steps' }] },
          { status: 422, statusText: 'x' }
        );
      expect(page.planError()).toBe('bad lr; bad steps');
    });

    it('falls back to the default message for an empty detail list', () => {
      start();
      jest.advanceTimersByTime(250);
      http.expectOne('/api/plan').flush({ detail: [] }, { status: 422, statusText: 'x' });
      expect(page.planError()).toBe('Could not plan this run');
    });

    it('clears the plan and sends nothing when there is no dataset', () => {
      start({ datasets: [] });
      jest.advanceTimersByTime(250);
      http.expectNone('/api/plan');
      expect(page.plan()).toBeNull();
    });
  });

  describe('queueing a run', () => {
    it('posts the request, reloads jobs and re-plans', () => {
      start();
      answerPlan();
      button('Add to queue').click();
      fixture.detectChanges();

      expect(button('Adding…').disabled).toBe(true);
      const req = http.expectOne('/api/train');
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toMatchObject({ dataset: 'cats', model: 'qwen_image_2_1' });
      req.flush({ job_id: 'j1' });
      http.expectOne('/api/jobs').flush({ items: [job({})] });
      fixture.detectChanges();

      expect(button('Add to queue').disabled).toBe(false);
      expect(all('.jobItem')).toHaveLength(1);
      // The output folder is now taken, so the preview is requested again.
      answerPlan();
    });

    it('shows the backend error and re-enables the button when queueing fails', () => {
      start();
      answerPlan();
      button('Add to queue').click();
      http
        .expectOne('/api/train')
        .flush({ detail: 'Model files missing' }, { status: 400, statusText: 'Bad' });
      fixture.detectChanges();

      expect(text('.error')).toBe('Model files missing');
      expect(button('Add to queue').disabled).toBe(false);
    });

    it('asks for a dataset and a model when none are selected', () => {
      start({ datasets: [], models: [] });
      button('Add to queue').click();
      fixture.detectChanges();

      http.expectNone('/api/train');
      expect(text('.error')).toBe('Select a dataset and a model to start training.');
    });
  });

  describe('jobs', () => {
    const queued = job({ job_id: 'q1', status: 'queued' });
    const running = job({ job_id: 'r1', status: 'running', progress: { step: 5, total: 50 } });

    it('renders jobs with model label, progress and an open link', () => {
      start({ jobs: [queued, running] });
      answerPlan();

      const items = all('.jobItem');
      expect(items).toHaveLength(2);
      expect(items[1].textContent).toContain('Qwen-Image 2.1 · automagic3 · 100 steps');
      expect(items[1].textContent).toContain('5 / 50 · running');
      expect(items[0].querySelector('a')?.getAttribute('href')).toBe('/jobs/q1');
    });

    it('falls back to the raw model id for unknown models', () => {
      start({ jobs: [job({ model: 'mystery' })] });
      answerPlan();
      expect(text('.jobMeta')).toContain('mystery ·');
      expect(page.modelLabel('mystery')).toBe('mystery');
    });

    it('offers Start for queued jobs only, disabled while another job runs', () => {
      start({ jobs: [queued, running] });
      answerPlan();
      expect(all('.jobActions button')).toHaveLength(1);
      expect((all('.jobActions button')[0] as HTMLButtonElement).disabled).toBe(true);
      expect(page.hasRunning()).toBe(true);
    });

    it('starts a queued job and reloads the list', () => {
      start({ jobs: [queued] });
      answerPlan();
      (all('.jobActions button')[0] as HTMLButtonElement).click();

      const req = http.expectOne('/api/jobs/q1/start');
      expect(req.request.method).toBe('POST');
      req.flush({ status: 'started' });
      http.expectOne('/api/jobs').flush({ items: [{ ...queued, status: 'running' }] });
      expect(page.jobs()[0].status).toBe('running');
    });

    it('ignores a failure to start a job', () => {
      start({ jobs: [queued] });
      answerPlan();
      page.startJob('q1');
      http.expectOne('/api/jobs/q1/start').flush('x', { status: 409, statusText: 'Conflict' });
      http.expectNone('/api/jobs');
      expect(page.error()).toBeNull();
    });

    it('starts the next job with the queue button and reloads', () => {
      start({ jobs: [queued] });
      answerPlan();
      button('Start next').click();
      fixture.detectChanges();
      expect(button('Starting…').disabled).toBe(true);

      http.expectOne('/api/jobs/start-next').flush({ status: 'started' });
      http.expectOne('/api/jobs').flush({ items: [] });
      fixture.detectChanges();
      expect(button('Start next').disabled).toBe(false);
    });

    it('re-enables the queue button when starting the next job fails', () => {
      start();
      answerPlan();
      page.startNext();
      http.expectOne('/api/jobs/start-next').flush('x', { status: 500, statusText: 'x' });
      expect(page.startingQueue()).toBe(false);
    });

    it('refreshes jobs on demand and keeps the old list when loading fails', () => {
      start({ jobs: [queued] });
      answerPlan();
      all('.cardActions button')[0].click();
      http.expectOne('/api/jobs').flush('x', { status: 500, statusText: 'x' });
      expect(page.jobs()).toHaveLength(1);
    });

    it('polls the job list every three seconds and stops after destroy', () => {
      start();
      answerPlan();
      jest.advanceTimersByTime(3000);
      http.expectOne('/api/jobs').flush({ items: [running] });
      expect(page.jobs()).toEqual([running]);

      fixture.destroy();
      jest.advanceTimersByTime(6000);
      http.expectNone('/api/jobs');
    });
  });
});
