import {
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  inject,
  signal,
  OnDestroy
} from '@angular/core';
import { RouterLink } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import {
  DatasetInfo,
  DurationMode,
  JobStatusResponse,
  ModelInfo,
  StartTrainRequest,
  TrainerApiService,
  TrainPlan
} from '../../services/trainer-api.service';

@Component({
  selector: 'app-trainer-page',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './trainer.page.html',
  styleUrl: './trainer.page.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class TrainerPage implements OnDestroy {
  private readonly api = inject(TrainerApiService);

  readonly datasets = signal<DatasetInfo[]>([]);
  readonly loading = signal(true);
  readonly error = signal<string | null>(null);
  readonly selectedDataset = signal<string>('');
  readonly dryRun = signal(false);
  readonly dryRunStepTime = signal(0.025);
  readonly starting = signal(false);
  readonly jobs = signal<JobStatusResponse[]>([]);
  readonly startingQueue = signal(false);

  readonly models = signal<ModelInfo[]>([]);
  readonly selectedModelId = signal<string>('');
  readonly optimizer = signal<string>('');
  readonly learningRate = signal<number | null>(null);
  readonly durationMode = signal<DurationMode>('steps');
  readonly epochs = signal<number>(70);
  /** null = auto: the backend picks a step budget from the dataset size. */
  readonly maxSteps = signal<number | null>(null);
  readonly saveEvery = signal<number | null>(null);
  readonly trigger = signal<string>('');

  readonly plan = signal<TrainPlan | null>(null);
  readonly planError = signal<string | null>(null);
  /** Bumped after a job is queued: its output folder is taken, so the preview must re-plan. */
  private readonly planRevision = signal(0);

  private pollTimer: number | null = null;

  readonly selectedInfo = computed(
    () => this.datasets().find((d) => d.name === this.selectedDataset()) ?? null
  );

  readonly selectedModel = computed(
    () => this.models().find((m) => m.id === this.selectedModelId()) ?? null
  );

  readonly selectedOptimizer = computed(
    () => this.selectedModel()?.optimizers.find((o) => o.name === this.optimizer()) ?? null
  );

  readonly hasRunning = computed(() => this.jobs().some((job) => job.status === 'running'));

  readonly modelsMissing = computed(() => this.selectedModel()?.missing_files ?? []);

  constructor() {
    this.loadDatasets();
    this.loadModels();
    this.loadJobs();
    this.pollTimer = window.setInterval(() => this.loadJobs(), 3000);

    // Re-plan (debounced) whenever anything that affects the run changes.
    effect((onCleanup) => {
      this.planRevision();
      const request = this.buildRequest();
      if (!request) {
        this.plan.set(null);
        return;
      }
      const timer = window.setTimeout(() => {
        this.api.planTraining(request).subscribe({
          next: (plan) => {
            this.plan.set(plan);
            this.planError.set(null);
          },
          error: (err) => {
            this.plan.set(null);
            this.planError.set(this.describeError(err, 'Could not plan this run'));
          }
        });
      }, 250);
      onCleanup(() => window.clearTimeout(timer));
    });
  }

  loadDatasets() {
    this.loading.set(true);
    this.error.set(null);
    this.api.listDatasets().subscribe({
      next: (items) => {
        this.datasets.set(items);
        if (!this.selectedDataset() && items.length > 0) {
          this.selectedDataset.set(items[0].name);
        }
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(err?.message ?? 'Failed to load datasets');
        this.loading.set(false);
      }
    });
  }

  loadModels() {
    this.api.listModels().subscribe({
      next: (items) => {
        this.models.set(items);
        if (!this.selectedModelId() && items.length > 0) {
          const preferred = items.find((m) => m.id === 'qwen_image_2_1') ?? items[0];
          this.selectModel(preferred.id);
        }
      },
      error: (err) => this.error.set(this.describeError(err, 'Failed to load models'))
    });
  }

  /** Switching model resets every model-specific field to that model's defaults. */
  selectModel(modelId: string) {
    const model = this.models().find((m) => m.id === modelId);
    if (!model) return;
    this.selectedModelId.set(model.id);
    this.durationMode.set(model.default_duration_mode);
    this.epochs.set(model.default_epochs);
    this.maxSteps.set(model.auto_steps ? null : model.default_max_steps);
    this.saveEvery.set(null);
    this.selectOptimizer(model.default_optimizer);
  }

  /** Each optimizer has its own sensible learning rate; picking one pre-fills it. */
  selectOptimizer(name: string) {
    this.optimizer.set(name);
    const opt = this.selectedModel()?.optimizers.find((o) => o.name === name);
    this.learningRate.set(opt ? opt.default_lr : null);
  }

  loadJobs() {
    this.api.listJobs().subscribe({
      next: (resp) => this.jobs.set(resp.items),
      error: () => undefined
    });
  }

  startNext() {
    this.startingQueue.set(true);
    this.api.startNextJob().subscribe({
      next: () => {
        this.startingQueue.set(false);
        this.loadJobs();
      },
      error: () => {
        this.startingQueue.set(false);
      }
    });
  }

  buildRequest(): StartTrainRequest | null {
    const dataset = this.selectedDataset();
    const model = this.selectedModelId();
    if (!dataset || !model) return null;
    const mode = this.durationMode();
    const lr = this.learningRate();
    const saveEvery = this.saveEvery();
    const maxSteps = this.maxSteps();
    const trigger = this.trigger().trim();
    return {
      dataset,
      model,
      optimizer: this.optimizer() || undefined,
      learning_rate: lr && lr > 0 ? lr : undefined,
      duration_mode: mode,
      epochs: mode === 'epochs' ? Math.max(1, Math.floor(this.epochs() || 1)) : undefined,
      max_steps: mode === 'steps' && maxSteps && maxSteps > 0 ? Math.floor(maxSteps) : undefined,
      save_every: saveEvery && saveEvery > 0 ? Math.floor(saveEvery) : undefined,
      trigger: trigger || undefined,
      dry_run: this.dryRun(),
      dry_run_step_time: this.dryRunStepTime()
    };
  }

  startTraining() {
    const body = this.buildRequest();
    if (!body) {
      this.error.set('Select a dataset and a model to start training.');
      return;
    }
    this.starting.set(true);
    this.error.set(null);

    this.api.startTraining(body).subscribe({
      next: () => {
        this.starting.set(false);
        this.loadJobs();
        this.planRevision.update((v) => v + 1);
      },
      error: (err) => {
        this.error.set(this.describeError(err, 'Failed to start training'));
        this.starting.set(false);
      }
    });
  }

  startJob(jobId: string) {
    this.api.startJob(jobId).subscribe({
      next: () => this.loadJobs(),
      error: () => undefined
    });
  }

  modelLabel(id: string): string {
    return this.models().find((m) => m.id === id)?.label ?? id;
  }

  private describeError(err: unknown, fallback: string): string {
    const e = err as { error?: { detail?: unknown }; message?: string };
    const detail = e?.error?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      return (
        detail
          .map((d: { msg?: string }) => d?.msg ?? '')
          .filter(Boolean)
          .join('; ') || fallback
      );
    }
    return e?.message ?? fallback;
  }

  ngOnDestroy() {
    if (this.pollTimer) {
      window.clearInterval(this.pollTimer);
    }
  }
}
