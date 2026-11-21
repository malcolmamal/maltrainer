import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  signal,
  OnDestroy
} from '@angular/core';
import { RouterLink } from '@angular/router';

import { describeHttpError } from '../../services/http-error';
import {
  JobStatusResponse,
  ModelInfo,
  TrainerApiService
} from '../../services/trainer-api.service';
import { SegmentOption, statusTone, UI } from '../../ui';

export type RunFilter = 'all' | 'active' | 'finished' | 'failed';

const FILTERS: Record<RunFilter, (job: JobStatusResponse) => boolean> = {
  all: () => true,
  active: (job) => job.status === 'running' || job.status === 'queued',
  finished: (job) => job.status === 'exited',
  failed: (job) => job.status === 'failed'
};

const LABELS: Record<RunFilter, string> = {
  all: 'All',
  active: 'Active',
  finished: 'Finished',
  failed: 'Failed'
};

/** Job start time as "Nov 20, 21:04" in the browser's locale, or a dash before it starts. */
export function formatStarted(epochSeconds: number): string {
  if (!epochSeconds) return '—';
  return new Date(epochSeconds * 1000).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
}

@Component({
  selector: 'app-runs-page',
  imports: [RouterLink, ...UI],
  templateUrl: './runs.page.html',
  styleUrl: './runs.page.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class RunsPage implements OnDestroy {
  private readonly api = inject(TrainerApiService);

  readonly jobs = signal<JobStatusResponse[]>([]);
  readonly models = signal<ModelInfo[]>([]);
  readonly loaded = signal(false);
  readonly filter = signal<RunFilter>('all');
  readonly startingQueue = signal(false);
  readonly jobsError = signal<string | null>(null);
  readonly actionError = signal<string | null>(null);

  readonly filterOptions = computed<SegmentOption<RunFilter>[]>(() =>
    (Object.keys(FILTERS) as RunFilter[]).map((value) => ({
      value,
      label: LABELS[value],
      detail: this.jobs().filter(FILTERS[value]).length
    }))
  );
  readonly visible = computed(() => this.jobs().filter(FILTERS[this.filter()]));
  readonly hasRunning = computed(() => this.jobs().some((job) => job.status === 'running'));
  readonly summary = computed(() => {
    const running = this.jobs().filter((j) => j.status === 'running').length;
    const queued = this.jobs().filter((j) => j.status === 'queued').length;
    return `${running} training, ${queued} queued. The queue runs one job at a time.`;
  });

  readonly tone = statusTone;
  readonly started = formatStarted;
  private readonly pollTimer: number;

  constructor() {
    this.api
      .listModels()
      .subscribe({ next: (items) => this.models.set(items), error: () => undefined });
    this.loadJobs();
    this.pollTimer = window.setInterval(() => this.loadJobs(), 3000);
  }

  loadJobs(): void {
    this.api.listJobs().subscribe({
      next: (resp) => {
        this.jobs.set(resp.items);
        this.jobsError.set(null);
        this.loaded.set(true);
      },
      error: (err) => {
        this.jobsError.set(describeHttpError(err, 'Failed to load runs'));
        this.loaded.set(true);
      }
    });
  }

  startNext(): void {
    this.startingQueue.set(true);
    this.act(this.api.startNextJob(), 'Failed to start the next job', () =>
      this.startingQueue.set(false)
    );
  }

  startJob(jobId: string): void {
    this.act(this.api.startJob(jobId), 'Failed to start the job');
  }

  modelLabel(id: string): string {
    return this.models().find((m) => m.id === id)?.label ?? id;
  }

  private act(
    call: ReturnType<TrainerApiService['startJob']>,
    fallback: string,
    done: () => void = () => undefined
  ): void {
    this.actionError.set(null);
    call.subscribe({
      next: () => {
        done();
        this.loadJobs();
      },
      error: (err) => {
        done();
        this.actionError.set(describeHttpError(err, fallback));
      }
    });
  }

  ngOnDestroy(): void {
    window.clearInterval(this.pollTimer);
  }
}
