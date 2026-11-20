import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  signal,
  OnDestroy
} from '@angular/core';
import { ActivatedRoute, RouterLink } from '@angular/router';

import { describeHttpError } from '../../services/http-error';
import { JobStatusResponse, TrainerApiService } from '../../services/trainer-api.service';
import { Fact, LogLine, LossPoint, percentOf, UI } from '../../ui';
import { addLossPoint, formatDuration, parseLoss } from './job-stream';

/**
 * Progress messages closer together than this are the backend's buffer replayed on connect, not
 * live training: tqdm refreshes at most ten times a second.
 */
const REPLAY_BURST_MS = 20;

@Component({
  selector: 'app-job-page',
  standalone: true,
  imports: [RouterLink, ...UI],
  templateUrl: './job.page.html',
  styleUrl: './job.page.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class JobPage implements OnDestroy {
  private readonly api = inject(TrainerApiService);
  private readonly route = inject(ActivatedRoute);

  readonly job = signal<JobStatusResponse | null>(null);
  readonly status = signal('running');
  readonly progressStep = signal(0);
  readonly progressTotal = signal(0);
  readonly progressLine = signal<string | null>(null);
  readonly progressStream = signal<'stdout' | 'stderr' | null>(null);
  readonly logs = signal<LogLine[]>([]);
  readonly loss = signal<LossPoint[]>([]);
  readonly error = signal<string | null>(null);
  /**
   * Where speed is measured from, and the latest progress (time in ms, step). Measuring from a
   * live message leaves out model loading and the replayed history (see REPLAY_BURST_MS).
   */
  readonly firstProgress = signal<{ at: number; step: number } | null>(null);
  readonly lastProgress = signal<{ at: number; step: number } | null>(null);

  readonly progressPercent = computed(() =>
    Math.round(percentOf(this.progressStep(), this.progressTotal()))
  );

  /** Seconds per step; null until two progress messages are some steps apart. */
  readonly secPerStep = computed(() => {
    const first = this.firstProgress();
    const last = this.lastProgress();
    if (!first || !last || last.step <= first.step || last.at <= first.at) return null;
    return (last.at - first.at) / 1000 / (last.step - first.step);
  });

  readonly timeLeft = computed(() => {
    const rate = this.secPerStep();
    const left = this.progressTotal() - this.progressStep();
    if (rate === null || this.status() !== 'running' || left <= 0) return null;
    return formatDuration(rate * left);
  });

  readonly subtitle = computed(() => {
    const job = this.job();
    return job ? `${job.model} · ${job.optimizer} · ${job.duration}` : '';
  });

  readonly facts = computed<Fact[]>(() => {
    const job = this.job();
    if (!job) return [];
    return [
      { label: 'Model', value: job.model, mono: true },
      { label: 'Optimizer', value: job.optimizer, mono: true },
      { label: 'Duration', value: job.duration },
      { label: 'Output folder', value: job.output_path, mono: true },
      { label: 'Config', value: job.config_path, mono: true }
    ];
  });

  private socket: WebSocket | null = null;

  constructor() {
    const id = this.route.snapshot.paramMap.get('id');
    if (!id) {
      this.error.set('Missing job id');
      return;
    }

    this.api.getJob(id).subscribe({
      next: (job) => this.applyJob(job, 0, 0),
      error: (err) => this.error.set(describeHttpError(err, 'Failed to load job'))
    });

    this.connectSocket(id);
  }

  connectSocket(jobId: string) {
    const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${scheme}://${window.location.host}/ws/jobs/${jobId}`;
    const socket = new WebSocket(url);
    this.socket = socket;

    socket.onmessage = (event) => {
      try {
        this.handle(JSON.parse(event.data));
      } catch {
        // ignore parse errors
      }
    };

    socket.onerror = () => {
      this.error.set('WebSocket connection error');
    };
  }

  private handle(payload: Record<string, unknown>): void {
    if (payload['type'] === 'log') {
      this.handleLog(payload);
    } else if (payload['type'] === 'progress') {
      const step = (payload['step'] as number) ?? 0;
      this.progressStep.set(step);
      this.progressTotal.set((payload['total'] as number) ?? 0);
      const mark = { at: Date.now(), step };
      const previous = this.lastProgress();
      // On connect the backend replays its buffer, so old progress arrives in one burst; restart
      // the measurement at each message of a burst so only live intervals count.
      if (!previous || mark.at - previous.at < REPLAY_BURST_MS) this.firstProgress.set(mark);
      this.lastProgress.set(mark);
    } else if (payload['type'] === 'status') {
      this.status.set((payload['status'] as string) ?? this.status());
    } else if (payload['type'] === 'job') {
      this.applyJob(payload['job'] as JobStatusResponse, this.progressStep(), this.progressTotal());
    }
  }

  private handleLog(payload: Record<string, unknown>): void {
    const next: LogLine = {
      stream: payload['stream'] as 'stdout' | 'stderr',
      line: payload['line'] as string
    };
    const point = parseLoss(next.line, this.progressStep());
    if (point) this.loss.update((points) => addLossPoint(points, point));
    if (payload['replace'] && payload['is_progress']) {
      this.progressLine.set(next.line);
      this.progressStream.set(next.stream);
    } else {
      this.logs.update((items) => [...items, next]);
    }
  }

  private applyJob(job: JobStatusResponse, step: number, total: number): void {
    this.job.set(job);
    this.status.set(job.status ?? this.status());
    this.progressStep.set(job.progress?.step ?? step);
    this.progressTotal.set(job.progress?.total ?? total);
  }

  ngOnDestroy() {
    this.socket?.close();
  }
}
