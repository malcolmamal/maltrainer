import { ChangeDetectionStrategy, Component, inject, signal, OnDestroy } from '@angular/core';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { CommonModule } from '@angular/common';

import { describeHttpError } from '../../services/http-error';
import { JobStatusResponse, TrainerApiService } from '../../services/trainer-api.service';

interface LogMessage {
  stream: 'stdout' | 'stderr';
  line: string;
}

@Component({
  selector: 'app-job-page',
  standalone: true,
  imports: [CommonModule, RouterLink],
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
  readonly logs = signal<LogMessage[]>([]);
  readonly error = signal<string | null>(null);

  private socket: WebSocket | null = null;

  constructor() {
    const id = this.route.snapshot.paramMap.get('id');
    if (!id) {
      this.error.set('Missing job id');
      return;
    }

    this.api.getJob(id).subscribe({
      next: (job) => {
        this.job.set(job);
        this.status.set(job.status);
        this.progressStep.set(job.progress?.step ?? 0);
        this.progressTotal.set(job.progress?.total ?? 0);
      },
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
        const payload = JSON.parse(event.data);
        if (payload.type === 'log') {
          const next = {
            stream: payload.stream as 'stdout' | 'stderr',
            line: payload.line as string
          };
          if (payload.replace && payload.is_progress) {
            this.progressLine.set(next.line);
            this.progressStream.set(next.stream);
          } else {
            this.logs.update((items) => [...items, next]);
          }
        } else if (payload.type === 'progress') {
          this.progressStep.set(payload.step ?? 0);
          this.progressTotal.set(payload.total ?? 0);
        } else if (payload.type === 'status') {
          this.status.set(payload.status ?? this.status());
        } else if (payload.type === 'job') {
          const job = payload.job as JobStatusResponse;
          this.job.set(job);
          this.status.set(job.status ?? this.status());
          this.progressStep.set(job.progress?.step ?? this.progressStep());
          this.progressTotal.set(job.progress?.total ?? this.progressTotal());
        }
      } catch {
        // ignore parse errors
      }
    };

    socket.onerror = () => {
      this.error.set('WebSocket connection error');
    };
  }

  progressPercent(): number {
    const total = this.progressTotal();
    if (!total) return 0;
    return Math.min(100, Math.round((this.progressStep() / total) * 100));
  }

  ngOnDestroy() {
    this.socket?.close();
  }
}
