import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import {
  DatasetInfo,
  JobStatusResponse,
  StartTrainRequest,
  TrainerApiService
} from '../../services/trainer-api.service';

@Component({
  selector: 'app-trainer-page',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './trainer.page.html',
  styleUrl: './trainer.page.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class TrainerPage {
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

  private pollTimer: number | null = null;

  readonly selectedInfo = computed(() =>
    this.datasets().find((d) => d.name === this.selectedDataset()) ?? null
  );

  readonly outputPath = computed(() => {
    const name = this.selectedDataset();
    if (!name) return '';
    return `output/${name}`;
  });

  readonly hasRunning = computed(() =>
    this.jobs().some((job) => job.status === 'running')
  );

  constructor() {
    this.loadDatasets();
    this.loadJobs();
    this.pollTimer = window.setInterval(() => this.loadJobs(), 3000);
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

  startTraining() {
    const dataset = this.selectedDataset();
    if (!dataset) {
      this.error.set('Select a dataset to start training.');
      return;
    }
    this.starting.set(true);
    this.error.set(null);

    const body: StartTrainRequest = {
      dataset,
      dry_run: this.dryRun(),
      dry_run_step_time: this.dryRunStepTime()
    };

    this.api.startTraining(body).subscribe({
      next: (resp) => {
        this.starting.set(false);
        this.loadJobs();
      },
      error: (err) => {
        this.error.set(err?.error?.detail ?? err?.message ?? 'Failed to start training');
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

  ngOnDestroy() {
    if (this.pollTimer) {
      window.clearInterval(this.pollTimer);
    }
  }
}
