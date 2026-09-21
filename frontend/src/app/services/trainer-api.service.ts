import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';

export interface DatasetInfo {
  name: string;
  path: string;
  image_count: number;
}

export type DurationMode = 'epochs' | 'steps';

export interface OptimizerInfo {
  name: string;
  label: string;
  default_lr: number;
  note: string;
}

export interface ModelInfo {
  id: string;
  label: string;
  optimizers: OptimizerInfo[];
  default_optimizer: string;
  default_duration_mode: DurationMode;
  default_epochs: number;
  default_max_steps: number;
  /** False when the trainer only runs whole epochs; a step budget is then rounded up. */
  native_max_steps: boolean;
  sec_per_step: number;
  models_root: string | null;
  missing_files: string[];
  /** True when leaving Max steps empty picks a step budget from the dataset size. */
  auto_steps: boolean;
  target_epochs: number;
  min_auto_steps: number;
  max_auto_steps: number;
  resolution_note: string;
}

export interface StartTrainRequest {
  dataset: string;
  model: string;
  optimizer?: string;
  learning_rate?: number;
  duration_mode: DurationMode;
  epochs?: number;
  max_steps?: number;
  save_every?: number;
  trigger?: string;
  dry_run: boolean;
  dry_run_step_time: number;
}

export interface TrainPlan {
  model: string;
  dataset: string;
  optimizer: string;
  learning_rate: number;
  duration_mode: DurationMode;
  image_count: number;
  steps_per_epoch: number;
  epochs: number;
  total_steps: number;
  save_every: number;
  output_path: string;
  duration: string;
  auto_steps: boolean;
  estimated_minutes: number | null;
  notes: string[];
  warnings: string[];
}

export interface StartTrainResponse {
  job_id: string;
  dataset: string;
  output_path: string;
  config_path: string;
  command: string[];
}

export interface JobStatusResponse {
  job_id: string;
  dataset: string;
  output_path: string;
  config_path: string;
  status: string;
  started_at: number;
  ended_at: number | null;
  exit_code: number | null;
  progress: { step: number; total: number };
  model: string;
  optimizer: string;
  duration: string;
}

export interface JobListResponse {
  items: JobStatusResponse[];
}

@Injectable({ providedIn: 'root' })
export class TrainerApiService {
  private readonly http = inject(HttpClient);

  listDatasets(): Observable<DatasetInfo[]> {
    return this.http.get<DatasetInfo[]>('/api/datasets');
  }

  listModels(): Observable<ModelInfo[]> {
    return this.http.get<ModelInfo[]>('/api/models');
  }

  planTraining(body: StartTrainRequest): Observable<TrainPlan> {
    return this.http.post<TrainPlan>('/api/plan', body);
  }

  startTraining(body: StartTrainRequest): Observable<StartTrainResponse> {
    return this.http.post<StartTrainResponse>('/api/train', body);
  }

  getJob(jobId: string): Observable<JobStatusResponse> {
    return this.http.get<JobStatusResponse>(`/api/jobs/${jobId}`);
  }

  listJobs(): Observable<JobListResponse> {
    return this.http.get<JobListResponse>('/api/jobs');
  }

  startNextJob(): Observable<{ status: string }> {
    return this.http.post<{ status: string }>('/api/jobs/start-next', {});
  }

  startJob(jobId: string): Observable<{ status: string }> {
    return this.http.post<{ status: string }>(`/api/jobs/${jobId}/start`, {});
  }
}
