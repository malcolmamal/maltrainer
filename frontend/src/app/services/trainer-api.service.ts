import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';

export interface DatasetInfo {
  name: string;
  path: string;
  image_count: number;
}

export interface StartTrainRequest {
  dataset: string;
  dry_run: boolean;
  dry_run_step_time: number;
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
