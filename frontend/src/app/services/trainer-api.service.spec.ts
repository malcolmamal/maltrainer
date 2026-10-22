import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';

import { StartTrainRequest, TrainerApiService } from './trainer-api.service';

const request: StartTrainRequest = {
  dataset: 'cats',
  model: 'qwen_image_2_1',
  duration_mode: 'steps',
  max_steps: 100,
  dry_run: true,
  dry_run_step_time: 0.01
};

describe('TrainerApiService', () => {
  let api: TrainerApiService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()]
    });
    api = TestBed.inject(TrainerApiService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('lists datasets with GET /api/datasets', () => {
    const datasets = [{ name: 'cats', path: '/d/cats', image_count: 3 }];
    let result: unknown;
    api.listDatasets().subscribe((value) => (result = value));

    const req = http.expectOne('/api/datasets');
    expect(req.request.method).toBe('GET');
    req.flush(datasets);
    expect(result).toEqual(datasets);
  });

  it('lists models with GET /api/models', () => {
    api.listModels().subscribe();
    const req = http.expectOne('/api/models');
    expect(req.request.method).toBe('GET');
    req.flush([]);
  });

  it('plans with POST /api/plan and the request body', () => {
    api.planTraining(request).subscribe();
    const req = http.expectOne('/api/plan');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual(request);
    req.flush({});
  });

  it('starts training with POST /api/train and the request body', () => {
    api.startTraining(request).subscribe();
    const req = http.expectOne('/api/train');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual(request);
    req.flush({});
  });

  it('reads one job with GET /api/jobs/:id', () => {
    api.getJob('abc').subscribe();
    const req = http.expectOne('/api/jobs/abc');
    expect(req.request.method).toBe('GET');
    req.flush({});
  });

  it('lists jobs with GET /api/jobs', () => {
    api.listJobs().subscribe();
    const req = http.expectOne('/api/jobs');
    expect(req.request.method).toBe('GET');
    req.flush({ items: [] });
  });

  it('starts the next queued job with an empty POST', () => {
    api.startNextJob().subscribe();
    const req = http.expectOne('/api/jobs/start-next');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({});
    req.flush({ status: 'started' });
  });

  it('starts a given job with an empty POST', () => {
    api.startJob('abc').subscribe();
    const req = http.expectOne('/api/jobs/abc/start');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({});
    req.flush({ status: 'started' });
  });

  it('surfaces HTTP errors to the subscriber', () => {
    let status = 0;
    api.getJob('missing').subscribe({ error: (err: { status: number }) => (status = err.status) });
    http.expectOne('/api/jobs/missing').flush('nope', { status: 404, statusText: 'Not Found' });
    expect(status).toBe(404);
  });
});
