import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'trainer' },
  {
    path: 'trainer',
    title: 'New run · maltrainer',
    loadComponent: () => import('./pages/trainer/trainer.page').then((m) => m.TrainerPage)
  },
  {
    path: 'runs',
    title: 'Runs · maltrainer',
    loadComponent: () => import('./pages/runs/runs.page').then((m) => m.RunsPage)
  },
  {
    path: 'jobs/:id',
    title: 'Run · maltrainer',
    loadComponent: () => import('./pages/job/job.page').then((m) => m.JobPage)
  }
];
