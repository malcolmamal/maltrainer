import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'trainer' },
  {
    path: 'trainer',
    loadComponent: () => import('./pages/trainer/trainer.page').then((m) => m.TrainerPage)
  },
  {
    path: 'jobs/:id',
    loadComponent: () => import('./pages/job/job.page').then((m) => m.JobPage)
  }
];
