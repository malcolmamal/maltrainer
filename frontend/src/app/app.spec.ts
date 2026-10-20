import { TestBed } from '@angular/core/testing';
import { provideRouter, Router } from '@angular/router';
import { Component } from '@angular/core';

import { App } from './app';
import { routes } from './app.routes';
import { appConfig } from './app.config';

@Component({ template: '<p class="stub">stub</p>' })
class StubPage {}

describe('App shell', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [provideRouter([{ path: 'trainer', component: StubPage }])]
    }).compileComponents();
  });

  it('renders the brand and nav links to the trainer', () => {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;

    expect(el.querySelector('.brand')?.textContent).toContain('Maltrainer');
    expect(el.querySelector('.brand')?.getAttribute('href')).toBe('/trainer');
    expect(el.querySelector('.nav a')?.textContent?.trim()).toBe('Trainer');
  });

  it('renders the routed page and marks the active nav link', async () => {
    const fixture = TestBed.createComponent(App);
    await TestBed.inject(Router).navigateByUrl('/trainer');
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;

    expect(el.querySelector('main .stub')).not.toBeNull();
    expect(el.querySelector('.nav a')?.classList.contains('active')).toBe(true);
  });
});

describe('routes and app config', () => {
  it('redirects the empty path to the trainer', () => {
    expect(routes[0]).toMatchObject({ path: '', redirectTo: 'trainer', pathMatch: 'full' });
  });

  it('lazy loads the trainer and job pages', async () => {
    const loaded = await Promise.all(
      routes
        .filter((r) => r.loadComponent)
        .map((r) => (r.loadComponent as () => Promise<unknown>)())
    );
    expect(routes.filter((r) => r.loadComponent).map((r) => r.path)).toEqual([
      'trainer',
      'jobs/:id'
    ]);
    expect(loaded.map((c) => (c as { name: string }).name)).toEqual(['TrainerPage', 'JobPage']);
  });

  it('provides application providers', () => {
    expect(appConfig.providers.length).toBeGreaterThan(0);
  });
});
