import { TestBed } from '@angular/core/testing';
import { provideRouter, Router } from '@angular/router';
import { Component } from '@angular/core';

import { App } from './app';
import { routes } from './app.routes';
import { appConfig } from './app.config';
import { ThemeService } from './services/theme.service';

@Component({ template: '<p class="stub">stub</p>' })
class StubPage {}

describe('App shell', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [
        provideRouter([
          { path: 'trainer', component: StubPage },
          { path: 'runs', component: StubPage }
        ])
      ]
    }).compileComponents();
  });

  afterEach(() => {
    localStorage.clear();
    delete document.documentElement.dataset['theme'];
  });

  it('renders the brand and the main navigation', () => {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    const links = Array.from(el.querySelectorAll('.navItem'));

    expect(el.querySelector('.brand')?.textContent).toContain('maltrainer');
    expect(el.querySelector('.brand')?.getAttribute('href')).toBe('/trainer');
    expect(links.map((a) => a.textContent?.trim())).toEqual(['New run', 'Runs']);
    expect(links.map((a) => a.getAttribute('href'))).toEqual(['/trainer', '/runs']);
  });

  it('renders the routed page and marks the active nav link', async () => {
    const fixture = TestBed.createComponent(App);
    await TestBed.inject(Router).navigateByUrl('/runs');
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    const links = Array.from(el.querySelectorAll('.navItem'));

    expect(el.querySelector('main .stub')).not.toBeNull();
    expect(links.map((a) => a.classList.contains('active'))).toEqual([false, true]);
    expect(links[1].getAttribute('aria-current')).toBe('page');
  });

  it('switches the theme from the rail', () => {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    const buttons = Array.from(el.querySelectorAll<HTMLButtonElement>('.railFoot button'));

    expect(buttons.map((b) => b.textContent?.trim())).toEqual(['Auto', 'Light', 'Dark']);
    buttons[1].click();
    fixture.detectChanges();
    TestBed.tick();
    expect(TestBed.inject(ThemeService).preference()).toBe('light');
    expect(document.documentElement.dataset['theme']).toBe('light');
    expect(buttons[1].getAttribute('aria-checked')).toBe('true');
  });
});

describe('routes and app config', () => {
  it('redirects the empty path to the trainer', () => {
    expect(routes[0]).toMatchObject({ path: '', redirectTo: 'trainer', pathMatch: 'full' });
  });

  it('lazy loads the trainer, runs and job pages', async () => {
    const lazy = routes.filter((r) => r.loadComponent);
    const loaded = await Promise.all(
      lazy.map((r) => (r.loadComponent as () => Promise<unknown>)())
    );
    expect(lazy.map((r) => r.path)).toEqual(['trainer', 'runs', 'jobs/:id']);
    expect(loaded.map((c) => (c as { name: string }).name)).toEqual([
      'TrainerPage',
      'RunsPage',
      'JobPage'
    ]);
  });

  it('provides application providers', () => {
    expect(appConfig.providers.length).toBeGreaterThan(0);
  });
});
