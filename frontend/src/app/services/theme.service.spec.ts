import { TestBed } from '@angular/core/testing';

import { THEME_KEY, ThemeService } from './theme.service';

describe('ThemeService', () => {
  const root = document.documentElement;

  function create(): ThemeService {
    const service = TestBed.inject(ThemeService);
    TestBed.tick();
    return service;
  }

  afterEach(() => {
    localStorage.clear();
    delete root.dataset['theme'];
    jest.restoreAllMocks();
  });

  it('follows the system theme when nothing is saved', () => {
    const theme = create();
    expect(theme.preference()).toBe('system');
    expect(root.dataset['theme']).toBeUndefined();
  });

  it('restores a saved theme and applies it to <html>', () => {
    localStorage.setItem(THEME_KEY, 'light');
    const theme = create();
    expect(theme.preference()).toBe('light');
    expect(root.dataset['theme']).toBe('light');
  });

  it('ignores a saved value that is not a theme', () => {
    localStorage.setItem(THEME_KEY, 'neon');
    expect(create().preference()).toBe('system');
  });

  it('saves a forced theme and forgets it when going back to system', () => {
    const theme = create();
    theme.preference.set('dark');
    TestBed.tick();
    expect(root.dataset['theme']).toBe('dark');
    expect(localStorage.getItem(THEME_KEY)).toBe('dark');

    theme.preference.set('system');
    TestBed.tick();
    expect(root.dataset['theme']).toBeUndefined();
    expect(localStorage.getItem(THEME_KEY)).toBeNull();
  });

  it('still themes the page when storage refuses access', () => {
    jest.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('denied');
    });
    jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('denied');
    });
    const theme = create();
    expect(theme.preference()).toBe('system');
    theme.preference.set('light');
    expect(() => TestBed.tick()).not.toThrow();
    expect(root.dataset['theme']).toBe('light');
  });
});
