import { DOCUMENT } from '@angular/common';
import { effect, inject, Injectable, signal } from '@angular/core';

export type ThemePreference = 'system' | 'light' | 'dark';

/** Storage key; index.html reads it too, to apply the theme before the first paint. */
export const THEME_KEY = 'mt-theme';

function readSaved(storage: Storage | undefined): ThemePreference {
  try {
    const saved = storage?.getItem(THEME_KEY);
    return saved === 'light' || saved === 'dark' ? saved : 'system';
  } catch {
    return 'system';
  }
}

/**
 * Light, dark, or whatever the OS says. "system" removes `data-theme`, so the
 * prefers-color-scheme media query in src/styles/_tokens.scss decides.
 */
@Injectable({ providedIn: 'root' })
export class ThemeService {
  private readonly document = inject(DOCUMENT);
  private readonly storage = this.document.defaultView?.localStorage;
  readonly preference = signal<ThemePreference>(readSaved(this.storage));

  constructor() {
    effect(() => {
      const pref = this.preference();
      const root = this.document.documentElement;
      if (pref === 'system') delete root.dataset['theme'];
      else root.dataset['theme'] = pref;
      try {
        if (pref === 'system') this.storage?.removeItem(THEME_KEY);
        else this.storage?.setItem(THEME_KEY, pref);
      } catch {
        // Private windows may refuse storage; the theme still applies for this visit.
      }
    });
  }
}
