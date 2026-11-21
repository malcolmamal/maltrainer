import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

import { ThemePreference, ThemeService } from './services/theme.service';
import { IconComponent, IconName, SegmentedComponent, SegmentOption } from './ui';

interface NavItem {
  path: string;
  label: string;
  icon: IconName;
}

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive, IconComponent, SegmentedComponent],
  templateUrl: './app.html',
  styleUrl: './app.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class App {
  readonly theme = inject(ThemeService);

  readonly nav: NavItem[] = [
    { path: '/trainer', label: 'New run', icon: 'plus' },
    { path: '/runs', label: 'Runs', icon: 'list' }
  ];

  readonly themeOptions: SegmentOption<ThemePreference>[] = [
    { value: 'system', label: 'Auto' },
    { value: 'light', label: 'Light' },
    { value: 'dark', label: 'Dark' }
  ];
}
