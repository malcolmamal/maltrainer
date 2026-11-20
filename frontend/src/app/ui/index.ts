// The maltrainer design system. Pages import from here, never from the files directly,
// and never restyle these components from outside. Catalogue and rules: frontend/AGENTS.md.
import { AlertComponent } from './alert.component';
import { ButtonComponent } from './button.component';
import { ChoiceComponent } from './choice.component';
import { CookTimelineComponent } from './cook-timeline.component';
import { FactListComponent } from './fact-list.component';
import { FieldComponent } from './field.component';
import { IconComponent } from './icon.component';
import { LogViewComponent } from './log-view.component';
import { LossChartComponent } from './loss-chart.component';
import { InputDirective, TableDirective } from './native.directives';
import { PageHeaderComponent } from './page-header.component';
import { PanelComponent } from './panel.component';
import { ProgressBarComponent } from './progress-bar.component';
import { SegmentedComponent } from './segmented.component';
import { StatusPillComponent } from './status-pill.component';
import { SwitchComponent } from './switch.component';

export * from './alert.component';
export * from './button.component';
export * from './choice.component';
export * from './cook-timeline.component';
export * from './fact-list.component';
export * from './field.component';
export * from './icon.component';
export * from './log-view.component';
export * from './loss-chart.component';
export * from './native.directives';
export * from './page-header.component';
export * from './panel.component';
export * from './progress-bar.component';
export * from './segmented.component';
export * from './status-pill.component';
export * from './switch.component';

/** Add to a standalone component's `imports` to get every design-system piece. */
export const UI = [
  AlertComponent,
  ButtonComponent,
  ChoiceComponent,
  CookTimelineComponent,
  FactListComponent,
  FieldComponent,
  IconComponent,
  InputDirective,
  LogViewComponent,
  LossChartComponent,
  PageHeaderComponent,
  PanelComponent,
  ProgressBarComponent,
  SegmentedComponent,
  StatusPillComponent,
  SwitchComponent,
  TableDirective
] as const;
