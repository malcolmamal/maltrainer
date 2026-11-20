import { Directive, input } from '@angular/core';

/**
 * Theme a native text input, number input, select or textarea (styles: src/styles/_controls.scss).
 * `uiInput="mono"` sets the value in the data font, for numbers, paths and names.
 */
@Directive({
  selector: 'input[uiInput], select[uiInput], textarea[uiInput]',
  host: { class: 'ui-input', '[class.ui-input--mono]': "uiInput() === 'mono'" }
})
export class InputDirective {
  readonly uiInput = input<'' | 'mono'>('');
}

/** Theme a data table. Wrap it in `<div class="ui-table-box">` so wide tables scroll. */
@Directive({
  selector: 'table[uiTable]',
  host: { class: 'ui-table' }
})
export class TableDirective {}
