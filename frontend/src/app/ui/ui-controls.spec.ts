import { Component, signal } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SegmentOption, UI } from '.';

@Component({
  imports: [...UI],
  template: `
    <button uiButton id="ghost">Ghost</button>
    <button uiButton variant="primary" id="primary">Go</button>
    <a uiButton variant="danger" size="sm" id="danger" href="#x">Stop</a>
    <ui-panel heading="Queue"
      ><span panelActions id="act">A</span>
      <p id="body">B</p></ui-panel
    >
    <ui-panel heading="Bare" id="bare" />
    <ui-page-header heading="Runs" description="All runs" id="hdr">
      <button uiButton>Act</button>
    </ui-page-header>
    <ui-page-header heading="cats" [monoHeading]="true" id="mono" />
    <ui-field label="Steps" hint="Leave empty for auto"
      ><input uiInput="mono" id="steps"
    /></ui-field>
    <ui-field label="Name" id="nohint"><select uiInput id="sel"></select></ui-field>
    <button uiChoice [selected]="picked() === 'a'" (click)="picked.set('a')" id="ca">A</button>
    <button uiChoice layout="card" [selected]="picked() === 'b'" id="cb">B</button>
    <ui-segmented ariaLabel="Mode" [options]="options" [(value)]="mode" />
    <ui-switch label="Dry run" [(checked)]="dry"><span id="extra">x</span></ui-switch>
    <ui-alert id="err">Broken</ui-alert>
    <ui-alert tone="warning" id="warn">Careful</ui-alert>
    <ui-alert tone="info" id="info">Note</ui-alert>
    <table uiTable id="tbl"></table>
    <ui-icon name="plus" [size]="12" id="icon" />
  `
})
class Host {
  readonly picked = signal('a');
  readonly mode = signal<'steps' | 'epochs'>('steps');
  readonly dry = signal(false);
  readonly options: SegmentOption<'steps' | 'epochs'>[] = [
    { value: 'steps', label: 'Steps', detail: 3 },
    { value: 'epochs', label: 'Epochs' }
  ];
}

describe('design system controls', () => {
  let fixture: ComponentFixture<Host>;
  let host: Host;
  const q = <T extends Element = HTMLElement>(sel: string): T =>
    fixture.nativeElement.querySelector(sel) as T;

  beforeEach(() => {
    fixture = TestBed.createComponent(Host);
    host = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('gives buttons and links one look per variant and size', () => {
    expect(q('#ghost').className).toContain('ui-button--ghost');
    expect(q('#ghost').className).not.toContain('ui-button--sm');
    expect(q('#primary').className).toContain('ui-button--primary');
    expect(q('#danger').className).toContain('ui-button--danger');
    expect(q('#danger').className).toContain('ui-button--sm');
  });

  it('labels a panel region by its heading and projects actions apart from content', () => {
    const panel = q('ui-panel');
    const h2 = panel.querySelector('h2') as HTMLElement;
    expect(h2.textContent).toBe('Queue');
    expect(panel.getAttribute('role')).toBe('region');
    expect(panel.getAttribute('aria-labelledby')).toBe(h2.id);
    expect(q('ui-panel .actions #act')).not.toBeNull();
    expect(q('ui-panel > #body')).not.toBeNull();
    expect(q('#bare h2').id).not.toBe(h2.id);
  });

  it('renders the page header with an optional description and data-font heading', () => {
    expect(q('#hdr h1').textContent).toBe('Runs');
    expect(q('#hdr .description').textContent).toBe('All runs');
    expect(q('#hdr .actions button')).not.toBeNull();
    expect(q('#mono .description')).toBeNull();
    expect(q('#mono h1').classList).toContain('mt-mono');
    expect(q('#hdr h1').classList).not.toContain('mt-mono');
  });

  it('wraps the control in its label and shows the hint only when given', () => {
    const label = q('ui-field label.field');
    expect(label.querySelector('.label')?.textContent).toBe('Steps');
    expect(label.querySelector('#steps')).not.toBeNull();
    expect(q('ui-field .mt-hint').textContent).toBe('Leave empty for auto');
    expect(q('#nohint .mt-hint')).toBeNull();
  });

  it('themes native inputs, selects and tables through classes', () => {
    expect(q('#steps').className).toBe('ui-input ui-input--mono');
    expect(q('#sel').className).toBe('ui-input');
    expect(q('#tbl').className).toBe('ui-table');
  });

  it('marks the selected choice for assistive tech and as a card when asked', () => {
    expect(q('#ca').getAttribute('aria-checked')).toBe('true');
    expect(q('#ca').getAttribute('role')).toBe('radio');
    expect(q('#ca').getAttribute('type')).toBe('button');
    expect(q('#cb').getAttribute('aria-checked')).toBe('false');
    expect(q('#cb').classList).toContain('ui-choice--card');
    host.picked.set('b');
    fixture.detectChanges();
    expect(q('#cb').classList).toContain('ui-choice--on');
    expect(q('#ca').classList).not.toContain('ui-choice--on');
  });

  it('switches the segmented value on click and shows details', () => {
    const seg = q('ui-segmented');
    const buttons = Array.from(seg.querySelectorAll('button'));
    expect(seg.getAttribute('aria-label')).toBe('Mode');
    expect(buttons.map((b) => b.getAttribute('aria-checked'))).toEqual(['true', 'false']);
    expect(buttons[0].querySelector('.detail')?.textContent).toBe('3');
    expect(buttons[1].querySelector('.detail')).toBeNull();
    buttons[1].click();
    fixture.detectChanges();
    expect(host.mode()).toBe('epochs');
    expect(buttons[1].classList).toContain('on');
  });

  it('flips the switch model from the checkbox and back', () => {
    const box = q<HTMLInputElement>('ui-switch input');
    expect(box.getAttribute('role')).toBe('switch');
    expect(q('ui-switch .label').textContent).toBe('Dry run');
    expect(q('ui-switch #extra')).not.toBeNull();
    box.checked = true;
    box.dispatchEvent(new Event('change'));
    fixture.detectChanges();
    expect(host.dry()).toBe(true);
    host.dry.set(false);
    fixture.detectChanges();
    expect(box.checked).toBe(false);
  });

  it('announces errors and warnings but not info notes, each with its icon', () => {
    expect(q('#err').getAttribute('role')).toBe('alert');
    expect(q('#err').className).toContain('ui-alert--error');
    expect(q('#warn').getAttribute('role')).toBe('alert');
    expect(q('#info').getAttribute('role')).toBe('status');
    expect(q('#err path').getAttribute('d')).toContain('M12 8v5');
    expect(q('#warn path').getAttribute('d')).toContain('M12 4l9 16');
    expect(q('#info path').getAttribute('d')).toContain('M12 11v5');
    expect(q('#info .body').textContent).toBe('Note');
  });

  it('draws icons at the requested size, hidden from assistive tech', () => {
    expect(q('#icon').getAttribute('aria-hidden')).toBe('true');
    expect(q('#icon svg').getAttribute('width')).toBe('12');
    expect(q('#icon path').getAttribute('d')).toBe('M12 5v14M5 12h14');
  });
});
