# Working on the maltrainer frontend

Angular 20, standalone components, signals, OnPush. Every screen is built from the design system in
`src/app/ui` and the tokens in `src/styles/_tokens.scss`. Read this file before you add or change a page.

## The rules (enforced by `src/app/design-system.guard.spec.ts`)

1. **Use the components.** A feature imports `UI` from `src/app/ui` (`imports: [...UI]`) and builds
   with the pieces below. If a piece is missing, add it to `src/app/ui` (see "Adding a component");
   never style a one-off look inside a page.
2. **No colour or font values outside the tokens.** Colours are written once, in `_tokens.scss`, for
   both themes. Everything else uses `var(--mt-*)`. No hex, `rgb()`, `hsl()`, `white`/`black`, and no
   `font-family` other than `var(--mt-font-*)`.
3. **Native controls go through the design system.** In pages and the shell: every `<button>` has
   `uiButton` or `uiChoice`; every `<input>`, `<select>` and `<textarea>` has `uiInput`; every `<table>`
   has `uiTable`.
4. **Pages own layout only.** Page `.scss` files arrange things (flex, grid, gaps, widths). They never
   restyle a design-system component or re-declare its classes (`.ui-*`, `.mt-display`, `.mt-mono`, …).
5. **Both themes, every time.** Dark is the default; light comes from the OS or the theme switch in
   the rail. Use tokens and both work. Check a new screen in both before calling it done.

## Catalogue

| Piece | Use it for | Example |
|---|---|---|
| `ui-page-header` | The h1 of every page, a one-line description, actions on the right | `<ui-page-header heading="Runs" description="…"><button uiButton>…</button></ui-page-header>` |
| `ui-panel` | A raised surface with a heading. Header buttons go in an element marked `panelActions`. Never nest panels. | `<ui-panel heading="Queue"><div panelActions>…</div>…</ui-panel>` |
| `button[uiButton]`, `a[uiButton]` | Every button and button-styled link. `variant`: `primary` (one per screen), `ghost` (default), `danger`. `size="sm"` in headers and rows. | `<button uiButton variant="primary">Add to queue</button>` |
| `button[uiChoice]` | Pick-one lists drawn as rows or cards (datasets, models). Wrap in `role="radiogroup"` with a label. `layout="card"` for grids. | `<button uiChoice [selected]="x" (click)="…">…</button>` |
| `ui-segmented` | 2–4 options in a small switch (steps/epochs, filters, theme). `detail` shows a count. | `<ui-segmented ariaLabel="Mode" [options]="opts" [(value)]="mode" />` |
| `ui-field` | A label around one control, with an optional hint | `<ui-field label="Optimizer" hint="…"><select uiInput>…</select></ui-field>` |
| `input/select/textarea[uiInput]` | Themed native controls. `uiInput="mono"` for numbers, names and paths. | `<input uiInput="mono" type="number" />` |
| `ui-switch` | An on/off setting; projected content sits beside it | `<ui-switch label="Dry run" [(checked)]="dryRun" />` |
| `ui-alert` | Messages. `tone`: `error` (default) and `warning` are announced; `info` is a note. Say what happened and what to do. | `<ui-alert tone="warning">…</ui-alert>` |
| `ui-status-pill` | A job status (`running`, `queued`, `exited`, `failed`) | `<ui-status-pill [status]="job.status" />` |
| `ui-progress-bar` | A thin bar in a list row. `tone` from `statusTone(status)`. | `<ui-progress-bar [value]="s" [max]="t" [tone]="tone(job.status)" />` |
| `ui-cook-timeline` | The signature element: a run's step axis with checkpoints, keep range, overcook zone and, when live, the current step | `<ui-cook-timeline [total]="2000" [saveEvery]="250" [current]="step" />` |
| `ui-fact-list` | Label/value pairs (plan numbers, settings). `mono: true` for paths. | `<ui-fact-list [facts]="facts" [columns]="2" />` |
| `ui-loss-chart` | Loss per step plus its moving average | `<ui-loss-chart [points]="loss()" [total]="total" />` |
| `ui-log-view` | Process output with a stderr filter, live progress line and follow-the-bottom | `<ui-log-view [lines]="logs()" [liveLine]="line()" />` |
| `table[uiTable]` | Data tables; wrap in `<div class="ui-table-box">` so wide tables scroll | `<div class="ui-table-box"><table uiTable>…</table></div>` |
| `ui-icon` | Stroke icons from the set in `icon.component.ts`. Never emoji or text glyphs. | `<ui-icon name="plus" [size]="18" />` |

Global helpers (from `src/styles/_base.scss`): `mt-display` (heading face), `mt-mono` (data only: names,
paths, numbers, logs; never as decoration), `mt-num` (tabular digits), `mt-muted`, `mt-hint` (quiet help
text under a control), `mt-visually-hidden` (text for screen readers only).

## Tokens

All in `src/styles/_tokens.scss`, defined for dark and light:

- Surfaces: `--mt-bg`, `--mt-rail`, `--mt-surface` (panels), `--mt-surface-2` (rows, inputs), `--mt-sunken`
  (logs, paths), `--mt-hover`, `--mt-line`, `--mt-line-strong`, `--mt-track`, `--mt-tick`.
- Text: `--mt-text`, `--mt-text-2`, `--mt-text-3` (all pass 4.5:1 on every surface).
- Meaning: `accent` = live / cooking / the main action; `good` = finished, the keep range; `warn`;
  `danger` = failed, overcooked. Each has `-text`, `-soft` (tinted background) and `-border`.
- Type: `--mt-font-ui`, `--mt-font-display`, `--mt-font-mono`. Space: `--mt-space-1…8` (4–40 px).
  Radius: `--mt-radius-sm/md/lg/xl/pill`. Controls: `--mt-control-h` (44 px touch target). Motion:
  `--mt-ease-out`.

A new colour is a new token pair (dark and light) in `_tokens.scss`, never a value in a component.

## Adding a component

1. Put it in `src/app/ui/<name>.component.ts`, selector prefix `ui` (`ui-thing` or `button[uiThing]`),
   OnPush, signal inputs. Styles only from tokens.
2. Export it from `src/app/ui/index.ts` and add it to `UI`.
3. Test it in `src/app/ui/*.spec.ts`: rendered output and states, not only that it exists.
4. Add a row to the catalogue above.

## Checks

From `frontend/`, before every commit (CI runs the same):

```bash
npm run lint            # eslint + prettier, TypeScript and templates
npm run test:coverage   # Jest, including the design-system guard; coverage floors may only go up
npm run build           # type-checks the templates
```
