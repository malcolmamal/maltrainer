/**
 * Keeps features on the design system (rules: frontend/AGENTS.md).
 *
 * - Colour values are written only in src/styles/_tokens.scss; everything else uses var(--mt-*).
 * - Fonts come from the tokens too (var(--mt-font-*)); no font-family literals.
 * - Outside src/app/ui, every button, input, select, textarea and table goes through the
 *   design system (uiButton / uiChoice, uiInput, uiTable), and no component re-declares the
 *   design system's own classes.
 */
import { readdirSync, readFileSync, statSync } from 'fs';
import { join, relative } from 'path';

const SRC = join(__dirname, '..');
const APP = join(SRC, 'app');
const UI_DIR = join(APP, 'ui');
const TOKENS = join(SRC, 'styles', '_tokens.scss');

function files(dir: string, ext: RegExp): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return files(path, ext);
    return ext.test(name) ? [path] : [];
  });
}

const rel = (path: string): string => relative(SRC, path).replace(/\\/g, '/');
const stripComments = (code: string): string =>
  code.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/.*$/gm, '$1');

const LITERAL = /`([\s\S]*?)`|'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)"/y;

/**
 * The text of every `key:` value in a .ts file: one string (backtick or quoted) or an array of
 * them, each occurrence in the file.
 */
export function inlineValues(source: string, key: string): string[] {
  const found: string[] = [];
  const starts = new RegExp(`\\b${key}\\s*:\\s*`, 'g');
  for (let m = starts.exec(source); m; m = starts.exec(source)) {
    let at = starts.lastIndex;
    const isArray = source[at] === '[';
    if (isArray) at++;
    for (;;) {
      while (/[\s,]/.test(source[at] ?? '')) at++;
      LITERAL.lastIndex = at;
      const lit = LITERAL.exec(source);
      if (!lit) break;
      found.push(lit[1] ?? lit[2] ?? lit[3]);
      at = LITERAL.lastIndex;
      if (!isArray) break;
    }
  }
  return found;
}

/** Styles: .scss files, and every inline `styles:` value of .ts components. */
function styleSources(): { file: string; css: string }[] {
  const scss = [...files(APP, /\.scss$/), ...files(join(SRC, 'styles'), /\.scss$/)]
    .filter((f) => f !== TOKENS)
    .map((file) => ({ file, css: readFileSync(file, 'utf8') }));
  const inline = files(APP, /(?<!\.spec)\.ts$/).flatMap((file) =>
    inlineValues(readFileSync(file, 'utf8'), 'styles').map((css) => ({ file, css }))
  );
  return [...scss, ...inline];
}

/** Templates outside the design system: .html files and every inline `template:` value. */
function featureTemplates(): { file: string; html: string }[] {
  const html = files(APP, /\.html$/)
    .filter((f) => !f.startsWith(UI_DIR))
    .map((file) => ({ file, html: readFileSync(file, 'utf8') }));
  const inline = files(APP, /(?<!\.spec)\.ts$/)
    .filter((f) => !f.startsWith(UI_DIR))
    .flatMap((file) =>
      inlineValues(readFileSync(file, 'utf8'), 'template').map((h) => ({ file, html: h }))
    );
  return [...html, ...inline];
}

const NAMED = 'white|black|red|green|blue|orange|yellow|purple|pink|gr[ae]y|cyan|magenta';
const COLOUR = new RegExp(
  `#[0-9a-f]{3,8}\\b|\\b(?:rgba?|hsla?|oklch|lab|lch)\\(|(?<![\\w-])(?:${NAMED})(?![\\w-])`,
  'i'
);
const FONT_FAMILY = /font-family\s*:(?!\s*var\(--mt-font-)/i;
const OWNED_CLASSES = /\.(?:ui-[a-z-]+|mt-(?:display|mono|num|muted|hint|visually-hidden))\b/;

/** Each opening tag of `tag` that lacks every one of `attrs`. */
function bareTags(html: string, tag: string, attrs: string[]): string[] {
  const tags = html.match(new RegExp(`<${tag}\\b[^>]*>`, 'gi')) ?? [];
  return tags.filter((t) => !attrs.some((a) => new RegExp(`\\s${a}(?:[\\s=>/]|$)`).test(t)));
}

describe('design-system guard', () => {
  it('finds the files it checks', () => {
    expect(styleSources().length).toBeGreaterThan(10);
    expect(featureTemplates().map((t) => rel(t.file))).toContain(
      'app/pages/trainer/trainer.page.html'
    );
  });

  it('writes colours only in the tokens file', () => {
    const offenders = styleSources().flatMap(({ file, css }) =>
      stripComments(css)
        .split('\n')
        .filter((line) => COLOUR.test(line))
        .map((line) => `${rel(file)}: ${line.trim()}`)
    );
    expect(offenders).toEqual([]);
  });

  it('takes fonts from the tokens', () => {
    const offenders = styleSources().flatMap(({ file, css }) =>
      stripComments(css)
        .split('\n')
        .filter((line) => FONT_FAMILY.test(line))
        .map((line) => `${rel(file)}: ${line.trim()}`)
    );
    expect(offenders).toEqual([]);
  });

  it('builds feature controls from the design system', () => {
    const offenders = featureTemplates().flatMap(({ file, html }) =>
      [
        ...bareTags(html, 'button', ['uiButton', 'uiChoice']),
        ...bareTags(html, 'input', ['uiInput']),
        ...bareTags(html, 'select', ['uiInput']),
        ...bareTags(html, 'textarea', ['uiInput']),
        ...bareTags(html, 'table', ['uiTable'])
      ].map((tag) => `${rel(file)}: ${tag.replace(/\s+/g, ' ')}`)
    );
    expect(offenders).toEqual([]);
  });

  it('leaves the design-system classes to the design system', () => {
    const offenders = styleSources()
      .filter(({ file }) => !file.startsWith(UI_DIR) && !rel(file).startsWith('styles/'))
      .flatMap(({ file, css }) =>
        stripComments(css)
          .split('\n')
          .filter((line) => OWNED_CLASSES.test(line))
          .map((line) => `${rel(file)}: ${line.trim()}`)
      );
    expect(offenders).toEqual([]);
  });

  it('catches what it is meant to catch', () => {
    expect(COLOUR.test('color: #fff;')).toBe(true);
    expect(COLOUR.test('background: rgba(0, 0, 0, 0.1);')).toBe(true);
    expect(COLOUR.test('color: white;')).toBe(true);
    expect(COLOUR.test('color: var(--mt-text);')).toBe(false);
    expect(COLOUR.test('white-space: nowrap;')).toBe(false);
    expect(COLOUR.test('color: red;')).toBe(true);
    expect(COLOUR.test('border: 1px solid Grey;')).toBe(true);
    expect(COLOUR.test('.redraw { gap: 2px }')).toBe(false);
    const ts = [
      'styles: [`a { color: var(--x) }`, \'.b[x="1"] { gap: 0 }\'],',
      'template: `<p>one</p>`,',
      "styles: '.c { color: red }',",
      'template: "<button>two</button>"'
    ].join('\n');
    expect(inlineValues(ts, 'styles')).toEqual([
      'a { color: var(--x) }',
      '.b[x="1"] { gap: 0 }',
      '.c { color: red }'
    ]);
    expect(inlineValues(ts, 'template')).toEqual(['<p>one</p>', '<button>two</button>']);
    expect(FONT_FAMILY.test('font-family: Arial;')).toBe(true);
    expect(FONT_FAMILY.test('font-family: var(--mt-font-mono);')).toBe(false);
    expect(bareTags('<button (click)="x()">', 'button', ['uiButton'])).toHaveLength(1);
    expect(bareTags('<button uiButton>', 'button', ['uiButton'])).toEqual([]);
    expect(bareTags('<button\n  uiButton\n  size="sm">', 'button', ['uiButton'])).toEqual([]);
    expect(bareTags('<input uiInputx>', 'input', ['uiInput'])).toHaveLength(1);
    expect(bareTags('<input uiInput="mono" />', 'input', ['uiInput'])).toEqual([]);
    expect(OWNED_CLASSES.test('.ui-button { color: red }')).toBe(true);
    expect(OWNED_CLASSES.test('.uiThing {')).toBe(false);
  });
});
