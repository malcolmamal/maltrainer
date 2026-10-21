import { HttpErrorResponse } from '@angular/common/http';

import { describeHttpError } from './http-error';

function failure(status: number, body: unknown): HttpErrorResponse {
  return new HttpErrorResponse({ status, statusText: 'x', error: body });
}

describe('describeHttpError', () => {
  it('prefers a string detail from the backend', () => {
    expect(describeHttpError(failure(400, { detail: 'Output folder exists' }), 'Failed')).toBe(
      'Output folder exists'
    );
  });

  it('joins the msg of every validation entry and skips empty ones', () => {
    const body = { detail: [{ msg: 'bad lr', loc: ['body', 'lr'] }, { msg: '' }, { msg: 'bad' }] };
    expect(describeHttpError(failure(422, body), 'Failed')).toBe('bad lr; bad');
  });

  it('ignores entries that are not objects with a msg', () => {
    expect(describeHttpError(failure(422, { detail: [null, 'x', { msg: 'ok' }] }), 'F')).toBe('ok');
  });

  it.each([
    ['an empty detail list', { detail: [] }],
    ['an empty string detail', { detail: '' }],
    ['a detail of another type', { detail: 5 }],
    ['a list without messages', { detail: [{ loc: [] }] }],
    ['a plain text body', 'boom'],
    ['no body', null]
  ])('falls back to the generic text with the status for %s', (_name, body) => {
    expect(describeHttpError(failure(500, body), 'Failed to load jobs')).toBe(
      'Failed to load jobs (HTTP 500)'
    );
  });

  it('says the server is unreachable when there is no HTTP status', () => {
    expect(describeHttpError(failure(0, null), 'Failed to load jobs')).toBe(
      'Failed to load jobs (server unreachable)'
    );
  });

  it('uses only the fallback for unknown values', () => {
    expect(describeHttpError(undefined, 'Failed')).toBe('Failed');
    expect(describeHttpError(new Error('raw'), 'Failed')).toBe('Failed');
    expect(describeHttpError('text', 'Failed')).toBe('Failed');
  });

  it('never shows the raw Angular message', () => {
    expect(describeHttpError(failure(404, null), 'Failed')).not.toContain('Http failure');
  });
});
