import { HttpErrorResponse } from '@angular/common/http';

/** FastAPI validation entries look like `{ msg, loc, type }`; only `msg` is shown. */
function detailMessage(detail: unknown): string {
  if (typeof detail === 'string') return detail;
  if (!Array.isArray(detail)) return '';
  return detail
    .map((d: unknown) => (d as { msg?: unknown } | null)?.msg)
    .filter((msg): msg is string => typeof msg === 'string' && msg !== '')
    .join('; ');
}

/**
 * Turn a failed HTTP call into text for the user.
 *
 * Prefers the backend's `detail` (a string, or the FastAPI validation list). Otherwise returns
 * `fallback` plus the HTTP status, or "server unreachable" for status 0. Anything that is not an
 * HttpErrorResponse yields `fallback` alone. Angular's raw `message` is never shown.
 */
export function describeHttpError(err: unknown, fallback: string): string {
  if (!(err instanceof HttpErrorResponse)) return fallback;
  const message = detailMessage((err.error as { detail?: unknown } | null)?.detail);
  if (message) return message;
  return err.status > 0 ? `${fallback} (HTTP ${err.status})` : `${fallback} (server unreachable)`;
}
