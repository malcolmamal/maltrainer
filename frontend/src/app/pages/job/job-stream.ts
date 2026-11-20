import { LossPoint } from '../../ui';

const LOSS = /\bloss[=:]\s*(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)/i;
const STEP = /\bstep[=:]\s*(\d+)/i;
/** tqdm's `12/40` counter, read like the backend's PROGRESS_RE (backend/streams.py). */
const COUNTER = /(?<![\d/.])(\d+)\s*\/\s*(\d+)(?![\d/.])/;

/**
 * Read the loss from a trainer's tqdm line (`12/40 [… loss=0.0794 …]`). The step is the line's
 * `step=`, else its `N/M` counter, else `fallbackStep`. The backend sends the line before the
 * progress message parsed from it, so the last progress step would be one behind.
 */
export function parseLoss(line: string, fallbackStep: number): LossPoint | null {
  const loss = LOSS.exec(line);
  if (!loss) return null;
  const value = Number(loss[1]);
  if (!Number.isFinite(value)) return null;
  const step = STEP.exec(line) ?? COUNTER.exec(line);
  return { step: step ? Number(step[1]) : fallbackStep, loss: value };
}

/** Append a point, replacing the last one when it is for the same step. */
export function addLossPoint(points: LossPoint[], point: LossPoint): LossPoint[] {
  const last = points[points.length - 1];
  if (last && last.step === point.step) return [...points.slice(0, -1), point];
  if (last && point.step < last.step) return points;
  return [...points, point];
}

/** "33 min", "2 h 05 min", "under a minute". */
export function formatDuration(seconds: number): string {
  if (seconds < 60) return 'under a minute';
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min`;
  return `${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, '0')} min`;
}
