import type { IterationSnapshot, ResultsSummary } from '@/api/types.generated';

/**
 * The scale every per-case mark and cell is drawn against. Two parts, both fixed for
 * the life of a branch:
 *
 * - Colour classes at absolute ratio thresholds. The same speedup gets the same colour
 *   in every iteration, every branch and both views — a value's class never depends on
 *   what any other case did. A continuous ramp scaled to the branch's own extremes was
 *   tried first: one 0.09x outlier stretched it to 1/16–16x and left every real value
 *   (0.3–2.4x) a 30% tint of grey.
 * - A log span for the per-iteration axis, computed once from the whole history so a
 *   mark at the same speedup lands in the same place every time. With a per-iteration
 *   span, rail4284 went 0.50x -> 0.51x and its mark moved a third of the track because
 *   a different case had moved.
 */

export const BASELINE = 1;

/** Class boundaries. 0.9x–1.1x is "about the same as the reference". */
export const CLASS_BOUNDS = [1 / 4, 1 / 2, 0.9, 1.1, 2, 4] as const;

/** -3 (worst) .. 0 (≈1x) .. +3 (best). */
export type SpeedupClass = -3 | -2 | -1 | 0 | 1 | 2 | 3;

export const CLASSES: SpeedupClass[] = [-3, -2, -1, 0, 1, 2, 3];

export function speedupClass(speedup: number): SpeedupClass {
    let i = 0;
    while (i < CLASS_BOUNDS.length && speedup >= CLASS_BOUNDS[i]) i++;
    return (i - 3) as SpeedupClass;
}

/**
 * Orange slower, blue faster, a neutral grey for "about the same": a diverging pair
 * with a neutral midpoint, brighter with magnitude because the surface is dark.
 * Validated on the card surface (#171717) with the dataviz palette checks — each arm
 * is a monotone one-hue ramp, and the arms sit ΔE 20 apart under protanopia. The
 * obvious green/orange "good/bad" pair measured ΔE 3.7 there and was dropped.
 */
const CLASS_COLORS = [
    'oklch(0.80 0.15 55)',
    'oklch(0.67 0.15 55)',
    'oklch(0.55 0.12 55)',
    'oklch(0.38 0 0)',
    'oklch(0.55 0.11 250)',
    'oklch(0.67 0.13 250)',
    'oklch(0.80 0.13 250)',
];

export function classColor(cls: SpeedupClass): string {
    return CLASS_COLORS[cls + 3];
}

/** Boundary labels for the legend, in the same vocabulary as the axis ticks. */
export function boundLabel(v: number): string {
    return v < 1 && Number.isInteger(1 / v) ? `1/${1 / v}` : String(v);
}

/**
 * What became of a case this iteration. `skipped` is a null summary — the case never
 * ran because an earlier case failed first. `failed` ran and validated nothing.
 */
export type CaseOutcome = 'ok' | 'failed' | 'skipped';

export function caseOutcome(summary: ResultsSummary | null | undefined): CaseOutcome {
    if (!summary) return 'skipped';
    return summary.has_success ? 'ok' : 'failed';
}

/** Case names across a branch, first-seen order, so a case added mid-run still shows. */
export function caseNames(iterations: IterationSnapshot[]): string[] {
    return Array.from(
        new Set(iterations.flatMap((it) => Object.keys(it.results_summary?.cases ?? {}))),
    );
}

export function caseSpeedups(summary: ResultsSummary | null | undefined): number[] {
    if (!summary?.cases) return [];
    return Object.values(summary.cases)
        .map((c) => c?.speedup)
        .filter((s): s is number => s != null && s > 0);
}

/** A log axis: both ends positive, lo < hi. */
export interface LogRange {
    lo: number;
    hi: number;
}

/**
 * Round a range outward to powers of √2 and open it to at least one doubling, so the
 * ends are near the data and every power of two inside is a gridline.
 */
export function niceRange(lo: number, hi: number): LogRange {
    let a = Math.floor(Math.log2(lo) * 2) / 2;
    let b = Math.ceil(Math.log2(hi) * 2) / 2;
    if (b - a < 1) {
        a = Math.floor(a);
        b = Math.max(Math.ceil(b), a + 1);
    }
    return { lo: 2 ** a, hi: 2 ** b };
}

const DEFAULT_RANGE = niceRange(0.5, 2);

/**
 * The per-case axis for a branch: every case's speedup in every iteration, none
 * pinned. The worst case is the point of that view, so it is never cut off — and the
 * range is fitted, not mirrored about 1x, so a branch whose cases all run 1.3–36x
 * does not spend half its track on 1/64x.
 */
export function branchCaseRange(iterations: IterationSnapshot[]): LogRange {
    const all = iterations.flatMap((it) => caseSpeedups(it.results_summary));
    if (!all.length) return DEFAULT_RANGE;
    return niceRange(Math.min(...all), Math.max(...all));
}

export const VISIBLE_BELOW_BEST = 8;

/**
 * The progress axis for a branch: from the best result down to 1/8 of it. Anything
 * lower is pinned at the edge (see BranchHistory) rather than allowed to set the
 * scale — one validated-but-pathological iteration (0.04x on a branch running 115–170x,
 * 43x among 1,500–8,000x; both recorded here) would otherwise flatten every real
 * result into the top few pixels. Genuine early progress is almost always within 8x
 * of where the branch ends up, so it stays on the axis.
 */
export function progressRange(values: number[]): LogRange {
    const positive = values.filter((s) => s > 0);
    if (!positive.length) return DEFAULT_RANGE;
    const best = Math.max(...positive);
    const lo = Math.max(Math.min(...positive), best / VISIBLE_BELOW_BEST);
    return niceRange(lo, best);
}

/** Every power of two inside the range, ends included. */
export function rangeTicks(range: LogRange): number[] {
    const ticks: number[] = [];
    const first = Math.ceil(Math.log2(range.lo) - 1e-9);
    const last = Math.floor(Math.log2(range.hi) + 1e-9);
    for (let k = first; k <= last; k++) ticks.push(2 ** k);
    return ticks;
}

/**
 * Which ticks to label: every one while they sit at least `minPx` apart along an
 * axis `extentPx` long, otherwise every 2nd, 4th… power of two, so labels never pile
 * up on a 72px strip spanning six doublings.
 */
export function isLabelledTick(t: number, range: LogRange, extentPx: number, minPx = 14): boolean {
    const steps = Math.log2(range.hi / range.lo);
    let stride = 1;
    while ((extentPx / steps) * stride < minPx) stride *= 2;
    const k = Math.round(Math.log2(t));
    return ((k % stride) + stride) % stride === 0;
}

export function inRange(v: number, range: LogRange): boolean {
    return v >= range.lo && v <= range.hi;
}

/**
 * Where a value sits along the axis, as a percentage: lo at `pad`, hi at 100 - pad,
 * the padding leaving room for the mark at the ends. Clamped, so a value off the
 * axis lands on its edge.
 */
export function axisPosition(v: number, range: LogRange, pad = 6): number {
    const t = (Math.log2(v) - Math.log2(range.lo)) / Math.log2(range.hi / range.lo);
    return pad + Math.max(0, Math.min(1, t)) * (100 - 2 * pad);
}

export function tickLabel(v: number): string {
    return v >= 1 ? `${v}x` : `1/${Math.round(1 / v)}`;
}
