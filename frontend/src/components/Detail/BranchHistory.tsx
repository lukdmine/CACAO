import { Fragment } from 'react';

import type { IterationSnapshot } from '@/api/types.generated';
import { formatSpeedup } from '@/utils/statusColors';
import {
    BASELINE,
    CLASSES,
    CLASS_BOUNDS,
    VISIBLE_BELOW_BEST,
    axisPosition,
    boundLabel,
    caseNames,
    classColor,
    inRange,
    isLabelledTick,
    progressRange,
    rangeTicks,
    speedupClass,
    tickLabel,
} from './caseScale';
import { CaseRows } from './CaseHistory';

/**
 * A branch's whole history at once: the headline speedup of every iteration as a
 * strip, and — when the problem declares cases — each case's speedup underneath as a
 * grid (CaseHistory.tsx), one column per iteration for both.
 *
 * The strip is a dot per iteration on a log axis, on a stem from the 1x line. When 1x
 * is off the axis the stem comes from whichever edge 1x lies beyond — the bottom for
 * a branch running 200–300x, the top for one stuck at 0.1–0.5x — so a stem always
 * points away from the reference: standing up means faster, hanging down means
 * slower. Every stem in a strip shares one foot, so longer is further from the
 * reference at a glance; what the length does not carry is magnitude — twice the
 * stem is not twice the speedup — and the gridline labels do that instead.
 * Dots and stems rather than a line because iterations are discrete attempts, not
 * samples of a continuous process — a line through them would draw a trend that was
 * never run. Not a one-row heatmap because one series is exactly where a plotted
 * value works, and seven classes would collapse the moves worth watching on a single
 * kernel (0.44x -> 0.55x is one class).
 *
 * The axis runs from the branch's best result down to 1/8 of it
 * (caseScale.progressRange). An iteration below that is pinned at the bottom edge as
 * a ▾ with its value in the tooltip, rather than allowed to set the scale.
 *
 * The headline is `results_summary.speedup`: the branch's speedup for a single-case
 * problem, the geometric mean over cases otherwise (utils/results.aggregate_case_summaries
 * puts the geomean there; the two fields agree on every recorded iteration).
 *
 * A column is an iteration that reached a result, success or not. Iterations that
 * never produced a summary — a compile failure, or one still running — have no column;
 * the iteration list below carries them.
 */

const LABEL_W = 64; // px
const MIN_CELL = 10; // px; narrower than this the grid scrolls rather than squashing
const GAP = 2; // px; the surface showing between cells is what separates them
const STRIP_H = 72; // px
const STRIP_PAD = 7; // % of the strip's height kept clear at each end for the mark

export function BranchHistory({
    iterations,
    activeIter,
    onSelectIter,
}: {
    iterations: IterationSnapshot[];
    activeIter?: number | null;
    onSelectIter?: (iter: number) => void;
}) {
    const rows = iterations.filter((it) => it.results_summary);
    if (rows.length < 2) return null;

    const n = rows.length;
    const names = caseNames(rows);
    const hasCases = names.length > 0;

    const headline = rows.map((it) => {
        const summary = it.results_summary!;
        const failed = !summary.has_success || summary.failed_case != null;
        const s = summary.speedup;
        return { failed, value: !failed && s != null && s > 0 ? s : null };
    });
    const range = progressRange(headline.flatMap((h) => (h.value != null ? [h.value] : [])));
    const ticks = rangeTicks(range);
    const hasBaseline = inRange(BASELINE, range);
    const yOf = (s: number) => 100 - axisPosition(s, range, STRIP_PAD); // % from the top
    const xOf = (i: number) => ((i + 0.5) / n) * 100;
    const bottomY = 100 - STRIP_PAD;
    const stemFoot = hasBaseline ? yOf(BASELINE) : BASELINE > range.hi ? 0 : 100;

    let best = -1;
    headline.forEach((h, i) => {
        if (h.value != null && (best < 0 || h.value > headline[best].value!)) best = i;
    });
    const anyPinned = headline.some((h) => h.value != null && h.value < range.lo);

    return (
        <div>
            <div className="mb-1.5 flex items-baseline gap-1.5 text-xs">
                <span className="font-medium text-muted-foreground">Branch history</span>
                <span className="text-muted-foreground/70">
                    {hasCases ? `geomean of ${names.length} cases` : 'speedup'}, {n} iterations
                </span>
            </div>

            {/* overflow-x alone computes to overflow-y: auto as well, and a 1px ring on the
                bottom row was enough to summon a vertical scrollbar. Hidden, with the
                padding to keep that ring inside. */}
            <div className="overflow-x-auto overflow-y-hidden">
                <div
                    className="grid items-center pb-px"
                    style={{
                        gap: GAP,
                        gridTemplateColumns: `${LABEL_W}px repeat(${n}, minmax(${MIN_CELL}px, 1fr))`,
                        minWidth: LABEL_W + n * (MIN_CELL + GAP),
                    }}
                >
                    <div />
                    {rows.map((it, i) => (
                        <div
                            key={`h-${it.iter_num}`}
                            className={`text-center text-[9px] tabular-nums ${
                                it.iter_num === activeIter
                                    ? 'font-medium text-foreground'
                                    : 'text-muted-foreground/60'
                            }`}
                        >
                            {i === 0 || it.iter_num % 5 === 0 || it.iter_num === activeIter
                                ? it.iter_num
                                : ''}
                        </div>
                    ))}

                    {/* The strip: tick labels in the label column, the plot across the rest. */}
                    <div className="relative" style={{ height: STRIP_H }}>
                        {ticks
                            .filter((t) => isLabelledTick(t, range, STRIP_H))
                            .map((t) => (
                                <span
                                    key={t}
                                    className={`absolute right-1 -translate-y-1/2 text-[9px] leading-none tabular-nums ${
                                        t === BASELINE ? 'text-foreground/70' : 'text-muted-foreground/60'
                                    }`}
                                    style={{ top: `${yOf(t)}%` }}
                                >
                                    {tickLabel(t)}
                                </span>
                            ))}
                    </div>
                    <div className="relative" style={{ height: STRIP_H, gridColumn: '2 / -1' }}>
                        {ticks.map((t) => (
                            <div
                                key={t}
                                className={`absolute inset-x-0 h-px ${
                                    t === BASELINE ? 'bg-foreground/50' : 'bg-foreground/15'
                                }`}
                                style={{ top: `${yOf(t)}%` }}
                            />
                        ))}
                        {rows.map((it, i) => {
                            const { failed, value } = headline[i];
                            const pinned = value != null && value < range.lo;
                            const x = xOf(i);
                            const color = value != null ? classColor(speedupClass(value)) : undefined;
                            const hint = `iter ${it.iter_num} · ${
                                value != null
                                    ? formatSpeedup(value) +
                                      (pinned ? ` — more than ${VISIBLE_BELOW_BEST}x below the best` : '')
                                    : failed
                                      ? 'failed — no configuration validated'
                                      : 'no baseline'
                            }`;

                            let mark;
                            if (value == null) {
                                // No value: a hollow marker on the baseline if there is one,
                                // else on the bottom edge.
                                mark = (
                                    <div
                                        className={`pointer-events-none absolute size-2 -translate-x-1/2 -translate-y-1/2 rounded-full border ${
                                            failed ? 'border-red-400/70' : 'border-foreground/15'
                                        }`}
                                        style={{ left: `${x}%`, top: `${hasBaseline ? yOf(BASELINE) : bottomY}%` }}
                                    />
                                );
                            } else if (pinned) {
                                mark = (
                                    <svg
                                        className="pointer-events-none absolute -translate-x-1/2"
                                        style={{ left: `${x}%`, top: `calc(${bottomY}% - 3px)` }}
                                        width="8"
                                        height="6"
                                        viewBox="0 0 8 6"
                                    >
                                        <polygon points="0,0 8,0 4,6" fill={color} />
                                    </svg>
                                );
                            } else {
                                const y = yOf(value);
                                const labelBelow = y < 30;
                                mark = (
                                    <>
                                        <div
                                            className="pointer-events-none absolute w-0.5 -translate-x-1/2"
                                            style={{
                                                left: `${x}%`,
                                                top: `${Math.min(y, stemFoot)}%`,
                                                height: `${Math.abs(y - stemFoot)}%`,
                                                background: color,
                                            }}
                                        />
                                        <div
                                            className="pointer-events-none absolute size-2 -translate-x-1/2 -translate-y-1/2 rounded-full"
                                            style={{ left: `${x}%`, top: `${y}%`, background: color }}
                                        />
                                        {i === best && (
                                            <span
                                                className={`pointer-events-none absolute text-[9px] leading-none tabular-nums text-foreground/80 ${
                                                    x < 15 ? '' : x > 85 ? '-translate-x-full' : '-translate-x-1/2'
                                                }`}
                                                style={{
                                                    left: `${x}%`,
                                                    ...(labelBelow
                                                        ? { top: `calc(${y}% + 6px)` }
                                                        : { bottom: `calc(${100 - y}% + 6px)` }),
                                                }}
                                            >
                                                {formatSpeedup(value)}
                                            </span>
                                        )}
                                    </>
                                );
                            }

                            return (
                                <Fragment key={it.iter_num}>
                                    {/* The hit target is the whole column, not the dot: hover shows
                                        the value, a click opens the iteration. It also carries the
                                        highlight for the open iteration. Marks let events through. */}
                                    <button
                                        type="button"
                                        className={`absolute inset-y-0 hover:bg-foreground/5 ${
                                            it.iter_num === activeIter ? 'bg-foreground/5' : ''
                                        }`}
                                        style={{ left: `${(i / n) * 100}%`, width: `${100 / n}%` }}
                                        title={hint}
                                        onClick={() => onSelectIter?.(it.iter_num)}
                                    />
                                    {mark}
                                </Fragment>
                            );
                        })}
                    </div>

                    {hasCases && (
                        <>
                            <div className="my-0.5 h-px bg-border" style={{ gridColumn: '1 / -1' }} />
                            <CaseRows
                                iterations={rows}
                                names={names}
                                activeIter={activeIter}
                                onSelectIter={onSelectIter}
                            />
                        </>
                    )}
                </div>
            </div>

            {/* Key: the seven classes with their boundaries, in the axis's own vocabulary.
                The strip's dots and the cells share it. */}
            <div className="mt-2 flex flex-wrap items-start gap-x-3 gap-y-1 text-[9px] text-muted-foreground/70">
                <div className="flex items-start gap-1.5">
                    <span className="leading-[8px]">slower</span>
                    <div>
                        <div className="flex gap-px">
                            {CLASSES.map((c) => (
                                <div
                                    key={c}
                                    className="h-2 w-6 rounded-[1px]"
                                    style={{ background: classColor(c) }}
                                />
                            ))}
                        </div>
                        <div className="relative h-3">
                            {CLASS_BOUNDS.map((b, i) => (
                                <span
                                    key={b}
                                    className="absolute top-0.5 -translate-x-1/2 tabular-nums"
                                    style={{ left: `${((i + 1) / CLASSES.length) * 100}%` }}
                                >
                                    {boundLabel(b)}
                                </span>
                            ))}
                        </div>
                    </div>
                    <span className="leading-[8px]">faster</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="h-2 w-2.5 rounded-[1px] border border-foreground/15" />
                    no result
                </div>
                <div className="flex items-center gap-1">
                    <div className="h-2 w-2.5 rounded-[1px] border border-red-400/70" />
                    failed
                </div>
                {anyPinned && (
                    <div className="flex items-center gap-1">
                        <svg width="8" height="6" viewBox="0 0 8 6" className="fill-muted-foreground/70">
                            <polygon points="0,0 8,0 4,6" />
                        </svg>
                        more than {VISIBLE_BELOW_BEST}x below the best
                    </div>
                )}
            </div>
        </div>
    );
}
