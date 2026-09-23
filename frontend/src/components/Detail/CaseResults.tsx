import { Fragment } from 'react';
import { XCircle, CircleDashed } from 'lucide-react';

import { formatSpeedup, formatTime } from '@/utils/statusColors';
import type { ResultsSummary } from '@/api/types.generated';
import {
    BASELINE,
    axisPosition,
    caseOutcome,
    classColor,
    inRange,
    isLabelledTick,
    rangeTicks,
    speedupClass,
    tickLabel,
    type LogRange,
} from './caseScale';

/**
 * Per-case tuning results for a problem declaring several input cases.
 *
 * Each case is a dot on a shared log axis, with a stem from the 1x line whenever 1x is
 * on the axis. A ratio is a distance from the baseline, not an amount from zero —
 * 2x and 1/2x are the same step in opposite directions — and the stem shows which way
 * and how far without hunting for the tick. Stem and dot wear the colour class the
 * branch heatmap gives that value (caseScale.ts), so the two views agree.
 *
 * The axis comes from the branch (caseScale.branchCaseRange), so a case at the same
 * speedup lands in the same place in every iteration.
 *
 * The flat summary above this deliberately mixes scales — best_time_us and
 * reference_time_us describe the PRIMARY case while speedup is the geometric mean over
 * all of them (utils/results.aggregate_case_summaries). Side by side they invite
 * arithmetic that does not work, which is what this table exists to replace.
 */

// Fixed rather than auto: the columns must line up across rows, and an axis whose
// length depends on how long its case name happens to be cannot be read.
const COLS = 'grid-cols-[minmax(0,84px)_58px_58px_minmax(110px,1fr)_44px]';
const TRACK_MIN_PX = 110; // the track's minimum width, which the label stride is sized for

export function CaseResults({ summary, range }: { summary: ResultsSummary; range: LogRange }) {
    const cases = summary.cases;
    if (!cases) return null;

    const entries = Object.entries(cases);
    const ticks = rangeTicks(range);
    const hasBaseline = inRange(BASELINE, range);
    const origin = hasBaseline ? axisPosition(BASELINE, range) : null;

    return (
        <div className="pt-1">
            <div className="mb-1 flex items-baseline gap-1.5 text-xs">
                <span className="font-medium text-muted-foreground">Per-case results</span>
                <span className="text-muted-foreground/70">{entries.length} cases</span>
            </div>

            {/* One grid for every row, not a grid per row: `auto` columns size to each
                row's own content, which left every axis a different length. No row gap,
                so the baseline and ticks run unbroken down the track column. */}
            <div className={`grid ${COLS} items-center gap-x-2 text-[10px]`}>
                <div className="pb-1 text-muted-foreground/70">case</div>
                <div className="pb-1 text-right text-muted-foreground/70">best</div>
                <div className="pb-1 text-right text-muted-foreground/70">reference</div>
                <div className="flex justify-between pb-1 text-[9px] text-muted-foreground/60">
                    <span>← slower</span>
                    <span>faster →</span>
                </div>
                <div className="pb-1 text-right text-muted-foreground/70">speedup</div>

                {entries.map(([name, caseSummary]) => {
                    const outcome = caseOutcome(caseSummary);
                    const speedup = caseSummary?.speedup ?? null;
                    const drawable = outcome === 'ok' && speedup != null && speedup > 0;
                    const color = drawable ? classColor(speedupClass(speedup)) : undefined;
                    const pos = drawable ? axisPosition(speedup, range) : null;
                    const isWorst = summary.worst_case === name && entries.length > 1;

                    const hint = caseSummary
                        ? `${name}: ${caseSummary.num_successful}/${caseSummary.num_total} configurations passed` +
                          (caseSummary.best_config
                              ? `\n${Object.entries(caseSummary.best_config)
                                    .map(([k, v]) => `${k}=${v}`)
                                    .join(', ')}`
                              : '')
                        : `${name}: never ran — an earlier case failed first`;

                    return (
                        <Fragment key={name}>
                            <div className="flex h-5 min-w-0 items-center gap-1" title={hint}>
                                {outcome === 'failed' && (
                                    <XCircle size={11} className="shrink-0 text-red-400" />
                                )}
                                {outcome === 'skipped' && (
                                    <CircleDashed size={11} className="shrink-0 text-muted-foreground/60" />
                                )}
                                <span className="truncate font-mono">{name}</span>
                                {isWorst && (
                                    <span className="shrink-0 text-[9px] text-muted-foreground/70">
                                        worst
                                    </span>
                                )}
                            </div>

                            <span
                                className="text-right font-mono tabular-nums text-muted-foreground"
                                title={hint}
                            >
                                {formatTime(caseSummary?.best_time_us ?? null)}
                            </span>
                            <span
                                className="text-right font-mono tabular-nums text-muted-foreground/60"
                                title={hint}
                            >
                                {formatTime(caseSummary?.reference_time_us ?? null)}
                            </span>

                            <div className="relative h-5" title={hint}>
                                {ticks.map((t) => (
                                    <div
                                        key={t}
                                        className={`absolute inset-y-0 w-px ${
                                            t === BASELINE ? 'bg-foreground/50' : 'bg-foreground/15'
                                        }`}
                                        style={{ left: `${axisPosition(t, range)}%` }}
                                    />
                                ))}
                                {pos != null ? (
                                    <>
                                        {origin != null && (
                                            <div
                                                className="absolute top-1/2 h-0.5 -translate-y-1/2"
                                                style={{
                                                    left: `${Math.min(origin, pos)}%`,
                                                    width: `${Math.abs(pos - origin)}%`,
                                                    background: color,
                                                }}
                                            />
                                        )}
                                        <div
                                            className="absolute top-1/2 size-2 -translate-x-1/2 -translate-y-1/2 rounded-full"
                                            style={{ left: `${pos}%`, background: color }}
                                        />
                                    </>
                                ) : (
                                    <span
                                        className="absolute inset-y-0 flex items-center text-[9px] text-muted-foreground/60"
                                        style={{ left: `${(origin ?? 0) + 2}%` }}
                                    >
                                        {outcome === 'failed'
                                            ? 'failed'
                                            : outcome === 'skipped'
                                              ? 'not run'
                                              : 'no baseline'}
                                    </span>
                                )}
                            </div>

                            <span className="text-right font-mono tabular-nums" title={hint}>
                                {formatSpeedup(speedup)}
                            </span>
                        </Fragment>
                    );
                })}

                {/* The axis, labelled once under the marks. */}
                <div className="col-start-4 relative h-3.5">
                    {ticks
                        .filter((t) => isLabelledTick(t, range, TRACK_MIN_PX))
                        .map((t) => (
                            <div
                                key={t}
                                className={`absolute top-0.5 -translate-x-1/2 text-[9px] tabular-nums ${
                                    t === BASELINE ? 'text-foreground/70' : 'text-muted-foreground/60'
                                }`}
                                style={{ left: `${axisPosition(t, range)}%` }}
                            >
                                {tickLabel(t)}
                            </div>
                        ))}
                </div>
            </div>
        </div>
    );
}
