import { Fragment } from 'react';

import type { IterationSnapshot } from '@/api/types.generated';
import { formatSpeedup } from '@/utils/statusColors';
import { caseOutcome, classColor, speedupClass, type CaseOutcome } from './caseScale';

/**
 * Every case's speedup across every iteration of a branch, as rows of a grid.
 *
 * These are children of the branch-history grid (BranchHistory.tsx), not a grid of
 * their own: the strip above plots one point per iteration, and a cell must sit
 * exactly under its point. Two grids with the same template would line up too, but
 * one grid cannot fail to.
 *
 * The per-iteration table answers "what happened this round". It structurally cannot
 * answer the question that actually decides what to do next — has this case ever
 * worked, and did I just break it? On a real branch here, mawi collapsed from 1.40x to
 * 0.18x at iteration 7 and recovered at 11, while transient never once cleared 1.00x
 * in twenty iterations. Neither fact is visible one iteration at a time.
 *
 * Cells take one of seven classes at fixed ratio thresholds (caseScale.ts) rather than
 * a continuous ramp: crossing 1x, halving and doubling are the events worth seeing,
 * and a step is legible at 10px where a 5% tint difference is not. A cell with no
 * result is hollow, never coloured, so absence cannot read as a value.
 */

function cellLook(outcome: CaseOutcome, speedup: number | null | undefined) {
    if (outcome === 'ok' && speedup != null && speedup > 0) {
        return { className: '', style: { background: classColor(speedupClass(speedup)) } };
    }
    if (outcome === 'failed') return { className: 'border border-red-400/70', style: undefined };
    return { className: 'border border-foreground/15', style: undefined };
}

function cellHint(outcome: CaseOutcome, speedup: number | null | undefined) {
    if (outcome === 'skipped') return 'not run — an earlier case failed first';
    if (outcome === 'failed') return 'failed — no configuration validated';
    return speedup != null ? formatSpeedup(speedup) : 'no baseline';
}

export function CaseRows({
    iterations,
    names,
    activeIter,
    onSelectIter,
}: {
    iterations: IterationSnapshot[];
    names: string[];
    activeIter?: number | null;
    onSelectIter?: (iter: number) => void;
}) {
    return (
        <>
            {names.map((name) => (
                <Fragment key={name}>
                    <div className="truncate pr-1 font-mono text-[10px]" title={name}>
                        {name}
                    </div>
                    {iterations.map((it) => {
                        const caseSummary = it.results_summary?.cases?.[name];
                        const outcome = caseOutcome(caseSummary);
                        const speedup = caseSummary?.speedup;
                        const { className, style } = cellLook(outcome, speedup);
                        return (
                            <button
                                key={`${name}-${it.iter_num}`}
                                type="button"
                                className={`h-4 rounded-[2px] hover:ring-1 hover:ring-foreground/40 ${className} ${
                                    it.iter_num === activeIter ? 'ring-1 ring-foreground/40' : ''
                                }`}
                                style={style}
                                title={`iter ${it.iter_num} · ${name} · ${cellHint(outcome, speedup)}`}
                                onClick={() => onSelectIter?.(it.iter_num)}
                            />
                        );
                    })}
                </Fragment>
            ))}
        </>
    );
}
