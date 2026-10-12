import { Fragment, type ReactNode } from 'react';

/**
 * A line of facts separated by middots — `● Running · iter 5/20 · 1.24 ms · 1.40x`.
 * Values are text, not chips: a pill says "distinct object", and these are facts about
 * one thing. Falsy items are skipped so callers can pass conditionals straight in.
 */
export function Facts({ items, className = '' }: { items: ReactNode[]; className?: string }) {
    const shown = items.filter(Boolean);
    return (
        <div className={`flex min-w-0 flex-wrap items-center gap-x-1.5 ${className}`}>
            {shown.map((item, i) => (
                <Fragment key={i}>
                    {i > 0 && <span className="text-muted-foreground/40">·</span>}
                    {item}
                </Fragment>
            ))}
        </div>
    );
}
