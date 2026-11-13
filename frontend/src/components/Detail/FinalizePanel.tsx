import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { formatSpeedup } from '@/utils/statusColors';
import type { FinalizeResponse, FinalizeState } from '@/api/client';
import type { FinalistState, KernelTags, Variant } from '@/api/types.generated';

const STATE_TEXT: Record<FinalizeState, string> = {
    not_run: 'Not run',
    finalizing: 'Finalizing',
    up_to_date: 'Up to date',
    stale: 'Stale (branches changed since this finalize)',
    failed: 'Failed',
    stopped: 'Stopped before it finished',
};

const STATE_TONE: Record<FinalizeState, string> = {
    not_run: 'text-muted-foreground',
    finalizing: 'text-amber-400',
    up_to_date: 'text-emerald-400',
    stale: 'text-amber-400',
    failed: 'text-red-400',
    stopped: 'text-amber-400',
};

/** Variant.summary is an open record; the speedup is the only field read from it. */
function speedupOf(v: Variant | null | undefined): number | null {
    const s = v?.summary?.speedup;
    return typeof s === 'number' ? s : null;
}

function tagLabels(t: KernelTags | null | undefined): string[] {
    if (!t) return [];
    return [
        t.tensor_cores && 'tensor cores',
        t.async_copy && 'async copy',
        t.precision && t.precision,
        t.sm_scaled && 'SM scaled',
    ].filter((x): x is string => !!x);
}

/** The variant the manifest reports: a retune that lost to its source is not it (D13). */
function reportedVariant(f: FinalistState): Variant {
    return f.reported === 'finalized' && f.finalized ? f.finalized : f.source;
}

function minArch(v: Variant): number | null {
    return v.arch?.winner_min_arch ?? v.tags?.min_arch ?? null;
}

const STAGE_TONE: Record<string, string> = {
    done: 'text-emerald-400',
    failed: 'text-red-400',
};

/** The branches above a finalist: `branches/a/branches/b` gives ["a"]. Its name is the last one. */
function parentBranches(rel: string): string[] {
    return rel.split('/').filter((p) => p && p !== 'branches').slice(0, -1);
}

// One stacked block per finalist rather than a table row: strategy names are long
// snake_case identifiers the browser cannot wrap, and six columns in the side panel
// scrolled sideways. `wrap-anywhere` lets the names break instead.
function FinalistItem({ f }: { f: FinalistState }) {
    const shown = reportedVariant(f);
    const arch = minArch(shown);
    const tags = tagLabels(shown.tags);
    const parents = parentBranches(f.source_branch);
    return (
        <li className="space-y-1 border-t py-1.5 first:border-t-0">
            <div className="flex items-baseline justify-between gap-2">
                <span className="min-w-0 font-medium wrap-anywhere" title={f.source_branch}>{f.name}</span>
                <span className="shrink-0 font-mono">
                    {formatSpeedup(speedupOf(f.source))}
                    {f.finalized && <> → {formatSpeedup(speedupOf(f.finalized))}</>}
                </span>
            </div>
            {parents.length > 0 && (
                <div className="text-muted-foreground wrap-anywhere">under {parents.join(' / ')}</div>
            )}
            {(tags.length > 0 || arch != null) && (
                <div className="flex flex-wrap items-center gap-1">
                    {tags.map((t) => (
                        <Badge key={t} variant="secondary" className="px-1.5 py-0 text-[10px]">{t}</Badge>
                    ))}
                    {arch != null && <span className="font-mono text-muted-foreground">min sm_{arch}</span>}
                </div>
            )}
            <div className="flex flex-wrap gap-x-3 text-muted-foreground">
                {Object.entries(f.stages).map(([stage, status]) => (
                    <span key={stage}>
                        {stage} <span className={STAGE_TONE[status] ?? 'text-foreground'}>{status}</span>
                    </span>
                ))}
                {f.reported === 'source' && f.finalized && <span>retune did not beat the source</span>}
            </div>
            {f.error && (
                <p className="text-red-400 wrap-anywhere" title={f.error}>
                    {f.error.length > 200 ? `${f.error.slice(0, 200)}...` : f.error}
                </p>
            )}
        </li>
    );
}

/** The finalize pass's results, shown on the root node. */
export function FinalizePanel({ data, starting }: { data: FinalizeResponse | null; starting: boolean }) {
    // `starting` covers the seconds between accepting the POST and the first manifest,
    // so the panel does not stay hidden while the toolbar already says "Finalizing...".
    if (!starting && (!data || (data.state === 'not_run' && !data.manifest))) return null;

    const state: FinalizeState = starting ? 'finalizing' : data!.state;
    const manifest = data?.manifest ?? null;
    const overview = data?.overview_markdown ?? null;
    const finalists = manifest?.finalists ?? [];
    const bests = Object.entries(manifest?.category_bests ?? {});

    return (
        <div className="space-y-2 text-xs">
            <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-muted-foreground">Finalize</span>
                <span className={STATE_TONE[state]}>{STATE_TEXT[state]}</span>
                {manifest && state === 'finalizing' && (
                    <span className="text-muted-foreground">({manifest.status})</span>
                )}
            </div>

            {manifest?.error && <p className="text-red-400">{manifest.error}</p>}

            {overview && (
                <Accordion type="single" collapsible defaultValue="overview">
                    <AccordionItem value="overview" className="border-b-0">
                        <AccordionTrigger className="py-1.5 text-xs font-medium text-muted-foreground hover:no-underline hover:text-foreground">
                            Overview
                        </AccordionTrigger>
                        <AccordionContent>
                            <pre className="mt-1 rounded-md bg-muted/40 p-2 text-[11px] font-mono whitespace-pre-wrap overflow-x-auto">{overview}</pre>
                        </AccordionContent>
                    </AccordionItem>
                </Accordion>
            )}

            {finalists.length > 0 && (
                <div>
                    <span className="block text-xs font-medium text-muted-foreground">Finalists</span>
                    <ul className="text-[11px]">
                        {finalists.map((f) => <FinalistItem key={f.name} f={f} />)}
                    </ul>
                </div>
            )}

            {bests.length > 0 && (
                <div>
                    <span className="block text-xs font-medium text-muted-foreground">Category bests</span>
                    <div className="mt-0.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px]">
                        {bests.map(([category, name]) => (
                            <span key={category} className="min-w-0 wrap-anywhere">
                                <span className="text-muted-foreground">{category}:</span> {name}
                            </span>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
