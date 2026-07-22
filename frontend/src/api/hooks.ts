import { useEffect, useState } from 'react';
import { useAppStore } from '@/store/appStore';
import { fetchProblems, fetchTree, fetchTreeConditional, fetchIteration, fetchModels } from './client';
import type { IterationDetail } from './types.generated';

export const POLL_INTERVAL_MS = 3000;

/**
 * Immediately refresh the problems list (call after create/update/delete).
 */
export async function refreshProblems() {
    try {
        const data = await fetchProblems();
        useAppStore.getState().setProblems(data.problems);
    } catch {
        // silent — next poll will pick it up
    }
}

// ETag of the tree currently in the store. Shared between the poll loop and
// refreshTree so a mutation-driven refresh doesn't leave the loop holding a
// stale validator. Cleared whenever we fetch unconditionally.
let treeEtag: string | null = null;

/**
 * Immediately refresh the tree for the active problem (call after any branch mutation).
 */
export async function refreshTree() {
    const { activeProblem, setTreeFromAPI, setRunStatus } = useAppStore.getState();
    if (!activeProblem) return;
    try {
        treeEtag = null;
        const data = await fetchTree(activeProblem);
        setTreeFromAPI(data.nodes, data.llm_model, data.llm_provider, data.token_usage, data.tuning_duration_s);
        setRunStatus(data.running ? 'running' : 'completed');
    } catch {
        // silent — next poll will pick it up
    }
}

/**
 * Hook that fetches available models once on mount.
 */
export function useFetchModels() {
    useEffect(() => {
        fetchModels()
            .then((data) => useAppStore.getState().setAvailableModels(data))
            .catch((err) => console.debug('[useFetchModels] Failed:', err));
    }, []);
}

/**
 * Hook that polls the backend for problems list.
 */
export function usePollProblems() {
    useEffect(() => {
        let cancelled = false;

        async function loop() {
            if (cancelled) return;
            try {
                const data = await fetchProblems();
                if (!cancelled) {
                    useAppStore.getState().setProblems(data.problems);
                }
            } catch (err) {
                console.debug('[usePollProblems] Backend not available:', err);
            }

            if (!cancelled) {
                setTimeout(loop, POLL_INTERVAL_MS * 3);
            }
        }

        void loop();
        return () => { cancelled = true; };
    }, []);
}

// Fetched iteration bodies, keyed by problem/branch/iter. Reopening an
// iteration renders from here instantly while a conditional request confirms it
// — the etag rides along, so that confirmation is a 0-byte 304.
//
// Capped because a single entry can be a few hundred KB of NCU metrics, and a
// long browsing session would otherwise pin every iteration it ever opened.
const ITER_CACHE_MAX = 40;
const iterCache = new Map<string, { detail: IterationDetail; etag: string | null }>();

function cacheIteration(key: string, detail: IterationDetail, etag: string | null) {
    iterCache.delete(key);  // re-insert so Map order is least-recently-used first
    iterCache.set(key, { detail, etag });
    while (iterCache.size > ITER_CACHE_MAX) {
        iterCache.delete(iterCache.keys().next().value!);
    }
}

/**
 * Fetch the full work products for one expanded iteration.
 *
 * The tree only carries presence flags, so the panel asks for a body when the
 * reader actually opens an iteration. `live` keeps the currently-running
 * iteration refreshing, so run output still tails in place.
 */
export function useIterationDetail(
    problem: string | null,
    branchId: string,
    iterNum: number | null,
    live: boolean,
): { detail: IterationDetail | null; loading: boolean } {
    const key = problem && iterNum !== null ? `${problem}/${branchId}/${iterNum}` : null;
    const [detail, setDetail] = useState<IterationDetail | null>(
        () => (key && iterCache.get(key)?.detail) || null,
    );
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (!key || !problem || iterNum === null) {
            setDetail(null);
            return;
        }

        const cached = iterCache.get(key);
        setDetail(cached?.detail ?? null);
        setLoading(!cached);

        let cancelled = false;
        let timer: ReturnType<typeof setTimeout>;

        async function poll() {
            try {
                const { data, etag } = await fetchIteration(
                    problem!, branchId, iterNum!, iterCache.get(key!)?.etag ?? null,
                );
                if (cancelled) return;
                if (data) {
                    cacheIteration(key!, data, etag);
                    setDetail(data);
                }
            } catch (err) {
                console.debug('[useIterationDetail] Error:', err);
            } finally {
                if (!cancelled) {
                    setLoading(false);
                    if (live) timer = setTimeout(poll, POLL_INTERVAL_MS);
                }
            }
        }

        // Always revalidate, even from cache: a run ending flips `live` off, and
        // without this pass the last iteration would keep whatever it happened
        // to hold at the final poll. A cache hit makes this a 0-byte 304.
        void poll();

        return () => { cancelled = true; clearTimeout(timer); };
    }, [key, problem, branchId, iterNum, live]);

    return { detail, loading };
}

/**
 * Hook that polls the backend for tree data of the active problem.
 */
export function usePollTree() {
    const activeProblem = useAppStore((s) => s.activeProblem);

    useEffect(() => {
        if (!activeProblem) return;

        let cancelled = false;
        treeEtag = null;  // a different problem's validator means nothing here

        async function loop() {
            if (cancelled) return;
            try {
                const { data, etag } = await fetchTreeConditional(activeProblem!, treeEtag);
                // A 304 leaves the store — and therefore every React Flow node —
                // exactly as it was. Nothing has changed, so nothing re-renders.
                if (!cancelled && data) {
                    treeEtag = etag;
                    const { setTreeFromAPI, setRunStatus } = useAppStore.getState();
                    setTreeFromAPI(data.nodes, data.llm_model, data.llm_provider, data.token_usage, data.tuning_duration_s);
                    setRunStatus(data.running ? 'running' : (data.nodes.length > 1 ? 'completed' : 'idle'));
                }
            } catch (err) {
                console.debug('[usePollTree] Error:', err);
            }

            if (!cancelled) {
                setTimeout(loop, POLL_INTERVAL_MS);
            }
        }

        void loop();
        return () => { cancelled = true; };
    }, [activeProblem]);
}
