import { useEffect } from 'react';
import { useAppStore } from '@/store/appStore';
import { fetchProblems, fetchTree, fetchModels } from './client';

const POLL_INTERVAL_MS = 3000;

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

/**
 * Immediately refresh the tree for the active problem (call after any branch mutation).
 */
export async function refreshTree() {
    const { activeProblem, setTreeFromAPI, setRunStatus } = useAppStore.getState();
    if (!activeProblem) return;
    try {
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

/**
 * Hook that polls the backend for tree data of the active problem.
 */
export function usePollTree() {
    const activeProblem = useAppStore((s) => s.activeProblem);

    useEffect(() => {
        if (!activeProblem) return;

        let cancelled = false;

        async function loop() {
            if (cancelled) return;
            try {
                const data = await fetchTree(activeProblem!);
                if (!cancelled) {
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
