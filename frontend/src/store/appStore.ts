import { create } from 'zustand';
import type { Problem, TokenUsage, TreeNode } from '@/api/types.generated';
import type { ModelsResponse, FinalizeSummary, FinalizeResponse } from '@/api/client';

// localStorage helpers
const LS_PROVIDER = 'cacao-selected-provider';
const LS_MODEL = 'cacao-selected-model';

function loadSelection(): { provider: string | null; model: string | null } {
    return {
        provider: localStorage.getItem(LS_PROVIDER),
        model: localStorage.getItem(LS_MODEL),
    };
}

function saveSelection(provider: string, model: string) {
    localStorage.setItem(LS_PROVIDER, provider);
    localStorage.setItem(LS_MODEL, model);
}

interface AppState {
    // Problems
    problems: Problem[];
    activeProblem: string | null;

    // Tree
    treeNodes: TreeNode[];
    finalize: FinalizeSummary | null;
    // The full finalize response, fetched once by useFinalize and tagged with its
    // problem so a switch never shows another problem's results.
    finalizeData: { problem: string; data: FinalizeResponse } | null;
    // Set when a finalize POST is accepted. The state read back from the server stays
    // not_run/up_to_date/stale until the child writes its manifest, seconds later.
    startedFinalize: boolean;
    selectedNodeId: string | null;

    // Run state
    runStatus: 'idle' | 'running' | 'completed' | 'error';
    tokenUsage: TokenUsage;

    // LLM info (from running/completed optimization)
    llmModel: string | null;
    llmProvider: string | null;

    // Model selection (user choice for next run)
    availableModels: ModelsResponse | null;
    selectedProvider: string | null;
    selectedModel: string | null;

    // UI state
    sidebarOpen: boolean;
    connected: boolean;
}

interface AppActions {
    setActiveProblem: (name: string | null) => void;
    selectNode: (id: string | null) => void;
    toggleSidebar: () => void;
    setProblems: (problems: Problem[], llmModel?: string, llmProvider?: string) => void;
    setTreeFromAPI: (nodes: TreeNode[], llmModel?: string, llmProvider?: string, tokenUsage?: TokenUsage | null, finalize?: FinalizeSummary | null) => void;
    setRunStatus: (status: AppState['runStatus']) => void;
    setFinalizeData: (problem: string, data: FinalizeResponse) => void;
    setStartedFinalize: (started: boolean) => void;
    setConnected: (connected: boolean) => void;
    setAvailableModels: (models: ModelsResponse) => void;
    setSelectedProvider: (provider: string) => void;
    setSelectedModel: (model: string) => void;
}

const saved = loadSelection();

export const useAppStore = create<AppState & AppActions>((set, get) => ({
    // Initial state — empty until backend responds
    problems: [],
    activeProblem: null,
    treeNodes: [],
    finalize: null,
    finalizeData: null,
    startedFinalize: false,
    selectedNodeId: null,
    runStatus: 'idle',
    tokenUsage: { api_calls: 0, prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    llmModel: null,
    llmProvider: null,
    availableModels: null,
    selectedProvider: saved.provider,
    selectedModel: saved.model,
    sidebarOpen: true,
    connected: false,

    // Actions
    setActiveProblem: (name) => set({ activeProblem: name, selectedNodeId: null, treeNodes: [], finalize: null, finalizeData: null, startedFinalize: false, tokenUsage: { api_calls: 0, prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } }),
    selectNode: (id) => set({ selectedNodeId: id }),
    toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

    // API-driven actions
    setProblems: (problems, llmModel, llmProvider) => set({
        problems,
        connected: true,
        ...(llmModel != null && { llmModel }),
        ...(llmProvider != null && { llmProvider }),
    }),
    setTreeFromAPI: (nodes, llmModel, llmProvider, tokenUsage, finalize) => set({
        treeNodes: nodes,
        finalize: finalize ?? null,
        connected: true,
        ...(llmModel != null && { llmModel }),
        ...(llmProvider != null && { llmProvider }),
        tokenUsage: tokenUsage ?? { api_calls: 0, prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    }),
    setRunStatus: (status) => set({ runStatus: status }),
    setFinalizeData: (problem, data) => set({ finalizeData: { problem, data } }),
    setStartedFinalize: (started) => set({ startedFinalize: started }),
    // A successful response sets it true as a side effect; only the poll loops'
    // catch blocks set it false, so a backend that dies mid-session is visible.
    setConnected: (connected) => set({ connected }),

    // Model selection
    setAvailableModels: (models) => {
        const state = get();
        const providers = Object.keys(models);
        // If no saved provider or saved provider no longer valid, pick first
        let provider = state.selectedProvider;
        if (!provider || !models[provider]) {
            provider = providers[0] ?? null;
        }
        let model = state.selectedModel;
        if (provider && (!model || !models[provider].available.includes(model))) {
            model = models[provider].default;
        }
        if (provider && model) saveSelection(provider, model);
        set({ availableModels: models, selectedProvider: provider, selectedModel: model });
    },
    setSelectedProvider: (provider) => {
        const models = get().availableModels;
        const defaultModel = models?.[provider]?.default ?? null;
        if (provider && defaultModel) saveSelection(provider, defaultModel);
        set({ selectedProvider: provider, selectedModel: defaultModel });
    },
    setSelectedModel: (model) => {
        const provider = get().selectedProvider;
        if (provider && model) saveSelection(provider, model);
        set({ selectedModel: model });
    },
}));

// Selector helpers
// Field selectors, not the bare hook: zustand's default identity selector re-renders
// every consumer on any set(), and this one is called at the App root.
export const useSelectedNode = (): TreeNode | null => {
    const treeNodes = useAppStore((s) => s.treeNodes);
    const selectedNodeId = useAppStore((s) => s.selectedNodeId);
    if (!selectedNodeId) return null;
    return treeNodes.find((n) => n.id === selectedNodeId) ?? null;
};
