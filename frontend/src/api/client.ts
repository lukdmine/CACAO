const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8003';
/**
 * Typed fetch wrapper for the backend API.
 */
async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
    const res = await fetch(`${API_BASE}${path}`, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options?.headers,
        },
    });
    if (!res.ok) {
        const body = await res.text();
        throw new Error(`API ${res.status}: ${body}`);
    }
    return res.json();
}

// =============================================================================
// Problem endpoints
// =============================================================================

export interface ProblemsResponse {
    problems: Array<{
        name: string;
        status: string;
        description: string;
    }>;
    llm_model?: string;
    llm_provider?: string;
}

export async function fetchProblems(): Promise<ProblemsResponse> {
    return apiFetch<ProblemsResponse>('/api/problems');
}

export interface ProblemDetailResponse {
    name: string;
    config: Record<string, unknown>;
    ref_kernel: string;
    ref_cpu: string;
    has_output: boolean;
}

export async function fetchProblemDetail(name: string): Promise<ProblemDetailResponse> {
    return apiFetch<ProblemDetailResponse>(`/api/problems/${name}/detail`);
}

// =============================================================================
// Tree endpoint
// =============================================================================

import type { TreeNode, TokenUsage } from '@/api/types.generated';

export interface TreeResponse {
    nodes: TreeNode[];
    analysis: string;
    strategies: unknown[];
    running: boolean;
    llm_model?: string;
    llm_provider?: string;
    token_usage?: TokenUsage | null;
    tuning_duration_s?: number | null;
}

export async function fetchTree(problemName: string): Promise<TreeResponse> {
    return apiFetch<TreeResponse>(`/api/problems/${problemName}/tree`);
}

// =============================================================================
// Create Problem
// =============================================================================

export interface CreateProblemData {
    slug: string;
    name: string;
    description: string;
    gpu: {
        index: number;
    };
    tuning?: {
        duration_s: number;
    };
    reference_type: 'cuda' | 'cpu_c';
    ref_function: string;
    ref_block_x: number;
    ref_block_y: number;
    ref_block_z: number;
    ref_kernel_code: string;
    ref_cpu_code: string;
    scalars: { name: string; dtype: string; value: number }[];
    vectors: { name: string; dtype: string; size: string; access: string; init: string; init_min?: number | null; init_max?: number | null; validate: boolean }[];
    grid_x: string;
    grid_y: string;
    grid_z: string;
    tolerance: number;
}

export async function createProblem(data: CreateProblemData) {
    return apiFetch<{ status: string; name: string; path: string }>('/api/problems', {
        method: 'POST',
        body: JSON.stringify(data),
    });
}

export async function updateProblem(name: string, data: CreateProblemData) {
    return apiFetch<{ status: string; name: string; path: string }>(`/api/problems/${name}`, {
        method: 'PUT',
        body: JSON.stringify(data),
    });
}

export async function deleteProblem(name: string) {
    return apiFetch<{ status: string; name: string }>(`/api/problems/${name}`, {
        method: 'DELETE',
    });
}

export async function cloneProblem(name: string, newName: string) {
    return apiFetch<{ status: string; name: string; source: string }>(
        `/api/problems/${name}/clone`,
        { method: 'POST', body: JSON.stringify({ new_name: newName }) },
    );
}

// =============================================================================
// Run / Resume
// =============================================================================

export interface RunConfig {
    max_iter?: number;
    max_depth?: number;
    path_budget?: number;
    timeout?: number;
    model?: string;
    provider?: string;
}

export async function runProblem(name: string, config?: RunConfig) {
    return apiFetch(`/api/problems/${name}/run`, {
        method: 'POST',
        body: JSON.stringify(config ?? {}),
    });
}

export async function resumeProblem(name: string, config?: RunConfig) {
    return apiFetch(`/api/problems/${name}/resume`, {
        method: 'POST',
        body: JSON.stringify(config ?? {}),
    });
}

// =============================================================================
// Models
// =============================================================================

export interface ModelsResponse {
    [provider: string]: {
        default: string;
        available: string[];
    };
}

export async function fetchModels(): Promise<ModelsResponse> {
    return apiFetch<ModelsResponse>('/api/models');
}

// =============================================================================
// Stop
// =============================================================================

export async function stopProblem(name: string) {
    return apiFetch(`/api/problems/${name}/stop`, {
        method: 'POST',
    });
}

export interface LogsResponse {
    log: string;
    lines: number;
    truncated: boolean;
}

export async function fetchLogs(name: string, tail = 500): Promise<LogsResponse> {
    return apiFetch<LogsResponse>(`/api/problems/${name}/logs?tail=${tail}`);
}

// =============================================================================
// Status
// =============================================================================

export async function fetchStatus() {
    return apiFetch<{ running_problems: Record<string, boolean>; active_count: number }>('/api/status');
}

// =============================================================================
// GPU
// =============================================================================

export interface GpuDevice {
    index: number;
    name?: string;
    model?: string;
    total_memory_mb?: number;
    free_memory_mb?: number;
    compute_capability?: string;
    sm_count?: number;
}

export async function fetchGpuDevices() {
    return apiFetch<{ devices: GpuDevice[] }>('/api/gpu/devices');
}

// =============================================================================
// Branch Control
// =============================================================================

export async function stopBranch(problem: string, branchId: string) {
    return apiFetch<{ status: string; branch: string }>(
        `/api/problems/${problem}/branches/${branchId}/stop`,
        { method: 'POST' },
    );
}

export async function resumeBranch(problem: string, branchId: string) {
    return apiFetch<{ status: string; branch: string }>(
        `/api/problems/${problem}/branches/${branchId}/resume`,
        { method: 'POST' },
    );
}

export async function messageBranch(problem: string, branchId: string, content: string) {
    return apiFetch<{ status: string; branch: string }>(
        `/api/problems/${problem}/branches/${branchId}/message`,
        { method: 'POST', body: JSON.stringify({ content }) },
    );
}

export async function changeDecision(
    problem: string,
    branchId: string,
    targetIter: number,
    content?: string,
) {
    return apiFetch<{ status: string; branch: string; to_iter: number }>(
        `/api/problems/${problem}/branches/${branchId}/change-decision`,
        { method: 'POST', body: JSON.stringify({ target_iter: targetIter, content }) },
    );
}

export async function configureBranch(
    problem: string,
    branchId: string,
    config: { max_iter?: number },
) {
    return apiFetch<{ status: string; branch: string }>(
        `/api/problems/${problem}/branches/${branchId}/config`,
        { method: 'POST', body: JSON.stringify(config) },
    );
}

export async function deleteBranch(problem: string, branchId: string) {
    return apiFetch<{ status: string; branch: string }>(
        `/api/problems/${problem}/branches/${branchId}`,
        { method: 'DELETE' },
    );
}
