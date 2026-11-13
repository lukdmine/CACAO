import type { FinalizeSummary } from '@/api/client';

/**
 * Tree node ids are "branch/" + the path under output/branches/ (api/tree.py), so
 * nested branches read "branch/a/branches/b". A finalist's source_branch is the path
 * under output/ ("branches/a/branches/b"). Dropping the leading "branches/" from the
 * latter and prefixing "branch/" gives the node id.
 */
export function nodeIdForSourceBranch(sourceBranch: string): string {
    return 'branch/' + sourceBranch.replace(/^branches\//, '');
}

export function isFinalist(finalize: FinalizeSummary | null | undefined, nodeId: string): boolean {
    return !!finalize?.finalists.some((f) => nodeIdForSourceBranch(f.source_branch) === nodeId);
}
