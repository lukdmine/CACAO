interface StatusStyle {
    color: string;
    text: string;
    border: string;
    label: string;
    animate?: boolean;
}

const STATUS_MAP: Record<string, StatusStyle> = {
    initialized: { color: 'bg-zinc-500', text: 'text-zinc-400', border: 'border-zinc-500/30', label: 'Initialized' },
    planning: { color: 'bg-teal-500', text: 'text-teal-400', border: 'border-teal-500/30', label: 'Planning', animate: true },
    implementing: { color: 'bg-blue-500', text: 'text-blue-400', border: 'border-blue-500/30', label: 'Implementing', animate: true },
    configuring: { color: 'bg-indigo-500', text: 'text-indigo-400', border: 'border-indigo-500/30', label: 'Configuring', animate: true },
    running: { color: 'bg-amber-500', text: 'text-amber-400', border: 'border-amber-500/30', label: 'Running', animate: true },
    profiling: { color: 'bg-purple-500', text: 'text-purple-400', border: 'border-purple-500/30', label: 'Profiling', animate: true },
    proposing: { color: 'bg-pink-500', text: 'text-pink-400', border: 'border-pink-500/30', label: 'Proposing', animate: true },
    deciding: { color: 'bg-orange-500', text: 'text-orange-400', border: 'border-orange-500/30', label: 'Deciding', animate: true },
    success: { color: 'bg-emerald-500', text: 'text-emerald-400', border: 'border-emerald-500/30', label: 'Success' },
    failed: { color: 'bg-red-500', text: 'text-red-400', border: 'border-red-500/30', label: 'Failed' },
    branching: { color: 'bg-cyan-500', text: 'text-cyan-400', border: 'border-cyan-500/30', label: 'Branching', animate: true },
    stopped: { color: 'bg-yellow-500', text: 'text-yellow-400', border: 'border-yellow-500/30', label: 'Stopped' },
    decided: { color: 'bg-orange-500', text: 'text-orange-400', border: 'border-orange-500/30', label: 'Advancing...', animate: true },
};

export function getStatusStyle(status: string): StatusStyle {
    return STATUS_MAP[status] ?? STATUS_MAP.initialized;
}

export function formatTime(us: number | null): string {
    if (us === null || us === undefined) return '—';
    if (us < 1) return `${(us * 1000).toFixed(1)} ns`;
    if (us < 1000) return `${us.toFixed(2)} µs`;
    return `${(us / 1000).toFixed(2)} ms`;
}

export function formatSpeedup(s: number | null): string {
    if (s === null || s === undefined) return '—';
    return `${s.toFixed(2)}x`;
}

export function formatTokens(n: number): string {
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
    return String(n);
}
