import { useAppStore } from '@/store/appStore';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { NewProblemDialog } from './NewProblemDialog';
import { CloneDialog } from './CloneDialog';
import { ChevronLeft, Trash2, Edit2, Copy } from 'lucide-react';
import { deleteProblem } from '@/api/client';
import { refreshProblems } from '@/api/hooks';
import { toast } from 'sonner';

import { useState } from 'react';

// The problem list has its own three states, none of them an iteration phase.
const PROBLEM_TONE: Record<string, { text: string; dot: string; animate?: boolean }> = {
    running: { text: 'text-amber-400', dot: 'bg-amber-400', animate: true },
    completed: { text: 'text-emerald-400', dot: 'bg-emerald-400' },
};
const IDLE_TONE = { text: 'text-muted-foreground', dot: 'bg-muted-foreground/50' };

export function ProblemSidebar() {
    const problems = useAppStore((s) => s.problems);
    const activeProblem = useAppStore((s) => s.activeProblem);
    const setActiveProblem = useAppStore((s) => s.setActiveProblem);
    const sidebarOpen = useAppStore((s) => s.sidebarOpen);
    const toggleSidebar = useAppStore((s) => s.toggleSidebar);
    const [cloneTarget, setCloneTarget] = useState<string | null>(null);

    async function handleDelete(e: React.MouseEvent, name: string) {
        e.stopPropagation();
        if (confirm(`Are you sure you want to delete problem '${name}'? This cannot be undone.`)) {
            try {
                await deleteProblem(name);
                if (activeProblem === name) {
                    setActiveProblem(null);
                }
                await refreshProblems();
            } catch (err) {
                toast.error(err instanceof Error ? err.message : 'Failed to delete problem');
            }
        }
    }

    if (!sidebarOpen) return null;

    return (
        <div className="w-80 border-r bg-card flex flex-col h-full min-h-0 overflow-hidden">
            {/* Header */}
            <div className="p-3 flex items-center justify-between border-b">
                <h2 className="font-semibold text-sm">Problems</h2>
                <button onClick={toggleSidebar} className="text-muted-foreground hover:text-foreground transition-colors cursor-pointer">
                    <ChevronLeft size={16} />
                </button>
            </div>

            {/* Problem list */}
            <ScrollArea className="flex-1 min-h-0">
                <div className="p-2 space-y-0.5">
                    {problems.map((problem) => {
                        const isActive = problem.name === activeProblem;
                        const tone = PROBLEM_TONE[problem.status] ?? IDLE_TONE;
                        return (
                            <div
                                key={problem.name}
                                onClick={() => setActiveProblem(problem.name)}
                                className={`group relative cursor-pointer rounded-md px-2.5 py-2 transition-colors ${
                                    isActive ? 'bg-accent' : 'hover:bg-accent/50'
                                }`}
                            >
                                <div className="flex items-center justify-between gap-2 min-w-0">
                                    <span className={`truncate text-sm ${isActive ? 'font-medium' : ''}`} title={problem.name}>
                                        {problem.name}
                                    </span>
                                    <span className={`flex shrink-0 items-center gap-1 text-[10px] ${tone.text}`}>
                                        <span
                                            className={`inline-block size-1.5 rounded-full ${tone.dot} ${
                                                tone.animate ? 'animate-pulse' : ''
                                            }`}
                                        />
                                        {problem.status}
                                    </span>
                                </div>
                                {problem.description && (
                                    <p
                                        className="mt-0.5 line-clamp-2 break-words pr-16 text-[11px] text-muted-foreground"
                                        title={problem.description}
                                    >
                                        {problem.description}
                                    </p>
                                )}
                                <div className="absolute bottom-1.5 right-1.5 flex gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
                                    <div onClick={(e) => e.stopPropagation()}>
                                        <NewProblemDialog
                                            mode="edit"
                                            editProblemName={problem.name}
                                            trigger={
                                                <button
                                                    className="text-muted-foreground hover:text-primary p-1"
                                                    title="Edit problem"
                                                >
                                                    <Edit2 size={13} />
                                                </button>
                                            }
                                        />
                                    </div>
                                    <button
                                        onClick={(e) => { e.stopPropagation(); setCloneTarget(problem.name); }}
                                        className="text-muted-foreground hover:text-primary p-1 cursor-pointer"
                                        title="Clone problem"
                                    >
                                        <Copy size={13} />
                                    </button>
                                    <button
                                        onClick={(e) => handleDelete(e, problem.name)}
                                        className="text-muted-foreground hover:text-destructive p-1 cursor-pointer"
                                        title="Delete problem"
                                    >
                                        <Trash2 size={13} />
                                    </button>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </ScrollArea>

            <Separator />

            {/* New problem button */}
            <div className="p-2">
                <NewProblemDialog />
            </div>

            <CloneDialog
                sourceName={cloneTarget ?? ""}
                open={cloneTarget !== null}
                onOpenChange={(v) => { if (!v) setCloneTarget(null); }}
            />
        </div>
    );
}
