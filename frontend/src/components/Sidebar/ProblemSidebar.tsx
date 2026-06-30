import { useAppStore } from '@/store/appStore';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { NewProblemDialog } from './NewProblemDialog';
import { CloneDialog } from './CloneDialog';
import { ChevronLeft, Trash2, Edit2, Copy } from 'lucide-react';
import { deleteProblem } from '@/api/client';
import { refreshProblems } from '@/api/hooks';
import { toast } from 'sonner';

import { useState } from 'react';

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
                <div className="p-2 space-y-1.5">
                    {problems.map((problem) => {
                        const isActive = problem.name === activeProblem;
                        return (
                            <Card
                                key={problem.name}
                                onClick={() => setActiveProblem(problem.name)}
                                className={`p-3 cursor-pointer transition-all duration-150 relative group ${isActive
                                    ? 'border-primary bg-primary/5 shadow-sm'
                                    : 'hover:bg-accent border-transparent'
                                    }`}
                            >
                                <div className="flex items-center justify-between gap-1">
                                    <span className={`text-sm font-medium truncate ${isActive ? 'text-primary' : ''}`} title={problem.name}>
                                        {problem.name}
                                    </span>
                                    <Badge
                                        variant="outline"
                                        className={`text-[10px] px-1.5 py-0 ${problem.status === 'running' ? 'text-amber-400 border-amber-500/30' :
                                            problem.status === 'completed' ? 'text-emerald-400 border-emerald-500/30' :
                                                'text-zinc-400 border-zinc-500/30'
                                            }`}
                                    >
                                        {problem.status}
                                    </Badge>
                                </div>
                                {problem.description && (
                                    <p className="text-[11px] text-muted-foreground mt-1 pr-6">{problem.description}</p>
                                )}
                                <div className="absolute bottom-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity flex gap-1">
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
                            </Card>
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
