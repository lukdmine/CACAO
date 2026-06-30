import { useState } from 'react';
import { useAppStore } from '@/store/appStore';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { runProblem, resumeProblem, stopProblem } from '@/api/client';
import { refreshTree } from '@/api/hooks';
import { toast } from 'sonner';
import { Play, RotateCcw, PanelLeft, Cpu, Wifi, WifiOff, Square, Loader2, Coins, Timer } from 'lucide-react';

export function Toolbar() {
    const activeProblem = useAppStore((s) => s.activeProblem);
    const runStatus = useAppStore((s) => s.runStatus);
    const sidebarOpen = useAppStore((s) => s.sidebarOpen);
    const toggleSidebar = useAppStore((s) => s.toggleSidebar);
    const treeNodes = useAppStore((s) => s.treeNodes);
    const connected = useAppStore((s) => s.connected);
    const setRunStatus = useAppStore((s) => s.setRunStatus);
    const llmModel = useAppStore((s) => s.llmModel);
    const llmProvider = useAppStore((s) => s.llmProvider);
    const tokenUsage = useAppStore((s) => s.tokenUsage);
    const availableModels = useAppStore((s) => s.availableModels);
    const selectedProvider = useAppStore((s) => s.selectedProvider);
    const selectedModel = useAppStore((s) => s.selectedModel);
    const setSelectedProvider = useAppStore((s) => s.setSelectedProvider);
    const setSelectedModel = useAppStore((s) => s.setSelectedModel);
    const problemTuningDurationS = useAppStore((s) => s.problemTuningDurationS);

    const [isSubmitting, setIsSubmitting] = useState(false);
    const [isStopping, setIsStopping] = useState(false);

    // Timeout override is tagged by the problem it was entered for, so switching
    // problems automatically reverts the input to that problem's yaml default —
    // no effect needed.
    const [timeoutOverride, setTimeoutOverride] = useState<{ problem: string | null; value: string }>({ problem: null, value: '' });
    const overrideActive = timeoutOverride.problem === activeProblem;
    const timeoutDisplay = overrideActive
        ? timeoutOverride.value
        : (problemTuningDurationS != null ? String(problemTuningDurationS) : '');

    const branchCount = treeNodes.filter((n) => n.id !== 'root').length;
    const hasExistingRun = branchCount > 0;

    const parsedTimeout = (() => {
        const t = timeoutDisplay.trim();
        if (!t) return null;
        const n = Number(t);
        return Number.isFinite(n) && n > 0 ? Math.round(n) : null;
    })();
    const sendTimeout = overrideActive && parsedTimeout != null && parsedTimeout !== problemTuningDurationS;

    const runConfig = {
        ...(selectedProvider && { provider: selectedProvider }),
        ...(selectedModel && { model: selectedModel }),
        ...(sendTimeout && { timeout: parsedTimeout }),
    };

    async function handleRun() {
        if (!activeProblem || isSubmitting) return;
        setIsSubmitting(true);
        try {
            await runProblem(activeProblem, runConfig);
            setRunStatus('running');
            await refreshTree();
        } catch (err) {
            toast.error('Failed to start optimization');
            console.error(err);
        } finally {
            setIsSubmitting(false);
        }
    }

    async function handleResume() {
        if (!activeProblem || isSubmitting) return;
        setIsSubmitting(true);
        try {
            await resumeProblem(activeProblem, runConfig);
            setRunStatus('running');
            await refreshTree();
        } catch (err) {
            toast.error('Failed to resume optimization');
            console.error(err);
        } finally {
            setIsSubmitting(false);
        }
    }

    const providers = availableModels ? Object.keys(availableModels) : [];
    const models = (selectedProvider && availableModels?.[selectedProvider]?.available) || [];

    return (
        <div className="h-12 border-b bg-card flex items-center px-3 gap-3 shrink-0">
            {/* Sidebar toggle */}
            {!sidebarOpen && (
                <button onClick={toggleSidebar} className="text-muted-foreground hover:text-foreground transition-colors cursor-pointer">
                    <PanelLeft size={18} />
                </button>
            )}

            {/* Problem name */}
            <div className="flex items-center gap-2">
                <Cpu size={16} className="text-muted-foreground" />
                <span className="font-semibold text-sm">{activeProblem ?? 'No problem selected'}</span>
            </div>

            {/* Model selection */}
            {activeProblem && runStatus !== 'running' && availableModels && (
                <div className="flex items-center gap-1.5">
                    <Select value={selectedProvider ?? ''} onValueChange={setSelectedProvider}>
                        <SelectTrigger size="sm" className="h-7 text-xs min-w-[90px]">
                            <SelectValue placeholder="Provider" />
                        </SelectTrigger>
                        <SelectContent>
                            {providers.map((p) => (
                                <SelectItem key={p} value={p} className="text-xs">{p}</SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                    <Select value={selectedModel ?? ''} onValueChange={setSelectedModel}>
                        <SelectTrigger size="sm" className="h-7 text-xs min-w-[140px]">
                            <SelectValue placeholder="Model" />
                        </SelectTrigger>
                        <SelectContent>
                            {models.map((m) => (
                                <SelectItem key={m} value={m} className="text-xs">{m}</SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                    <div
                        className="flex items-center gap-1.5"
                        title={
                            problemTuningDurationS != null
                                ? `Tuner budget (seconds). Prefilled from problem.yaml tuning.duration_s = ${problemTuningDurationS}.`
                                : 'Tuner budget (seconds). Not set in problem.yaml — leave blank to use the system default.'
                        }
                    >
                        <Timer size={13} className="text-muted-foreground shrink-0" />
                        <Input
                            type="number"
                            min={1}
                            value={timeoutDisplay}
                            onChange={(e) =>
                                setTimeoutOverride({ problem: activeProblem, value: e.target.value })
                            }
                            placeholder={problemTuningDurationS != null ? String(problemTuningDurationS) : 'default'}
                            className="h-7 text-xs w-[110px]"
                        />
                        <span className="text-[10px] text-muted-foreground shrink-0">s</span>
                    </div>
                </div>
            )}

            {/* Run / Resume buttons */}
            {activeProblem && runStatus !== 'running' && (
                <div className="flex items-center gap-1.5">
                    <Button
                        size="sm"
                        className="text-xs h-7"
                        onClick={handleRun}
                        disabled={isSubmitting || !connected}
                    >
                        {hasExistingRun ? <RotateCcw size={12} className="mr-1" /> : <Play size={12} className="mr-1" />}
                        {hasExistingRun ? 'Rerun' : 'Run'}
                    </Button>
                    {hasExistingRun && (
                        <Button
                            size="sm"
                            variant="outline"
                            className="text-xs h-7"
                            onClick={handleResume}
                            disabled={isSubmitting || !connected}
                        >
                            <Play size={12} className="mr-1" />
                            Resume
                        </Button>
                    )}
                </div>
            )}
            {runStatus === 'running' && (
                <div className="flex items-center gap-1.5">
                    {(llmProvider || llmModel) && (
                        <Badge variant="secondary" className="text-xs font-normal">
                            {[llmProvider, llmModel].filter(Boolean).join(' · ')}
                        </Badge>
                    )}
                    <Badge variant="outline" className="text-xs text-amber-400 border-amber-500/30">
                        <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse mr-1.5 inline-block" />
                        Running...
                    </Badge>
                    <Button
                        size="sm"
                        variant="destructive"
                        className="text-xs h-7"
                        disabled={isStopping}
                        onClick={async () => {
                            if (!activeProblem || isStopping) return;
                            setIsStopping(true);
                            try {
                                await stopProblem(activeProblem);
                                setRunStatus('idle');
                                await refreshTree();
                            } catch (err) {
                                toast.error('Failed to stop optimization');
                                console.error(err);
                            } finally {
                                setIsStopping(false);
                            }
                        }}
                    >
                        {isStopping
                            ? <><Loader2 size={10} className="mr-1 animate-spin" />Stopping...</>
                            : <><Square size={10} className="mr-1" />Stop</>
                        }
                    </Button>
                </div>
            )}

            {/* Status badge */}
            {runStatus === 'completed' && (
                <Badge variant="outline" className="text-xs text-emerald-400 border-emerald-500/30">
                    completed
                </Badge>
            )}

            {/* Spacer */}
            <div className="flex-1" />

            {/* Stats */}
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
                {connected && tokenUsage.total_tokens > 0 && (
                    <>
                        <div className="flex items-center gap-1 text-muted-foreground/70" title={`API calls: ${tokenUsage.api_calls.toLocaleString()}\nPrompt: ${tokenUsage.prompt_tokens.toLocaleString()}\nCompletion: ${tokenUsage.completion_tokens.toLocaleString()}`}>
                            <Coins size={13} />
                            <span>{tokenUsage.total_tokens.toLocaleString()} tokens</span>
                        </div>
                        <Separator orientation="vertical" className="h-4" />
                    </>
                )}
                <div className="flex items-center gap-1">
                    {connected ? (
                        <>
                            <Wifi size={13} className="text-emerald-400" />
                            <span className="text-emerald-400">Connected</span>
                        </>
                    ) : (
                        <>
                            <WifiOff size={13} className="text-zinc-500" />
                            <span className="text-zinc-500">Connecting...</span>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
