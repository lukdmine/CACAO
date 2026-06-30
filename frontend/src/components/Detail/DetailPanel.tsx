import { useState } from 'react';
import { useSelectedNode } from '@/store/appStore';
import { useAppStore } from '@/store/appStore';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { getStatusStyle, formatTime, formatSpeedup } from '@/utils/statusColors';
import { stopBranch, resumeBranch, messageBranch, changeDecision, configureBranch, deleteBranch } from '@/api/client';
import { refreshTree } from '@/api/hooks';
import { toast } from 'sonner';
import { X, Square, Play, MessageSquare, Send, Settings2, Timer, Zap, CheckCircle2, XCircle, Trash2, RefreshCw, AlertTriangle, Loader2 } from 'lucide-react';

function branchIdFromNodeId(nodeId: string): string {
    return nodeId.replace(/^branch\//, '');
}

export function DetailPanel() {
    const node = useSelectedNode();
    const selectNode = useAppStore((s) => s.selectNode);
    const activeProblem = useAppStore((s) => s.activeProblem);
    const runStatus = useAppStore((s) => s.runStatus);
    const treeNodes = useAppStore((s) => s.treeNodes);
    const isRunning = runStatus === 'running';

    const [messageText, setMessageText] = useState('');
    const [isSending, setIsSending] = useState(false);

    const [changeIter, setChangeIter] = useState<number | null>(null);
    const [changeMessage, setChangeMessage] = useState('');
    const [isChanging, setIsChanging] = useState(false);

    const [editingMaxIter, setEditingMaxIter] = useState(false);
    const [maxIterValue, setMaxIterValue] = useState('');

    const [isStopping, setIsStopping] = useState(false);
    const [isResuming, setIsResuming] = useState(false);
    const [isSavingMaxIter, setIsSavingMaxIter] = useState(false);

    const [confirmDelete, setConfirmDelete] = useState(false);
    const [isDeleting, setIsDeleting] = useState(false);

    if (!node) {
        return (
            <div className="h-full flex items-center justify-center text-muted-foreground text-sm p-6">
                <p className="text-center">Click a node in the tree to view details</p>
            </div>
        );
    }

    const style = getStatusStyle(node.status);
    const isRoot = node.id === 'root';
    const branchId = isRoot ? '' : branchIdFromNodeId(node.id);
    const isActive = !isRoot && !['success', 'failed', 'branching'].includes(node.status);
    const isStopped = node.status === 'stopped';

    async function handleStop() {
        if (!activeProblem || !branchId || isStopping) return;
        setIsStopping(true);
        try {
            await stopBranch(activeProblem, branchId);
            await refreshTree();
        } catch (e) { toast.error('Failed to stop branch'); console.error(e); }
        finally { setIsStopping(false); }
    }

    async function handleResume() {
        if (!activeProblem || !branchId || isResuming) return;
        setIsResuming(true);
        try {
            await resumeBranch(activeProblem, branchId);
            await refreshTree();
        } catch (e) { toast.error('Failed to resume branch'); console.error(e); }
        finally { setIsResuming(false); }
    }

    async function handleSendMessage() {
        if (!activeProblem || !branchId || !messageText.trim()) return;
        setIsSending(true);
        try {
            await messageBranch(activeProblem, branchId, messageText.trim());
            setMessageText('');
            await refreshTree();
        } catch (e) { toast.error('Failed to send message'); console.error(e); }
        finally { setIsSending(false); }
    }

    async function handleChangeDecision() {
        if (!activeProblem || !branchId || changeIter === null) return;
        setIsChanging(true);
        try {
            await changeDecision(activeProblem, branchId, changeIter, changeMessage.trim() || undefined);
            setChangeIter(null);
            setChangeMessage('');
            await refreshTree();
        } catch (e) { toast.error('Failed to change decision'); console.error(e); }
        finally { setIsChanging(false); }
    }

    const isChangeRevert = changeIter !== null && changeIter < (node?.iter_num ?? 0);

    async function handleSaveMaxIter() {
        if (!activeProblem || !branchId || isSavingMaxIter) return;
        const val = parseInt(maxIterValue, 10);
        if (isNaN(val) || val < 1) return;
        setIsSavingMaxIter(true);
        try {
            await configureBranch(activeProblem, branchId, { max_iter: val });
            setEditingMaxIter(false);
            await refreshTree();
        } catch (e) { toast.error('Failed to update max iterations'); console.error(e); }
        finally { setIsSavingMaxIter(false); }
    }

    const hasChildren = treeNodes.some((n) => n.parentId === node.id);
    const isDeletableLeaf = !isRoot && !hasChildren;

    async function handleDelete() {
        if (!activeProblem || !branchId) return;
        setIsDeleting(true);
        try {
            await deleteBranch(activeProblem, branchId);
            setConfirmDelete(false);
            selectNode(null);
            await refreshTree();
        } catch (e) { toast.error('Failed to delete branch'); console.error(e); }
        finally { setIsDeleting(false); }
    }

    return (
        <ScrollArea className="h-full">
            <div className="p-4 space-y-4">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <h2 className="font-bold text-lg">{node.strategy.name}</h2>
                    <button onClick={() => selectNode(null)} className="text-muted-foreground hover:text-foreground transition-colors cursor-pointer">
                        <X size={18} />
                    </button>
                </div>

                {/* Status + metrics */}
                <div className="flex items-center gap-2 flex-wrap">
                    <Badge variant="outline" className={`${style.text} ${style.border}`}>
                        <span className={`w-2 h-2 rounded-full ${style.color} mr-1.5 inline-block ${style.animate ? 'animate-pulse' : ''}`} />
                        {style.label}
                    </Badge>
                    {node.iter_num > 0 && (
                        <Badge variant="secondary" className="text-xs">
                            Iter {node.iter_num}/{node.max_iter}
                        </Badge>
                    )}
                    {node.best_time_us !== null && (
                        <Badge variant="secondary" className="text-xs font-mono">
                            {formatTime(node.best_time_us)}
                        </Badge>
                    )}
                    {node.speedup !== null && (
                        <Badge variant="secondary" className="text-xs font-mono text-emerald-400">
                            {formatSpeedup(node.speedup)}
                        </Badge>
                    )}
                </div>

                {/* Branch controls */}
                {!isRoot && (
                    <div className="flex items-center gap-1.5 flex-wrap">
                        {isActive && !isStopped && (
                            <Button size="sm" variant="outline" className="text-xs h-7" onClick={handleStop} disabled={isStopping}>
                                {isStopping ? <Loader2 size={10} className="mr-1 animate-spin" /> : <Square size={10} className="mr-1" />}
                                {isStopping ? 'Stopping...' : 'Stop'}
                            </Button>
                        )}
                        {isStopped && (
                            <Button size="sm" variant="outline" className="text-xs h-7 text-emerald-400 border-emerald-500/30" onClick={handleResume} disabled={isResuming}>
                                {isResuming ? <Loader2 size={10} className="mr-1 animate-spin" /> : <Play size={10} className="mr-1" />}
                                {isResuming ? 'Resuming...' : 'Resume'}
                            </Button>
                        )}
                        <Button
                            size="sm"
                            variant="ghost"
                            className="text-xs h-7"
                            onClick={() => { setEditingMaxIter(true); setMaxIterValue(String(node.max_iter)); }}
                        >
                            <Settings2 size={10} className="mr-1" />
                            Max iter: {node.max_iter}
                        </Button>
                        {isDeletableLeaf && (
                            <div title={isRunning ? "Stop the optimizer before deleting branches" : "Delete branch"} className="inline-block cursor-help">
                                <Button
                                    size="sm"
                                    variant="ghost"
                                    className="text-xs h-7 text-red-400 hover:text-red-300 hover:bg-red-500/10"
                                    disabled={isRunning}
                                    style={isRunning ? { pointerEvents: "none" } : {}}
                                    onClick={() => setConfirmDelete(true)}
                                >
                                    <Trash2 size={10} className="mr-1" />
                                    Delete
                                </Button>
                            </div>
                        )}
                    </div>
                )}

                {/* Max iter editor dialog */}
                <Dialog open={editingMaxIter} onOpenChange={setEditingMaxIter}>
                    <DialogContent className="sm:max-w-sm">
                        <DialogHeader>
                            <DialogTitle>Change Iteration Limit</DialogTitle>
                            <DialogDescription>Set the maximum iterations for this branch.</DialogDescription>
                        </DialogHeader>
                        <input
                            type="number"
                            min={1}
                            value={maxIterValue}
                            onChange={(e) => setMaxIterValue(e.target.value)}
                            className="w-full rounded border bg-muted px-3 py-2 text-sm font-mono"
                            onKeyDown={(e) => e.key === 'Enter' && handleSaveMaxIter()}
                        />
                        <DialogFooter>
                            <Button variant="outline" size="sm" onClick={() => setEditingMaxIter(false)}>Cancel</Button>
                            <Button size="sm" onClick={handleSaveMaxIter} disabled={isSavingMaxIter}>
                                {isSavingMaxIter ? 'Saving...' : 'Save'}
                            </Button>
                        </DialogFooter>
                    </DialogContent>
                </Dialog>

                {/* Message input */}
                {!isRoot && (isActive || isStopped) && (
                    <div className="flex gap-1.5">
                        <input
                            value={messageText}
                            onChange={(e) => setMessageText(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                            placeholder="Send a message to this branch..."
                            className="flex-1 rounded border bg-muted px-3 py-1.5 text-xs placeholder:text-muted-foreground/50"
                        />
                        <Button size="sm" variant="outline" className="h-7 px-2" onClick={handleSendMessage} disabled={isSending || !messageText.trim()}>
                            <Send size={12} />
                        </Button>
                    </div>
                )}

                {/* User messages display */}
                {(node.user_messages?.length ?? 0) > 0 && (
                    <>
                        <Separator />
                        <div>
                            <span className="text-xs font-medium text-muted-foreground mb-1.5 block flex items-center gap-1">
                                <MessageSquare size={12} /> User Messages
                            </span>
                            <div className="space-y-1">
                                {node.user_messages!.map((m, i) => (
                                    <div key={i} className="text-xs p-2 bg-blue-950/30 border border-blue-500/20 rounded">
                                        <span className="text-muted-foreground">iter {m.iter_num}:</span>{' '}
                                        <span>{m.content}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </>
                )}

                <Separator />

                {/* Strategy info */}
                <div className="space-y-2 text-sm">
                    <div>
                        <span className="text-muted-foreground text-xs font-medium">Description</span>
                        <p className="mt-0.5">{node.strategy.description}</p>
                    </div>
                    <div>
                        <span className="text-muted-foreground text-xs font-medium">Hypothesis</span>
                        <p className="mt-0.5">{node.strategy.hypothesis}</p>
                    </div>
                    {node.strategy.key_parameters.length > 0 && (
                        <div>
                            <span className="text-muted-foreground text-xs font-medium">Parameters</span>
                            <div className="flex flex-wrap gap-1 mt-1">
                                {node.strategy.key_parameters.map((p) => (
                                    <Badge key={p} variant="outline" className="text-xs font-mono">{p}</Badge>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                {/* Iteration accordion */}
                {node.iterations.length > 0 && (
                    <>
                        <Separator />
                        <div>
                            <span className="text-xs font-medium text-muted-foreground mb-2 block">Iterations</span>
                            <Accordion type="single" collapsible defaultValue={`iter-${node.iterations[node.iterations.length - 1].iter_num}`}>
                                {node.iterations.map((iter) => (
                                    <AccordionItem
                                        key={iter.iter_num}
                                        value={`iter-${iter.iter_num}`}
                                        className="data-[state=open]:pl-3 data-[state=open]:pr-1"
                                    >
                                        <AccordionTrigger className="text-sm py-2">
                                            <div className="flex items-center gap-2">
                                                <span>Iteration {iter.iter_num}</span>
                                                {iter.decision && (
                                                    <Badge
                                                        variant="outline"
                                                        className={`text-[10px] px-1.5 py-0 ${iter.decision.action === 'stop' ? 'text-emerald-400 border-emerald-500/30' :
                                                            iter.decision.action === 'retry' ? 'text-red-400 border-red-500/30' :
                                                                iter.decision.action === 'branch' ? 'text-cyan-400 border-cyan-500/30' :
                                                                    'text-blue-400 border-blue-500/30'
                                                            }`}
                                                    >
                                                        {iter.decision.action}
                                                    </Badge>
                                                )}
                                                {iter.results_summary?.best_time_us != null && (
                                                    <Badge variant="secondary" className="text-[10px] px-1.5 py-0 font-mono">
                                                        <Timer size={10} className="mr-0.5" />
                                                        {formatTime(iter.results_summary.best_time_us)}
                                                    </Badge>
                                                )}
                                                {iter.results_summary?.speedup != null && (
                                                    <Badge variant="secondary" className="text-[10px] px-1.5 py-0 font-mono text-emerald-400">
                                                        <Zap size={10} className="mr-0.5" />
                                                        {formatSpeedup(iter.results_summary.speedup)}
                                                    </Badge>
                                                )}
                                            </div>
                                        </AccordionTrigger>
                                        <AccordionContent>
                                            <div className="ml-1 border-l-2 border-primary/30 pl-3">
                                                {((!isRoot && iter.decision) || iter.results_summary) && (
                                                    <div className="space-y-2 mb-3">
                                                        {!isRoot && (iter.decision || iter.status === 'decided') && (
                                                            <div className="flex items-start justify-start">
                                                                <Button
                                                                    size="sm"
                                                                    variant="outline"
                                                                    className="text-xs h-6 text-amber-400 border-amber-500/30 hover:bg-amber-500/10"
                                                                    onClick={() => { setChangeIter(iter.iter_num); setChangeMessage(''); }}
                                                                >
                                                                    <RefreshCw size={10} className="mr-1" />
                                                                    {iter.decision ? 'Change Decision' : 'Retry Iteration'}
                                                                </Button>
                                                            </div>
                                                        )}

                                                        {iter.results_summary && (
                                                            <div className={`w-full p-2.5 rounded border ${iter.results_summary.has_success
                                                                ? 'bg-emerald-950/20 border-emerald-500/20'
                                                                : iter.results_summary.num_total > 0
                                                                    ? 'bg-red-950/20 border-red-500/20'
                                                                    : 'bg-muted border-border'
                                                                }`}>
                                                                <div className="flex items-center gap-1.5 mb-2">
                                                                    {iter.results_summary.has_success
                                                                        ? <CheckCircle2 size={13} className="text-emerald-400" />
                                                                        : iter.results_summary.num_total > 0
                                                                            ? <XCircle size={13} className="text-red-400" />
                                                                            : null
                                                                    }
                                                                    <span className="font-medium">Tuning Results</span>
                                                                    <span className="text-muted-foreground ml-auto">
                                                                        {iter.results_summary.num_successful}/{iter.results_summary.num_total} configs passed
                                                                    </span>
                                                                </div>
                                                                {iter.results_summary.best_time_us != null && (
                                                                    <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
                                                                        <div>
                                                                            <span className="text-muted-foreground">Best time:</span>{' '}
                                                                            <span className="font-mono font-medium">{formatTime(iter.results_summary.best_time_us)}</span>
                                                                        </div>
                                                                        {iter.results_summary.reference_time_us != null && (
                                                                            <div>
                                                                                <span className="text-muted-foreground">Ref time:</span>{' '}
                                                                                <span className="font-mono">{formatTime(iter.results_summary.reference_time_us)}</span>
                                                                            </div>
                                                                        )}
                                                                        {iter.results_summary.speedup != null && (
                                                                            <div>
                                                                                <span className="text-muted-foreground">Speedup:</span>{' '}
                                                                                <span className={`font-mono font-medium ${iter.results_summary.speedup >= 1 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                                                    {formatSpeedup(iter.results_summary.speedup)}
                                                                                </span>
                                                                            </div>
                                                                        )}
                                                                    </div>
                                                                )}
                                                                {iter.results_summary.best_config && (
                                                                    <div className="mt-1.5 flex flex-wrap gap-1">
                                                                        {Object.entries(iter.results_summary.best_config).map(([k, v]) => (
                                                                            <Badge key={k} variant="outline" className="text-[10px] font-mono px-1.5 py-0">
                                                                                {k}={v}
                                                                            </Badge>
                                                                        ))}
                                                                    </div>
                                                                )}
                                                            </div>
                                                        )}
                                                    </div>
                                                )}

                                                <Accordion type="multiple" className="text-xs">
                                                    {iter.plan && iter.iter_num === 1 && (
                                                        <AccordionItem value={`iter-${iter.iter_num}-plan`}>
                                                            <AccordionTrigger className="py-1.5 text-xs">Plan</AccordionTrigger>
                                                            <AccordionContent>
                                                                <div className="border-l border-primary/20 pl-3">
                                                                    <pre className="mt-1 p-2 bg-muted rounded text-[11px] whitespace-pre-wrap overflow-x-auto">{iter.plan}</pre>
                                                                </div>
                                                            </AccordionContent>
                                                        </AccordionItem>
                                                    )}

                                                    {iter.kernel_code && (
                                                        <AccordionItem value={`iter-${iter.iter_num}-kernel`}>
                                                            <AccordionTrigger className="py-1.5 text-xs">Kernel Code</AccordionTrigger>
                                                            <AccordionContent>
                                                                <div className="border-l border-primary/20 pl-3">
                                                                    <pre className="mt-1 p-2 bg-zinc-900 rounded text-[11px] text-green-300 whitespace-pre-wrap overflow-x-auto font-mono">{iter.kernel_code}</pre>
                                                                </div>
                                                            </AccordionContent>
                                                        </AccordionItem>
                                                    )}

                                                    {iter.params_json && iter.params_json !== '{}' && (
                                                        <AccordionItem value={`iter-${iter.iter_num}-params`}>
                                                            <AccordionTrigger className="py-1.5 text-xs">Parameters</AccordionTrigger>
                                                            <AccordionContent>
                                                                <div className="border-l border-primary/20 pl-3">
                                                                    <pre className="mt-1 p-2 bg-muted rounded text-[11px] whitespace-pre-wrap overflow-x-auto font-mono">{iter.params_json}</pre>
                                                                </div>
                                                            </AccordionContent>
                                                        </AccordionItem>
                                                    )}

                                                    {iter.run_output && (
                                                        <AccordionItem value={`iter-${iter.iter_num}-output`}>
                                                            <AccordionTrigger className="py-1.5 text-xs">Run Output</AccordionTrigger>
                                                            <AccordionContent>
                                                                <div className="border-l border-primary/20 pl-3">
                                                                    <pre className={`mt-1 p-2 rounded text-[11px] whitespace-pre-wrap overflow-x-auto font-mono ${iter.run_output.includes('ERROR') ? 'bg-red-950/50 text-red-300' : 'bg-zinc-900 text-zinc-300'}`}>
                                                                        {iter.run_output}
                                                                    </pre>
                                                                </div>
                                                            </AccordionContent>
                                                        </AccordionItem>
                                                    )}

                                                    {iter.ncu_metrics && Object.keys(iter.ncu_metrics).length > 0 && (
                                                        <AccordionItem value={`iter-${iter.iter_num}-ncu`}>
                                                            <AccordionTrigger className="py-1.5 text-xs">NCU Metrics</AccordionTrigger>
                                                            <AccordionContent>
                                                                <div className="border-l border-primary/20 pl-3">
                                                                    <div className="mt-1 overflow-hidden">
                                                                        <table className="text-[11px]">
                                                                            <tbody>
                                                                                {Object.entries(iter.ncu_metrics).map(([key, val]) => (
                                                                                    <tr key={key} className="border-b last:border-0">
                                                                                        <td className="py-1 px-2 text-muted-foreground font-mono truncate max-w-[380px]">{key}</td>
                                                                                        <td className="py-1 px-2 font-mono whitespace-nowrap">{String(val)}</td>
                                                                                    </tr>
                                                                                ))}
                                                                            </tbody>
                                                                        </table>
                                                                    </div>
                                                                </div>
                                                            </AccordionContent>
                                                        </AccordionItem>
                                                    )}

                                                    {iter.proposal && (
                                                        <AccordionItem value={`iter-${iter.iter_num}-proposal`}>
                                                            <AccordionTrigger className="py-1.5 text-xs">Optimization Proposal</AccordionTrigger>
                                                            <AccordionContent>
                                                                <div className="border-l border-pink-500/30 pl-3">
                                                                    <pre className="mt-1 p-2 bg-pink-950/20 border border-pink-500/20 rounded text-[11px] whitespace-pre-wrap overflow-x-auto text-pink-100">{iter.proposal}</pre>
                                                                </div>
                                                            </AccordionContent>
                                                        </AccordionItem>
                                                    )}

                                                    {iter.decision && (
                                                        <AccordionItem value={`iter-${iter.iter_num}-decision`}>
                                                            <AccordionTrigger className="py-1.5 text-xs">Decision</AccordionTrigger>
                                                            <AccordionContent>
                                                                <div className="border-l border-primary/20 pl-3">
                                                                    <div className="mt-1 p-2 bg-muted rounded space-y-1">
                                                                        <div><span className="text-muted-foreground">Action:</span> <span className="font-medium">{iter.decision.action}</span></div>
                                                                        <div><span className="text-muted-foreground">Reasoning:</span> {iter.decision.reasoning}</div>
                                                                        {iter.decision.feedback && <div><span className="text-muted-foreground">Feedback:</span> {iter.decision.feedback}</div>}
                                                                        {iter.decision.error_analysis && (
                                                                            <div className="mt-1 p-2 bg-red-950/30 rounded border border-red-500/20">
                                                                                <div className="font-medium text-red-400 mb-1">Error Analysis</div>
                                                                                <div><span className="text-muted-foreground">Type:</span> {iter.decision.error_analysis.error_type}</div>
                                                                                <div><span className="text-muted-foreground">Cause:</span> {iter.decision.error_analysis.root_cause}</div>
                                                                                <div><span className="text-muted-foreground">Fix:</span> {iter.decision.error_analysis.suggested_fix}</div>
                                                                            </div>
                                                                        )}
                                                                    </div>
                                                                </div>
                                                            </AccordionContent>
                                                        </AccordionItem>
                                                    )}
                                                </Accordion>
                                            </div>
                                        </AccordionContent>
                                    </AccordionItem>
                                ))}
                            </Accordion>
                        </div>
                    </>
                )}
            </div>

            {/* Change Decision dialog */}
            <Dialog open={changeIter !== null} onOpenChange={(open) => { if (!open) setChangeIter(null); }}>
                <DialogContent className="sm:max-w-md">
                    <DialogHeader>
                        <DialogTitle>
                            {isChangeRevert
                                ? `Revert to Iteration ${changeIter}`
                                : `Continue from Iteration ${changeIter}`
                            }
                        </DialogTitle>
                        <DialogDescription>
                            {isChangeRevert
                                ? `This will delete all work after iteration ${changeIter} (including sub-branches) and let the agent re-decide from that point.`
                                : 'This will override the agent\'s decision and continue optimizing this branch.'
                            }
                        </DialogDescription>
                    </DialogHeader>
                    {isChangeRevert && (
                        <div className="flex items-start gap-2 p-2.5 rounded border bg-red-950/30 border-red-500/20 text-sm">
                            <AlertTriangle size={16} className="text-red-400 shrink-0 mt-0.5" />
                            <span className="text-red-300">
                                All iterations after iteration {changeIter} and any child branches will be <strong>permanently deleted</strong>.
                            </span>
                        </div>
                    )}
                    <div className="space-y-2">
                        <label className="text-sm text-muted-foreground">Message for the agent (optional):</label>
                        <textarea
                            value={changeMessage}
                            onChange={(e) => setChangeMessage(e.target.value)}
                            placeholder={isChangeRevert
                                ? 'e.g. Try using shared memory instead...'
                                : 'e.g. Keep going, the speedup is not good enough yet...'
                            }
                            rows={3}
                            className="w-full rounded border bg-muted px-3 py-2 text-sm placeholder:text-muted-foreground/50 resize-none"
                        />
                    </div>
                    <DialogFooter>
                        <Button variant="outline" size="sm" onClick={() => setChangeIter(null)}>Cancel</Button>
                        <Button
                            size="sm"
                            className="bg-amber-600 hover:bg-amber-700 text-white"
                            onClick={handleChangeDecision}
                            disabled={isChanging}
                        >
                            <RefreshCw size={12} className="mr-1" />
                            {isChanging ? 'Applying...' : isChangeRevert ? 'Revert & Redo' : 'Continue'}
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            {/* Delete confirmation dialog */}
            <Dialog open={confirmDelete} onOpenChange={setConfirmDelete}>
                <DialogContent className="sm:max-w-sm">
                    <DialogHeader>
                        <DialogTitle>Delete branch?</DialogTitle>
                        <DialogDescription>
                            This will permanently delete the branch "{node.strategy.name}" and all its iteration data. This cannot be undone.
                        </DialogDescription>
                    </DialogHeader>
                    <DialogFooter>
                        <Button variant="outline" size="sm" onClick={() => setConfirmDelete(false)}>Cancel</Button>
                        <Button
                            size="sm"
                            variant="destructive"
                            onClick={handleDelete}
                            disabled={isDeleting}
                        >
                            <Trash2 size={12} className="mr-1" />
                            {isDeleting ? 'Deleting...' : 'Delete'}
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </ScrollArea>
    );
}
