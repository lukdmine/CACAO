import { useState, type ReactNode } from 'react';
import { useSelectedNode } from '@/store/appStore';
import { useAppStore } from '@/store/appStore';
import { Button } from '@/components/ui/button';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Facts } from '@/components/Facts';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { getStatusStyle, formatTime, formatSpeedup } from '@/utils/statusColors';
import { CaseResults } from './CaseResults';
import { BranchHistory } from './BranchHistory';
import { branchCaseRange } from './caseScale';
import { stopBranch, resumeBranch, messageBranch, changeDecision, configureBranch, deleteBranch } from '@/api/client';
import { refreshTree, useAnalysis, useIterationDetail } from '@/api/hooks';
import { toast } from 'sonner';
import { X, Square, Play, MessageSquare, Send, Settings2, CheckCircle2, XCircle, Trash2, RefreshCw, AlertTriangle, Loader2 } from 'lucide-react';

function branchIdFromNodeId(nodeId: string): string {
    return nodeId.replace(/^branch\//, '');
}

/** Section label. The one place medium weight is used besides the title. */
function Caption({ children }: { children: ReactNode }) {
    return <span className="block text-xs font-medium text-muted-foreground">{children}</span>;
}

/** Every block of code or output shares one quiet surface; colour is reserved for data. */
const PRE = 'mt-1 rounded-md bg-muted/40 p-2 text-[11px] font-mono whitespace-pre-wrap overflow-x-auto';

/** A decision word is coloured only when it is an outcome, not a continuation. */
function decisionTone(action: string): string {
    if (action === 'retry') return 'text-red-400';
    if (action === 'stop') return 'text-emerald-400';
    return 'text-muted-foreground';
}

/** Stand-in for a section whose body is still in flight, or never arrived.
 *  The tree told us the section exists, so an empty body once the request has
 *  settled means the fetch failed rather than that there is nothing to show. */
function Pending({ loading }: { loading: boolean }) {
    return (
        <div className="mt-1 flex items-center gap-1.5 p-2 text-[11px] text-muted-foreground">
            {loading ? (
                <><Loader2 size={11} className="animate-spin" /> Loading…</>
            ) : (
                <>Could not load this section.</>
            )}
        </div>
    );
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

    // Which iteration is expanded. Held per node so switching branches resets,
    // and left unset until the reader picks one — an untouched panel tracks the
    // newest iteration, which is what you want while a branch is running.
    const [openState, setOpenState] = useState<{ nodeId: string | null; iter: number | null }>(
        { nodeId: null, iter: null },
    );

    // Derived above the early return, because the iteration fetch below is a
    // hook and hooks cannot sit behind a conditional.
    const nodeId = node?.id ?? null;
    const isRoot = nodeId === 'root';
    const branchId = node && !isRoot ? branchIdFromNodeId(node.id) : '';
    const isActive = !!node && !isRoot
        && !['success', 'failed', 'branching'].includes(node.status);

    const iterations = node?.iterations ?? [];
    // One axis for every per-case mark in this branch. Derived from the whole
    // history so a case at the same speedup lands in the same place every time.
    const caseRange = branchCaseRange(iterations);
    const lastIterNum = iterations.length ? iterations[iterations.length - 1].iter_num : null;
    const openIter = openState.nodeId === nodeId ? openState.iter : lastIterNum;

    // A click in the branch history opens that iteration and brings it into view.
    // The scroll waits a frame so the accordion has re-rendered with it open.
    const selectIter = (iter: number) => {
        setOpenState({ nodeId, iter });
        requestAnimationFrame(() => {
            document
                .getElementById(`iter-item-${iter}`)
                ?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        });
    };

    // Phase-1 work product, and the root node is the only place it belongs: it is
    // about the problem, not about any one branch's bet on it.
    const { analysis, loading: analysisLoading } = useAnalysis(activeProblem, isRoot, isRunning);

    // Only the newest iteration of a running branch can still change; everything
    // else is immutable and served from cache after the first open.
    const { detail: openDetail, loading: detailLoading } = useIterationDetail(
        activeProblem,
        branchId,
        openIter,
        isRunning && isActive && openIter === lastIterNum,
    );

    if (!node) {
        return (
            <div className="h-full flex items-center justify-center text-muted-foreground text-sm p-6">
                <p className="text-center">Click a node in the tree to view details</p>
            </div>
        );
    }

    const style = getStatusStyle(node.status);
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
            <div className="space-y-5 p-4">
                {/* Header: the name, then status and the headline numbers as one line. */}
                <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0 space-y-1">
                        <h2 className="text-base font-medium leading-tight">{node.strategy.name}</h2>
                        <Facts
                            className="text-xs text-muted-foreground"
                            items={[
                                <span key="status" className={`flex items-center gap-1.5 ${style.text}`}>
                                    <span
                                        className={`inline-block size-1.5 rounded-full ${style.color} ${
                                            style.animate ? 'animate-pulse' : ''
                                        }`}
                                    />
                                    {style.label}
                                </span>,
                                node.iter_num > 0 && <span key="iter">iter {node.iter_num}/{node.max_iter}</span>,
                                node.best_time_us !== null && (
                                    <span key="time" className="font-mono">{formatTime(node.best_time_us)}</span>
                                ),
                                node.speedup !== null && (
                                    <span key="speedup" className="font-mono text-foreground">
                                        {formatSpeedup(node.speedup)}
                                    </span>
                                ),
                            ]}
                        />
                    </div>
                    <button
                        onClick={() => selectNode(null)}
                        className="shrink-0 cursor-pointer text-muted-foreground transition-colors hover:text-foreground"
                    >
                        <X size={16} />
                    </button>
                </div>

                {/* Branch controls */}
                {!isRoot && (
                    <div className="-ml-2 flex flex-wrap items-center gap-0.5">
                        {isActive && !isStopped && (
                            <Button size="sm" variant="ghost" className="h-7 text-xs" onClick={handleStop} disabled={isStopping}>
                                {isStopping ? <Loader2 size={10} className="mr-1 animate-spin" /> : <Square size={10} className="mr-1" />}
                                {isStopping ? 'Stopping...' : 'Stop'}
                            </Button>
                        )}
                        {isStopped && (
                            <Button size="sm" variant="ghost" className="h-7 text-xs" onClick={handleResume} disabled={isResuming}>
                                {isResuming ? <Loader2 size={10} className="mr-1 animate-spin" /> : <Play size={10} className="mr-1" />}
                                {isResuming ? 'Resuming...' : 'Resume'}
                            </Button>
                        )}
                        <Button
                            size="sm"
                            variant="ghost"
                            className="h-7 text-xs text-muted-foreground"
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
                                    className="h-7 text-xs text-red-400/80 hover:bg-red-500/10 hover:text-red-300"
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
                            className="flex-1 rounded-md border bg-transparent px-3 py-1.5 text-xs placeholder:text-muted-foreground/50"
                        />
                        <Button size="sm" variant="ghost" className="h-7 px-2" onClick={handleSendMessage} disabled={isSending || !messageText.trim()}>
                            <Send size={12} />
                        </Button>
                    </div>
                )}

                {/* User messages */}
                {(node.user_messages?.length ?? 0) > 0 && (
                    <div>
                        <Caption>
                            <span className="flex items-center gap-1"><MessageSquare size={12} /> Messages</span>
                        </Caption>
                        <div className="mt-1.5 space-y-1 border-l border-border pl-2.5 text-xs">
                            {node.user_messages!.map((m, i) => (
                                <div key={i}>
                                    <span className="text-muted-foreground">iter {m.iter_num}</span>{' '}
                                    <span>{m.content}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Strategy */}
                <div className="space-y-2.5 text-sm">
                    <div>
                        <Caption>Description</Caption>
                        <p className="mt-0.5">{node.strategy.description}</p>
                    </div>
                    <div>
                        <Caption>Hypothesis</Caption>
                        <p className="mt-0.5">{node.strategy.hypothesis}</p>
                    </div>
                    {node.strategy.key_parameters.length > 0 && (
                        <div>
                            <Caption>Parameters</Caption>
                            <div className="mt-0.5 flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[11px] text-muted-foreground">
                                {node.strategy.key_parameters.map((p) => (
                                    <span key={p}>{p}</span>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                {/* Phase 1's read of the problem and its reference kernel. Root only —
                    every branch inherits it, so repeating it per branch would say the
                    same thing four times. Collapsed for the same reason as the plan. */}
                {isRoot && (analysis || analysisLoading || isRunning) && (
                    analysis ? (
                        <Accordion type="single" collapsible className="text-xs">
                            <AccordionItem value="analysis" className="border-b-0">
                                <AccordionTrigger className="py-1.5 text-xs font-medium text-muted-foreground hover:no-underline hover:text-foreground">
                                    Analysis
                                </AccordionTrigger>
                                <AccordionContent>
                                    <pre className={PRE}>{analysis}</pre>
                                </AccordionContent>
                            </AccordionItem>
                        </Accordion>
                    ) : (
                        <p className="flex items-center gap-1.5 py-1.5 text-xs text-muted-foreground">
                            <Loader2 size={11} className="animate-spin" />
                            {analysisLoading ? 'Loading analysis…' : 'Analyzing the reference kernel…'}
                        </p>
                    )
                )}

                {/* Branch-level: planning runs once per branch, so the plan sits outside
                    the iteration list rather than being repeated inside iteration 1.
                    Collapsed by default — it is ~10k chars and read once. */}
                {node.plan && (
                    <Accordion type="single" collapsible className="text-xs">
                        <AccordionItem value="branch-plan" className="border-b-0">
                            <AccordionTrigger className="py-1.5 text-xs font-medium text-muted-foreground hover:no-underline hover:text-foreground">
                                Plan
                            </AccordionTrigger>
                            <AccordionContent>
                                <pre className={PRE}>{node.plan}</pre>
                            </AccordionContent>
                        </AccordionItem>
                    </Accordion>
                )}

                {/* Branch history — every iteration's speedup at once, with the
                    per-case grid underneath when the problem declares cases. Renders
                    once there are two iterations with results to compare. Clicking a
                    column opens that iteration below. */}
                <BranchHistory iterations={iterations} activeIter={openIter} onSelectIter={selectIter} />

                {/* Iterations */}
                {node.iterations.length > 0 && (
                    <>
                        <Separator />
                        <div>
                            <Caption>Iterations</Caption>
                            <Accordion
                                type="single"
                                collapsible
                                className="mt-1"
                                value={openIter !== null ? `iter-${openIter}` : ''}
                                onValueChange={(v) => setOpenState({
                                    nodeId,
                                    iter: v ? Number(v.slice('iter-'.length)) : null,
                                })}
                            >
                                {node.iterations.map((iter) => {
                                    // Only the expanded iteration has a fetched body.
                                    const detail = openDetail?.iter_num === iter.iter_num ? openDetail : null;
                                    function body<T>(v: T | null | undefined, render: (v: T) => ReactNode) {
                                        return v ? render(v) : <Pending loading={detailLoading} />;
                                    }
                                    const rs = iter.results_summary;
                                    const canChange = !isRoot && (iter.decision || iter.status === 'decided');
                                    return (
                                    <AccordionItem
                                        key={iter.iter_num}
                                        id={`iter-item-${iter.iter_num}`}
                                        value={`iter-${iter.iter_num}`}
                                    >
                                        <AccordionTrigger className="py-2 text-sm hover:no-underline">
                                            <Facts
                                                className="text-xs"
                                                items={[
                                                    <span key="name" className="text-sm text-foreground">Iteration {iter.iter_num}</span>,
                                                    iter.decision && (
                                                        <span key="decision" className={decisionTone(iter.decision.action)}>
                                                            {iter.decision.action}
                                                        </span>
                                                    ),
                                                    rs?.best_time_us != null && (
                                                        <span key="time" className="font-mono text-muted-foreground">
                                                            {formatTime(rs.best_time_us)}
                                                        </span>
                                                    ),
                                                    rs?.speedup != null && (
                                                        <span key="speedup" className="font-mono text-foreground/80">
                                                            {formatSpeedup(rs.speedup)}
                                                        </span>
                                                    ),
                                                ]}
                                            />
                                        </AccordionTrigger>
                                        <AccordionContent>
                                            <div className="ml-1 space-y-3 border-l border-border pl-3">
                                                {rs && (
                                                    <div className="space-y-1.5 text-xs">
                                                        <div className="flex items-center gap-1.5">
                                                            {rs.has_success
                                                                ? <CheckCircle2 size={12} className="text-emerald-400" />
                                                                : rs.num_total > 0
                                                                    ? <XCircle size={12} className="text-red-400" />
                                                                    : null
                                                            }
                                                            <span className="font-medium">Tuning results</span>
                                                            <span className="text-muted-foreground">
                                                                {rs.num_successful}/{rs.num_total} configs passed
                                                                {rs.cases ? ' · primary case' : ''}
                                                            </span>
                                                        </div>
                                                        {rs.best_time_us != null && (
                                                            <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
                                                                <span className="text-muted-foreground">Best time</span>
                                                                <span className="font-mono">{formatTime(rs.best_time_us)}</span>
                                                                {rs.reference_time_us != null && (
                                                                    <>
                                                                        <span className="text-muted-foreground">Reference</span>
                                                                        <span className="font-mono">{formatTime(rs.reference_time_us)}</span>
                                                                    </>
                                                                )}
                                                                {rs.speedup != null && (
                                                                    <>
                                                                        <span className="text-muted-foreground">
                                                                            {rs.cases ? 'Speedup (geomean)' : 'Speedup'}
                                                                        </span>
                                                                        <span className="font-mono">{formatSpeedup(rs.speedup)}</span>
                                                                    </>
                                                                )}
                                                            </div>
                                                        )}
                                                        {rs.best_config && (
                                                            <div className="flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[11px] text-muted-foreground">
                                                                {rs.cases && <span>primary:</span>}
                                                                {Object.entries(rs.best_config).map(([k, v]) => (
                                                                    <span key={k}>{k}={v}</span>
                                                                ))}
                                                            </div>
                                                        )}
                                                        <CaseResults summary={rs} range={caseRange} />
                                                    </div>
                                                )}

                                                {/* Bodies come from the per-iteration fetch; the tree only
                                                    says which sections exist. `body` renders the spinner
                                                    while that request is still in flight. */}
                                                <Accordion type="multiple" className="text-xs">
                                                    {iter.has.kernel_code && (
                                                        <AccordionItem value={`iter-${iter.iter_num}-kernel`}>
                                                            <AccordionTrigger className="py-1.5 text-xs hover:no-underline">Kernel code</AccordionTrigger>
                                                            <AccordionContent>
                                                                {body(detail?.kernel_code, (v) => <pre className={PRE}>{v}</pre>)}
                                                            </AccordionContent>
                                                        </AccordionItem>
                                                    )}

                                                    {iter.has.framework_cpp && (
                                                        <AccordionItem value={`iter-${iter.iter_num}-framework`}>
                                                            <AccordionTrigger className="py-1.5 text-xs hover:no-underline">Framework driver</AccordionTrigger>
                                                            <AccordionContent>
                                                                {body(detail?.framework_cpp, (v) => <pre className={PRE}>{v}</pre>)}
                                                            </AccordionContent>
                                                        </AccordionItem>
                                                    )}

                                                    {iter.has.run_output && (
                                                        <AccordionItem value={`iter-${iter.iter_num}-output`}>
                                                            <AccordionTrigger className="py-1.5 text-xs hover:no-underline">Run output</AccordionTrigger>
                                                            <AccordionContent>
                                                                {body(detail?.run_output, (v) => (
                                                                    <pre className={`${PRE} ${v.includes('ERROR') ? 'border-l-2 border-red-500/60' : ''}`}>
                                                                        {v}
                                                                    </pre>
                                                                ))}
                                                            </AccordionContent>
                                                        </AccordionItem>
                                                    )}

                                                    {iter.has.ncu_metrics && (
                                                        <AccordionItem value={`iter-${iter.iter_num}-ncu`}>
                                                            <AccordionTrigger className="py-1.5 text-xs hover:no-underline">NCU metrics</AccordionTrigger>
                                                            <AccordionContent>
                                                                {body(detail?.ncu_metrics, (metrics) => (
                                                                    <div className="mt-1 overflow-hidden">
                                                                        <table className="text-[11px]">
                                                                            <tbody>
                                                                                {Object.entries(metrics).map(([key, val]) => (
                                                                                    <tr key={key} className="border-b last:border-0">
                                                                                        <td className="max-w-[380px] truncate px-2 py-1 font-mono text-muted-foreground">{key}</td>
                                                                                        <td className="whitespace-nowrap px-2 py-1 font-mono">{String(val)}</td>
                                                                                    </tr>
                                                                                ))}
                                                                            </tbody>
                                                                        </table>
                                                                    </div>
                                                                ))}
                                                            </AccordionContent>
                                                        </AccordionItem>
                                                    )}

                                                    {iter.has.proposal && (
                                                        <AccordionItem value={`iter-${iter.iter_num}-proposal`}>
                                                            <AccordionTrigger className="py-1.5 text-xs hover:no-underline">Optimization proposal</AccordionTrigger>
                                                            <AccordionContent>
                                                                {body(detail?.proposal, (v) => (
                                                                    <div className="mt-1 whitespace-pre-wrap text-[11px] leading-relaxed">{v}</div>
                                                                ))}
                                                            </AccordionContent>
                                                        </AccordionItem>
                                                    )}

                                                    {iter.decision && (
                                                        <AccordionItem value={`iter-${iter.iter_num}-decision`}>
                                                            <AccordionTrigger className="py-1.5 text-xs hover:no-underline">Decision</AccordionTrigger>
                                                            <AccordionContent>
                                                                <div className="mt-1 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[11px]">
                                                                    <span className="text-muted-foreground">Action</span>
                                                                    <span className="font-medium">{iter.decision.action}</span>
                                                                    <span className="text-muted-foreground">Reasoning</span>
                                                                    <span>{iter.decision.reasoning}</span>
                                                                    {iter.decision.feedback && (
                                                                        <>
                                                                            <span className="text-muted-foreground">Feedback</span>
                                                                            <span>{iter.decision.feedback}</span>
                                                                        </>
                                                                    )}
                                                                </div>
                                                                {iter.decision.error_analysis && (
                                                                    <div className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 border-l-2 border-red-500/60 pl-2 text-[11px]">
                                                                        <span className="col-span-2 font-medium text-red-400">Error analysis</span>
                                                                        <span className="text-muted-foreground">Type</span>
                                                                        <span>{iter.decision.error_analysis.error_type}</span>
                                                                        <span className="text-muted-foreground">Cause</span>
                                                                        <span>{iter.decision.error_analysis.root_cause}</span>
                                                                        <span className="text-muted-foreground">Fix</span>
                                                                        <span>{iter.decision.error_analysis.suggested_fix}</span>
                                                                    </div>
                                                                )}
                                                            </AccordionContent>
                                                        </AccordionItem>
                                                    )}
                                                </Accordion>

                                                {/* The override lives after everything it overrides. A real button,
                                                    because a text link here was missed; neutral, because amber made
                                                    it look like a status. */}
                                                {canChange && (
                                                    <Button
                                                        size="sm"
                                                        variant="outline"
                                                        className="h-7 text-xs"
                                                        onClick={() => { setChangeIter(iter.iter_num); setChangeMessage(''); }}
                                                    >
                                                        <RefreshCw size={11} className="mr-1" />
                                                        {iter.decision ? 'Change decision' : 'Retry iteration'}
                                                    </Button>
                                                )}
                                            </div>
                                        </AccordionContent>
                                    </AccordionItem>
                                    );
                                })}
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
