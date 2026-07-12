import { useState, useEffect } from 'react';
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
    DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { createProblem, updateProblem, fetchProblemDetail, fetchGpuDevices, previewInputs, type CreateProblemData, type GpuDevice, type ArgSpec, type BufferSpec, type ScalarSpec, type Placement } from '@/api/client';
import { refreshProblems } from '@/api/hooks';
import { Plus, Trash2, ChevronLeft, ChevronRight, ChevronUp, ChevronDown, Loader2, AlertTriangle } from 'lucide-react';

// ── Default form state ───────────────────────────────────────────────────────
function defaultForm(): CreateProblemData {
    return {
        slug: '', name: '', description: '',
        gpu: { index: 0 },
        reference_type: 'cuda',
        ref_function: '', ref_block_x: 256, ref_block_y: 1, ref_block_z: 1,
        ref_kernel_code: '',
        ref_cpu_code: '',
        inputs: {
            headers: [],
            shared_setup: '',
            args: [
                { kind: 'scalar', name: 'N', dtype: 'int', value: 1024, placements: ['host', 'define'] },
                { kind: 'buffer', name: 'input', dtype: 'float', size: 'N', access: 'read', init: 'random', min: null, max: null, validate: false },
                { kind: 'buffer', name: 'output', dtype: 'float', size: 'N', access: 'write', init: 'zeros', min: null, max: null, validate: true },
            ],
        },
        global_size_type: 'cuda',
        grid_x: 'N', grid_y: '1', grid_z: '1',
        tolerance: 0.05,
        tuning: { duration_s: 100 },
    };
}

/** Args passed to the reference kernel, in declaration order: buffers, plus `runtime` scalars. */
function boundaryOf(args: ArgSpec[]): ArgSpec[] {
    return args.filter((a) => a.kind === 'buffer' || a.placements.includes('runtime'));
}

const PLACEMENTS: Placement[] = ['host', 'define', 'runtime'];

const PLACEMENT_HELP: Record<Placement, string> = {
    host: 'inline constexpr in inputs.hpp — sizes the data generators',
    define: '-D macro for NVRTC — a compile-time constant inside the kernels (name must be UPPERCASE)',
    runtime: 'passed as a kernel argument — joins the reference-signature boundary',
};

const STEPS = ['Basics', 'GPU & Kernel', 'Arguments', 'Grid & Review'];

interface NewProblemDialogProps {
    onCreated?: (name: string) => void;
    mode?: 'create' | 'edit';
    editProblemName?: string;
    trigger?: React.ReactNode;
}

export function NewProblemDialog({ onCreated, mode = 'create', editProblemName, trigger }: NewProblemDialogProps) {
    const [open, setOpen] = useState(false);
    const [step, setStep] = useState(0);
    const [form, setForm] = useState<CreateProblemData>(defaultForm);
    const [error, setError] = useState('');
    const [submitting, setSubmitting] = useState(false);
    const [loading, setLoading] = useState(false);
    const [gpuDevices, setGpuDevices] = useState<GpuDevice[]>([]);
    const [loadingGpus, setLoadingGpus] = useState(false);
    const [warnings, setWarnings] = useState<string[]>([]);
    const [previewHpp, setPreviewHpp] = useState('');

    useEffect(() => {
        if (open && gpuDevices.length === 0) {
            setLoadingGpus(true);
            fetchGpuDevices().then((res) => {
                setGpuDevices(res.devices);
            }).catch(e => {
                console.error("Failed to fetch GPUs", e);
            }).finally(() => {
                setLoadingGpus(false);
            });
        }
    }, [open, gpuDevices.length]);

    // Live inputs.hpp preview, rendered by the backend so there is exactly one generator.
    // Debounced: the boundary changes on every keystroke in the args table.
    useEffect(() => {
        if (!open || step !== 3) return;
        const id = setTimeout(() => {
            previewInputs(form.inputs)
                .then((r) => setPreviewHpp(r.inputs_hpp))
                .catch((e) => setPreviewHpp(`// ${e instanceof Error ? e.message : String(e)}`));
        }, 300);
        return () => clearTimeout(id);
    }, [open, step, form.inputs]);

    function update<K extends keyof CreateProblemData>(key: K, value: CreateProblemData[K]) {
        setForm((prev) => ({ ...prev, [key]: value }));
    }

    function reset() {
        setForm(defaultForm());
        setStep(0);
        setError('');
        setWarnings([]);
        setPreviewHpp('');
    }

    async function loadProblemData() {
        if (mode !== 'edit' || !editProblemName) return;
        setLoading(true);
        try {
            const data = await fetchProblemDetail(editProblemName);
            const ref = (data.config.reference ?? {}) as { type?: 'cuda' | 'cpu_c'; function?: string; block?: { x?: number; y?: number; z?: number } };
            const grid = (data.config.grid ?? {}) as { x?: string; y?: string; z?: string };

            // The boundary comes from the structured spec, never from the generated
            // inputs.hpp. A problem with no inputs.yaml yet loads an empty boundary
            // rather than silently inventing one.
            if (!data.inputs) {
                setError(
                    `'${editProblemName}' has no inputs.yaml — it predates framework mode. ` +
                    'Saving will define its I/O boundary from scratch.'
                );
            }

            setForm({
                slug: data.name,
                name: (data.config.name as string) || data.name,
                description: (data.config.description as string) || '',
                gpu: (data.config.gpu as CreateProblemData['gpu']) || defaultForm().gpu,
                reference_type: ref.type || 'cuda',
                ref_function: ref.function || '',
                ref_block_x: ref.block?.x ?? 256,
                ref_block_y: ref.block?.y ?? 1,
                ref_block_z: ref.block?.z ?? 1,
                ref_kernel_code: data.ref_kernel || '',
                ref_cpu_code: data.ref_cpu || '',
                inputs: data.inputs ?? { headers: [], shared_setup: '', args: [] },
                global_size_type: (data.config.global_size_type as CreateProblemData['global_size_type']) || 'cuda',
                grid_x: grid.x ? String(grid.x) : 'N',
                grid_y: grid.y ? String(grid.y) : '1',
                grid_z: grid.z ? String(grid.z) : '1',
                tolerance: ((data.config.validation as { tolerance?: number } | undefined)?.tolerance) ?? 0.05,
                tuning: {
                    duration_s:
                        (data.config.tuning as { duration_s?: number } | undefined)?.duration_s ?? 100,
                },
            });
        } catch (err) {
            setError('Failed to load problem data: ' + (err instanceof Error ? err.message : String(err)));
        } finally {
            setLoading(false);
        }
    }

    // ── Argument list helpers ────────────────────────────────────────────────
    const args = form.inputs.args;

    function setArgs(next: ArgSpec[]) {
        setForm((prev) => ({ ...prev, inputs: { ...prev.inputs, args: next } }));
    }

    function patchArg(i: number, patch: Partial<ScalarSpec> & Partial<BufferSpec>) {
        setArgs(args.map((a, j) => (j === i ? ({ ...a, ...patch } as ArgSpec) : a)));
    }

    /** Exactly one buffer is compared against the reference, so this behaves as a radio. */
    function setValidated(i: number) {
        setArgs(args.map((a, j) => (a.kind === 'buffer' ? { ...a, validate: j === i } : a)));
    }

    function togglePlacement(i: number, p: Placement) {
        const s = args[i] as ScalarSpec;
        const next = s.placements.includes(p)
            ? s.placements.filter((x) => x !== p)
            : [...s.placements, p];
        patchArg(i, { placements: next });
    }

    function addScalar() {
        setArgs([...args, { kind: 'scalar', name: '', dtype: 'int', value: 0, placements: ['host', 'define'] }]);
    }

    function addBuffer() {
        setArgs([...args, { kind: 'buffer', name: '', dtype: 'float', size: '', access: 'read', init: 'random', min: null, max: null, validate: false }]);
    }

    async function handleSubmit() {
        setError('');
        setSubmitting(true);
        try {
            const res = mode === 'create'
                ? await createProblem(form)
                : await updateProblem(editProblemName!, form);

            // Saved either way: a signature mismatch is reported, not blocked (the regex
            // cannot parse every legal declaration). Keep the dialog open so it is read.
            if (res.warnings?.length) {
                setWarnings(res.warnings);
                await refreshProblems();
                return;
            }

            setOpen(false);
            reset();
            onCreated?.(form.slug);
            await refreshProblems();
        } catch (err) {
            setError(err instanceof Error ? err.message : `Failed to ${mode} problem`);
        } finally {
            setSubmitting(false);
        }
    }

    // Auto-generate slug from display name
    function handleNameChange(displayName: string) {
        update('name', displayName);
        if (!form.slug || form.slug === form.name.toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, '')) {
            update('slug', displayName.toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, ''));
        }
    }

    function moveItem<T>(items: T[], fromIndex: number, toIndex: number): T[] {
        if (toIndex < 0 || toIndex >= items.length || fromIndex === toIndex) {
            return items;
        }

        const next = [...items];
        const [item] = next.splice(fromIndex, 1);
        next.splice(toIndex, 0, item);
        return next;
    }

    function formatGpuSubtitle(gpu: GpuDevice): string {
        if (typeof gpu.total_memory_mb === 'number' && Number.isFinite(gpu.total_memory_mb)) {
            return `${(gpu.total_memory_mb / 1024).toFixed(1)} GB Memory`;
        }

        const parts: string[] = [];
        if (gpu.compute_capability) {
            parts.push(`CC ${gpu.compute_capability}`);
        }
        if (typeof gpu.sm_count === 'number' && Number.isFinite(gpu.sm_count)) {
            parts.push(`${gpu.sm_count} SMs`);
        }

        return parts.length > 0 ? parts.join(' · ') : 'Memory unavailable';
    }

    return (
        <Dialog open={open} onOpenChange={(v) => {
            setOpen(v);
            if (!v) reset();
            else if (mode === 'edit') loadProblemData();
        }}>
            <DialogTrigger asChild>
                {trigger || (
                    <Button variant="outline" className="w-full text-sm" size="sm">
                        <Plus size={14} className="mr-1.5" />
                        New Problem
                    </Button>
                )}
            </DialogTrigger>
            <DialogContent className="sm:max-w-3xl max-h-[85vh] flex flex-col overflow-hidden">
                <DialogHeader>
                    <DialogTitle>{mode === 'create' ? 'New Problem' : 'Edit Problem'}</DialogTitle>
                    <DialogDescription>
                        {mode === 'create' ? 'Create a new CUDA kernel optimization problem.' : 'Modify an existing problem.'}
                    </DialogDescription>
                </DialogHeader>

                {loading && mode === 'edit' && (
                    <div className="flex items-center justify-center py-10">
                        <Loader2 className="animate-spin text-muted-foreground" />
                    </div>
                )}

                {(!loading || mode !== 'edit') && (
                    <>
                        {/* Step indicator */}
                        <div className="flex items-center gap-1 mb-2">
                            {STEPS.map((label, i) => (
                                <button
                                    key={label}
                                    onClick={() => setStep(i)}
                                    className={`text-xs px-2 py-1 rounded transition-colors cursor-pointer ${i === step ? 'bg-primary text-primary-foreground' :
                                        i < step ? 'bg-primary/20 text-primary' :
                                            'bg-muted text-muted-foreground'
                                        }`}
                                >
                                    {i + 1}. {label}
                                </button>
                            ))}
                        </div>

                        <Separator />

                        <div className="overflow-y-auto flex-1 min-h-0">
                        {/* ── Step 1: Basics ────────────────────────────────────────────── */}
                        {step === 0 && (
                            <div className="space-y-4 py-2">
                                <div className="space-y-2">
                                    <Label>Display Name *</Label>
                                    <Input
                                        placeholder="e.g. GEMM, Convolution 2D"
                                        value={form.name}
                                        onChange={(e) => handleNameChange(e.target.value)}
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label>Slug (directory name)</Label>
                                    <Input
                                        placeholder="e.g. gemm, convolution_2d"
                                        value={form.slug}
                                        onChange={(e) => update('slug', e.target.value)}
                                        className="font-mono"
                                        disabled={mode === 'edit'}
                                    />
                                    <p className="text-[11px] text-muted-foreground">Lowercase, underscores only. This becomes the folder name under <code>problems/</code></p>
                                </div>
                                <div className="space-y-2">
                                    <Label>Description *</Label>
                                    <textarea
                                        className="w-full rounded-md border bg-background px-3 py-2 text-sm min-h-[100px] resize-y"
                                        placeholder="e.g. General Matrix-Matrix Multiplication: C = A * B"
                                        value={form.description}
                                        onChange={(e) => update('description', e.target.value)}
                                    />
                                </div>
                            </div>
                        )}

                        {/* ── Step 2: GPU & Kernel ──────────────────────────────────────── */}
                        {step === 1 && (
                            <div className="space-y-4 py-2">
                                <div className="space-y-2">
                                    <Label>Target GPU Device</Label>
                                    {loadingGpus ? (
                                        <div className="text-xs text-muted-foreground flex items-center gap-2">
                                            <Loader2 size={12} className="animate-spin" /> Detecting GPUs...
                                        </div>
                                    ) : gpuDevices.length === 0 ? (
                                        <div className="text-xs text-destructive">No GPUs detected automatically. Optimization will fall back to default logic.</div>
                                    ) : (
                                        <div className="grid gap-2">
                                            {gpuDevices.map((gpu) => (
                                                <button
                                                    key={gpu.index}
                                                    onClick={() => update('gpu', { index: gpu.index })}
                                                    className={`text-left text-sm px-3 py-2 rounded-md border transition-colors cursor-pointer ${form.gpu.index === gpu.index
                                                        ? 'border-primary bg-primary/10 text-primary'
                                                        : 'border-border hover:border-primary/50 text-foreground'
                                                        }`}
                                                >
                                                    <div className="font-medium">[{gpu.index}] {gpu.name || gpu.model || `GPU ${gpu.index}`}</div>
                                                    <div className="text-xs opacity-70 mt-0.5">
                                                        {formatGpuSubtitle(gpu)}
                                                    </div>
                                                </button>
                                            ))}
                                        </div>
                                    )}
                                </div>

                                <Separator />

                                <div className="grid grid-cols-2 gap-3">
                                    <div className="space-y-1.5">
                                        <Label className="text-xs">Reference Type</Label>
                                        <select
                                            className="h-8 w-full rounded border bg-background px-2 text-xs"
                                            value={form.reference_type}
                                            onChange={(e) => update('reference_type', e.target.value as CreateProblemData['reference_type'])}
                                        >
                                            <option value="cuda">CUDA reference kernel</option>
                                            <option value="cpu_c">CPU C/C++ reference</option>
                                        </select>
                                    </div>
                                    <div className="space-y-1.5">
                                        <Label className="text-xs" title="OpenCL: grid is total work-items (KTT divides by the per-config block). CUDA: grid is the number of blocks.">
                                            Global Size Type
                                        </Label>
                                        <select
                                            className="h-8 w-full rounded border bg-background px-2 text-xs"
                                            value={form.global_size_type}
                                            onChange={(e) => update('global_size_type', e.target.value as CreateProblemData['global_size_type'])}
                                        >
                                            <option value="cuda">cuda — grid = number of blocks</option>
                                            <option value="opencl">opencl — grid = total work-items</option>
                                        </select>
                                    </div>
                                </div>

                                {form.reference_type === 'cpu_c' && (
                                    <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-100 flex gap-2">
                                        <AlertTriangle size={14} className="shrink-0 mt-0.5" />
                                        <span>
                                            Framework mode validates only against a CUDA reference kernel. A C reference
                                            is saved and kept, but the problem <strong>cannot be run</strong> until
                                            CPU-reference support lands.
                                        </span>
                                    </div>
                                )}

                                <div className="grid grid-cols-4 gap-3">
                                    <div className="space-y-1.5">
                                        <Label className="text-xs">Ref Function *</Label>
                                        <Input
                                            placeholder="e.g. gemm_reference"
                                            className="text-xs font-mono"
                                            value={form.ref_function}
                                            onChange={(e) => update('ref_function', e.target.value)}
                                        />
                                    </div>
                                    <div className="space-y-1.5">
                                        <Label className="text-xs">Block X</Label>
                                        <Input type="number" className="text-xs" value={form.ref_block_x}
                                            onChange={(e) => update('ref_block_x', parseInt(e.target.value) || 1)}
                                            disabled={form.reference_type !== 'cuda'} />
                                    </div>
                                    <div className="space-y-1.5">
                                        <Label className="text-xs">Block Y</Label>
                                        <Input type="number" className="text-xs" value={form.ref_block_y}
                                            onChange={(e) => update('ref_block_y', parseInt(e.target.value) || 1)}
                                            disabled={form.reference_type !== 'cuda'} />
                                    </div>
                                    <div className="space-y-1.5">
                                        <Label className="text-xs">Block Z</Label>
                                        <Input type="number" className="text-xs" value={form.ref_block_z}
                                            onChange={(e) => update('ref_block_z', parseInt(e.target.value) || 1)}
                                            disabled={form.reference_type !== 'cuda'} />
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <Label>{form.reference_type === 'cuda' ? 'Reference Kernel Code *' : 'CPU Reference Code *'}</Label>
                                    <textarea
                                        className="w-full rounded-md border bg-zinc-950 text-zinc-100 font-mono text-xs px-3 py-2 min-h-[200px] resize-y"
                                        placeholder={form.reference_type === 'cuda'
                                            ? 'extern "C" __global__ void my_kernel(...) {\n  // Reference implementation\n}'
                                            : 'extern "C" void reference(float* in, float* out, int N) {\n  // CPU reference implementation\n}'}
                                        value={form.reference_type === 'cuda' ? form.ref_kernel_code : form.ref_cpu_code}
                                        onChange={(e) => {
                                            if (form.reference_type === 'cuda') {
                                                update('ref_kernel_code', e.target.value);
                                            } else {
                                                update('ref_cpu_code', e.target.value);
                                            }
                                        }}
                                        spellCheck={false}
                                    />
                                </div>
                            </div>
                        )}

                        {/* ── Step 3: Arguments ─────────────────────────────────────────── */}
                        {step === 2 && (
                            <div className="space-y-3 py-2">
                                <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-100 flex gap-2">
                                    <AlertTriangle size={14} className="shrink-0 mt-0.5" />
                                    <span>
                                        <strong>Order is the reference kernel's signature.</strong> Arguments bind by
                                        position, not by name — a wrong order makes the reference compute from shuffled
                                        inputs and validation compare against garbage, with no error. Buffers, and
                                        scalars marked <code className="font-mono">runtime</code>, are passed; a
                                        host-only scalar is declared but not passed, so it may sit anywhere.
                                    </span>
                                </div>

                                {/* Derived boundary — what the reference kernel actually receives */}
                                <div className="rounded-md border bg-zinc-950 px-3 py-2 text-xs font-mono text-zinc-300">
                                    <span className="text-zinc-500">{form.ref_function || 'reference'}(</span>
                                    {boundaryOf(args).map((a, i) => (
                                        <span key={a.name + i}>
                                            {i > 0 && <span className="text-zinc-500">, </span>}
                                            <span className={a.kind === 'scalar' ? 'text-sky-400' : 'text-emerald-400'}>
                                                {a.name || '?'}
                                            </span>
                                        </span>
                                    ))}
                                    <span className="text-zinc-500">)</span>
                                    {boundaryOf(args).length === 0 && <span className="text-zinc-600 italic"> — nothing passed</span>}
                                </div>

                                <div className="flex items-center justify-between">
                                    <Label>Arguments</Label>
                                    <div className="flex gap-2">
                                        <Button size="sm" variant="outline" className="text-xs h-6" onClick={addScalar}>
                                            <Plus size={12} className="mr-1" /> Scalar
                                        </Button>
                                        <Button size="sm" variant="outline" className="text-xs h-6" onClick={addBuffer}>
                                            <Plus size={12} className="mr-1" /> Buffer
                                        </Button>
                                    </div>
                                </div>

                                {args.length === 0 && (
                                    <p className="text-xs text-muted-foreground italic">No arguments defined.</p>
                                )}

                                {args.map((a, i) => (
                                    <div key={i} className="rounded-md border border-border/50 p-2 space-y-1.5">
                                        <div className="flex items-center gap-2">
                                            {/* Reorder — this list IS the signature order */}
                                            <div className="flex flex-col gap-0.5">
                                                <button type="button" className="text-muted-foreground hover:text-foreground disabled:opacity-40"
                                                    disabled={i === 0}
                                                    onClick={() => setArgs(moveItem(args, i, i - 1))}
                                                    aria-label={`Move ${a.name || i + 1} up`}>
                                                    <ChevronUp size={14} />
                                                </button>
                                                <button type="button" className="text-muted-foreground hover:text-foreground disabled:opacity-40"
                                                    disabled={i === args.length - 1}
                                                    onClick={() => setArgs(moveItem(args, i, i + 1))}
                                                    aria-label={`Move ${a.name || i + 1} down`}>
                                                    <ChevronDown size={14} />
                                                </button>
                                            </div>

                                            <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded shrink-0 ${a.kind === 'scalar' ? 'bg-sky-500/15 text-sky-400' : 'bg-emerald-500/15 text-emerald-400'
                                                }`}>
                                                {a.kind}
                                            </span>

                                            <Input className="text-xs font-mono flex-1 min-w-0" placeholder="Name" value={a.name}
                                                onChange={(e) => patchArg(i, { name: e.target.value })} />

                                            <select className="h-8 rounded border bg-background px-2 text-xs"
                                                value={a.dtype}
                                                onChange={(e) => patchArg(i, { dtype: e.target.value as 'int' | 'float' })}>
                                                <option value="float">float</option>
                                                <option value="int">int</option>
                                            </select>

                                            {a.kind === 'scalar' ? (
                                                <Input type="number" className="text-xs w-24" value={a.value}
                                                    onChange={(e) => patchArg(i, { value: parseFloat(e.target.value) || 0 })} />
                                            ) : (
                                                <>
                                                    <Input className="text-xs font-mono w-28" placeholder="size expr" value={a.size}
                                                        onChange={(e) => patchArg(i, { size: e.target.value })}
                                                        title="C++ expression over host-const scalars, e.g. kSizeM * kSizeK" />
                                                    <select className="h-8 rounded border bg-background px-2 text-xs"
                                                        value={a.access}
                                                        onChange={(e) => {
                                                            const access = e.target.value as BufferSpec['access'];
                                                            patchArg(i, { access, init: access === 'write' ? 'zeros' : a.init });
                                                        }}>
                                                        <option value="read">read</option>
                                                        <option value="write">write</option>
                                                        <option value="readwrite">readwrite</option>
                                                    </select>
                                                </>
                                            )}

                                            <button className="text-muted-foreground hover:text-destructive"
                                                onClick={() => setArgs(args.filter((_, j) => j !== i))}
                                                aria-label={`Remove ${a.name || i + 1}`}>
                                                <Trash2 size={14} />
                                            </button>
                                        </div>

                                        {/* Scalar: placement multi-select (a scalar can be several of these at once) */}
                                        {a.kind === 'scalar' && (
                                            <div className="flex items-center gap-3 pl-7 flex-wrap">
                                                {PLACEMENTS.map((p) => (
                                                    <label key={p} className="flex items-center gap-1 text-xs" title={PLACEMENT_HELP[p]}>
                                                        <input type="checkbox"
                                                            checked={(a as ScalarSpec).placements.includes(p)}
                                                            onChange={() => togglePlacement(i, p)} />
                                                        <span className="font-mono">{p}</span>
                                                    </label>
                                                ))}
                                                {(a as ScalarSpec).placements.includes('define') && a.name !== a.name.toUpperCase() && (
                                                    <span className="text-[11px] text-amber-400">
                                                        -D macros must be UPPERCASE (they collide with NVRTC headers)
                                                    </span>
                                                )}
                                            </div>
                                        )}

                                        {/* Buffer: init + validate */}
                                        {a.kind === 'buffer' && (
                                            <div className="flex items-center gap-2 pl-7 flex-wrap">
                                                <select className="h-7 rounded border bg-background px-2 text-xs"
                                                    value={a.init}
                                                    onChange={(e) => patchArg(i, { init: e.target.value as BufferSpec['init'] })}>
                                                    <option value="zeros">zeros</option>
                                                    <option value="random">random</option>
                                                    <option value="custom">custom</option>
                                                </select>

                                                {a.init === 'random' && (
                                                    <>
                                                        <span className="text-xs text-muted-foreground">range:</span>
                                                        <Input type="number" className="text-xs w-20 h-7" placeholder="min"
                                                            value={a.min ?? ''}
                                                            onChange={(e) => patchArg(i, { min: e.target.value === '' ? null : parseFloat(e.target.value) })} />
                                                        <span className="text-xs text-muted-foreground">to</span>
                                                        <Input type="number" className="text-xs w-20 h-7" placeholder="max"
                                                            value={a.max ?? ''}
                                                            onChange={(e) => patchArg(i, { max: e.target.value === '' ? null : parseFloat(e.target.value) })} />
                                                    </>
                                                )}

                                                {/* Exactly one buffer is compared against the reference -> radio, not checkbox */}
                                                <label className="flex items-center gap-1 text-xs ml-auto"
                                                    title="The buffer compared against the reference kernel. Exactly one.">
                                                    <input type="radio" name="validated" checked={a.validate}
                                                        onChange={() => setValidated(i)} />
                                                    validated
                                                </label>
                                            </div>
                                        )}

                                        {a.kind === 'buffer' && a.init === 'custom' && (
                                            <div className="pl-7">
                                                <textarea
                                                    className="w-full rounded-md border bg-zinc-950 text-zinc-100 font-mono text-[11px] px-2 py-1.5 min-h-[90px] resize-y"
                                                    placeholder={`std::vector<${a.dtype}> v(${a.size || 'N'});\n// build the data however you like\nreturn v;`}
                                                    value={a.body ?? ''}
                                                    onChange={(e) => patchArg(i, { body: e.target.value })}
                                                    spellCheck={false}
                                                />
                                                <p className="text-[11px] text-muted-foreground mt-1">
                                                    Body of <code className="font-mono">gen_{a.name || 'name'}()</code> — must
                                                    return a <code className="font-mono">std::vector&lt;{a.dtype}&gt;</code>.
                                                    May use scalars, never tuning parameters.
                                                </p>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}


                        {/* ── Step 4: Grid & Review ────────────────────────────────────── */}
                        {step === 3 && (
                            <div className="space-y-4 py-2">
                                <div className="grid grid-cols-5 gap-3">
                                    <div className="space-y-1.5">
                                        <Label className="text-xs">Grid X</Label>
                                        <Input className="text-xs font-mono" value={form.grid_x}
                                            onChange={(e) => update('grid_x', e.target.value)} />
                                    </div>
                                    <div className="space-y-1.5">
                                        <Label className="text-xs">Grid Y</Label>
                                        <Input className="text-xs font-mono" value={form.grid_y}
                                            onChange={(e) => update('grid_y', e.target.value)} />
                                    </div>
                                    <div className="space-y-1.5">
                                        <Label className="text-xs">Grid Z</Label>
                                        <Input className="text-xs font-mono" value={form.grid_z}
                                            onChange={(e) => update('grid_z', e.target.value)} />
                                    </div>
                                    <div className="space-y-1.5">
                                        <Label className="text-xs">Tolerance</Label>
                                        <Input type="number" step="0.001" className="text-xs" value={form.tolerance}
                                            onChange={(e) => update('tolerance', parseFloat(e.target.value) || 0.05)} />
                                    </div>
                                    <div className="space-y-1.5">
                                        <Label className="text-xs" title="Wall-clock budget (seconds) for one tuner run; includes reference computation.">Tuner budget (s)</Label>
                                        <Input type="number" min={1} step={1} className="text-xs"
                                            value={form.tuning?.duration_s ?? 100}
                                            onChange={(e) => update('tuning', { duration_s: parseInt(e.target.value, 10) || 100 })} />
                                    </div>
                                </div>

                                <Separator />

                                {/* Review summary */}
                                <div className="space-y-2">
                                    <Label>Summary</Label>
                                    <div className="rounded-md border bg-zinc-950 p-3 text-xs font-mono text-zinc-300 space-y-1">
                                        <div><span className="text-zinc-500">name:</span> {form.name || '—'}</div>
                                        <div><span className="text-zinc-500">slug:</span> {form.slug || '—'}</div>
                                        <div><span className="text-zinc-500">gpu:</span> Index {form.gpu.index} {gpuDevices.find(g => g.index === form.gpu.index)?.name ? `(${gpuDevices.find(g => g.index === form.gpu.index)?.name})` : ''}</div>
                                        <div><span className="text-zinc-500">ref:</span> {form.ref_function || '—'} ({form.reference_type}) block=[{form.ref_block_x}, {form.ref_block_y}, {form.ref_block_z}]</div>
                                        <div>
                                            <span className="text-zinc-500">boundary:</span>{' '}
                                            {boundaryOf(args).map(a => a.name).join(', ') || 'none'}
                                        </div>
                                        <div>
                                            <span className="text-zinc-500">validated:</span>{' '}
                                            {args.find(a => a.kind === 'buffer' && a.validate)?.name ?? (
                                                <span className="text-amber-400">none — exactly one buffer must be validated</span>
                                            )}
                                        </div>
                                        <div><span className="text-zinc-500">grid:</span> [{form.grid_x}, {form.grid_y}, {form.grid_z}] ({form.global_size_type})  tolerance={form.tolerance}  tuner={form.tuning?.duration_s ?? 100}s</div>
                                    </div>
                                </div>

                                {/* Generated inputs.hpp — rendered by the backend, so what you see is what is written */}
                                <div className="space-y-2">
                                    <Label>Generated inputs.hpp</Label>
                                    <pre className="rounded-md border bg-zinc-950 p-3 text-[11px] font-mono text-emerald-300 max-h-72 overflow-auto whitespace-pre">
                                        {previewHpp || '// …'}
                                    </pre>
                                </div>
                            </div>
                        )}

                        {error && (
                            <p className="text-sm text-destructive">{error}</p>
                        )}

                        {/* Saved, but something needs looking at. A signature mismatch cannot
                            be blocked (the parse is best-effort), so it must be read. */}
                        {warnings.length > 0 && (
                            <div className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 space-y-1">
                                <div className="flex items-center gap-2 text-xs font-medium text-amber-200">
                                    <AlertTriangle size={14} />
                                    Saved with warnings
                                </div>
                                {warnings.map((w, i) => (
                                    <p key={i} className="text-xs text-amber-100/90 pl-5">{w}</p>
                                ))}
                                <div className="pl-5 pt-1">
                                    <Button size="sm" variant="outline" className="h-6 text-xs"
                                        onClick={() => { setOpen(false); reset(); onCreated?.(form.slug); }}>
                                        Close anyway
                                    </Button>
                                </div>
                            </div>
                        )}
                        </div>

                        <DialogFooter className="flex items-center justify-between">
                            <div>
                                {step > 0 && (
                                    <Button variant="outline" size="sm" onClick={() => setStep(step - 1)}>
                                        <ChevronLeft size={14} className="mr-1" /> Back
                                    </Button>
                                )}
                            </div>
                            <div className="flex gap-2">
                                <Button variant="outline" size="sm" onClick={() => { setOpen(false); reset(); }}>
                                    Cancel
                                </Button>
                                {step < STEPS.length - 1 ? (
                                    <Button size="sm" onClick={() => setStep(step + 1)}>
                                        Next <ChevronRight size={14} className="ml-1" />
                                    </Button>
                                ) : (
                                    <Button size="sm" onClick={handleSubmit} disabled={submitting || !form.slug || !form.name || !form.ref_function || (form.reference_type === 'cuda' ? !form.ref_kernel_code : !form.ref_cpu_code)}>
                                        {submitting && <Loader2 size={14} className="mr-1 animate-spin" />}
                                        {mode === 'create' ? 'Create Problem' : 'Save Changes'}
                                    </Button>
                                )}
                            </div>
                        </DialogFooter>
                    </>
                )}
            </DialogContent>
        </Dialog>
    );
}
