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
import { createProblem, updateProblem, fetchProblemDetail, fetchGpuDevices, previewInputs, uploadProblemInput, type CreateProblemData, type GpuDevice, type ArgSpec, type BufferSpec, type ScalarSpec, type Placement, type ReferenceType, type ProblemDetailResponse } from '@/api/client';
import { refreshProblems } from '@/api/hooks';
import { Plus, Trash2, ChevronLeft, ChevronRight, ChevronUp, ChevronDown, Loader2, AlertTriangle, Info, Upload } from 'lucide-react';

// ── Default form state ───────────────────────────────────────────────────────
function defaultForm(): CreateProblemData {
    return {
        slug: '', name: '', description: '',
        gpu: { index: 0 },
        reference_type: 'cuda',
        ref_function: '', ref_block_x: 256, ref_block_y: 1, ref_block_z: 1,
        ref_kernel_code: '',
        ref_cpu_code: '',
        ref_python_code: '',
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

/** What each reference kind is, and what it receives. One table rather than a branch per
 *  call site — the previous two-way splits were spread across ten of them. */
const REFERENCE_KINDS: Record<ReferenceType, {
    label: string;
    field: 'ref_kernel_code' | 'ref_cpu_code' | 'ref_python_code';
    codeLabel: string;
    placeholder: string;
    /** Whether the reference binds its arguments positionally. Python does not. */
    positional: boolean;
    note: string;
}> = {
    cuda: {
        label: 'CUDA reference kernel',
        field: 'ref_kernel_code',
        codeLabel: 'Reference Kernel Code *',
        placeholder: 'extern "C" __global__ void my_kernel(...) {\n  // Reference implementation\n}',
        positional: true,
        note: '',
    },
    cpu_c: {
        label: 'CPU C/C++ reference',
        field: 'ref_cpu_code',
        codeLabel: 'CPU Reference Code *',
        placeholder: 'void reference(const float* in, float* out) {\n  // Runs on the host\n}',
        positional: true,
        note: 'Compiled and linked into the driver, and run on the host once per validated buffer. '
            + 'It takes every buffer as a pointer, in the order listed under Arguments, and reads '
            + 'scalars as -D macros — so scalars are not parameters, and block size does not apply.',
    },
    python: {
        label: 'Python reference',
        field: 'ref_python_code',
        codeLabel: 'Python Reference Code *',
        placeholder: 'import numpy as np\n\ndef reference(scalars, buffers):\n'
            + '    # scalars: {"N": 1024, ...}   buffers: {"input": np.ndarray, ...}\n'
            + '    return buffers["input"] * 2   # flat array for the validated buffer',
        positional: false,
        note: 'Run on the host once per validated buffer, via a subprocess. It receives '
            + 'def f(scalars, buffers) — two dicts keyed by NAME, with buffers as numpy arrays — '
            + 'and returns the validated buffer as a flat array. Argument order does not apply, '
            + 'and block size does not either. Requires numpy.',
    },
};

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
    // Picked binaries for init=file buffers, keyed by buffer name; uploaded after save.
    const [inputFiles, setInputFiles] = useState<Record<string, File>>({});
    // What the server already holds for each file buffer (edit mode only).
    const [serverInputFiles, setServerInputFiles] = useState<ProblemDetailResponse['input_files']>({});

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
            previewInputs(form.inputs, form.reference_type, form.ref_function)
                .then((r) => setPreviewHpp(r.inputs_hpp))
                .catch((e) => setPreviewHpp(`// ${e instanceof Error ? e.message : String(e)}`));
        }, 300);
        return () => clearTimeout(id);
    }, [open, step, form.inputs, form.reference_type, form.ref_function]);

    function update<K extends keyof CreateProblemData>(key: K, value: CreateProblemData[K]) {
        setForm((prev) => ({ ...prev, [key]: value }));
    }

    function reset() {
        setForm(defaultForm());
        setStep(0);
        setError('');
        setWarnings([]);
        setPreviewHpp('');
        setInputFiles({});
        setServerInputFiles({});
    }

    async function loadProblemData() {
        if (mode !== 'edit' || !editProblemName) return;
        setLoading(true);
        try {
            const data = await fetchProblemDetail(editProblemName);
            setServerInputFiles(data.input_files ?? {});
            const ref = (data.config.reference ?? {}) as { type?: ReferenceType; function?: string; block?: { x?: number; y?: number; z?: number } };
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
                ref_python_code: data.ref_python || '',
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

    const refKind = REFERENCE_KINDS[form.reference_type];

    // What the reference is handed, in order. A CUDA kernel takes the boundary (buffers +
    // runtime scalars); a C reference takes every buffer as a pointer with scalars as -D
    // macros. A python reference takes dicts keyed by name, so order does not apply and
    // there is no positional signature to preview.
    const refParams = form.reference_type === 'cuda'
        ? boundaryOf(args)
        : args.filter((a) => a.kind === 'buffer');

    function setArgs(next: ArgSpec[]) {
        setForm((prev) => ({ ...prev, inputs: { ...prev.inputs, args: next } }));
    }

    function patchArg(i: number, patch: Partial<ScalarSpec> & Partial<BufferSpec>) {
        setArgs(args.map((a, j) => (j === i ? ({ ...a, ...patch } as ArgSpec) : a)));
    }

    /** Buffers checked against the reference — at least one; a multi-output kernel has several. */
    function toggleValidated(i: number) {
        setArgs(args.map((a, j) => (j === i && a.kind === 'buffer' ? { ...a, validate: !a.validate } : a)));
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

    // Picked files are keyed by buffer NAME (it rides along through reorder/remove).
    // A rename migrates the key; a duplicate/empty name can only exist mid-typing —
    // the server rejects duplicates on save, before any upload runs.
    function pickInputFile(bufferName: string, e: React.ChangeEvent<HTMLInputElement>) {
        const f = e.target.files?.[0];
        e.target.value = ''; // re-picking the same file must fire onChange again
        if (!f || !bufferName) return;
        setInputFiles((prev) => ({ ...prev, [bufferName]: f }));
        const a = args.find((x) => x.name === bufferName);
        if (a?.kind === 'buffer' && !a.file_name) {
            patchArg(args.indexOf(a), { file_name: f.name });
        }
    }

    function renameArg(i: number, newName: string) {
        const oldName = args[i]?.name;
        patchArg(i, { name: newName });
        if (oldName && oldName !== newName && inputFiles[oldName]) {
            setInputFiles((prev) => {
                const next = { ...prev };
                next[newName] = next[oldName];
                delete next[oldName];
                return next;
            });
        }
    }

    function setInit(i: number, init: BufferSpec['init']) {
        patchArg(i, { init });
        if (init !== 'file') {
            const name = args[i]?.name;
            if (name) setInputFiles((prev) => {
                const next = { ...prev };
                delete next[name];
                return next;
            });
        }
    }

    async function handleSubmit() {
        setError('');
        setSubmitting(true);
        try {
            const res = mode === 'create'
                ? await createProblem(form)
                : await updateProblem(editProblemName!, form);

            // Upload the picked init=file binaries now that the problem is saved:
            // the server derives file_name from the spec it just persisted and
            // byte-checks against size × sizeof(dtype). Targets are the spec's own
            // buffers, so a removed arg's stale pick is simply never sent.
            const uploaded: string[] = [];
            const uploadWarnings: string[] = [];
            for (const a of form.inputs.args) {
                if (a.kind !== 'buffer' || a.init !== 'file') continue;
                const f = inputFiles[a.name];
                if (!f) continue;
                try {
                    await uploadProblemInput(res.name, a.name, f);
                    uploaded.push(a.name);
                } catch (err) {
                    uploadWarnings.push(
                        `'${a.name}': ${err instanceof Error ? err.message : String(err)}`
                    );
                }
            }
            if (uploaded.length) {
                setInputFiles((prev) => {
                    const next = { ...prev };
                    for (const n of uploaded) delete next[n];
                    return next;
                });
            }

            // Saved either way: a signature mismatch or failed upload is reported, not
            // blocked. Keep the dialog open so it is read (and the upload retried).
            const allWarnings = [...(res.warnings ?? []), ...uploadWarnings];
            if (allWarnings.length) {
                setWarnings(allWarnings);
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

    function fmtBytes(n: number | null): string {
        if (n === null) return '';
        const units = ['B', 'KB', 'MB', 'GB'];
        let v = n;
        let u = 0;
        while (v >= 1024 && u < units.length - 1) { v /= 1024; u += 1; }
        return `${u === 0 ? v : v.toFixed(1)} ${units[u]}`;
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
                                            onChange={(e) => update('reference_type', e.target.value as ReferenceType)}
                                        >
                                            {(Object.keys(REFERENCE_KINDS) as ReferenceType[]).map((k) => (
                                                <option key={k} value={k}>{REFERENCE_KINDS[k].label}</option>
                                            ))}
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

                                {refKind.note && (
                                    <div className="rounded-md border border-sky-500/30 bg-sky-500/10 px-3 py-2 text-xs text-sky-100 flex gap-2">
                                        <Info size={14} className="shrink-0 mt-0.5" />
                                        <span>{refKind.note}</span>
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
                                    <Label>{refKind.codeLabel}</Label>
                                    <textarea
                                        className="w-full rounded-md border bg-zinc-950 text-zinc-100 font-mono text-xs px-3 py-2 min-h-[200px] resize-y"
                                        placeholder={refKind.placeholder}
                                        value={form[refKind.field]}
                                        onChange={(e) => update(refKind.field, e.target.value)}
                                        spellCheck={false}
                                    />
                                </div>
                            </div>
                        )}

                        {/* ── Step 3: Arguments ─────────────────────────────────────────── */}
                        {step === 2 && (
                            <div className="space-y-3 py-2">
                                <div className={`rounded-md border px-3 py-2 text-xs flex gap-2 ${refKind.positional
                                    ? 'border-amber-500/30 bg-amber-500/10 text-amber-100'
                                    : 'border-sky-500/30 bg-sky-500/10 text-sky-100'}`}>
                                    {refKind.positional
                                        ? <AlertTriangle size={14} className="shrink-0 mt-0.5" />
                                        : <Info size={14} className="shrink-0 mt-0.5" />}
                                    <span>
                                        {form.reference_type === 'cuda' && (
                                            <><strong>Order is the reference's signature.</strong> Arguments bind by
                                            position, not by name — a wrong order makes the reference compute from
                                            shuffled inputs and validation compare against garbage, with no error.
                                            Buffers, and scalars marked <code className="font-mono">runtime</code>, are
                                            passed; a host-only scalar is declared but not passed, so it may sit anywhere.</>
                                        )}
                                        {form.reference_type === 'cpu_c' && (
                                            <><strong>Order is the reference's signature.</strong> Arguments bind by
                                            position, not by name — a wrong order makes the reference compute from
                                            shuffled inputs and validation compare against garbage, with no error. A C
                                            reference takes <strong>every buffer</strong> as a pointer; scalars are{' '}
                                            <code className="font-mono">-D</code> macros, never parameters.</>
                                        )}
                                        {form.reference_type === 'python' && (
                                            <><strong>Order does not matter for a Python reference.</strong> It receives{' '}
                                            <code className="font-mono">scalars</code> and <code className="font-mono">buffers</code>{' '}
                                            as dicts keyed by name, so it reads what it needs. Order still defines the
                                            kernel's own signature, which the configure step binds by position.</>
                                        )}
                                    </span>
                                </div>

                                {/* What the reference actually receives — each kind takes something different */}
                                <div className="rounded-md border bg-zinc-950 px-3 py-2 text-xs font-mono text-zinc-300">
                                    {refKind.positional ? (
                                        <>
                                            <span className="text-zinc-500">
                                                {form.reference_type === 'cuda' ? '__global__ void ' : 'void '}
                                                {form.ref_function || 'reference'}(
                                            </span>
                                            {refParams.map((a, i) => (
                                                <span key={a.name + i}>
                                                    {i > 0 && <span className="text-zinc-500">, </span>}
                                                    <span className={a.kind === 'scalar' ? 'text-sky-400' : 'text-emerald-400'}>
                                                        {a.kind === 'buffer' && form.reference_type === 'cpu_c'
                                                            ? `${a.access === 'read' ? 'const ' : ''}${a.dtype}* ${a.name || '?'}`
                                                            : a.name || '?'}
                                                    </span>
                                                </span>
                                            ))}
                                            <span className="text-zinc-500">)</span>
                                            {refParams.length === 0 && <span className="text-zinc-600 italic"> — nothing passed</span>}
                                        </>
                                    ) : (
                                        // Python takes dicts keyed by name — there is no positional signature to show,
                                        // so show the keys it will actually find in them.
                                        <>
                                            <span className="text-zinc-500">def {form.ref_function || 'reference'}(scalars, buffers)</span>
                                            <div className="mt-1 text-[11px]">
                                                <span className="text-zinc-500">scalars: </span>
                                                {args.filter(a => a.kind === 'scalar').map(a => a.name).join(', ') || <span className="text-zinc-600 italic">none</span>}
                                            </div>
                                            <div className="text-[11px]">
                                                <span className="text-zinc-500">buffers: </span>
                                                {args.filter(a => a.kind === 'buffer').map(a => a.name).join(', ') || <span className="text-zinc-600 italic">none</span>}
                                            </div>
                                        </>
                                    )}
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
                                                onChange={(e) => renameArg(i, e.target.value)} />

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
                                                    onChange={(e) => setInit(i, e.target.value as BufferSpec['init'])}>
                                                    <option value="zeros">zeros</option>
                                                    <option value="random">random</option>
                                                    <option value="custom">custom</option>
                                                    <option value="file">file</option>
                                                </select>

                                                {a.init === 'file' && (
                                                    <>
                                                        <label className="h-7 inline-flex items-center gap-1 rounded border px-2 text-xs cursor-pointer hover:bg-accent"
                                                            title="Raw little-endian binary of the buffer dtype; byte count must equal size × sizeof(dtype). Uploaded when the problem is saved.">
                                                            <input type="file" className="hidden"
                                                                onChange={(e) => pickInputFile(a.name, e)} />
                                                            <Upload size={12} />
                                                            {inputFiles[a.name] ? inputFiles[a.name].name : 'choose file…'}
                                                        </label>
                                                        <Input className="text-xs font-mono w-40 h-7" placeholder="as inputs/…"
                                                            value={a.file_name ?? ''}
                                                            onChange={(e) => patchArg(i, { file_name: e.target.value || null })}
                                                            title={`Stored as problems/${form.slug || '<slug>'}/inputs/${a.file_name || '<file>'}`} />
                                                        {mode === 'edit' && serverInputFiles[a.name] && (
                                                            <span className={`text-[11px] ${serverInputFiles[a.name].exists ? 'text-emerald-400' : 'text-amber-400'}`}>
                                                                {serverInputFiles[a.name].exists
                                                                    ? `on server (${fmtBytes(serverInputFiles[a.name].bytes)})${inputFiles[a.name] ? ' — will be replaced' : ''}`
                                                                    : 'not uploaded yet'}
                                                            </span>
                                                        )}
                                                    </>
                                                )}

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

                                                {/* Several buffers may be validated — a multi-output kernel checks each */}
                                                <label className="flex items-center gap-1 text-xs ml-auto"
                                                    title="Compare this buffer against the reference. At least one; a multi-output kernel validates several.">
                                                    <input type="checkbox" checked={a.validate}
                                                        onChange={() => toggleValidated(i)} />
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
                                        <div>
                                            <span className="text-zinc-500">ref:</span> {form.ref_function || '—'} ({form.reference_type})
                                            {form.reference_type === 'cuda' && ` block=[${form.ref_block_x}, ${form.ref_block_y}, ${form.ref_block_z}]`}
                                        </div>
                                        <div>
                                            <span className="text-zinc-500">ref args:</span>{' '}
                                            {refKind.positional
                                                ? (refParams.map(a => a.name).join(', ') || 'none')
                                                : 'scalars + buffers dicts (by name)'}
                                        </div>
                                        <div>
                                            <span className="text-zinc-500">validated:</span>{' '}
                                            {args.filter(a => a.kind === 'buffer' && a.validate).map(a => a.name).join(', ') || (
                                                <span className="text-amber-400">none — at least one buffer must be validated</span>
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
                                    <Button size="sm" onClick={handleSubmit} disabled={submitting || !form.slug || !form.name || !form.ref_function || !form[refKind.field]}>
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
