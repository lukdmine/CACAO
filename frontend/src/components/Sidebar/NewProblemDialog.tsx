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
import { createProblem, updateProblem, fetchProblemDetail, fetchGpuDevices, type CreateProblemData, type GpuDevice } from '@/api/client';
import { refreshProblems } from '@/api/hooks';
import { Plus, Trash2, ChevronLeft, ChevronRight, ChevronUp, ChevronDown, Loader2 } from 'lucide-react';

// ── Default form state ───────────────────────────────────────────────────────
function defaultForm(): CreateProblemData {
    return {
        slug: '', name: '', description: '',
        gpu: { index: 0 },
        reference_type: 'cuda',
        ref_function: '', ref_block_x: 256, ref_block_y: 1, ref_block_z: 1,
        ref_kernel_code: '',
        ref_cpu_code: '',
        scalars: [{ name: 'N', dtype: 'int', value: 1024 }],
        vectors: [{ name: 'input', dtype: 'float', size: 'N', access: 'read', init: 'random', init_min: null, init_max: null, validate: false },
        { name: 'output', dtype: 'float', size: 'N', access: 'write', init: 'zeros', init_min: null, init_max: null, validate: true }],
        grid_x: 'N', grid_y: '1', grid_z: '1',
        tolerance: 0.05,
        tuning: { duration_s: 100 },
    };
}

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

    function update<K extends keyof CreateProblemData>(key: K, value: CreateProblemData[K]) {
        setForm((prev) => ({ ...prev, [key]: value }));
    }

    function reset() {
        setForm(defaultForm());
        setStep(0);
        setError('');
    }

    async function loadProblemData() {
        if (mode !== 'edit' || !editProblemName) return;
        setLoading(true);
        try {
            const data = await fetchProblemDetail(editProblemName);
            const loadedForm: CreateProblemData = {
                slug: data.name,
                name: (data.config.name as string) || data.name,
                description: (data.config.description as string) || '',
                gpu: (data.config.gpu as CreateProblemData['gpu']) || defaultForm().gpu,
                reference_type: ((data.config.reference as any)?.type as CreateProblemData['reference_type']) || 'cuda',
                ref_function: ((data.config.reference as any)?.function as string) || '',
                ref_block_x: ((data.config.reference as any)?.block_x as number) || 256,
                ref_block_y: ((data.config.reference as any)?.block_y as number) || 1,
                ref_block_z: ((data.config.reference as any)?.block_z as number) || 1,
                ref_kernel_code: data.ref_kernel || '',
                ref_cpu_code: data.ref_cpu || '',
                scalars: (data.config.scalars as CreateProblemData['scalars']) || [],
                vectors: ((data.config.vectors as CreateProblemData['vectors']) || []).map(v => ({ ...v, size: String(v.size) })),
                grid_x: ((data.config.grid as any)?.x as string) || 'N',
                grid_y: ((data.config.grid as any)?.y as string) || '1',
                grid_z: ((data.config.grid as any)?.z as string) || '1',
                tolerance: ((data.config.validation as any)?.tolerance as number) || 0.05,
                tuning: {
                    duration_s:
                        (data.config.tuning as { duration_s?: number } | undefined)?.duration_s ?? 100,
                },
            };
            setForm(loadedForm);
        } catch (err) {
            setError('Failed to load problem data: ' + (err instanceof Error ? err.message : String(err)));
        } finally {
            setLoading(false);
        }
    }

    async function handleSubmit() {
        setError('');
        setSubmitting(true);
        try {
            if (mode === 'create') {
                await createProblem(form);
            } else {
                await updateProblem(editProblemName!, form);
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

                                <div className="space-y-1.5">
                                    <Label className="text-xs">Reference Type</Label>
                                    <select
                                        className="h-8 rounded border bg-background px-2 text-xs"
                                        value={form.reference_type}
                                        onChange={(e) => update('reference_type', e.target.value as CreateProblemData['reference_type'])}
                                    >
                                        <option value="cuda">CUDA reference kernel</option>
                                        <option value="cpu_c">CPU C/C++ reference</option>
                                    </select>
                                </div>

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
                            <div className="space-y-4 py-2">
                                <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-100">
                                    Vector order matters. Vectors are passed to kernels and CPU references in the exact order shown here. Use the up/down buttons to reorder them. Scalars on the other hand are passed as defines (not arguments - order does not matter).
                                </div>

                                {/* Scalars */}
                                <div>
                                    <div className="flex items-center justify-between mb-2">
                                        <Label>Scalar Arguments</Label>
                                        <Button size="sm" variant="outline" className="text-xs h-6"
                                            onClick={() => update('scalars', [...form.scalars, { name: '', dtype: 'int', value: 0 }])}>
                                            <Plus size={12} className="mr-1" /> Add
                                        </Button>
                                    </div>
                                    {form.scalars.map((s, i) => (
                                        <div key={i} className="flex items-center gap-2 mb-1.5">
                                            <Input className="text-xs font-mono flex-1" placeholder="Name" value={s.name}
                                                onChange={(e) => {
                                                    const scalars = [...form.scalars];
                                                    scalars[i] = { ...s, name: e.target.value };
                                                    update('scalars', scalars);
                                                }} />
                                            <select className="h-8 rounded border bg-background px-2 text-xs"
                                                value={s.dtype} onChange={(e) => {
                                                    const scalars = [...form.scalars];
                                                    scalars[i] = { ...s, dtype: e.target.value };
                                                    update('scalars', scalars);
                                                }}>
                                                <option value="int">int</option>
                                                <option value="float">float</option>
                                            </select>
                                            <Input type="number" className="text-xs w-24" value={s.value}
                                                onChange={(e) => {
                                                    const scalars = [...form.scalars];
                                                    scalars[i] = { ...s, value: parseFloat(e.target.value) || 0 };
                                                    update('scalars', scalars);
                                                }} />
                                            <button className="text-muted-foreground hover:text-destructive"
                                                onClick={() => update('scalars', form.scalars.filter((_, j) => j !== i))}>
                                                <Trash2 size={14} />
                                            </button>
                                        </div>
                                    ))}
                                    {form.scalars.length === 0 && (
                                        <p className="text-xs text-muted-foreground italic">No scalars defined</p>
                                    )}
                                </div>

                                <Separator />

                                {/* Vectors */}
                                <div>
                                    <div className="flex items-center justify-between mb-2">
                                        <Label>Vector Arguments</Label>
                                        <Button size="sm" variant="outline" className="text-xs h-6"
                                            onClick={() => update('vectors', [...form.vectors, { name: '', dtype: 'float', size: 'N', access: 'read', init: 'random', init_min: null, init_max: null, validate: false }])}>
                                            <Plus size={12} className="mr-1" /> Add
                                        </Button>
                                    </div>
                                    {form.vectors.map((v, i) => (
                                        <div key={i} className="rounded-md border border-border/50 p-2 mb-2 space-y-1.5">
                                            {/* Row 1: name, dtype, size, access, reorder, delete */}
                                            <div className="flex items-center gap-2">
                                                <div className="flex flex-col gap-0.5">
                                                    <button
                                                        type="button"
                                                        className="text-muted-foreground hover:text-foreground disabled:opacity-40"
                                                        disabled={i === 0}
                                                        onClick={() => update('vectors', moveItem(form.vectors, i, i - 1))}
                                                        aria-label={`Move vector ${v.name || i + 1} up`}
                                                    >
                                                        <ChevronUp size={14} />
                                                    </button>
                                                    <button
                                                        type="button"
                                                        className="text-muted-foreground hover:text-foreground disabled:opacity-40"
                                                        disabled={i === form.vectors.length - 1}
                                                        onClick={() => update('vectors', moveItem(form.vectors, i, i + 1))}
                                                        aria-label={`Move vector ${v.name || i + 1} down`}
                                                    >
                                                        <ChevronDown size={14} />
                                                    </button>
                                                </div>
                                                <Input className="text-xs font-mono flex-1 min-w-0" placeholder="Name" value={v.name}
                                                    onChange={(e) => {
                                                        const vectors = [...form.vectors];
                                                        vectors[i] = { ...v, name: e.target.value };
                                                        update('vectors', vectors);
                                                    }} />
                                                <select className="h-8 rounded border bg-background px-2 text-xs"
                                                    value={v.dtype} onChange={(e) => {
                                                        const vectors = [...form.vectors];
                                                        vectors[i] = { ...v, dtype: e.target.value };
                                                        update('vectors', vectors);
                                                    }}>
                                                    <option value="float">float</option>
                                                    <option value="double">double</option>
                                                    <option value="int">int</option>
                                                </select>
                                                <Input className="text-xs font-mono w-24" placeholder="Size expr" value={v.size}
                                                    onChange={(e) => {
                                                        const vectors = [...form.vectors];
                                                        vectors[i] = { ...v, size: e.target.value };
                                                        update('vectors', vectors);
                                                    }} />
                                                <select className="h-8 rounded border bg-background px-2 text-xs"
                                                    value={v.access} onChange={(e) => {
                                                        const vectors = [...form.vectors];
                                                        const newAccess = e.target.value;
                                                        const newInit = newAccess === 'write' ? 'zeros' : v.init;
                                                        vectors[i] = { ...v, access: newAccess, init: newInit };
                                                        update('vectors', vectors);
                                                    }}>
                                                    <option value="read">read</option>
                                                    <option value="write">write</option>
                                                </select>
                                                <button className="text-muted-foreground hover:text-destructive"
                                                    onClick={() => update('vectors', form.vectors.filter((_, j) => j !== i))}>
                                                    <Trash2 size={14} />
                                                </button>
                                            </div>
                                            {/* Row 2: init, init_min/max, validate */}
                                            <div className="flex items-center gap-2 pl-7">
                                                <select className="h-7 rounded border bg-background px-2 text-xs"
                                                    value={v.init} onChange={(e) => {
                                                        const vectors = [...form.vectors];
                                                        vectors[i] = { ...v, init: e.target.value };
                                                        update('vectors', vectors);
                                                    }}>
                                                    <option value="zeros">zeros</option>
                                                    <option value="random">random</option>
                                                </select>
                                                {v.init === 'random' && (
                                                    <>
                                                        <span className="text-xs text-muted-foreground">range:</span>
                                                        <Input type="number" className="text-xs w-20 h-7" placeholder="min"
                                                            value={v.init_min ?? ''}
                                                            onChange={(e) => {
                                                                const vectors = [...form.vectors];
                                                                vectors[i] = { ...v, init_min: e.target.value === '' ? null : parseFloat(e.target.value) };
                                                                update('vectors', vectors);
                                                            }}
                                                            title="Random init minimum (optional)"
                                                        />
                                                        <span className="text-xs text-muted-foreground">to</span>
                                                        <Input type="number" className="text-xs w-20 h-7" placeholder="max"
                                                            value={v.init_max ?? ''}
                                                            onChange={(e) => {
                                                                const vectors = [...form.vectors];
                                                                vectors[i] = { ...v, init_max: e.target.value === '' ? null : parseFloat(e.target.value) };
                                                                update('vectors', vectors);
                                                            }}
                                                            title="Random init maximum (optional, inclusive)"
                                                        />
                                                    </>
                                                )}
                                                <label className="flex items-center gap-1 text-xs ml-auto">
                                                    <input type="checkbox" checked={v.validate}
                                                        onChange={(e) => {
                                                            const vectors = [...form.vectors];
                                                            vectors[i] = { ...v, validate: e.target.checked };
                                                            update('vectors', vectors);
                                                        }} />
                                                    validate
                                                </label>
                                            </div>
                                        </div>
                                    ))}
                                </div>
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
                                        <div><span className="text-zinc-500">ref:</span> {form.ref_function || '—'}  block=[{form.ref_block_x}, {form.ref_block_y}, {form.ref_block_z}]</div>
                                        <div><span className="text-zinc-500">scalars:</span> {form.scalars.map(s => `${s.name}=${s.value}`).join(', ') || 'none'}</div>
                                        <div><span className="text-zinc-500">vectors:</span> {form.vectors.map(v => `${v.name}(${v.access})`).join(', ') || 'none'}</div>
                                        <div><span className="text-zinc-500">grid:</span> [{form.grid_x}, {form.grid_y}, {form.grid_z}]  tolerance={form.tolerance}  tuner={form.tuning?.duration_s ?? 100}s</div>
                                        <div><span className="text-zinc-500">code:</span> {form.ref_kernel_code ? `${form.ref_kernel_code.split('\n').length} lines` : '—'}</div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {error && (
                            <p className="text-sm text-destructive">{error}</p>
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
