import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import type { TreeNode } from '@/api/types.generated';
import { Facts } from '@/components/Facts';
import { formatTime, formatSpeedup } from '@/utils/statusColors';
import { useAppStore } from '@/store/appStore';
import { Cpu } from 'lucide-react';

function RootNodeComponent({ data, id }: NodeProps) {
    const node = data as unknown as TreeNode;
    const selectedNodeId = useAppStore((s) => s.selectedNodeId);
    const selectNode = useAppStore((s) => s.selectNode);
    const isSelected = selectedNodeId === id;

    // The same card as every strategy, told apart by what it says and a firmer
    // border — not an inverted slab, which was the loudest thing on the canvas.
    return (
        <div
            onClick={() => selectNode(id)}
            className={`w-[280px] cursor-pointer rounded-md border bg-card text-card-foreground transition-colors ${
                isSelected ? 'border-foreground/70' : 'border-foreground/30 hover:border-foreground/50'
            }`}
        >
            <div className="space-y-1.5 p-3">
                <div className="flex items-center gap-2">
                    <Cpu size={14} className="shrink-0 text-muted-foreground" />
                    <span className="truncate text-sm font-medium">{node.strategy.name}</span>
                </div>

                <p className="line-clamp-2 text-xs text-muted-foreground">{node.strategy.description}</p>

                {node.best_time_us !== null && (
                    <Facts
                        className="pt-0.5 text-xs text-muted-foreground"
                        items={[
                            <span key="time" className="font-mono">{formatTime(node.best_time_us)}</span>,
                            node.speedup !== null && (
                                <span key="speedup" className="font-mono font-medium text-foreground">
                                    {formatSpeedup(node.speedup)}
                                </span>
                            ),
                        ]}
                    />
                )}
            </div>

            <Handle type="source" position={Position.Bottom} className="!bg-muted-foreground !w-2 !h-2" />
        </div>
    );
}

export const RootNode = memo(RootNodeComponent);
