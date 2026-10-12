import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import type { TreeNode } from '@/api/types.generated';
import { Facts } from '@/components/Facts';
import { getStatusStyle, formatTime, formatSpeedup } from '@/utils/statusColors';
import { useAppStore } from '@/store/appStore';

function StrategyNodeComponent({ data, id }: NodeProps) {
    const node = data as unknown as TreeNode;
    const selectedNodeId = useAppStore((s) => s.selectedNodeId);
    const selectNode = useAppStore((s) => s.selectNode);
    const style = getStatusStyle(node.status);
    const isSelected = selectedNodeId === id;

    return (
        <div
            onClick={() => selectNode(id)}
            className={`w-[260px] cursor-pointer rounded-md border bg-card text-card-foreground transition-colors ${
                isSelected ? 'border-foreground/60' : 'border-border hover:border-foreground/30'
            } ${style.animate ? 'animate-pulse-subtle' : ''}`}
        >
            <Handle type="target" position={Position.Top} className="!bg-muted-foreground !w-2 !h-2" />

            <div className="space-y-1.5 p-3">
                <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-sm font-medium">{node.strategy.name}</span>
                    <span className={`flex shrink-0 items-center gap-1 text-[10px] ${style.text}`}>
                        <span
                            className={`inline-block size-1.5 rounded-full ${style.color} ${
                                style.animate ? 'animate-pulse' : ''
                            }`}
                        />
                        {style.label}
                    </span>
                </div>

                <Facts
                    className="text-xs text-muted-foreground"
                    items={[
                        node.iter_num > 0 && <span key="iter">iter {node.iter_num}/{node.max_iter}</span>,
                        node.best_time_us !== null && (
                            <span key="time" className="font-mono">{formatTime(node.best_time_us)}</span>
                        ),
                        node.speedup !== null && (
                            <span key="speedup" className="font-mono font-medium text-foreground">
                                {formatSpeedup(node.speedup)}
                            </span>
                        ),
                    ]}
                />
            </div>

            <Handle type="source" position={Position.Bottom} className="!bg-muted-foreground !w-2 !h-2" />
        </div>
    );
}

export const StrategyNode = memo(StrategyNodeComponent);
