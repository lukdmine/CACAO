import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import type { TreeNode } from '@/api/types.generated';
import { Badge } from '@/components/ui/badge';
import { getStatusStyle, formatTime, formatSpeedup } from '@/utils/statusColors';
import { useAppStore } from '@/store/appStore';
import { Timer, Zap, IterationCcw } from 'lucide-react';

function StrategyNodeComponent({ data, id }: NodeProps) {
    const node = data as unknown as TreeNode;
    const selectedNodeId = useAppStore((s) => s.selectedNodeId);
    const selectNode = useAppStore((s) => s.selectNode);
    const style = getStatusStyle(node.status);
    const isSelected = selectedNodeId === id;

    return (
        <div
            onClick={() => selectNode(id)}
            className={`
        w-[260px] rounded-lg border-2 cursor-pointer transition-all duration-200
        bg-card text-card-foreground shadow-md hover:shadow-lg
        ${isSelected ? 'border-primary ring-2 ring-primary/20' : style.border}
        ${style.animate ? 'animate-pulse-subtle' : ''}
      `}
        >
            <Handle type="target" position={Position.Top} className="!bg-muted-foreground !w-2 !h-2" />

            <div className="p-3 space-y-2">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <span className="font-semibold text-sm truncate flex-1">{node.strategy.name}</span>
                    <Badge
                        variant="outline"
                        className={`text-[10px] px-1.5 py-0 ${style.text} ${style.border}`}
                    >
                        <span className={`w-1.5 h-1.5 rounded-full ${style.color} mr-1 inline-block ${style.animate ? 'animate-pulse' : ''}`} />
                        {style.label}
                    </Badge>
                </div>

                {/* Metrics row */}
                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                    {node.iter_num > 0 && (
                        <div className="flex items-center gap-1">
                            <IterationCcw size={12} />
                            <span>{node.iter_num}/{node.max_iter}</span>
                        </div>
                    )}
                    {node.best_time_us !== null && (
                        <div className="flex items-center gap-1">
                            <Timer size={12} />
                            <span>{formatTime(node.best_time_us)}</span>
                        </div>
                    )}
                    {node.speedup !== null && (
                        <div className="flex items-center gap-1">
                            <Zap size={12} />
                            <span className="text-emerald-400 font-medium">{formatSpeedup(node.speedup)}</span>
                        </div>
                    )}
                </div>

            </div>

            <Handle type="source" position={Position.Bottom} className="!bg-muted-foreground !w-2 !h-2" />
        </div>
    );
}

export const StrategyNode = memo(StrategyNodeComponent);
