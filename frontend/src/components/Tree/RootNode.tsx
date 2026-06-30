import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import type { TreeNode } from '@/api/types.generated';
import { formatTime, formatSpeedup } from '@/utils/statusColors';
import { useAppStore } from '@/store/appStore';
import { Cpu, Timer, Zap } from 'lucide-react';

function RootNodeComponent({ data, id }: NodeProps) {
    const node = data as unknown as TreeNode;
    const selectedNodeId = useAppStore((s) => s.selectedNodeId);
    const selectNode = useAppStore((s) => s.selectNode);
    const isSelected = selectedNodeId === id;

    return (
        <div
            onClick={() => selectNode(id)}
            className={`
        w-[280px] rounded-xl border-2 cursor-pointer transition-all duration-200
        bg-primary text-primary-foreground shadow-lg hover:shadow-xl
        ${isSelected ? 'ring-2 ring-ring' : 'border-primary'}
      `}
        >
            <div className="p-4 space-y-2">
                {/* Header */}
                <div className="flex items-center gap-2">
                    <Cpu size={18} />
                    <span className="font-bold text-base">{node.strategy.name}</span>
                </div>

                <p className="text-xs opacity-80 line-clamp-2">{node.strategy.description}</p>

                {/* Best result */}
                {node.best_time_us !== null && (
                    <div className="flex items-center gap-4 text-sm pt-1">
                        <div className="flex items-center gap-1">
                            <Timer size={14} />
                            <span className="font-mono">{formatTime(node.best_time_us)}</span>
                        </div>
                        {node.speedup !== null && (
                            <div className="flex items-center gap-1">
                                <Zap size={14} />
                                <span className="font-bold">{formatSpeedup(node.speedup)}</span>
                            </div>
                        )}
                    </div>
                )}
            </div>

            <Handle type="source" position={Position.Bottom} className="!bg-primary-foreground !w-3 !h-3" />
        </div>
    );
}

export const RootNode = memo(RootNodeComponent);
