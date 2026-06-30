import { useMemo, useCallback, useEffect } from 'react';
import {
    ReactFlow,
    Background,
    Controls,
    BackgroundVariant,
    useNodesState,
    useEdgesState,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { useAppStore } from '@/store/appStore';
import { buildTreeLayout } from './treeLayout';
import { StrategyNode } from './StrategyNode';
import { RootNode } from './RootNode';

const nodeTypes = {
    strategyNode: StrategyNode,
    rootNode: RootNode,
};

export function TreeCanvas() {
    const treeNodes = useAppStore((s) => s.treeNodes);
    const selectNode = useAppStore((s) => s.selectNode);

    const { nodes: layoutNodes, edges: layoutEdges } = useMemo(
        () => buildTreeLayout(treeNodes),
        [treeNodes]
    );

    const [nodes, setNodes, onNodesChange] = useNodesState(layoutNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(layoutEdges);

    // Sync React Flow state when store data changes (e.g. from API polling)
    useEffect(() => {
        setNodes(layoutNodes);
        setEdges(layoutEdges);
    }, [layoutNodes, layoutEdges, setNodes, setEdges]);

    const onPaneClick = useCallback(() => {
        selectNode(null);
    }, [selectNode]);

    return (
        <div className="h-full w-full">
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onPaneClick={onPaneClick}
                nodeTypes={nodeTypes}
                fitView
                fitViewOptions={{ padding: 0.2 }}
                minZoom={0.3}
                maxZoom={2}
                proOptions={{ hideAttribution: true }}
            >
                <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="hsl(var(--muted-foreground) / 0.15)" />
                <Controls className="!bg-card !border-border !shadow-md [&>button]:!bg-card [&>button]:!border-border [&>button]:!text-foreground" />
            </ReactFlow>
        </div>
    );
}
