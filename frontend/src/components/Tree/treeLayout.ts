import dagre from 'dagre';
import type { Node, Edge } from '@xyflow/react';
import type { TreeNode } from '@/api/types.generated';

const NODE_WIDTH = 260;
const NODE_HEIGHT = 120;
const ROOT_NODE_WIDTH = 280;
const ROOT_NODE_HEIGHT = 140;

/**
 * Convert our flat TreeNode[] into React Flow nodes + edges with dagre layout.
 */
export function buildTreeLayout(treeNodes: TreeNode[]): { nodes: Node[]; edges: Edge[] } {
    const g = new dagre.graphlib.Graph();
    g.setDefaultEdgeLabel(() => ({}));
    g.setGraph({ rankdir: 'TB', nodesep: 40, ranksep: 100 });

    // Add nodes — root gets larger dimensions to avoid overlap
    for (const node of treeNodes) {
        const isRoot = node.id === 'root';
        g.setNode(node.id, {
            width: isRoot ? ROOT_NODE_WIDTH : NODE_WIDTH,
            height: isRoot ? ROOT_NODE_HEIGHT : NODE_HEIGHT,
        });
    }

    // Add edges
    const edges: Edge[] = [];
    for (const node of treeNodes) {
        if (node.parentId) {
            const edgeId = `${node.parentId}->${node.id}`;
            g.setEdge(node.parentId, node.id);
            edges.push({
                id: edgeId,
                source: node.parentId,
                target: node.id,
                type: 'smoothstep',
                animated: node.status !== 'success' && node.status !== 'failed',
                style: { stroke: '#71717a', strokeWidth: 2 },
            });
        }
    }

    dagre.layout(g);

    // Map to React Flow nodes
    const nodes: Node[] = treeNodes.map((treeNode) => {
        const pos = g.node(treeNode.id);
        const isRoot = treeNode.id === 'root';
        const w = isRoot ? ROOT_NODE_WIDTH : NODE_WIDTH;
        const h = isRoot ? ROOT_NODE_HEIGHT : NODE_HEIGHT;
        return {
            id: treeNode.id,
            type: isRoot ? 'rootNode' : 'strategyNode',
            position: { x: pos.x - w / 2, y: pos.y - h / 2 },
            data: treeNode as unknown as Record<string, unknown>,
        };
    });

    return { nodes, edges };
}
