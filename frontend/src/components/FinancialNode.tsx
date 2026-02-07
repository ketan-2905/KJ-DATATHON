"use client";
import React, { useCallback, useState } from 'react';
import ReactFlow, {
    Background,
    Controls,
    NodeProps,
    Handle,
    Position,
    useNodesState,
    useEdgesState,
    addEdge,
    Connection,
    Edge,
    MarkerType
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Plus, Trash2, Network } from 'lucide-react';
import initialData from '../data/financialNodes.json';

// --- Types ---
interface NodeData {
    label: string;
    type: string;
    details: string;
}

// --- Custom Node Component ---
const FinancialNode = ({ data, id }: NodeProps<NodeData>) => {
    const isCCP = data.type === 'CCP';

    return (
        <div className="relative group">
            {/* Node Circle */}
            <div className={`
                w-16 h-16 md:w-20 md:h-20 rounded-full 
                flex items-center justify-center 
                bg-white border-2 
                ${isCCP ? 'border-indigo-500 shadow-indigo-500/20' : 'border-slate-200 hover:border-indigo-400'}
                shadow-sm hover:shadow-xl
                transition-all duration-300 transform group-hover:scale-105
                cursor-pointer z-10
            `}>
                {/* Logo / Letter */}
                <span className={`
                    text-xl md:text-2xl font-bold font-serif
                    ${isCCP ? 'text-indigo-600' : 'text-slate-700'}
                `}>
                    {isCCP ? 'C' : 'B'}
                </span>
            </div>

            {/* Connection Handles */}
            <Handle
                type="target"
                position={Position.Top}
                className="w-3 h-3 !bg-slate-300 hover:!bg-indigo-500 transition-colors"
                style={{ top: -5 }}
            />
            <Handle
                type="source"
                position={Position.Bottom}
                className="w-3 h-3 !bg-slate-300 hover:!bg-indigo-500 transition-colors"
                style={{ bottom: -5 }}
            />
            <Handle
                type="target"
                position={Position.Left}
                className="w-3 h-3 !bg-slate-300 hover:!bg-indigo-500 transition-colors"
                style={{ left: -5 }}
            />
            <Handle
                type="source"
                position={Position.Right}
                className="w-3 h-3 !bg-slate-300 hover:!bg-indigo-500 transition-colors"
                style={{ right: -5 }}
            />

            {/* Hover Detail Card */}
            <div className="
                absolute top-full left-1/2 -translate-x-1/2 mt-4 
                w-48 p-3 rounded-xl 
                bg-white/90 backdrop-blur-md border border-slate-100 shadow-xl shadow-slate-200/50
                opacity-0 invisible group-hover:opacity-100 group-hover:visible 
                transition-all duration-300 z-50 pointer-events-none
                transform translate-y-2 group-hover:translate-y-0
            ">
                <div className="flex items-center gap-2 mb-2 border-b border-slate-100 pb-2">
                    <div className={`w-1.5 h-1.5 rounded-full ${isCCP ? 'bg-indigo-500' : 'bg-emerald-400'}`}></div>
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">{data.type}</span>
                </div>

                <h3 className="font-bold text-slate-800 text-sm mb-1">{data.label}</h3>
                <p className="text-[10px] text-slate-500 leading-relaxed mb-2">
                    {data.details}
                </p>
                <div className="text-[9px] text-slate-400 font-mono">
                    ID: {id}
                </div>
            </div>
        </div>
    );
};

const nodeTypes = { custom: FinancialNode };

export default function FinancialInfrastructureMap() {
    // Initialize nodes and edges from JSON
    const [nodes, setNodes, onNodesChange] = useNodesState(initialData.nodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialData.edges);

    // --- Interaction Handlers ---

    // Manual Connection
    const onConnect = useCallback(
        (params: Connection) => setEdges((eds) => addEdge({
            ...params,
            animated: true,
            style: { stroke: '#94a3b8', strokeWidth: 1.5 },
            type: 'default'
        }, eds)),
        [setEdges],
    );

    // Delete selected edges/nodes (handled natively by onEdgesChange/onNodesChange using Delete key, 
    // but we can add UI button if needed, standard behavior is usually sufficient if key events are captured)

    // Add New Bank Node
    const addNewBank = () => {
        const id = `bank-${nodes.length + Math.floor(Math.random() * 1000)}`;
        const newNode = {
            id,
            type: 'custom',
            position: {
                x: Math.random() * 600 + 50,
                y: Math.random() * 400 + 50,
            },
            data: {
                label: `New Bank`,
                type: 'Bank',
                details: 'Newly joined member bank.'
            },
        };

        // Automatically connect to CCP
        const newEdge = {
            id: `e-${id}-ccp`,
            source: id,
            target: 'ccp-1',
            animated: true,
            style: { stroke: '#6366f1' },
        };

        setNodes((nds) => nds.concat(newNode));
        setEdges((eds) => eds.concat(newEdge));
    };

    return (
        <div className="flex w-full h-screen bg-slate-50 overflow-hidden font-sans">
            {/* Left Sidebar */}
            <aside className="w-72 h-full bg-white border-r border-slate-200 hidden xl:flex flex-col p-6 z-10 shrink-0 shadow-[4px_0_24px_-12px_rgba(0,0,0,0.1)]">
                <div className="mb-8">
                    <h1 className="text-xl font-bold text-slate-900 tracking-tighter flex items-center gap-2">
                        <Network className="text-indigo-600" size={24} />
                        Equilibrium
                    </h1>
                    <p className="text-[10px] text-slate-400 mt-1 uppercase tracking-widest font-semibold pl-8">Network Admin</p>
                </div>

                <div className="space-y-4">
                    <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100">
                        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Controls</h3>
                        <button
                            onClick={addNewBank}
                            className="w-full py-3 px-4 bg-indigo-600 hover:bg-indigo-700 active:scale-95 text-white rounded-xl shadow-lg shadow-indigo-500/20 transition-all flex items-center justify-center gap-2 text-sm font-semibold"
                        >
                            <Plus size={16} /> Add Bank Node
                        </button>
                        <p className="text-[10px] text-slate-400 mt-3 text-center leading-relaxed">
                            New nodes automatically connect to Central Counterparty (CCP).
                        </p>
                    </div>

                    <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100">
                        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Instructions</h3>
                        <ul className="text-xs text-slate-500 space-y-2 list-disc pl-4">
                            <li>Drag handles to connect nodes manually.</li>
                            <li>Select a connection and press <kbd className="font-mono bg-white px-1 rounded border border-slate-200">Backspace</kbd> to delete.</li>
                            <li>Drag nodes to rearrange the network topology.</li>
                        </ul>
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 h-full relative bg-slate-50/50">
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    nodeTypes={nodeTypes}
                    fitView
                    attributionPosition="bottom-right"
                    proOptions={{ hideAttribution: true }}
                    minZoom={0.5}
                    maxZoom={1.5}
                    className="bg-slate-50"
                >
                    <Background color="#94a3b8" gap={32} size={1} className="opacity-20" />
                    <Controls className="!bg-white !border-slate-200 !text-slate-600 !shadow-sm !rounded-xl overflow-hidden m-4" />
                </ReactFlow>
            </main>

            {/* Right Sidebar */}
            <aside className="w-80 h-full bg-white border-l border-slate-200 hidden lg:flex flex-col p-6 z-10 shrink-0 shadow-[-4px_0_24px_-12px_rgba(0,0,0,0.1)]">
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider">System Status</h2>
                    <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                        <span className="text-xs font-medium text-emerald-600">Live</span>
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
                    <div className="space-y-3">
                        {nodes.map(node => (
                            <div key={node.id} className="p-3 bg-white border border-slate-100 rounded-xl hover:border-indigo-100 transition-colors group">
                                <div className="flex items-center justify-between mb-1">
                                    <span className="text-xs font-bold text-slate-700">{node.data.label}</span>
                                    <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${node.data.type === 'CCP' ? 'bg-indigo-50 text-indigo-600' : 'bg-slate-100 text-slate-500'}`}>
                                        {node.data.type}
                                    </span>
                                </div>
                                <div className="text-[10px] text-slate-400 truncate">
                                    ID: {node.id}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </aside>
        </div>
    );
}