"use client";
import React, { useState, useEffect, useCallback } from 'react';
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
    MarkerType,
    Node,
    Edge,
    EdgeProps,
    getBezierPath,
    getStraightPath,
    BaseEdge,
    EdgeLabelRenderer,
    useReactFlow
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Network, Plus, Trash2, Send, Save, Activity, Search, X } from 'lucide-react';
import { useParams } from 'next/navigation';

// --- CUSTOM EDGE WITH DELETE BUTTON ---
const CustomEdge = ({
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    style = {},
    markerEnd,
}: EdgeProps) => {
    const { setEdges } = useReactFlow();
    const [edgePath, labelX, labelY] = getStraightPath({
        sourceX,
        sourceY,
        sourcePosition,
        targetX,
        targetY,
        targetPosition,
    });

    const onEdgeClick = (evt: React.MouseEvent) => {
        evt.stopPropagation();
        setEdges((edges) => edges.filter((edge) => edge.id !== id));
    };

    return (
        <>
            <BaseEdge path={edgePath} markerEnd={markerEnd} style={style} />
            <EdgeLabelRenderer>
                <div
                    style={{
                        position: 'absolute',
                        transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
                        fontSize: 12,
                        pointerEvents: 'all',
                    }}
                    className="nodrag nopan group"
                >
                    <button
                        className="w-4 h-4 bg-slate-100 hover:bg-red-500 hover:text-white rounded-full flex items-center justify-center text-[10px] transition-all shadow-sm border border-slate-300 opacity-0 group-hover:opacity-100"
                        onClick={onEdgeClick}
                        aria-label="Delete Connection"
                        title="Delete Connection"
                    >
                        ×
                    </button>
                    {/* Invisible hover area to make it easier to trigger the button visibility */}
                    <div className="absolute inset-0 w-8 h-8 -translate-x-1/2 -translate-y-1/2 rounded-full cursor-pointer hover:bg-slate-500/10 group-hover:bg-transparent -z-10" />
                </div>
            </EdgeLabelRenderer>
        </>
    );
};


// --- NODE COMPONENT ---
const FinancialNode = ({ data, id }: NodeProps) => {
    const isCCP = data.type === 'CCP';
    const isUnconnected = data.isUnconnected;

    return (
        <div className="relative group/node">
            <div className={`w-16 h-16 md:w-20 md:h-20 rounded-full flex items-center justify-center bg-white border-2 
                ${isUnconnected ? 'border-red-500 shadow-red-500/20' :
                    isCCP ? 'border-indigo-600 shadow-indigo-500/20' : 'border-slate-300 hover:border-indigo-400'}
                shadow-md hover:shadow-xl transition-all duration-300 transform hover:scale-105 z-10 cursor-pointer`}>
                <span className={`text-xl md:text-2xl font-bold font-serif ${isCCP ? 'text-indigo-600' : 'text-slate-700'}`}>
                    {isCCP ? 'C' : isUnconnected ? '!' : data.label ? data.label[0] : 'B'}
                </span>
            </div>

            {/* center handle (target) - existing */}
            <Handle
                type="target"
                position={Position.Top}
                className="opacity-0 w-1 h-1"
                style={{ top: '50%', left: '50%', transform: 'translate(-50%, -50%)', pointerEvents: 'none' }} // Visible handle is below
                id="center"
            />

            {/* NEW: center handle (source) - ensuring center-to-center connections work both ways */}
            <Handle
                type="source"
                position={Position.Top}
                className="opacity-0 w-1 h-1"
                style={{ top: '50%', left: '50%', transform: 'translate(-50%, -50%)', pointerEvents: 'none' }}
                id="center"
            />


            {/* Interaction Handles: Visible ONLY on Hover (Visual guide, but we will force connection to 'center') */}
            <Handle
                type="source"
                position={Position.Bottom}
                className="w-3 h-3 bg-slate-300 hover:bg-indigo-500 border border-white transition-all z-20 opacity-0 group-hover/node:opacity-100"
                style={{ bottom: -6 }}
            />
            <Handle
                type="source"
                position={Position.Top}
                className="w-3 h-3 bg-slate-300 hover:bg-indigo-500 border border-white transition-all z-20 opacity-0 group-hover/node:opacity-100"
                style={{ top: -6 }}
            />

            {/* Tooltip */}
            <div className="absolute top-full left-1/2 -translate-x-1/2 mt-3 w-40 p-3 bg-white/90 backdrop-blur-md rounded-xl border border-slate-100 shadow-xl opacity-0 invisible group-hover/node:opacity-100 group-hover/node:visible transition-all duration-300 z-50 pointer-events-none transform translate-y-2 group-hover/node:translate-y-0 text-center">
                <span className="text-[10px] font-bold text-indigo-500 uppercase tracking-widest block mb-1">{data.type}</span>
                <h3 className="font-bold text-slate-800 text-sm mb-1">{data.label}</h3>
                <p className="text-[9px] text-slate-400">{data.details}</p>
                {isUnconnected && <p className="text-[9px] text-red-500 font-bold mt-1">Unconnected</p>}
            </div>
        </div>
    );
};

const nodeTypes = { custom: FinancialNode };
const edgeTypes = { custom: CustomEdge };

export default function SimulationWorkspace() {
    const { id: simulationId } = useParams();
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);
    const [input, setInput] = useState("");
    const [messages, setMessages] = useState<{ role: 'user' | 'ai', content: string }[]>([]);
    const [isLoading, setIsLoading] = useState(false);

    // Search State
    const [isSearchOpen, setIsSearchOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState("");
    const [searchResults, setSearchResults] = useState<any[]>([]);
    const [isSearching, setIsSearching] = useState(false);

    // Load Simulation Data
    useEffect(() => {
        const fetchSimulation = async () => {
            if (!simulationId) return;
            try {
                const res = await fetch(`/api/simulation/${simulationId}`);
                if (res.ok) {
                    const data = await res.json();
                    setNodes(data.nodes || []);
                    // Ensure loaded edges use custom type
                    const loadedEdges = (data.edges || []).map((e: any) => ({ ...e, type: 'custom' }));
                    setEdges(loadedEdges);
                }
            } catch (err) {
                console.error("Failed to load simulation:", err);
            }
        };
        fetchSimulation();
    }, [simulationId, setNodes, setEdges]); // eslint-disable-line

    // Save Simulation Data
    const saveSimulation = async () => {
        if (!simulationId) return;
        try {
            await fetch(`/api/simulation/${simulationId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ nodes, edges })
            });
            console.log("Saved simulation state.");
        } catch (err) {
            console.error("Failed to save simulation:", err);
        }
    };

    // Auto-save
    useEffect(() => {
        if (nodes.length > 0) {
            const timer = setTimeout(saveSimulation, 2000);
            return () => clearTimeout(timer);
        }
    }, [nodes, edges]); // eslint-disable-line

    // Check for unconnected nodes to update styling
    useEffect(() => {
        setNodes((nds) => nds.map((node) => {
            const isConnected = edges.some(e => e.source === node.id || e.target === node.id);
            if (node.data.isUnconnected !== !isConnected) {
                return { ...node, data: { ...node.data, isUnconnected: !isConnected } };
            }
            return node;
        }));
    }, [edges, setNodes]);


    // Handle Chat
    const handleSend = async () => {
        if (!input.trim()) return;

        const userMsg = input;
        setInput("");
        setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
        setIsLoading(true);

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: userMsg,
                    currentGraph: { nodes, edges }
                })
            });

            if (!res.ok) throw new Error("AI failed to respond");

            const action = await res.json();

            if (action.nodesToAdd) {
                const newNodes = action.nodesToAdd.map((n: any) => ({
                    ...n,
                    id: n.id || `node-${Date.now()}-${Math.random()}`
                }));
                // Prevent duplicates
                setNodes((nds) => {
                    const existingIds = new Set(nds.map(n => n.id));
                    const uniqueNewNodes = newNodes.filter((n: any) => !existingIds.has(n.id));
                    return [...nds, ...uniqueNewNodes];
                });
            }

            if (action.edgesToAdd) {
                setEdges((eds) => {
                    const existingIds = new Set(eds.map(e => e.id));
                    const uniqueNewEdges = action.edgesToAdd.filter((e: any) => !existingIds.has(e.id))
                        .map((e: any) => ({ ...e, type: 'custom' })); // Ensure custom type
                    return [...eds, ...uniqueNewEdges];
                });
            }

            if (action.nodesToRemove) {
                setNodes((nds) => nds.filter((n) => !action.nodesToRemove.includes(n.id)));
            }

            if (action.edgesToRemove) {
                setEdges((eds) => eds.filter((e) => !action.edgesToRemove.includes(e.id)));
            }

            if (action.message) {
                setMessages(prev => [...prev, { role: 'ai', content: action.message }]);
            }

        } catch (err) {
            console.error(err);
            setMessages(prev => [...prev, { role: 'ai', content: "Sorry, I encountered an error. Please try again." }]);
        } finally {
            setIsLoading(false);
        }
    };

    // SEARCH INSTITUTIONS
    const handleSearch = async (val: string) => {
        setSearchQuery(val);
        if (val.length < 2) {
            setSearchResults([]);
            return;
        }

        setIsSearching(true);
        try {
            // Use our proxy API
            const res = await fetch(`/api/finance/search?q=${val}`);
            if (res.ok) {
                const data = await res.json();
                setSearchResults(data.results || []);
            }
        } catch (err) {
            console.error("Search failed", err);
        } finally {
            setIsSearching(false);
        }
    };

    // ADD INSTITUTION NODE
    const addInstitution = (item: any) => {
        const id = item.id || `bank-${Date.now()}`;
        // Random position "NEAR" center as requested (reduced radius 100-150)
        const angle = Math.random() * Math.PI * 2;
        const radius = 120 + Math.random() * 60; // Was 250+50
        const x = 400 + Math.cos(angle) * radius;
        const y = 300 + Math.sin(angle) * radius;

        const newNode: Node = {
            id,
            type: 'custom',
            position: { x, y },
            data: {
                label: item.name,
                type: 'Financial Institution',
                details: `${item.exchange} - ${item.ticker}`,
                isUnconnected: false // Will be connected
            }
        };

        // Find CCP to connect to
        const ccp = nodes.find(n => n.data.type === 'CCP');
        const newEdge: Edge | null = ccp ? {
            id: `e-${id}-${ccp.id}`,
            source: id,
            target: ccp.id,
            type: 'custom', // Use custom edge
            sourceHandle: 'center', // STRICTLY CONNECT FROM CENTER
            targetHandle: 'center', // STRICTLY CONNECT TO CENTER
            animated: true,
            style: { stroke: '#94a3b8', strokeWidth: 2 }
        } : null;

        setNodes((nds) => nds.concat(newNode));
        if (newEdge) {
            setEdges((eds) => eds.concat(newEdge));
        }

        // Reset Search
        setSearchQuery("");
        setSearchResults([]);
        setIsSearchOpen(false);
    };

    // HANDLE CONNECTIONS
    const onConnect = useCallback((params: Connection) => {
        setEdges((eds) => {
            // 1. Prevent Double Links (source->target OR target->source)
            const exists = eds.some(e =>
                (e.source === params.source && e.target === params.target) ||
                (e.source === params.target && e.target === params.source)
            );

            if (exists) return eds;


            const sourceNode = nodes.find(n => n.id === params.source);
            const targetNode = nodes.find(n => n.id === params.target);

            const isSourceCCP = sourceNode?.data.type === 'CCP';
            const isTargetCCP = targetNode?.data.type === 'CCP';

            // Logic: CCP to Bank (or vice versa) -> Gray
            // Bank to Bank -> Blue
            let edgeColor = '#3b82f6'; // Default Blue (bank-to-bank)
            if (isSourceCCP || isTargetCCP) {
                edgeColor = '#94a3b8'; // Gray
            }

            return addEdge({
                ...params,
                type: 'custom', // Use Custom Edge
                // FORCE THE VISUAL TO BE CENTER-TO-CENTER
                // We overwrite whatever handle the user dragged from/to with our 'center' handle logic
                sourceHandle: 'center',
                targetHandle: 'center',
                animated: true,
                style: { stroke: edgeColor, strokeWidth: 2 }
            }, eds);
        });
    }, [setEdges, nodes]); // Added nodes to dependency

    // DELETE NODE
    const deleteNode = (id: string) => {
        setNodes((nds) => nds.filter((n) => n.id !== id));
        setEdges((eds) => eds.filter((e) => e.source !== id && e.target !== id));
    };

    return (
        <div className="flex w-full h-screen bg-slate-50 overflow-hidden font-sans">
            {/* LEFT SIDEBAR: CHAT */}
            <aside className="w-80 h-full bg-white border-r border-slate-200 flex flex-col p-4 z-20 shadow-lg">
                <div className="mb-4 flex items-center justify-between">
                    <h2 className="text-sm font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2">
                        <Network size={16} /> Data Pilot
                    </h2>
                    <span className="px-2 py-0.5 bg-emerald-100 text-emerald-700 text-[10px] rounded-full font-bold uppercase">Online</span>
                </div>

                <div className="flex-1 overflow-y-auto mb-4 space-y-4 pr-1 custom-scrollbar">
                    {messages.length === 0 && (
                        <div className="text-center text-slate-400 text-sm mt-10 p-4 border-2 border-dashed border-slate-100 rounded-xl">
                            <p>Ask me to simulate scenarios.</p>
                        </div>
                    )}
                    {messages.map((msg, i) => (
                        <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`
                                max-w-[85%] p-3 rounded-2xl text-xs leading-relaxed shadow-sm
                                ${msg.role === 'user'
                                    ? 'bg-indigo-600 text-white rounded-tr-none'
                                    : 'bg-slate-100 text-slate-700 rounded-tl-none border border-slate-200'}
                            `}>
                                {msg.content}
                            </div>
                        </div>
                    ))}
                    {isLoading && (
                        <div className="flex justify-start">
                            <div className="bg-slate-50 text-slate-400 p-3 rounded-2xl rounded-tl-none border border-slate-100 text-xs animate-pulse">
                                Simulating...
                            </div>
                        </div>
                    )}
                </div>

                <div className="relative">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                        placeholder="Describe network changes..."
                        className="w-full pl-4 pr-12 py-3 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all text-slate-700"
                    />
                    <button
                        onClick={handleSend}
                        disabled={isLoading}
                        className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors"
                    >
                        <Send size={14} />
                    </button>
                </div>
            </aside>

            {/* MAIN CANVAS */}
            <main className="flex-1 h-full relative">
                {/* Floating Controls */}
                <div className="absolute top-4 left-4 z-10 flex gap-2">
                    <button
                        onClick={() => setIsSearchOpen(true)}
                        className="bg-white hover:bg-slate-50 text-slate-700 px-4 py-2 rounded-xl shadow-lg shadow-slate-200/50 border border-slate-200 flex items-center gap-2 text-xs font-bold transition-all active:scale-95"
                    >
                        <Plus size={14} className="text-indigo-600" /> Add Institution
                    </button>
                    <button
                        onClick={saveSimulation}
                        className="bg-white hover:bg-slate-50 text-slate-700 px-4 py-2 rounded-xl shadow-lg shadow-slate-200/50 border border-slate-200 flex items-center gap-2 text-xs font-bold transition-all active:scale-95"
                    >
                        <Save size={14} className="text-emerald-500" /> Save
                    </button>
                </div>

                {/* SEARCH MODAL (Simple overlay) */}
                {isSearchOpen && (
                    <div className="absolute top-16 left-4 w-80 bg-white rounded-xl shadow-2xl border border-slate-200 z-50 overflow-hidden flex flex-col max-h-[400px]">
                        <div className="p-3 border-b border-slate-100 flex items-center gap-2">
                            <Search size={14} className="text-slate-400" />
                            <input
                                autoFocus
                                className="flex-1 text-sm outline-none text-slate-700 placeholder:text-slate-400"
                                placeholder="Search (e.g., Reliance)..."
                                value={searchQuery}
                                onChange={(e) => handleSearch(e.target.value)}
                            />
                            <button onClick={() => setIsSearchOpen(false)} className="text-slate-400 hover:text-slate-600">
                                <X size={14} />
                            </button>
                        </div>

                        <div className="flex-1 overflow-y-auto custom-scrollbar bg-slate-50">
                            {isSearching ? (
                                <div className="p-4 text-center text-xs text-slate-400 animate-pulse">Searching global markets...</div>
                            ) : searchResults.length > 0 ? (
                                <div className="divide-y divide-slate-100">
                                    {searchResults.map((item) => (
                                        <button
                                            key={item.id}
                                            onClick={() => addInstitution(item)}
                                            className="w-full text-left p-3 hover:bg-white transition-colors flex flex-col gap-0.5"
                                        >
                                            <div className="flex justify-between items-center">
                                                <span className="font-bold text-slate-700 text-xs truncate w-48">{item.name}</span>
                                                <span className="text-[9px] bg-slate-200 text-slate-600 px-1.5 py-0.5 rounded ml-2">{item.exchange}</span>
                                            </div>
                                            <span className="text-[10px] text-slate-400 font-mono">{item.ticker}</span>
                                        </button>
                                    ))}
                                </div>
                            ) : searchQuery.length > 2 && (
                                <div className="p-4 text-center text-xs text-slate-400">No results found.</div>
                            )}
                        </div>
                    </div>
                )}

                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    nodeTypes={nodeTypes}
                    edgeTypes={edgeTypes}
                    fitView
                    attributionPosition="bottom-right"
                    proOptions={{ hideAttribution: true }}
                    minZoom={0.2}
                    maxZoom={2}
                    className="bg-slate-50/50"
                >
                    <Background color="#cbd5e1" gap={24} size={1} />
                    <Controls className="!bg-white !border-slate-200 !text-slate-600 !shadow-sm !rounded-xl overflow-hidden m-4" />
                </ReactFlow>
            </main>

            {/* RIGHT SIDEBAR: DETAILS */}
            <aside className="w-72 h-full bg-white border-l border-slate-200 hidden lg:flex flex-col p-6 z-10 shrink-0 shadow-lg">
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider">Institutions</h2>
                    <div className="flex items-center gap-1.5 px-2 py-1 bg-indigo-50 rounded-md border border-indigo-100">
                        <Activity size={12} className="text-indigo-600" />
                        <span className="text-[10px] font-bold text-indigo-700">{nodes.length}</span>
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar space-y-3">
                    {nodes.map(node => (
                        <div key={node.id} className="group p-3 bg-white border border-slate-100 rounded-xl hover:border-indigo-200 hover:shadow-md transition-all cursor-default relative">
                            <div className="flex items-center justify-between mb-1">
                                <span className="text-xs font-bold text-slate-700 truncate max-w-[120px]">{node.data.label}</span>
                                {/* Delete Button on Sidebar Item */}
                                <button
                                    onClick={(e) => { e.stopPropagation(); deleteNode(node.id); }}
                                    className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-50 text-slate-300 hover:text-red-500 rounded transition-all"
                                >
                                    <Trash2 size={12} />
                                </button>
                            </div>
                            <div className="flex items-center justify-start gap-2 text-[10px] text-slate-400 mt-2">
                                <span className={`px-1.5 py-0.5 rounded-full font-bold uppercase tracking-wide text-[8px] ${node.data.isUnconnected ? 'bg-red-50 text-red-500' : 'bg-slate-100 text-slate-500'}`}>
                                    {node.data.isUnconnected ? 'Disconnected' : 'Active'}
                                </span>
                                <span className="font-mono opacity-50">#{node.id.slice(0, 6)}...</span>
                            </div>
                        </div>
                    ))}
                </div>
            </aside>
        </div>
    );
}
