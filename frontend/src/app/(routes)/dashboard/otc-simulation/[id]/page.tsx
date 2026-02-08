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
    getStraightPath,
    BaseEdge,
    EdgeLabelRenderer,
    useReactFlow
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Plus, Trash2, Play, Loader2, TrendingUp, AlertTriangle, Zap, Search, X, Activity, Settings } from 'lucide-react';
import { useParams } from 'next/navigation';
import NewsFeed from '@/components/NewsFeed';

// --- Types ---
interface NewsItem {
    id: string;
    ticker: string;
    company_name: string;
    headline: string;
    summary: string;
    sentiment: 'positive' | 'negative' | 'neutral';
    confidence: number;
    timestamp: string;
    is_breaking: boolean;
}

interface CCPFunds {
    initial_margin: number;
    variation_margin_flow: number;
    default_fund: number;
    ccp_capital: number;
    units: string;
    vm_is_flow: boolean;
}

interface CongestionOutput {
    level: string;
    most_congested_node: string;
    max_score: number;
}

interface CascadeOutput {
    status: string;
    failed_nodes: string[];
    failed_edges: { source: string; target: string }[];
    failure_ratio: number;
    cascade_depth: number;
    analysis: string;
}

interface SimulationResult {
    end_date: string;
    used_tickers: string[];
    masked_tickers: string[];
    latest_features: Record<string, number>;
    latest_payoff_S: number;
    predicted_next_systemic_risk: number;
    congestion: CongestionOutput;
    cascade: CascadeOutput;
    ccp_funds: CCPFunds;
}

// Custom Edge Component (same as CCP page)
const CustomEdge = ({
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    style = {},
    markerEnd,
    data,
}: EdgeProps) => {
    const { setEdges } = useReactFlow();
    const [edgePath, labelX, labelY] = getStraightPath({
        sourceX,
        sourceY,
        targetX,
        targetY,
    });

    const onEdgeClick = (evt: React.MouseEvent) => {
        evt.stopPropagation();
        setEdges((edges) => edges.filter((edge) => edge.id !== id));
    };

    const isFailed = data?.isFailed || false;
    const isCongested = data?.isCongested || false;
    const hasIssue = isFailed || isCongested;

    const getEdgeColor = () => {
        if (isFailed) return { primary: '#ef4444', secondary: '#f59e0b', bg: '#fbbf24' };
        if (isCongested) return { primary: '#f97316', secondary: '#eab308', bg: '#fcd34d' };
        return { primary: '#a5b4fc', secondary: '#a5b4fc', bg: '#a5b4fc' };
    };
    const colors = getEdgeColor();

    return (
        <>
            <BaseEdge
                path={edgePath}
                markerEnd={hasIssue ? undefined : markerEnd}
                style={hasIssue
                    ? { ...style, stroke: colors.bg, strokeWidth: 4, opacity: 0.3 }
                    : { ...style, strokeDasharray: '5,5' }
                }
            />
            {hasIssue && (
                <path
                    d={edgePath}
                    fill="none"
                    stroke={isFailed ? "url(#cascadeGradient)" : "url(#congestionGradient)"}
                    strokeWidth={isFailed ? 4 : 3}
                    strokeDasharray={isFailed ? "10,5" : "8,4"}
                    style={{
                        animation: isFailed ? 'flowAnimation 1s linear infinite' : 'pulseAnimation 2s ease-in-out infinite',
                    }}
                />
            )}
            <defs>
                <linearGradient id="cascadeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#ef4444" />
                    <stop offset="50%" stopColor="#f59e0b" />
                    <stop offset="100%" stopColor="#ef4444" />
                </linearGradient>
                <linearGradient id="congestionGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#f97316" />
                    <stop offset="50%" stopColor="#eab308" />
                    <stop offset="100%" stopColor="#f97316" />
                </linearGradient>
                <style>
                    {`
                        @keyframes flowAnimation {
                            0% { stroke-dashoffset: 30; }
                            100% { stroke-dashoffset: 0; }
                        }
                        @keyframes pulseAnimation {
                            0%, 100% { opacity: 0.5; stroke-width: 3; }
                            50% { opacity: 1; stroke-width: 5; }
                        }
                    `}
                </style>
            </defs>
            <EdgeLabelRenderer>
                <div
                    style={{
                        position: 'absolute',
                        transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
                        pointerEvents: 'all',
                    }}
                >
                    <button
                        onClick={onEdgeClick}
                        className="w-5 h-5 bg-white border border-slate-300 rounded-full flex items-center justify-center hover:bg-red-50 hover:border-red-400 transition-colors shadow-sm"
                    >
                        <Trash2 size={10} className="text-slate-500 hover:text-red-500" />
                    </button>
                </div>
            </EdgeLabelRenderer>
        </>
    );
};

// --- Financial Node ---
const FinancialNode = ({ data, id, selected }: NodeProps & { selected?: boolean }) => {
    const isCCP = data.type === 'CCP';
    const isUnconnected = data.isUnconnected;
    const isSelected = selected || data.isSelected;
    const defaultProbability = data.defaultProbability || 0;
    const isCascading = defaultProbability >= 1.0;
    const isRisk = defaultProbability > 0.5;

    const getNodeStyle = () => {
        if (isSelected) return 'border-indigo-600 shadow-indigo-500/30 bg-indigo-50 ring-2 ring-indigo-400';
        if (isUnconnected) return 'border-red-500 shadow-red-500/20 bg-white';
        if (isCascading) return 'border-red-600 shadow-red-600/40 bg-red-100 animate-pulse';
        if (isRisk) return 'border-red-500 shadow-red-500/20 bg-red-50';
        if (defaultProbability > 0) return 'border-amber-500 shadow-amber-500/20 bg-amber-50';
        return 'border-emerald-500 shadow-emerald-500/20 bg-white';
    };

    return (
        <div className="relative group/node">
            {/* Animated dashed ring for cascade state */}
            {isCascading && (
                <>
                    <svg className="absolute -inset-3 w-[calc(100%+24px)] h-[calc(100%+24px)] z-0" viewBox="0 0 100 100">
                        <circle
                            cx="50"
                            cy="50"
                            r="45"
                            fill="none"
                            stroke="url(#cascadeGradientNode)"
                            strokeWidth="3"
                            strokeDasharray="8 4"
                            style={{
                                animation: 'dashFlow 1s linear infinite',
                            }}
                        />
                        <defs>
                            <linearGradient id="cascadeGradientNode" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stopColor="#ef4444" />
                                <stop offset="50%" stopColor="#f59e0b" />
                                <stop offset="100%" stopColor="#ef4444" />
                            </linearGradient>
                        </defs>
                    </svg>
                    <style>{`
                        @keyframes dashFlow {
                            0% { stroke-dashoffset: 24; }
                            100% { stroke-dashoffset: 0; }
                        }
                    `}</style>
                </>
            )}

            {/* Risk ring for high probability */}
            {isRisk && !isCascading && (
                <div className="absolute -inset-2 rounded-full border-2 border-dashed border-red-400 animate-spin" style={{ animationDuration: '8s' }} />
            )}

            <div className={`w-16 h-16 md:w-20 md:h-20 rounded-full flex items-center justify-center border-2 
                ${getNodeStyle()}
                shadow-md hover:shadow-xl transition-all duration-300 transform hover:scale-105 z-10 cursor-pointer`}>
                <div className="text-center">
                    <span className={`text-xs md:text-sm font-bold ${isCCP ? 'text-indigo-600' : isCascading ? 'text-red-700' : 'text-slate-700'}`}>
                        {data.label?.split('.')[0]?.slice(0, 8) || 'B'}
                    </span>
                    {defaultProbability > 0 && (
                        <div className={`text-[8px] font-bold ${isCascading ? 'text-red-700' : 'text-red-500'}`}>
                            <Zap size={8} className="inline" /> {(defaultProbability * 100).toFixed(0)}%
                        </div>
                    )}
                </div>
            </div>

            <Handle type="target" position={Position.Top} className="opacity-0 w-1 h-1" style={{ top: '50%', left: '50%', transform: 'translate(-50%, -50%)', pointerEvents: 'none' }} id="center" />
            <Handle type="source" position={Position.Top} className="opacity-0 w-1 h-1" style={{ top: '50%', left: '50%', transform: 'translate(-50%, -50%)', pointerEvents: 'none' }} id="center" />
            <Handle type="source" position={Position.Bottom} className="w-3 h-3 bg-slate-300 hover:bg-indigo-500 border border-white transition-all z-20 opacity-0 group-hover/node:opacity-100" style={{ bottom: -6 }} />
            <Handle type="source" position={Position.Top} className="w-3 h-3 bg-slate-300 hover:bg-indigo-500 border border-white transition-all z-20 opacity-0 group-hover/node:opacity-100" style={{ top: -6 }} />

            <div className="absolute top-full left-1/2 -translate-x-1/2 mt-3 w-40 p-3 bg-white/90 backdrop-blur-md rounded-xl border border-slate-100 shadow-xl opacity-0 invisible group-hover/node:opacity-100 group-hover/node:visible transition-all duration-300 z-50 pointer-events-none transform translate-y-2 group-hover/node:translate-y-0 text-center">
                <span className="text-[10px] font-bold text-indigo-500 uppercase tracking-widest block mb-1">{data.type}</span>
                <h3 className="font-bold text-slate-800 text-sm mb-1">{data.label}</h3>
                <p className="text-[9px] text-slate-400">{data.details}</p>
                {isUnconnected && <p className="text-[9px] text-red-500 font-bold mt-1">Unconnected</p>}
                {isCascading && <p className="text-[9px] text-red-600 font-bold mt-1 animate-pulse">⚠️ DEFAULTED</p>}
            </div>
        </div>
    );
};

const nodeTypes = { custom: FinancialNode };
const edgeTypes = { custom: CustomEdge };

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

export default function OTCSimulationWorkspace() {
    const { id: simulationId } = useParams();
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);

    // Node Selection & Default State
    const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
    const [defaultProbabilityByNode, setDefaultProbabilityByNode] = useState<Record<string, number>>({});

    // News State
    const [newsFeed, setNewsFeed] = useState<NewsItem[]>([]);
    const [breakingNews, setBreakingNews] = useState<NewsItem | null>(null);

    // Simulation State
    const [simulation, setSimulation] = useState<SimulationResult | null>(null);
    const [simulationLoading, setSimulationLoading] = useState(false);

    // Search State
    const [isSearchOpen, setIsSearchOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState("");
    const [searchResults, setSearchResults] = useState<any[]>([]);
    const [isSearching, setIsSearching] = useState(false);

    // Get selected node
    const selectedNode = nodes.find(n => n.id === selectedNodeId);

    // Get ticker from node
    const getNodeTicker = (node: Node | undefined) => {
        if (!node) return null;
        if (node.data.ticker) return node.data.ticker;
        const details = node.data.details || '';
        const match = details.match(/([A-Z]+)\.NS/) || details.match(/NSE\s*-\s*([A-Z]+)/);
        return match ? `${match[1]}.NS` : null;
    };

    const selectedNodeTicker = getNodeTicker(selectedNode);

    // Get all tickers from nodes
    const getTickersFromNodes = () => {
        return nodes
            .map(node => getNodeTicker(node))
            .filter((ticker): ticker is string => ticker !== null);
    };

    // Connect nodes
    const onConnect = useCallback((params: Connection) => {
        setEdges((eds) => addEdge({
            ...params,
            type: 'custom',
            markerEnd: { type: MarkerType.ArrowClosed, width: 15, height: 15, color: '#6366f1' },
            style: { stroke: '#a5b4fc', strokeWidth: 2 },
        }, eds));
    }, [setEdges]);

    // Delete node
    const deleteNode = (nodeId: string) => {
        setNodes((nds) => nds.filter((node) => node.id !== nodeId));
        setEdges((eds) => eds.filter((edge) => edge.source !== nodeId && edge.target !== nodeId));
        if (selectedNodeId === nodeId) setSelectedNodeId(null);
    };

    // Track connected nodes
    useEffect(() => {
        const connectedNodeIds = new Set<string>();
        edges.forEach(edge => {
            connectedNodeIds.add(edge.source);
            connectedNodeIds.add(edge.target);
        });

        setNodes(nds => nds.map(node => {
            const isConnected = connectedNodeIds.has(node.id);
            return { ...node, data: { ...node.data, isUnconnected: !isConnected } };
        }));
    }, [edges, setNodes]);

    // Node click handler
    const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
        setSelectedNodeId(node.id);
    }, []);

    const onPaneClick = useCallback(() => {
        setSelectedNodeId(null);
    }, []);

    // Update default probability
    const updateDefaultProbability = (value: number) => {
        if (!selectedNodeTicker || !selectedNodeId) {
            console.warn('No valid ticker found for selected node:', selectedNode);
            return;
        }
        setDefaultProbabilityByNode(prev => ({
            ...prev,
            [selectedNodeTicker]: value
        }));

        // Update node data so it turns red
        setNodes(nds => nds.map(node => {
            if (node.id === selectedNodeId) {
                return { ...node, data: { ...node.data, defaultProbability: value } };
            }
            return node;
        }));

        // If at 100%, trigger default
        if (value >= 1.0 && selectedNode) {
            triggerDefault(selectedNode, selectedNodeTicker);
        }
    };

    // Trigger default - generate breaking news and cascade
    const triggerDefault = async (node: Node, ticker: string) => {
        try {
            // Generate breaking news
            const response = await fetch('/api/news/default', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ticker,
                    company_name: node.data.label,
                    default_magnitude: 1.0
                })
            });

            if (response.ok) {
                const data = await response.json();
                setBreakingNews(data.news);

                // After news is generated, show cascade animation on edges connected to this node
                // This stays persistent to visualize the cascade spreading through the network
                setEdges(eds => eds.map(edge => {
                    if (edge.source === node.id || edge.target === node.id) {
                        return { ...edge, data: { ...edge.data, isFailed: true } };
                    }
                    return edge;
                }));

                // Wait 2-3 seconds for "market reaction", then run simulation
                setTimeout(() => {
                    runCascadeSimulation(ticker);
                }, 2500);
            }
        } catch (err) {
            console.error('Failed to generate default news:', err);
        }
    };

    // Run cascade simulation
    const runCascadeSimulation = async (defaultedTicker: string) => {
        const tickers = getTickersFromNodes();

        if (tickers.length === 0) {
            console.warn('No tickers found for simulation');
            return;
        }

        setSimulationLoading(true);
        console.log('Running cascade simulation with tickers:', tickers, 'shocked_node:', defaultedTicker);

        try {
            const payload = {
                tickers,
                start: '2023-01-01',
                shocked_node: defaultedTicker,
                shock_magnitude: 1.0,
                connectivity_strength: 1.5,
                liquidity_buffer_level: 0.5,
                ccp_strictness: 0.5,
                correlation_regime: 0.8,
                steps: 10
            };
            console.log('Simulation payload:', payload);

            const response = await fetch(`${API_URL}/simulate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            console.log('Simulation response status:', response.status);

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Simulation error response:', errorText);
                throw new Error(`Simulation failed: ${errorText}`);
            }

            const result = await response.json();
            console.log('Simulation result:', result);

            setSimulation(result);

            // Update edge visualization
            const failedEdgeSet = new Set(result.cascade.failed_edges.map((e: any) => `${e.source}-${e.target}`));
            const congestedNodes = new Set([result.congestion.most_congested_node]);

            setEdges(edges => edges.map(edge => {
                const edgeKey = `${edge.source}-${edge.target}`;
                const isFailed = failedEdgeSet.has(edgeKey);
                const isCongested = congestedNodes.has(edge.source) || congestedNodes.has(edge.target);

                return {
                    ...edge,
                    data: { ...edge.data, isFailed, isCongested }
                };
            }));

        } catch (err) {
            console.error('Cascade simulation failed:', err);
        } finally {
            setSimulationLoading(false);
        }
    };

    // Search handler
    const handleSearch = async (query: string) => {
        setSearchQuery(query);
        if (!query.trim()) {
            setSearchResults([]);
            return;
        }

        setIsSearching(true);
        try {
            const res = await fetch(`/api/finance/search?q=${encodeURIComponent(query)}`);
            if (res.ok) {
                const data = await res.json();
                setSearchResults(data.results || []);
            }
        } catch (err) {
            console.error("Search failed:", err);
        } finally {
            setIsSearching(false);
        }
    };

    const addInstitutionFromSearch = (institution: any) => {
        const lastNode = nodes[nodes.length - 1];
        const ticker = institution.ticker || institution.symbol;
        const newNode: Node = {
            id: `node-${Date.now()}`,
            type: 'custom',
            position: { x: lastNode ? lastNode.position.x + 150 : 300, y: lastNode ? lastNode.position.y + (Math.random() * 100 - 50) : 200 },
            data: {
                label: institution.name || ticker,
                type: 'Bank',
                details: `${institution.exchange || 'NSE'} - ${ticker}`,
                ticker: ticker ? (ticker.endsWith('.NS') ? ticker : `${ticker}.NS`) : undefined
            }
        };
        console.log('Adding node:', newNode);
        setNodes(nds => {
            const updated = [...nds, newNode];
            console.log('Updated nodes:', updated);
            return updated;
        });
        setIsSearchOpen(false);
        setSearchQuery("");
        setSearchResults([]);
    };

    return (
        <div className="flex w-full h-screen bg-slate-50 overflow-hidden font-sans">
            {/* LEFT SIDEBAR: NEWS FEED */}
            <NewsFeed
                breakingNews={breakingNews}
                onBreakingNewsDisplayed={() => setBreakingNews(null)}
            />

            {/* MAIN CANVAS */}
            <main className="flex-1 h-full relative">
                {/* Top Bar */}
                <div className="absolute top-4 left-4 right-4 z-10 flex items-center justify-between">
                    <div className="flex items-center gap-2 bg-white px-4 py-2 rounded-xl shadow-lg border border-slate-200">
                        <Activity size={18} className="text-indigo-600" />
                        <span className="text-sm font-bold text-slate-900">OTC Network Simulation</span>
                    </div>

                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setIsSearchOpen(true)}
                            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl shadow-lg transition-colors text-sm font-medium"
                        >
                            <Plus size={16} />
                            Add Institution
                        </button>
                    </div>
                </div>

                {/* Search Modal */}
                {isSearchOpen && (
                    <div className="absolute top-16 left-4 w-80 bg-white rounded-xl shadow-2xl border border-slate-200 z-50 overflow-hidden flex flex-col max-h-[400px]">
                        <div className="p-3 border-b border-slate-100 flex items-center gap-2">
                            <Search size={14} className="text-slate-400" />
                            <input
                                autoFocus
                                className="flex-1 text-sm outline-none text-slate-700 placeholder:text-slate-400"
                                placeholder="Search institutions..."
                                value={searchQuery}
                                onChange={(e) => handleSearch(e.target.value)}
                            />
                            <button onClick={() => { setIsSearchOpen(false); setSearchQuery(""); setSearchResults([]); }} className="p-1 hover:bg-slate-100 rounded-full">
                                <X size={14} className="text-slate-400" />
                            </button>
                        </div>
                        <div className="flex-1 overflow-y-auto">
                            {isSearching && <div className="p-4 text-center text-xs text-slate-400"><Loader2 size={14} className="animate-spin inline mr-1" /> Searching...</div>}
                            {!isSearching && searchResults.length > 0 && searchResults.map((inst, i) => (
                                <button
                                    key={i}
                                    onClick={() => addInstitutionFromSearch(inst)}
                                    className="w-full p-3 text-left hover:bg-slate-50 border-b border-slate-100 transition-colors"
                                >
                                    <span className="text-sm font-medium text-slate-800">{inst.name || inst.ticker}</span>
                                    <span className="text-[10px] text-slate-400 block">{inst.ticker} • {inst.exchange || 'N/A'}</span>
                                </button>
                            ))}
                            {!isSearching && searchQuery && searchResults.length === 0 && (
                                <div className="p-4 text-center text-xs text-slate-400">No results found</div>
                            )}
                        </div>
                    </div>
                )}

                {/* Node Inspector Panel */}
                {selectedNode && (
                    <div className="absolute bottom-4 left-4 w-72 bg-white rounded-xl shadow-2xl border border-slate-200 z-50 p-4">
                        <h3 className="text-xs font-bold text-slate-800 mb-1 uppercase tracking-wider flex items-center gap-2">
                            <Zap size={12} className="text-red-500" /> Default Trigger
                        </h3>
                        <p className="text-sm font-bold text-indigo-600 mb-3">{selectedNode.data.label}</p>
                        {selectedNodeTicker ? (
                            <div>
                                <label className="text-[10px] text-slate-500 flex justify-between">
                                    <span>Default Probability</span>
                                    <span className="font-bold text-red-600">{((defaultProbabilityByNode[selectedNodeTicker] || 0) * 100).toFixed(0)}%</span>
                                </label>
                                <input
                                    type="range"
                                    min="0"
                                    max="1"
                                    step="0.05"
                                    value={defaultProbabilityByNode[selectedNodeTicker] || 0}
                                    onChange={e => updateDefaultProbability(parseFloat(e.target.value))}
                                    onInput={e => updateDefaultProbability(parseFloat((e.target as HTMLInputElement).value))}
                                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-red-600"
                                />
                                <div className="flex justify-between text-[9px] text-slate-400 mt-1">
                                    <span>Stable</span>
                                    <span>Default</span>
                                </div>
                                {(defaultProbabilityByNode[selectedNodeTicker] || 0) > 0.75 && (
                                    <div className="mt-2 text-xs text-amber-700 bg-amber-50 p-2 rounded">
                                        ⚠️ High default risk
                                    </div>
                                )}
                                <button
                                    onClick={() => updateDefaultProbability(0)}
                                    className="mt-3 w-full py-1.5 text-[10px] bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-lg transition-colors"
                                >
                                    Reset
                                </button>
                            </div>
                        ) : (
                            <div className="space-y-2">
                                <div className="text-xs text-amber-700 bg-amber-50 p-2 rounded border border-amber-200">
                                    ⚠️ No ticker found for this node.
                                </div>
                                <p className="text-[10px] text-slate-500">
                                    <strong>Solution:</strong> Delete this node and re-add it using the "Add Institution" search button.
                                </p>
                                <button
                                    onClick={(e) => { e.stopPropagation(); deleteNode(selectedNode.id); setSelectedNodeId(null); }}
                                    className="w-full py-1.5 text-[10px] bg-red-50 hover:bg-red-100 text-red-600 rounded-lg transition-colors font-medium"
                                >
                                    Delete This Node
                                </button>
                            </div>
                        )}
                    </div>
                )}

                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    onNodeClick={onNodeClick}
                    onPaneClick={onPaneClick}
                    nodeTypes={nodeTypes}
                    edgeTypes={edgeTypes}
                    fitView
                >
                    <Background color="#cbd5e1" gap={20} />
                    <Controls />
                </ReactFlow>
            </main>

            {/* RIGHT SIDEBAR: RESULTS */}
            <aside className="w-80 h-full bg-white border-l border-slate-200 flex flex-col p-4 shadow-lg overflow-y-auto">
                <h2 className="text-sm font-bold text-slate-900 mb-4">Simulation Results</h2>

                {simulationLoading && (
                    <div className="flex items-center justify-center h-32">
                        <Loader2 className="animate-spin text-indigo-600" size={32} />
                    </div>
                )}

                {simulation && !simulationLoading && (
                    <div className="space-y-4">
                        {/* Systemic Risk */}
                        <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 border border-indigo-200 rounded-xl p-4">
                            <h3 className="text-xs font-bold text-indigo-900 mb-2 uppercase tracking-wider flex items-center gap-2">
                                <TrendingUp size={12} /> Systemic Risk
                            </h3>
                            <div className="text-3xl font-black text-indigo-900">
                                {(simulation.predicted_next_systemic_risk * 100).toFixed(1)}%
                            </div>
                            <p className="text-[10px] text-indigo-700 mt-2">Network risk level</p>
                        </div>

                        {/* Payoff */}
                        <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
                            <h3 className="text-xs font-bold text-slate-800 mb-2 uppercase tracking-wider">💰 Payoff S_T</h3>
                            <div className="text-2xl font-bold text-slate-900">
                                {simulation.latest_payoff_S.toFixed(4)}
                            </div>
                        </div>

                        {/* CCP Funds */}
                        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
                            <h3 className="text-xs font-bold text-amber-900 mb-3 uppercase tracking-wider">CCP Funds</h3>
                            <div className="grid grid-cols-2 gap-2 text-xs">
                                <div>
                                    <p className="text-amber-700">Initial Margin</p>
                                    <p className="font-bold text-amber-900">{simulation.ccp_funds.initial_margin.toFixed(1)}</p>
                                </div>
                                <div>
                                    <p className="text-amber-700">Variation Margin</p>
                                    <p className="font-bold text-amber-900">{simulation.ccp_funds.variation_margin_flow.toFixed(1)}</p>
                                </div>
                                <div>
                                    <p className="text-amber-700">Default Fund</p>
                                    <p className="font-bold text-amber-900">{simulation.ccp_funds.default_fund.toFixed(1)}</p>
                                </div>
                                <div>
                                    <p className="text-amber-700">CCP Capital</p>
                                    <p className="font-bold text-amber-900">{simulation.ccp_funds.ccp_capital.toFixed(1)}</p>
                                </div>
                            </div>
                        </div>

                        {/* Congestion */}
                        <div className="bg-cyan-50 border border-cyan-200 rounded-xl p-4">
                            <h3 className="text-xs font-bold text-cyan-900 mb-3 uppercase tracking-wider">Congestion</h3>
                            <div className="space-y-2">
                                <div className="flex justify-between text-xs">
                                    <span className="text-cyan-700">Level</span>
                                    <span className="font-bold text-cyan-900 uppercase">{simulation.congestion.level}</span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span className="text-cyan-700">Most Congested</span>
                                    <span className="font-bold text-cyan-900">{simulation.congestion.most_congested_node}</span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span className="text-cyan-700">Max Score</span>
                                    <span className="font-bold text-cyan-900">{simulation.congestion.max_score.toFixed(2)}</span>
                                </div>
                            </div>
                        </div>

                        {/* Cascade */}
                        <div className="bg-red-50 border border-red-200 rounded-xl p-4">
                            <h3 className="text-xs font-bold text-red-900 mb-3 uppercase tracking-wider">Cascade</h3>
                            <div className="space-y-2">
                                <div className="flex justify-between text-xs">
                                    <span className="text-red-700">Status</span>
                                    <span className="font-bold text-red-900 uppercase">{simulation.cascade.status}</span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span className="text-red-700">Failure Ratio</span>
                                    <span className="font-bold text-red-900">{(simulation.cascade.failure_ratio * 100).toFixed(0)}%</span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span className="text-red-700">Cascade Depth</span>
                                    <span className="font-bold text-red-900">{simulation.cascade.cascade_depth}</span>
                                </div>
                            </div>
                            <div className="mt-3 pt-3 border-t border-red-200">
                                <p className="text-[9px] text-red-700 font-medium">Analysis:</p>
                                <p className="text-[10px] text-red-600 italic mt-1">{simulation.cascade.analysis}</p>
                            </div>
                        </div>
                    </div>
                )}

                {!simulation && !simulationLoading && (
                    <div className="text-center py-8 text-xs text-slate-400">
                        Set a bank to 100% default probability to trigger cascade simulation
                    </div>
                )}

                {/* Institutions List */}
                <div className="mt-6 pt-6 border-t border-slate-200">
                    <h3 className="text-xs font-bold text-slate-900 mb-3 uppercase tracking-wider">Institutions</h3>
                    <div className="space-y-2">
                        {nodes.length === 0 ? (
                            <div className="text-center py-4 text-[10px] text-slate-400">
                                No institutions yet. Click "Add Institution" to start.
                            </div>
                        ) : (
                            nodes.map(node => (
                                <div key={node.id} className="bg-slate-50 border border-slate-200 rounded-lg p-2 flex items-center justify-between group">
                                    <div className="flex-1">
                                        <div className="text-xs font-medium text-slate-900">{node.data.label}</div>
                                        <div className="text-[9px] text-slate-400">{node.data.details}</div>
                                    </div>
                                    <button
                                        onClick={() => deleteNode(node.id)}
                                        className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 rounded transition-all"
                                    >
                                        <Trash2 size={12} className="text-red-500" />
                                    </button>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </aside>
        </div>
    );
}
