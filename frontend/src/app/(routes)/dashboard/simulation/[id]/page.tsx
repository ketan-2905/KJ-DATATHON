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
import { Network, Plus, Trash2, Send, Save, Activity, Search, X, Play, Loader2, TrendingUp, AlertTriangle, Zap, Settings, Sliders, Calendar } from 'lucide-react';
import { useParams } from 'next/navigation';
import { CCPAnalyticsPanel } from '@/components/CCPAnalyticsPanel';

// --- Types ---
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

// --- API URL ---
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

// --- CUSTOM EDGE WITH DELETE BUTTON ---
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

    // Check if edge is failed (from cascade) or congested
    const isFailed = data?.isFailed || false;
    const isCongested = data?.isCongested || false;
    const hasIssue = isFailed || isCongested;

    // Determine colors based on state
    const getEdgeColor = () => {
        if (isFailed) return { primary: '#ef4444', secondary: '#f59e0b', bg: '#fbbf24' }; // Red-Yellow for cascade
        if (isCongested) return { primary: '#f97316', secondary: '#eab308', bg: '#fcd34d' }; // Orange-Yellow for congestion
        return { primary: '#a5b4fc', secondary: '#a5b4fc', bg: '#a5b4fc' }; // Default indigo
    };
    const colors = getEdgeColor();

    return (
        <>
            {/* Base edge */}
            <BaseEdge
                path={edgePath}
                markerEnd={hasIssue ? undefined : markerEnd}
                style={hasIssue
                    ? { ...style, stroke: colors.bg, strokeWidth: 4, opacity: 0.3 }
                    : { ...style, strokeDasharray: '5,5' }
                }
            />
            {/* Animated flowing edge for cascade/congestion */}
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
            {/* SVG Gradient and Animation Definitions */}
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
                        transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
                        pointerEvents: 'all',
                    }}
                    className="nodrag nopan absolute"
                >
                    <button
                        className={`w-5 h-5 ${isFailed ? 'bg-gradient-to-r from-red-500 to-amber-500 border-red-600 animate-pulse' : isCongested ? 'bg-gradient-to-r from-orange-500 to-yellow-500 border-orange-600 animate-pulse' : 'bg-white border-slate-300'} border-2 rounded-full flex items-center justify-center text-slate-400 hover:text-red-500 hover:border-red-400 transition-colors shadow-lg ${hasIssue ? 'opacity-100' : 'opacity-0 hover:opacity-100'}`}
                        onClick={onEdgeClick}
                    >
                        <X size={10} className={hasIssue ? 'text-white' : ''} />
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
    const shockMagnitude = data.shockMagnitude || 0;

    const getNodeStyle = () => {
        if (isSelected) return 'border-indigo-600 shadow-indigo-500/30 bg-indigo-50 ring-2 ring-indigo-400';
        if (isUnconnected) return 'border-red-500 shadow-red-500/20 bg-white';
        if (isCCP) return 'border-indigo-600 shadow-indigo-500/20 bg-white';
        if (shockMagnitude > 0.5) return 'border-red-500 shadow-red-500/20 bg-red-50';
        if (shockMagnitude > 0) return 'border-amber-500 shadow-amber-500/20 bg-amber-50';
        return 'border-emerald-500 shadow-emerald-500/20 bg-white';
    };

    return (
        <div className="relative group/node">
            <div className={`w-16 h-16 md:w-20 md:h-20 rounded-full flex items-center justify-center border-2 
                ${getNodeStyle()}
                shadow-md hover:shadow-xl transition-all duration-300 transform hover:scale-105 z-10 cursor-pointer`}>
                <div className="text-center">
                    <span className={`text-xs md:text-sm font-bold ${isCCP ? 'text-indigo-600' : 'text-slate-700'}`}>
                        {data.label?.split('.')[0]?.slice(0, 8) || (isCCP ? 'CCP' : 'B')}
                    </span>
                    {shockMagnitude > 0 && !isCCP && (
                        <div className="text-[8px] text-red-500 font-bold">
                            <Zap size={8} className="inline" /> {(shockMagnitude * 100).toFixed(0)}%
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

    // Simulation State
    const [simulation, setSimulation] = useState<SimulationResult | null>(null);
    const [simulationLoading, setSimulationLoading] = useState(false);
    const [simulationError, setSimulationError] = useState<string | null>(null);

    // Node Selection & Shock State
    const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
    const [shockMagnitudeByNode, setShockMagnitudeByNode] = useState<Record<string, number>>({});

    // Global Simulation Sliders
    const [connectivityStrength, setConnectivityStrength] = useState(1.0);
    const [liquidityBufferLevel, setLiquidityBufferLevel] = useState(1.0);
    const [ccpStrictness, setCcpStrictness] = useState(0.5);
    const [correlationRegime, setCorrelationRegime] = useState(0.7);
    const [steps, setSteps] = useState(10);
    const [showSliders, setShowSliders] = useState(false);
    const [startDate, setStartDate] = useState('2023-01-01');  // Default start date

    // Search State
    const [isSearchOpen, setIsSearchOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState("");
    const [searchResults, setSearchResults] = useState<any[]>([]);
    const [isSearching, setIsSearching] = useState(false);

    // Agent Mode State
    const [isAgentMode, setIsAgentMode] = useState(false);

    // Helper to extraction ticker safely
    const getNodeTicker = (node: Node | undefined): string | null => {
        if (!node || node.data.type === 'CCP') return null;
        // 1. Try explicit ticker data
        if (node.data.ticker) return node.data.ticker;
        // 2. Try regex match for .NS
        const nsMatch = node.data.details?.match(/([A-Z0-9]+\.NS)/);
        if (nsMatch) return nsMatch[1];
        // 3. Try regex match for NSE - SYMBOL
        const nseMatch = node.data.details?.match(/NSE - ([A-Z0-9]+)/);
        if (nseMatch) return `${nseMatch[1]}.NS`;
        // 4. Fallback: if details looks like a ticker
        if (node.data.details?.match(/^[A-Z0-9]+$/)) return `${node.data.details}.NS`;
        return null;
    };

    // Convert Payoff S_t to Systemic Risk Percentage
    // Based on interpretation scale:
    // S_t 0.00-0.02 → 0-20% (Very calm)
    // S_t 0.03-0.07 → 20-50% (Normal/mild stress)
    // S_t 0.08-0.15 → 50-80% (Elevated risk)
    // S_t >0.15 → 80-100% (Dangerous/cascading)
    const payoffToRiskPercentage = (payoff: number): number => {
        if (payoff <= 0.02) {
            // 0.00-0.02 → 0-20%
            return (payoff / 0.02) * 20;
        } else if (payoff <= 0.07) {
            // 0.03-0.07 → 20-50%
            return 20 + ((payoff - 0.02) / 0.05) * 30;
        } else if (payoff <= 0.15) {
            // 0.08-0.15 → 50-80%
            return 50 + ((payoff - 0.07) / 0.08) * 30;
        } else {
            // >0.15 → 80-100%
            // Cap at 100% for very high payoffs
            return Math.min(100, 80 + ((payoff - 0.15) / 0.15) * 20);
        }
    };

    // Get selected node data
    const selectedNode = nodes.find(n => n.id === selectedNodeId);
    const selectedNodeTicker = getNodeTicker(selectedNode);
    const isCCPSelected = selectedNode?.data.type === 'CCP';

    // Load Simulation Data
    useEffect(() => {
        const fetchSimulation = async () => {
            if (!simulationId) return;
            try {
                const res = await fetch(`/api/simulation/${simulationId}`);
                if (res.ok) {
                    const data = await res.json();
                    setNodes(data.nodes || []);
                    const loadedEdges = (data.edges || []).map((e: any) => ({ ...e, type: 'custom' }));
                    setEdges(loadedEdges);
                }
            } catch (err) {
                console.error("Failed to load simulation:", err);
            }
        };
        fetchSimulation();
    }, [simulationId, setNodes, setEdges]);

    // Update node selection visual
    useEffect(() => {
        setNodes(nds => nds.map(n => ({
            ...n,
            data: {
                ...n.data,
                isSelected: n.id === selectedNodeId,
                shockMagnitude: shockMagnitudeByNode[getNodeTicker(n) || ''] || 0
            }
        })));
    }, [selectedNodeId, shockMagnitudeByNode, setNodes]);

    // Save Simulation Data
    const saveSimulation = async () => {
        if (!simulationId) return;
        try {
            await fetch(`/api/simulation/${simulationId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ nodes, edges })
            });
        } catch (err) {
            console.error("Failed to save simulation:", err);
        }
    };

    const onConnect = useCallback((params: Connection) => {
        setEdges((eds) => addEdge({
            ...params,
            type: 'custom',
            markerEnd: { type: MarkerType.ArrowClosed, width: 15, height: 15, color: '#6366f1' },
            style: { stroke: '#a5b4fc', strokeWidth: 2 },
        }, eds));
    }, [setEdges]);

    const deleteNode = (nodeId: string) => {
        setNodes((nds) => nds.filter((node) => node.id !== nodeId));
        setEdges((eds) => eds.filter((edge) => edge.source !== nodeId && edge.target !== nodeId));
        if (selectedNodeId === nodeId) setSelectedNodeId(null);
    };

    useEffect(() => {
        const connectedNodeIds = new Set<string>();
        edges.forEach(edge => {
            connectedNodeIds.add(edge.source);
            connectedNodeIds.add(edge.target);
        });

        setNodes(nds => nds.map(node => {
            const isConnected = connectedNodeIds.has(node.id) || node.data.type === 'CCP';
            return { ...node, data: { ...node.data, isUnconnected: !isConnected } };
        }));
    }, [edges, setNodes]);

    // Get tickers from nodes
    const getTickersFromNodes = (): string[] => {
        return nodes
            .filter(n => n.data.type !== 'CCP')
            .map(n => getNodeTicker(n))
            .filter((t): t is string => t !== null);
    };

    // Handle node click
    const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
        setSelectedNodeId(node.id);
    }, []);

    // Handle pane click (deselect)
    const onPaneClick = useCallback(() => {
        setSelectedNodeId(null);
    }, []);

    // Update shock magnitude for selected node
    const updateShockMagnitude = (value: number) => {
        if (isCCPSelected) {
            console.warn('Cannot shock CCP node');
            return;
        }
        if (!selectedNodeTicker) {
            console.warn('No valid ticker found for selected node:', selectedNode);
            return;
        }
        setShockMagnitudeByNode(prev => ({
            ...prev,
            [selectedNodeTicker]: value
        }));
    };

    // Run Simulation API call
    const runSimulation = async () => {
        const tickers = getTickersFromNodes();

        if (tickers.length === 0) {
            setSimulationError('No valid tickers found. Add institutions with NSE tickers.');
            return;
        }

        // Find shocked node (node with highest shock, or first shocked node)
        const shockedEntries = Object.entries(shockMagnitudeByNode).filter(([t, v]) => v > 0 && tickers.includes(t));
        let shockedNode = shockedEntries.length > 0 ? shockedEntries.sort((a, b) => b[1] - a[1])[0][0] : tickers[0];
        let shockMagnitude = shockedEntries.length > 0 ? shockMagnitudeByNode[shockedNode] : 0.3;

        setSimulationLoading(true);
        setSimulationError(null);

        try {
            const response = await fetch(`${API_URL}/simulate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    tickers,
                    start: startDate,
                    shocked_node: shockedNode,
                    shock_magnitude: shockMagnitude,
                    connectivity_strength: connectivityStrength,
                    liquidity_buffer_level: liquidityBufferLevel,
                    ccp_strictness: ccpStrictness,
                    correlation_regime: correlationRegime,
                    steps
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Simulation failed');
            }

            const result: SimulationResult = await response.json();
            setSimulation(result);

            // Update edges to show failed connections (red edges) and congestion (orange edges)
            setEdges(currentEdges => currentEdges.map(edge => {
                // Find ticker for source and target
                const edgeSourceTicker = getNodeTicker(nodes.find(n => n.id === edge.source));
                const edgeTargetTicker = getNodeTicker(nodes.find(n => n.id === edge.target));

                // Check if edge is failed (cascade)
                const isFailed = result.cascade?.failed_edges?.some(fe =>
                    (fe.source === edgeSourceTicker && fe.target === edgeTargetTicker) ||
                    (fe.source === edgeTargetTicker && fe.target === edgeSourceTicker) ||
                    result.cascade.failed_nodes.includes(edgeSourceTicker || '') ||
                    result.cascade.failed_nodes.includes(edgeTargetTicker || '')
                ) || false;

                // Check if edge is congested (connected to most congested node)
                const mostCongestedNode = result.congestion?.most_congested_node;
                const isCongested = result.congestion?.level !== 'low' && (
                    edgeSourceTicker === mostCongestedNode ||
                    edgeTargetTicker === mostCongestedNode
                );

                return {
                    ...edge,
                    data: { ...edge.data, isFailed, isCongested }
                };
            }));

        } catch (err) {
            setSimulationError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setSimulationLoading(false);
        }
    };

    // Auto-run simulation when parameters change (with debouncing)
    useEffect(() => {
        // Only auto-run if we have nodes and a previous simulation exists
        if (nodes.length === 0 || !simulation) {
            console.log('Auto-run skipped: no nodes or no initial simulation');
            return;
        }

        console.log('Auto-run scheduled: parameters changed');
        const timer = setTimeout(() => {
            console.log('Auto-running simulation...');
            runSimulation();
        }, 1000);

        return () => {
            console.log('Auto-run cancelled: parameters changed again');
            clearTimeout(timer);
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [shockMagnitudeByNode, connectivityStrength, liquidityBufferLevel, ccpStrictness, correlationRegime, steps, nodes.length]);

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
                    currentGraph: { nodes, edges },
                    isAgentMode
                })
            });

            if (!res.ok) throw new Error("AI failed to respond");

            const action = await res.json();

            // Handle node addition
            if (action.type === 'add_nodes' && action.nodes) {
                const lastNode = nodes[nodes.length - 1];
                const startX = lastNode ? lastNode.position.x + 150 : 300;
                const startY = lastNode ? lastNode.position.y : 200;
                const newNodes = action.nodes.map((n: any, i: number) => ({
                    id: `node-${Date.now()}-${i}`,
                    type: 'custom',
                    position: { x: startX + (i * 130), y: startY + (Math.random() * 80 - 40) },
                    data: n.data || { label: n.label, type: 'Bank', details: n.details }
                }));
                setNodes(nds => [...nds, ...newNodes]);

                // Auto-connect to CCP in agent mode
                if (isAgentMode) {
                    // Find or create CCP node
                    let ccpNode = nodes.find(n => n.data.type === 'CCP');
                    if (!ccpNode) {
                        ccpNode = {
                            id: 'ccp-1',
                            type: 'custom',
                            position: { x: 600, y: 300 },
                            data: { label: 'CCP', type: 'CCP', details: 'Central Counterparty' }
                        };
                        setNodes(nds => [...nds, ccpNode!]);
                    }

                    // Create edges between new nodes and CCP
                    setTimeout(() => {
                        const newEdges = newNodes.flatMap((node: any) => [
                            {
                                id: `edge-${node.id}-ccp`,
                                source: node.id,
                                target: ccpNode!.id,
                                type: 'custom',
                                markerEnd: { type: MarkerType.ArrowClosed, width: 15, height: 15, color: '#6366f1' },
                                style: { stroke: '#a5b4fc', strokeWidth: 2 }
                            },
                            {
                                id: `edge-ccp-${node.id}`,
                                source: ccpNode!.id,
                                target: node.id,
                                type: 'custom',
                                markerEnd: { type: MarkerType.ArrowClosed, width: 15, height: 15, color: '#6366f1' },
                                style: { stroke: '#a5b4fc', strokeWidth: 2 }
                            }
                        ]);
                        setEdges(eds => [...eds, ...newEdges]);
                    }, 100);
                }
            }

            // Handle node removal by matching label/ticker
            if (action.nodesToRemove && action.nodesToRemove.length > 0) {
                const nodesToRemoveSet: Set<string> = new Set(action.nodesToRemove.map((n: any) => String(n).toLowerCase()));

                // Find nodes to remove by matching label, ticker, or id
                const nodeIdsToRemove = nodes
                    .filter(node => {
                        const label = node.data.label?.toLowerCase() || '';
                        const ticker = getNodeTicker(node)?.toLowerCase() || '';
                        const details = node.data.details?.toLowerCase() || '';
                        const nodeId = node.id.toLowerCase();


                        return [...nodesToRemoveSet].some((removePattern) =>
                            label.includes(removePattern) ||
                            ticker.includes(removePattern) ||
                            details.includes(removePattern) ||
                            nodeId === removePattern
                        );
                    })
                    .map(n => n.id);

                // Remove the nodes
                setNodes(nds => nds.filter(node => !nodeIdsToRemove.includes(node.id)));

                // Remove connected edges
                setEdges(eds => eds.filter(edge =>
                    !nodeIdsToRemove.includes(edge.source) &&
                    !nodeIdsToRemove.includes(edge.target)
                ));
            }

            // Handle edge removal
            if (action.edgesToRemove && action.edgesToRemove.length > 0) {
                const edgeIdsToRemove = new Set(action.edgesToRemove);
                setEdges(eds => eds.filter(edge => !edgeIdsToRemove.has(edge.id)));
            }

            setMessages(prev => [...prev, { role: 'ai', content: action.message || 'Operation completed.' }]);

        } catch (err) {
            setMessages(prev => [...prev, { role: 'ai', content: 'Sorry, something went wrong.' }]);
        } finally {
            setIsLoading(false);
        }
    };

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
        const newNode = {
            id: `node-${Date.now()}`,
            type: 'custom',
            position: { x: lastNode ? lastNode.position.x + 150 : 300, y: lastNode ? lastNode.position.y + (Math.random() * 100 - 50) : 200 },
            data: {
                label: institution.name || ticker,
                type: institution.type || 'Bank',
                details: `${institution.exchange || 'NSE'} - ${ticker}`,
                ticker: ticker ? (ticker.endsWith('.NS') ? ticker : `${ticker}.NS`) : undefined
            }
        };
        setNodes(nds => [...nds, newNode]);
        setIsSearchOpen(false);
        setSearchQuery("");
        setSearchResults([]);
    };

    return (
        <div className="flex w-full h-screen bg-slate-50 overflow-hidden font-sans">
            {/* LEFT SIDEBAR: CHAT */}
            <aside className="w-80 h-full bg-white border-r border-slate-200 flex flex-col p-4 z-20 shadow-lg">
                <div className="flex items-center gap-3 mb-6 pt-2">
                    <div className="p-2 bg-gradient-to-br from-indigo-500 to-indigo-600 rounded-xl shadow-lg">
                        <Network size={20} className="text-white" />
                    </div>
                    <div className="flex-1">
                        <h1 className="text-md font-bold text-slate-800">Simulation</h1>
                        <p className="text-[10px] text-slate-400">AI-Powered Network</p>
                    </div>
                    <button
                        onClick={() => setIsAgentMode(!isAgentMode)}
                        className={`group relative p-2 rounded-lg transition-all duration-300 ${isAgentMode
                            ? 'bg-purple-600 shadow-lg shadow-purple-500/30'
                            : 'bg-slate-100 hover:bg-slate-200'
                            }`}
                        title={isAgentMode ? "Exit Agent Mode" : "Enter Agent Mode"}
                    >
                        <svg
                            width="18"
                            height="18"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            className={`transition-all duration-300 ${isAgentMode ? 'text-white' : 'text-slate-600 group-hover:text-purple-600'
                                }`}
                        >
                            <path d="M4 7l4-4l4 4" />
                            <path d="M8 3v9" />
                            <path d="M12 18l4 3l4-3" />
                            <path d="M16 21v-9" />
                        </svg>
                        {isAgentMode && (
                            <span className="absolute -top-1 -right-1 flex h-3 w-3">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-3 w-3 bg-purple-500"></span>
                            </span>
                        )}
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto space-y-3 mb-4 pr-1 custom-scrollbar">
                    {messages.map((msg, i) => (
                        <div key={i} className={`p-3 rounded-xl text-xs ${msg.role === 'ai' ? 'bg-slate-100 text-slate-700' : 'bg-indigo-600 text-white ml-4'}`}>
                            {msg.content}
                        </div>
                    ))}
                    {isLoading && (
                        <div className="p-3 bg-slate-100 rounded-xl text-xs text-slate-500 italic flex items-center gap-2">
                            <Loader2 size={12} className="animate-spin" /> Thinking...
                        </div>
                    )}
                </div>

                <div className="relative">
                    {isAgentMode && (
                        <div className="mb-2 flex items-center gap-2 px-3 py-1.5 bg-gradient-to-r from-purple-50 to-indigo-50 rounded-lg border border-purple-200">
                            <div className="flex items-center gap-1.5">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="text-purple-600">
                                    <path d="M4 7l4-4l4 4" />
                                    <path d="M8 3v9" />
                                    <path d="M12 18l4 3l4-3" />
                                    <path d="M16 21v-9" />
                                </svg>
                                <span className="text-[10px] font-bold text-purple-700 uppercase tracking-wider">Agent Mode Active</span>
                            </div>
                            <span className="ml-auto w-2 h-2 rounded-full bg-purple-500 animate-pulse"></span>
                        </div>
                    )}
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                        placeholder={isAgentMode ? "Paste auto-simulation prompt here..." : "Describe network changes..."}
                        className={`w-full pl-4 pr-12 py-3 border rounded-xl text-sm focus:outline-none focus:ring-2 transition-all text-slate-700 ${isAgentMode
                            ? 'bg-purple-50/50 border-purple-200 focus:ring-purple-500 placeholder:text-purple-400'
                            : 'bg-slate-50 border-slate-200 focus:ring-indigo-500'
                            }`}
                    />
                    <button
                        onClick={handleSend}
                        disabled={isLoading}
                        className={`absolute right-2 top-1/2 -translate-y-1/2 p-1.5 text-white rounded-lg disabled:opacity-50 transition-all ${isAgentMode
                            ? 'bg-purple-600 hover:bg-purple-700'
                            : 'bg-indigo-600 hover:bg-indigo-700'
                            }`}
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
                    <button
                        onClick={() => setShowSliders(!showSliders)}
                        className={`px-4 py-2 rounded-xl shadow-lg flex items-center gap-2 text-xs font-bold transition-all active:scale-95 ${showSliders ? 'bg-indigo-600 text-white' : 'bg-white text-slate-700 border border-slate-200'}`}
                    >
                        <Sliders size={14} /> Controls
                    </button>
                    <button
                        onClick={runSimulation}
                        disabled={simulationLoading || nodes.filter(n => n.data.type !== 'CCP').length === 0}
                        className={`px-4 py-2 rounded-xl shadow-lg flex items-center gap-2 text-xs font-bold transition-all active:scale-95 ${simulationLoading || nodes.filter(n => n.data.type !== 'CCP').length === 0
                            ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                            : 'bg-emerald-600 hover:bg-emerald-700 text-white shadow-emerald-500/20'
                            }`}
                    >
                        {simulationLoading ? (
                            <><Loader2 className="animate-spin" size={14} /> Simulating...</>
                        ) : (
                            <><Play size={14} /> Run Analysis</>
                        )}
                    </button>
                </div>

                {/* Global Sliders Panel */}
                {showSliders && (
                    <div className="absolute top-16 left-4 w-72 bg-white rounded-xl shadow-2xl border border-slate-200 z-40 p-4">
                        <h3 className="text-xs font-bold text-slate-800 mb-3 uppercase tracking-wider flex items-center gap-2">
                            <Settings size={12} /> Simulation Controls
                        </h3>
                        <div className="space-y-3">
                            <div className="pb-3 mb-3 border-b border-slate-100">
                                <label className="text-[10px] text-slate-500 flex items-center gap-1 mb-1">
                                    <Calendar size={10} /> Start Date
                                </label>
                                <input
                                    type="date"
                                    value={startDate}
                                    onChange={e => setStartDate(e.target.value)}
                                    max={new Date().toISOString().split('T')[0]}
                                    min="2020-01-01"
                                    className="w-full px-2 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-slate-700"
                                />
                            </div>
                            <div>
                                <label className="text-[10px] text-slate-500 flex justify-between"><span>Connectivity Strength</span><span>{connectivityStrength.toFixed(1)}</span></label>
                                <input type="range" min="0.1" max="2" step="0.1" value={connectivityStrength} onChange={e => setConnectivityStrength(parseFloat(e.target.value))} className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600" />
                            </div>
                            <div>
                                <label className="text-[10px] text-slate-500 flex justify-between"><span>Liquidity Buffer</span><span>{liquidityBufferLevel.toFixed(1)}</span></label>
                                <input type="range" min="0.1" max="2" step="0.1" value={liquidityBufferLevel} onChange={e => setLiquidityBufferLevel(parseFloat(e.target.value))} className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600" />
                            </div>
                            <div>
                                <label className="text-[10px] text-slate-500 flex justify-between"><span>CCP Strictness</span><span>{(ccpStrictness * 100).toFixed(0)}%</span></label>
                                <input type="range" min="0" max="1" step="0.1" value={ccpStrictness} onChange={e => setCcpStrictness(parseFloat(e.target.value))} className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600" />
                            </div>
                            <div>
                                <label className="text-[10px] text-slate-500 flex justify-between"><span>Correlation Regime</span><span>{(correlationRegime * 100).toFixed(0)}%</span></label>
                                <input type="range" min="0" max="1" step="0.1" value={correlationRegime} onChange={e => setCorrelationRegime(parseFloat(e.target.value))} className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600" />
                            </div>
                            <div>
                                <label className="text-[10px] text-slate-500 flex justify-between"><span>Cascade Steps</span><span>{steps}</span></label>
                                <input type="range" min="1" max="50" step="1" value={steps} onChange={e => setSteps(parseInt(e.target.value))} className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600" />
                            </div>
                        </div>
                    </div>
                )}

                {/* Node Inspector Panel */}
                {selectedNode && !isCCPSelected && (
                    <div className="absolute bottom-4 left-4 w-72 bg-white rounded-xl shadow-2xl border border-slate-200 z-50 p-4">
                        <h3 className="text-xs font-bold text-slate-800 mb-1 uppercase tracking-wider flex items-center gap-2">
                            <Zap size={12} className="text-amber-500" /> Node Inspector
                        </h3>
                        <p className="text-sm font-bold text-indigo-600 mb-3">{selectedNode.data.label}</p>
                        {selectedNodeTicker ? (
                            <div>
                                <label className="text-[10px] text-slate-500 flex justify-between">
                                    <span>Shock Magnitude</span>
                                    <span className="font-bold text-red-500">{((shockMagnitudeByNode[selectedNodeTicker] || 0) * 100).toFixed(0)}%</span>
                                </label>
                                <input
                                    type="range"
                                    min="0"
                                    max="1"
                                    step="0.05"
                                    value={shockMagnitudeByNode[selectedNodeTicker] || 0}
                                    onChange={e => updateShockMagnitude(parseFloat(e.target.value))}
                                    onInput={e => updateShockMagnitude(parseFloat((e.target as HTMLInputElement).value))}
                                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-red-500"
                                />
                                <div className="flex justify-between text-[9px] text-slate-400 mt-1">
                                    <span>No Shock</span>
                                    <span>Max Shock</span>
                                </div>
                                <button
                                    onClick={() => updateShockMagnitude(0)}
                                    className="mt-3 w-full py-1.5 text-[10px] bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-lg transition-colors"
                                >
                                    Clear Shock
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

                {/* CCP Analytics Panel */}
                {selectedNode && isCCPSelected && (
                    <div className="absolute top-0 right-0 h-full z-40 animate-in slide-in-from-right duration-300">
                        <CCPAnalyticsPanel onClose={() => setSelectedNodeId(null)} />
                    </div>
                )}

                {/* SEARCH MODAL */}
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

            {/* RIGHT SIDEBAR: RESULTS */}
            <aside className="w-72 h-full bg-white border-l border-slate-200 flex flex-col p-6 z-10 shrink-0 shadow-lg overflow-y-auto">
                {/* Results Section */}
                {(simulation || simulationError) && (
                    <div className="mb-6 pb-6 border-b border-slate-100">
                        <div className="flex items-center justify-between mb-4">
                            <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider">Risk Analysis</h2>
                            {simulation && (
                                <div className="flex items-center gap-1">
                                    <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                                    <span className="text-[10px] font-medium text-emerald-600">Done</span>
                                </div>
                            )}
                        </div>

                        {simulationError && (
                            <div className="p-3 bg-red-50 border border-red-200 rounded-xl flex items-start gap-2">
                                <AlertTriangle className="text-red-500 shrink-0 mt-0.5" size={12} />
                                <p className="text-[10px] text-red-600">{simulationError}</p>
                            </div>
                        )}

                        {simulation && (
                            <div className="space-y-3">
                                {/* Systemic Risk */}
                                <div className="p-3 bg-gradient-to-br from-indigo-50 to-white rounded-xl border border-indigo-100">
                                    <h3 className="text-[9px] font-bold text-indigo-600 uppercase tracking-widest mb-1">Systemic Risk</h3>
                                    <div className="flex items-end gap-1">
                                        <span className="text-2xl font-bold text-slate-900">
                                            {payoffToRiskPercentage(simulation.latest_payoff_S).toFixed(1)}
                                        </span>
                                        <span className="text-sm text-slate-400 mb-0.5">%</span>
                                    </div>
                                    <div className="text-[9px] text-slate-500 mt-1">
                                        {simulation.latest_payoff_S <= 0.02 && '🟢 Very Calm'}
                                        {simulation.latest_payoff_S > 0.02 && simulation.latest_payoff_S <= 0.07 && '🟡 Normal Stress'}
                                        {simulation.latest_payoff_S > 0.07 && simulation.latest_payoff_S <= 0.15 && '🟠 Elevated Risk'}
                                        {simulation.latest_payoff_S > 0.15 && '🔴 Dangerous'}
                                    </div>
                                </div>

                                {/* Payoff */}
                                <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                                    <h3 className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1 flex items-center gap-1">
                                        <TrendingUp size={10} /> Payoff S_t
                                    </h3>
                                    <span className="text-lg font-bold text-slate-900">{simulation.latest_payoff_S.toFixed(4)}</span>
                                </div>

                                {/* CCP Funds */}
                                {simulation.ccp_funds && (
                                    <div className="p-3 bg-amber-50 rounded-xl border border-amber-100">
                                        <h3 className="text-[9px] font-bold text-amber-700 uppercase tracking-widest mb-2">CCP Funds</h3>
                                        <div className="grid grid-cols-2 gap-2">
                                            <div>
                                                <span className="text-[9px] text-slate-500 block">Initial Margin</span>
                                                <span className="text-sm font-bold text-slate-900">{simulation.ccp_funds.initial_margin.toFixed(1)}</span>
                                            </div>
                                            <div>
                                                <span className="text-[9px] text-slate-500 block">Variation Margin</span>
                                                <span className="text-sm font-bold text-slate-900">{simulation.ccp_funds.variation_margin_flow.toFixed(1)}</span>
                                            </div>
                                            <div>
                                                <span className="text-[9px] text-slate-500 block">Default Fund</span>
                                                <span className="text-sm font-bold text-slate-900">{simulation.ccp_funds.default_fund.toFixed(1)}</span>
                                            </div>
                                            <div>
                                                <span className="text-[9px] text-slate-500 block">CCP Capital</span>
                                                <span className="text-sm font-bold text-slate-900">{simulation.ccp_funds.ccp_capital.toFixed(1)}</span>
                                            </div>
                                        </div>
                                        <div className="text-[9px] text-slate-400 text-center mt-2">
                                            Data: {simulation.end_date}
                                        </div>
                                    </div>
                                )}

                                {/* Congestion */}
                                <div className={`p-3 rounded-xl border ${simulation.congestion.level === 'high' ? 'bg-red-50 border-red-200' : simulation.congestion.level === 'medium' ? 'bg-amber-50 border-amber-200' : 'bg-emerald-50 border-emerald-200'}`}>
                                    <h3 className="text-[9px] font-bold uppercase tracking-widest mb-2 text-slate-600">Congestion</h3>
                                    <div className="flex justify-between text-[10px]">
                                        <span className="text-slate-500">Level</span>
                                        <span className={`font-bold uppercase ${simulation.congestion.level === 'high' ? 'text-red-600' : simulation.congestion.level === 'medium' ? 'text-amber-600' : 'text-emerald-600'}`}>
                                            {simulation.congestion.level}
                                        </span>
                                    </div>
                                    <div className="flex justify-between text-[10px] mt-1">
                                        <span className="text-slate-500">Most Congested</span>
                                        <span className="font-bold text-slate-700">{simulation.congestion.most_congested_node.split('.')[0]}</span>
                                    </div>
                                    <div className="flex justify-between text-[10px] mt-1">
                                        <span className="text-slate-500">Max Score</span>
                                        <span className="font-bold text-slate-700">{simulation.congestion.max_score.toFixed(2)}</span>
                                    </div>
                                </div>

                                {/* Cascade */}
                                <div className={`p-3 rounded-xl border ${simulation.cascade.status === 'cascade' ? 'bg-red-50 border-red-200' : 'bg-emerald-50 border-emerald-200'}`}>
                                    <h3 className="text-[9px] font-bold uppercase tracking-widest mb-2 text-slate-600">Cascade</h3>
                                    <div className="flex justify-between text-[10px]">
                                        <span className="text-slate-500">Status</span>
                                        <span className={`font-bold uppercase ${simulation.cascade.status === 'cascade' ? 'text-red-600' : 'text-emerald-600'}`}>
                                            {simulation.cascade.status}
                                        </span>
                                    </div>
                                    <div className="flex justify-between text-[10px] mt-1">
                                        <span className="text-slate-500">Failure Ratio</span>
                                        <span className="font-bold text-slate-700">{(simulation.cascade.failure_ratio * 100).toFixed(0)}%</span>
                                    </div>
                                    <div className="flex justify-between text-[10px] mt-1">
                                        <span className="text-slate-500">Cascade Depth</span>
                                        <span className="font-bold text-slate-700">{simulation.cascade.cascade_depth}</span>
                                    </div>
                                    {simulation.cascade.failed_nodes.length > 0 && (
                                        <div className="mt-2 pt-2 border-t border-slate-200">
                                            <span className="text-[9px] text-slate-500">Failed Nodes:</span>
                                            <div className="flex flex-wrap gap-1 mt-1">
                                                {simulation.cascade.failed_nodes.map(n => (
                                                    <span key={n} className="px-1.5 py-0.5 bg-red-100 text-red-600 rounded text-[8px] font-bold">{n.split('.')[0]}</span>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                    {simulation.cascade.analysis && (
                                        <div className="mt-2 pt-2 border-t border-slate-200">
                                            <span className="text-[9px] text-slate-500">Analysis:</span>
                                            <p className="text-[9px] text-slate-600 mt-1 italic">{simulation.cascade.analysis}</p>
                                        </div>
                                    )}
                                </div>

                                <div className="text-[9px] text-slate-400 text-center">
                                    Data: {simulation.end_date}
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* Institutions List */}
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider">Institutions</h2>
                    <div className="flex items-center gap-1.5 px-2 py-1 bg-indigo-50 rounded-md border border-indigo-100">
                        <Activity size={12} className="text-indigo-600" />
                        <span className="text-[10px] font-bold text-indigo-700">{nodes.length}</span>
                    </div>
                </div>

                <div className="flex-1 min-h-[200px] overflow-y-auto pr-2 custom-scrollbar space-y-2">
                    {nodes.length === 0 ? (
                        <div className="flex flex-col items-center justify-center h-full text-center py-8">
                            <Activity size={24} className="text-slate-300 mb-2" />
                            <p className="text-xs text-slate-400">No institutions yet</p>
                            <p className="text-[10px] text-slate-300 mt-1">Click "Add Institution" to start</p>
                        </div>
                    ) : (
                        nodes.map(node => {
                            const ticker = getNodeTicker(node) || '';
                            const shockValue = shockMagnitudeByNode[ticker] || 0;
                            return (
                                <div
                                    key={node.id}
                                    onClick={() => setSelectedNodeId(node.id)}
                                    className={`group p-3 border rounded-xl hover:shadow-md transition-all cursor-pointer relative ${selectedNodeId === node.id ? 'bg-indigo-50 border-indigo-300' : 'bg-white border-slate-100 hover:border-indigo-200'}`}
                                >
                                    <div className="flex items-center justify-between mb-1">
                                        <span className="text-xs font-bold text-slate-700 truncate max-w-[140px]">{node.data.label}</span>
                                        <button
                                            onClick={(e) => { e.stopPropagation(); deleteNode(node.id); }}
                                            className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-50 text-slate-300 hover:text-red-500 rounded transition-all"
                                        >
                                            <Trash2 size={12} />
                                        </button>
                                    </div>
                                    {/* Ticker and Type */}
                                    <div className="text-[9px] text-slate-400 mb-2">
                                        {ticker && <span className="font-mono">{ticker}</span>}
                                        {node.data.type && <span className="ml-2 text-indigo-500">• {node.data.type}</span>}
                                    </div>
                                    <div className="flex items-center justify-start gap-2 text-[10px] text-slate-400">
                                        <span className={`px-1.5 py-0.5 rounded-full font-bold uppercase tracking-wide text-[8px] ${node.data.isUnconnected ? 'bg-red-50 text-red-500' : 'bg-emerald-50 text-emerald-600'}`}>
                                            {node.data.isUnconnected ? 'Disconnected' : 'Connected'}
                                        </span>
                                        {shockValue > 0 && (
                                            <span className="px-1.5 py-0.5 rounded-full bg-red-100 text-red-600 text-[8px] font-bold flex items-center gap-0.5">
                                                <Zap size={8} /> {(shockValue * 100).toFixed(0)}%
                                            </span>
                                        )}
                                    </div>
                                </div>
                            );
                        })
                    )}
                </div>
            </aside>
        </div>
    );
}
