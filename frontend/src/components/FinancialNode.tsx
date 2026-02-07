"use client";
import React, { useCallback, useState, useEffect } from 'react';
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
    Node,
    Edge,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Play, Network, AlertTriangle, TrendingUp, Loader2 } from 'lucide-react';
import configData from '../../config.json';

// --- Types ---
interface NodeData {
    label: string;
    type: string;
    ticker?: string;
    details: string;
    risk?: number;
    isActive?: boolean;
}

interface PredictionResult {
    predicted_next_systemic_risk: number;
    latest_S_t: number;
    latest_features: Record<string, number>;
    used_tickers: string[];
    masked_tickers: string[];
    end_date: string;
}

// --- Custom Node Component ---
const FinancialNode = ({ data, id }: NodeProps<NodeData>) => {
    const isCCP = data.type === 'CCP';
    const isActive = data.isActive !== false;
    const riskLevel = data.risk || 0;

    // Color based on risk level
    const getRiskColor = () => {
        if (!isActive) return 'border-slate-200 bg-slate-50 opacity-50';
        if (isCCP) return 'border-indigo-500 shadow-indigo-500/20 bg-white';
        if (riskLevel > 0.7) return 'border-red-500 shadow-red-500/20 bg-red-50';
        if (riskLevel > 0.4) return 'border-amber-500 shadow-amber-500/20 bg-amber-50';
        return 'border-emerald-500 shadow-emerald-500/20 bg-white';
    };

    return (
        <div className="relative group">
            <div className={`
                w-16 h-16 md:w-20 md:h-20 rounded-full 
                flex items-center justify-center 
                border-2 
                ${getRiskColor()}
                shadow-sm hover:shadow-xl
                transition-all duration-300 transform group-hover:scale-105
                cursor-pointer z-10
            `}>
                <div className="text-center">
                    <span className={`
                        text-xs md:text-sm font-bold
                        ${isCCP ? 'text-indigo-600' : isActive ? 'text-slate-700' : 'text-slate-400'}
                    `}>
                        {data.label.split('.')[0]}
                    </span>
                </div>
            </div>

            <Handle type="target" position={Position.Top} className="w-3 h-3 !bg-slate-300 hover:!bg-indigo-500" style={{ top: -5 }} />
            <Handle type="source" position={Position.Bottom} className="w-3 h-3 !bg-slate-300 hover:!bg-indigo-500" style={{ bottom: -5 }} />
            <Handle type="target" position={Position.Left} className="w-3 h-3 !bg-slate-300 hover:!bg-indigo-500" style={{ left: -5 }} />
            <Handle type="source" position={Position.Right} className="w-3 h-3 !bg-slate-300 hover:!bg-indigo-500" style={{ right: -5 }} />

            {/* Hover Card */}
            <div className="absolute top-full left-1/2 -translate-x-1/2 mt-4 w-48 p-3 rounded-xl bg-white/90 backdrop-blur-md border border-slate-100 shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-300 z-50 pointer-events-none">
                <div className="flex items-center gap-2 mb-2 border-b border-slate-100 pb-2">
                    <div className={`w-1.5 h-1.5 rounded-full ${isCCP ? 'bg-indigo-500' : 'bg-emerald-400'}`}></div>
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">{data.type}</span>
                </div>
                <h3 className="font-bold text-slate-800 text-sm mb-1">{data.label}</h3>
                <p className="text-[10px] text-slate-500 leading-relaxed">{data.details}</p>
                {data.ticker && <div className="text-[9px] text-slate-400 font-mono mt-2">Ticker: {data.ticker}</div>}
            </div>
        </div>
    );
};

const nodeTypes = { custom: FinancialNode };

// --- API URL ---
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

export default function FinancialInfrastructureMap() {
    const [nodes, setNodes, onNodesChange] = useNodesState<NodeData>([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);
    const [selectedTickers, setSelectedTickers] = useState<string[]>([]);
    const [prediction, setPrediction] = useState<PredictionResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const allTickers = configData.tickers;

    // Initialize CCP node
    useEffect(() => {
        const ccpNode: Node<NodeData> = {
            id: 'ccp',
            type: 'custom',
            position: { x: 400, y: 300 },
            data: {
                label: 'CCP',
                type: 'CCP',
                details: 'Central Counterparty Clearing House',
                isActive: true
            }
        };
        setNodes([ccpNode]);
    }, [setNodes]);

    // Toggle ticker selection
    const toggleTicker = (ticker: string) => {
        setSelectedTickers(prev => {
            if (prev.includes(ticker)) {
                // Remove ticker and its node
                setNodes(nds => nds.filter(n => n.id !== ticker));
                setEdges(eds => eds.filter(e => e.source !== ticker && e.target !== ticker));
                return prev.filter(t => t !== ticker);
            } else {
                // Add ticker as node
                const angle = (prev.length * 2 * Math.PI) / Math.max(allTickers.length, 8);
                const radius = 200;
                const newNode: Node<NodeData> = {
                    id: ticker,
                    type: 'custom',
                    position: {
                        x: 400 + radius * Math.cos(angle),
                        y: 300 + radius * Math.sin(angle)
                    },
                    data: {
                        label: ticker.replace('.NS', ''),
                        type: 'Bank',
                        ticker: ticker,
                        details: `NSE Stock: ${ticker}`,
                        isActive: true
                    }
                };

                const newEdge: Edge = {
                    id: `e-${ticker}-ccp`,
                    source: ticker,
                    target: 'ccp',
                    animated: true,
                    style: { stroke: '#6366f1' }
                };

                setNodes(nds => [...nds, newNode]);
                setEdges(eds => [...eds, newEdge]);
                return [...prev, ticker];
            }
        });
        setPrediction(null);
        setError(null);
    };

    // Manual connection
    const onConnect = useCallback(
        (params: Connection) => setEdges((eds) => addEdge({
            ...params,
            animated: true,
            style: { stroke: '#94a3b8', strokeWidth: 1.5 },
        }, eds)),
        [setEdges],
    );

    // Run prediction
    const runPrediction = async () => {
        if (selectedTickers.length === 0) {
            setError('Please select at least one stock');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tickers: selectedTickers })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Prediction failed');
            }

            const result: PredictionResult = await response.json();
            setPrediction(result);

            // Update node visuals based on result
            setNodes(nds => nds.map(node => {
                if (node.id === 'ccp') return node;
                const isMasked = result.masked_tickers.includes(node.id);
                return {
                    ...node,
                    data: {
                        ...node.data,
                        isActive: !isMasked,
                        risk: isMasked ? 0 : Math.random() // TODO: Get actual risk from API if available
                    }
                };
            }));

        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex w-full h-screen bg-slate-50 overflow-hidden font-sans">
            {/* Left Sidebar - Stock Picker */}
            <aside className="w-80 h-full bg-white border-r border-slate-200 flex flex-col p-6 z-10 shrink-0 shadow-[4px_0_24px_-12px_rgba(0,0,0,0.1)]">
                <div className="mb-6">
                    <h1 className="text-xl font-bold text-slate-900 tracking-tighter flex items-center gap-2">
                        <Network className="text-indigo-600" size={24} />
                        Equilibrium
                    </h1>
                    <p className="text-[10px] text-slate-400 mt-1 uppercase tracking-widest font-semibold pl-8">Systemic Risk</p>
                </div>

                {/* Stock Picker */}
                <div className="flex-1 overflow-y-auto">
                    <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Select Stocks</h3>
                    <div className="space-y-2">
                        {allTickers.map(ticker => (
                            <button
                                key={ticker}
                                onClick={() => toggleTicker(ticker)}
                                className={`w-full py-2 px-3 
                                    
                                    rounded-xl text-left text-sm font-medium transition-all ${selectedTickers.includes(ticker)
                                        ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/20'
                                        : 'bg-slate-50 text-slate-600 hover:bg-slate-100 border border-slate-100'
                                    }`}
                            >
                                {ticker.replace('.NS', '')}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Run Button */}
                <div className="mt-4 pt-4 border-t border-slate-100">
                    <button
                        onClick={runPrediction}
                        disabled={loading || selectedTickers.length === 0}
                        className={`w-full py-4 px-4 rounded-xl font-bold transition-all flex items-center justify-center gap-2 ${loading || selectedTickers.length === 0
                            ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                            : 'bg-emerald-600 hover:bg-emerald-700 text-white shadow-lg shadow-emerald-500/20 active:scale-95'
                            }`}
                    >
                        {loading ? (
                            <><Loader2 className="animate-spin" size={18} /> Running...</>
                        ) : (
                            <><Play size={18} /> Run Analysis</>
                        )}
                    </button>
                    <p className="text-[10px] text-slate-400 mt-2 text-center">
                        {selectedTickers.length} stocks selected
                    </p>
                </div>
            </aside>

            {/* Main Canvas */}
            <main className="flex-1 h-full relative bg-slate-50/50">
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    nodeTypes={nodeTypes}
                    fitView
                    proOptions={{ hideAttribution: true }}
                    minZoom={0.5}
                    maxZoom={1.5}
                    className="bg-slate-50"
                >
                    <Background color="#94a3b8" gap={32} size={1} className="opacity-20" />
                    <Controls className="!bg-white !border-slate-200 !text-slate-600 !shadow-sm !rounded-xl overflow-hidden m-4" />
                </ReactFlow>
            </main>

            {/* Right Sidebar - Results */}
            <aside className="w-80 h-full bg-white border-l border-slate-200 flex flex-col p-6 z-10 shrink-0 shadow-[-4px_0_24px_-12px_rgba(0,0,0,0.1)]">
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider">Risk Analysis</h2>
                    {prediction && (
                        <div className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                            <span className="text-xs font-medium text-emerald-600">Complete</span>
                        </div>
                    )}
                </div>

                {error && (
                    <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-xl flex items-start gap-2">
                        <AlertTriangle className="text-red-500 shrink-0 mt-0.5" size={16} />
                        <p className="text-xs text-red-600">{error}</p>
                    </div>
                )}

                {prediction ? (
                    <div className="space-y-4">
                        {/* Systemic Risk Score */}
                        <div className="p-4 bg-gradient-to-br from-indigo-50 to-white rounded-2xl border border-indigo-100">
                            <h3 className="text-[10px] font-bold text-indigo-600 uppercase tracking-widest mb-2">Systemic Risk Score</h3>
                            <div className="flex items-end gap-2">
                                <span className="text-4xl font-bold text-slate-900">
                                    {(prediction.predicted_next_systemic_risk * 100).toFixed(1)}
                                </span>
                                <span className="text-lg text-slate-400 mb-1">%</span>
                            </div>
                            <p className="text-[10px] text-slate-500 mt-2">Predicted for next period</p>
                        </div>

                        {/* Latest S_t */}
                        <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100">
                            <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2 flex items-center gap-2">
                                <TrendingUp size={12} /> Current Payoff
                            </h3>
                            <span className="text-2xl font-bold text-slate-900">
                                {prediction.latest_S_t.toFixed(4)}
                            </span>
                        </div>

                        {/* Features */}
                        <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100">
                            <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-3">Features</h3>
                            <div className="space-y-2 text-xs">
                                {Object.entries(prediction.latest_features).slice(0, 6).map(([key, value]) => (
                                    <div key={key} className="flex justify-between">
                                        <span className="text-slate-500">{key}</span>
                                        <span className="font-mono text-slate-700">{typeof value === 'number' ? value.toFixed(4) : value}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Meta */}
                        <div className="text-[10px] text-slate-400 text-center">
                            Data as of {prediction.end_date}
                        </div>
                    </div>
                ) : (
                    <div className="flex-1 flex items-center justify-center">
                        <div className="text-center text-slate-400">
                            <Network size={48} className="mx-auto mb-4 opacity-20" />
                            <p className="text-sm">Select stocks and click<br /><strong>Run Analysis</strong></p>
                        </div>
                    </div>
                )}
            </aside>
        </div>
    );
}