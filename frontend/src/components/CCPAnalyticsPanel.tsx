"use client";

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, AreaChart, Area, Legend, Cell } from 'recharts';
import { Activity, ShieldCheck, TrendingUp, Network, X } from 'lucide-react';

// Helper function: Convert Payoff S_t to Risk Percentage
const payoffToRiskPercentage = (payoff: number): number => {
    if (payoff <= 0.02) {
        return (payoff / 0.02) * 20;
    } else if (payoff <= 0.07) {
        return 20 + ((payoff - 0.02) / 0.05) * 30;
    } else if (payoff <= 0.15) {
        return 50 + ((payoff - 0.07) / 0.08) * 30;
    } else {
        return Math.min(100, 80 + ((payoff - 0.15) / 0.15) * 20);
    }
};

const generateMockData = () => {
    const data = [];
    for (let i = 1; i <= 30; i++) {
        // Generate a stress scenario: calm, then stress peak, then calm
        const stressFactor = i > 12 && i < 22 ?
            0.5 + 0.4 * Math.sin((i - 12) * Math.PI / 10) : // Peak around day 17
            0.1 + Math.random() * 0.1;

        // Generate S_t payoff values (typical range 0-0.3)
        const payoff_S_t = Math.max(0, Math.min(0.3, stressFactor * 0.25 + (Math.random() * 0.05)));

        // Convert to risk percentage using interpretation scale
        const riskPercentage = payoffToRiskPercentage(payoff_S_t);

        // Buffers follow stress but smoother
        const im_base = 1500;
        const im = im_base + (stressFactor * 2000);

        // Default fund reacts sharply to high stress
        const df_base = 500;
        const df = df_base + (Math.pow(stressFactor, 2) * 3000);

        // VM Flow is volatile, spikes during stress
        const vm_flow = (stressFactor * 800) + (Math.random() * 200);

        data.push({
            day: `T+${i}`,
            payoff_st: payoff_S_t.toFixed(4),
            risk: riskPercentage.toFixed(1),
            im: Math.round(im),
            df: Math.round(df),
            capital: 1000, // Constant CCP Skin-in-the-game
            vm_flow: Math.round(vm_flow),
        });
    }
    return data;
};

const driversData = [
    { name: 'Connectivity (λmax)', value: 0.82, fill: '#6366f1' }, // Indigo-500
    { name: 'Mean Risk', value: 0.54, fill: '#ec4899' }, // Pink-500
    { name: 'Std Risk', value: 0.35, fill: '#f59e0b' }, // Amber-500
    { name: 'Lag (St-1)', value: 0.65, fill: '#10b981' }, // Emerald-500
];

interface CCPAnalyticsPanelProps {
    onClose?: () => void;
}

export function CCPAnalyticsPanel({ onClose }: CCPAnalyticsPanelProps) {
    const data = generateMockData();

    return (
        <div className="flex flex-col h-full bg-slate-50/90 backdrop-blur-sm border-l border-slate-200 shadow-xl overflow-hidden w-full max-w-2xl">
            <div className="p-4 border-b border-slate-200 flex items-center justify-between bg-white">
                <div>
                    <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                        <ShieldCheck className="text-indigo-600" size={20} />
                        CCP Risk Analytics
                    </h2>
                    <p className="text-xs text-slate-500 mt-1">
                        Real-time systemic risk monitoring & buffer adequacy
                    </p>
                </div>
                {onClose && (
                    <button onClick={onClose} className="p-2 hover:bg-slate-100 rounded-full text-slate-400 hover:text-slate-600 transition-colors">
                        <X size={18} />
                    </button>
                )}
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {/* 1. Systemic Risk Over Time */}
                <Card className="shadow-sm border-slate-200">
                    <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium flex items-center gap-2 text-slate-800">
                            <Activity className="w-4 h-4 text-indigo-500" />
                            Systemic Risk Over Time
                        </CardTitle>
                        <CardDescription className="text-xs">
                            Risk percentage based on Payoff S_t (0-100%)
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="h-[200px] w-full mt-2">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={data}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                                    <XAxis dataKey="day" hide interval={4} />
                                    <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#64748b' }} axisLine={false} tickLine={false} />
                                    <Tooltip
                                        contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                        labelStyle={{ color: '#64748b', fontSize: '10px', marginBottom: '4px' }}
                                        itemStyle={{ fontSize: '11px', fontWeight: 500 }}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="risk"
                                        stroke="#6366f1"
                                        strokeWidth={2}
                                        dot={false}
                                        activeDot={{ r: 4, strokeWidth: 0 }}
                                        name="Systemic Risk"
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </CardContent>
                </Card>

                {/* 2. CCP Buffer Requirements */}
                <Card className="shadow-sm border-slate-200">
                    <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium flex items-center gap-2 text-slate-800">
                            <ShieldCheck className="w-4 h-4 text-emerald-500" />
                            Buffer Requirements Breakdown
                        </CardTitle>
                        <CardDescription className="text-xs">
                            Margins & default fund expansion vs. static capital
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="h-[200px] w-full mt-2">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={data}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                                    <XAxis dataKey="day" hide />
                                    <YAxis tick={{ fontSize: 10, fill: '#64748b' }} axisLine={false} tickLine={false} />
                                    <Tooltip
                                        contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                        labelStyle={{ color: '#64748b', fontSize: '10px' }}
                                        itemStyle={{ fontSize: '11px' }}
                                    />
                                    <Legend iconType="circle" wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
                                    <Area type="monotone" dataKey="im" stackId="1" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.8} name="Initial Margin" />
                                    <Area type="monotone" dataKey="df" stackId="1" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.8} name="Default Fund" />
                                    <Area type="monotone" dataKey="capital" stackId="1" stroke="#10b981" fill="#10b981" fillOpacity={0.8} name="CCP Capital" />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </CardContent>
                </Card>

                {/* 3. VM Flow Intensity */}
                <Card className="shadow-sm border-slate-200">
                    <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium flex items-center gap-2 text-slate-800">
                            <TrendingUp className="w-4 h-4 text-amber-500" />
                            Variation Margin Flow Intensity
                        </CardTitle>
                        <CardDescription className="text-xs">
                            Daily liquidity pressure (Flow magnitude, not balance)
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="h-[200px] w-full mt-2">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={data}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                                    <XAxis dataKey="day" hide />
                                    <YAxis tick={{ fontSize: 10, fill: '#64748b' }} axisLine={false} tickLine={false} />
                                    <Tooltip
                                        contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                        cursor={{ fill: '#f1f5f9' }}
                                        labelStyle={{ color: '#64748b', fontSize: '10px' }}
                                        itemStyle={{ fontSize: '11px' }}
                                    />
                                    <Bar dataKey="vm_flow" fill="#f43f5e" name="VM Flow ($)" radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </CardContent>
                </Card>

                {/* 4. Network Stress Drivers */}
                <Card className="shadow-sm border-slate-200">
                    <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium flex items-center gap-2 text-slate-800">
                            <Network className="w-4 h-4 text-purple-500" />
                            Network Stress Drivers
                        </CardTitle>
                        <CardDescription className="text-xs">
                            Structural factors driving current systemic risk
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="h-[200px] w-full mt-2">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart layout="vertical" data={driversData} margin={{ left: 0, right: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#e2e8f0" />
                                    <XAxis type="number" domain={[0, 1]} hide />
                                    <YAxis
                                        dataKey="name"
                                        type="category"
                                        tick={{ fontSize: 10, fill: '#475569', fontWeight: 500 }}
                                        width={100}
                                        axisLine={false}
                                        tickLine={false}
                                    />
                                    <Tooltip
                                        contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                        cursor={{ fill: '#f1f5f9' }}
                                        itemStyle={{ fontSize: '11px' }}
                                    />
                                    <Bar dataKey="value" name="Impact Score" radius={[0, 4, 4, 0]} barSize={24}>
                                        {
                                            driversData.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={entry.fill} />
                                            ))
                                        }
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
