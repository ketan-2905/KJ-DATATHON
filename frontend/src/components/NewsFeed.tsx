'use client';

import { useEffect, useState } from 'react';
import { TrendingUp, TrendingDown, Minus, Newspaper, AlertTriangle } from 'lucide-react';
import indianStocks from '@/data/indianStocks.json';

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

interface NewsFeedProps {
    breakingNews?: NewsItem | null;
    onBreakingNewsDisplayed?: () => void;
}

export default function NewsFeed({ breakingNews, onBreakingNewsDisplayed }: NewsFeedProps) {
    const [news, setNews] = useState<NewsItem[]>([]);
    const [isLoading, setIsLoading] = useState(false);

    // Load initial news feed from predefined 15 stocks
    useEffect(() => {
        const loadNews = async () => {
            setIsLoading(true);
            try {
                // Get tickers from the predefined stocks list
                const tickers = indianStocks.stocks.map(s => s.ticker);

                const response = await fetch('/api/news/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ tickers, count: 10 })
                });

                if (response.ok) {
                    const data = await response.json();
                    setNews(data.news || []);
                }
            } catch (err) {
                console.error('Failed to load news:', err);
            } finally {
                setIsLoading(false);
            }
        };

        loadNews();
    }, []); // Load once on mount

    // Add breaking news to top of feed
    useEffect(() => {
        if (breakingNews) {
            setNews(prev => [breakingNews, ...prev]);
            if (onBreakingNewsDisplayed) {
                onBreakingNewsDisplayed();
            }
        }
    }, [breakingNews, onBreakingNewsDisplayed]);

    const getSentimentIcon = (sentiment: string) => {
        if (sentiment === 'positive') return <TrendingUp size={14} className="text-emerald-600" />;
        if (sentiment === 'negative') return <TrendingDown size={14} className="text-red-600" />;
        return <Minus size={14} className="text-slate-400" />;
    };

    const getSentimentBadge = (sentiment: string, confidence: number) => {
        const colors = {
            positive: 'bg-emerald-50 text-emerald-700 border-emerald-200',
            negative: 'bg-red-50 text-red-700 border-red-200',
            neutral: 'bg-slate-50 text-slate-600 border-slate-200'
        };

        return (
            <div className={`flex items-center gap-1 px-2 py-0.5 rounded-full border text-[9px] font-bold uppercase ${colors[sentiment as keyof typeof colors] || colors.neutral}`}>
                {getSentimentIcon(sentiment)}
                {sentiment} | {(confidence * 100).toFixed(0)}%
            </div>
        );
    };

    const formatTime = (timestamp: string) => {
        const date = new Date(timestamp);
        const now = new Date();
        const diffMs = now.getTime() - date.getTime();
        const diffMins = Math.floor(diffMs / 60000);

        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins} min${diffMins === 1 ? '' : 's'} ago`;
        const diffHours = Math.floor(diffMins / 60);
        if (diffHours < 24) return `${diffHours} hour${diffHours === 1 ? '' : 's'} ago`;
        return date.toLocaleDateString();
    };

    return (
        <aside className="w-80 h-full bg-white border-r border-slate-200 flex flex-col p-4 z-20 shadow-lg overflow-hidden">
            {/* Header */}
            <div className="flex items-center gap-3 mb-4 pt-2">
                <div className="p-2 bg-gradient-to-br from-slate-700 to-slate-800 rounded-xl shadow-lg">
                    <Newspaper size={20} className="text-white" />
                </div>
                <div>
                    <h2 className="text-sm font-bold text-slate-900">Market News</h2>
                    <p className="text-[10px] text-slate-400">Live Financial Feed</p>
                </div>
            </div>

            {/* News Feed */}
            <div className="flex-1 overflow-y-auto space-y-2 custom-scrollbar pr-2">
                {isLoading ? (
                    <div className="flex items-center justify-center h-32">
                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-slate-700"></div>
                    </div>
                ) : news.length === 0 ? (
                    <div className="text-center py-8 text-xs text-slate-400">
                        No news available. Add institutions to see market updates.
                    </div>
                ) : (
                    news.map((item) => (
                        <div
                            key={item.id}
                            className={`p-3 rounded-xl border transition-all ${item.is_breaking
                                ? 'bg-red-50 border-red-200 shadow-md animate-pulse-slow'
                                : 'bg-slate-50 border-slate-100 hover:shadow-sm'
                                }`}
                        >
                            {/* Breaking Badge */}
                            {item.is_breaking && (
                                <div className="flex items-center gap-1 mb-2">
                                    <AlertTriangle size={12} className="text-red-600 animate-pulse" />
                                    <span className="text-[9px] font-black text-red-600 uppercase tracking-wider">Breaking News</span>
                                </div>
                            )}

                            {/* Company */}
                            <div className="text-[10px] font-bold text-indigo-600 mb-1">
                                {item.company_name}
                            </div>

                            {/* Headline */}
                            <h3 className={`text-xs font-bold mb-2 ${item.is_breaking ? 'text-red-900' : 'text-slate-900'}`}>
                                {item.headline}
                            </h3>

                            {/* Summary */}
                            {item.summary && (
                                <p className="text-[10px] text-slate-600 mb-2 line-clamp-2">
                                    {item.summary}
                                </p>
                            )}

                            {/* Footer */}
                            <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-200">
                                {getSentimentBadge(item.sentiment, item.confidence)}
                                <span className="text-[9px] text-slate-400">
                                    {formatTime(item.timestamp)}
                                </span>
                            </div>
                        </div>
                    ))
                )}
            </div>

            {/* Custom Scrollbar Styles */}
            <style jsx>{`
                .custom-scrollbar::-webkit-scrollbar {
                    width: 6px;
                }
                .custom-scrollbar::-webkit-scrollbar-track {
                    background: #f1f5f9;
                    border-radius: 10px;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb {
                    background: #cbd5e1;
                    border-radius: 10px;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb:hover {
                    background: #94a3b8;
                }
                .animate-pulse-slow {
                    animation: pulse 3s cubic-bezier(0.4, 0, 0.6, 1) 2;
                }
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.8; }
                }
            `}</style>
        </aside>
    );
}
