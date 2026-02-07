import { NextResponse } from 'next/server';

export async function GET(req: Request) {
    const { searchParams } = new URL(req.url);
    const query = searchParams.get('q');

    if (!query) {
        return NextResponse.json({ results: [] });
    }

    const url = `https://query1.finance.yahoo.com/v1/finance/search?q=${encodeURIComponent(query)}`;

    try {
        const response = await fetch(url, {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        });

        if (!response.ok) {
            throw new Error(`Yahoo API responded with ${response.status}`);
        }

        const data = await response.json();

        const results = (data.quotes || []).map((item: any) => ({
            name: item.shortname || item.longname || item.symbol,
            ticker: item.symbol,
            exchange: item.exchange,
            id: item.symbol,
            type: item.quoteType || 'Equity'
        }));

        return NextResponse.json({ results });
    } catch (error) {
        console.error('Finance Search Error:', error);
        return NextResponse.json({ results: [], error: 'Failed to fetch data' }, { status: 500 });
    }
}
