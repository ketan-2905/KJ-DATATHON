import { NextResponse } from 'next/server';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

export async function POST(req: Request) {
    try {
        const body = await req.json();

        const response = await fetch(`${API_URL}/news/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body)
        });

        if (!response.ok) {
            throw new Error(`Backend responded with ${response.status}`);
        }

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error('News generate error:', error);
        return NextResponse.json(
            { error: 'Failed to generate news' },
            { status: 500 }
        );
    }
}
