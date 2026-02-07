import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET(req: Request, { params }: { params: { id: string } }) {
    try {
        const { id } = await params
        const filePath = path.join(process.cwd(), `src/data/simulations/${id}.json`);

        if (!fs.existsSync(filePath)) {
            return NextResponse.json({ error: 'Simulation not found' }, { status: 404 });
        }

        const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
        return NextResponse.json(data);
    } catch (error) {
        return NextResponse.json({ error: 'Failed to load simulation' }, { status: 500 });
    }
}

export async function POST(req: Request, { params }: { params: { id: string } }) {
    try {
        const { id } = await params
        const { nodes, edges } = await req.json();
        const filePath = path.join(process.cwd(), `src/data/simulations/${id}.json`);

        fs.writeFileSync(filePath, JSON.stringify({ nodes, edges }, null, 2));

        return NextResponse.json({ success: true });
    } catch (error) {
        return NextResponse.json({ error: 'Failed to save simulation' }, { status: 500 });
    }
}
