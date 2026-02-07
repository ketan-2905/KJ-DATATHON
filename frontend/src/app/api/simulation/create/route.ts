import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function POST(req: Request) {
    try {
        const { name } = await req.json();
        const id = name.toLowerCase().replace(/[^a-z0-9]+/g, '-') + '-' + Date.now().toString().slice(-4);

        // 1. Create Simulation Data File
        const initialData = {
            nodes: [
                {
                    id: 'ccp-1',
                    type: 'custom',
                    position: { x: 400, y: 300 },
                    data: { label: 'CCP-1', type: 'CCP', details: 'Central Clearing House' }
                }
            ],
            edges: []
        };

        const simulationsDir = path.join(process.cwd(), 'src/data/simulations');
        if (!fs.existsSync(simulationsDir)) {
            fs.mkdirSync(simulationsDir, { recursive: true });
        }

        fs.writeFileSync(
            path.join(simulationsDir, `${id}.json`),
            JSON.stringify(initialData, null, 2)
        );

        // 2. Update Dashboard Index
        const dashboardPath = path.join(process.cwd(), 'src/data/dashboard.json');
        let dashboard = [];
        if (fs.existsSync(dashboardPath)) {
            dashboard = JSON.parse(fs.readFileSync(dashboardPath, 'utf8'));
        }

        const newEntry = {
            id,
            name,
            createdAt: new Date().toISOString(),
            path: `./simulations/${id}.json`
        };

        dashboard.unshift(newEntry);
        fs.writeFileSync(dashboardPath, JSON.stringify(dashboard, null, 2));

        return NextResponse.json({ success: true, id });
    } catch (error) {
        console.error('Failed to create simulation:', error);
        return NextResponse.json({ success: false, error: 'Internal Server Error' }, { status: 500 });
    }
}
