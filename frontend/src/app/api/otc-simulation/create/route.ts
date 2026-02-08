import { NextResponse } from 'next/server';

export async function POST() {
    try {
        // Generate a unique ID for the OTC simulation
        const simulationId = `otc-${Date.now()}`;

        // You can optionally store this in a database
        // For now, just return the ID

        return NextResponse.json({
            id: simulationId,
            type: 'otc',
            created_at: new Date().toISOString()
        });
    } catch (error) {
        console.error('OTC simulation creation error:', error);
        return NextResponse.json(
            { error: 'Failed to create OTC simulation' },
            { status: 500 }
        );
    }
}
