import { NextResponse } from 'next/server';
import OpenAI from 'openai';

const client = new OpenAI({
    baseURL: 'https://api.featherless.ai/v1',
    apiKey: process.env.NEXT_FEATHERLESS_API_KEY,
});

export async function POST(req: Request) {
    try {
        const { message, currentGraph, isAgentMode } = await req.json();

        // Agent Mode Auto-Simulation Detection
        const isAutoSimPrompt = message.toLowerCase().includes('hdfcbank') &&
            message.toLowerCase().includes('icicibank') &&
            message.toLowerCase().includes('kotakbank') &&
            message.toLowerCase().includes('sbin');

        if (isAgentMode && isAutoSimPrompt) {
            // Auto-generate structured response for the simulation
            const response = {
                type: 'add_nodes',
                nodes: [
                    {
                        label: 'HDFC Bank',
                        data: {
                            label: 'HDFC Bank',
                            type: 'Bank',
                            details: 'NSE - HDFCBANK',
                            ticker: 'HDFCBANK.NS'
                        }
                    },
                    {
                        label: 'ICICI Bank',
                        data: {
                            label: 'ICICI Bank',
                            type: 'Bank',
                            details: 'NSE - ICICIBANK',
                            ticker: 'ICICIBANK.NS'
                        }
                    },
                    {
                        label: 'Kotak Bank',
                        data: {
                            label: 'Kotak Mahindra Bank',
                            type: 'Bank',
                            details: 'NSE - KOTAKBANK',
                            ticker: 'KOTAKBANK.NS'
                        }
                    },
                    {
                        label: 'SBI',
                        data: {
                            label: 'State Bank of India',
                            type: 'Bank',
                            details: 'NSE - SBIN',
                            ticker: 'SBIN.NS'
                        }
                    }
                ],
                message: '✅ Agent Mode: Added HDFC Bank, ICICI Bank, Kotak Mahindra Bank, and State Bank of India to the canvas. Connect them to the CCP node and click "Run Analysis" to start the simulation.'
            };
            return NextResponse.json(response);
        }

        const systemPrompt = `
You are a specialized Financial Network Architect AI. 
You control a React Flow graph visualization of a financial system.

Current Graph State:
- Node Count: ${currentGraph.nodes.length}
- Edge Count: ${currentGraph.edges.length}
- Existing Node IDs: ${currentGraph.nodes.map((n: any) => n.id).join(', ')}
${isAgentMode ? '\n**AGENT MODE ACTIVE**: You have enhanced capabilities to automatically execute multi-step workflows.' : ''}

Your Task:
Interpret the user's natural language request to MODIFIY the graph.
You can ADD nodes (Banks), ADD edges (Connections), REMOVE items, or clear the graph.

CRITICAL RULES:
1. **JSON ONLY**: You must return ONLY a raw JSON object. Do not wrap it in markdown code blocks.
2. **Structure**: 
   {
     "action": "update" | "clear",
     "nodesToAdd": [ { "id": "bank-X", "type": "custom", "position": { "x": 0, "y": 0 }, "data": { "label": "Bank X", "type": "Bank", "details": "..." } } ],
     "edgesToAdd": [ { "id": "e-X-Y", "source": "X", "target": "Y", "animated": true, "style": { "stroke": "#..." } } ],
     "nodesToRemove": [ "id1" ],
     "edgesToRemove": [ "id1" ],
     "message": "Start with a short confirmation."
   }
3. **Connectivity**: 
   - Unless specified otherwise, connect ALL new banks to the Central Counterparty (id: 'ccp-1').
   - If 'ccp-1' does not exist, you should probably create it first if the user asks for a fresh start.
4. **Positioning**:
   - You MUST generate 'x' and 'y' coordinates for new nodes. 
   - Try to arrange them in a circle around the center (400, 300) or in a grid if many.
   - Do not place them all at (0,0).
${isAgentMode ? `
5. **AGENT MODE SPECIAL INSTRUCTIONS**:
   - When user mentions specific tickers (e.g., HDFCBANK.NS, ICICIBANK.NS), create nodes with proper ticker data
   - Include ticker field in node data: { "ticker": "SYMBOL.NS" }
   - Format details as: "NSE - SYMBOL"
   - Be proactive and execute complete workflows automatically
` : ''}

User Request: "${message}"
`;

        const completion = await client.chat.completions.create({
            model: 'Qwen/Qwen2.5-7B-Instruct',
            messages: [
                { role: 'system', content: systemPrompt },
                { role: 'user', content: message }
            ],
            temperature: 0.2, // Low temperature for consistent JSON
            max_tokens: 1000,
        });

        let aiContent = completion.choices[0].message.content || "{}";

        // Clean up potential markdown formatting
        aiContent = aiContent.replace(/^```json\s*/, '').replace(/\s*```$/, '');

        // Validate JSON parsing
        try {
            JSON.parse(aiContent);
        } catch (e) {
            console.error("AI JSON Parse Error:", aiContent);
            // Fallback or error handling
            return NextResponse.json({
                message: "I configured the network, but there was a data formatting error. Please try again.",
                raw: aiContent
            }, { status: 422 });
        }

        return new NextResponse(aiContent, {
            headers: { 'Content-Type': 'application/json' }
        });

    } catch (error: any) {
        console.error('AI Route Error:', error);
        return NextResponse.json({ error: error.message || "Internal Server Error" }, { status: 500 });
    }
}
