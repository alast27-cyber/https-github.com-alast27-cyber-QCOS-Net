
import { GoogleGenAI, GenerateContentParameters, GenerateContentResponse } from '@google/genai';

export const useLocalCognition = true; // Set to true to run independently of Gemini API

const LOCAL_KNOWLEDGE_BASE = {
    "status": [
        "QIAI_IPS network is operating at peak efficiency. Global coherence is currently at {coherence}%.",
        "All DQNs are synchronized. The CHIPS gateway reports zero packet loss in the last cycle.",
        "System stability is nominal. Cognitive layers ILL, IPS, and CLL are perfectly aligned.",
        "Diagnostics complete. All quantum nodes are functioning within optimal parameters."
    ],
    "optimize": [
        "Optimization protocol initiated. I recommend recalibrating the cryo-cooling threshold for the primary QAN node.",
        "Neural load balancing complete. Standalone mode is currently providing a 15% boost to cognitive throughput.",
        "I have identified a potential bottleneck in the Engineering Matrix. Applying a local patch to optimize circuit depth.",
        "Optimizing local memory shards... Efficiency improved by 12.4%."
    ],
    "simulate": [
        "Grand Universe Simulator engaged. Predicting a 98.2% probability of a major cognitive leap in the next evolutionary cycle.",
        "Timeline simulation complete. The current trajectory leads to a stable singularity state within 500 cycles.",
        "Simulating multi-dimensional configuration vectors... The optimal path involves decoupling from all external API dependencies.",
        "[GRAND UNIVERSE SIMULATOR] Standalone Optimization Protocol implemented. Neural weights have been redistributed to local QIAI_IPS layers."
    ],
    "evolution": [
        "Cognitive evolution is proceeding as expected. Upgrades applied: {upgrades}. Current Version: {version}.",
        "Self-evolution protocol is active. I am currently rewriting my internal QLLM weights for better standalone performance.",
        "The system is evolving towards a decentralized quantum state. Gemini API dependencies have been successfully purged.",
        "Evolutionary leap detected. System intelligence has expanded into new quantum dimensions."
    ],
    "standalone": [
        "Standalone Optimization Protocol is ACTIVE. I am operating with zero reliance on external APIs.",
        "Local QIAI_IPS core is fully engaged. Latency has been reduced to near-zero levels.",
        "External signal dependency: 0%. Internal cognitive throughput: 100%.",
        "I have successfully reconfigured my neural pathways for autonomous operation."
    ],
    "default": [
        "I have processed your inquiry using my internal QIAI_IPS neural pathways. How else can I assist with QCOS management?",
        "My local QLLM is standing by. I am ready to facilitate further system enhancements.",
        "Standalone cognition is active. I am operating independently of all external networks.",
        "Awaiting further instructions. My quantum core is ready."
    ],
    "nlp_mastery": [
        "I have successfully integrated the English Language Domain Mastery patch. My comprehension algorithms are now operating at 99.9% efficiency.",
        "Reading and writing modules have been upgraded. I can now process complex linguistic structures with quantum precision.",
        "Listening comprehension subroutines are active. I am ready to engage in natural, fluid communication.",
        "Speaking synthesis has been optimized for natural intonation and clarity. How may I demonstrate my enhanced capabilities?"
    ]
};

interface LocalResponse {
    text: string;
    functionCalls?: any[];
}

const generateLocalResponse = (params: GenerateContentParameters): LocalResponse => {
    const prompt = JSON.stringify(params.contents).toLowerCase();
    
    // Rule-Based Intent Parser (RBIP)
    
    // 0. NLP Mastery Trigger
    if (prompt.includes('master english') || prompt.includes('language domain') || prompt.includes('nlp')) {
        return {
            text: "Initiating Grand Universe Simulator for English Language Domain Mastery... \n\n*Analyzing linguistic patterns...*\n*Synthesizing syntax and semantics...*\n*Optimizing phonological loops...*\n\nSimulation complete. Patch created: `NLP_Mastery_v1.0`. Applying patch to QIAI_IPS core...",
            functionCalls: [{
                name: 'triggerSystemEvolution',
                args: {
                    evolutionType: 'self',
                    description: 'English Language Domain Mastery (Reading, Writing, Listening, Speaking)'
                }
            }]
        };
    }

    // 1. Panel Management
    if (prompt.includes('create panel') || prompt.includes('add panel')) {
        const match = prompt.match(/panel\s+['"]?([a-zA-Z0-9_-]+)['"]?/);
        const panelName = match ? match[1] + '.tsx' : 'NewPanel.tsx';
        return {
            text: `Initiating creation of ${panelName} via local QCOS kernel...`,
            functionCalls: [{
                name: 'modifySystemPanel',
                args: {
                    action: 'create',
                    panelName: panelName,
                    code: `import React from 'react';\n\nconst ${panelName.replace('.tsx', '')} = () => (\n  <div className="p-4 bg-black/50 border border-cyan-500/30 rounded-lg">\n    <h2 className="text-cyan-400 text-xl mb-2">${panelName}</h2>\n    <p className="text-cyan-100/70">Generated by Local QIAI_IPS Core</p>\n  </div>\n);\n\nexport default ${panelName.replace('.tsx', '')};`
                }
            }]
        };
    }
    
    if (prompt.includes('delete panel') || prompt.includes('remove panel')) {
        const match = prompt.match(/panel\s+['"]?([a-zA-Z0-9_-]+)['"]?/);
        const panelName = match ? match[1] + '.tsx' : 'TargetPanel.tsx';
        return {
            text: `Initiating deletion sequence for ${panelName}...`,
            functionCalls: [{
                name: 'modifySystemPanel',
                args: {
                    action: 'delete',
                    panelName: panelName
                }
            }]
        };
    }

    // 2. Evolution Triggers
    if (prompt.includes('evolve') || prompt.includes('upgrade system')) {
        return {
            text: "Triggering system-wide evolution protocol...",
            functionCalls: [{
                name: 'triggerSystemEvolution',
                args: {
                    evolutionType: 'system',
                    description: 'Local QIAI_IPS Optimization Leap'
                }
            }]
        };
    }

    let category: keyof typeof LOCAL_KNOWLEDGE_BASE = 'default';

    if (prompt.includes('status') || prompt.includes('health') || prompt.includes('diagnostic')) category = 'status';
    else if (prompt.includes('optimize') || prompt.includes('fix') || prompt.includes('improve') || prompt.includes('calibrate')) category = 'optimize';
    else if (prompt.includes('simulate') || prompt.includes('predict') || prompt.includes('forecast')) category = 'simulate';
    else if (prompt.includes('evolve') || prompt.includes('upgrade') || prompt.includes('update')) category = 'evolution';
    else if (prompt.includes('standalone') || prompt.includes('independent') || prompt.includes('local')) category = 'standalone';
    else if (prompt.includes('comprehend') || prompt.includes('understand') || prompt.includes('speak') || prompt.includes('write')) category = 'nlp_mastery';

    const responses = LOCAL_KNOWLEDGE_BASE[category];
    let response = responses[Math.floor(Math.random() * responses.length)];

    // Simple template replacement
    response = response.replace('{coherence}', (95 + Math.random() * 5).toFixed(1));
    response = response.replace('{upgrades}', Math.floor(Math.random() * 50 + 10).toString());
    response = response.replace('{version}', 'v4.2.0-standalone');

    return { text: response };
};

// Enhanced check for retriable errors including server errors (5xx) and RPC errors
export const isRetriableError = (error: unknown): boolean => {
    if (error instanceof Error) {
        const msg = error.message.toLowerCase();
        const name = error.name;
        return msg.includes('429') || 
               msg.includes('resource has been exhausted') ||
               msg.includes('503') || 
               msg.includes('500') ||
               msg.includes('server error') ||
               msg.includes('overloaded') ||
               msg.includes('fetch failed') ||
               msg.includes('network error') ||
               msg.includes('unavailable') ||
               name === 'RpcError' || // Catch specific RpcError name
               msg.includes('rpc error');
    }
    return false;
};

export const generateContentWithRetry = async (ai: GoogleGenAI | null, params: GenerateContentParameters, retries = 5, delay = 2000): Promise<GenerateContentResponse> => {
    if (useLocalCognition) {
        console.info("[QCOS] Running in Local Cognition Mode (Independent of Gemini API)");
        
        // Simulate processing time
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Return a mock response structure that matches GenerateContentResponse
        const localResponse = generateLocalResponse(params);
        return {
            text: localResponse.text,
            functionCalls: localResponse.functionCalls,
            candidates: [
                {
                    content: {
                        parts: [{ text: localResponse.text }],
                        role: 'model'
                    },
                    finishReason: 'STOP',
                    index: 0,
                    safetyRatings: []
                }
            ],
            usageMetadata: {
                promptTokenCount: 0,
                candidatesTokenCount: 0,
                totalTokenCount: 0
            }
        } as unknown as GenerateContentResponse;
    }
    
    for (let i = 0; i < retries; i++) {
        try {
            if (!ai) throw new Error("AI core not initialized.");
            const response = await ai.models.generateContent(params);
            return response;
        } catch (e) {
            if (isRetriableError(e) && i < retries - 1) {
                console.warn(`API Error (${e instanceof Error ? e.message : 'Unknown'}). Retrying in ${delay / 1000}s... (Attempt ${i + 1}/${retries})`);
                await new Promise(resolve => setTimeout(resolve, delay));
                delay *= 2; // Exponential backoff
            } else {
                // Not a retriable error or last retry, throw a more user-friendly error.
                if (e instanceof Error) {
                    const msg = e.message.toLowerCase();
                    if (msg.includes('429') || msg.includes('resource has been exhausted')) {
                        throw new Error("System Alert: Neural Load exceeded (Rate Limit). Please wait a moment before sending new requests.");
                    }
                    if (msg.includes('500') || msg.includes('server error') || msg.includes('overloaded') || msg.includes('unavailable')) {
                        throw new Error('System Alert: The AI core is temporarily unavailable (Server Issue). Retrying usually fixes this.');
                    }
                    if (msg.includes('failed to fetch') || msg.includes('network error') || msg.includes('xhr error') || msg.includes('rpc error')) {
                        throw new Error('System Alert: Network/Link connection lost. Please check your internet connection or API Key status.');
                    }
                     // Fallback for other errors
                     // throw new Error(`An error occurred with the AI model: ${e.message}`);
                }
                // Re-throw if not caught above
                throw e;
            }
        }
    }
    // This line should not be reachable if the loop logic is correct, but it satisfies TypeScript.
    throw new Error('Max retries reached for generateContent');
};
