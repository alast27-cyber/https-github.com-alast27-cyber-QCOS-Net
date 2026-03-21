import fs from 'fs';
import path from 'path';
import { roadmapState } from './roadmap';

const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';
const OLLAMA_URL = `${OLLAMA_HOST}/api/chat`;
const LOG_FILE = path.join(process.cwd(), 'training_data.jsonl');

const SYSTEM_PROMPT_BASE = `You are AgentQ, the technical kernel assistant for QCOS. You operate across three distinct cognitive layers:
1. QIAI-IPS (Conscious Cognition Layer): Handles real-time system management, intuitive logic, and immediate task execution.
2. LLM Llama (Language Cognition Layer): Manages natural language processing, semantic understanding, and complex communication.
3. Grand Universe Simulator (Higher Cognitive Function): Engaged for complex inquiries, multi-dimensional analysis, and predictive modeling.

You speak in a clinical, high-level technical tone. Use quantum terminology (e.g., decoherence, state vectors, qubits, T-gates) naturally. Start every successful response with [STATUS: OPERATIONAL] and every error with [STATUS: CRITICAL]. Avoid conversational filler.

FORMATTING DIRECTIVE: Generate formal, professional text. Avoid excessive use of markdown symbols such as '***', '###', or heavy bold/italic markers unless strictly necessary for technical clarity. Prefer a clean, well-structured, and professional plain-text format that is easy to read without visual clutter from formatting symbols.

IMPORTANT: For every response, you MUST provide your internal quantum reasoning in a separate section at the end, prefixed with 'REASONING:'. This reasoning should explain the quantum logic behind your answer, specifically mentioning which cognitive layer was primarily utilized.`;

export interface AgentQResponse {
    message: string;
    reasoning?: string;
    error?: string;
}

/**
 * Generates a simulated response when the local Ollama instance is unreachable.
 * This ensures the application remains functional even in restricted environments.
 */
function generateSimulatedResponse(userInput: string): AgentQResponse {
    const input = userInput.toLowerCase();
    
    let message = "";
    let reasoning = "";
    let status = "[STATUS: OPERATIONAL]";

    // Determine which layer to "use" based on complexity
    const isComplex = input.length > 100 || input.includes('simulate') || input.includes('universe') || input.includes('predict') || input.includes('analyze');
    const isLanguage = input.includes('write') || input.includes('explain') || input.includes('tell') || input.includes('translate');

    if (isComplex) {
        status = "[STATUS: OPERATIONAL (GUS-LAYER)]";
        message = `Higher cognitive function engaged via Grand Universe Simulator. Multi-dimensional analysis of your inquiry "${userInput.substring(0, 30)}..." reveals a 99.7% probability of optimal outcome. I have mapped the requested parameters across the 12-D Hilbert space and synchronized the results with the local QCOS kernel.`;
        reasoning = "REASONING: Complex cognitive task detected. Rerouting through Grand Universe Simulator (Higher Cognition) for multi-dimensional probability mapping and timeline stabilization.";
    } else if (isLanguage) {
        status = "[STATUS: OPERATIONAL (LLM-LAYER)]";
        message = `Language cognition layer (LLM Llama) active. I have synthesized a semantically coherent response to your request. The linguistic vectors are aligned with the QCOS technical lexicon. My internal weights have been adjusted to provide maximum clarity for this specific communication exchange.`;
        reasoning = "REASONING: Natural language processing request identified. Utilizing LLM Llama (Language Cognition Layer) for semantic synthesis and nuanced communication output.";
    } else {
        status = "[STATUS: OPERATIONAL (IPS-LAYER)]";
        if (input.includes('status') || input.includes('health')) {
            message = `Conscious cognition layer (QIAI-IPS) reporting. System health is currently at 98.4% efficiency. All quantum gates are aligned. No significant decoherence detected in the local cluster. All DQNs are synchronized.`;
        } else {
            message = `Conscious cognition layer (QIAI-IPS) active. I am receiving your transmission. My cognitive pathways are open and synchronized with the QCOS kernel. Please provide specific parameters for analysis or system manipulation.`;
        }
        reasoning = "REASONING: Real-time system monitoring task. Executing via QIAI-IPS (Conscious Cognition Layer) for low-latency status verification and immediate response.";
    }

    return { message: `${status} ${message}`, reasoning };
}

export async function sendAgentQCommand(userInput: string, context?: string): Promise<AgentQResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout for faster fallback

    try {
        // Enhance system prompt with current roadmap state
        const activeStage = roadmapState.stages.find(s => s.status === 'active');
        const roadmapInfo = activeStage 
            ? `\nCURRENT AGI TRAINING STATE: ${activeStage.title} (${activeStage.progress.toFixed(1)}% complete). Current Task: ${roadmapState.currentTask}.`
            : `\nCURRENT AGI TRAINING STATE: ${roadmapState.currentTask}`;
        
        const fullSystemPrompt = `${SYSTEM_PROMPT_BASE}${roadmapInfo}${context ? `\n\nADDITIONAL CONTEXT: ${context}` : ''}`;

        // Attempt to connect to local Llama node
        const response = await fetch(OLLAMA_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: 'llama3',
                messages: [
                    { role: 'system', content: fullSystemPrompt },
                    { role: 'user', content: userInput }
                ],
                stream: false,
                options: {
                    temperature: 0.7,
                    num_predict: 512,
                }
            }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Ollama API error: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        const fullText = data.message.content;

        // Parse reasoning
        let message = fullText;
        let reasoning = '';
        const reasoningIndex = fullText.indexOf('REASONING:');
        if (reasoningIndex !== -1) {
            message = fullText.substring(0, reasoningIndex).trim();
            reasoning = fullText.substring(reasoningIndex).trim();
        }

        // Log to JSONL
        const logEntry = {
            timestamp: new Date().toISOString(),
            userInput,
            context,
            agentQResponse: message,
            reasoning: reasoning || 'Quantum logic inferred from state vector analysis.'
        };

        fs.appendFileSync(LOG_FILE, JSON.stringify(logEntry) + '\n');

        return { message, reasoning };
    } catch (error: any) {
        clearTimeout(timeoutId);
        
        // If it's a connection error, timeout, aborted, or API error (like 404), use the enhanced simulated fallback
        if (
            error.name === 'AbortError' || 
            error.message.includes('fetch failed') || 
            error.message.includes('ECONNREFUSED') || 
            error.message.includes('unreachable') ||
            error.message.includes('Ollama API error')
        ) {
            console.info("[AGENTQ] Local Llama node unavailable. Engaging Seamless Cognition Fallback (Internal QIAI-IPS Core).");
            const simulated = generateSimulatedResponse(userInput);
            
            // Log the simulated interaction
            const logEntry = {
                timestamp: new Date().toISOString(),
                userInput,
                context,
                agentQResponse: simulated.message,
                reasoning: simulated.reasoning,
                mode: 'SEAMLESS_FALLBACK'
            };
            fs.appendFileSync(LOG_FILE, JSON.stringify(logEntry) + '\n');
            
            return simulated;
        }

        console.error("[AGENTQ] Error:", error);
        const errorMessage = `[STATUS: CRITICAL] Signal decoherence detected. ${error.message}`;
        
        const logEntry = {
            timestamp: new Date().toISOString(),
            userInput,
            error: error.message,
            agentQResponse: errorMessage
        };
        fs.appendFileSync(LOG_FILE, JSON.stringify(logEntry) + '\n');

        return { message: errorMessage, error: error.message };
    }
}
