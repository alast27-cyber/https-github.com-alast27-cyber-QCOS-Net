
import { GoogleGenAI, GenerateContentParameters, GenerateContentResponse } from '@google/genai';

const isApiDisabled = true; // Global flag to disable API calls

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

export const generateContentWithRetry = async (ai: GoogleGenAI, params: GenerateContentParameters, retries = 5, delay = 2000): Promise<GenerateContentResponse> => {
    if (isApiDisabled) {
        console.warn("Gemini API calls are disabled by the operator.");
        // Simulate a delay to make the UI feel responsive without making a call
        await new Promise(resolve => setTimeout(resolve, 500));
        throw new Error("All Gemini API calls are currently disabled by the operator.");
    }
    
    for (let i = 0; i < retries; i++) {
        try {
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
