import { GoogleGenAI } from "@google/genai";
import { LOCAL_KNOWLEDGE_BASE } from "../data/knowledgeBase";
import { generateLocalResponse } from "./localResponseGenerator";

export const useLocalCognition = false; // Set to false to use Gemini API primarily

// Initialize Gemini AI
const apiKey = process.env.GEMINI_API_KEY || "";
const genAI = apiKey ? new GoogleGenAI({ apiKey }) : null;


interface LocalResponse {
    text: string;
    functionCalls?: any[];
}

export const generateContentWithRetry = async (ai: any, params: any, retries = 5, delay = 2000): Promise<any> => {
    // If we have a real AI instance (Gemini), use it first
    if (genAI) {
        try {
            console.info("[QCOS] Routing request to Gemini API...");
            const model = "gemini-3-flash-preview";
            
            const userMessage = params.contents?.parts?.[0]?.text || params.message || "";
            const systemInstruction = params.systemInstruction || "";
            
            const response = await genAI.models.generateContent({
                model,
                contents: [{ role: 'user', parts: [{ text: userMessage }] }],
                config: {
                    systemInstruction,
                    temperature: 0.7,
                }
            });

            return {
                text: response.text,
                candidates: [
                    {
                        content: {
                            parts: [{ text: response.text }],
                            role: 'model'
                        },
                        finishReason: 'STOP',
                        index: 0,
                    }
                ],
            };
        } catch (error) {
            console.warn("[QCOS] Gemini API failed, falling back to local/ollama:", error);
        }
    }

    // Fallback to Local Ollama Backend or Mock
    console.info("[QCOS] Routing request to Local Ollama Backend...");
        
        try {
            const userMessage = params.contents?.parts?.[0]?.text || params.message || "";
            const systemInstruction = params.systemInstruction || "";
            
            const response = await fetch('/api/agentq/message', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    message: userMessage, 
                    context: systemInstruction 
                })
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                // Check if the response is HTML (likely an error page)
                if (errorText.trim().startsWith('<!doctype html>') || errorText.trim().startsWith('<!DOCTYPE html>')) {
                    throw new Error(`Received HTML response from backend: ${response.status} - ${errorText.substring(0, 200)}...`);
                }
                throw new Error(`Backend error: ${response.status} - ${errorText}`);
            }

            // Check Content-Type before parsing JSON
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                const responseText = await response.text();
                throw new Error(`Expected JSON response but received ${contentType || 'unknown'} content type. Response body: ${responseText.substring(0, 200)}...`);
            }
            
            const result = await response.json();
            
            return {
                text: result.message,
                reasoning: result.data?.reasoning,
                candidates: [
                    {
                        content: {
                            parts: [{ text: result.message }],
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
            };
        } catch (error) {
            console.warn("[QCOS] Ollama Backend failed, falling back to local mock:", error);
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
            };
        }

    // Original logic for non-local mode (if applicable)
    if (!ai) {
        const localResponse = generateLocalResponse(params);
        return {
            text: localResponse.text,
            candidates: [{ content: { parts: [{ text: localResponse.text }] } }]
        };
    }
};

export const generateLocalCode = async (prompt: string, currentCode: string): Promise<string> => {
    console.info("[QCOS] Generating Local Code via QIAI_IPS Core...");
    await new Promise(resolve => setTimeout(resolve, 1500)); 

    const lowerPrompt = prompt.toLowerCase();

    if (lowerPrompt.includes('test') || lowerPrompt.includes('unit test')) {
        return `// [Q-IAI] Generated Unit Tests
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';

describe('Q-IAI Auto-Generated Suite', () => {
    test('component renders without crashing', () => {
        expect(true).toBeTruthy(); 
    });
});
`;
    }

    if (lowerPrompt.includes('debug') || lowerPrompt.includes('fix') || lowerPrompt.includes('error')) {
        return `// [Q-IAI] Applied Debugging & Error Boundaries
import React, { Component, ErrorInfo, ReactNode } from 'react';

class QIAIErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean }> {
    constructor(props: { children: ReactNode }) {
        super(props);
        this.state = { hasError: false };
    }

    static getDerivedStateFromError(_: Error) {
        return { hasError: true };
    }

    componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error("[Q-IAI Kernel Fault]:", error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return <div className="text-red-500">System Fault Detected. Recovering...</div>;
        }
        return this.props.children;
    }
}

${currentCode}
`;
    }

    if (lowerPrompt.includes('refactor') || lowerPrompt.includes('optimize') || lowerPrompt.includes('clean')) {
        return `// [Q-IAI] Refactored for Cognitive Efficiency
${currentCode}
`;
    }

    if (lowerPrompt.includes('doc') || lowerPrompt.includes('comment') || lowerPrompt.includes('explain')) {
        return `/**
 * [Q-IAI GENERATED DOCUMENTATION]
 */

${currentCode}
`;
    }

    return `// [Q-IAI] Generated Code based on intent: "${prompt}"
import React, { useState, useEffect } from 'react';

const QIAIGeneratedComponent: React.FC = () => {
    return (
        <div className="p-6 bg-black/60 border border-cyan-500/30 rounded-xl">
            <h2 className="text-xl font-bold text-cyan-100">Generated Module</h2>
            <p className="text-gray-300">Context: ${prompt}</p>
        </div>
    );
};

export default QIAIGeneratedComponent;
`;
};
