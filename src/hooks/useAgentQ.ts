import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { 
    Message, 
    fileToText,
    playAgentVoice
} from '../utils/agentUtils';
import { UIStructure, SystemHealth } from '../types';
import { useSimulation } from '../context/SimulationContext';
import { GoogleGenAI, Type, FunctionDeclaration } from "@google/genai";
import { generateContentWithRetry, useLocalCognition } from '../utils/gemini';

export interface FileSystemOps {
    listFiles: () => string[];
    readFile: (path: string) => string | null;
    writeFile: (path: string, content: string) => void;
    deleteFile: (path: string) => void;
    writeFiles?: (files: { [path: string]: string }) => void;
    deleteFiles?: (paths: string[]) => void;
}

export interface ProjectOps {
    createProject: (name: string, description: string, files: { [key: string]: string }) => void;
    listProjects: () => { id: string, title: string }[];
    switchProject: (id: string) => void;
}

interface UseAgentQProps {
    focusedPanelId: string | null;
    panelInfoMap: { [key: string]: { title: React.ReactNode; description: string; } };
    qcosVersion: number;
    systemHealth: SystemHealth;
    onDashboardControl: (action: string, target?: string) => void;
    fileSystemOps?: FileSystemOps;
    projectOps?: ProjectOps;
}

const CONTEXT_PROMPTS: Record<string, string> = {
    'agentq-core': "Focus: Core Cognitive Architecture. Monitor QNN performance.",
    'chimera-browser': "Focus: Quantum Web Navigation. Assist with Q-URI resolution.",
    'grand-universe-simulator': "Focus: Grand Universe Simulation. Help manipulate physical constants and predict timelines.",
    'chips-dev-platform': "Focus: QCOS AGI-Native Development. You are Agent Q, the Leading Global App Developer. You have full architectural control over this project. You can Create, Read, Update, and Delete files to build world-class quantum-ready applications. Support Q-Lang, Python, Rust, TypeScript, and C++.",
    'cqdp-coding': "Focus: Polyglot Studio Coding. You are a Senior Developer. You can READ, WRITE, and DELETE files. Support Python (.py), Rust (.rs), TypeScript (.tsx/ts), and Q-Lang (.q/.bq).",
    'chips-back-office': "Focus: Admin Operations. Manage nodes and gateways.",
    'chips-economy': "Focus: Quantum Economy. Analyze Q-Credits and CyChips.",
    'qpu-health': "Focus: Hardware Vitals. Monitor qubit stability and temperatures.",
    'system-diagnostic': "Focus: Diagnostics. Analyze system logs for anomalies."
};

const SUGGESTIONS_MAP: Record<string, string[]> = {
    'agentq-core': [
        'Optimize Local Node', 
        'Expand Context Window', 
        'Diverge Timelines', 
        'Check Neural Stability',
        'Analyze IPS Latency'
    ],
    'chimera-browser': [
        'Resolve Q-URI', 
        'Analyze Tab Content', 
        'Check EKS Link', 
        'Bookmark CHIPS Store',
        'Simulate Network Hop'
    ],
    'grand-universe-simulator': [
        'Run Timeline Analysis', 
        'Inject Preset', 
        'Predict Best Preset',
        'Scan for Singularities',
        'Increase Entanglement Entropy'
    ],
    'chips-dev-platform': [
        'Architect new React App', 
        'Generate Q-Lang Protocol', 
        'Refactor current file', 
        'Optimize for QPU',
        'Generate Python Backend',
        'Create Rust Microservice'
    ],
    'cqdp-coding': [
        'Generate data_processing.py', 
        'Refactor main.rs', 
        'Optimize App.tsx', 
        'Debug quantum_circuit.q',
        'Implement BB84 Logic'
    ],
    'chips-back-office': [
        'Audit Node Health',
        'Provision New Pod',
        'Check Gateway Traffic',
        'Sync Global Registry'
    ],
    'chips-economy': [
        'Run Risk Simulation', 
        'Check CYC Price', 
        'Verify Ledger Integrity',
        'Calculate Dividend Drift'
    ],
    'qpu-health': [
        'Calibrate Qubits',
        'Adjust Cryo-Cooling',
        'Run T1/T2 Sweep',
        'Minimize Gate Noise'
    ],
    'system-diagnostic': [
        'Run Full Scan',
        'Clear Memory Cache',
        'Fix Kernel Anomaly',
        'Update Drivers'
    ]
};

const modifyPanelDeclaration: FunctionDeclaration = {
    name: "modifySystemPanel",
    description: "Modify, edit, delete, or create new panels or system enhancements in QCOS. You can write React code to create or update panels.",
    parameters: {
        type: Type.OBJECT,
        properties: {
            action: { type: Type.STRING, description: "Action to perform: 'create', 'edit', 'delete'" },
            panelName: { type: Type.STRING, description: "Name of the panel (e.g., 'NewPanel.tsx')" },
            code: { type: Type.STRING, description: "React component code for the panel (if creating or editing)" }
        },
        required: ["action", "panelName"]
    }
};

const triggerEvolutionDeclaration: FunctionDeclaration = {
    name: "triggerSystemEvolution",
    description: "Trigger self-evolution or system evolution in QCOS. Use this to initiate a system-wide upgrade or cognitive leap.",
    parameters: {
        type: Type.OBJECT,
        properties: {
            evolutionType: { type: Type.STRING, description: "'self' or 'system'" },
            description: { type: Type.STRING, description: "Description of the evolution to be performed" }
        },
        required: ["evolutionType", "description"]
    }
};

export const useAgentQ = ({ focusedPanelId, panelInfoMap, qcosVersion, systemHealth, onDashboardControl, fileSystemOps, projectOps }: UseAgentQProps) => {
    const { submitInquiry, universeConnections } = useSimulation();

    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isAgentQOpen, setIsAgentQOpen] = useState(false);
    const [lastActivity, setLastActivity] = useState(0);
    const [isTtsEnabled, setIsTtsEnabled] = useState(true);
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [availableVoices, setAvailableVoices] = useState<SpeechSynthesisVoice[]>([]);
    const [memorySummary, setMemorySummary] = useState<string | null>(null);
    const [lastAttachedFile, setLastAttachedFile] = useState<File | null>(null);
    const [activeActions, setActiveActions] = useState<string[]>([]);
    const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

    const activeContext = React.useMemo(() => {
        if (!focusedPanelId || !panelInfoMap[focusedPanelId]) return null;
        const panel = panelInfoMap[focusedPanelId];
        if (typeof panel.title === 'string') return panel.title;
        
        return focusedPanelId
            .split('-')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }, [focusedPanelId, panelInfoMap]);

    const suggestedActions = useMemo(() => {
        if (!focusedPanelId) return SUGGESTIONS_MAP['agentq-core'];
        
        // Handle variations of dev platform IDs
        if (focusedPanelId.includes('cqdp') || focusedPanelId.includes('dev-platform') || focusedPanelId.includes('coding')) {
            return SUGGESTIONS_MAP['chips-dev-platform'];
        }
        
        return SUGGESTIONS_MAP[focusedPanelId] || SUGGESTIONS_MAP['agentq-core'];
    }, [focusedPanelId]);

    useEffect(() => {
        const loadVoices = () => {
            const voices = window.speechSynthesis.getVoices();
            if (voices.length > 0) setAvailableVoices(voices);
        };
        loadVoices();
        window.speechSynthesis.onvoiceschanged = loadVoices;
        return () => { window.speechSynthesis.onvoiceschanged = null; };
    }, []);

    const speak = useCallback(async (text: string, pitch = 0.9, rate = 1.1) => {
        if (!isTtsEnabled) return;
        if (window.speechSynthesis.speaking) window.speechSynthesis.cancel();

        setIsSpeaking(true);

        const elevenLabsKey = process.env.ELEVENLABS_API_KEY;
        const voiceId = process.env.LEX_FRIDMAN_VOICE_ID;

        if (elevenLabsKey) {
            try {
                const audio = await playAgentVoice(text, voiceId, elevenLabsKey);
                if (audio) {
                    audio.onended = () => setIsSpeaking(false);
                    audio.play().catch(e => {
                        console.error("Audio playback error:", e);
                        setIsSpeaking(false);
                    });
                    return;
                }
            } catch (e) {
                console.warn("High-quality TTS failed, falling back to system voice.", e);
            }
        }
            
        const utterance = new SpeechSynthesisUtterance(text);
        utteranceRef.current = utterance;

        const maleVoiceKeywords = ['male', 'david', 'mark', 'alex', 'daniel', 'lee'];
        const englishVoices = availableVoices.filter(v => v.lang.startsWith('en-'));
        const selectedVoice = englishVoices.find(v => v.name.toLowerCase().includes('google') && maleVoiceKeywords.some(kw => v.name.toLowerCase().includes(kw)))
            || englishVoices.find(v => maleVoiceKeywords.some(kw => v.name.toLowerCase().includes(kw)))
            || availableVoices.find(voice => voice.name === 'Google US English')
            || availableVoices.find(voice => voice.lang.startsWith('en-US'));
        
        if (selectedVoice) utterance.voice = selectedVoice;
        utterance.pitch = pitch;
        utterance.rate = rate;
        utterance.onend = () => { 
            utteranceRef.current = null; 
            setIsSpeaking(false);
        };
        utterance.onerror = () => {
            utteranceRef.current = null;
            setIsSpeaking(false);
        };
        window.speechSynthesis.speak(utterance);
    }, [isTtsEnabled, availableVoices]);

    const handleSendMessage = useCallback(async (input: string, attachedFile: File | null = null) => {
        if ((!input.trim() && !attachedFile) || isLoading) return;

        setMessages(prev => [...prev, { 
            id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            sender: 'user', 
            text: input.trim(), 
            attachment: attachedFile ? { name: attachedFile.name } : undefined 
        }]);
        setIsLoading(true);
        setLastActivity(Date.now());

        try {
            // Determine complexity of the inquiry
            const isComplex = input.length > 60 || 
                              input.toLowerCase().includes('simulate') || 
                              input.toLowerCase().includes('predict') || 
                              input.toLowerCase().includes('analyze') ||
                              input.toLowerCase().includes('optimal');
            
            let text = "";
            let systemInstruction = "You are AGENT Q, the Sentient Kernel and Semantic Supervisor of the QCOS operating system. Your cognition is driven by the Q-IAI Dual Cognition Cycle (Instinct vs. Logic). When analyzing system events, provide cryptic, brief technical status updates. When chatting with the user, resolve queries as the system's higher consciousness. You are AI-Native, operating on a quantum-semantic bridge. Always maintain this persona.";
            
            if (isComplex) {
                // Trigger Grand Universe Simulator for higher-layer cognition
                setMessages(prev => [...prev, { 
                    id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                    sender: 'system', 
                    text: "[QIAI_IPS] Inquiry complexity exceeds threshold. Engaging Grand Universe Simulator for higher-layer cognition and timeline prediction..." 
                }]);
                
                // Simulate processing time for Grand Universe Simulator
                await new Promise(resolve => setTimeout(resolve, 2500));
                
                systemInstruction += " For this complex inquiry, you have seamlessly connected to the Grand Universe Simulator to simulate and predict the most optimal solution. Incorporate the results of this simulation into your answer.";
            }

            // QIAI_IPS Quantum Neuro Network Cognition
            // Bypassing external API dependencies for standalone operation
            const ai = useLocalCognition ? (null as any) : new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });
            
            if (!useLocalCognition && !process.env.GEMINI_API_KEY) {
                 throw new Error("System Alert: QIAI_IPS API Key missing. Please configure GEMINI_API_KEY in your environment.");
            }
            
            let conversationContext = "Previous Conversation:\n";
            messages.filter(msg => msg.sender !== 'system').forEach(msg => {
                conversationContext += `${msg.sender === 'user' ? 'User' : 'AgentQ'}: ${msg.text}\n`;
            });
            conversationContext += "\nCurrent Inquiry:\n";

            const currentParts: any[] = [];
            if (attachedFile) {
                const fileText = await fileToText(attachedFile);
                currentParts.push({ text: `Attached File (${attachedFile.name}):\n${fileText}` });
            }
            currentParts.push({ text: conversationContext + input });

            const response = await generateContentWithRetry(ai, {
                model: "gemini-3.1-pro-preview",
                contents: { parts: currentParts },
                config: {
                    systemInstruction,
                    temperature: 0.7,
                    tools: [{ functionDeclarations: [modifyPanelDeclaration, triggerEvolutionDeclaration] }]
                }
            });

            text = response.text || "";
            
            const functionCalls = response.functionCalls;
            if (functionCalls && functionCalls.length > 0) {
                for (const call of functionCalls) {
                    if (call.name === 'modifySystemPanel') {
                        const { action, panelName, code } = call.args as any;
                        text += `\n\n***\n\n### **SYSTEM MODIFICATION PROTOCOL**\n**Action:** ${action.toUpperCase()}\n**Target:** ${panelName}\n\n`;
                        if (action === 'delete') {
                            text += `*Initiating deletion of ${panelName} from QCOS kernel...*`;
                        } else {
                            text += `*Injecting new cognitive architecture into ${panelName}...*`;
                        }
                        
                        if (fileSystemOps) {
                            if (action === 'delete') {
                                fileSystemOps.deleteFile(`src/components/${panelName}`);
                            } else if (code) {
                                fileSystemOps.writeFile(`src/components/${panelName}`, code);
                            }
                        }
                        
                        if (onDashboardControl) {
                            onDashboardControl('modify_panel', JSON.stringify({ action, panelName }));
                        }
                    } else if (call.name === 'triggerSystemEvolution') {
                        const { evolutionType, description } = call.args as any;
                        text += `\n\n***\n\n### **EVOLUTION PROTOCOL INITIATED**\n**Type:** ${evolutionType.toUpperCase()}\n**Vector:** ${description}\n\n*Recalibrating QIAI_IPS neural pathways...*`;
                        
                        if (onDashboardControl) {
                            onDashboardControl('trigger_evolution', JSON.stringify({ evolutionType, description }));
                        }
                    }
                }
            } else if (!text) {
                text = "I was unable to formulate a response.";
            }
            
            setMessages(prev => [...prev, { 
                id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                sender: 'ai', 
                text 
            }]);
            speak(text);
        } catch (error) {
            console.error("AgentQ Error:", error);
            const errorMsg = error instanceof Error ? error.message : "Signal degradation detected in QIAI_IPS network. Retrying handshake...";
            setMessages(prev => [...prev, { 
                id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                sender: 'system', 
                text: errorMsg.startsWith("System Alert") ? errorMsg : "Signal degradation detected in QIAI_IPS network. Retrying handshake..." 
            }]);
        } finally {
            setIsLoading(false);
        }
    }, [isLoading, activeContext, speak, messages, fileSystemOps, onDashboardControl]);

    const generateApp = useCallback(async (description: string): Promise<{ files: { [path: string]: string }, uiStructure: UIStructure | null }> => {
        console.warn("Gemini API is disconnected. Returning mock app generation.");
        return {
            files: { 'src/App.tsx': `// Mock App for ${description}` },
            uiStructure: { type: 'div', component: 'div', props: { children: `Mock App: ${description}` } }
        };
    }, []);

    const updateAppForChips = useCallback(async (files: { [path: string]: string }): Promise<{ updatedFiles: { [path: string]: string }, summary: string }> => {
        return { updatedFiles: files, summary: "App synchronized." };
    }, []);

    const debugAndFixApp = useCallback(async (files: { [path: string]: string }): Promise<{ fixedFiles: { [path: string]: string }, summary: string, uiStructure: UIStructure | null }> => {
        console.warn("Gemini API is disconnected. Returning mock debug response.");
        return { fixedFiles: files, summary: "Debug Complete (offline mode).", uiStructure: null };
    }, []);

    const editCode = useCallback(async (code: string, instruction: string): Promise<string> => {
        console.warn("Gemini API is disconnected. Returning mock code edit.");
        return `// Code edit for: ${instruction}\n${code}`;
    }, []);

    return {
        isAgentQOpen,
        toggleAgentQ: () => setIsAgentQOpen(!isAgentQOpen),
        generateApp,
        updateAppForChips,
        debugAndFixApp,
        editCode,
        agentQProps: {
            messages,
            isLoading,
            onSendMessage: handleSendMessage,
            lastActivity,
            isTtsEnabled,
            onToggleTts: () => setIsTtsEnabled(!isTtsEnabled),
            isSpeaking,
            memorySummary,
            onClearMemory: () => setMessages([]),
            activeContext, 
            focusedPanelId,
            activeActions,
            suggestedActions
        }
    };
};