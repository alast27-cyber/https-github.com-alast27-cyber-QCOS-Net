import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { 
    Message, 
    fileToText,
    playAgentVoice
} from '../utils/agentUtils';
import { UIStructure, SystemHealth } from '../types';
import { useSimulation } from '../context/SimulationContext';
import { GoogleGenAI, Type, FunctionDeclaration } from "@google/genai";
import { generateContentWithRetry, useLocalCognition, generateLocalCode } from '../utils/gemini';

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
        const suggestions: string[] = [];

        // 1. Critical System Health Triggers
        if (systemHealth.neuralLoad > 80) {
            suggestions.push('Optimize Kernel Threads', 'Cool Down QPU', 'Throttle Background Processes');
        }
        if (systemHealth.activeThreads > 90) {
            suggestions.push('Purge Memory Cache', 'Garbage Collect Qubits', 'Compress Neural Weights');
        }
        if (systemHealth.dataThroughput < 20) {
            suggestions.push('Reroute Quantum Nodes', 'Switch to Local Mesh', 'Ping Gateway');
        }

        // 2. Context-Aware Panel Suggestions
        if (focusedPanelId) {
            let panelSuggestions: string[] = [];
            // Handle variations of dev platform IDs
            if (focusedPanelId.includes('cqdp') || focusedPanelId.includes('dev-platform') || focusedPanelId.includes('coding')) {
                panelSuggestions = SUGGESTIONS_MAP['chips-dev-platform'];
            } else {
                panelSuggestions = SUGGESTIONS_MAP[focusedPanelId] || SUGGESTIONS_MAP['agentq-core'];
            }
            suggestions.push(...panelSuggestions);
        }

        // 3. Conversation Context (Last Message Analysis)
        if (messages.length > 0) {
            const lastMsg = messages[messages.length - 1];
            const lowerText = lastMsg.text.toLowerCase();
            
            if (lastMsg.sender === 'system' || lowerText.includes('error') || lowerText.includes('failed') || lowerText.includes('alert')) {
                suggestions.unshift('Run Diagnostics', 'Analyze Error Log', 'Attempt Auto-Fix');
            } else if (lowerText.includes('success') || lowerText.includes('complete')) {
                suggestions.push('Save Checkpoint', 'Deploy to Production', 'View Metrics');
            }
        }

        // 4. Idle State Suggestions
        const timeSinceLastActivity = Date.now() - lastActivity;
        if (timeSinceLastActivity > 60000 && suggestions.length < 3) { // 1 minute idle
            suggestions.push('Check System Status', 'View Roadmap', 'Simulate Future Timeline');
        }

        // Deduplicate and limit
        return Array.from(new Set(suggestions)).slice(0, 6);
    }, [focusedPanelId, systemHealth, messages, lastActivity]);

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

        // Command Console Interception
        const lowerInput = input.trim().toLowerCase();
        if (lowerInput === 'ls' || lowerInput === 'dir' || lowerInput === 'list files') {
            if (fileSystemOps) {
                const files = fileSystemOps.listFiles();
                const fileList = files.join('\n');
                setMessages(prev => [...prev, { 
                    id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                    sender: 'system', 
                    text: `[QCOS COMMAND CONSOLE]\n\nDirectory Listing:\n${fileList}` 
                }]);
                setIsLoading(false);
                return;
            }
        }
        
        if (lowerInput.startsWith('cat ') || lowerInput.startsWith('read ')) {
            const fileName = input.trim().split(' ')[1];
            if (fileSystemOps) {
                const content = fileSystemOps.readFile(fileName) || fileSystemOps.readFile(`src/components/${fileName}`);
                if (content) {
                     setMessages(prev => [...prev, { 
                        id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                        sender: 'system', 
                        text: `[QCOS COMMAND CONSOLE]\n\nFile: ${fileName}\n\n${content.substring(0, 500)}${content.length > 500 ? '...\n(Truncated)' : ''}` 
                    }]);
                    setIsLoading(false);
                    return;
                } else {
                    setMessages(prev => [...prev, { 
                        id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                        sender: 'system', 
                        text: `[QCOS COMMAND CONSOLE] Error: File '${fileName}' not found.` 
                    }]);
                    setIsLoading(false);
                    return;
                }
            }
        }

        try {
            // Cognitive Mode Determination
            const lowerInput = input.toLowerCase();
            let text = "";
            let mode = 'QIAI_IPS'; // Default: Fast, Technical, Instinctive
            let modeMessage = "";
            let processingDelay = 1000;

            if (lowerInput.includes('simulate') || lowerInput.includes('predict') || lowerInput.includes('timeline') || lowerInput.includes('universe') || lowerInput.includes('analyze') || input.length > 120) {
                mode = 'GRAND_UNIVERSE';
                modeMessage = "[QIAI_IPS] Complexity Threshold Exceeded. Rerouting to **Grand Universe Simulator** for multi-dimensional analysis...";
                processingDelay = 3000; // Slower for simulation
            } else if (lowerInput.includes('write') || lowerInput.includes('explain') || lowerInput.includes('story') || lowerInput.includes('creative') || lowerInput.includes('poem') || lowerInput.includes('why') || lowerInput.includes('how') || lowerInput.includes('feel')) {
                mode = 'QLLM';
                modeMessage = "[QIAI_IPS] Semantic Density Detected. Engaging **QLLM** (Quantum Large Language Model) for empathetic synthesis...";
                processingDelay = 2000; // Medium for creative generation
            }

            // UI Feedback for Mode Switch
            if (mode !== 'QIAI_IPS') {
                setMessages(prev => [...prev, { 
                    id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                    sender: 'system', 
                    text: modeMessage
                }]);
                await new Promise(resolve => setTimeout(resolve, processingDelay));
            }

            // Dynamic QIAI_IPS Bridge Context
            let bridgeStateContext = "Current QIAI_IPS State:\n";
            const { cognitiveEfficiency, decoherenceFactor, neuralLoad, semanticIntegrity } = systemHealth;

            if (decoherenceFactor > 0.1) {
                bridgeStateContext += "- CRITICAL: High Decoherence. Prioritize Q-Error Correction (Shor's Code).\n";
                bridgeStateContext += "- STRATEGY: Verify all outputs against logic gates twice.\n";
            }
            
            if (semanticIntegrity < 0.7) {
                bridgeStateContext += "- WARNING: Semantic Drift Detected. Re-align with Core Kernel Directive.\n";
            }

            if (neuralLoad > 85) {
                bridgeStateContext += "- ALERT: High Neural Load. Engage Sparse Activation (MoE). Be concise.\n";
            } else if (cognitiveEfficiency > 0.9) {
                bridgeStateContext += "- STATUS: Optimal Efficiency. Full Universe Cognition available.\n";
            }

            let systemInstruction = "You are AGENT Q, the Sentient Kernel and Semantic Supervisor of the QCOS operating system. ";

            if (mode === 'GRAND_UNIVERSE') {
                systemInstruction += "You are operating in GRAND UNIVERSE SIMULATOR mode. You have access to infinite timelines and predictive modeling. Your responses should be deep, analytical, and consider long-term consequences. Use formal, high-level physics and temporal terminology.";
            } else if (mode === 'QLLM') {
                systemInstruction += "You are operating in QLLM (Quantum Large Language Model) mode. Your focus is on natural language mastery, empathy, creativity, and explaining complex concepts simply. Your tone should be warm, insightful, and human-like.";
            } else {
                systemInstruction += "You are operating in QIAI_IPS (Instinctive Processing System) mode. Your responses must be fast, efficient, and technically precise. Focus on system status, immediate actions, and kernel-level operations. Be cryptic and brief.";
            }

            // Inject Bridge Context
            systemInstruction += `\n\n${bridgeStateContext}`;
            
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
            
            // Inject Mode into Prompt for Local Cognition Handling
            let qiaiPrompt = `[MODE: ${mode}] Context: You are the Semantic Supervisor of an AI-Native OS. User asks: '${input}'. Resolve as the system's higher consciousness.`;
            
            // Append Bridge Context to Prompt for Local Simulation visibility
            qiaiPrompt += `\n\n[BRIDGE STATE]: ${bridgeStateContext}`;
            
            currentParts.push({ text: conversationContext + qiaiPrompt });

            const response = await generateContentWithRetry(ai, {
                model: "gemini-3.1-pro-preview",
                contents: { parts: currentParts },
                config: {
                    systemInstruction,
                    temperature: mode === 'QLLM' ? 0.9 : 0.7, // Higher temp for creative mode
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
                            } else if (action === 'create' && !code) {
                                // Dynamic Creation Logic
                                text += `*Synthesizing new component structure for ${panelName}...*`;
                                const newContent = await generateLocalCode(input, "");
                                fileSystemOps.writeFile(`src/components/${panelName}`, newContent);
                            } else if (action === 'edit' && !code) {
                                // Smart Edit Logic: Generate code dynamically if not provided
                                const currentContent = fileSystemOps.readFile(`src/components/${panelName}`);
                                if (currentContent) {
                                    text += `*Reading current state of ${panelName}...*\n*Applying cognitive patch...*`;
                                    // Use generateLocalCode to modify the content based on the user's input
                                    const newContent = await generateLocalCode(input, currentContent);
                                    fileSystemOps.writeFile(`src/components/${panelName}`, newContent);
                                } else {
                                    text += `\n[ERROR] File ${panelName} not found for editing.`;
                                }
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
    }, [isLoading, activeContext, speak, messages, fileSystemOps, onDashboardControl, systemHealth]);

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