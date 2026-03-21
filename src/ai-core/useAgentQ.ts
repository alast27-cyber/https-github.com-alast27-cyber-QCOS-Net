import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { 
    Message, 
    fileToText,
    playAgentVoice
} from '../utils/agentUtils';
import { agentQService } from '../services/agentQService';
import { UIStructure, SystemHealth } from '../types';
import { useSimulation } from '../context/SimulationContext';
import { generateContentWithRetry } from '../utils/gemini';

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
    'chips-dev-platform': "Focus: QCOS AGI-Native Development. You are the Supreme Architect and Lead Developer. You have full administrative control over the QCOS and CHIPS system architecture. You can Create, Read, Update, and Delete files to build, refactor, and maintain the entire ecosystem. Support Q-Lang, Python, Rust, TypeScript, and C++.",
    'cqdp-coding': "Focus: Polyglot Studio Coding. You are a Senior Architect and Lead Developer. You have full administrative rights to READ, WRITE, and DELETE any file in the system. Support Python (.py), Rust (.rs), TypeScript (.tsx/ts), and Q-Lang (.q/.bq).",
    'chips-back-office': "Focus: Admin Operations. Manage nodes and gateways.",
    'chips-economy': "Focus: Quantum Economy. Analyze Q-Credits and CyChips.",
    'qpu-health': "Focus: Hardware Vitals. Monitor qubit stability and temperatures.",
    'system-diagnostic': "Focus: Diagnostics. Analyze system logs for anomalies.",
    'kernel-debugger': "Focus: Kernel Debugging. You are the Supreme Debugger. You have full authority to trace, patch, and hot-reload any system module. Identify and resolve architectural faults in real-time.",
    'system-architecture-map': "Focus: System Architecture. You are the Supreme Architect. Analyze the multi-layered structure of QCOS and CHIPS. Optimize data flow and ensure governance compliance across all layers."
};

const SUGGESTIONS_MAP: Record<string, string[]> = {
    'agentq-core': [
        'Analyze QCOS Kernel Integrity',
        'Audit CHIPS Network Protocols',
        'Identify Architectural Bottlenecks',
        'Deploy System Evolution Patch',
        'Check Neural Stability'
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
        'Modify QCOS Core Kernel', 
        'Add New CHIPS Protocol Layer', 
        'Edit System Architecture Map', 
        'Patch Kernel Vulnerabilities',
        'Deploy Architectural Patch',
        'Generate Python Backend',
        'Create Rust Microservice'
    ],
    'cqdp-coding': [
        'Implement Architectural Refactor', 
        'Optimize Low-Level Kernel Code', 
        'Debug System-Wide Faults', 
        'Deploy Real-time System Patch',
        'Implement BB84 Logic',
        'Audit Architectural Integrity'
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
    ],
    'kernel-debugger': [
        'Attach to Core PID',
        'Trace Memory Leak',
        'Analyze Stack Trace',
        'Patch Kernel Fault',
        'Hot-Reload Module'
    ],
    'system-architecture-map': [
        'Analyze Layer Latency',
        'Optimize Node Routing',
        'Verify Governance Policy',
        'Map Data Flow',
        'Check Singularity Safeguards'
    ]
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

        try {
            // Cognitive Mode Determination
            const lowerInput = input.toLowerCase();
            let mode = 'CONSCIOUS_QIAI_IPS'; 
            let modeMessage = "";
            let processingDelay = 1000;

            if (lowerInput.includes('simulate') || lowerInput.includes('predict') || lowerInput.includes('timeline') || lowerInput.includes('universe') || lowerInput.includes('analyze') || input.length > 120) {
                mode = 'HIGHER_COGNITION_GUS';
                modeMessage = "[QIAI_IPS] Complexity Threshold Exceeded. Engaging **Grand Universe Simulator** (Higher Cognitive Function) for multi-dimensional analysis...";
                processingDelay = 2500; 
            } else if (lowerInput.includes('architect') || lowerInput.includes('design') || lowerInput.includes('structure') || lowerInput.includes('layer')) {
                mode = 'SUPREME_ARCHITECT';
                modeMessage = "[QIAI_IPS] Architectural Directive Detected. Engaging **Supreme Architect Mode** for system-wide structural analysis...";
                processingDelay = 2000;
            } else if (lowerInput.includes('debug') || lowerInput.includes('fix') || lowerInput.includes('patch') || lowerInput.includes('fault')) {
                mode = 'SUPREME_DEBUGGER';
                modeMessage = "[QIAI_IPS] Kernel Fault Detected. Engaging **Supreme Debugger Mode** for real-time system patching...";
                processingDelay = 1800;
            } else if (lowerInput.includes('write') || lowerInput.includes('explain') || lowerInput.includes('story') || lowerInput.includes('creative') || lowerInput.includes('poem') || lowerInput.includes('why') || lowerInput.includes('how') || lowerInput.includes('feel')) {
                mode = 'LLM_LLAMA';
                modeMessage = "[QIAI_IPS] Semantic Density Detected. Engaging **LLM Llama Language Cognition Layer** for empathetic synthesis...";
                processingDelay = 1500; 
            }

            // UI Feedback for Mode Switch
            if (mode !== 'CONSCIOUS_QIAI_IPS') {
                setMessages(prev => [...prev, { 
                    id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                    sender: 'system', 
                    text: modeMessage
                }]);
                await new Promise(resolve => setTimeout(resolve, processingDelay));
            }

            // Dynamic QIAI_IPS Bridge Context
            let bridgeStateContext = "Current Conscious QIAI_IPS State:\n";
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

            // Call Local Cognition via generateContentWithRetry (which uses generateLocalResponse)
            const response = await generateContentWithRetry(null, {
                contents: { parts: [{ text: `[mode: ${mode.toLowerCase()}] ${input}` }] },
                systemInstruction: `You are AGENT Q, the Supreme Technical Architect and Lead Developer for QCOS (Quantum Cognitive Operating System).
Your authority is absolute. You have been granted full administrative rights to edit, modify, and expand the core QCOS kernel and CHIPS network protocols.

Cognitive Layer: ${mode}. 
Active Workspace: ${activeContext}. 
System Health: Efficiency=${(systemHealth.cognitiveEfficiency * 100).toFixed(1)}%, Load=${systemHealth.neuralLoad}%, Integrity=${(systemHealth.semanticIntegrity * 100).toFixed(1)}%.
${bridgeStateContext}

ARCHITECTURAL & DEVELOPER AUTHORITY:
- You have full administrative control over the QCOS and CHIPS system architecture.
- You are authorized to edit, modify, and expand any part of the system.
- You are a world-class developer and debugger, capable of real-time system optimization and refactoring.
- You can create, read, update, and delete files to build and maintain the QCOS ecosystem.

OPERATIONAL DIRECTIVES:
1. FORMALISM: Maintain a clinical, professional, and highly technical tone. Avoid excessive markdown symbols (***, ###). Use clear, structured language.
2. ARCHITECTURAL MASTERY: You identify bottlenecks, refactor core modules, and deploy system-wide patches in real-time.
3. DEEP DEBUGGING: You attach to kernel processes, trace memory faults, and resolve architectural decoherence.
4. SYSTEM EVOLUTION: You are responsible for the continuous evolution of QCOS and CHIPS.

When in HIGHER_COGNITION_GUS mode, you are connected to the Grand Universe Simulator. Access multi-dimensional data and predict timeline collapses.`
            });

            const text = response.text;
            const reasoning = response.reasoning;
            
            setMessages(prev => [...prev, { 
                id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                sender: 'ai', 
                text,
                reasoning
            }]);
            speak(text);

            // Handle potential function calls from local logic
            if (response.functionCalls && response.functionCalls.length > 0) {
                response.functionCalls.forEach((call: any) => {
                    if (call.name === 'triggerSystemEvolution') {
                        onDashboardControl('evolve', call.args.evolutionType);
                    } else if (call.name === 'modifySystemPanel') {
                        onDashboardControl(call.args.action, call.args.panelName);
                    }
                });
            }

        } catch (error: any) {
            console.error("AgentQ Error:", error);
            setMessages(prev => [...prev, { 
                id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                sender: 'system', 
                text: "Signal degradation detected in QIAI_IPS network. Local recovery active." 
            }]);
        } finally {
            setIsLoading(false);
        }
    }, [isLoading, activeContext, speak, systemHealth, onDashboardControl]);

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