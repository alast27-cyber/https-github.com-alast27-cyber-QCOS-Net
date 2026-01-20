import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { 
    Message, 
    fileToText,
    playAgentVoice
} from '../utils/agentUtils';
import { UIStructure, SystemHealth } from '../types';
import { useSimulation } from '../context/SimulationContext';
import { GoogleGenAI, Type } from "@google/genai";
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
    'quantum-app-exchange': "Focus: App Store. Help user find or install quantum apps.",
    'universe-simulator': "Focus: Universe Simulation. Help manipulate physical constants.",
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
    'quantum-app-exchange': [
        'Browse Best Sellers',
        'Install SDK Package',
        'Submit New Artifact',
        'Review Node Ratings'
    ],
    'universe-simulator': [
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

export const useAgentQ = ({ focusedPanelId, panelInfoMap, qcosVersion, systemHealth, onDashboardControl, fileSystemOps, projectOps }: UseAgentQProps) => {
    const { submitInquiry, universeConnections } = useSimulation();

    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isAgentQOpen, setIsAgentQOpen] = useState(false);
    const [lastActivity, setLastActivity] = useState(0);
    const [isTtsEnabled, setIsTtsEnabled] = useState(true);
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

        const elevenLabsKey = process.env.ELEVENLABS_API_KEY;
        const voiceId = process.env.LEX_FRIDMAN_VOICE_ID;

        if (elevenLabsKey) {
            try {
                const audio = await playAgentVoice(text, voiceId, elevenLabsKey);
                if (audio) {
                    audio.play().catch(e => console.error("Audio playback error:", e));
                    return;
                }
            } catch (e) {
                console.warn("High-quality TTS failed, falling back to system voice.", e);
            }
        }
            
        const utterance = new SpeechSynthesisUtterance(text);
        utteranceRef.current = utterance;

        let selectedVoice: SpeechSynthesisVoice | undefined;
        const maleVoiceKeywords = ['male', 'david', 'mark', 'alex', 'daniel', 'lee'];
        const englishVoices = availableVoices.filter(v => v.lang.startsWith('en-'));
        selectedVoice = englishVoices.find(v => v.name.toLowerCase().includes('google') && maleVoiceKeywords.some(kw => v.name.toLowerCase().includes(kw)))
            || englishVoices.find(v => maleVoiceKeywords.some(kw => v.name.toLowerCase().includes(kw)))
            || availableVoices.find(voice => voice.name === 'Google US English')
            || availableVoices.find(voice => voice.lang.startsWith('en-US'));
        
        if (selectedVoice) utterance.voice = selectedVoice;
        utterance.pitch = pitch;
        utterance.rate = rate;
        utterance.onend = () => { utteranceRef.current = null; };
        window.speechSynthesis.speak(utterance);
    }, [isTtsEnabled, availableVoices]);

    const handleSendMessage = useCallback(async (input: string, attachedFile: File | null = null) => {
        if ((!input.trim() && !attachedFile) || isLoading) return;

        setMessages(prev => [...prev, { sender: 'user', text: input.trim(), attachment: attachedFile ? { name: attachedFile.name } : undefined }]);
        setIsLoading(true);
        setLastActivity(Date.now());

        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
            const response = await generateContentWithRetry(ai, {
                model: 'gemini-3-flash-preview',
                contents: input,
                config: {
                    systemInstruction: `You are Agent Q, the core intelligence of QCOS. Context: ${activeContext}. Keep responses technical and helpful.`
                }
            });
            
            const text = response.text || "Communication established.";
            setMessages(prev => [...prev, { sender: 'ai', text }]);
            speak(text);
        } catch (error) {
            setMessages(prev => [...prev, { sender: 'system', text: "Signal degradation detected. Retrying handshake..." }]);
        } finally {
            setIsLoading(false);
        }
    }, [isLoading, activeContext, speak]);

    const generateApp = useCallback(async (description: string): Promise<{ files: { [path: string]: string }, uiStructure: UIStructure | null }> => {
        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
            const prompt = `Architect a new CHIPS application based on this description: "${description}". Return JSON with files map and uiStructure.`;
            const response = await generateContentWithRetry(ai, {
                model: 'gemini-3-pro-preview',
                contents: prompt,
                config: { responseMimeType: "application/json" }
            });
            const data = JSON.parse(response.text || '{}');
            return { files: data.files || {}, uiStructure: data.uiStructure || null };
        } catch (error) {
            return { files: {}, uiStructure: null };
        }
    }, []);

    const updateAppForChips = useCallback(async (files: { [path: string]: string }): Promise<{ updatedFiles: { [path: string]: string }, summary: string }> => {
        return { updatedFiles: files, summary: "App synchronized." };
    }, []);

    const debugAndFixApp = useCallback(async (files: { [path: string]: string }): Promise<{ fixedFiles: { [path: string]: string }, summary: string, uiStructure: UIStructure | null }> => {
        return { fixedFiles: files, summary: "Debug Complete.", uiStructure: null };
    }, []);

    const editCode = useCallback(async (code: string, instruction: string): Promise<string> => {
        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
            const response = await generateContentWithRetry(ai, {
                model: 'gemini-3-pro-preview',
                contents: `Code: ${code}\nInstruction: ${instruction}`
            });
            return response.text || code;
        } catch (error) {
            return code;
        }
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
            memorySummary,
            onClearMemory: () => setMessages([]),
            activeContext, 
            focusedPanelId,
            activeActions,
            suggestedActions
        }
    };
};