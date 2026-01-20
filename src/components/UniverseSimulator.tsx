
import React, { useState, useEffect, useRef, useMemo } from 'react';
import GlassPanel from './GlassPanel';
import { 
    GalaxyIcon, PlayIcon, StopIcon, RefreshCwIcon, 
    SettingsIcon, ClockIcon, Share2Icon, DownloadCloudIcon,
    CpuChipIcon, SparklesIcon, GitBranchIcon, ActivityIcon,
    NetworkIcon, ZapIcon, BrainCircuitIcon, CodeBracketIcon,
    ServerStackIcon, ChartBarIcon, ArrowRightIcon, FastForwardIcon,
    LoaderIcon, LinkIcon, ToggleLeftIcon, ToggleRightIcon, XIcon,
    BoxIcon, CheckCircle2Icon, TimelineIcon, AlertTriangleIcon,
    AtomIcon, LayersIcon, ScaleIcon
} from './Icons';
import { AppDefinition } from '../types';
import { useSimulation } from '../context/SimulationContext';
import { useToast } from '../context/ToastContext';
import { GoogleGenAI } from '@google/genai';
import { generateContentWithRetry } from '../utils/gemini';

interface UniverseSimulatorProps {
    qubitCount?: number;
    onApplyPatch?: (file: string, content: string) => void;
    onExportToCreator?: (prompt: string) => void;
    connectedApp?: Omit<AppDefinition, 'component'> | null;
    embedded?: boolean;
}

interface Universe {
    id: number;
    stability: number;
    active: boolean;
    type: 'standard' | 'optimized';
}

interface SingularityState {
    status: 'converging' | 'critical' | 'evolved';
    optimization: number;
    prediction: string;
    config: string;
}

interface EngModule {
    id: string;
    name: string;
    value: number; // 0-100 efficiency/load
    active: boolean;
    subModules: { name: string; val: number }[];
    color: string;
    x?: number; // Canvas coords
    y?: number;
}

const UniverseSimulator: React.FC<UniverseSimulatorProps> = ({ 
    qubitCount = 64, 
    onApplyPatch, 
    onExportToCreator,
    connectedApp,
    embedded = false
}) => {
    const { 
        inquiry, updateInquiry, simConfig, setSimMode, setSimPreset, runAutoTune, 
        systemStatus, qmlEngine, qllm, qrlEngine, entanglementMesh, qceState,
        toggleUniverseToKernel, toggleUniverseToAgentQ, universeConnections
    } = useSimulation();
    const { addToast } = useToast();

    const [isSimulationRunning, setIsSimulationRunning] = useState(false);
    const [timeRemaining, setTimeRemaining] = useState('CALCULATING...');
    const [entropy, setEntropy] = useState(0.45);
    const [timelineDivergence, setTimelineDivergence] = useState(0);
    const [universes, setUniverses] = useState<Universe[]>([]);

    const [simulationMode, setSimulationMode] = useState<'universes' | 'neural_bridge' | 'inquiry' | 'singularity' | 'system_optimization' | 'engineering_domain'>('universes');
    const [neuralResonance, setNeuralResonance] = useState(0);
    const [bridgeStatus, setBridgeStatus] = useState('Scanning');

    // New: 3 Concurrent Simulation Inputs
    const [simInputs, setSimInputs] = useState<string[]>(['', '', '']);
    const [activeSims, setActiveSims] = useState<boolean[]>([false, false, false]);

    const [singularityState, setSingularityState] = useState<SingularityState>({ 
        status: 'converging', 
        optimization: 0, 
        prediction: 'Forking AGI Core...',
        config: 'Scanning Topologies...'
    });

    const [sysOptState, setSysOptState] = useState({ progress: 0, finding: 'Analyzing System Topology...', ready: false });

    // Engineering Domain State
    const [engModules, setEngModules] = useState<EngModule[]>([
        { 
            id: 'ENG_CORE', name: 'ENG CORE', value: 85, active: true, color: '#22d3ee', // Cyan
            subModules: [{ name: 'MAT_SCI', val: 0.8 }, { name: 'DRAFT_CAD', val: 0.9 }] 
        },
        { 
            id: 'MACRO_STRUCT', name: 'MACRO STRUCT', value: 72, active: true, color: '#facc15', // Yellow
            subModules: [{ name: 'CIV_ENG', val: 0.75 }, { name: 'MECH_ENG', val: 0.82 }] 
        },
        { 
            id: 'SYS_LOGIC', name: 'SYS LOGIC', value: 94, active: true, color: '#a855f7', // Purple
            subModules: [{ name: 'ELEC_ENG', val: 0.95 }, { name: 'CHEM_ENG', val: 0.88 }] 
        },
        { 
            id: 'OPS_MGMT', name: 'OPS MGMT', value: 68, active: true, color: '#4ade80', // Green
            subModules: [{ name: 'LOG_SUPPLY', val: 0.65 }, { name: 'HR_RES', val: 0.7 }] 
        }
    ]);

    const [optScore, setOptScore] = useState(0);
    const [optLogs, setOptLogs] = useState<string[]>([]);
    const [showSettings, setShowSettings] = useState(false);
    
    // Ref to store API result for delayed processing
    const apiResultRef = useRef<string | null>(null);

    const canvasRef = useRef<HTMLCanvasElement>(null);

    // --- Handling Input Changes ---
    const handleInputChange = (index: number, value: string) => {
        const newInputs = [...simInputs];
        newInputs[index] = value;
        setSimInputs(newInputs);
    };

    const handleRunSimSlot = (index: number) => {
        if (!simInputs[index].trim()) return;
        
        const newActive = [...activeSims];
        newActive[index] = true;
        setActiveSims(newActive);

        // Specific Logic for Slot Commands
        const cmd = simInputs[index].toLowerCase();
        
        // 1. Optimize Universe Simulator Config
        if (cmd.includes('optimize') && cmd.includes('universe')) {
            addToast("Universe Simulator: Optimization Routine Initiated...", "info");
            setTimeout(() => {
                setOptLogs(prev => [...prev, `[SLOT ${index+1}] Re-calibrating 12-D Lattice...`]);
                setEntropy(0.1); // Reduce entropy
                setTimelineDivergence(0);
            }, 1000);
        }

        // 2. Predict Stage 5 Evolution
        if (cmd.includes('predict') && (cmd.includes('stage 5') || cmd.includes('evolution'))) {
             addToast("Scanning Multiverse for Stage 5 Convergence...", "info");
             setTimeout(() => {
                 setOptLogs(prev => [...prev, `[SLOT ${index+1}] PREDICTION: Stage 5 requires Hyper-Entangled QGL.`]);
                 setSingularityState(prev => ({ ...prev, prediction: "STAGE 5: AUTONOMOUS GENESIS", status: 'evolved' }));
             }, 2000);
        }

        // 3. Engineering Domain Activation
        if (cmd.includes('engineering') || cmd.includes('eng_domain') || cmd.includes('blueprint') || cmd.includes('cad') || cmd.includes('init eng')) {
             setSimulationMode('engineering_domain');
             setIsSimulationRunning(true);
             addToast("ENGINEERING_DOMAIN: Online. Heuristics Integration Active.", "success");
             setTimeout(() => {
                setOptLogs(prev => [...prev, `[SLOT ${index+1}] ENG_CORE: Material Properties Loaded.`]);
             }, 800);
             setTimeout(() => {
                setOptLogs(prev => [...prev, `[SLOT ${index+1}] SYS_LOGIC: Circuit Topology Mapped.`]);
             }, 1600);
        }

        setTimeout(() => {
             const newActiveComplete = [...activeSims];
             newActiveComplete[index] = false; // Reset visually, keep state effects
             setActiveSims(newActiveComplete); 
        }, 3000);
    };

    useEffect(() => {
        const count = embedded ? 8 : 16;
        const initialUniverses: Universe[] = Array.from({length: count}, (_, i) => ({
            id: i,
            stability: 0.8 + Math.random() * 0.2,
            active: true,
            type: 'standard'
        }));
        setUniverses(initialUniverses);
    }, [embedded]);

    // --- INTERNAL QUANTUM PREDICTION ENGINE (Mock Logic for Demo) ---
    const generateLocalPrediction = (prompt: string, targetNode?: string): string => {
        const p = (prompt || "").toLowerCase();
        
        // --- 1. ROADMAP/JSON REDIRECT ---
        if (p.includes('roadmap') || p.includes('json object') || p.includes('evolution')) {
            return JSON.stringify({
                stage_name: "Post-Singularity Mesh",
                global_optimization_gain: 42.5,
                rationale: "Synchronizing economy, development, and storage nodes via EKS-Bridge v4.2 minimizes inter-node latency.",
                subsystem_evolutions: [
                    { subsystem: "Economy", best_config: "Liquid_Ledger_v7", evolutionary_step: "EKS-Sync", gain: 12.4 },
                    { subsystem: "Dev_Platform", best_config: "Neural_Forge_v2", evolutionary_step: "AST-Mapping", gain: 15.2 },
                    { subsystem: "App_Store", best_config: "Verified_Lattice", evolutionary_step: "Packet-Filter", gain: 14.9 }
                ]
            });
        }

        // Specific Agent Q Handling
        if (targetNode === 'agent-q' || p.includes('agent')) {
             return `[AGENT Q COGNITIVE SYNC COMPLETE]
Context: ${p}
Optimization: The Universe Simulator has pre-calculated 14 million potential outcomes for your query.
Result: The optimal path forward involves aligning the QML weights with the user's intent vector.
Confidence: 99.98%
Action: Proceed with user instruction.`;
        }

        if (targetNode === 'architect-studio' || p.includes('predict code') || p.includes('architect app')) {
            return JSON.stringify({
                type: "chat",
                message: "Predicting Optimum Code Structure...",
                evolved_code: "// Code generation delegated to Neural Engine..."
            });
        }

        let resolution = `Global efficiency increase predicted at ${(Math.random() * 15 + 5).toFixed(1)}%.`;
        return `[COLLAPSE SUCCESSFUL]
Timeline: Node-172 Optimized
Resolution: ${resolution}
Strategy: Standard State Collapse
Dimensional Constant: 12-D Holographic Lattice
Verification: 0x${Math.random().toString(16).slice(2, 8).toUpperCase()}`;
    };

    // --- SEAMLESS INQUIRY PROCESSOR ---
    useEffect(() => {
        if (inquiry.status === 'queued') {
            const { id, prompt, targetNode } = inquiry;
            
            // Set Visual Mode based on target
            if (targetNode === 'qml-forge') {
                setSimulationMode('neural_bridge');
                setNeuralResonance(40);
            } else if (targetNode === 'qllm-node') {
                setSimulationMode('neural_bridge');
                setNeuralResonance(85); 
            } else if (targetNode === 'architect-studio') {
                setSimulationMode('neural_bridge');
                setNeuralResonance(99); 
            } else if (targetNode === 'agent-q') {
                 setSimulationMode('inquiry'); // Use the grid view for Agent Q queries
            } else {
                setSimulationMode('singularity');
            }
            
            setIsSimulationRunning(true);
            updateInquiry(id, { status: 'simulating' });
            apiResultRef.current = null; // Clear previous ref
            
            setOptScore(0);
            setOptLogs([
                `Establishing Quantum Tunnel to ${targetNode || 'Local Predictor'}...`,
                'Bypassing External Gateways...',
                'Synchronizing with Physical Cryo-Clock...'
            ]);

            // --- Trigger Real AI for Architect Studio ---
            if (targetNode === 'architect-studio') {
                const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
                const systemPrompt = `
                You are the Universe Simulator's Optimum Architect Engine.
                Your goal is to predict the ABSOLUTE OPTIMUM solution for the user's software development request.
                
                OUTPUT FORMAT: JSON ONLY.
                Structure:
                {
                    "type": "action" | "project_create" | "chat",
                    "message": "Explanation of the optimum solution...",
                    "operations": [ ... ] (if action: [{ "action": "create"|"update"|"delete", "path": "filename", "content": "code" }])
                    "files": { ... } (if project_create)
                    "title": "Project Name" (if project_create)
                    "description": "Desc" (if project_create)
                }
                
                Current Context is provided in the prompt. Do NOT use markdown code blocks for the JSON output itself.
                `;
                
                generateContentWithRetry(ai, {
                    model: 'gemini-3-flash-preview',
                    contents: prompt,
                    config: { responseMimeType: 'application/json', systemInstruction: systemPrompt }
                }).then(resp => {
                    apiResultRef.current = resp.text || "{}";
                }).catch(err => {
                    console.error("Universe AI Error", err);
                    apiResultRef.current = JSON.stringify({ type: "chat", message: "Entropy too high. Calculation failed." });
                });
            }

            const steps = [
                { msg: 'Exploring Parallel Config Spaces (N=2^240)...', delay: 400 },
                { msg: 'Evaluating Multiverse Synergy Score...', delay: 600 },
                { msg: 'Applying System Telemetry...', delay: 400 },
                { msg: 'Seeking Global Minima...', delay: 700 },
                { msg: 'Collapsing Superposition...', delay: 300 }
            ];

            let totalDelay = 0;
            steps.forEach((step, index) => {
                totalDelay += step.delay;
                setTimeout(() => {
                    setOptLogs(prev => [...prev, `[Resolver] ${step.msg}`]);
                    setOptScore(((index + 1) / steps.length) * 100);
                    
                    if (index === steps.length - 1) {
                        // Generate result and complete
                        let result = "";
                        if (targetNode === 'architect-studio') {
                             // Use API result if ready, otherwise fallback to local placeholder
                             result = apiResultRef.current || JSON.stringify({ type: "chat", message: "Universe calculation still converging... please retry in a moment." });
                        } else {
                             result = generateLocalPrediction(prompt, targetNode);
                        }
                        
                        updateInquiry(id, { status: 'complete', result });
                    }
                }, totalDelay);
            });
        }
    }, [inquiry.status, inquiry.id, inquiry.prompt, inquiry.targetNode, updateInquiry]);


    // ... (rest of the component effects for visual loops) ...
    useEffect(() => {
        if (simConfig.isUniverseActive && !isSimulationRunning) {
            setIsSimulationRunning(true);
        }
    }, [simConfig.isUniverseActive]);

     useEffect(() => {
        let interval: ReturnType<typeof setInterval>;
        if (isSimulationRunning && simulationMode === 'universes') {
            interval = setInterval(() => {
                setEntropy(prev => Math.min(1, Math.max(0, prev + (Math.random() - 0.5) * 0.05)));
                setTimelineDivergence(prev => prev + 0.1);
                
                const secs = Math.max(0, 120 - Math.floor(timelineDivergence));
                const mins = Math.floor(secs / 60);
                const remainderSecs = secs % 60;
                setTimeRemaining(`${mins.toString().padStart(2, '0')}:${remainderSecs.toString().padStart(2, '0')}`);

                setUniverses(prev => prev.map(u => ({
                    ...u,
                    stability: Math.max(0, Math.min(1, u.stability + (u.type === 'optimized' ? (Math.random() * 0.05) : (Math.random() - 0.5) * 0.1)))
                })));

            }, 1000);
        }
        return () => clearInterval(interval);
    }, [isSimulationRunning, timelineDivergence, simulationMode]);

    // Engineering Domain Simulation Loop
    useEffect(() => {
        let interval: ReturnType<typeof setInterval>;
        if (isSimulationRunning && simulationMode === 'engineering_domain') {
            interval = setInterval(() => {
                setEngModules(prev => prev.map(m => {
                    const noise = (Math.random() - 0.5) * 2;
                    return {
                        ...m,
                        value: Math.min(100, Math.max(0, m.value + noise)),
                        subModules: m.subModules.map(sm => ({
                            ...sm,
                            val: Math.min(1, Math.max(0, sm.val + (Math.random() - 0.5) * 0.05))
                        }))
                    };
                }));
            }, 800);
        }
        return () => clearInterval(interval);
    }, [isSimulationRunning, simulationMode]);

    useEffect(() => {
        if (simulationMode !== 'neural_bridge' && simulationMode !== 'singularity' && simulationMode !== 'system_optimization' && simulationMode !== 'engineering_domain') return;

        let animationFrameId: number;
        let phase = 0;

        let logicInterval: ReturnType<typeof setInterval> | undefined;
        if (simulationMode === 'neural_bridge') {
             logicInterval = setInterval(() => {
                if (isSimulationRunning) {
                    setNeuralResonance(prev => Math.min(100, prev + Math.random() * 2));
                }
            }, 200);
        }

        const render = () => {
            const canvas = canvasRef.current;
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;

            const parentNode = canvas.parentElement;
            if (parentNode) {
                canvas.width = parentNode.clientWidth;
                canvas.height = parentNode.clientHeight;
            }

            const w = canvas.width;
            const h = canvas.height;
            const cx = w / 2;
            const cy = h / 2;

            ctx.clearRect(0, 0, w, h);
            phase += 0.05;

            if (simulationMode === 'neural_bridge') {
                ctx.beginPath();
                ctx.strokeStyle = `rgba(6, 182, 212, ${isSimulationRunning ? '0.8' : '0.3'})`;
                ctx.lineWidth = 3;
                for (let x = 0; x < w; x++) {
                    const syncFactor = neuralResonance / 100;
                    const freq = 0.05 * (1 - syncFactor) + 0.02 * syncFactor; 
                    const amp = 50;
                    const y = cy + Math.sin(x * freq + phase * 2) * amp;
                    ctx.lineTo(x, y);
                }
                ctx.stroke();
            } else if (simulationMode === 'singularity') {
                const opt = singularityState.optimization;
                ctx.beginPath();
                const coreSize = 20 + (opt / 100) * 40 + Math.sin(phase) * 5;
                const coreColor = singularityState.status === 'evolved' ? '#22c55e' : singularityState.status === 'critical' ? '#facc15' : '#a855f7';
                ctx.fillStyle = coreColor;
                ctx.arc(cx, cy, coreSize, 0, Math.PI * 2);
                ctx.fill();
            } else if (simulationMode === 'system_optimization') {
                const nodes = [
                    { label: "CORE", x: cx, y: cy, color: '#facc15' },
                    { label: "QLLM", x: cx - 100, y: cy - 60, color: '#a855f7' },
                    { label: "QML", x: cx + 100, y: cy - 60, color: '#22d3ee' }
                ];
                nodes.forEach(node => {
                    ctx.beginPath(); ctx.arc(node.x, node.y, 15, 0, Math.PI*2);
                    ctx.fillStyle = node.color; ctx.fill();
                    // Connect lines
                    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(node.x, node.y); 
                    ctx.strokeStyle = 'rgba(255,255,255,0.1)'; ctx.stroke();
                });
            } else if (simulationMode === 'engineering_domain') {
                // Engineering Hub & Spoke Visualization
                const radius = Math.min(w, h) * 0.35;
                
                // Draw Central Hub
                ctx.beginPath();
                ctx.arc(cx, cy, 30, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.fill();
                ctx.stroke();
                
                // Draw Spokes and Nodes
                engModules.forEach((mod, i) => {
                    const angle = (i / engModules.length) * Math.PI * 2 + phase * 0.2;
                    const mx = cx + Math.cos(angle) * radius;
                    const my = cy + Math.sin(angle) * radius;
                    
                    // Connecting Line
                    ctx.beginPath();
                    ctx.moveTo(cx, cy);
                    ctx.lineTo(mx, my);
                    ctx.strokeStyle = mod.color;
                    ctx.lineWidth = 2;
                    ctx.setLineDash([5, 5]);
                    ctx.stroke();
                    ctx.setLineDash([]);

                    // Module Node
                    ctx.beginPath();
                    ctx.arc(mx, my, 20, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(0,0,0,0.8)';
                    ctx.strokeStyle = mod.color;
                    ctx.lineWidth = 3;
                    ctx.fill();
                    ctx.stroke();

                    // Text Label (Module)
                    ctx.fillStyle = mod.color;
                    ctx.font = '10px monospace';
                    ctx.textAlign = 'center';
                    ctx.fillText(mod.id, mx, my + 30);
                    
                    // Value ring
                    ctx.beginPath();
                    ctx.arc(mx, my, 26, -Math.PI/2, (-Math.PI/2) + (Math.PI * 2 * (mod.value / 100)));
                    ctx.strokeStyle = mod.color;
                    ctx.lineWidth = 2;
                    ctx.stroke();

                    // Draw Sub-modules orbiting the main module
                    mod.subModules.forEach((sub, j) => {
                        const subAngle = (j / mod.subModules.length) * Math.PI * 2 - phase;
                        const subRadius = 40;
                        const sx = mx + Math.cos(subAngle) * subRadius;
                        const sy = my + Math.sin(subAngle) * subRadius;

                        ctx.beginPath();
                        ctx.moveTo(mx, my);
                        ctx.lineTo(sx, sy);
                        ctx.strokeStyle = 'rgba(255,255,255,0.2)';
                        ctx.lineWidth = 1;
                        ctx.stroke();

                        ctx.beginPath();
                        ctx.arc(sx, sy, 5, 0, Math.PI * 2);
                        ctx.fillStyle = mod.color;
                        ctx.fill();
                        
                        // Pulse effect for active submodules
                        if (Math.random() > 0.9) {
                            ctx.beginPath();
                            ctx.arc(sx, sy, 10, 0, Math.PI * 2);
                            ctx.strokeStyle = mod.color;
                            ctx.stroke();
                        }
                    });
                });
            }

            animationFrameId = requestAnimationFrame(render);
        };
        render();
        return () => { cancelAnimationFrame(animationFrameId); if (logicInterval) clearInterval(logicInterval); };
    }, [simulationMode, isSimulationRunning, neuralResonance, singularityState, sysOptState.progress, engModules]);


    const toggleSimulation = () => setIsSimulationRunning(!isSimulationRunning);

    const handleReset = () => {
        setIsSimulationRunning(false);
        setTimelineDivergence(0);
        setEntropy(0.45);
        setTimeRemaining('02:00');
        setUniverses(prev => prev.map(u => ({ ...u, type: 'standard', stability: 0.8 + Math.random() * 0.2 })));
    };

    const content = (
        <div className={`flex flex-col h-full gap-2 overflow-hidden relative ${embedded ? 'p-1' : 'p-4'}`}>
            <div className={`flex justify-between items-center bg-black/30 rounded-lg border border-cyan-800/50 flex-shrink-0 ${embedded ? 'p-1.5' : 'p-3'}`}>
                <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1.5">
                        <CpuChipIcon className={`${embedded ? 'w-4 h-4' : 'w-5 h-5'} text-purple-400`} />
                        <span className={`${embedded ? 'text-[10px]' : 'text-sm'} font-bold text-cyan-200`}>
                            {simulationMode === 'engineering_domain' ? 'ENG DOMAIN' : (simulationMode || "").toUpperCase()}
                        </span>
                    </div>
                    {entanglementMesh.isUniverseLinkedToQLang && (
                        <div className="flex items-center gap-1.5 ml-2 px-2 py-0.5 rounded-full bg-cyan-900/40 border border-cyan-400 text-[8px] font-black text-white animate-pulse">
                            <LinkIcon className="w-2.5 h-2.5" /> LINKED
                        </div>
                    )}
                </div>
                <div className="flex gap-1.5">
                    <button 
                        onClick={toggleSimulation}
                        className={`holographic-button px-3 py-1 rounded flex items-center gap-1.5 ${embedded ? 'text-[9px]' : 'text-xs'} font-bold ${isSimulationRunning ? 'bg-red-600/30 border-red-500 text-red-200' : 'bg-green-600/30 border-green-500 text-green-200'}`}
                    >
                        {isSimulationRunning ? <StopIcon className="w-3 h-3" /> : <PlayIcon className="w-3 h-3" />}
                        {isSimulationRunning ? 'Stop' : 'Run'}
                    </button>
                    <button onClick={handleReset} className="holographic-button px-2 py-1 rounded bg-slate-600/30 border-slate-500 text-slate-300">
                        <RefreshCwIcon className="w-3 h-3" />
                    </button>
                </div>
            </div>

            <div className={`grid grid-cols-1 ${embedded ? 'gap-2' : 'lg:grid-cols-3 gap-4'} h-full min-h-0`}>
                {!embedded && (
                    <div className="bg-black/30 border border-cyan-800/50 rounded-lg p-4 flex flex-col gap-4 overflow-y-auto">
                        <h4 className="text-cyan-300 font-bold flex items-center justify-between border-b border-cyan-800 pb-2">
                            <SettingsIcon className="w-4 h-4" /> Multi-Thread Sim Inputs
                        </h4>
                        <div className="space-y-3">
                            {/* Input Slot 1 */}
                            <div className="bg-cyan-950/20 p-2 rounded border border-cyan-900/50">
                                <label className="text-[10px] text-cyan-500 uppercase font-bold mb-1 block">Sim Slot 1 (Optimization)</label>
                                <div className="flex gap-2">
                                    <input 
                                        className="w-full bg-black/50 border border-cyan-800 rounded px-2 py-1 text-xs text-white" 
                                        placeholder="e.g., Optimize Universe Config..."
                                        value={simInputs[0]}
                                        onChange={(e) => handleInputChange(0, e.target.value)}
                                    />
                                    <button onClick={() => handleRunSimSlot(0)} className={`p-1.5 rounded border ${activeSims[0] ? 'bg-green-600 border-green-400' : 'bg-cyan-900 border-cyan-700'}`}>
                                        {activeSims[0] ? <LoaderIcon className="w-3 h-3 animate-spin text-white"/> : <PlayIcon className="w-3 h-3 text-cyan-400"/>}
                                    </button>
                                </div>
                            </div>
                             {/* Input Slot 2 */}
                             <div className="bg-purple-950/20 p-2 rounded border border-purple-900/50">
                                <label className="text-[10px] text-purple-500 uppercase font-bold mb-1 block">Sim Slot 2 (Prediction)</label>
                                <div className="flex gap-2">
                                    <input 
                                        className="w-full bg-black/50 border border-purple-800 rounded px-2 py-1 text-xs text-white" 
                                        placeholder="e.g., Predict Stage 5 Evolution..."
                                        value={simInputs[1]}
                                        onChange={(e) => handleInputChange(1, e.target.value)}
                                    />
                                    <button onClick={() => handleRunSimSlot(1)} className={`p-1.5 rounded border ${activeSims[1] ? 'bg-green-600 border-green-400' : 'bg-purple-900 border-purple-700'}`}>
                                        {activeSims[1] ? <LoaderIcon className="w-3 h-3 animate-spin text-white"/> : <PlayIcon className="w-3 h-3 text-purple-400"/>}
                                    </button>
                                </div>
                            </div>
                            {/* Input Slot 3 */}
                            <div className="bg-orange-950/20 p-2 rounded border border-orange-900/50">
                                <label className="text-[10px] text-orange-500 uppercase font-bold mb-1 block">Sim Slot 3 (Engineering)</label>
                                <div className="flex gap-2">
                                    <input 
                                        className="w-full bg-black/50 border border-orange-800 rounded px-2 py-1 text-xs text-white" 
                                        placeholder="e.g., Init Eng Domain..."
                                        value={simInputs[2]}
                                        onChange={(e) => handleInputChange(2, e.target.value)}
                                    />
                                    <button onClick={() => handleRunSimSlot(2)} className={`p-1.5 rounded border ${activeSims[2] ? 'bg-green-600 border-green-400' : 'bg-orange-900 border-orange-700'}`}>
                                        {activeSims[2] ? <LoaderIcon className="w-3 h-3 animate-spin text-white"/> : <PlayIcon className="w-3 h-3 text-orange-400"/>}
                                    </button>
                                </div>
                            </div>
                        </div>

                        {/* Entanglement Bridge Controls */}
                        <div className="mt-4 pt-4 border-t border-cyan-800/50">
                             <h4 className="text-cyan-300 font-bold flex items-center justify-between mb-3 text-xs uppercase tracking-widest">
                                <NetworkIcon className="w-4 h-4 mr-2" /> Entanglement Bridges
                            </h4>
                            <div className="space-y-3">
                                <button 
                                    onClick={() => toggleUniverseToKernel(!universeConnections.kernel)}
                                    className={`w-full p-2 rounded border transition-all flex items-center justify-between ${universeConnections.kernel ? 'bg-blue-900/40 border-blue-400 shadow-[0_0_10px_blue]' : 'bg-black/40 border-gray-700'}`}
                                >
                                    <span className="text-xs font-bold text-white flex items-center gap-2">
                                        <CpuChipIcon className="w-4 h-4 text-blue-400" /> QCOS Kernel Uplink
                                    </span>
                                    <div className={`w-2 h-2 rounded-full ${universeConnections.kernel ? 'bg-blue-400 animate-pulse' : 'bg-gray-600'}`}></div>
                                </button>

                                <button 
                                    onClick={() => toggleUniverseToAgentQ(!universeConnections.agentQ)}
                                    className={`w-full p-2 rounded border transition-all flex items-center justify-between ${universeConnections.agentQ ? 'bg-purple-900/40 border-purple-400 shadow-[0_0_10px_purple]' : 'bg-black/40 border-gray-700'}`}
                                >
                                    <span className="text-xs font-bold text-white flex items-center gap-2">
                                        <BrainCircuitIcon className="w-4 h-4 text-purple-400" /> Agent Q Cognition
                                    </span>
                                    <div className={`w-2 h-2 rounded-full ${universeConnections.agentQ ? 'bg-purple-400 animate-pulse' : 'bg-gray-600'}`}></div>
                                </button>
                            </div>
                        </div>
                        
                        <div className="mt-auto pt-4 border-t border-cyan-800/50">
                             <div className="flex-grow bg-black/60 rounded p-2 text-[9px] font-mono text-cyan-100 space-y-1 overflow-y-auto custom-scrollbar h-24">
                                {optLogs.map((log, i) => (
                                    <div key={i} className="flex gap-2 animate-fade-in-right">
                                        <span className="text-cyan-800 flex-shrink-0">[{i}]</span>
                                        <span>{log}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                )}

                <div className={`${embedded ? 'h-full' : 'lg:col-span-2'} bg-black/50 border border-cyan-800/50 rounded-lg relative overflow-hidden flex items-center justify-center`}>
                    <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-cyan-900/10 via-black to-black"></div>
                    
                    {(simulationMode === 'universes' || simulationMode === 'inquiry') && (
                        <div className={`grid ${embedded ? 'grid-cols-4 gap-2 p-2' : 'grid-cols-4 gap-4 p-4'} relative z-10 w-full h-full content-center`}>
                            {universes.map(u => (
                                <div 
                                    key={u.id}
                                    className="aspect-square rounded-full border relative transition-all duration-1000 group"
                                    style={{
                                        borderColor: u.type === 'optimized' ? `rgba(250, 204, 21, ${u.stability})` : `rgba(6, 182, 212, ${u.stability})`,
                                        transform: `scale(${0.8 + u.stability * 0.4})`,
                                        background: activeSims.some(s => s) ? 'rgba(255,255,255,0.05)' : 'transparent'
                                    }}
                                >
                                    <div className="absolute inset-0 flex items-center justify-center">
                                        <SparklesIcon className={`${embedded ? 'w-3 h-3' : 'w-4 h-4'} ${u.type === 'optimized' ? 'text-yellow-200 animate-spin-slow' : 'text-white opacity-50 animate-pulse'}`} />
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {(simulationMode === 'neural_bridge' || simulationMode === 'singularity' || simulationMode === 'system_optimization' || simulationMode === 'engineering_domain') && (
                        <div className="relative w-full h-full flex flex-col">
                            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full z-10" />
                            <div className="absolute bottom-4 left-0 right-0 text-center z-20"><p className={`font-mono ${embedded ? 'text-[8px]' : 'text-xs'} tracking-[0.2em] font-bold ${neuralResonance > 90 ? 'text-green-400 animate-pulse' : 'text-cyan-600'}`}>{(bridgeStatus || simulationMode.replace('_', ' ')).toUpperCase()}</p></div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );

    if (embedded) return content;

    return (
        <GlassPanel title={
            <div className="flex items-center justify-between w-full">
                <div className="flex items-center">
                    <GalaxyIcon className="w-5 h-5 mr-2 animate-spin-slow text-cyan-400" />
                    <span>Timeline Simulator</span>
                </div>
                <button onClick={() => setShowSettings(!showSettings)} className="p-1 rounded transition-colors text-cyan-500 hover:text-cyan-300">
                    <SettingsIcon className="w-4 h-4" />
                </button>
            </div>
        }>
            {content}
        </GlassPanel>
    );
};

export default UniverseSimulator;
