
import React, { useState, useEffect, useRef, useCallback } from 'react';
import GlassPanel from './GlassPanel';
import { 
    SparklesIcon, BrainCircuitIcon, DatabaseIcon, 
    RocketLaunchIcon, ActivityIcon,
    CodeBracketIcon, AtomIcon, GalaxyIcon, StopIcon,
    PlayIcon, UploadCloudIcon, SaveIcon,
    ServerStackIcon, CheckCircle2Icon
} from './Icons';
import MemoryMatrix from './MemoryMatrix';
import { useAgentQ } from '../hooks/useAgentQ';
import { useSimulation } from '../context/SimulationContext';
import { useToast } from '../context/ToastContext';

// --- Types ---
interface CognitiveEngine {
    id: 'QLLM' | 'QML' | 'QRL' | 'QGL' | 'QDL';
    label: string;
    load: number;
    active: boolean;
    color: string;
    icon: React.FC<{className?: string}>;
}

interface UniverseSim {
    id: number;
    targetName: string;
    status: 'INITIALIZING' | 'OPTIMIZING' | 'CONVERGING' | 'UPGRADING' | 'SAVING';
    optimizationScore: number;
    currentVersion: string;
    nextVersion: string;
    predictionLog: string;
    engines: CognitiveEngine[];
    stability: number;
    upgradesApplied: number;
}

// Factory function to create fresh engine instances with Icon references intact
const getInitialEngines = (): CognitiveEngine[] => [
    { id: 'QLLM', label: 'Semantic Core', load: 0, active: false, color: 'text-purple-400', icon: CodeBracketIcon },
    { id: 'QML', label: 'QCA Memory', load: 0, active: false, color: 'text-cyan-400', icon: BrainCircuitIcon },
    { id: 'QRL', label: 'Strategy Loop', load: 0, active: false, color: 'text-orange-400', icon: RocketLaunchIcon },
    { id: 'QGL', label: 'Generative UI', load: 0, active: false, color: 'text-pink-400', icon: SparklesIcon },
    { id: 'QDL', label: 'Deep Logic', load: 0, active: false, color: 'text-blue-400', icon: AtomIcon },
];

const INITIAL_UNIVERSES: UniverseSim[] = [
    {
        id: 1,
        targetName: "Neural Programming Core",
        status: 'CONVERGING', 
        optimizationScore: 88,
        currentVersion: 'v4.5.0',
        nextVersion: 'v4.6.0', 
        predictionLog: "Optimizing synaptic BCI weights...",
        engines: getInitialEngines(),
        stability: 99.4,
        upgradesApplied: 12
    },
    {
        id: 2,
        targetName: "Quantum Engineering Matrix",
        status: 'OPTIMIZING',
        optimizationScore: 45,
        currentVersion: 'v2.4.0', 
        nextVersion: 'v2.5.0', 
        predictionLog: "Refactoring QDL stack hierarchy...",
        engines: getInitialEngines(),
        stability: 99.2,
        upgradesApplied: 8
    },
    {
        id: 3,
        targetName: "Material Science Foundry",
        status: 'INITIALIZING',
        optimizationScore: 12,
        currentVersion: 'v3.8.0',
        nextVersion: 'v3.9.0', 
        predictionLog: "Converging VQE molecular states...",
        engines: getInitialEngines(),
        stability: 98.9,
        upgradesApplied: 6
    }
];

interface QuantumCognitiveArchitectureProps {
    onApplyPatch?: (file: string, content: string) => void;
}

const QuantumCognitiveArchitecture: React.FC<QuantumCognitiveArchitectureProps> = ({ onApplyPatch }) => {
    const { agentQProps } = useAgentQ({
        focusedPanelId: null,
        panelInfoMap: {},
        qcosVersion: 4.5,
        systemHealth: {} as any,
        onDashboardControl: () => {}
    });
    const { addToast } = useToast();
    const { messages, lastActivity } = agentQProps;
    const { systemStatus, updateQIAIIPS, qiaiIps } = useSimulation();

    // --- State ---
    const [universes, setUniverses] = useState<UniverseSim[]>(INITIAL_UNIVERSES);
    const [isSimulating, setIsSimulating] = useState(false);
    const [consolidationStatus, setConsolidationStatus] = useState<'Idle' | 'Consolidating' | 'Complete'>('Idle');
    const [globalCoherence, setGlobalCoherence] = useState(99.9);

    // --- Helper to trigger upgrades to the Context ---
    const commitUpgradeToCore = useCallback((universeName: string, version: string) => {
        // 1. Boost Global IPS Stats based on specific vector
        let boostType = "General";
        
        if (universeName.includes("Neural")) {
            boostType = "Synaptic Efficiency";
            updateQIAIIPS({ qcl: { ...qiaiIps.qcl, coherence: Math.min(1.0, qiaiIps.qcl.coherence + 0.005), load: 45 } });
        } else if (universeName.includes("Engineering")) {
            boostType = "Circuit Depth";
            updateQIAIIPS({ qips: { ...qiaiIps.qips, coherence: Math.min(1.0, qiaiIps.qips.coherence + 0.005) } });
        } else if (universeName.includes("Material")) {
            boostType = "Molecular Simulation Speed";
            updateQIAIIPS({ qil: { ...qiaiIps.qil, coherence: Math.min(1.0, qiaiIps.qil.coherence + 0.005) } });
        }

        // 2. Notify User
        addToast(`Auto-Evolve: ${universeName} upgraded to ${version}`, "success");
        addToast(`System Optimized: ${boostType} Increased`, "info");
        
        // 3. Update Visuals
        setGlobalCoherence(prev => Math.min(100, prev + 0.05));
    }, [updateQIAIIPS, qiaiIps, addToast]);

    // --- Simulation Loop ---
    useEffect(() => {
        let interval: ReturnType<typeof setInterval>;

        if (isSimulating) {
            interval = setInterval(() => {
                setUniverses(prevUniverses => prevUniverses.map(u => {
                    let { 
                        optimizationScore, status, currentVersion, nextVersion, 
                        predictionLog, engines, upgradesApplied, stability 
                    } = u;

                    // 1. State Machine Transitions
                    let nextScore = optimizationScore;
                    let nextStatus = status;

                    if (status === 'SAVING') {
                        // Hold in SAVING state briefly
                        if (Math.random() > 0.8) {
                            nextStatus = 'INITIALIZING';
                            predictionLog = "Awaiting next evolutionary cycle...";
                        }
                        return { ...u, status: nextStatus, predictionLog };
                    }

                    // Normal Progress
                    const increment = (Math.random() * 2.5) + 0.5;
                    nextScore += increment;

                    // Phase Logic for Next Level Prediction
                    if (nextScore < 30) {
                        nextStatus = 'INITIALIZING';
                        if (Math.random() > 0.9) {
                            if (u.id === 1) predictionLog = "Scanning Neural Pathways...";
                            if (u.id === 2) predictionLog = "Calibrating Engineering Matrix...";
                            if (u.id === 3) predictionLog = "Loading Molecular Data...";
                        }
                    } else if (nextScore < 70) {
                        nextStatus = 'OPTIMIZING';
                        
                        // --- EVOLUTION PREDICTION LOGIC ---
                        if (Math.random() > 0.8) {
                            const parts = currentVersion.replace('v','').split('.').map(Number);
                            const [maj, min] = parts;
                            
                            // High stability allows for Evolutionary Leaps
                            const canLeap = stability > 98.5 && Math.random() > 0.6;
                            const isMaxMinor = min >= 9;

                            if (isMaxMinor) {
                                nextVersion = `v${maj + 1}.0.0`;
                                predictionLog = `Singularity Threshold Met. Target: ${nextVersion}`;
                            } else if (canLeap) {
                                if (Math.random() > 0.7) {
                                    nextVersion = `v${maj + 1}.0.0`;
                                    predictionLog = `Major Leap Predicted: ${nextVersion}`;
                                } else {
                                    const nextMin = Math.min(9, min + 2);
                                    nextVersion = `v${maj}.${nextMin}.0`;
                                    predictionLog = `Optimizing Vector to ${nextVersion}`;
                                }
                            } else {
                                nextVersion = `v${maj}.${min + 1}.0`;
                                predictionLog = `Standard Evolution to ${nextVersion}`;
                            }
                        }

                    } else if (nextScore < 100) {
                        nextStatus = 'CONVERGING';
                        if (Math.random() > 0.9) predictionLog = `Compiling ${nextVersion} patch...`;
                    } else {
                        // --- TRIGGER UPGRADE ---
                        nextStatus = 'SAVING'; // Brief pause state
                        nextScore = 0;
                        
                        // Apply the predicted version
                        currentVersion = nextVersion;
                        
                        // Calculate *future* next version for display
                        const parts = currentVersion.replace('v','').split('.').map(Number);
                        const [maj, min] = parts;
                        const tempNext = min >= 9 ? `v${maj + 1}.0.0` : `v${maj}.${min + 1}.0`;
                        nextVersion = tempNext;

                        upgradesApplied += 1;
                        stability = Math.min(100, stability + 0.2);

                        predictionLog = `>>> AUTO-IMPLEMENTED: ${currentVersion}`;
                        
                        // Dispatch Side Effect (Applying Patches)
                        let patchName = "generic.q";
                        let patchContent = "// Optimization";
                        
                        if (u.id === 1) {
                            patchName = `neural_synapse_opt_${currentVersion}.q`;
                            patchContent = `// NEURAL PROGRAMMING PATCH\n// TARGET: BCI & SYNAPTIC BRIDGE\nOPTIMIZE_WEIGHTS(LAYER_4);\nINCREASE_DOPAMINE_SENSITIVITY();`;
                        } else if (u.id === 2) {
                            patchName = `eng_topology_${currentVersion}.q`;
                            patchContent = `// QUANTUM ENGINEERING PATCH\n// TARGET: QDL HIERARCHY\nREFACTOR_STACK_DEPTH(12);\nMINIMIZE_GATE_NOISE();`;
                        } else if (u.id === 3) {
                             patchName = `mat_vqe_solver_${currentVersion}.q`;
                             patchContent = `// MATERIAL SCIENCE PATCH\n// TARGET: MOLECULAR FOUNDRY\nCONVERGE_GROUND_STATE();\nSIMULATE_FOLDING_PATH();`;
                        }

                        if (onApplyPatch) {
                            onApplyPatch(patchName, patchContent);
                        }
                        commitUpgradeToCore(u.targetName, currentVersion);
                    }

                    // 2. Engine Load Balancing (Visuals)
                    const nextEngines = engines.map(eng => {
                        let active = false;
                        let load = 5 + Math.random() * 10; // Idle

                        if (u.id === 1) { // Neural
                            if ((eng.id === 'QLLM' || eng.id === 'QML') && (nextStatus === 'OPTIMIZING')) { active = true; load = 85 + Math.random() * 15; }
                        } else if (u.id === 2) { // Engineering
                            if ((eng.id === 'QDL' || eng.id === 'QRL') && (nextStatus === 'CONVERGING')) { active = true; load = 80 + Math.random() * 20; }
                        } else { // Material Science
                            if ((eng.id === 'QGL' || eng.id === 'QRL') && (nextStatus === 'INITIALIZING')) { active = true; load = 75; }
                        }
                        
                        // All engines surge during Upgrade
                        if (nextStatus === 'SAVING') { active = true; load = 100; }

                        return { ...eng, active, load };
                    });

                    return {
                        ...u,
                        optimizationScore: nextScore,
                        status: nextStatus,
                        currentVersion,
                        nextVersion,
                        predictionLog,
                        engines: nextEngines,
                        stability,
                        upgradesApplied
                    };
                }));
            }, 100); // 100ms Tick
        }

        return () => clearInterval(interval);
    }, [isSimulating, commitUpgradeToCore, onApplyPatch]);

    const handleStartAll = () => {
        setIsSimulating(true);
        addToast("3-Vector Simulation Initiated. Auto-Evolution Active.", "success");
    };

    const handleStopAll = () => {
        setIsSimulating(false);
        addToast("Simulation Halted. Evolution Paused.", "warning");
    };

    const handleConsolidate = () => {
        setConsolidationStatus('Consolidating');
        setTimeout(() => {
            setConsolidationStatus('Complete');
            setTimeout(() => setConsolidationStatus('Idle'), 2000);
        }, 3000);
    };

    return (
        <GlassPanel title={
            <div className="flex items-center justify-between w-full">
                <div className="flex items-center">
                    <BrainCircuitIcon className="w-5 h-5 mr-2 text-yellow-400" />
                    <span>Quantum Cognitive Architecture (QCA)</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className={`flex items-center gap-2 px-2 py-1 rounded border ${isSimulating ? 'bg-green-900/30 border-green-500 text-green-400' : 'bg-gray-800 border-gray-600 text-gray-500'}`}>
                        <ActivityIcon className={`w-3 h-3 ${isSimulating ? 'animate-pulse' : ''}`} />
                        <span className="text-[10px] font-bold uppercase">{isSimulating ? 'EVOLUTION RUNNING' : 'SYSTEM IDLE'}</span>
                    </div>
                </div>
            </div>
        }>
            <div className="flex flex-col h-full gap-4 p-4 overflow-y-auto custom-scrollbar">
                
                {/* Top Section: Memory Matrix */}
                <div className="h-1/3 min-h-[180px] w-full flex flex-col gap-2">
                     <div className="flex justify-between items-center px-1">
                        <div className="flex items-center gap-2">
                            <span className="text-[10px] text-yellow-500 uppercase font-bold">System Memory Engrams</span>
                            <span className="text-[9px] bg-black/40 px-2 py-0.5 rounded text-cyan-500 font-mono border border-cyan-900">
                                Coherence: {(globalCoherence || 0).toFixed(4)}%
                            </span>
                        </div>
                        <div className="flex gap-2">
                             <span className="text-[10px] text-gray-400 font-mono">Load: {((systemStatus?.neuralLoad || 0) * 12.5).toFixed(0)} TB/s</span>
                             <button onClick={handleConsolidate} disabled={consolidationStatus !== 'Idle'} className="text-[9px] bg-yellow-900/30 text-yellow-200 px-2 rounded border border-yellow-700 hover:bg-yellow-800/50 transition-colors">
                                 {consolidationStatus}
                             </button>
                        </div>
                    </div>
                    <div className="flex-grow bg-black/40 border border-yellow-800/30 rounded-lg p-3 overflow-hidden relative">
                        <MemoryMatrix 
                            lastActivity={lastActivity || Date.now()} 
                            memorySummary="System Cognitive State"
                            interactive={true}
                            messages={messages} 
                        />
                    </div>
                </div>

                {/* Bottom Section: Multi-Vector Universe Simulators */}
                <div className="flex-grow flex flex-col gap-2 min-h-0">
                    <div className="flex justify-between items-center border-t border-gray-800 pt-2">
                        <span className="text-[10px] text-cyan-400 font-bold uppercase tracking-widest flex items-center gap-2">
                             <GalaxyIcon className="w-3 h-3 animate-spin-slow" /> 3-Vector Universe Simulator
                        </span>
                        <div className="flex gap-2">
                            {!isSimulating ? (
                                <button onClick={handleStartAll} className="holographic-button px-3 py-1 bg-green-600/20 border-green-500 text-green-300 text-[10px] font-bold rounded flex items-center gap-2 hover:bg-green-600/40">
                                    <PlayIcon className="w-3 h-3" /> Initiate Evolution Loop
                                </button>
                            ) : (
                                <button onClick={handleStopAll} className="holographic-button px-3 py-1 bg-red-600/20 border-red-500 text-red-300 text-[10px] font-bold rounded flex items-center gap-2 hover:bg-red-600/40">
                                    <StopIcon className="w-3 h-3" /> Halt Simulation
                                </button>
                            )}
                        </div>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 flex-grow min-h-0">
                        {universes.map((u) => (
                            <div key={u.id} className={`bg-black/30 border rounded-xl p-3 flex flex-col relative overflow-hidden group transition-all duration-500 ${u.status === 'SAVING' ? 'border-green-500 shadow-[0_0_20px_rgba(34,197,94,0.3)]' : 'border-cyan-900/50'}`}>
                                
                                {/* Background Effect */}
                                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-cyan-500 to-transparent opacity-30"></div>
                                <div className={`absolute inset-0 opacity-0 transition-opacity pointer-events-none ${u.status === 'SAVING' ? 'bg-green-500/10 opacity-100' : 'bg-cyan-500/5 group-hover:opacity-100'}`}></div>

                                {/* Header */}
                                <div className="flex justify-between items-start mb-3 z-10">
                                    <div className="min-w-0">
                                        <h4 className="text-xs font-black text-white uppercase tracking-tight truncate">{u.targetName}</h4>
                                        <div className="flex items-center gap-2 mt-1">
                                            <span className="text-[10px] font-mono text-green-400 bg-green-900/30 px-1.5 rounded border border-green-800">{u.currentVersion}</span>
                                            <span className="text-[9px] text-cyan-600 font-mono">→ {u.nextVersion}</span>
                                        </div>
                                    </div>
                                    <div className={`flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-full border shadow-[0_0_10px_rgba(6,182,212,0.3)] bg-black/60 ${u.status === 'SAVING' ? 'border-green-400 bg-green-900/20' : 'border-cyan-700'}`}>
                                        {u.status === 'SAVING' ? (
                                            <SaveIcon className="w-4 h-4 text-green-400 animate-bounce" />
                                        ) : (
                                            <GalaxyIcon className={`w-5 h-5 text-cyan-300 ${isSimulating ? 'animate-spin-slow' : ''}`} />
                                        )}
                                    </div>
                                </div>

                                {/* Status & Progress */}
                                <div className="mb-4 z-10">
                                    <div className="flex justify-between text-[8px] text-gray-400 mb-1 uppercase font-bold">
                                        <span className={u.status === 'SAVING' ? 'text-green-400 animate-pulse' : 'text-cyan-600'}>{u.status === 'SAVING' ? 'AUTO-UPGRADING' : u.status}</span>
                                        <span>{Math.floor(u.optimizationScore || 0)}%</span>
                                    </div>
                                    <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
                                        <div 
                                            className={`h-full transition-all duration-300 ${u.status === 'SAVING' ? 'bg-green-500' : 'bg-gradient-to-r from-cyan-600 via-purple-500 to-white'}`}
                                            style={{ width: `${u.optimizationScore || 0}%` }}
                                        ></div>
                                    </div>
                                    <div className={`mt-2 text-[9px] font-mono h-8 overflow-hidden border-l-2 pl-2 leading-tight transition-colors ${u.status === 'SAVING' ? 'text-green-300 border-green-500' : 'text-cyan-200/80 border-cyan-800'}`}>
                                        {u.predictionLog}
                                    </div>
                                </div>

                                {/* Cognitive Engines (Auto-Configuring) */}
                                <div className="mt-auto z-10">
                                    <p className="text-[8px] text-gray-600 uppercase font-bold mb-1 flex justify-between">
                                        <span>Active Cognitive Matrix</span>
                                        {u.status === 'SAVING' && <span className="text-green-500">UPGRADING CORE</span>}
                                    </p>
                                    <div className="flex justify-between items-center bg-black/40 p-1.5 rounded-lg border border-gray-800">
                                        {u.engines.map((eng) => (
                                            <div 
                                                key={eng.id} 
                                                className={`relative flex items-center justify-center w-6 h-6 rounded transition-all duration-500 ${eng.active ? `${eng.color} bg-white/10 scale-110 shadow-lg` : 'text-gray-700 grayscale scale-90'}`}
                                                title={`${eng.label}: ${(eng.load || 0).toFixed(0)}% Load`}
                                            >
                                                <eng.icon className="w-3.5 h-3.5" />
                                                {eng.active && (
                                                    <div className="absolute -bottom-1 -right-1 w-1.5 h-1.5 bg-green-500 rounded-full border border-black animate-pulse"></div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                    <div className="flex justify-between items-center mt-2 text-[8px] font-mono text-gray-500">
                                        <span className="flex items-center gap-1">
                                            Agent Q: <span className="text-purple-400 font-bold">LVL {12 + u.upgradesApplied}</span>
                                        </span>
                                        <span>Fidelity: {(u.stability || 0).toFixed(1)}%</span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

            </div>
        </GlassPanel>
    );
};

export default QuantumCognitiveArchitecture;
