
import React, { useState, useEffect, useCallback, useRef } from 'react';
import GlassPanel from './GlassPanel';
import SyntaxHighlighter from './SyntaxHighlighter';
import { 
    FileCodeIcon, CheckCircle2Icon, LoaderIcon, 
    CpuChipIcon, SparklesIcon,
    GitBranchIcon, BrainCircuitIcon,
    ServerCogIcon, 
    ToggleLeftIcon, ToggleRightIcon, 
    ActivityIcon, ZapIcon, BugAntIcon,
    RefreshCwIcon, FastForwardIcon, LinkIcon,
    GlobeIcon, AtomIcon, RocketLaunchIcon, CodeBracketIcon,
    PlayIcon, CommandIcon, StopIcon
} from './Icons';
import { useToast } from '../context/ToastContext';
import { useSimulation } from '../context/SimulationContext';
import UniverseSimulator from './UniverseSimulator';

interface QCOSGatewayProps {
  codebase: { [key: string]: string };
  onApplyPatch: (filePath: string, newContent: string) => void;
  editCode?: (code: string, instruction: string) => Promise<string>;
  onMaximizeSubPanel?: (id: string) => void;
}

// --- Sub-component: QIAI-IPS Lattice Layer ---
const IPSLayerNode: React.FC<{
    label: string;
    subLabel: string;
    coherence: number;
    load: number;
    status: string;
    color: string;
    isActive: boolean;
}> = ({ label, subLabel, coherence, load, status, color, isActive }) => {
    const colorMap: Record<string, string> = {
        cyan: 'text-cyan-400 border-cyan-500/50 bg-cyan-950/30 shadow-[0_0_15px_rgba(34,211,238,0.2)]',
        purple: 'text-purple-400 border-purple-500/50 bg-purple-950/30 shadow-[0_0_15px_rgba(168,85,247,0.2)]',
        gold: 'text-yellow-400 border-yellow-500/50 bg-yellow-950/30 shadow-[0_0_15px_rgba(234,179,8,0.2)]',
    };

    return (
        <div className={`relative flex flex-col p-4 rounded-xl border-2 transition-all duration-700 ${isActive ? colorMap[color] : 'border-gray-800 opacity-40 grayscale bg-black/20'}`}>
            <div className="flex justify-between items-start mb-3">
                <div>
                    <h4 className="text-[11px] font-black uppercase tracking-widest flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full bg-current ${isActive ? 'animate-pulse' : ''}`}></div>
                        {label}
                    </h4>
                    <p className="text-[9px] opacity-70 italic mt-0.5">{subLabel}</p>
                </div>
                <div className={`text-[8px] font-mono px-2 py-0.5 rounded border border-white/10 bg-black/40 ${isActive ? 'animate-pulse' : ''}`}>
                    {status}
                </div>
            </div>
            <div className="space-y-2">
                <div className="space-y-1">
                    <div className="flex justify-between items-center text-[8px] font-mono">
                        <span className="opacity-60 uppercase tracking-tighter">Stack Coherence</span>
                        <span className="text-white font-bold">{(coherence * 100).toFixed(2)}%</span>
                    </div>
                    <div className="w-full h-1 bg-black/40 rounded-full overflow-hidden border border-white/5">
                        <div className="h-full bg-current transition-all duration-1000" style={{ width: `${coherence * 100}%` }}></div>
                    </div>
                </div>
                <div className="space-y-1">
                    <div className="flex justify-between items-center text-[8px] font-mono">
                        <span className="opacity-60 uppercase tracking-tighter">Synaptic Tension</span>
                        <span className="text-white font-bold">{load}%</span>
                    </div>
                    <div className="w-full h-1 bg-black/40 rounded-full overflow-hidden border border-white/5">
                        <div className="h-full bg-white opacity-30" style={{ width: `${load}%` }}></div>
                    </div>
                </div>
            </div>
        </div>
    );
};

const CognitiveEngineCard: React.FC<{
    id: string;
    label: string;
    version: number;
    progress: number;
    icon: React.FC<{className?: string}>;
    color: string;
}> = ({ id, label, version, progress, icon: Icon, color }) => (
    <div className={`flex flex-col bg-black/40 border border-${color}-500/30 p-2 rounded-lg relative overflow-hidden group`}>
        <div className={`absolute top-0 left-0 w-full h-0.5 bg-${color}-500 opacity-50`}></div>
        <div className="flex justify-between items-start mb-2">
            <Icon className={`w-4 h-4 text-${color}-400 group-hover:scale-110 transition-transform`} />
            <span className={`text-[8px] font-mono text-${color}-300 bg-${color}-900/20 px-1 rounded`}>v{version}.0</span>
        </div>
        <div className="mt-auto">
            <div className="flex justify-between items-end mb-1">
                <span className="text-[9px] font-bold text-gray-300">{label}</span>
                <span className="text-[8px] text-gray-500">{Math.floor(progress)}%</span>
            </div>
            <div className="w-full h-1 bg-gray-800 rounded-full overflow-hidden">
                <div className={`h-full bg-${color}-500 transition-all duration-300`} style={{ width: `${progress}%` }}></div>
            </div>
        </div>
        {/* Sync Indicator */}
        <div className="absolute top-1 right-1">
            <LinkIcon className="w-2 h-2 text-white/20 group-hover:text-white/60" />
        </div>
    </div>
);

const QCOSGateway: React.FC<QCOSGatewayProps> = ({ codebase, onApplyPatch, editCode, onMaximizeSubPanel }) => {
    const { qiaiIps, updateQIAIIPS, submitInquiry, inquiry, qceState } = useSimulation();
    const { addToast } = useToast();
    const [activeTab, setActiveTab] = useState<'source' | 'ips' | 'cue'>('cue');
    const [selectedFile, setSelectedFile] = useState<string | null>(null);
    
    // Cue Simulation State
    const [cues, setCues] = useState<string[]>([
        'Stochastic Hamiltonian Evolution', 
        'Quantum Walk Propagation', 
        'Decoherence-Induced Phase Mapping'
    ]);
    const [cueStatus, setCueStatus] = useState<('IDLE' | 'QUEUED' | 'TRANSMITTING' | 'ACTIVE' | 'EVOLVING' | 'PATCHING' | 'IMPLEMENTED')[]>(['IDLE', 'IDLE', 'IDLE']);
    const [isContinuous, setIsContinuous] = useState(false);
    const simulationInterval = useRef<ReturnType<typeof setInterval> | null>(null);
    const [evoVersion, setEvoVersion] = useState(4.5);

    // IPS State
    const [isSimulating, setIsSimulating] = useState(false);
    const [simLogs, setSimLogs] = useState<string[]>([]);
    const [prediction, setPrediction] = useState<any>(null);

    const runSimulation = useCallback(async (type: 'debug' | 'fix' | 'optimize' | 'evolve') => {
        setIsSimulating(true);
        setPrediction(null);
        const timestamp = new Date().toLocaleTimeString();
        setSimLogs(prev => [`[${timestamp}] INITIALIZING QIAI-IPS ${(type || "").toUpperCase()} SEQUENCE...`, ...prev].slice(0, 50));
        
        // Dynamic Layer Activation
        updateQIAIIPS({
            qil: { ...qiaiIps.qil, status: 'INGESTING', load: 88 },
            qips: { ...qiaiIps.qips, status: 'SOLVING', load: 94 },
            qcl: { ...qiaiIps.qcl, status: 'GOVERNING', load: 45 }
        });

        const fileList = Object.keys(codebase).join(', ');
        const prompt = `Act as Agent Q's QIAI-IPS Core. Run a 12-dimensional simulation of the following project scripts: ${fileList}. 
        Task: ${(type || "").toUpperCase()}. 
        Return result as JSON: { 
            "stage_prediction": "string", 
            "optimization_report": "string", 
            "fidelity_delta": "number", 
            "fix_applied": "string",
            "next_evolution_vector": "string",
            "affected_engines": ["string"]
        }`;

        submitInquiry(prompt, 'qiai-ips-core', 'universe-solver');
    }, [submitInquiry, updateQIAIIPS, qiaiIps, codebase]);

    const handleCueChange = (index: number, value: string) => {
        const newCues = [...cues];
        newCues[index] = value;
        setCues(newCues);
    };

    const toggleContinuousEvolution = () => {
        if (isContinuous) {
            // STOP
            if (simulationInterval.current) clearInterval(simulationInterval.current);
            setIsContinuous(false);
            setCueStatus(prev => prev.map(() => 'IDLE'));
            addToast("Evolutionary Loop Halted.", "warning");
        } else {
            // START
            const activeIndices = cues.map((c, i) => c.trim() ? i : -1).filter(i => i !== -1);
            if (activeIndices.length === 0) {
                addToast("No simulation vectors defined.", "error");
                return;
            }

            setIsContinuous(true);
            addToast("Initializing Continuous Background Evolution...", "success");

            // Loop Logic
            simulationInterval.current = setInterval(() => {
                // 1. Select a random active cue to process
                const targetIdx = activeIndices[Math.floor(Math.random() * activeIndices.length)];
                const cueName = cues[targetIdx];

                // 2. Visual Update: Synthesis Phase
                setCueStatus(prev => prev.map((s, i) => i === targetIdx ? 'EVOLVING' : (s === 'IMPLEMENTED' ? 'ACTIVE' : s)));

                setTimeout(() => {
                    // 3. Increment Version (Predicting next stage)
                    setEvoVersion(prev => {
                        const next = parseFloat((prev + 0.01).toFixed(2));
                        
                        // 4. Generate Contextual Patch Code
                        const patchName = `auto_evo_v${next}_${cueName.split(' ')[0].toLowerCase()}.q`;
                        const sanitizedCue = cueName.replace(/[^a-zA-Z0-9 ]/g, "").toUpperCase().replace(/ /g, "_");
                        const patchContent = `// AUTO-GENERATED EVOLUTION PATCH v${next}
// VECTOR_SOURCE: ${cueName}
// STATUS: IMPLEMENTED
// Q-ENTROPY: ${(Math.random()).toFixed(4)}

MODULE EVOLUTION_KERNEL {
    TARGET_NODE = "GATEWAY_CORE";
    OPTIMIZATION_VECTOR = "${sanitizedCue}";
    
    // Heuristic application of simulation result
    FUNCTION UPGRADE_KERNEL() {
        QREG q_state[8];
        INIT_STATE(q_state, "${sanitizedCue}");
        APPLY_OPTIMIZATION(q_state);
        MEASURE q_state -> KERNEL_CONFIG;
        RETURN "SYSTEM_UPGRADED";
    }
}

// Applying patch immediately...
UPGRADE_KERNEL();`;

                        // 5. Apply Patch (Implementation)
                        onApplyPatch(patchName, patchContent);
                        
                        // 6. Submit to Universe to register the event
                        submitInquiry(
                            `System Upgrade v${next} Implemented. Vector: ${cueName}. Patch applied to kernel.`, 
                            'gateway-evolution', 
                            'universe-solver'
                        );

                        return next;
                    });

                    // 7. Visual Update: Implementation Phase
                    setCueStatus(prev => prev.map((s, i) => i === targetIdx ? 'PATCHING' : s));
                    
                    // 8. Confirmation
                    setTimeout(() => {
                        setCueStatus(prev => prev.map((s, i) => i === targetIdx ? 'IMPLEMENTED' : s));
                        addToast(`Patch Implemented: ${cueName}`, "success");
                    }, 800);

                }, 1500); // Synthesis Time

            }, 4000); // 4 Second Cycle
        }
    };

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (simulationInterval.current) clearInterval(simulationInterval.current);
        };
    }, []);

    useEffect(() => {
        if (isSimulating && inquiry.status === 'complete' && inquiry.result) {
            try {
                const data = JSON.parse(inquiry.result.match(/\{[\s\S]*\}/)?.[0] || inquiry.result);
                setPrediction(data);
                setIsSimulating(false);
                setSimLogs(prev => [`[CORE] Lattice Analysis Converged.`, ...prev]);
                
                updateQIAIIPS({
                    qil: { ...qiaiIps.qil, status: 'IDLE', load: 12 },
                    qips: { ...qiaiIps.qips, status: 'IDLE', load: 24 },
                    qcl: { ...qiaiIps.qcl, status: 'IDLE', load: 5 }
                });

                if (data.fix_applied && codebase['App.tsx']) {
                    addToast("Predictive Fix Suggested for System Core", "info");
                }
            } catch (e) {
                console.error("Simulation parse failed", e);
                setIsSimulating(false);
                updateQIAIIPS({
                    qil: { ...qiaiIps.qil, status: 'IDLE', load: 0 },
                    qips: { ...qiaiIps.qips, status: 'IDLE', load: 0 },
                    qcl: { ...qiaiIps.qcl, status: 'IDLE', load: 0 }
                });
            }
        }
    }, [inquiry, isSimulating, codebase, qiaiIps, updateQIAIIPS, addToast]);

    return (
        <div className="h-full flex flex-col gap-0 bg-black/40 overflow-hidden relative text-cyan-100">
            {/* Header Tabs */}
            <div className="flex border-b border-cyan-900/50 bg-black/40">
                <button 
                    onClick={() => setActiveTab('ips')}
                    className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-widest flex items-center justify-center gap-2 transition-colors ${activeTab === 'ips' ? 'bg-cyan-900/20 text-cyan-300 border-b-2 border-cyan-500' : 'text-gray-500 hover:text-cyan-400'}`}
                >
                    <BrainCircuitIcon className="w-3 h-3" /> QIAI-IPS Core
                </button>
                <button 
                    onClick={() => setActiveTab('source')}
                    className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-widest flex items-center justify-center gap-2 transition-colors ${activeTab === 'source' ? 'bg-purple-900/20 text-purple-300 border-b-2 border-purple-500' : 'text-gray-500 hover:text-purple-400'}`}
                >
                    <GlobeIcon className="w-3 h-3" /> Gateway Source
                </button>
                <button 
                    onClick={() => setActiveTab('cue')}
                    className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-widest flex items-center justify-center gap-2 transition-colors ${activeTab === 'cue' ? 'bg-green-900/20 text-green-300 border-b-2 border-green-500' : 'text-gray-500 hover:text-green-400'}`}
                >
                    <FastForwardIcon className="w-3 h-3" /> Cue Simulation
                </button>
            </div>

            <div className="flex-grow min-h-0 relative p-4 overflow-y-auto custom-scrollbar">
                {activeTab === 'ips' && (
                    <div className="grid grid-cols-1 gap-6">
                        {/* IPS Layer Visualization */}
                        <div className="flex flex-col gap-3">
                            <IPSLayerNode 
                                label="QCL-NN" 
                                subLabel="Cognition Layer (Governing)" 
                                color="gold" 
                                coherence={qiaiIps.qcl.coherence} 
                                load={qiaiIps.qcl.load} 
                                status={qiaiIps.qcl.status} 
                                isActive={true} 
                            />
                            <div className="h-4 w-0.5 bg-gradient-to-b from-yellow-500/50 to-purple-500/50 mx-auto"></div>
                            <IPSLayerNode 
                                label="QIPS-NN" 
                                subLabel="Instinctive Solver (Heuristics)" 
                                color="purple" 
                                coherence={qiaiIps.qips.coherence} 
                                load={qiaiIps.qips.load} 
                                status={qiaiIps.qips.status} 
                                isActive={true} 
                            />
                            <div className="h-4 w-0.5 bg-gradient-to-b from-purple-500/50 to-cyan-500/50 mx-auto"></div>
                            <IPSLayerNode 
                                label="QIL-NN" 
                                subLabel="Intuitive Learner (Data)" 
                                color="cyan" 
                                coherence={qiaiIps.qil.coherence} 
                                load={qiaiIps.qil.load} 
                                status={qiaiIps.qil.status} 
                                isActive={true} 
                            />
                        </div>

                        {/* Controls */}
                        <div className="grid grid-cols-2 gap-3">
                            <button onClick={() => runSimulation('optimize')} disabled={isSimulating} className="holographic-button py-3 bg-cyan-900/20 border-cyan-500/50 text-cyan-200 text-[10px] font-black uppercase tracking-widest flex items-center justify-center gap-2 hover:bg-cyan-900/40">
                                {isSimulating ? <LoaderIcon className="w-3 h-3 animate-spin"/> : <SparklesIcon className="w-3 h-3" />}
                                Optimize
                            </button>
                            <button onClick={() => runSimulation('evolve')} disabled={isSimulating} className="holographic-button py-3 bg-purple-900/20 border-purple-500/50 text-purple-200 text-[10px] font-black uppercase tracking-widest flex items-center justify-center gap-2 hover:bg-purple-900/40">
                                {isSimulating ? <LoaderIcon className="w-3 h-3 animate-spin"/> : <FastForwardIcon className="w-3 h-3" />}
                                Evolve
                            </button>
                        </div>

                        {/* Logs */}
                        <div className="h-32 bg-black/60 border border-cyan-900/30 rounded-lg p-2 font-mono text-[9px] text-gray-500 overflow-y-auto custom-scrollbar shadow-inner">
                            {simLogs.map((log, i) => (
                                <div key={i} className="mb-0.5 border-b border-white/5 pb-0.5 last:border-0 truncate">
                                    {log}
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {activeTab === 'source' && (
                    <div className="h-full flex flex-col gap-3">
                        {/* Upper Half: Universe Simulator Integration */}
                        <div className="flex-grow bg-black/40 rounded-xl border border-cyan-800/50 relative overflow-hidden group">
                             <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-cyan-500 to-transparent opacity-50 z-20"></div>
                             
                             <div className="absolute top-2 left-3 z-20 flex items-center gap-2">
                                <span className="text-[10px] font-black text-cyan-400 uppercase tracking-widest bg-black/60 px-2 py-0.5 rounded border border-cyan-900 backdrop-blur-sm">
                                    Universe Simulator Root
                                </span>
                                <div className="flex items-center gap-1 text-[8px] text-green-400 bg-green-900/20 px-1.5 py-0.5 rounded border border-green-800 animate-pulse">
                                    <LinkIcon className="w-2 h-2" /> SYNCED
                                </div>
                             </div>

                             {/* Embedded Simulator */}
                             <div className="w-full h-full transform scale-95 origin-center opacity-90 group-hover:scale-100 group-hover:opacity-100 transition-all duration-700">
                                 <UniverseSimulator embedded={true} />
                             </div>
                        </div>

                        {/* Lower Half: Synchronized Cognitive Engines */}
                        <div className="h-28 grid grid-cols-5 gap-2">
                            <CognitiveEngineCard 
                                id="qllm" 
                                label="QLLM Core" 
                                version={qceState.currentStage.QLLM} 
                                progress={qceState.evolutionProgress.QLLM} 
                                icon={CodeBracketIcon} 
                                color="purple" 
                            />
                            <CognitiveEngineCard 
                                id="qml" 
                                label="QML Forge" 
                                version={qceState.currentStage.QML} 
                                progress={qceState.evolutionProgress.QML} 
                                icon={BrainCircuitIcon} 
                                color="cyan" 
                            />
                            <CognitiveEngineCard 
                                id="qrl" 
                                label="QRL Strat" 
                                version={qceState.currentStage.QRL} 
                                progress={qceState.evolutionProgress.QRL} 
                                icon={RocketLaunchIcon} 
                                color="orange" 
                            />
                            <CognitiveEngineCard 
                                id="qgl" 
                                label="QGL Gen" 
                                version={qceState.currentStage.QGL} 
                                progress={qceState.evolutionProgress.QGL} 
                                icon={SparklesIcon} 
                                color="pink" 
                            />
                            <CognitiveEngineCard 
                                id="qdl" 
                                label="QDL Deep" 
                                version={qceState.currentStage.QDL} 
                                progress={qceState.evolutionProgress.QDL} 
                                icon={AtomIcon} 
                                color="blue" 
                            />
                        </div>
                    </div>
                )}

                {activeTab === 'cue' && (
                    <div className="h-full flex flex-col gap-4">
                        <div className="bg-black/40 border border-green-900/50 p-4 rounded-xl">
                            <h4 className="text-sm font-bold text-green-300 uppercase tracking-widest mb-2 flex items-center gap-2">
                                <CommandIcon className="w-4 h-4" /> Multi-Vector Simulation & Evolution
                            </h4>
                            <p className="text-xs text-gray-400 mb-4">
                                Define simultaneous simulation domains. The gateway will run a continuous background loop to predict evolution stages and auto-apply optimization patches to the system core.
                            </p>
                            
                            <div className="space-y-3">
                                {[0, 1, 2].map(i => (
                                    <div key={i} className="flex items-center gap-3">
                                        <div className="flex-shrink-0 w-16 text-[9px] font-mono text-cyan-600 uppercase text-right">
                                            Vector {String.fromCharCode(65 + i)}
                                        </div>
                                        <div className="flex-grow relative">
                                            <input 
                                                type="text" 
                                                value={cues[i]}
                                                onChange={(e) => handleCueChange(i, e.target.value)}
                                                placeholder={`Enter simulation cue for Vector ${String.fromCharCode(65 + i)}...`}
                                                className="w-full bg-black/50 border border-cyan-800 rounded px-3 py-2 text-xs text-white focus:border-cyan-500 outline-none transition-colors"
                                                disabled={isContinuous}
                                            />
                                            {cueStatus[i] !== 'IDLE' && (
                                                <div className={`absolute right-2 top-1/2 -translate-y-1/2 text-[8px] font-bold px-1.5 py-0.5 rounded bg-black/60 border animate-pulse ${
                                                    cueStatus[i] === 'PATCHING' ? 'border-green-500 text-green-300' :
                                                    cueStatus[i] === 'IMPLEMENTED' ? 'border-green-400 text-green-400 bg-green-900/20' :
                                                    cueStatus[i] === 'EVOLVING' ? 'border-purple-500 text-purple-300' :
                                                    'border-cyan-700 text-cyan-300'
                                                }`}>
                                                    {cueStatus[i]}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="mt-auto">
                            <div className="mb-2 flex justify-between items-center text-[9px] font-mono text-cyan-500 bg-black/30 p-2 rounded border border-cyan-900/30">
                                <span>SYSTEM VERSION: v{evoVersion.toFixed(2)}</span>
                                <span className={isContinuous ? "text-green-400 animate-pulse" : "text-gray-500"}>
                                    {isContinuous ? "AUTO-EVOLUTION: ENGAGED" : "AUTO-EVOLUTION: STANDBY"}
                                </span>
                            </div>
                            <button 
                                onClick={toggleContinuousEvolution}
                                className={`w-full py-4 border font-black text-sm uppercase tracking-[0.2em] rounded-xl transition-all flex items-center justify-center gap-3 active:scale-95 group ${
                                    isContinuous 
                                        ? 'bg-red-600/20 border-red-500 text-red-300 hover:bg-red-600/30' 
                                        : 'bg-gradient-to-r from-green-600/20 to-cyan-600/20 border-green-500/50 text-green-300 hover:from-green-600/30 hover:to-cyan-600/30'
                                }`}
                            >
                                {isContinuous ? <StopIcon className="w-5 h-5 group-hover:text-white" /> : <PlayIcon className="w-5 h-5 group-hover:text-white" />}
                                {isContinuous ? 'Halt Evolution Loop' : 'Initiate Continuous Evolution'}
                            </button>
                            <p className="text-[8px] text-center text-gray-600 mt-2 font-mono uppercase">
                                Continuous background simulation will predict optimal configurations and auto-patch the kernel.
                            </p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default QCOSGateway;
