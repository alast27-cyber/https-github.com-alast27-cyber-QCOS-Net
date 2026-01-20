import React, { useEffect, useRef, useState } from 'react';
import { NetworkIcon, GalaxyIcon, BrainCircuitIcon, AtomIcon, SparklesIcon, RocketLaunchIcon, CodeBracketIcon, StopIcon, PlayIcon, AcademicCapIcon, CheckCircle2Icon, LoaderIcon, RefreshCwIcon, ServerStackIcon, SaveIcon, ActivityIcon, GlobeIcon, BoxIcon, BanknotesIcon, ShieldCheckIcon, CpuChipIcon } from './Icons';
import { useSimulation } from '../context/SimulationContext';
import { useToast } from '../context/ToastContext';
import MemoryMatrix from './MemoryMatrix';
import { MaximizeIcon } from './Icons';

// --- Types ---
interface Domain {
    name: string;
    progress: number;
    status: string;
    color: string;
}

interface EngineConfig {
    id: string;
    label: string;
    icon: any;
    color: string;
    borderColor: string;
    description: string;
    simSteps: string[];
}

interface TrainingSession {
    active: boolean;
    engineId: string | null;
    progress: number;
    logs: string[];
    currentStepIndex: number;
    modelArtifact: { name: string; accuracy: string; parameters: string } | null;
}

const ENGINES: EngineConfig[] = [
    { 
        id: 'QLLM', 
        label: 'QLLM', 
        icon: CodeBracketIcon, 
        color: 'text-purple-400', 
        borderColor: 'border-purple-500',
        description: 'Semantic Probability Tree & IPSNN',
        simSteps: ['Tokenizing Manifold', 'Weight Entanglement', 'Gradient Synthesis', 'Context Collapse']
    },
    { 
        id: 'QCE', 
        label: 'QCE', 
        icon: AtomIcon, 
        color: 'text-cyan-400', 
        borderColor: 'border-cyan-500',
        description: 'Quantum Chromodynamics Engine',
        simSteps: ['Gluon Binding', 'Quark Mapping', 'Lattice Simulation', 'Hadron Sync']
    },
    { 
        id: 'QRL', 
        label: 'QRL', 
        icon: SparklesIcon, 
        color: 'text-green-400', 
        borderColor: 'border-green-500',
        description: 'Reinforcement Learning Agent',
        simSteps: ['Policy Iteration', 'Reward Shaping', 'Value Drift Fix', 'Actor-Critic Sync']
    },
    { 
        id: 'QNN', 
        label: 'QNN', 
        icon: BrainCircuitIcon, 
        color: 'text-blue-400', 
        borderColor: 'border-blue-500',
        description: 'Neural Network Fabric',
        simSteps: ['Synapse Mapping', 'Axon Routing', 'Dendrite Growth', 'Pulse Modulation']
    }
];

const DistributedCognitiveArchitecture: React.FC<{ activeDataStreams?: string[] }> = ({ activeDataStreams = [] }) => {
    const { addToast } = useToast();
    const simulation = useSimulation();

    // Defensive Destructuring: Fixes 'currentStage' of undefined
    const training = simulation?.training ?? { isActive: false, currentStage: 0, logs: [], coherence: 0 };
    const evolution = simulation?.evolution ?? { isActive: false, currentStage: 0, logs: [] };
    
    const [isSimulating, setIsSimulating] = useState(false);
    const [session, setSession] = useState<TrainingSession>({
        active: false,
        engineId: null,
        progress: 0,
        logs: [],
        currentStepIndex: 0,
        modelArtifact: null
    });

    const runEngineSimulation = (engineId: string) => {
        if (session.active) return;
        const engine = ENGINES.find(e => e.id === engineId);
        if (!engine) return;

        setSession({
            active: true,
            engineId,
            progress: 0,
            logs: [`Initialising ${engine.label} Protocol...`],
            currentStepIndex: 0,
            modelArtifact: null
        });

        addToast(`${engine.label} Simulation Sequence Started`, 'info');
    };

    return (
        <div className="flex-grow flex flex-col gap-4 min-h-0 relative overflow-hidden">
            {/* Main Viz Area */}
            <div className="flex-grow grid grid-cols-1 md:grid-cols-4 gap-4 min-h-0">
                {ENGINES.map((engine) => {
                    // Safety check: Is this specific engine active in the context?
                    const isContextActive = (training?.isActive && training?.domain === engine.id) || 
                                          (evolution?.isActive && engine.id === 'QCE');
                    
                    const isLocalActive = session.active && session.engineId === engine.id;

                    return (
                        <div 
                            key={engine.id}
                            className={`relative p-3 rounded-lg border transition-all duration-500 flex flex-col gap-2 ${
                                isLocalActive || isContextActive 
                                ? `${engine.borderColor} bg-black/60 shadow-[0_0_15px_rgba(0,0,0,0.5)]` 
                                : 'border-white/5 bg-black/20 opacity-60 hover:opacity-100'
                            }`}
                        >
                            <div className="flex justify-between items-start">
                                <engine.icon className={`w-5 h-5 ${engine.color}`} />
                                {isLocalActive && <LoaderIcon className={`w-3 h-3 animate-spin ${engine.color}`} />}
                            </div>
                            
                            <div>
                                <h4 className="text-[10px] font-black tracking-tighter text-white">{engine.label}</h4>
                                <p className="text-[8px] text-gray-500 leading-tight uppercase">{engine.description}</p>
                            </div>

                            <div className="mt-auto space-y-1">
                                <div className="flex justify-between text-[8px] font-mono">
                                    <span className="text-gray-600">STATE:</span>
                                    <span className={isLocalActive || isContextActive ? engine.color : 'text-gray-700'}>
                                        {isLocalActive ? 'EXECUTING' : isContextActive ? 'SYNCED' : 'IDLE'}
                                    </span>
                                </div>
                                <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
                                    <div 
                                        className={`h-full transition-all duration-1000 ${engine.color.replace('text', 'bg')}`}
                                        style={{ width: `${isLocalActive ? session.progress : isContextActive ? 100 : 0}%` }}
                                    />
                                </div>
                            </div>

                            {!session.active && !isContextActive && (
                                <button 
                                    onClick={() => runEngineSimulation(engine.id)}
                                    className="absolute inset-0 z-10 opacity-0 cursor-pointer"
                                />
                            )}
                        </div>
                    );
                })}
            </div>

            {/* Terminal Interface */}
            <div className="h-32 bg-black/60 rounded border border-white/5 p-2 font-mono text-[9px] flex flex-col gap-1">
                <div className="flex justify-between items-center text-gray-500 border-b border-white/5 pb-1 mb-1">
                    <span className="flex items-center gap-1"><ActivityIcon className="w-3 h-3"/> MESH_TELEMETRY</span>
                    <span>STAGE: {training?.currentStage ?? 0}/4</span>
                </div>
                <div className="flex-grow overflow-y-auto custom-scrollbar text-cyan-500/80">
                    {session.active ? (
                        session.logs.map((log, i) => <div key={i}>{`> ${log}`}</div>)
                    ) : training?.isActive ? (
                        (training?.logs ?? []).slice(-5).map((log, i) => <div key={i} className="text-purple-400">{`> [REMOTE] ${log}`}</div>)
                    ) : (
                        <div className="text-gray-700 italic">Waiting for Engine Instruction...</div>
                    )}
                </div>
            </div>

            {/* Bottom Controls */}
            <div className="absolute bottom-2 right-2 pointer-events-auto text-right">
                 <div className="flex items-center justify-end gap-2 mb-1 pointer-events-none">
                     <NetworkIcon className="w-4 h-4 text-purple-400" />
                     <span className="text-xs font-bold text-white uppercase tracking-wider">Quantum Cognition Engine</span>
                 </div>
                 <div className="flex items-center gap-2">
                     <div className="text-[9px] text-cyan-500 font-mono bg-black/60 px-2 py-1 rounded w-fit ml-auto border border-cyan-900/50">
                         Entanglement: 100% | Background Sim: <span className={isSimulating ? "text-green-400" : "text-yellow-400"}>{isSimulating ? 'Active' : 'Paused'}</span>
                     </div>
                     <button 
                        onClick={() => setIsSimulating(!isSimulating)}
                        className={`p-1 rounded border transition-colors ${isSimulating ? 'bg-red-900/30 border-red-500 text-red-400 hover:bg-red-900/50' : 'bg-green-900/30 border-green-500 text-green-400 hover:bg-green-900/50'}`}
                     >
                        {isSimulating ? <StopIcon className="w-3 h-3" /> : <PlayIcon className="w-3 h-3" />}
                     </button>
                 </div>
            </div>
        </div>
    );
};

export default DistributedCognitiveArchitecture;