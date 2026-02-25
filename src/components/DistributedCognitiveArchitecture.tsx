
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

// --- Engine Definitions based on Docs ---
const ENGINES: EngineConfig[] = [
    { 
        id: 'QLLM', 
        label: 'QLLM', 
        icon: CodeBracketIcon, 
        color: 'text-purple-400', 
        borderColor: 'border-purple-500',
        description: 'Semantic Probability Tree & IPSNN Optimization',
        simSteps: [
            "Injecting Neuro-Semantic Data...",
            "Mapping Synaptic Weights to Qubits...",
            "Generating Semantic Probability Tree...",
            "Evolving Neural Superposition...",
            "Calculating Branching Trajectories...",
            "IPSNN: Minimizing Global Energy...",
            "Collapsing Wavefunction to Optimal Thought...",
            "Synthesizing Neuro-Response..."
        ]
    },
    { 
        id: 'QML', 
        label: 'QCA (Memory)', 
        icon: BrainCircuitIcon, 
        color: 'text-cyan-400', 
        borderColor: 'border-cyan-500',
        description: 'Quantum Cognitive Architecture (QSIL, QWP, QIE)',
        simSteps: [
            "QSIL: Transducing Sensory Data...",
            "QWP: Loading Working Memory Register...",
            "QIE: Running Subconscious Annealing...",
            "Neuro-Pattern Probability > 95%...",
            "QCOU: Triggering Memory Consolidation...",
            "QLTM: Mapping to Neural Network...",
            "QEC: Stabilizing Synaptic Coherence...",
            "QCA: Cognitive State Converged."
        ]
    },
    { 
        id: 'QRL', 
        label: 'QRL', 
        icon: RocketLaunchIcon, 
        color: 'text-orange-400', 
        borderColor: 'border-orange-500',
        description: 'Variational Circuit Policy Optimization',
        simSteps: [
            "Encoding State S into Qubits...",
            "Executing Variational Circuit...",
            "Measuring Action Probability...",
            "Collapsing to Classical Action...",
            "Evaluating Dopamine/Reward Signal...",
            "Updating Theta Parameters...",
            "Amplifying Amplitude...",
            "Policy Converged."
        ]
    },
    { 
        id: 'QGL', 
        label: 'QGL', 
        icon: SparklesIcon, 
        color: 'text-pink-400', 
        borderColor: 'border-pink-500',
        description: 'QGAN Minimax Game (Gen vs Disc)',
        simSteps: [
            "Initializing Generator State...",
            "Applying Entangling Gates...",
            "Generator: Creating Neuro-State...",
            "Discriminator: Sampling Real vs Fake...",
            "Calculating Minimax Loss...",
            "Classical Update: Adjusting Parameters...",
            "Checking Nash Equilibrium...",
            "Convergence Reached."
        ]
    },
    { 
        id: 'QDL', 
        label: 'QDL', 
        icon: AtomIcon, 
        color: 'text-blue-400', 
        borderColor: 'border-blue-500',
        description: 'Deep Quantum Stacking & QCNN',
        simSteps: [
            "Initializing QCNN Layers...",
            "Applying Spatial Filter Gates...",
            "Executing Pooling Layer...",
            "Checking for Barren Plateaus...",
            "Applying Data Re-uploading...",
            "Entangling Dense Layer...",
            "Backpropagating via Parameter Shift...",
            "Deep Architecture Optimized."
        ]
    },
];

const qiaiLayers = [
    { id: 'QCL', short: 'QCL-NN', name: 'Cognition Layer' },
    { id: 'QIPS', short: 'QIPS-NN', name: 'Instinctive Solver' },
    { id: 'QIL', short: 'QIL-NN', name: 'Intuitive Learner' }
];

const DistributedCognitiveArchitecture: React.FC<{ activeDataStreams: string[] }> = ({ activeDataStreams }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { qceState, updateQIAIIPS, qiaiIps } = useSimulation();
    const { addToast } = useToast();
    const [isSimulating, setIsSimulating] = useState(true);
    const timeRef = useRef(0);

    // --- Training State ---
    const [training, setTraining] = useState<TrainingSession>({
        active: false,
        engineId: null,
        progress: 0,
        logs: [],
        currentStepIndex: 0,
        modelArtifact: null
    });

    // --- Domain App Development State ---
    const [domains, setDomains] = useState<Domain[]>([
        { name: 'Neuro-Mapping', progress: 12, status: 'Designing', color: 'text-cyan-400' },
        { name: 'Synaptic Bridge', progress: 45, status: 'Building', color: 'text-purple-400' },
        { name: 'Cognitive Core', progress: 78, status: 'Optimizing', color: 'text-green-400' }
    ]);

    // --- Mastery Simulation Loop ---
    useEffect(() => {
        if (!isSimulating) return;
        const interval = setInterval(() => {
            setDomains(prevDomains => prevDomains.map(d => {
                let nextProgress = d.progress + (Math.random() * 1.5) + 0.2;
                let nextStatus = d.status;
                if (nextProgress < 30) nextStatus = 'Researching';
                else if (nextProgress < 60) nextStatus = 'Prototyping';
                else if (nextProgress < 90) nextStatus = 'Refining';
                else if (nextProgress < 100) nextStatus = 'Deploying';
                if (nextProgress >= 100) return { ...d, progress: 0, status: 'Iterating Next Gen' };
                return { ...d, progress: nextProgress, status: nextStatus };
            }));
        }, 100); 
        return () => clearInterval(interval);
    }, [isSimulating]);

    // --- Engine Training Loop ---
    useEffect(() => {
        if (!training.active || !training.engineId) return;

        const engine = ENGINES.find(e => e.id === training.engineId);
        if (!engine) return;

        const interval = setInterval(() => {
            setTraining(prev => {
                if (prev.progress >= 100) {
                    clearInterval(interval);
                    
                    // Generate specific artifact names based on Engine Type
                    let artifactName = `${engine.id}-NeuroModel-v${Math.floor(Math.random() * 10)}.0`;
                    let paramType = "Q-Params";
                    
                    if (engine.id === 'QLLM') { artifactName = `Semantic-Tree-v${Math.floor(Math.random() * 5 + 1)}.0`; paramType = "Tokens"; }
                    else if (engine.id === 'QML') { artifactName = `QCA-Memory-Matrix-v${Math.floor(Math.random() * 10)}.0`; paramType = "Engrams"; }
                    else if (engine.id === 'QRL') { artifactName = `Policy-Gradient-v${Math.floor(Math.random() * 20)}.0`; paramType = "States"; }
                    else if (engine.id === 'QGL') { artifactName = `QGAN-Generator-v${Math.floor(Math.random() * 5)}.0`; paramType = "Vectors"; }
                    else if (engine.id === 'QDL') { artifactName = `Deep-Stack-v${Math.floor(Math.random() * 8)}.0`; paramType = "Layers"; }

                    // --- AUTO UPDATE CORE ---
                    // Push results to QCOS / Agent Q Core
                    if (updateQIAIIPS) {
                         updateQIAIIPS({
                             qcl: { ...qiaiIps.qcl, coherence: Math.min(1.0, qiaiIps.qcl.coherence + 0.02), load: 45 },
                             qips: { ...qiaiIps.qips, coherence: Math.min(1.0, qiaiIps.qips.coherence + 0.01) },
                             globalSync: Math.min(100, qiaiIps.globalSync + 1.2)
                         });
                    }
                    addToast(`Neuro Science: ${artifactName} integrated into Agent Q Core.`, "success");

                    return {
                        ...prev,
                        active: false,
                        logs: [...prev.logs, ">>> MODEL GENERATION COMPLETE - CORE UPDATED"],
                        modelArtifact: {
                            name: artifactName,
                            accuracy: (95 + Math.random() * 4.9).toFixed(2) + '%',
                            parameters: `${Math.floor(Math.random() * 5000)} ${paramType}`
                        }
                    };
                }

                const newProgress = Math.min(100, prev.progress + 1.5);
                const stepIndex = Math.floor((newProgress / 100) * engine.simSteps.length);
                
                let newLogs = prev.logs;
                if (stepIndex > prev.currentStepIndex && engine.simSteps[stepIndex]) {
                    newLogs = [...prev.logs, `[${new Date().toLocaleTimeString()}] ${engine.simSteps[stepIndex]}`].slice(-6);
                }

                return {
                    ...prev,
                    progress: newProgress,
                    currentStepIndex: stepIndex,
                    logs: newLogs
                };
            });
        }, 100);

        return () => clearInterval(interval);
    }, [training.active, training.engineId, updateQIAIIPS, qiaiIps, addToast]);

    const startCueTraining = (engineId: string) => {
        setTraining({
            active: true,
            engineId: engineId,
            progress: 0,
            logs: [`[NEURO-SCI] Initiating ${engineId} Synaptic Mapping...`],
            currentStepIndex: 0,
            modelArtifact: null
        });
    };

    // --- Canvas Rendering ---
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let frameId: number;

        const resize = () => {
            if(canvas.parentElement) {
                canvas.width = canvas.parentElement.clientWidth;
                canvas.height = canvas.parentElement.clientHeight;
            }
        };
        resize();
        window.addEventListener('resize', resize);

        const render = () => {
            timeRef.current += 0.01;
            const t = timeRef.current;
            const w = canvas.width;
            const h = canvas.height;
            const cx = w / 2;
            const cy = h / 2;
            ctx.clearRect(0, 0, w, h);

            // 1. Entanglement Lines
            const outerRadius = Math.min(w, h) * 0.35;
            
            // Draw connections based on training state
            ENGINES.forEach((eng, i) => {
                const angle = (i / 5) * Math.PI * 2 - Math.PI / 2;
                const x = cx + Math.cos(angle) * outerRadius;
                const y = cy + Math.sin(angle) * outerRadius;

                // Line to center
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(cx, cy);
                
                const isTrainingThis = training.active && training.engineId === eng.id;
                
                if (isTrainingThis) {
                    ctx.strokeStyle = eng.color.replace('text-', 'rgba(').replace('-400', ', 200, 255, 0.8)').replace('cyan', '0, 255, 255').replace('purple', '168, 85, 247').replace('orange', '251, 146, 60').replace('pink', '244, 114, 182').replace('blue', '96, 165, 250'); // Approx colors
                    ctx.lineWidth = 3;
                    ctx.setLineDash([5, 5]);
                    ctx.lineDashOffset = -t * 20;
                } else {
                    ctx.strokeStyle = 'rgba(0, 255, 255, 0.1)';
                    ctx.lineWidth = 1;
                    ctx.setLineDash([]);
                }
                ctx.stroke();

                // Training Ring (Progress)
                if (isTrainingThis) {
                    ctx.beginPath();
                    ctx.arc(x, y, 40, -Math.PI/2, (-Math.PI/2) + (Math.PI * 2 * (training.progress / 100)));
                    ctx.strokeStyle = '#fff';
                    ctx.lineWidth = 2;
                    ctx.setLineDash([]);
                    ctx.stroke();
                }
            });

            frameId = requestAnimationFrame(render);
        };

        if (isSimulating) {
            render();
        }

        return () => {
            cancelAnimationFrame(frameId);
            window.removeEventListener('resize', resize);
        };
    }, [isSimulating, training]);
    
    return (
        <div className="w-full h-full relative bg-black/20 overflow-hidden flex items-center justify-center">
            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
            
            {/* --- 1. Grand Universe Simulator (Center) --- */}
            <div className={`absolute z-30 flex flex-col items-center justify-center w-32 h-32 bg-black/80 rounded-full border-2 border-cyan-500/50 shadow-[0_0_30px_rgba(6,182,212,0.3)] ${isSimulating ? 'animate-pulse-slow' : ''}`}>
                <GalaxyIcon className={`w-10 h-10 text-white ${isSimulating ? 'animate-spin-slow' : ''}`} />
                <span className="text-[9px] font-black text-cyan-300 mt-1 uppercase tracking-widest text-center">Grand Universe<br/>Sim</span>
                <span className="text-[7px] text-gray-400 font-mono">QIAI-IPS Core</span>
            </div>

            {/* --- 2. QIAI-IPS Layers (Inner Orbit) --- */}
            <div className={`absolute w-[220px] h-[220px] rounded-full border border-purple-500/20 border-dashed pointer-events-none ${isSimulating ? 'animate-spin-reverse-slow' : ''}`}></div>
            {qiaiLayers.map((layer, i) => {
                const angle = (i / 3) * Math.PI * 2;
                const radius = 35; 
                const top = 50 + Math.sin(angle) * radius;
                const left = 50 + Math.cos(angle) * radius;
                return (
                    <div 
                        key={i}
                        className="absolute w-16 h-16 flex flex-col items-center justify-center bg-black/60 backdrop-blur-md rounded-xl border border-purple-500/40 shadow-lg z-20 pointer-events-none"
                        style={{ top: `${top}%`, left: `${left}%`, transform: 'translate(-50%, -50%)' }}
                    >
                        <span className="text-[8px] font-bold text-purple-300 uppercase">{layer.short}</span>
                        <div className="w-8 h-1 bg-gray-800 rounded-full overflow-hidden mt-1">
                            <div className={`h-full bg-purple-500 ${isSimulating ? 'animate-pulse' : ''}`} style={{width: '80%'}}></div>
                        </div>
                    </div>
                );
            })}

            {/* --- 3. Domain App Development Matrix (Top Left HUD) --- */}
            <div className="absolute top-2 left-2 z-40 flex flex-col gap-2">
                <div className="bg-black/60 p-2 rounded-lg border border-cyan-900/50 backdrop-blur-sm w-48">
                    <h4 className="text-[10px] font-black text-white uppercase tracking-widest mb-2 flex items-center gap-1">
                        <AcademicCapIcon className="w-3 h-3 text-yellow-400" /> Neuro Science Dev
                    </h4>
                    <div className="space-y-2">
                        {domains.map((d, i) => (
                            <div key={d.name} className="flex flex-col">
                                <div className="flex justify-between items-end mb-0.5">
                                    <span className={`text-[9px] font-bold ${d.color}`}>{d.name}</span>
                                    <span className="text-[7px] text-gray-400 font-mono uppercase">{d.status}</span>
                                </div>
                                <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden border border-white/5">
                                    <div 
                                        className={`h-full transition-all duration-300 ${d.progress > 95 ? 'bg-white shadow-[0_0_10px_white]' : d.color.replace('text', 'bg')}`} 
                                        style={{ width: `${d.progress}%` }}
                                    ></div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="flex flex-wrap gap-1 w-48">
                    {['Synaptic Map', 'Cognitive', 'Neural', 'Predictive'].map((mode, i) => (
                        <div key={i} className="flex items-center gap-1 bg-black/60 px-2 py-0.5 rounded border border-gray-800">
                            <div className={`w-1 h-1 rounded-full ${i % 2 === 0 ? 'bg-cyan-500' : 'bg-purple-500'} ${isSimulating ? 'animate-pulse' : ''}`}></div>
                            <span className="text-[7px] font-mono text-gray-400 uppercase tracking-tight">{mode}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* --- 4. Quantum Model Forge (Right Side HUD) --- */}
            <div className="absolute top-2 right-2 z-40 flex flex-col gap-2 w-64">
                <div className="bg-black/80 backdrop-blur-md p-3 rounded-lg border border-purple-500/30">
                    <h4 className="text-[10px] font-black text-white uppercase tracking-widest mb-2 flex items-center gap-2 border-b border-purple-800/50 pb-1">
                        <SparklesIcon className="w-3 h-3 text-purple-400" /> Neuro-Evolution Forge
                    </h4>
                    
                    {training.active ? (
                        <div className="space-y-3 animate-fade-in">
                            <div className="flex items-center justify-between">
                                <span className="text-xs font-bold text-cyan-300">Training: {training.engineId}</span>
                                <span className="text-[10px] font-mono text-white">{training.progress.toFixed(0)}%</span>
                            </div>
                            <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden border border-white/10">
                                <div className="h-full bg-gradient-to-r from-purple-600 to-cyan-400 transition-all duration-200" style={{ width: `${training.progress}%` }}></div>
                            </div>
                            <div className="h-24 bg-black/50 rounded border border-cyan-900/30 p-2 font-mono text-[8px] text-gray-300 overflow-hidden flex flex-col justify-end">
                                {training.logs.map((log, i) => (
                                    <div key={i} className="truncate animate-fade-in-right">{log}</div>
                                ))}
                            </div>
                            <div className="text-center text-[9px] text-cyan-500 animate-pulse">
                                MAPPING SYNAPTIC QUANTUM STATE...
                            </div>
                        </div>
                    ) : training.modelArtifact ? (
                         <div className="space-y-2 animate-fade-in text-center p-2 bg-green-900/20 border border-green-500/30 rounded">
                            <CheckCircle2Icon className="w-8 h-8 text-green-400 mx-auto mb-1" />
                            <h5 className="text-sm font-bold text-white">Neuro-Model Created</h5>
                            <p className="text-xs text-green-300 font-mono">{training.modelArtifact.name}</p>
                            <div className="flex justify-center gap-4 text-[9px] text-gray-300 mt-1">
                                <span>Acc: {training.modelArtifact.accuracy}</span>
                                <span>{training.modelArtifact.parameters}</span>
                            </div>
                            <button onClick={() => setTraining(p => ({...p, modelArtifact: null}))} className="mt-2 text-[10px] text-cyan-400 hover:text-white underline">Dismiss</button>
                         </div>
                    ) : (
                        <div className="text-center p-4 text-gray-500 text-[10px] italic border border-dashed border-gray-700 rounded">
                            Select an engine node to initialize Neuro Science Training.
                        </div>
                    )}
                </div>
            </div>

            {/* --- 5. 5 Quantum Cognition Engines (Outer Orbit) --- */}
            {ENGINES.map((eng, i) => {
                const angle = (i / 5) * Math.PI * 2 - Math.PI / 2;
                const radius = 42;
                const top = 50 + Math.sin(angle) * radius;
                const left = 50 + Math.cos(angle) * radius;
                
                // Get Evolution Stage from Context
                const stage = qceState.currentStage[eng.id as keyof typeof qceState.currentStage] || 4;
                const progress = qceState.evolutionProgress[eng.id as keyof typeof qceState.evolutionProgress] || 0;
                const isTraining = training.active && training.engineId === eng.id;

                return (
                    <button 
                        key={eng.id}
                        onClick={() => !training.active && startCueTraining(eng.id)}
                        disabled={training.active}
                        className={`absolute w-24 flex flex-col items-center bg-black/80 backdrop-blur-lg rounded-lg border ${isTraining ? 'border-white shadow-[0_0_20px_white] scale-110 z-50' : `${eng.borderColor} shadow-[0_0_20px_rgba(0,0,0,0.5)] z-20`} p-2 transition-all duration-300 hover:scale-110 group`}
                        style={{ top: `${top}%`, left: `${left}%`, transform: 'translate(-50%, -50%)' }}
                    >
                        <div className="flex items-center gap-2 mb-1">
                            <eng.icon className={`w-4 h-4 ${isTraining ? 'text-white animate-spin' : eng.color}`} />
                            <span className={`text-[10px] font-black ${isTraining ? 'text-white' : eng.color}`}>{eng.label}</span>
                        </div>
                        <div className="w-full h-1 bg-gray-800 rounded-full overflow-hidden mb-1">
                            <div className={`h-full ${eng.color.replace('text', 'bg')}`} style={{ width: `${progress}%` }}></div>
                        </div>
                        <div className="flex justify-between w-full text-[7px] font-mono text-gray-400">
                            <span>{isTraining ? 'LEARNING' : `OPT: ${Math.floor(progress)}%`}</span>
                            <span className="text-white font-bold">v{stage}.0</span>
                        </div>
                        
                        {/* Tooltip Description */}
                        <div className="absolute top-full mt-2 w-32 bg-black/90 text-white text-[8px] p-2 rounded border border-gray-700 opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50">
                            {eng.description}
                        </div>

                        {stage >= 5 && (
                            <div className="absolute -top-2 -right-2 w-4 h-4 bg-yellow-500 rounded-full flex items-center justify-center animate-bounce shadow-lg">
                                <span className="text-[8px] text-black font-bold">5</span>
                            </div>
                        )}
                    </button>
                );
            })}
            
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
                        title={isSimulating ? "Stop Cued Simulation" : "Start Cued Simulation"}
                     >
                        {isSimulating ? <StopIcon className="w-3 h-3" /> : <PlayIcon className="w-3 h-3" />}
                     </button>
                 </div>
            </div>
        </div>
    );
};

export default DistributedCognitiveArchitecture;
