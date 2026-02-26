import React, { useState, useEffect, useRef } from 'react';
import { ActivityIcon, BrainCircuitIcon, CheckCircle2Icon, CpuChipIcon, DatabaseIcon, LayersIcon, LockIcon, PlayIcon, ShieldCheckIcon, StopIcon } from './Icons';

// --- Types ---
interface RoadmapStage {
    id: string;
    title: string;
    description: string;
    progress: number; // 0 to 100
    status: 'pending' | 'active' | 'completed';
    tasks: string[];
}

interface TrainingLog {
    timestamp: number;
    message: string;
    type: 'info' | 'success' | 'warning' | 'patch';
}

// --- Constants (from PDF) ---
const INITIAL_STAGES: RoadmapStage[] = [
    {
        id: 'phase-1',
        title: 'Phase 1: Core Architectural Progress (GME)',
        description: 'Mixture-of-Experts (MoE) implementation, Sparse Activation, and Expert Specialization.',
        progress: 100,
        status: 'completed',
        tasks: ['Model Scaling & Efficiency', 'Expert Specialization (S\'MoRE)', 'Multimodal Integration']
    },
    {
        id: 'phase-2',
        title: 'Phase 2: Multi-Domain Generalization',
        description: 'Scientific Reasoning, Life Sciences, and Ethical Alignment.',
        progress: 45,
        status: 'active',
        tasks: ['2.3: Scientific Reasoning (Causal Modeling)', '2.4: Life Sciences (GNNs, ABM)', '2.5: Philosophy & Alignment (Ethical Guardrails)']
    },
    {
        id: 'phase-3',
        title: 'Phase 3: Generalization, Autonomy, Refinement',
        description: 'Cross-Domain Stress Testing, Self-Improvement Loop, and Final Certification.',
        progress: 0,
        status: 'pending',
        tasks: ['3.1: Cross-Domain Stress Testing', '3.2: Self-Improvement Loop', '3.3: Final Certification']
    },
    {
        id: 'phase-4',
        title: 'Phase 4: Reality-Grounded Integration & Robotics',
        description: 'Anchoring GME reasoning in sensory-motor data and real-time physical constraints.',
        progress: 0,
        status: 'pending',
        tasks: ['4.1: Embodied Sensory Fusion (GEA)', '4.2: Sim-to-Real Transfer (Physics/Eng)', '4.3: Real-Time Causal Observation']
    },
    {
        id: 'phase-5',
        title: 'Phase 5: Multi-Agent Societal Simulations',
        description: 'Moving to a "society of GMEs" to observe emergent social, economic, and political behaviors.',
        progress: 0,
        status: 'pending',
        tasks: ['5.1: Agent-Based Macro-Modeling (ABM)', '5.2: Collaborative Expert Negotiation', '5.3: Language & Dialect Evolution']
    },
    {
        id: 'phase-6',
        title: 'Phase 6: Recursive Self-Architecting',
        description: 'Transitioning from updating weights to updating architecture (NAS, Expert Spawning).',
        progress: 0,
        status: 'pending',
        tasks: ['6.1: Neural Architecture Search (NAS)', '6.2: Expert Spawning', '6.3: Synaptic Growth Optimization']
    },
    {
        id: 'phase-7',
        title: 'Phase 7: Global-Scale Problem Solving (The "Oracle" Test)',
        description: 'Applying GME to solve "Grand Challenges" like climate change and disease modeling.',
        progress: 0,
        status: 'pending',
        tasks: ['7.1: Climate & Ecological Engineering', '7.2: Universal Disease Modeling', '7.3: Ethical Policy Synthesis']
    },
    {
        id: 'phase-8',
        title: 'Phase 8: Transcendental Reasoning & Meta-Philosophy',
        description: 'Addressing the "Hard Problem of Consciousness" and internal logic verification.',
        progress: 0,
        status: 'pending',
        tasks: ['8.1: Formal Self-Verification', '8.2: Metacognitive Intuition', '8.3: Universal Ethics Alignment']
    },
    {
        id: 'phase-9',
        title: 'Phase 9: Hardware-Software Co-Evolution',
        description: 'Transitioning to neuromorphic and quantum-secure infrastructure.',
        progress: 0,
        status: 'pending',
        tasks: ['9.1: Neuromorphic Integration', '9.2: Quantum Acceleration']
    },
    {
        id: 'phase-10',
        title: 'Phase 10: Full AGI Realization & Deployment',
        description: 'Autonomous, cross-domain mastery with human-level safety and alignment.',
        progress: 0,
        status: 'pending',
        tasks: ['10.1: Continuous AGI Loop', '10.2: The Alignment Anchor', '10.3: Final Certification (G-Score)']
    }
];

const AgiTrainingSimulationRoadmap: React.FC = () => {
    // --- State ---
    const [stages, setStages] = useState<RoadmapStage[]>(INITIAL_STAGES);
    const [isTraining, setIsTraining] = useState(true);
    const [logs, setLogs] = useState<TrainingLog[]>([]);
    const [currentTask, setCurrentTask] = useState<string>('Initializing Training Protocols...');
    
    // --- Persistence ---
    useEffect(() => {
        const savedState = localStorage.getItem('qiai_training_state');
        if (savedState) {
            try {
                const parsed = JSON.parse(savedState);
                
                // Check if we have new phases to add (Migration logic)
                if (parsed.stages && parsed.stages.length < INITIAL_STAGES.length) {
                    console.log("Migrating roadmap state: Adding new phases...");
                    const mergedStages = [
                        ...parsed.stages, 
                        ...INITIAL_STAGES.slice(parsed.stages.length)
                    ];
                    setStages(mergedStages);
                } else {
                    setStages(parsed.stages);
                }
                
                // Force training to start as per user request
                setIsTraining(true);
                setLogs(parsed.logs || []);
            } catch (e) {
                console.error("Failed to load training state", e);
                // Fallback to initial state
                setStages(INITIAL_STAGES);
                setIsTraining(true);
            }
        } else {
            // No saved state, start fresh
            setIsTraining(true);
        }
    }, []);

    useEffect(() => {
        const stateToSave = { stages, isTraining, logs: logs.slice(-50) }; // Keep last 50 logs
        localStorage.setItem('qiai_training_state', JSON.stringify(stateToSave));
    }, [stages, isTraining, logs]);

    // --- Simulation Loop ---
    useEffect(() => {
        let interval: NodeJS.Timeout;

        if (isTraining) {
            interval = setInterval(() => {
                setStages(prevStages => {
                    const newStages = [...prevStages];
                    const activeStageIndex = newStages.findIndex(s => s.status === 'active');

                    if (activeStageIndex !== -1) {
                        const activeStage = newStages[activeStageIndex];
                        
                        // Increment Progress
                        const increment = Math.random() * 0.5; // Random progress
                        let newProgress = activeStage.progress + increment;

                        // Update Task Description based on sub-progress
                        const taskIndex = Math.floor((newProgress / 100) * activeStage.tasks.length);
                        const currentTaskName = activeStage.tasks[Math.min(taskIndex, activeStage.tasks.length - 1)];
                        setCurrentTask(`Training: ${currentTaskName} (${newProgress.toFixed(1)}%)`);

                        // Stage Completion Logic
                        if (newProgress >= 100) {
                            newProgress = 100;
                            newStages[activeStageIndex].status = 'completed';
                            newStages[activeStageIndex].progress = 100;
                            
                            // Activate next stage
                            if (activeStageIndex + 1 < newStages.length) {
                                newStages[activeStageIndex + 1].status = 'active';
                                addLog(`Phase Completed: ${activeStage.title}`, 'success');
                                addLog(`Initiating: ${newStages[activeStageIndex + 1].title}`, 'info');
                            } else {
                                setIsTraining(false);
                                addLog('ALL TRAINING PHASES COMPLETE. SYSTEM OPTIMIZED.', 'success');
                            }
                            
                            // Generate "Code Patch"
                            generateCodePatch(activeStage.title);
                        } else {
                            newStages[activeStageIndex].progress = newProgress;
                            
                            // Random Log Generation
                            if (Math.random() > 0.95) {
                                generateRandomLog(activeStage.title);
                            }
                        }
                    }
                    return newStages;
                });
            }, 1000); // Update every second
        }

        return () => clearInterval(interval);
    }, [isTraining]);

    // --- Helpers ---
    const addLog = (message: string, type: TrainingLog['type'] = 'info') => {
        setLogs(prev => [...prev, { timestamp: Date.now(), message, type }].slice(-50));
    };

    const generateRandomLog = (phase: string) => {
        const messages = [
            "Optimizing synaptic weights...",
            "Pruning redundant neural pathways...",
            "Validating causal inference chain...",
            "Integrating cross-domain knowledge graph...",
            "Reducing loss function in sub-sector 7...",
            "Calibrating ethical guardrails...",
            "Simulating counter-factual scenarios..."
        ];
        const msg = messages[Math.floor(Math.random() * messages.length)];
        addLog(`[${phase}] ${msg}`, 'info');
    };

    const generateCodePatch = (phase: string) => {
        const patchName = `PATCH-${Date.now().toString().slice(-6)}-${phase.split(' ')[1]}`;
        addLog(`GENERATING SYSTEM UPDATE: ${patchName}`, 'patch');
        addLog(`Applying ${patchName} to QCOS Kernel...`, 'warning');
        // In a real system, this would trigger a file write or API call
    };

    const toggleTraining = () => setIsTraining(!isTraining);

    const resetSimulation = () => {
        setStages(INITIAL_STAGES);
        setLogs([]);
        setIsTraining(true);
        localStorage.removeItem('qiai_training_state');
    };

    // --- Render ---
    return (
        <div className="flex flex-col h-full bg-black/40 border border-cyan-500/30 rounded-xl overflow-hidden">
            {/* Header */}
            <div className="bg-cyan-950/30 p-3 border-b border-cyan-500/30 flex justify-between items-center">
                <div className="flex items-center gap-2">
                    <BrainCircuitIcon className="w-5 h-5 text-cyan-400 animate-pulse" />
                    <h3 className="text-sm font-bold text-cyan-100 uppercase tracking-widest">QIAI-IPS Training Roadmap</h3>
                </div>
                <div className="flex gap-2">
                    <button onClick={toggleTraining} className={`p-1.5 rounded border ${isTraining ? 'bg-red-500/20 border-red-500 text-red-300' : 'bg-green-500/20 border-green-500 text-green-300'}`}>
                        {isTraining ? <StopIcon className="w-4 h-4" /> : <PlayIcon className="w-4 h-4" />}
                    </button>
                    <button onClick={resetSimulation} className="p-1.5 rounded border bg-gray-700/30 border-gray-500 text-gray-300 hover:bg-gray-600/50">
                        <ActivityIcon className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {/* Main Content */}
            <div className="flex-grow flex flex-col md:flex-row overflow-hidden">
                {/* Stages List */}
                <div className="w-full md:w-3/4 p-4 overflow-y-auto grid grid-cols-1 md:grid-cols-2 gap-4">
                    {stages.map((stage) => (
                        <div key={stage.id} className={`p-4 rounded-lg border ${stage.status === 'active' ? 'bg-cyan-900/20 border-cyan-400 shadow-[0_0_15px_rgba(34,211,238,0.1)]' : 'bg-black/40 border-gray-700 opacity-80'}`}>
                            <div className="flex justify-between items-start mb-2">
                                <h4 className={`text-sm font-bold ${stage.status === 'active' ? 'text-cyan-300' : stage.status === 'completed' ? 'text-green-400' : 'text-gray-400'}`}>
                                    {stage.title}
                                </h4>
                                {stage.status === 'completed' && <CheckCircle2Icon className="w-5 h-5 text-green-500" />}
                                {stage.status === 'active' && <ActivityIcon className="w-5 h-5 text-cyan-400 animate-spin-slow" />}
                                {stage.status === 'pending' && <LockIcon className="w-4 h-4 text-gray-600" />}
                            </div>
                            <p className="text-xs text-gray-400 mb-3">{stage.description}</p>
                            
                            {/* Progress Bar */}
                            <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden mb-2">
                                <div 
                                    className={`h-full transition-all duration-500 ${stage.status === 'completed' ? 'bg-green-500' : 'bg-cyan-500'}`} 
                                    style={{ width: `${stage.progress}%` }}
                                ></div>
                            </div>
                            <div className="flex justify-between text-[10px] text-gray-500 font-mono">
                                <span>{stage.status.toUpperCase()}</span>
                                <span>{stage.progress.toFixed(1)}%</span>
                            </div>

                            {/* Active Tasks */}
                            {stage.status === 'active' && (
                                <div className="mt-3 p-2 bg-black/50 rounded border border-cyan-900/50">
                                    <div className="text-[10px] text-cyan-500 uppercase font-bold mb-1">Current Focus:</div>
                                    <div className="text-xs text-cyan-100 font-mono animate-pulse">{currentTask}</div>
                                </div>
                            )}
                        </div>
                    ))}
                </div>

                {/* Logs & Patches Panel */}
                <div className="w-full md:w-1/4 bg-black/60 border-l border-cyan-500/30 flex flex-col">
                    <div className="p-2 border-b border-cyan-900/50 text-[10px] font-bold text-cyan-500 uppercase tracking-widest flex items-center gap-2">
                        <DatabaseIcon className="w-3 h-3" /> System Logs & Patches
                    </div>
                    <div className="flex-grow overflow-y-auto p-2 space-y-1 font-mono text-[10px]">
                        {logs.slice().reverse().map((log, i) => (
                            <div key={i} className={`p-1.5 rounded border-l-2 ${
                                log.type === 'success' ? 'border-green-500 bg-green-900/10 text-green-300' :
                                log.type === 'warning' ? 'border-yellow-500 bg-yellow-900/10 text-yellow-300' :
                                log.type === 'patch' ? 'border-purple-500 bg-purple-900/20 text-purple-200' :
                                'border-cyan-500 bg-cyan-900/10 text-cyan-300'
                            }`}>
                                <span className="opacity-50 mr-2">[{new Date(log.timestamp).toLocaleTimeString()}]</span>
                                {log.message}
                            </div>
                        ))}
                    </div>
                    
                    {/* Stats Footer */}
                    <div className="p-2 border-t border-cyan-900/50 bg-black/80 text-[9px] text-gray-500 flex justify-between">
                        <span>CPU Load: {(30 + Math.random() * 40).toFixed(0)}%</span>
                        <span>Memory: {(40 + Math.random() * 20).toFixed(0)}%</span>
                        <span>Qubits: 240/240</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AgiTrainingSimulationRoadmap;
