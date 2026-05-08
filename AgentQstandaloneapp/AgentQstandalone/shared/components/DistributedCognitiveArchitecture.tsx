
import React, { useEffect, useRef, useState } from 'react';
import { NetworkIcon, GalaxyIcon, BrainCircuitIcon, AtomIcon, SparklesIcon, RocketLaunchIcon, CodeBracketIcon, StopIcon, PlayIcon, AcademicCapIcon, CheckCircle2Icon, ActivityIcon } from './Icons';
import { useSimulation } from '../context/SimulationContext';
import { useToast } from '../context/ToastContext';

interface EngineConfig {
    id: string;
    label: string;
    icon: any;
    color: string;
    borderColor: string;
    description: string;
    simSteps: string[];
}

const ENGINES: EngineConfig[] = [
    { id: 'QLLM', label: 'QLLM', icon: CodeBracketIcon, color: 'text-purple-400', borderColor: 'border-purple-500', description: 'Semantic Probability Tree', simSteps: ["Mapping Weights...", "Synthesizing Thought..."] },
    { id: 'QML', label: 'QCA', icon: BrainCircuitIcon, color: 'text-cyan-400', borderColor: 'border-cyan-500', description: 'Quantum Cognitive Architecture', simSteps: ["Loading Memory...", "Consolidating..."] },
    { id: 'QRL', label: 'QRL', icon: RocketLaunchIcon, color: 'text-orange-400', borderColor: 'border-orange-500', description: 'Policy Optimization', simSteps: ["Encoding State...", "Action Choice..."] },
];

const DistributedCognitiveArchitecture: React.FC<{ activeDataStreams?: string[] }> = ({ activeDataStreams }) => {
    const { qceState } = useSimulation();
    const { addToast } = useToast();
    const [isSimulating, setIsSimulating] = useState(true);

    return (
        <div className="w-full h-full relative bg-black/20 overflow-hidden flex items-center justify-center p-4">
            <div className="grid grid-cols-1 gap-4 w-full">
                {ENGINES.map((eng) => {
                    const progress = qceState.evolutionProgress[eng.id as keyof typeof qceState.evolutionProgress] || 0;
                    return (
                        <div key={eng.id} className={`bg-black/60 p-3 rounded-lg border ${eng.borderColor} flex flex-col gap-2`}>
                            <div className="flex items-center gap-2">
                                <eng.icon className={`w-4 h-4 ${eng.color}`} />
                                <span className={`text-xs font-bold ${eng.color}`}>{eng.label}</span>
                            </div>
                            <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
                                <div className={`h-full ${eng.color.replace('text', 'bg')}`} style={{ width: `${progress}%` }}></div>
                            </div>
                            <p className="text-[10px] text-gray-400 italic">{eng.description}</p>
                        </div>
                    );
                })}
            </div>
            
            <div className="absolute bottom-2 right-2">
                <button onClick={() => setIsSimulating(!isSimulating)} className={`p-1 rounded border ${isSimulating ? 'text-red-400 border-red-400' : 'text-green-400 border-green-400'}`}>
                    {isSimulating ? <StopIcon className="w-4 h-4" /> : <PlayIcon className="w-4 h-4" />}
                </button>
            </div>
        </div>
    );
};

export default DistributedCognitiveArchitecture;
