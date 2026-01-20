import React, { useState, useEffect, useRef } from 'react';
import { LogInIcon, CpuChipIcon, FileCodeIcon, CrosshairIcon, SendIcon, CheckCircle2Icon } from './Icons';

type Stage = 'idle' | 'ingestion' | 'activation' | 'compilation' | 'targeting' | 'dispatch' | 'complete';

const stages: { id: Stage, name: string, icon: React.FC<{className?: string}>, description: string }[] = [
    { id: 'ingestion', name: 'Ingestion', icon: LogInIcon, description: 'QAN API Gateway receives the Q-URI request.' },
    { id: 'activation', name: 'QSC Activation', icon: CpuChipIcon, description: 'Quantum Semantic Compiler parses the Algorithm Domain.' },
    { id: 'compilation', name: 'Compilation', icon: FileCodeIcon, description: 'QSC uses Task Reference to generate Q-Lang script.' },
    { id: 'targeting', name: 'Targeting', icon: CrosshairIcon, description: 'QSC sets TARGET_SCOPE from DQN Alias & initiates EKS.' },
    { id: 'dispatch', name: 'Dispatch', icon: SendIcon, description: 'Secured CHIPS packet is transmitted to target DQN.' },
];

const Q_URI = "CHIPS://rigel.grover.search/DB_7bit_User101";
const URI_PARTS = {
    protocol: "CHIPS://",
    dqnAlias: "rigel",
    algorithmDomain: ".grover.search",
    taskReference: "/DB_7bit_User101"
};

const QANExecutionSimulator: React.FC = () => {
    const [currentStage, setCurrentStage] = useState<Stage>('idle');
    const stageIndex = stages.findIndex(s => s.id === currentStage);
    const hasStarted = useRef(false);

    const handleDispatch = () => {
        if(currentStage === 'idle') {
            setCurrentStage('ingestion');
        }
    };

    useEffect(() => {
        if (currentStage === 'idle') {
            const delay = hasStarted.current ? 4000 : 1500; // Loop with a delay, but start faster on first load
            const timer = setTimeout(() => {
                handleDispatch();
                hasStarted.current = true;
            }, delay);
            return () => clearTimeout(timer);
        }
    }, [currentStage]);

    useEffect(() => {
        let timer: number;
        if (currentStage !== 'idle' && currentStage !== 'complete') {
            timer = window.setTimeout(() => {
                const nextStageIndex = stageIndex + 1;
                if (nextStageIndex < stages.length) {
                    setCurrentStage(stages[nextStageIndex].id);
                } else {
                    setCurrentStage('complete');
                }
            }, 1500); // Time per stage
        } else if (currentStage === 'complete') {
            timer = window.setTimeout(() => setCurrentStage('idle'), 4000);
        }
        return () => clearTimeout(timer);
    }, [currentStage, stageIndex]);
    
    return (
        <div className="h-full flex flex-col">
            <h3 className="text-cyan-300 text-sm tracking-widest text-left flex-shrink-0 mb-2">QAN RESOLUTION & EXECUTION FLOW</h3>
            
            {/* Q-URI Display */}
            <div className="bg-black/30 p-2 rounded-md border border-cyan-900 font-mono text-xs mb-4">
                <span className="text-cyan-400">{URI_PARTS.protocol}</span>
                <span className="text-yellow-300">{URI_PARTS.dqnAlias}</span>
                <span className="text-purple-400">{URI_PARTS.algorithmDomain}</span>
                <span className="text-green-400">{URI_PARTS.taskReference}</span>
            </div>

            {/* Stages Timeline */}
            <div className="flex-grow space-y-2 flex flex-col justify-around">
                {stages.map((stage, index) => {
                    const isCompleted = stageIndex >= index && currentStage !== stage.id;
                    const isActive = stage.id === currentStage;
                    const isPending = stageIndex < index;

                    return (
                        <div key={stage.id} className="flex items-start space-x-3">
                            <div className="flex flex-col items-center">
                                <div className={`w-8 h-8 rounded-full flex items-center justify-center border-2 transition-all duration-300
                                    ${isActive ? 'bg-cyan-500/30 border-cyan-400 animate-pulse animate-bloom-pulse' : isCompleted ? 'bg-green-500/30 border-green-500' : 'bg-slate-800/50 border-cyan-800'}`}>
                                    {isCompleted ? <CheckCircle2Icon className="w-5 h-5 text-green-400"/> : <stage.icon className={`w-5 h-5 ${isActive ? 'text-cyan-300' : 'text-cyan-600'}`} />}
                                </div>
                                {index < stages.length - 1 && <div className={`w-0.5 h-4 mt-1 transition-colors duration-300 ${isCompleted ? 'bg-green-500' : 'bg-cyan-800'}`} />}
                            </div>
                            <div>
                                <h4 className={`font-bold text-sm transition-colors ${isActive || isCompleted ? 'text-white' : 'text-cyan-700'}`}>{stage.name}</h4>
                                <p className={`text-xs transition-colors ${isActive ? 'text-cyan-300' : isCompleted ? 'text-gray-400' : 'text-cyan-800'}`}>{stage.description}</p>
                            </div>
                        </div>
                    );
                })}
            </div>
            
             <button 
                onClick={handleDispatch}
                disabled={currentStage !== 'idle'}
                className="mt-4 w-full bg-cyan-500/30 hover:bg-cyan-500/50 border border-cyan-500/50 text-cyan-200 font-bold py-2 px-4 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
                {currentStage === 'idle' ? 'Dispatch Q-URI Request' : currentStage === 'complete' ? 'Complete' : `Processing: ${currentStage.toUpperCase()}`}
            </button>
        </div>
    );
};

export default QANExecutionSimulator;