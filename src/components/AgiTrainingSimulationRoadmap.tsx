import React from 'react';
import { ActivityIcon, BrainCircuitIcon, CheckCircle2Icon, DatabaseIcon, LockIcon, PlayIcon, StopIcon } from './Icons';
import { useSimulation } from '../context/SimulationContext';

const AgiTrainingSimulationRoadmap: React.FC = () => {
    const { roadmapState, toggleRoadmapTraining, resetRoadmap } = useSimulation();
    const { stages, isTraining, logs, currentTask } = roadmapState;

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
                    <button onClick={toggleRoadmapTraining} className={`p-1.5 rounded border ${isTraining ? 'bg-red-500/20 border-red-500 text-red-300' : 'bg-green-500/20 border-green-500 text-green-300'}`}>
                        {isTraining ? <StopIcon className="w-4 h-4" /> : <PlayIcon className="w-4 h-4" />}
                    </button>
                    <button onClick={resetRoadmap} className="p-1.5 rounded border bg-gray-700/30 border-gray-500 text-gray-300 hover:bg-gray-600/50">
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
