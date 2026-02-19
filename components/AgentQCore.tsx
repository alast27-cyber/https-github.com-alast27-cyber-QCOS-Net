import React, { useState } from 'react';
import { 
    CpuChipIcon, SparklesIcon, ZapIcon, 
    BrainCircuitIcon, AcademicCapIcon, 
    SettingsIcon, XIcon, ToggleLeftIcon, 
    ToggleRightIcon, GalaxyIcon, BanknotesIcon, HeartIcon,
    RefreshCwIcon, PlayIcon, ActivityIcon, ArrowTrendingUpIcon,
    LockIcon, ShieldCheckIcon, EyeIcon
} from './Icons';
import { SystemHealth } from '../types';
import IAIKernelStatus from './IAIKernelStatus';
import DistributedCognitiveArchitecture from './DistributedCognitiveArchitecture';
import { useSimulation } from '../context/SimulationContext';
import GlassPanel from './GlassPanel';

interface AgentQCoreProps {
    systemHealth: SystemHealth;
    isRecalibrating: boolean;
    isUpgrading: boolean;
    activeDataStreams: string[];
    onMaximizeSubPanel?: (id: string) => void;
}

const TelemetryRow: React.FC<{ name: string; value: number; unit: string; trend: 'rising' | 'falling' | 'stable'; color: string }> = ({ name, value, unit, trend, color }) => (
    <div className="flex items-center justify-between bg-black/30 p-2 rounded border border-cyan-900/30">
        <div className="flex items-center gap-2">
            <ActivityIcon className={`w-3 h-3 ${color}`} />
            <span className="text-[10px] text-gray-400 uppercase tracking-tight">{name}</span>
        </div>
        <div className="flex items-center gap-2">
            <span className={`text-xs font-mono font-bold ${color}`}>{(value || 0).toFixed(1)}{unit}</span>
            <ArrowTrendingUpIcon className={`w-3 h-3 ${trend === 'rising' ? 'text-green-400 rotate-0' : trend === 'falling' ? 'text-red-400 rotate-180' : 'text-yellow-400 rotate-90'} transition-transform duration-500`} />
        </div>
    </div>
);

const AgentQCore: React.FC<AgentQCoreProps> = ({ systemHealth, isRecalibrating, isUpgrading, activeDataStreams, onMaximizeSubPanel }) => {
    const { 
        simConfig, setSimMode, 
        startToESimulation, startMathSimulation, 
        startEcoSimulation, startNeuroSimulation, 
        toggleAutomation, training, telemetryFeeds 
    } = useSimulation();
    const [showSettings, setShowSettings] = useState(false);

    const renderTrainingCategory = (
        title: string, 
        icon: React.ReactNode, 
        color: string, 
        options: { label: string, action: () => void }[]
    ) => (
        <div className="space-y-1.5">
            <p className={`text-[9px] uppercase font-bold flex items-center gap-1 ${color}`}>
                {icon} {title}
            </p>
            <div className="grid grid-cols-2 gap-1.5">
                {options.map((opt, i) => (
                    <button 
                        key={i}
                        onClick={() => { opt.action(); setShowSettings(false); }}
                        className="text-[9px] text-left px-1.5 py-1 bg-black/40 border border-cyan-900/50 rounded hover:bg-cyan-900/30 hover:border-cyan-500 transition-all text-cyan-100 truncate"
                    >
                        {opt.label}
                    </button>
                ))}
            </div>
        </div>
    );

    return (
        <div className="h-full grid grid-cols-1 md:grid-cols-2 gap-3 relative">
            {/* Expanded Training Forge Overlay */}
            {showSettings && (
                <div className="absolute top-10 right-2 z-50 w-72 bg-black/95 backdrop-blur-xl border border-cyan-500/50 rounded-lg p-3 shadow-2xl animate-fade-in-up">
                    <div className="flex justify-between items-center mb-3 border-b border-cyan-800 pb-2">
                        <h4 className="text-xs font-bold text-white flex items-center gap-2">
                            <SparklesIcon className="w-3.5 h-3.5 text-cyan-400" /> Recursive Forge
                        </h4>
                        <button onClick={() => setShowSettings(false)} className="text-cyan-500 hover:text-white"><XIcon className="w-4 h-4" /></button>
                    </div>
                    
                    <div className="space-y-4 max-h-[60vh] overflow-y-auto pr-2 custom-scrollbar">
                        <div className="bg-gradient-to-r from-purple-900/40 to-cyan-900/40 p-2 rounded border border-purple-500/50 mb-3 shadow-[0_0_10px_rgba(168,85,247,0.2)]">
                            <div className="flex justify-between items-center">
                                <div>
                                    <h5 className="text-[9px] font-bold text-white uppercase tracking-widest">Autonomous Engine</h5>
                                </div>
                                <button 
                                    onClick={toggleAutomation}
                                    className={`px-2 py-1 text-[9px] font-bold rounded transition-all flex items-center gap-1.5 ${training.isAutomated ? 'bg-purple-600 text-white animate-pulse' : 'bg-black/60 border border-purple-500 text-purple-300'}`}
                                >
                                    {training.isAutomated ? <RefreshCwIcon className="w-2.5 h-2.5 animate-spin" /> : <PlayIcon className="w-2.5 h-2.5" />}
                                    {training.isAutomated ? 'LOOPING' : 'START LOOP'}
                                </button>
                            </div>
                        </div>

                        {/* Navigation to Self-Evolution Panel */}
                        <button 
                            onClick={() => { onMaximizeSubPanel?.('agentq-self-evo'); setShowSettings(false); }}
                            className="w-full py-2 bg-black/60 border border-cyan-500/50 text-cyan-300 text-[10px] font-bold rounded flex items-center justify-center gap-2 hover:bg-cyan-900/40"
                        >
                            <EyeIcon className="w-3 h-3" /> View Evolution Matrix
                        </button>

                        {renderTrainingCategory(
                            "Formal Mathematics",
                            <AcademicCapIcon className="w-2.5 h-2.5" />,
                            "text-yellow-400",
                            [
                                { label: 'Topology', action: () => startMathSimulation('Topology') },
                                { label: 'Number Theory', action: () => startMathSimulation('Number_Theory') },
                                { label: 'Calculus', action: () => startMathSimulation('Calculus') },
                                { label: 'Algebra', action: () => startMathSimulation('Algebra') },
                            ]
                        )}

                        {renderTrainingCategory(
                            "Economic Predictors",
                            <BanknotesIcon className="w-2.5 h-2.5" />,
                            "text-emerald-400",
                            [
                                { label: 'Market Dynamics', action: () => startEcoSimulation('Market_Dynamics') },
                                { label: 'Game Theory', action: () => startEcoSimulation('Game_Theory') },
                                { label: 'Resource Alloc', action: () => startEcoSimulation('Resource_Alloc') },
                            ]
                        )}

                        {renderTrainingCategory(
                            "Neural Science",
                            <HeartIcon className="w-2.5 h-2.5" />,
                            "text-rose-400",
                            [
                                { label: 'Mapping', action: () => startNeuroSimulation('Synaptic_Mapping') },
                                { label: 'Connectomics', action: () => startNeuroSimulation('Connectomics') },
                                { label: 'Decoding', action: () => startNeuroSimulation('Neural_Decoding') },
                            ]
                        )}

                        <div className="space-y-2 border-t border-cyan-900/50 pt-2">
                            <button 
                                onClick={() => { startToESimulation(); setShowSettings(false); }} 
                                disabled={training.domain === 'PHYSICS' && training.isActive}
                                className="w-full holographic-button py-2 px-2 text-[10px] font-bold flex items-center justify-center gap-2 bg-purple-900/30 border-purple-500/50 text-purple-200 disabled:opacity-50"
                            >
                                <GalaxyIcon className="w-3.5 h-3.5" />
                                {training.domain === 'PHYSICS' && training.isActive ? "Synthesizing..." : "Synthesize ToE"}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Left Panel: Backend Status & Telemetry */}
            <div className="h-full min-h-0 flex flex-col gap-2">
                <GlassPanel 
                    onMaximize={() => onMaximizeSubPanel?.('qiai-kernel-status')}
                    title={
                        <div className="flex items-center justify-between w-full pr-2">
                            <div className="flex items-center gap-2">
                                 <CpuChipIcon className="w-3.5 h-3.5 text-cyan-400" />
                                 <span className="text-[10px]">Kernel & Telemetry</span>
                            </div>
                            <div className="flex items-center gap-1 text-[9px] text-green-400 bg-green-900/20 px-2 py-0.5 rounded border border-green-800 animate-pulse">
                                <LockIcon className="w-2.5 h-2.5" /> SECURE
                            </div>
                        </div>
                    }
                >
                    <div className="flex flex-col h-full relative p-1 overflow-hidden">
                        <div className="absolute top-1 right-1 z-10">
                            <button onClick={(e) => { e.stopPropagation(); setShowSettings(!showSettings); }} className="p-1 rounded transition-colors text-cyan-500 hover:text-white">
                                <SettingsIcon className="w-3.5 h-3.5" />
                            </button>
                        </div>
                        
                        {/* Upper: Kernel Status */}
                        <div className="flex-shrink-0 mb-2">
                             {training.isActive ? (
                                <div className="text-center p-2 animate-fade-in">
                                    <div className="relative mb-2">
                                        <BrainCircuitIcon className={`w-8 h-8 mx-auto relative z-10 ${
                                            training.domain === 'MATH' ? 'text-yellow-400' : 
                                            training.domain === 'ECONOMICS' ? 'text-emerald-400' : 
                                            training.domain === 'NEUROSCIENCE' ? 'text-rose-400' : 
                                            'text-purple-400'
                                        } animate-pulse-bright`} />
                                    </div>
                                    <h3 className="text-[9px] font-bold text-white uppercase">{training.domain} FORGE</h3>
                                    <div className="mt-2 w-20 h-0.5 bg-gray-800 rounded-full mx-auto overflow-hidden">
                                        <div className="h-full bg-cyan-500 animate-flow-right" style={{ width: `${training.coherence * 100}%` }}></div>
                                    </div>
                                    <button 
                                        onClick={() => onMaximizeSubPanel?.('agentq-self-evo')}
                                        className="mt-2 text-[8px] bg-purple-900/30 text-purple-200 px-2 py-1 rounded border border-purple-600 hover:bg-purple-800/50"
                                    >
                                        Open Visualizer
                                    </button>
                                </div>
                            ) : (
                                <div className="w-full flex flex-col items-center">
                                    <IAIKernelStatus isRecalibrating={false} />
                                </div>
                            )}
                        </div>

                        {/* Lower: Telemetry Stream */}
                        <div className="flex-grow flex flex-col bg-black/20 rounded-lg border border-cyan-900/30 p-2 overflow-y-auto custom-scrollbar">
                            <h4 className="text-[9px] font-bold text-cyan-500 uppercase mb-2 border-b border-cyan-800/50 pb-1 flex items-center gap-1">
                                <ActivityIcon className="w-3 h-3" /> QAI-IPS Live Stream
                            </h4>
                            <div className="space-y-1.5">
                                {telemetryFeeds.map((feed, i) => (
                                    <TelemetryRow 
                                        key={i} 
                                        name={feed.name} 
                                        value={feed.value} 
                                        unit={feed.unit}
                                        trend={feed.trend}
                                        color={i === 0 ? 'text-purple-400' : i === 1 ? 'text-green-400' : i === 2 ? 'text-blue-400' : 'text-yellow-400'} 
                                    />
                                ))}
                                <TelemetryRow 
                                    name="Channel Encryption" 
                                    value={99.9} 
                                    unit="%" 
                                    trend="stable" 
                                    color="text-green-400"
                                />
                            </div>
                        </div>
                    </div>
                </GlassPanel>
            </div>

            {/* Right Panel: Mesh */}
            <div className="h-full min-h-0">
                <GlassPanel 
                    onMaximize={() => onMaximizeSubPanel?.('qiai-cognitive-mesh')}
                    title={
                        <div className="flex items-center gap-2">
                            <BrainCircuitIcon className="w-3.5 h-3.5 text-purple-400" />
                            <span className="text-[10px]">Mesh Topology</span>
                        </div>
                    }
                >
                    <div className="h-full flex flex-col min-h-0">
                        <DistributedCognitiveArchitecture activeDataStreams={activeDataStreams} />
                    </div>
                </GlassPanel>
            </div>
        </div>
    );
};

export default AgentQCore;