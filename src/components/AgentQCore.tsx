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
            <span className={`text-xs font-bold ${color}`}>{value}{unit}</span>
            {trend === 'rising' && <ArrowTrendingUpIcon className="w-2.5 h-2.5 text-green-400" />}
        </div>
    </div>
);

const AgentQCore: React.FC<AgentQCoreProps> = ({ 
    systemHealth, 
    isRecalibrating, 
    isUpgrading, 
    activeDataStreams,
    onMaximizeSubPanel 
}) => {
    const simulation = useSimulation();
    
    // Safety Fallbacks
    const training = simulation?.training ?? { isActive: false, logs: [], coherence: 0, loss: 0, epoch: 0 };
    const evolution = simulation?.evolution ?? { isActive: false, logs: [] };
    const toggleTraining = simulation?.toggleTraining ?? (() => {});
    const toggleEvolution = simulation?.toggleEvolution ?? (() => {});

    return (
        <div className="h-full grid grid-rows-2 gap-4 p-1 min-h-0 overflow-hidden">
            {/* Top Section: Training & System Status */}
            <div className="grid grid-cols-12 gap-4 min-h-0">
                <div className="col-span-12 lg:col-span-7 h-full">
                    <GlassPanel 
                        onMaximize={() => onMaximizeSubPanel?.('qiai-kernel-status')}
                        title={
                            <div className="flex items-center gap-2">
                                <CpuChipIcon className="w-3.5 h-3.5 text-cyan-400" />
                                <span className="text-[10px]">Neural Core Status</span>
                            </div>
                        }
                    >
                        <div className="h-full flex flex-col gap-4">
                            {/* Training Controls */}
                            <div className="bg-cyan-950/20 p-3 rounded-lg border border-cyan-500/20">
                                <div className="flex justify-between items-center mb-3">
                                    <div className="flex items-center gap-2">
                                        <div className={`w-2 h-2 rounded-full ${training.isActive ? 'bg-green-400 animate-pulse' : 'bg-gray-600'}`} />
                                        <span className="text-[10px] font-bold text-cyan-100">AUTONOMOUS_LEARNING</span>
                                    </div>
                                    <button 
                                        onClick={toggleTraining}
                                        className={`px-3 py-1 rounded text-[9px] font-black transition-all ${training.isActive ? 'bg-red-500/20 text-red-400 border border-red-500/40' : 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40'}`}
                                    >
                                        {training.isActive ? 'ABORT_TRAINING' : 'INIT_TRAINING'}
                                    </button>
                                </div>
                                
                                <div className="grid grid-cols-3 gap-2">
                                    <div className="text-center p-2 bg-black/40 rounded border border-cyan-900/30">
                                        <div className="text-[8px] text-gray-500 uppercase">Coherence</div>
                                        <div className="text-sm font-bold text-cyan-400">{(training.coherence ?? 0).toFixed(1)}%</div>
                                    </div>
                                    <div className="text-center p-2 bg-black/40 rounded border border-cyan-900/30">
                                        <div className="text-[8px] text-gray-500 uppercase">Loss Rate</div>
                                        <div className="text-sm font-bold text-purple-400">{(training.loss ?? 0).toFixed(4)}</div>
                                    </div>
                                    <div className="text-center p-2 bg-black/40 rounded border border-cyan-900/30">
                                        <div className="text-[8px] text-gray-500 uppercase">Epoch</div>
                                        <div className="text-sm font-bold text-white">{training.epoch ?? 0}</div>
                                    </div>
                                </div>
                            </div>

                            {/* Logs Section */}
                            <div className="flex-grow min-h-0 bg-black/40 rounded-lg border border-cyan-900/30 p-2 font-mono text-[9px] overflow-hidden flex flex-col">
                                <div className="flex items-center gap-2 mb-1 text-gray-500 border-b border-cyan-900/20 pb-1">
                                    <TerminalIcon className="w-3 h-3" />
                                    <span>LIVE_COGNITION_LOGS</span>
                                </div>
                                <div className="flex-grow overflow-y-auto custom-scrollbar space-y-1">
                                    {/* Safe Map Implementation */}
                                    {(training.logs ?? []).length > 0 ? (
                                        (training.logs ?? []).slice(-10).map((log, i) => (
                                            <div key={`train-log-${i}`} className="text-cyan-400/80">
                                                <span className="text-cyan-700 mr-2">[{new Date().toLocaleTimeString()}]</span>
                                                {log}
                                            </div>
                                        ))
                                    ) : (
                                        <div className="text-gray-600 italic">Waiting for neural link...</div>
                                    )}
                                </div>
                            </div>
                        </div>
                    </GlassPanel>
                </div>

                <GlassPanel 
                    className="col-span-12 lg:col-span-5"
                    title={
                        <div className="flex items-center gap-2">
                            <ActivityIcon className="w-3.5 h-3.5 text-green-400" />
                            <span className="text-[10px]">Vital Metrics</span>
                        </div>
                    }
                >
                    <div className="h-full flex flex-col justify-between">
                        <div className="space-y-2">
                            <TelemetryRow 
                                name="IPS Throughput" 
                                value={systemHealth?.ipsThroughput ?? 0} 
                                unit=" Th/s" 
                                trend="rising" 
                                color="text-cyan-400" 
                            />
                            <TelemetryRow 
                                name="Neural Load" 
                                value={systemHealth?.neuralLoad ?? 0} 
                                unit="%" 
                                trend="stable" 
                                color={(systemHealth?.neuralLoad ?? 0) > 80 ? 'text-red-400' : 'text-green-400'} 
                            />
                            <TelemetryRow 
                                name="Active Threads" 
                                value={systemHealth?.activeThreads ?? 0} 
                                unit="" 
                                trend="rising" 
                                color="text-purple-400" 
                            />
                        </div>

                        <div className="mt-4 pt-4 border-t border-cyan-900/30">
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-[9px] text-gray-500 uppercase tracking-widest">Evolution Progress</span>
                                <span className="text-[10px] text-cyan-400 font-bold">{evolution.isActive ? 'EVOLVING' : 'STANDBY'}</span>
                            </div>
                            <div className="w-full h-1 bg-cyan-900/30 rounded-full overflow-hidden">
                                <div 
                                    className="h-full bg-gradient-to-r from-cyan-500 to-purple-500 transition-all duration-1000"
                                    style={{ width: `${evolution.isActive ? 100 : 0}%` }}
                                />
                            </div>
                            <div className="mt-3 flex gap-2">
                                <button 
                                    onClick={toggleEvolution}
                                    className="flex-grow py-2 bg-purple-500/10 hover:bg-purple-500/20 border border-purple-500/30 rounded text-[9px] font-bold text-purple-300 transition-all"
                                >
                                    {evolution.isActive ? 'STOP_EVOLUTION' : 'START_EVOLUTION'}
                                </button>
                            </div>
                        </div>
                    </div>
                </GlassPanel>
            </div>

            {/* Bottom Section: Mesh Topology */}
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

const TerminalIcon = ({ className }: { className?: string }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>
);

export default AgentQCore;