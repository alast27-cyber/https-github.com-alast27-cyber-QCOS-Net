
import React from 'react';
import { CpuChipIcon, ActivityIcon, ServerCogIcon, GalaxyIcon, TerminalIcon } from './Icons';
import { useSimulation } from '../context/SimulationContext';

const IAIKernelStatus: React.FC<{ isRecalibrating: boolean }> = ({ isRecalibrating }) => {
    const { systemStatus, universeConnections, injectApp } = useSimulation();

    return (
        <div className="w-full p-4 flex flex-col gap-4">
            <div className="flex items-center justify-center mb-2 relative">
                {universeConnections.kernel && (
                    <div className="absolute -top-4 left-1/2 -translate-x-1/2 text-[8px] bg-blue-900/80 text-blue-200 px-2 py-0.5 rounded border border-blue-500 animate-pulse flex items-center gap-1 z-10 whitespace-nowrap">
                        <GalaxyIcon className="w-2.5 h-2.5" /> UNIVERSE ACCELERATION
                    </div>
                )}
                <div className={`w-24 h-24 rounded-full border-4 flex items-center justify-center relative ${isRecalibrating ? 'border-yellow-500 animate-pulse' : (universeConnections.kernel ? 'border-blue-400 shadow-[0_0_40px_blue]' : 'border-cyan-500 shadow-[0_0_30px_cyan]')}`}>
                    <CpuChipIcon className={`w-12 h-12 ${universeConnections.kernel ? 'text-white drop-shadow-[0_0_10px_white]' : 'text-white'}`} />
                    <div className={`absolute inset-0 border-t-2 border-transparent ${universeConnections.kernel ? 'border-t-blue-300 animate-spin-fast' : 'border-t-white animate-spin'} rounded-full`}></div>
                </div>
            </div>
            
            <div className="text-center">
                <h3 className="text-lg font-bold text-white uppercase tracking-widest">QCOS Kernel v4.2</h3>
                <p className={`text-xs font-mono mt-1 ${isRecalibrating ? 'text-yellow-400' : 'text-green-400'}`}>
                    {isRecalibrating ? 'RECALIBRATING...' : 'OPERATIONAL'}
                </p>
            </div>

            <div className="grid grid-cols-2 gap-3 mt-2">
                <div className="bg-black/40 p-2 rounded border border-cyan-900/50 flex flex-col items-center">
                    <ServerCogIcon className="w-4 h-4 text-purple-400 mb-1" />
                    <span className="text-[10px] text-gray-400 uppercase">Threads</span>
                    <span className="text-sm font-mono text-white">{universeConnections.kernel ? '∞' : systemStatus.activeThreads}</span>
                </div>
                <div className="bg-black/40 p-2 rounded border border-cyan-900/50 flex flex-col items-center">
                    <ActivityIcon className="w-4 h-4 text-blue-400 mb-1" />
                    <span className="text-[10px] text-gray-400 uppercase">IPS Load</span>
                    <span className="text-sm font-mono text-white">{systemStatus.ipsThroughput} T/s</span>
                </div>
            </div>

            <button 
                onClick={() => injectApp('qos-kernel-manager')}
                className="w-full mt-2 py-2 holographic-button bg-cyan-900/30 border border-cyan-500 text-cyan-300 text-[10px] font-black uppercase rounded flex items-center justify-center gap-2 hover:bg-cyan-900/50"
            >
                <TerminalIcon className="w-3.5 h-3.5" /> Launch Kernel Console
            </button>
        </div>
    );
};

export default IAIKernelStatus;
