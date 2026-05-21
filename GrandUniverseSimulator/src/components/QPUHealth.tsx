
import React from 'react';
import GlassPanel from './GlassPanel';
import { CpuChipIcon, ThermometerIcon, ActivityIcon, ZapIcon } from './Icons';
import { SystemHealth } from '../types';
import QuantumSystemSimulator from './QuantumSystemSimulator';

interface QPUHealthProps {
    systemHealth: SystemHealth;
    onMaximize?: () => void;
}

const QPUHealth: React.FC<QPUHealthProps> = ({ systemHealth, onMaximize }) => {
    return (
        <GlassPanel 
            onMaximize={onMaximize}
            title={
                <div className="flex items-center">
                    <CpuChipIcon className="w-5 h-5 mr-2 text-green-400" /> Quantum Processing Unit Vitals
                </div>
            }
        >
            <div className="p-4 h-full flex flex-col gap-4">
                <div className="grid grid-cols-2 gap-4 flex-shrink-0">
                    <div className="bg-black/30 p-3 rounded-lg border border-cyan-800/50 flex flex-col items-center">
                        <ThermometerIcon className="w-6 h-6 text-red-400 mb-1" />
                        <p className="text-[10px] text-gray-400 uppercase">Temp</p>
                        <p className="text-xl font-mono text-white">12mK</p>
                    </div>
                    <div className="bg-black/30 p-3 rounded-lg border border-cyan-800/50 flex flex-col items-center">
                        <ActivityIcon className="w-6 h-6 text-green-400 mb-1" />
                        <p className="text-[10px] text-gray-400 uppercase">Coherence</p>
                        <p className="text-xl font-mono text-white">{(100 - (systemHealth.decoherenceFactor || 0) * 1000).toFixed(2)}us</p>
                    </div>
                </div>
                
                <div className="flex-grow bg-black/20 rounded-lg p-1 border border-cyan-900/30 flex flex-col overflow-hidden relative">
                    <div className="px-2 pt-2 pb-1 flex justify-between items-center z-10">
                        <h4 className="text-xs font-bold text-cyan-500 uppercase">Qubit Topology Load</h4>
                        <span className="text-[9px] text-gray-500 font-mono">16-Qubit Simulation</span>
                    </div>
                    <div className="flex-grow relative min-h-0">
                         {/* Integrated Simulator with Holographic Rendering in Embedded Mode */}
                         <QuantumSystemSimulator embedded={true} />
                    </div>
                </div>

                <div className="flex-shrink-0 flex items-center justify-between text-xs bg-cyan-950/30 p-2 rounded border border-cyan-800/50">
                    <div className="flex items-center gap-2">
                        <ZapIcon className="w-4 h-4 text-yellow-400" />
                        <span className="text-gray-300">Power Draw</span>
                    </div>
                    <span className="font-mono text-white">1.2 kW</span>
                </div>
            </div>
        </GlassPanel>
    );
};

export default QPUHealth;
