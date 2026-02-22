
import React, { useState, useEffect } from 'react';
import { ActivityIcon, ZapIcon, LayersIcon, CpuChipIcon } from './Icons';

interface QuantumExecutionFlowProps {
    ipsThroughput: number;
}

const QuantumExecutionFlow: React.FC<QuantumExecutionFlowProps> = ({ ipsThroughput }) => {
    const [pulses, setPulses] = useState<{ id: number; x: number; color: string }[]>([]);

    useEffect(() => {
        const interval = setInterval(() => {
            setPulses(prev => [
                ...prev.map(p => ({ ...p, x: p.x + 2 })),
                { id: Date.now(), x: 0, color: Math.random() > 0.5 ? '#22d3ee' : '#a855f7' }
            ].filter(p => p.x < 100));
        }, 200);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="h-full flex flex-col p-4 bg-black/40 rounded-lg border border-cyan-900/30 overflow-hidden">
            <div className="flex justify-between items-center mb-4">
                <div className="flex items-center gap-2">
                    <ActivityIcon className="w-4 h-4 text-cyan-400" />
                    <span className="text-xs font-bold uppercase tracking-widest text-cyan-200">Execution Pipeline</span>
                </div>
                <div className="text-[10px] font-mono text-cyan-500">
                    THROUGHPUT: <span className="text-white">{ipsThroughput.toFixed(0)} IPS</span>
                </div>
            </div>

            <div className="flex-grow relative flex flex-col justify-around py-4">
                {[0, 1, 2].map(row => (
                    <div key={row} className="h-8 w-full bg-cyan-950/20 rounded-full relative overflow-hidden border border-cyan-900/20">
                        <div className="absolute inset-0 flex items-center justify-between px-4 opacity-20">
                            <LayersIcon className="w-3 h-3" />
                            <ZapIcon className="w-3 h-3" />
                            <CpuChipIcon className="w-3 h-3" />
                        </div>
                        {pulses.filter((_, i) => i % 3 === row).map(p => (
                            <div 
                                key={p.id}
                                className="absolute top-1/2 -translate-y-1/2 w-4 h-1 rounded-full blur-[2px] animate-pulse"
                                style={{ 
                                    left: `${p.x}%`, 
                                    backgroundColor: p.color,
                                    boxShadow: `0 0 10px ${p.color}`
                                }}
                            />
                        ))}
                    </div>
                ))}
            </div>

            <div className="mt-4 grid grid-cols-3 gap-2">
                {['L1_CACHE', 'Q_BUFFER', 'V_GRADIENT'].map((label, idx) => (
                    <div key={label} className="bg-black/40 border border-cyan-900/20 p-2 rounded text-center">
                        <p className="text-[8px] text-cyan-600 font-bold">{label}</p>
                        <div className="h-1 w-full bg-cyan-900/30 rounded-full mt-1 overflow-hidden">
                            <div className="h-full bg-cyan-500 animate-pulse" style={{ width: `${40 + (idx * 15) % 40}%` }}></div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default QuantumExecutionFlow;
