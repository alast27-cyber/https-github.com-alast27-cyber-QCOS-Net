import React, { useState, useEffect } from 'react';
import { CpuChipIcon, ActivityIcon, ZapIcon, LoaderIcon } from './Icons';

interface QuantumSystemSimulatorProps {
    embedded?: boolean;
}

const QuantumSystemSimulator: React.FC<QuantumSystemSimulatorProps> = ({ embedded }) => {
    const [status, setStatus] = useState<'IDLE' | 'SIMULATING' | 'COMPLETE'>('IDLE');
    const [progress, setProgress] = useState(0);

    const startSimulation = () => {
        setStatus('SIMULATING');
        setProgress(0);
    };

    useEffect(() => {
        if (status === 'SIMULATING') {
            const interval = setInterval(() => {
                setProgress(prev => {
                    if (prev >= 100) {
                        clearInterval(interval);
                        setStatus('COMPLETE');
                        return 100;
                    }
                    return prev + 2;
                });
            }, 100);
            return () => clearInterval(interval);
        }
    }, [status]);

    return (
        <div className={`h-full flex flex-col ${embedded ? 'p-0 bg-transparent border-none' : 'p-4 bg-black/40 rounded-lg border border-cyan-900/30'} overflow-hidden relative`}>
            {!embedded && (
                <div className="flex justify-between items-center mb-4 z-10">
                    <div className="flex items-center gap-2">
                        <CpuChipIcon className="w-4 h-4 text-cyan-400" />
                        <span className="text-xs font-bold uppercase tracking-widest text-cyan-200">System Simulator</span>
                    </div>
                    <div className="text-[10px] font-mono text-cyan-500">
                        MODE: <span className="text-white">FULL_STATE_VECTOR</span>
                    </div>
                </div>
            )}

            <div className="flex-grow flex flex-col gap-4 min-h-0">
                <div className="bg-black/30 border border-cyan-900/20 rounded-lg p-4 flex flex-col items-center justify-center relative overflow-hidden flex-grow">
                    {status === 'IDLE' && (
                        <button 
                            onClick={startSimulation}
                            className="px-6 py-2 bg-cyan-600/20 border border-cyan-500 rounded-full text-xs font-bold text-cyan-300 hover:bg-cyan-500/20 transition-all"
                        >
                            INITIALIZE SYSTEM SIM
                        </button>
                    )}
                    {status === 'SIMULATING' && (
                        <div className="flex flex-col items-center gap-4">
                            <LoaderIcon className="w-12 h-12 text-cyan-400 animate-spin" />
                            <div className="w-48 h-2 bg-cyan-900/30 rounded-full overflow-hidden border border-cyan-800/30">
                                <div className="h-full bg-cyan-500 transition-all duration-300" style={{ width: `${progress}%` }}></div>
                            </div>
                            <p className="text-[10px] font-mono text-cyan-500 uppercase tracking-widest">Processing Hilbert Space...</p>
                        </div>
                    )}
                    {status === 'COMPLETE' && (
                        <div className="flex flex-col items-center gap-4 animate-fade-in">
                            <ZapIcon className="w-12 h-12 text-yellow-400" />
                            <p className="text-xs font-bold text-green-400 uppercase tracking-widest">Simulation Converged</p>
                            <button 
                                onClick={() => setStatus('IDLE')}
                                className="text-[10px] text-cyan-500 hover:text-cyan-300 underline"
                            >
                                Reset Parameters
                            </button>
                        </div>
                    )}
                </div>

                <div className="grid grid-cols-2 gap-2">
                    <div className="bg-black/40 border border-cyan-900/20 p-2 rounded">
                        <p className="text-[8px] text-cyan-600 font-bold uppercase">Complexity</p>
                        <p className="text-xs font-mono text-cyan-200">2^50 States</p>
                    </div>
                    <div className="bg-black/40 border border-cyan-900/20 p-2 rounded">
                        <p className="text-[8px] text-cyan-600 font-bold uppercase">Memory Usage</p>
                        <p className="text-xs font-mono text-cyan-200">1.2 TB (Virtual)</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default QuantumSystemSimulator;
