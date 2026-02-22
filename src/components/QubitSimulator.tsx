import React, { useState, useEffect } from 'react';
import { AtomIcon, ActivityIcon } from './Icons';

const QubitSimulator: React.FC = () => {
    const [qubits, setQubits] = useState<{ id: number; state: number; phase: number }[]>([]);

    useEffect(() => {
        const initialQubits = Array.from({ length: 8 }).map((_, i) => ({
            id: i,
            state: Math.random(),
            phase: Math.random() * 360
        }));
        setQubits(initialQubits);

        const interval = setInterval(() => {
            setQubits(prev => prev.map(q => ({
                ...q,
                state: Math.max(0, Math.min(1, q.state + (Math.random() - 0.5) * 0.1)),
                phase: (q.phase + 5) % 360
            })));
        }, 100);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="h-full flex flex-col p-4 bg-black/40 rounded-lg border border-cyan-900/30 overflow-hidden relative">
            <div className="flex justify-between items-center mb-4 z-10">
                <div className="flex items-center gap-2">
                    <AtomIcon className="w-4 h-4 text-cyan-400" />
                    <span className="text-xs font-bold uppercase tracking-widest text-cyan-200">Qubit Array Sim</span>
                </div>
                <div className="text-[10px] font-mono text-cyan-500">
                    COHERENCE: <span className="text-white">99.8%</span>
                </div>
            </div>

            <div className="flex-grow grid grid-cols-4 gap-4 p-2">
                {qubits.map(q => (
                    <div key={q.id} className="flex flex-col items-center gap-2">
                        <div className="relative w-12 h-12 flex items-center justify-center">
                            <div 
                                className="absolute inset-0 rounded-full border border-cyan-500/20 animate-spin-slow"
                                style={{ transform: `rotate(${q.phase}deg)` }}
                            >
                                <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1.5 h-1.5 bg-cyan-400 rounded-full shadow-[0_0_8px_#22d3ee]"></div>
                            </div>
                            <div 
                                className="w-6 h-6 rounded-full bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center"
                                style={{ opacity: q.state }}
                            >
                                <div className="w-2 h-2 bg-cyan-400 rounded-full"></div>
                            </div>
                        </div>
                        <span className="text-[8px] font-mono text-cyan-600">Q{q.id}</span>
                    </div>
                ))}
            </div>

            <div className="mt-4 flex items-center justify-between text-[9px] font-mono text-cyan-500/60 border-t border-cyan-900/20 pt-2">
                <div className="flex items-center gap-1">
                    <ActivityIcon className="w-3 h-3" />
                    GATE_TIME: 12ns
                </div>
                <span>TEMP: 15mK</span>
            </div>
        </div>
    );
};

export default QubitSimulator;
