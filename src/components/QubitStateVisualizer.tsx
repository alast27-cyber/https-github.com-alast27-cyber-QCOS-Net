
import React, { useState, useEffect } from 'react';

const QUBIT_COUNT = 32;

type QubitState = 'zero' | 'one' | 'superposition';

interface QubitStateVisualizerProps {
    qubitStability: number;
}

const Qubit = React.memo(({ state, isSelected }: { state: QubitState; isSelected: boolean }) => {
    // Enhanced visuals for the qubit
    const baseClasses = "w-6 h-6 rounded-full transition-all duration-500 relative flex items-center justify-center";
    
    // Inner core style
    const getCoreStyle = () => {
        switch(state) {
            case 'zero': return "bg-blue-500 shadow-[0_0_10px_theme(colors.blue.400)]";
            case 'one': return "bg-purple-600 shadow-[0_0_10px_theme(colors.purple.500)]";
            case 'superposition': return "bg-gradient-to-tr from-blue-400 to-purple-500 animate-pulse shadow-[0_0_15px_theme(colors.cyan.400)]";
        }
    };

    // Orbital ring for superposition or active state
    const ringClasses = state === 'superposition' ? "absolute inset-[-4px] border border-cyan-400/30 rounded-full animate-spin-slow" : "";

    return (
        <div className="relative group cursor-pointer p-2">
            {/* Selection Indicator */}
            {isSelected && (
                <div className="absolute inset-0 border-2 border-yellow-400 rounded-lg animate-pulse shadow-[0_0_15px_theme(colors.yellow.500/50%)]"></div>
            )}
            
            {/* The Qubit Sphere */}
            <div className={`${baseClasses} ${getCoreStyle()}`}>
                {/* Highlight/Reflection for 3D effect */}
                <div className="absolute top-1 left-1 w-2 h-2 bg-white/40 rounded-full blur-[1px]"></div>
                
                {/* State Text Overlay (Tiny) */}
                <span className="text-[8px] font-bold text-white/90 z-10 font-mono">
                    {state === 'zero' ? '|0⟩' : state === 'one' ? '|1⟩' : 'Ψ'}
                </span>
            </div>
            
            {/* Outer Rings */}
            <div className={ringClasses}></div>
        </div>
    );
});

const QubitStateVisualizer: React.FC<QubitStateVisualizerProps> = ({ qubitStability }) => {
    const [qubitStates, setQubitStates] = useState<QubitState[]>(() => 
        Array.from({ length: QUBIT_COUNT }, () => 'zero')
    );
    const [selectedQubit, setSelectedQubit] = useState<number | null>(null);

    useEffect(() => {
        const interval = setInterval(() => {
            if (selectedQubit !== null) return; // Pause updates if interacting

            setQubitStates(states => {
                const newStates = [...states];
                // Simulate noise/decoherence
                if (Math.random() > 0.6) {
                    const idx = Math.floor(Math.random() * QUBIT_COUNT);
                    const rand = Math.random();
                    newStates[idx] = rand < 0.4 ? 'zero' : rand < 0.8 ? 'one' : 'superposition';
                }
                return newStates;
            });
        }, Math.max(100, qubitStability)); // Prevent too fast updates

        return () => clearInterval(interval);
    }, [qubitStability, selectedQubit]);

    const handleQubitClick = (index: number) => {
        setSelectedQubit(prev => (prev === index ? null : index));
    };

    const handleSetState = (newState: QubitState) => {
        if (selectedQubit === null) return;
        setQubitStates(prev => {
            const next = [...prev];
            next[selectedQubit] = newState;
            return next;
        });
    };

    return (
        <div className="w-full h-full flex flex-col relative bg-black/40 rounded-lg overflow-hidden">
            <div className="p-2 border-b border-cyan-900/50 flex justify-between items-center bg-cyan-950/20">
                <h3 className="text-xs font-bold tracking-widest text-cyan-300">QUBIT REGISTER MAP</h3>
                <div className="flex gap-2 text-[10px] font-mono text-cyan-500">
                    <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-blue-500"></div>|0⟩</span>
                    <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-purple-600"></div>|1⟩</span>
                    <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-gradient-to-tr from-blue-400 to-purple-500"></div>Ψ</span>
                </div>
            </div>

            <div className="flex-grow p-4 overflow-y-auto custom-scrollbar">
                <div className="grid grid-cols-8 gap-4 justify-items-center">
                    {qubitStates.map((state, i) => (
                        <div key={i} className="flex flex-col items-center">
                            <div onClick={() => handleQubitClick(i)}>
                                <Qubit state={state} isSelected={selectedQubit === i} />
                            </div>
                            <span className={`text-[9px] font-mono mt-1 ${selectedQubit === i ? 'text-yellow-400 font-bold' : 'text-cyan-700'}`}>
                                q{i}
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Control Panel Overlay */}
            <div className={`absolute bottom-0 left-0 right-0 bg-black/90 border-t border-cyan-500/50 p-3 transition-transform duration-300 flex flex-col gap-2 z-20 ${selectedQubit !== null ? 'translate-y-0' : 'translate-y-full'}`}>
                <div className="flex justify-between items-center">
                    <span className="text-xs font-bold text-white">
                        Control: <span className="text-yellow-400 font-mono">q{selectedQubit}</span>
                    </span>
                    <button onClick={() => setSelectedQubit(null)} className="text-gray-500 hover:text-white">
                        ✕
                    </button>
                </div>
                <div className="flex gap-2 justify-center">
                    <button onClick={() => handleSetState('zero')} className="flex-1 py-1.5 bg-blue-900/40 border border-blue-500 rounded text-xs text-blue-200 hover:bg-blue-800/60 transition-colors font-mono">
                        |0⟩ Ground
                    </button>
                    <button onClick={() => handleSetState('one')} className="flex-1 py-1.5 bg-purple-900/40 border border-purple-500 rounded text-xs text-purple-200 hover:bg-purple-800/60 transition-colors font-mono">
                        |1⟩ Excited
                    </button>
                    <button onClick={() => handleSetState('superposition')} className="flex-1 py-1.5 bg-cyan-900/40 border border-cyan-500 rounded text-xs text-cyan-200 hover:bg-cyan-800/60 transition-colors font-mono">
                        H(Ψ) Superpose
                    </button>
                </div>
                <p className="text-[9px] text-center text-gray-500 italic">
                    Manual override pauses coherence drift for selected qubit.
                </p>
            </div>
        </div>
    );
};

export default QubitStateVisualizer;
