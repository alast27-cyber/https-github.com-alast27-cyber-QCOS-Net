import React, { useState, useEffect } from 'react';
import { ShieldCheckIcon, LockIcon, KeyIcon, ActivityIcon } from './Icons';

const QuantumProtocolSimulator: React.FC = () => {
    const [activeProtocol, setActiveProtocol] = useState('BB84');
    const [isSimulating, setIsSimulating] = useState(false);

    const [entropy, setEntropy] = useState('');

    useEffect(() => {
        setEntropy('0x' + Math.random().toString(16).slice(2, 10).toUpperCase());
    }, []);

    const protocols = [
        { id: 'BB84', name: 'BB84 QKD', security: 'ULTRA' },
        { id: 'E91', name: 'E91 Entanglement', security: 'MAX' },
        { id: 'B92', name: 'B92 Single-State', security: 'HIGH' }
    ];

    const startSim = () => {
        setIsSimulating(true);
        setEntropy('0x' + Math.random().toString(16).slice(2, 10).toUpperCase());
        setTimeout(() => setIsSimulating(false), 3000);
    };

    return (
        <div className="h-full flex flex-col p-4 bg-black/40 rounded-lg border border-green-900/30 overflow-hidden">
            <div className="flex justify-between items-center mb-4">
                <div className="flex items-center gap-2">
                    <ShieldCheckIcon className="w-4 h-4 text-green-400" />
                    <span className="text-xs font-bold uppercase tracking-widest text-green-200">Protocol Simulator</span>
                </div>
                <div className="flex items-center gap-1 text-[9px] text-green-500 font-mono">
                    <LockIcon className="w-3 h-3" />
                    ENCRYPTION: <span className="text-white">QUANTUM_SECURE</span>
                </div>
            </div>

            <div className="grid grid-cols-3 gap-2 mb-4">
                {protocols.map(p => (
                    <button 
                        key={p.id}
                        onClick={() => setActiveProtocol(p.id)}
                        className={`p-2 rounded border text-[10px] font-bold transition-all ${activeProtocol === p.id ? 'bg-green-600/20 border-green-500 text-white' : 'bg-black/40 border-green-900 text-green-600 hover:bg-green-900/20'}`}
                    >
                        {p.id}
                    </button>
                ))}
            </div>

            <div className="flex-grow bg-black/30 border border-green-900/20 rounded-lg p-4 flex flex-col items-center justify-center relative overflow-hidden">
                {isSimulating ? (
                    <div className="flex flex-col items-center gap-4 animate-pulse">
                        <KeyIcon className="w-12 h-12 text-green-400" />
                        <div className="flex flex-col items-center">
                            <p className="text-xs font-bold text-white uppercase tracking-widest">Generating Keys...</p>
                            <p className="text-[10px] font-mono text-green-500 mt-1">Entropy: {entropy}</p>
                        </div>
                    </div>
                ) : (
                    <div className="flex flex-col items-center gap-4">
                        <ActivityIcon className="w-12 h-12 text-green-900" />
                        <button 
                            onClick={startSim}
                            className="px-6 py-2 bg-green-600/20 border border-green-500 rounded-full text-xs font-bold text-green-300 hover:bg-green-500/20 transition-all"
                        >
                            START SIMULATION
                        </button>
                    </div>
                )}
                
                <div className="absolute top-2 right-2 flex gap-1">
                    <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-ping"></div>
                    <div className="w-1.5 h-1.5 rounded-full bg-green-500"></div>
                </div>
            </div>

            <div className="mt-4 p-2 bg-black/40 border border-green-900/20 rounded text-[8px] font-mono text-green-500/60">
                <p>&gt; INITIALIZING {activeProtocol} HANDSHAKE...</p>
                <p>&gt; DETECTING EAVESDROPPERS: NONE</p>
                <p>&gt; KEY_RATE: 1.2 MB/S</p>
            </div>
        </div>
    );
};

export default QuantumProtocolSimulator;
