
import React, { useState, useEffect } from 'react';
import GlassPanel from './GlassPanel';
import { LockIcon, ArrowRightIcon, RefreshCwIcon, ShieldCheckIcon, AlertTriangleIcon } from './Icons';

type Basis = '+' | 'x';
type Bit = 0 | 1;

const QKDSimulator: React.FC = () => {
    const [step, setStep] = useState(0);
    const [aliceBits, setAliceBits] = useState<Bit[]>([]);
    const [aliceBases, setAliceBases] = useState<Basis[]>([]);
    const [bobBases, setBobBases] = useState<Basis[]>([]);
    const [results, setResults] = useState<(Bit | null)[]>([]);
    const [finalKey, setFinalKey] = useState<string>('');
    const [eavesdropper, setEavesdropper] = useState(false);

    const generateSequence = () => {
        const length = 8;
        const bits: Bit[] = Array.from({length}, () => Math.random() > 0.5 ? 1 : 0);
        const aBases: Basis[] = Array.from({length}, () => Math.random() > 0.5 ? '+' : 'x');
        const bBases: Basis[] = Array.from({length}, () => Math.random() > 0.5 ? '+' : 'x');
        
        setAliceBits(bits);
        setAliceBases(aBases);
        setBobBases(bBases);
        setResults([]);
        setFinalKey('');
        setStep(1);
    };

    const runTransmission = () => {
        setStep(2);
        setTimeout(() => {
            const measured: (Bit | null)[] = aliceBits.map((bit, i) => {
                // If bases match, perfect transmission.
                if (aliceBases[i] === bobBases[i]) {
                    // Eavesdropper noise simulation
                    if (eavesdropper && Math.random() > 0.75) {
                        return bit === 1 ? 0 : 1; // Bit flip error
                    }
                    return bit;
                }
                // If bases mismatch, 50% chance of 0 or 1
                return Math.random() > 0.5 ? 1 : 0;
            });
            setResults(measured);
            setStep(3);
        }, 1500);
    };

    const siftKeys = () => {
        setStep(4);
        const key = aliceBits.filter((_, i) => aliceBases[i] === bobBases[i]).join('');
        setFinalKey(key);
    };

    const reset = () => {
        setStep(0);
        setAliceBits([]);
        setFinalKey('');
    };

    return (
        <GlassPanel title={<div className="flex items-center"><LockIcon className="w-5 h-5 mr-2 text-green-400" /> BB84 QKD Protocol Sim</div>}>
            <div className="flex flex-col h-full p-4 gap-4 text-sm">
                
                <div className="flex justify-between items-center bg-black/30 p-2 rounded border border-green-900/50">
                    <div className="flex items-center gap-2">
                        <span className="text-cyan-400 font-bold">Status:</span>
                        <span className="text-white">
                            {step === 0 ? 'Idle' : step === 1 ? 'Prepared' : step === 2 ? 'Transmitting...' : step === 3 ? 'Received' : 'Key Generated'}
                        </span>
                    </div>
                    <div className="flex items-center gap-2">
                        <label className="text-[10px] text-red-400 uppercase font-bold">Eavesdropper (Eve)</label>
                        <button onClick={() => setEavesdropper(!eavesdropper)} className={`w-8 h-4 rounded-full p-0.5 transition-colors ${eavesdropper ? 'bg-red-600' : 'bg-gray-700'}`}>
                            <div className={`w-3 h-3 bg-white rounded-full transition-transform ${eavesdropper ? 'translate-x-4' : ''}`}></div>
                        </button>
                    </div>
                </div>

                <div className="flex-grow flex flex-col justify-center gap-6 relative">
                    {/* Channel Visual */}
                    {step === 2 && (
                        <div className="absolute top-1/2 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 to-green-500 animate-pulse opacity-50"></div>
                    )}

                    {/* Alice */}
                    <div className="bg-blue-900/20 p-3 rounded-lg border border-blue-500/30">
                        <h4 className="text-blue-300 font-bold mb-2">Alice (Sender)</h4>
                        {step >= 1 && (
                            <div className="grid grid-cols-8 gap-1">
                                {aliceBits.map((b, i) => (
                                    <div key={i} className="flex flex-col items-center p-1 bg-black/40 rounded">
                                        <span className="text-white font-mono">{b}</span>
                                        <span className="text-[10px] text-blue-400">{aliceBases[i]}</span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    <div className="flex justify-center text-gray-500">
                        <ArrowRightIcon className={`w-6 h-6 rotate-90 md:rotate-0 ${step === 2 ? 'text-white animate-bounce' : ''}`} />
                    </div>

                    {/* Bob */}
                    <div className="bg-green-900/20 p-3 rounded-lg border border-green-500/30">
                        <h4 className="text-green-300 font-bold mb-2">Bob (Receiver)</h4>
                        {step >= 1 && (
                            <div className="grid grid-cols-8 gap-1">
                                {bobBases.map((base, i) => (
                                    <div key={i} className="flex flex-col items-center p-1 bg-black/40 rounded relative">
                                        <span className={`text-[10px] mb-1 ${step >= 3 && aliceBases[i] === base ? 'text-green-400 font-bold' : 'text-green-600'}`}>{base}</span>
                                        <span className="text-white font-mono">{step >= 3 ? results[i] : '?'}</span>
                                        {step >= 4 && aliceBases[i] !== base && (
                                            <div className="absolute inset-0 bg-black/80 flex items-center justify-center">
                                                <span className="text-red-500 text-[10px]">X</span>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>

                {step === 4 && (
                    <div className="bg-black/40 p-4 rounded-lg border border-green-500 flex flex-col items-center animate-fade-in-up">
                        <h4 className="text-green-400 font-bold mb-2 flex items-center gap-2"><ShieldCheckIcon className="w-5 h-5"/> Sifted Shared Key</h4>
                        <div className="font-mono text-xl text-white tracking-widest bg-black/60 px-4 py-2 rounded border border-green-900">
                            {finalKey}
                        </div>
                        {eavesdropper && (
                            <div className="mt-2 text-red-400 text-xs flex items-center gap-1 animate-pulse">
                                <AlertTriangleIcon className="w-3 h-3"/> High Error Rate Detected (Eve)
                            </div>
                        )}
                    </div>
                )}

                <div className="mt-auto">
                    {step === 0 && <button onClick={generateSequence} className="holographic-button w-full py-2 bg-blue-600/30 border-blue-500 text-white rounded font-bold">Initialize Protocol</button>}
                    {step === 1 && <button onClick={runTransmission} className="holographic-button w-full py-2 bg-cyan-600/30 border-cyan-500 text-white rounded font-bold">Transmit Qubits</button>}
                    {step === 3 && <button onClick={siftKeys} className="holographic-button w-full py-2 bg-purple-600/30 border-purple-500 text-white rounded font-bold">Sift Bases (Classical Channel)</button>}
                    {step === 4 && <button onClick={reset} className="holographic-button w-full py-2 bg-gray-600/30 border-gray-500 text-gray-300 rounded font-bold flex items-center justify-center gap-2"><RefreshCwIcon className="w-4 h-4"/> Reset</button>}
                </div>

            </div>
        </GlassPanel>
    );
};

export default QKDSimulator;
