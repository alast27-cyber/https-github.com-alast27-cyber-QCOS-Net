import React, { useState, useEffect } from 'react';
import { ActivityIcon, ZapIcon } from './Icons';

const QuantumSpeedometer: React.FC = () => {
    const [speed, setSpeed] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setSpeed(prev => {
                const target = 85 + Math.random() * 10;
                return prev + (target - prev) * 0.1;
            });
        }, 100);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="h-full flex flex-col p-4 bg-black/40 rounded-lg border border-cyan-900/30 overflow-hidden relative">
            <div className="flex justify-between items-center mb-4 z-10">
                <div className="flex items-center gap-2">
                    <ActivityIcon className="w-4 h-4 text-cyan-400" />
                    <span className="text-xs font-bold uppercase tracking-widest text-cyan-200">Quantum Velocity</span>
                </div>
                <div className="text-[10px] font-mono text-cyan-500">
                    STATUS: <span className="text-white">SUPERLUMINAL</span>
                </div>
            </div>

            <div className="flex-grow flex flex-col items-center justify-center relative">
                <div className="relative w-48 h-48 flex items-center justify-center">
                    <svg className="w-full h-full transform -rotate-90">
                        <circle 
                            cx="96" cy="96" r="80" 
                            fill="none" stroke="rgba(6, 182, 212, 0.1)" strokeWidth="8" 
                        />
                        <circle 
                            cx="96" cy="96" r="80" 
                            fill="none" stroke="url(#speedGrad)" strokeWidth="8" 
                            strokeDasharray="502.4"
                            strokeDashoffset={502.4 * (1 - speed / 100)}
                            className="transition-all duration-300 ease-out"
                        />
                        <defs>
                            <linearGradient id="speedGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stopColor="#06b6d4" />
                                <stop offset="100%" stopColor="#22d3ee" />
                            </linearGradient>
                        </defs>
                    </svg>
                    
                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <span className="text-4xl font-black text-white drop-shadow-[0_0_10px_rgba(34,211,238,0.5)]">
                            {speed.toFixed(1)}
                        </span>
                        <span className="text-[10px] font-bold text-cyan-500 uppercase tracking-widest">Q-Units/s</span>
                    </div>
                </div>

                <div className="mt-4 flex gap-4">
                    <div className="flex items-center gap-1 text-[9px] font-mono text-cyan-600">
                        <ZapIcon className="w-3 h-3" />
                        PEAK: 98.2
                    </div>
                    <div className="flex items-center gap-1 text-[9px] font-mono text-cyan-600">
                        <ActivityIcon className="w-3 h-3" />
                        AVG: 87.5
                    </div>
                </div>
            </div>

            <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(circle_at_center,rgba(6,182,212,0.05)_0%,transparent_70%)]"></div>
        </div>
    );
};

export default QuantumSpeedometer;
