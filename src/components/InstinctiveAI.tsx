// /src/components/InstinctiveAI.tsx
import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import GlassPanel from './GlassPanel';
import { 
    BrainCircuitIcon, CpuChipIcon, ZapIcon, ActivityIcon, 
    ShieldCheckIcon, AlertTriangleIcon, RefreshCwIcon,
    ArrowRightIcon, LayersIcon, NetworkIcon
} from './Icons';
import { useSimulation } from '../context/SimulationContext';

const InstinctiveAI: React.FC = () => {
    const { systemStatus } = useSimulation();
    
    // Local simulation of IMOS for UI demonstration
    const [energy, setEnergy] = useState(1000);
    const [status, setStatus] = useState<'STABLE' | 'DEGRADED' | 'CRITICAL'>('STABLE');
    const [activeLayer, setActiveLayer] = useState<'ILL' | 'IPS' | 'CLL'>('ILL');
    const [interrupts, setInterrupts] = useState<any[]>([]);
    
    // Topological Configuration (10 columns)
    const columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]; // Node counts per column
    
    useEffect(() => {
        const interval = setInterval(() => {
            const newInterrupt = {
                id: Date.now(),
                type: Math.random() > 0.7 ? 'CONFLICT' : 'DATA',
                value: Math.floor(Math.random() * 100),
                timestamp: new Date().toLocaleTimeString()
            };
            setInterrupts(prev => [newInterrupt, ...prev].slice(0, 5));
            
            // Energy minimization logic
            setEnergy(prev => Math.max(0, prev - (Math.random() * 2)));
            
            if (energy < 300) setStatus('CRITICAL');
            else if (energy < 600) setStatus('DEGRADED');
            else setStatus('STABLE');
            
        }, 3000);
        return () => clearInterval(interval);
    }, [energy]);

    return (
        <div className="space-y-6 p-4">
            {/* Header / Status */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <GlassPanel title="IMOS Status" className="p-4 flex items-center justify-between">
                    <div className="flex items-center justify-between w-full">
                        <div>
                            <p className="text-xs uppercase tracking-wider text-white/50">System State</p>
                            <h3 className={`text-xl font-bold ${status === 'STABLE' ? 'text-emerald-400' : status === 'DEGRADED' ? 'text-amber-400' : 'text-rose-400'}`}>
                                {status}
                            </h3>
                        </div>
                        <div className={`p-3 rounded-full ${status === 'STABLE' ? 'bg-emerald-500/20' : 'bg-rose-500/20'}`}>
                            {status === 'STABLE' ? <ShieldCheckIcon className="w-6 h-6 text-emerald-400" /> : <AlertTriangleIcon className="w-6 h-6 text-rose-400" />}
                        </div>
                    </div>
                </GlassPanel>

                <GlassPanel title="Energy Budget" className="p-4 flex items-center justify-between">
                    <div className="flex items-center justify-between w-full">
                        <div>
                            <p className="text-xs uppercase tracking-wider text-white/50">Minimization Goal</p>
                            <h3 className="text-xl font-bold text-sky-400">{energy.toFixed(0)} J</h3>
                        </div>
                        <div className="p-3 rounded-full bg-sky-500/20">
                            <ZapIcon className="w-6 h-6 text-sky-400" />
                        </div>
                    </div>
                </GlassPanel>

                <GlassPanel title="Active Threads" className="p-4 flex items-center justify-between">
                    <div className="flex items-center justify-between w-full">
                        <div>
                            <p className="text-xs uppercase tracking-wider text-white/50">Neural Load</p>
                            <h3 className="text-xl font-bold text-violet-400">{systemStatus.activeThreads}</h3>
                        </div>
                        <div className="p-3 rounded-full bg-violet-500/20">
                            <ActivityIcon className="w-6 h-6 text-violet-400" />
                        </div>
                    </div>
                </GlassPanel>
            </div>

            {/* 10-Column Topological Configuration */}
            <GlassPanel title="Topological Configuration" className="p-6">
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-lg font-bold flex items-center gap-2">
                        <NetworkIcon className="w-5 h-5 text-indigo-400" />
                        IAI Neural Network
                    </h2>
                    <div className="flex gap-2">
                        {['ILL', 'IPS', 'CLL'].map(l => (
                            <button 
                                key={l}
                                onClick={() => setActiveLayer(l as any)}
                                className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${activeLayer === l ? 'bg-indigo-500 text-white' : 'bg-white/5 text-white/50 hover:bg-white/10'}`}
                            >
                                {l}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="relative h-64 w-full flex justify-between items-center px-4">
                    {/* Central Node Line */}
                    <div className="absolute left-1/2 top-0 bottom-0 w-px bg-white/10 dashed" style={{ transform: 'translateX(-50%)' }} />
                    <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-10">
                        <div className="w-12 h-12 rounded-full bg-indigo-500/20 border border-indigo-500/50 flex items-center justify-center shadow-[0_0_20px_rgba(99,102,241,0.3)]">
                            <BrainCircuitIcon className="w-6 h-6 text-indigo-400 animate-pulse" />
                        </div>
                        <p className="text-[10px] uppercase tracking-tighter text-indigo-300 absolute -bottom-6 left-1/2 -translate-x-1/2 whitespace-nowrap">Central Node</p>
                    </div>

                    {columns.map((count, colIndex) => (
                        <div key={colIndex} className="flex flex-col gap-2 items-center z-0">
                            {Array.from({ length: count }).map((_, nodeIndex) => (
                                <motion.div 
                                    key={nodeIndex}
                                    initial={{ scale: 0 }}
                                    animate={{ scale: 1 }}
                                    className={`w-3 h-3 rounded-full border ${activeLayer === 'ILL' ? 'bg-emerald-500/20 border-emerald-500/50' : activeLayer === 'IPS' ? 'bg-sky-500/20 border-sky-500/50' : 'bg-violet-500/20 border-violet-500/50'}`}
                                />
                            ))}
                            <p className="text-[8px] text-white/30 mt-2">C{colIndex + 1}</p>
                        </div>
                    ))}
                </div>
            </GlassPanel>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Layer Details */}
                <GlassPanel title="Layer Architecture" className="p-6">
                    <h3 className="text-sm font-bold uppercase tracking-widest text-white/40 mb-4 flex items-center gap-2">
                        <LayersIcon className="w-4 h-4" />
                        Active: {activeLayer}
                    </h3>
                    
                    <div className="space-y-4">
                        {activeLayer === 'ILL' && (
                            <div className="space-y-3">
                                <div className="p-3 rounded-lg bg-white/5 border border-white/10">
                                    <h4 className="text-xs font-bold text-emerald-400 mb-1">Intuitive Learning Layer (ILL)</h4>
                                    <p className="text-[11px] text-white/60 leading-relaxed">
                                        Analogous to the peripheral nervous system. Manages real-time data ingestion and normalization.
                                        Identifies "Opposing Aspects" and "Three Contradictions".
                                    </p>
                                </div>
                                <div className="grid grid-cols-2 gap-2">
                                    <div className="p-2 rounded bg-white/5 border border-white/10 text-[10px]">
                                        <p className="text-white/40 uppercase">Logic</p>
                                        <p className="text-white/80">Dichotomous Tensor</p>
                                    </div>
                                    <div className="p-2 rounded bg-white/5 border border-white/10 text-[10px]">
                                        <p className="text-white/40 uppercase">Output</p>
                                        <p className="text-white/80">Normalized Vector</p>
                                    </div>
                                </div>
                            </div>
                        )}

                        {activeLayer === 'IPS' && (
                            <div className="space-y-3">
                                <div className="p-3 rounded-lg bg-white/5 border border-white/10">
                                    <h4 className="text-xs font-bold text-sky-400 mb-1">Instinctive Problem Solving (IPS)</h4>
                                    <p className="text-[11px] text-white/60 leading-relaxed">
                                        Heuristic-based decision-making and pattern matching. Sparsely connected network of specialized Instinct Circuits.
                                    </p>
                                </div>
                                <div className="p-2 rounded bg-white/5 border border-white/10 text-[10px]">
                                    <p className="text-white/40 uppercase">Active Circuits</p>
                                    <div className="flex gap-1 mt-1">
                                        {[1, 2, 3, 4].map(i => (
                                            <div key={i} className="w-2 h-2 rounded-full bg-sky-500 animate-pulse" />
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}

                        {activeLayer === 'CLL' && (
                            <div className="space-y-3">
                                <div className="p-3 rounded-lg bg-white/5 border border-white/10">
                                    <h4 className="text-xs font-bold text-violet-400 mb-1">Cognition Learning Layer (CLL)</h4>
                                    <p className="text-[11px] text-white/60 leading-relaxed">
                                        Seat of higher-order reasoning and long-term planning. Triggered by novelty, conflict, or negative feedback.
                                    </p>
                                </div>
                                <div className="flex items-center gap-2 text-[10px] text-violet-300">
                                    <RefreshCwIcon className="w-3 h-3 animate-spin-slow" />
                                    <span>Synthesizing new instinct circuits...</span>
                                </div>
                            </div>
                        )}
                    </div>
                </GlassPanel>

                {/* Real-time Interrupts */}
                <GlassPanel title="SIPL I/O Abstraction" className="p-6">
                    <h3 className="text-sm font-bold uppercase tracking-widest text-white/40 mb-4 flex items-center gap-2">
                        <ActivityIcon className="w-4 h-4" />
                        Interrupt Stream
                    </h3>
                    <div className="space-y-2">
                        <AnimatePresence mode="popLayout">
                            {interrupts.map(int => (
                                <motion.div 
                                    key={int.id}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: 20 }}
                                    className="flex items-center justify-between p-2 rounded bg-white/5 border border-white/10"
                                >
                                    <div className="flex items-center gap-3">
                                        <div className={`w-1.5 h-1.5 rounded-full ${int.type === 'CONFLICT' ? 'bg-rose-500' : 'bg-emerald-500'}`} />
                                        <span className="text-[10px] font-mono text-white/70">{int.timestamp}</span>
                                        <span className="text-[11px] font-medium">{int.type} detected</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className="text-[10px] text-white/40">Value: {int.value}</span>
                                        <ArrowRightIcon className="w-3 h-3 text-white/20" />
                                        <span className={`text-[10px] font-bold ${int.type === 'CONFLICT' ? 'text-rose-400' : 'text-emerald-400'}`}>
                                            {int.type === 'CONFLICT' ? 'ESCALATE' : 'IPS_MATCH'}
                                        </span>
                                    </div>
                                </motion.div>
                            ))}
                        </AnimatePresence>
                    </div>
                </GlassPanel>
            </div>
        </div>
    );
};

export default InstinctiveAI;

