import React, { useState, useEffect, useRef } from 'react';
import { 
    DatabaseIcon, 
    SparklesIcon, 
    ActivityIcon, 
    FileJsonIcon, 
    ShieldCheckIcon,
    TerminalIcon,
    RefreshCwIcon,
    BrainCircuitIcon,
    XIcon
} from './Icons';
import { motion, AnimatePresence } from 'framer-motion';

interface LogEntry {
    instruction: string;
    context: string;
    response: string;
    metadata: {
        timestamp: string;
        quantum_logic_flag: boolean;
    };
}

const GoldDatasetCuration: React.FC = () => {
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [isLogging, setIsLogging] = useState(true);
    const [lastAugmentation, setLastAugmentation] = useState<string | null>(null);
    const [status, setStatus] = useState<'OPERATIONAL' | 'AUGMENTING' | 'OFFLINE'>('OPERATIONAL');
    const scrollRef = useRef<HTMLDivElement>(null);

    // Simulate incoming logs for the "Gold Dataset"
    useEffect(() => {
        if (!isLogging) return;

        const interval = setInterval(() => {
            const isQuantum = Math.random() > 0.5;
            const newEntry: LogEntry = {
                instruction: isQuantum 
                    ? "Analyze the decoherence rate of a superconducting qubit." 
                    : "Optimize the memory allocation for the QCOS kernel.",
                context: "You are AgentQ, a Clinical, high-level Technical OS Assistant...",
                response: "[STATUS: OPERATIONAL] Analysis complete. Decoherence rate is within nominal parameters (0.02% drift).",
                metadata: {
                    timestamp: new Date().toISOString(),
                    quantum_logic_flag: isQuantum
                }
            };

            setLogs(prev => [...prev, newEntry].slice(-50));
            
            // Randomly trigger augmentation simulation
            if (Math.random() > 0.8) {
                setStatus('AUGMENTING');
                setTimeout(() => {
                    setLastAugmentation(`Synthetic Expansion: Generated 3 variations for "${newEntry.instruction.substring(0, 20)}..."`);
                    setStatus('OPERATIONAL');
                }, 1500);
            }
        }, 3000);

        return () => clearInterval(interval);
    }, [isLogging]);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    return (
        <div className="h-full flex flex-col bg-slate-950 text-cyan-100 font-mono p-4 border border-cyan-500/30 rounded-lg overflow-hidden relative">
            {/* Header */}
            <div className="flex justify-between items-center mb-4 border-b border-cyan-500/20 pb-2">
                <div className="flex items-center gap-2">
                    <DatabaseIcon className="w-5 h-5 text-amber-400" />
                    <h3 className="text-sm font-bold tracking-widest uppercase">Gold Dataset Pipeline</h3>
                </div>
                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2 px-2 py-1 bg-black/40 rounded border border-cyan-500/20">
                        <div className={`w-2 h-2 rounded-full ${status === 'OPERATIONAL' ? 'bg-green-500 animate-pulse' : status === 'AUGMENTING' ? 'bg-amber-500 animate-bounce' : 'bg-red-500'}`}></div>
                        <span className="text-[10px] font-bold tracking-tighter">AGENTQ: {status}</span>
                    </div>
                    <button 
                        onClick={() => setIsLogging(!isLogging)}
                        className={`p-1 rounded hover:bg-cyan-500/20 transition-colors ${isLogging ? 'text-cyan-400' : 'text-slate-500'}`}
                    >
                        <ActivityIcon className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-3 gap-2 mb-4">
                <div className="bg-black/40 p-2 border border-cyan-500/10 rounded">
                    <div className="text-[9px] text-cyan-500/60 uppercase">Total Samples</div>
                    <div className="text-lg font-bold text-cyan-400">{logs.length * 3 + 1240}</div>
                </div>
                <div className="bg-black/40 p-2 border border-cyan-500/10 rounded">
                    <div className="text-[9px] text-cyan-500/60 uppercase">Quantum Logic</div>
                    <div className="text-lg font-bold text-amber-400">
                        {Math.floor((logs.filter(l => l.metadata.quantum_logic_flag).length / (logs.length || 1)) * 100)}%
                    </div>
                </div>
                <div className="bg-black/40 p-2 border border-cyan-500/10 rounded">
                    <div className="text-[9px] text-cyan-500/60 uppercase">Augmentation</div>
                    <div className="text-lg font-bold text-purple-400">3.0x</div>
                </div>
            </div>

            {/* Log Viewer */}
            <div 
                ref={scrollRef}
                className="flex-1 overflow-y-auto space-y-2 pr-2 custom-scrollbar text-[11px]"
            >
                <AnimatePresence initial={false}>
                    {logs.map((log, i) => (
                        <motion.div 
                            key={log.metadata.timestamp}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="p-2 bg-black/20 border-l-2 border-cyan-500/40 rounded-r"
                        >
                            <div className="flex justify-between items-start mb-1">
                                <span className="text-cyan-500/80 font-bold">INSTRUCTION:</span>
                                <span className="text-[9px] text-slate-500">{new Date(log.metadata.timestamp).toLocaleTimeString()}</span>
                            </div>
                            <div className="text-slate-300 mb-2 italic">"{log.instruction}"</div>
                            
                            <div className="flex items-center gap-2 mb-1">
                                <span className="text-amber-500/80 font-bold uppercase text-[9px]">Response:</span>
                                {log.metadata.quantum_logic_flag && (
                                    <span className="px-1 bg-amber-500/20 text-amber-400 text-[8px] rounded border border-amber-500/30 font-bold">QUANTUM_LOGIC</span>
                                )}
                            </div>
                            <div className="text-cyan-100/90 font-mono leading-tight">
                                {log.response}
                            </div>
                        </motion.div>
                    ))}
                </AnimatePresence>
            </div>

            {/* Augmentation Alert */}
            {lastAugmentation && (
                <motion.div 
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="absolute bottom-4 left-4 right-4 p-2 bg-purple-900/40 border border-purple-500/40 rounded backdrop-blur-md flex items-center gap-2"
                >
                    <SparklesIcon className="w-4 h-4 text-purple-400" />
                    <div className="text-[10px] text-purple-100 font-bold truncate">
                        {lastAugmentation}
                    </div>
                    <button onClick={() => setLastAugmentation(null)} className="ml-auto text-purple-400 hover:text-white">
                        <XIcon className="w-3 h-3" />
                    </button>
                </motion.div>
            )}

            {/* Footer / Controls */}
            <div className="mt-4 pt-2 border-t border-cyan-500/20 flex justify-between items-center">
                <div className="flex items-center gap-2 text-[10px] text-cyan-500/60">
                    <FileJsonIcon className="w-3 h-3" />
                    <span>gold_dataset.jsonl</span>
                </div>
                <div className="flex gap-2">
                    <button 
                        onClick={() => {
                            setStatus('AUGMENTING');
                            setTimeout(() => {
                                setLastAugmentation(`Synthetic Expansion: Generated 3 variations for manual validation.`);
                                setStatus('OPERATIONAL');
                            }, 2000);
                        }}
                        className="px-2 py-1 bg-purple-500/10 hover:bg-purple-500/20 border border-purple-500/30 rounded text-[10px] font-bold text-purple-400 transition-colors flex items-center gap-1"
                    >
                        <SparklesIcon className="w-3 h-3" />
                        AUGMENT
                    </button>
                    <button className="px-2 py-1 bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/30 rounded text-[10px] font-bold transition-colors flex items-center gap-1">
                        <RefreshCwIcon className="w-3 h-3" />
                        RE-SYNC
                    </button>
                    <button className="px-2 py-1 bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 rounded text-[10px] font-bold text-amber-400 transition-colors flex items-center gap-1">
                        <ShieldCheckIcon className="w-3 h-3" />
                        VALIDATE
                    </button>
                    <a 
                        href="/scripts/gold_dataset_pipeline.py" 
                        download 
                        className="px-2 py-1 bg-slate-500/10 hover:bg-slate-500/20 border border-slate-500/30 rounded text-[10px] font-bold text-slate-400 transition-colors flex items-center gap-1"
                    >
                        <TerminalIcon className="w-3 h-3" />
                        PY_SCRIPT
                    </a>
                </div>
            </div>
        </div>
    );
};

export default GoldDatasetCuration;
