import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { TerminalIcon, BugAntIcon, ActivityIcon, RefreshCwIcon, ShieldCheckIcon, CodeIcon, ChevronRightIcon } from './Icons';

const KernelDebugger: React.FC = () => {
    const [logs, setLogs] = useState<{ id: number, time: string, msg: string, type: 'info' | 'error' | 'success' | 'debug' | 'user' }[]>([
        { id: 1, time: new Date().toLocaleTimeString(), msg: 'Kernel Debugger Initialized. AgentQ Supreme Authority Verified.', type: 'info' as const },
        { id: 2, time: new Date().toLocaleTimeString(), msg: 'Attaching to QCOS Core Process [PID: 0x1337]...', type: 'debug' as const },
        { id: 3, time: new Date().toLocaleTimeString(), msg: 'Memory Map: 0x0000 - 0xFFFF Synchronized.', type: 'success' as const },
    ]);
    const [isScanning, setIsScanning] = useState(false);
    const [command, setCommand] = useState('');
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    const addLog = (msg: string, type: 'info' | 'error' | 'success' | 'debug' | 'user' = 'info') => {
        setLogs(prev => [...prev, { id: Date.now(), time: new Date().toLocaleTimeString(), msg, type }]);
    };

    const handleCommand = (e: React.FormEvent) => {
        e.preventDefault();
        if (!command.trim()) return;

        addLog(command, 'user');
        
        const cmd = command.toLowerCase();
        if (cmd.includes('scan')) {
            setIsScanning(true);
            addLog('Initiating Deep System Scan...', 'debug');
            setTimeout(() => {
                addLog('Analyzing Architectural Integrity...', 'info');
            }, 1000);
            setTimeout(() => {
                addLog('System Restored. Architectural Fidelity: 100%.', 'success');
                setIsScanning(false);
            }, 3000);
        } else if (cmd.includes('patch')) {
            addLog('Applying Architectural Patch 0x99-B...', 'debug');
            setTimeout(() => addLog('Patch successfully deployed to QCOS Kernel.', 'success'), 1500);
        } else if (cmd.includes('help')) {
            addLog('Available commands: scan, patch, clear, status, help', 'info');
        } else if (cmd.includes('status')) {
            addLog('QCOS Kernel: STABLE | CHIPS Network: OPTIMAL | AgentQ: SUPREME', 'success');
        } else {
            addLog(`Command '${command}' recognized. Processing architectural directive...`, 'info');
        }

        setCommand('');
    };

    return (
        <div className="h-full flex flex-col bg-black/40 rounded-lg border border-red-500/20 overflow-hidden font-mono text-xs">
            <div className="p-2 border-b border-red-500/20 bg-red-500/5 flex items-center justify-between">
                <div className="flex items-center gap-2 text-red-400 font-bold">
                    <TerminalIcon className="w-4 h-4" />
                    KERNEL_DEBUGGER_V4.2
                </div>
                <div className="flex gap-2">
                    <button 
                        onClick={() => handleCommand({ preventDefault: () => {}, target: { value: 'scan' } } as any)}
                        disabled={isScanning}
                        className={`px-2 py-1 rounded border border-red-500/30 text-[10px] flex items-center gap-1 transition-colors ${isScanning ? 'opacity-50 cursor-not-allowed' : 'hover:bg-red-500/20'}`}
                    >
                        <BugAntIcon className={`w-3 h-3 ${isScanning ? 'animate-pulse' : ''}`} />
                        SCAN
                    </button>
                    <button 
                        onClick={() => setLogs([])}
                        className="px-2 py-1 rounded border border-slate-500/30 text-[10px] flex items-center gap-1 hover:bg-slate-500/20 transition-colors"
                    >
                        <RefreshCwIcon className="w-3 h-3" />
                        CLEAR
                    </button>
                </div>
            </div>

            <div 
                ref={scrollRef}
                className="flex-1 overflow-y-auto p-2 space-y-1 scrollbar-thin scrollbar-thumb-red-500/20"
            >
                {logs.map(log => (
                    <div key={log.id} className="flex gap-2 animate-in fade-in slide-in-from-left-2 duration-300">
                        <span className="text-slate-500 shrink-0">[{log.time}]</span>
                        <span className={`
                            ${log.type === 'error' ? 'text-red-400' : ''}
                            ${log.type === 'success' ? 'text-green-400' : ''}
                            ${log.type === 'debug' ? 'text-purple-400' : ''}
                            ${log.type === 'info' ? 'text-cyan-400' : ''}
                            ${log.type === 'user' ? 'text-yellow-400' : ''}
                        `}>
                            {log.type === 'debug' && '> '}
                            {log.type === 'user' && '$ '}
                            {log.msg}
                        </span>
                    </div>
                ))}
                {isScanning && (
                    <div className="flex items-center gap-2 text-red-400 animate-pulse">
                        <span className="text-slate-500">[{new Date().toLocaleTimeString()}]</span>
                        <span>DEBUG_PROCESS_ACTIVE...</span>
                    </div>
                )}
            </div>

            <form onSubmit={handleCommand} className="p-2 border-t border-red-500/20 bg-black/60 flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isScanning ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`} />
                <div className="flex-1 relative">
                    <ChevronRightIcon className="absolute left-0 top-1/2 -translate-y-1/2 w-3 h-3 text-red-500/50" />
                    <input 
                        type="text"
                        value={command}
                        onChange={(e) => setCommand(e.target.value)}
                        placeholder="Enter architectural directive..."
                        className="w-full bg-transparent border-none focus:ring-0 text-[10px] text-red-400 pl-4 placeholder:text-red-900/50"
                    />
                </div>
                <div className="flex gap-2">
                    <ShieldCheckIcon className="w-3 h-3 text-slate-500" />
                    <CodeIcon className="w-3 h-3 text-slate-500" />
                </div>
            </form>
        </div>
    );
};

export default KernelDebugger;
