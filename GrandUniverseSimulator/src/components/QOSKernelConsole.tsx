
import React, { useState, useEffect, useRef } from 'react';
import GlassPanel from './GlassPanel';
import { 
    TerminalIcon, CpuChipIcon, ActivityIcon, LayersIcon, 
    PlayIcon, StopIcon, AlertTriangleIcon, CheckCircle2Icon,
    GitBranchIcon, ZapIcon, ServerStackIcon, GlobeIcon, CodeBracketIcon
} from './Icons';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';

// --- Types Mirroring Python Framework ---
interface QuantumJob {
    id: string;
    priority: number; // 0 (High) to 10 (Low)
    circuit: string;
    minFidelity: number;
    status: 'QUEUED' | 'OPTIMIZING' | 'EXECUTING' | 'MITIGATING' | 'COMPLETE' | 'FAILED';
}

interface LogEntry {
    id: number;
    timestamp: string;
    level: 'INFO' | 'WARNING' | 'ERROR';
    module: 'KERNEL' | 'SCHEDULER' | 'HAL' | 'OPTIMIZER' | 'MITIGATION' | 'API';
    message: string;
}

interface HealthResponse {
    status: string;
    timestamp: string;
    kernel: {
        state: string;
        uptime_ticks: number;
    };
    scheduler: {
        queue_depth: number;
        status: string;
    };
    hal: {
        global_fidelity: number;
        qubits_online: number;
    };
}

const QOSKernelConsole: React.FC = () => {
    const [isRunning, setIsRunning] = useState(false);
    const [uptimeTicks, setUptimeTicks] = useState(0);
    const [queue, setQueue] = useState<QuantumJob[]>([]);
    const [activeJob, setActiveJob] = useState<QuantumJob | null>(null);
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [backendFidelity, setBackendFidelity] = useState(0.96);
    const [qubitHealth, setQubitHealth] = useState<number[]>(Array(8).fill(0.98));
    
    // API Simulation State
    const [healthResponse, setHealthResponse] = useState<HealthResponse | null>(null);
    const [isApiLoading, setIsApiLoading] = useState(false);
    
    const logEndRef = useRef<HTMLDivElement>(null);

    // --- Helpers ---
    const addLog = (module: LogEntry['module'], level: LogEntry['level'], message: string) => {
        setLogs(prev => [...prev.slice(-49), {
            id: Date.now() + Math.random(),
            timestamp: new Date().toLocaleTimeString(),
            module,
            level,
            message
        }]);
    };

    useEffect(() => {
        if (logEndRef.current) logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    const handleHealthCheck = () => {
        setIsApiLoading(true);
        addLog('API', 'INFO', 'Incoming Request: GET /health (External Source)');
        
        setTimeout(() => {
            const sysStatus = isRunning 
                ? (backendFidelity > 0.9 ? 'OPERATIONAL' : 'DEGRADED') 
                : 'OFFLINE';

            const response: HealthResponse = {
                status: sysStatus,
                timestamp: new Date().toISOString(),
                kernel: {
                    state: isRunning ? 'RUNNING' : 'STOPPED',
                    uptime_ticks: uptimeTicks
                },
                scheduler: {
                    queue_depth: queue.length,
                    status: queue.length > 5 ? 'CONGESTED' : 'IDLE'
                },
                hal: {
                    global_fidelity: parseFloat(backendFidelity.toFixed(4)),
                    qubits_online: 8
                }
            };
            setHealthResponse(response);
            setIsApiLoading(false);
            addLog('API', 'INFO', `Response 200 OK: System ${sysStatus}`);
        }, 600);
    };

    // --- Simulation Loop (Mirrors Python asyncio loop) ---
    useEffect(() => {
        if (!isRunning) return;

        const interval = setInterval(() => {
            setUptimeTicks(t => t + 1);

            // 1. Randomly Fluctuate Hardware Health (HAL Simulation)
            setQubitHealth(prev => prev.map(q => Math.max(0.7, Math.min(1.0, q + (Math.random() - 0.5) * 0.02))));
            const avgFid = qubitHealth.reduce((a,b)=>a+b,0) / qubitHealth.length;
            setBackendFidelity(avgFid);

            // 2. Incoming Job Generator (User Simulation)
            if (Math.random() > 0.85 && queue.length < 8) {
                const prio = Math.floor(Math.random() * 10);
                const newJob: QuantumJob = {
                    id: `JOB-${Math.random().toString(36).substr(2, 6).toUpperCase()}`,
                    priority: prio,
                    circuit: prio < 3 ? 'Grover_Search_Oracle' : 'Bell_State_Prep',
                    minFidelity: prio < 3 ? 0.95 : 0.85,
                    status: 'QUEUED'
                };
                setQueue(prev => [...prev, newJob].sort((a, b) => a.priority - b.priority));
                addLog('SCHEDULER', 'INFO', `Job ${newJob.id} queued with priority ${newJob.priority}`);
            }

            // 3. Kernel Execution Logic
            if (!activeJob && queue.length > 0) {
                // Peek next job
                const nextJob = queue[0];

                // Fidelity Check (Qubit Health)
                if (avgFid < nextJob.minFidelity) {
                    addLog('SCHEDULER', 'WARNING', `Backend fidelity (${avgFid.toFixed(3)}) < Req (${nextJob.minFidelity}). Re-queueing ${nextJob.id}.`);
                    // Move to back of queue to prevent blocking (simple round robin fallback)
                    setQueue(prev => {
                        const [head, ...tail] = prev;
                        return [...tail, head];
                    });
                } else {
                    // Start Processing
                    setQueue(prev => prev.slice(1));
                    setActiveJob({ ...nextJob, status: 'OPTIMIZING' });
                    addLog('KERNEL', 'INFO', `Processing Job ${nextJob.id}`);
                }
            } else if (activeJob) {
                // Progression State Machine
                switch (activeJob.status) {
                    case 'OPTIMIZING':
                        addLog('OPTIMIZER', 'INFO', 'Running Lookahead Swap Mapper...');
                        setTimeout(() => setActiveJob(j => j ? { ...j, status: 'EXECUTING' } : null), 800);
                        break;
                    case 'EXECUTING':
                        addLog('HAL', 'INFO', 'Submitting circuit to Qiskit backend...');
                        setTimeout(() => setActiveJob(j => j ? { ...j, status: 'MITIGATING' } : null), 1200);
                        break;
                    case 'MITIGATING':
                        addLog('MITIGATION', 'INFO', 'Applying ZNE error suppression...');
                        setTimeout(() => setActiveJob(j => j ? { ...j, status: 'COMPLETE' } : null), 800);
                        break;
                    case 'COMPLETE':
                        addLog('KERNEL', 'INFO', `Job ${activeJob.id} Complete. Counts: { "00": 512, "11": 512 }`);
                        setActiveJob(null);
                        break;
                }
            }

        }, 1000);

        return () => clearInterval(interval);
    }, [isRunning, activeJob, queue, qubitHealth]);

    return (
        <GlassPanel title={<div className="flex items-center"><TerminalIcon className="w-5 h-5 mr-2 text-yellow-400" /> QOS Kernel Console</div>}>
            <div className="h-full flex flex-col gap-4 p-4 font-mono text-xs">
                
                {/* Top Control Bar */}
                <div className="flex justify-between items-center bg-black/40 p-3 rounded-lg border border-yellow-900/50">
                    <div className="flex items-center gap-4">
                        <div className="flex flex-col">
                            <span className="text-[10px] text-gray-500 uppercase font-bold">Kernel State</span>
                            <span className={`text-sm font-bold ${isRunning ? 'text-green-400' : 'text-red-400'}`}>
                                {isRunning ? 'ACTIVE (EVENT LOOP)' : 'HALTED'}
                            </span>
                        </div>
                        <div className="h-8 w-px bg-gray-800"></div>
                        <div className="flex flex-col">
                            <span className="text-[10px] text-gray-500 uppercase font-bold">Scheduler</span>
                            <span className="text-sm text-cyan-300">{queue.length} Pending</span>
                        </div>
                    </div>
                    <button 
                        onClick={() => setIsRunning(!isRunning)}
                        className={`px-6 py-2 rounded font-bold flex items-center gap-2 transition-all ${isRunning ? 'bg-red-900/30 border border-red-600 text-red-200 hover:bg-red-900/50' : 'bg-green-900/30 border border-green-600 text-green-200 hover:bg-green-900/50'}`}
                    >
                        {isRunning ? <StopIcon className="w-4 h-4" /> : <PlayIcon className="w-4 h-4" />}
                        {isRunning ? 'Stop Kernel' : 'Boot Kernel'}
                    </button>
                </div>

                <div className="flex-grow grid grid-cols-1 lg:grid-cols-3 gap-4 min-h-0">
                    
                    {/* Left: Priority Scheduler */}
                    <div className="bg-black/30 border border-cyan-900/30 rounded-lg p-3 flex flex-col">
                        <h4 className="text-[10px] font-bold text-cyan-500 uppercase tracking-widest mb-3 flex items-center gap-2">
                            <LayersIcon className="w-4 h-4" /> Priority Scheduler
                        </h4>
                        
                        {/* Active Job Slot */}
                        <div className="mb-4 bg-cyan-950/20 border border-cyan-500/30 p-3 rounded-lg relative overflow-hidden">
                            <p className="text-[9px] text-cyan-400 uppercase mb-1">Active Context</p>
                            {activeJob ? (
                                <div>
                                    <div className="flex justify-between items-center mb-1">
                                        <span className="text-white font-bold">{activeJob.id}</span>
                                        <span className="text-yellow-400">PRI: {activeJob.priority}</span>
                                    </div>
                                    <p className="text-gray-400 mb-2 truncate">{activeJob.circuit}</p>
                                    <div className="flex items-center gap-2">
                                        <div className="flex-grow h-1.5 bg-gray-800 rounded-full overflow-hidden">
                                            <div className={`h-full bg-green-500 transition-all duration-500 ${
                                                activeJob.status === 'OPTIMIZING' ? 'w-1/4' :
                                                activeJob.status === 'EXECUTING' ? 'w-2/4' :
                                                activeJob.status === 'MITIGATING' ? 'w-3/4' : 'w-full'
                                            }`}></div>
                                        </div>
                                        <span className="text-[8px] text-green-400">{activeJob.status}</span>
                                    </div>
                                </div>
                            ) : (
                                <div className="text-gray-600 italic text-center py-2">IDLE - Awaiting Instructions</div>
                            )}
                        </div>

                        {/* Queue */}
                        <div className="flex-grow overflow-y-auto custom-scrollbar space-y-2">
                            {queue.map(job => (
                                <div key={job.id} className="bg-black/40 border border-gray-800 p-2 rounded flex justify-between items-center group hover:border-cyan-800 transition-colors">
                                    <div>
                                        <span className="block text-gray-300 font-bold">{job.id}</span>
                                        <span className="text-[9px] text-gray-500">{job.circuit}</span>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-[9px] bg-gray-800 px-1.5 py-0.5 rounded text-gray-300 mb-1">P{job.priority}</div>
                                        <div className="text-[9px] text-red-400">Min Fid: {job.minFidelity}</div>
                                    </div>
                                </div>
                            ))}
                            {queue.length === 0 && <div className="text-center text-gray-700 mt-10">Queue Empty</div>}
                        </div>
                    </div>

                    {/* Middle: Kernel Logs & API */}
                    <div className="bg-black/30 border border-cyan-900/30 rounded-lg p-3 flex flex-col font-mono">
                        <div className="flex-grow flex flex-col min-h-0">
                            <h4 className="text-[10px] font-bold text-cyan-500 uppercase tracking-widest mb-3 flex items-center gap-2">
                                <TerminalIcon className="w-4 h-4" /> System Log
                            </h4>
                            <div className="flex-grow overflow-y-auto custom-scrollbar space-y-1 mb-2">
                                {logs.map(log => (
                                    <div key={log.id} className="flex gap-2 text-[10px] break-all">
                                        <span className="text-gray-600 flex-shrink-0">[{log.timestamp}]</span>
                                        <span className={`font-bold w-16 flex-shrink-0 ${
                                            log.module === 'KERNEL' ? 'text-purple-400' :
                                            log.module === 'HAL' ? 'text-red-400' :
                                            log.module === 'SCHEDULER' ? 'text-yellow-400' :
                                            log.module === 'API' ? 'text-green-400' :
                                            'text-blue-400'
                                        }`}>{log.module}:</span>
                                        <span className={log.level === 'WARNING' ? 'text-yellow-200' : 'text-gray-300'}>{log.message}</span>
                                    </div>
                                ))}
                                <div ref={logEndRef} />
                            </div>
                        </div>

                        {/* Simulated API Endpoint Interface */}
                        <div className="mt-2 pt-2 border-t border-cyan-800/50">
                            <div className="flex items-center justify-between mb-2">
                                <h4 className="text-[10px] font-bold text-green-400 uppercase tracking-widest flex items-center gap-2">
                                    <GlobeIcon className="w-3 h-3" /> External Endpoint
                                </h4>
                                <button 
                                    onClick={handleHealthCheck}
                                    disabled={isApiLoading}
                                    className="px-2 py-1 bg-green-900/20 border border-green-500/50 rounded text-[9px] text-green-300 hover:bg-green-900/40 transition-colors"
                                >
                                    {isApiLoading ? 'Pinging...' : 'GET /health'}
                                </button>
                            </div>
                            <div className="bg-black/50 p-2 rounded border border-green-900/30 h-24 overflow-y-auto custom-scrollbar">
                                {healthResponse ? (
                                    <pre className="text-[9px] text-green-300 font-mono whitespace-pre-wrap">
                                        {JSON.stringify(healthResponse, null, 2)}
                                    </pre>
                                ) : (
                                    <div className="h-full flex items-center justify-center text-gray-600 text-[10px] italic">
                                        Waiting for request...
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Right: HAL & Hardware Abstraction */}
                    <div className="bg-black/30 border border-cyan-900/30 rounded-lg p-3 flex flex-col">
                        <h4 className="text-[10px] font-bold text-cyan-500 uppercase tracking-widest mb-3 flex items-center gap-2">
                            <CpuChipIcon className="w-4 h-4" /> HAL: Hardware Health
                        </h4>
                        
                        <div className="flex justify-between items-center mb-4 bg-black/40 p-2 rounded border border-gray-800">
                            <span className="text-gray-400">Global Fidelity</span>
                            <span className={`text-xl font-bold ${backendFidelity > 0.9 ? 'text-green-400' : 'text-yellow-400'}`}>
                                {(backendFidelity * 100).toFixed(2)}%
                            </span>
                        </div>

                        <div className="flex-grow">
                             <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={qubitHealth.map((val, i) => ({ id: `q${i}`, val: val * 100 }))} layout="vertical">
                                    <XAxis type="number" domain={[0, 100]} hide />
                                    <YAxis dataKey="id" type="category" width={30} tick={{fontSize: 10, fill: '#666'}} />
                                    <Tooltip cursor={{fill: 'transparent'}} contentStyle={{backgroundColor: '#000', borderColor: '#333'}} />
                                    <Bar dataKey="val" barSize={12} radius={[0, 4, 4, 0]}>
                                        {qubitHealth.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry > 0.9 ? '#22c55e' : entry > 0.8 ? '#facc15' : '#ef4444'} />
                                        ))}
                                    </Bar>
                                </BarChart>
                             </ResponsiveContainer>
                        </div>

                        <div className="mt-4 pt-4 border-t border-gray-800">
                             <div className="flex items-center gap-2 text-[10px] text-gray-500 mb-2">
                                 <GitBranchIcon className="w-3 h-3" /> 
                                 <span>Topology: Linear [0-7]</span>
                             </div>
                             <div className="flex items-center gap-2 text-[10px] text-gray-500">
                                 <ZapIcon className="w-3 h-3" /> 
                                 <span>Mitigation: ZNE + M3 Active</span>
                             </div>
                        </div>
                    </div>

                </div>
            </div>
        </GlassPanel>
    );
};

export default QOSKernelConsole;
