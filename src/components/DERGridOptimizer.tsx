
import React, { useState, useEffect, useRef } from 'react';
import GlassPanel from './GlassPanel';
import { 
    ZapIcon, TerminalIcon, ServerCogIcon, CheckCircle2Icon, 
    ActivityIcon, CpuChipIcon, RefreshCwIcon, AlertTriangleIcon,
    GitBranchIcon, Share2Icon
} from './Icons';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, AreaChart, Area } from 'recharts';

// --- Types ---
type DeployStep = 0 | 1 | 2 | 3 | 4 | 5;

interface LogLine {
    text: string;
    type: 'cmd' | 'info' | 'success' | 'warn';
    id: number;
}

interface GridNode {
    id: string;
    type: 'SOLAR' | 'WIND' | 'STORAGE' | 'CONSUMER';
    x: number;
    y: number;
    load: number; // 0-100
    status: 'OPTIMAL' | 'STRESSED' | 'CRITICAL';
}

const INITIAL_LOGS: LogLine[] = [
    { id: 1, type: 'info', text: 'Initializing QCOS DER Deployment Sequence...' }
];

const GRID_NODES: GridNode[] = [
    { id: 'PV-01', type: 'SOLAR', x: 20, y: 30, load: 45, status: 'OPTIMAL' },
    { id: 'WT-04', type: 'WIND', x: 80, y: 20, load: 78, status: 'OPTIMAL' },
    { id: 'BATT-A', type: 'STORAGE', x: 50, y: 50, load: 12, status: 'OPTIMAL' },
    { id: 'SUB-99', type: 'CONSUMER', x: 20, y: 80, load: 85, status: 'STRESSED' },
    { id: 'IND-X', type: 'CONSUMER', x: 80, y: 80, load: 92, status: 'CRITICAL' },
];

const DERGridOptimizer: React.FC = () => {
    // --- State ---
    const [step, setStep] = useState<DeployStep>(0);
    const [logs, setLogs] = useState<LogLine[]>(INITIAL_LOGS);
    const [activeView, setActiveView] = useState<'TERMINAL' | 'DASHBOARD'>('TERMINAL');
    const [gridMetrics, setGridMetrics] = useState({ frequency: 60.00, voltage: 120.0, fidelity: 0.999 });
    const [telemetryHistory, setTelemetryHistory] = useState<{time: number, freq: number, load: number}[]>([]);
    
    const logsEndRef = useRef<HTMLDivElement>(null);

    // --- Helpers ---
    const addLog = (text: string, type: LogLine['type'] = 'info') => {
        setLogs(prev => [...prev, { id: Date.now() + Math.random(), text, type }]);
    };

    useEffect(() => {
        if (logsEndRef.current) {
            logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [logs]);

    // --- Deployment Sequence Effect ---
    useEffect(() => {
        let timeout: ReturnType<typeof setTimeout>;

        const runSequence = async () => {
            if (step === 0) {
                // Step 1: Load Module
                timeout = setTimeout(() => {
                    addLog('> qmod load --persistent mod_der_mgmt.ko', 'cmd');
                    setTimeout(() => {
                        addLog('[KERNEL] Module mod_der_mgmt loaded. Bridge established @ 0x884F.', 'success');
                        setStep(1);
                    }, 800);
                }, 1000);
            } else if (step === 1) {
                // Step 2: Config
                timeout = setTimeout(() => {
                    addLog('> reading /etc/qos/der_config.yaml', 'cmd');
                    addLog('   Resource: 12 Qubits | Coherence_Min: 100us | Proto: IEC_61850', 'info');
                    setStep(2);
                }, 1200);
            } else if (step === 2) {
                // Step 3: QVM Init
                timeout = setTimeout(() => {
                    addLog('> qvm create --name der_runtime --config /etc/qos/der_config.yaml', 'cmd');
                    setTimeout(() => {
                        addLog('[QVM] Provisioning isolated execution environment...', 'info');
                        addLog('[QVM] der_runtime initialized. PID: 4421', 'success');
                        setStep(3);
                    }, 1500);
                }, 1000);
            } else if (step === 3) {
                // Step 4: Deploy Circuit
                timeout = setTimeout(() => {
                    addLog('> qexec deploy --src ./der_grid_optimizer.qasm --target der_runtime', 'cmd');
                    setTimeout(() => {
                        addLog('[Q-SCHEDULER] Circuit injected into quantum stream.', 'info');
                        addLog('[Q-SCHEDULER] Optimization topology mapped.', 'success');
                        setStep(4);
                    }, 1500);
                }, 1000);
            } else if (step === 4) {
                // Step 5: Verify
                timeout = setTimeout(() => {
                    addLog('> qstat -v der_runtime', 'cmd');
                    addLog('   Status: ACTIVE', 'success');
                    addLog('   Entanglement_Link: STABLE', 'success');
                    addLog('   Fidelity: 0.9992', 'success');
                    setTimeout(() => {
                        addLog('>>> LAUNCHING DASHBOARD INTERFACE <<<', 'warn');
                        setTimeout(() => setActiveView('DASHBOARD'), 1500);
                    }, 1000);
                }, 1000);
            }
        };

        runSequence();
        return () => clearTimeout(timeout);
    }, [step]);

    // --- Live Data Simulation ---
    useEffect(() => {
        if (activeView !== 'DASHBOARD') return;

        const interval = setInterval(() => {
            setGridMetrics(prev => ({
                frequency: 60 + (Math.random() - 0.5) * 0.05,
                voltage: 120 + (Math.random() - 0.5) * 1.5,
                fidelity: Math.min(0.9999, Math.max(0.95, prev.fidelity + (Math.random() - 0.5) * 0.001))
            }));

            setTelemetryHistory(prev => {
                const newPt = {
                    time: Date.now(),
                    freq: 60 + (Math.random() - 0.5) * 0.1,
                    load: 45 + Math.random() * 10
                };
                return [...prev.slice(-30), newPt];
            });
        }, 1000);

        return () => clearInterval(interval);
    }, [activeView]);

    // --- Render Methods ---

    const renderTerminal = () => (
        <div className="h-full bg-black/80 font-mono text-xs p-4 overflow-y-auto custom-scrollbar rounded-lg border border-cyan-900/50 shadow-inner">
            {logs.map(log => (
                <div key={log.id} className={`mb-1 ${
                    log.type === 'cmd' ? 'text-cyan-400' : 
                    log.type === 'success' ? 'text-green-400' : 
                    log.type === 'warn' ? 'text-yellow-400 font-bold' : 
                    'text-gray-300'
                }`}>
                    {log.type === 'cmd' ? '$ ' : log.type === 'success' ? '✔ ' : ''}{log.text}
                </div>
            ))}
            <div ref={logsEndRef} />
            {step < 5 && <div className="animate-pulse text-cyan-500">_</div>}
        </div>
    );

    const renderDashboard = () => (
        <div className="h-full flex flex-col gap-4 animate-fade-in">
            {/* Header Telemetry */}
            <div className="flex justify-between items-center bg-black/40 p-3 rounded-lg border border-cyan-800/50">
                <div className="flex gap-6">
                    <div className="flex flex-col">
                        <span className="text-[10px] text-gray-400 uppercase font-bold">Grid Frequency</span>
                        <span className="text-xl font-mono text-white font-bold">{gridMetrics.frequency.toFixed(3)} Hz</span>
                    </div>
                    <div className="flex flex-col">
                        <span className="text-[10px] text-gray-400 uppercase font-bold">Voltage</span>
                        <span className="text-xl font-mono text-cyan-300 font-bold">{gridMetrics.voltage.toFixed(1)} kV</span>
                    </div>
                    <div className="flex flex-col">
                        <span className="text-[10px] text-gray-400 uppercase font-bold">Q-Fidelity</span>
                        <span className={`text-xl font-mono font-bold ${gridMetrics.fidelity > 0.98 ? 'text-green-400' : 'text-yellow-400'}`}>
                            {(gridMetrics.fidelity * 100).toFixed(2)}%
                        </span>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    <div className="px-3 py-1 rounded bg-green-900/30 border border-green-500/50 text-green-300 text-xs font-bold flex items-center gap-2">
                        <ActivityIcon className="w-4 h-4 animate-pulse" /> OPTIMIZER ACTIVE
                    </div>
                    <div className="px-3 py-1 rounded bg-purple-900/30 border border-purple-500/50 text-purple-300 text-xs font-bold flex items-center gap-2">
                        <Share2Icon className="w-4 h-4" /> EKS STABLE
                    </div>
                </div>
            </div>

            {/* Main Content Split */}
            <div className="flex-grow grid grid-cols-1 lg:grid-cols-3 gap-4 min-h-0">
                
                {/* Left: Grid Visualization */}
                <div className="lg:col-span-2 bg-black/40 rounded-lg border border-cyan-900/50 relative overflow-hidden flex flex-col">
                    <div className="absolute top-2 left-2 z-10 bg-black/60 px-2 py-1 rounded text-[10px] text-cyan-500 font-bold border border-cyan-900">
                        IEC 61850 TOPOLOGY MAP
                    </div>
                    
                    <div className="flex-grow relative">
                        {/* Background Grid */}
                        <div className="absolute inset-0 holographic-grid opacity-20"></div>
                        
                        {/* Power Lines (SVG) */}
                        <svg className="absolute inset-0 w-full h-full pointer-events-none">
                             {/* Central Hub Lines */}
                             <line x1="50%" y1="50%" x2="20%" y2="30%" stroke="rgba(34, 211, 238, 0.3)" strokeWidth="2" />
                             <line x1="50%" y1="50%" x2="80%" y2="20%" stroke="rgba(34, 211, 238, 0.3)" strokeWidth="2" />
                             <line x1="50%" y1="50%" x2="20%" y2="80%" stroke="rgba(239, 68, 68, 0.4)" strokeWidth="2" strokeDasharray="5,5" className="animate-pulse" />
                             <line x1="50%" y1="50%" x2="80%" y2="80%" stroke="rgba(239, 68, 68, 0.4)" strokeWidth="2" strokeDasharray="5,5" className="animate-pulse" />
                             
                             {/* Power Particles */}
                             <circle r="3" fill="#fff" className="animate-flow-path">
                                 <animateMotion dur="2s" repeatCount="indefinite" path="M 50% 50% L 20% 80%" />
                             </circle>
                             <circle r="3" fill="#22d3ee" className="animate-flow-path" style={{animationDelay: '1s'}}>
                                 <animateMotion dur="3s" repeatCount="indefinite" path="M 20% 30% L 50% 50%" />
                             </circle>
                        </svg>

                        {/* Nodes */}
                        {GRID_NODES.map(node => (
                            <div 
                                key={node.id}
                                className={`absolute w-16 h-16 -ml-8 -mt-8 flex flex-col items-center justify-center rounded-full border-2 bg-black/80 backdrop-blur-sm transition-all duration-500 z-20 ${
                                    node.status === 'OPTIMAL' ? 'border-green-500 shadow-[0_0_20px_rgba(34,197,94,0.3)]' : 
                                    node.status === 'STRESSED' ? 'border-yellow-500 shadow-[0_0_20px_rgba(234,179,8,0.3)]' : 
                                    'border-red-500 shadow-[0_0_30px_rgba(239,68,68,0.5)] animate-pulse'
                                }`}
                                style={{ left: `${node.x}%`, top: `${node.y}%` }}
                            >
                                <div className="text-white mb-1">
                                    {node.type === 'SOLAR' && <ZapIcon className="w-6 h-6 text-yellow-300" />}
                                    {node.type === 'WIND' && <ActivityIcon className="w-6 h-6 text-blue-300" />}
                                    {node.type === 'STORAGE' && <ServerCogIcon className="w-6 h-6 text-purple-300" />}
                                    {node.type === 'CONSUMER' && <CpuChipIcon className="w-6 h-6 text-white" />}
                                </div>
                                <span className="text-[9px] font-bold text-gray-300">{node.id}</span>
                                <div className="absolute -bottom-6 bg-black/80 px-2 py-0.5 rounded text-[8px] font-mono border border-gray-700">
                                    {node.load}% LOAD
                                </div>
                            </div>
                        ))}

                        {/* Center Hub */}
                        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-24 h-24 rounded-full border-4 border-cyan-500/50 flex items-center justify-center bg-black/50 z-10">
                            <div className="text-center">
                                <CpuChipIcon className="w-8 h-8 text-cyan-400 mx-auto mb-1 animate-pulse" />
                                <span className="text-[9px] font-black text-white block">Q-CORE</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Right: Controls & Quantum State */}
                <div className="col-span-1 flex flex-col gap-4">
                    
                    {/* Quantum Register State */}
                    <div className="bg-black/30 border border-purple-900/50 rounded-lg p-4 flex-shrink-0">
                        <h4 className="text-xs font-bold text-purple-300 mb-3 uppercase tracking-widest flex items-center gap-2">
                            <GitBranchIcon className="w-4 h-4" /> Optimization Circuit
                        </h4>
                        <div className="grid grid-cols-4 gap-2 mb-3">
                            {Array.from({length: 12}).map((_, i) => (
                                <div key={i} className="flex flex-col items-center gap-1">
                                    <div className="w-full h-8 bg-gray-900 rounded relative overflow-hidden border border-purple-900/30">
                                        <div 
                                            className="absolute bottom-0 left-0 w-full bg-purple-500 transition-all duration-300"
                                            style={{ height: `${Math.random() * 100}%`, opacity: 0.7 }}
                                        ></div>
                                    </div>
                                    <span className="text-[7px] font-mono text-gray-500">q{i}</span>
                                </div>
                            ))}
                        </div>
                        <div className="text-[9px] text-gray-400 font-mono text-center">
                            State Vector: |Ψ⟩ = α|0⟩ + β|1⟩ (Superposition Active)
                        </div>
                    </div>

                    {/* Load Graph */}
                    <div className="bg-black/30 border border-cyan-900/50 rounded-lg p-2 flex-grow min-h-[150px] flex flex-col">
                        <h4 className="text-[10px] font-bold text-cyan-500 uppercase tracking-widest mb-2 px-2">Grid Frequency Trend</h4>
                        <div className="flex-grow relative">
                             <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={telemetryHistory}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                    <XAxis dataKey="time" hide />
                                    <YAxis domain={['auto', 'auto']} hide />
                                    <Tooltip contentStyle={{backgroundColor: '#000', fontSize: '10px'}} itemStyle={{color: '#fff'}} />
                                    <Line type="monotone" dataKey="freq" stroke="#22d3ee" strokeWidth={2} dot={false} isAnimationActive={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Manual Controls */}
                    <div className="flex flex-col gap-2 mt-auto">
                        <button className="holographic-button py-3 bg-red-900/20 border-red-500/50 text-red-200 text-xs font-bold rounded flex items-center justify-center gap-2 hover:bg-red-900/40">
                             <AlertTriangleIcon className="w-4 h-4" /> Inject Load Anomaly
                        </button>
                        <div className="grid grid-cols-2 gap-2">
                             <button className="holographic-button py-2 bg-yellow-900/20 border-yellow-500/50 text-yellow-200 text-[10px] font-bold rounded hover:bg-yellow-900/40">
                                 PID Fallback
                             </button>
                             <button className="holographic-button py-2 bg-green-900/20 border-green-500/50 text-green-200 text-[10px] font-bold rounded hover:bg-green-900/40">
                                 <RefreshCwIcon className="w-3 h-3 inline mr-1" /> Re-Optimize
                             </button>
                        </div>
                    </div>

                </div>
            </div>
        </div>
    );

    return (
        <GlassPanel title={
            <div className="flex items-center">
                <ZapIcon className="w-5 h-5 mr-2 text-yellow-400" />
                <span>DER Grid Optimizer</span>
            </div>
        }>
            {activeView === 'TERMINAL' ? renderTerminal() : renderDashboard()}
        </GlassPanel>
    );
};

export default DERGridOptimizer;
