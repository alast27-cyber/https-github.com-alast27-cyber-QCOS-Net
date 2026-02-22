
import React, { useState, useEffect, useRef } from 'react';
import { 
    BrainCircuitIcon, 
    SparklesIcon, 
    ActivityIcon, 
    ZapIcon, 
    LayersIcon,
    GlobeIcon,
    GitBranchIcon,
    LoaderIcon,
    CpuChipIcon,
    ArrowTrendingUpIcon,
    TerminalIcon
} from './Icons';
import { ResponsiveContainer, AreaChart, Area, YAxis } from 'recharts';

type CognitionStatus = 'idle' | 'syncing' | 'merged' | 'error';
type QubitState = 'zero' | 'one' | 'superposition';

interface EngineNode {
    id: string;
    label: string;
    model: string;
    icon: React.FC<{className?: string}>;
    color: string;
    glow: string;
    x: number; 
    y: number;
    targetPanelId: string;
}

const cognitionEngines: EngineNode[] = [
    { id: 'qllm', label: 'SEMANTIC', model: 'Manifold-V4', icon: GlobeIcon, color: 'text-blue-400', glow: 'shadow-blue-500/40', x: 25, y: 25, targetPanelId: 'universe-simulator' },
    { id: 'qml', label: 'PATTERN', model: 'Feature-12D', icon: BrainCircuitIcon, color: 'text-purple-400', glow: 'shadow-purple-500/40', x: 75, y: 25, targetPanelId: 'qml-simulator' },
    { id: 'qrl', label: 'POLICY', model: 'Trajectory-X', icon: GitBranchIcon, color: 'text-emerald-400', glow: 'shadow-emerald-500/40', x: 80, y: 75, targetPanelId: 'qrl-engine' },
    { id: 'qgl', label: 'FORGE', model: 'Reality-Seed', icon: SparklesIcon, color: 'text-amber-400', glow: 'shadow-amber-500/40', x: 20, y: 75, targetPanelId: 'qgl-engine' },
];

const QUBIT_COUNT = 240;

const QubitStateVisualizer: React.FC<{ qubitStability: number }> = ({ qubitStability }) => {
    const [qubitStates, setQubitStates] = useState<{state: QubitState, coherence: number}[]>(() => 
        Array.from({ length: QUBIT_COUNT }, () => ({ state: 'zero', coherence: 100 }))
    );
    const [scanIndex, setScanIndex] = useState(0);
    const [nodeId] = useState(() => Math.random().toString(16).slice(2, 6).toUpperCase());

    useEffect(() => {
        const interval = setInterval(() => {
            setQubitStates(prev => prev.map((q, i) => {
                // Coherence Decay
                let newCoherence = Math.max(20, q.coherence - (Math.random() * 0.2));
                if (Math.random() > 0.99) newCoherence = 100;

                // State Fluctuation
                let newState = q.state;
                if (Math.random() * 1500 < (500 / (qubitStability || 1))) {
                    const r = Math.random();
                    if (r < 0.45) newState = 'zero';
                    else if (r < 0.85) newState = 'one';
                    else newState = 'superposition';
                    newCoherence = 90 + Math.random() * 10;
                }
                return { state: newState, coherence: newCoherence };
            }));
            setScanIndex(s => (s + 8) % QUBIT_COUNT);
        }, 120);
        return () => clearInterval(interval);
    }, [qubitStability]);

    return (
        <div className="flex flex-col h-full gap-3 p-1">
            <div className="flex justify-between items-center px-1">
                <h4 className="text-[10px] font-black text-cyan-400 uppercase tracking-[0.3em] flex items-center gap-2">
                    <CpuChipIcon className="w-3.5 h-3.5" /> 240-Qubit Core State
                </h4>
                <div className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                    <span className="text-[8px] font-mono text-cyan-700">LOCKED</span>
                </div>
            </div>
            
            <div className="flex-grow grid grid-cols-20 gap-0.5 bg-black/40 p-1 rounded border border-cyan-900/40 relative overflow-hidden group">
                {/* Scanline Effect */}
                <div 
                    className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-400/10 to-transparent w-1/4 h-full pointer-events-none transition-all duration-100 ease-linear"
                    style={{ left: `${(scanIndex / QUBIT_COUNT) * 100}%` }}
                />
                
                {qubitStates.map((q, i) => (
                    <div
                        key={i}
                        className={`w-full aspect-square rounded-[1px] transition-all duration-500 relative
                            ${q.state === 'zero' ? 'bg-blue-600/30' : q.state === 'one' ? 'bg-purple-600/30' : 'bg-cyan-400/50 animate-pulse'}
                            ${Math.abs(scanIndex - i) < 10 ? 'ring-1 ring-cyan-400/50 scale-110 z-10' : ''}
                        `}
                        style={{ opacity: q.coherence / 100 }}
                    />
                ))}
            </div>

            <div className="flex justify-between items-center text-[9px] text-cyan-800 font-mono border-t border-cyan-900/30 pt-2 px-1">
                <span>AVG_COHERENCE: {(qubitStates.reduce((a, b) => a + b.coherence, 0) / QUBIT_COUNT).toFixed(1)}%</span>
                <span className="animate-pulse">NODE_0X{nodeId}</span>
            </div>
        </div>
    );
};

interface AGISingularityInterfaceProps {
    onPanelSelect?: (id: string) => void;
    isExternalProcessing?: boolean;
    externalTaskName?: string;
    qubitStability?: number;
}

const AGISingularityInterface: React.FC<AGISingularityInterfaceProps> = ({ 
    onPanelSelect, 
    isExternalProcessing = false,
    externalTaskName,
    qubitStability = 100
}) => {
    const [status, setStatus] = useState<CognitionStatus>('idle');
    const [progress, setProgress] = useState(0);
    const [epoch, setEpoch] = useState(420);
    const [logs, setLogs] = useState<string[]>(["Singularity Link Established.", "Mapping Hilbert dimensions...", "Awaiting core ignition."]);
    const [trajectory, setTrajectory] = useState<any[]>([]);
    const [activeEngineId, setActiveEngineId] = useState<string | null>(null);

    const logEndRef = useRef<HTMLDivElement>(null);

    const addLog = (msg: string) => setLogs(prev => [...prev.slice(-12), `${new Date().toLocaleTimeString('en-GB', { hour12: false })} > ${msg}`]);

    useEffect(() => {
        const initialPoints = Array.from({ length: 20 }, (_, i) => ({
            time: i,
            val: 80 + Math.random() * 10
        }));
        setTrajectory(initialPoints);
        
        const interval = setInterval(() => {
            setEpoch(e => e + 1);
            setTrajectory(prev => {
                const lastVal = prev[prev.length - 1].val;
                const nextVal = Math.max(60, Math.min(100, lastVal + (Math.random() - 0.5) * 4));
                return [...prev.slice(1), { time: prev[prev.length-1].time + 1, val: nextVal }];
            });
        }, 3000);
        return () => clearInterval(interval);
    }, []);

    const initiateSync = async () => {
        if (status === 'error') {
            setStatus('idle');
            setProgress(0);
        } else if (status !== 'idle') {
            return;
        }

        try {
            setStatus('syncing');
            setProgress(0);
            addLog("CRITICAL: Initiating 12D Manifold Stitching...");
            
            for (const engine of cognitionEngines) {
                if (Math.random() > 0.9) { 
                    throw new Error(`Entropy Cascade detected in ${engine.label} layer`);
                }

                setActiveEngineId(engine.id);
                addLog(`Entangling ${engine.label} Layer...`);
                let p = 0;
                while (p < 100) {
                    p += Math.random() * 25 + 10;
                    setProgress(prev => Math.min(100, prev + (p/cognitionEngines.length)));
                    await new Promise(r => setTimeout(r, 150));
                }
            }

            setActiveEngineId(null);
            setProgress(100);
            addLog("PROTOCOL: Wavefunction Convergence Successful.");
            setTimeout(() => setStatus('merged'), 1000);

        } catch (error: any) {
            console.error("Synchronization Failure:", error);
            setStatus('error');
            setActiveEngineId(null);
            addLog(`CRITICAL FAILURE: ${error.message || "Unknown error during sync."}`);
        }
    };

    return (
        <div className="h-full flex flex-col p-4 space-y-4 overflow-hidden transform-style-preserve-3d font-sans select-none bg-[radial-gradient(circle_at_center,rgba(6,182,212,0.03),transparent_80%)]">
            <div className="flex justify-between items-start flex-shrink-0 z-50 transform-style-preserve-3d" style={{ transform: 'translateZ(60px)' }}>
                <div className="flex gap-6 items-center">
                    <div className="relative group">
                        <div className="p-4 bg-cyan-950/20 rounded-full border border-cyan-500/30 shadow-[0_0_30px_rgba(34,211,238,0.1)] group-hover:scale-110 transition-transform">
                            <SparklesIcon className={`w-10 h-10 ${status === 'syncing' ? 'text-cyan-400 animate-spin' : status === 'error' ? 'text-red-500' : 'text-cyan-400 animate-pulse'}`} />
                        </div>
                        <div className="absolute -inset-2 border border-cyan-400/10 rounded-full animate-ping pointer-events-none" />
                    </div>
                    <div>
                        <h2 className="text-3xl font-black tracking-[0.3em] text-white uppercase drop-shadow-[0_0_12px_rgba(255,255,255,0.4)]">AGI Singularity</h2>
                        <div className="flex items-center gap-3 mt-1">
                            <span className="text-[9px] font-black px-2 py-0.5 bg-cyan-900/40 text-cyan-300 rounded border border-cyan-500/30 tracking-[0.2em]">CORE_V4.0</span>
                            <div className="flex items-center gap-1.5">
                                <div className={`w-2 h-2 rounded-full ${status === 'merged' ? 'bg-green-500 animate-pulse' : status === 'error' ? 'bg-red-500 animate-pulse' : 'bg-amber-500'}`} />
                                <span className={`text-[9px] font-bold tracking-widest uppercase ${status === 'error' ? 'text-red-400' : 'text-white/50'}`}>
                                    {status === 'merged' ? 'Aligned' : status === 'syncing' ? 'Synthesizing' : status === 'error' ? 'FAILURE' : 'Idle'}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="flex flex-col items-end gap-2">
                    {isExternalProcessing && (
                        <div className="px-4 py-2 bg-rose-500/10 border border-rose-500/40 rounded text-rose-300 text-[10px] font-black animate-pulse flex items-center gap-3 shadow-[0_0_20px_rgba(244,63,94,0.2)]">
                            <ZapIcon className="w-4 h-4" /> AGENT_Q_ACTIVE: {externalTaskName?.substring(0, 15)}...
                        </div>
                    )}
                    <div className="flex gap-4">
                        <div className="text-right">
                            <p className="text-[9px] font-black text-cyan-600 tracking-tighter uppercase">Global Uptime</p>
                            <p className="text-sm font-mono text-white">99.999%</p>
                        </div>
                        <div className="text-right">
                            <p className="text-[9px] font-black text-cyan-600 tracking-tighter uppercase">Coherence</p>
                            <p className="text-sm font-mono text-white">{(88 + progress/10).toFixed(2)}%</p>
                        </div>
                    </div>
                </div>
            </div>

            <div className="flex-grow grid grid-cols-12 gap-6 items-center min-h-0 relative">
                <div className="col-span-3 flex flex-col gap-4 z-40 transform-style-preserve-3d" style={{ transform: 'translateZ(40px)' }}>
                    <div className="bg-black/40 border border-cyan-900/40 p-4 rounded-xl backdrop-blur-md">
                        <h3 className="text-[10px] font-black text-cyan-500 uppercase tracking-widest mb-4 border-b border-cyan-800/30 pb-2 flex items-center gap-2">
                            <ActivityIcon className="w-4 h-4" /> Manifold Vitals
                        </h3>
                        <div className="space-y-4">
                            {['Spatial', 'Temporal', 'Logical', 'Abstract'].map((dim, i) => (
                                <div key={dim} className="space-y-1">
                                    <div className="flex justify-between text-[8px] font-black text-cyan-700 uppercase tracking-tighter">
                                        <span>{dim} Stability</span>
                                        <span>{(80 + (i * 5) + Math.random()).toFixed(1)}%</span>
                                    </div>
                                    <div className="w-full h-1 bg-cyan-900/30 rounded-full overflow-hidden">
                                        <div 
                                            className="h-full bg-gradient-to-r from-cyan-600 to-cyan-300 transition-all duration-1000"
                                            style={{ width: `${80 + (i * 5)}%` }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="bg-black/40 border border-cyan-900/40 p-4 rounded-xl backdrop-blur-md flex-grow min-h-0 flex flex-col">
                        <h3 className="text-[10px] font-black text-cyan-500 uppercase tracking-widest mb-2 flex items-center gap-2">
                            <ArrowTrendingUpIcon className="w-4 h-4" /> Entropy Drift
                        </h3>
                        <div className="flex-grow min-h-0">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={trajectory}>
                                    <defs>
                                        <linearGradient id="drift-grad" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.3}/>
                                            <stop offset="95%" stopColor="#22d3ee" stopOpacity={0}/>
                                        </linearGradient>
                                    </defs>
                                    <Area type="monotone" dataKey="val" stroke="#22d3ee" fill="url(#drift-grad)" isAnimationActive={false} strokeWidth={2} />
                                    <YAxis domain={[0, 100]} hide />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>

                <div className="col-span-6 h-full relative flex items-center justify-center perspective-[1000px]">
                    <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-40" viewBox="0 0 100 100" preserveAspectRatio="none">
                        {cognitionEngines.map(e => (
                            <path 
                                key={e.id}
                                d={`M 50 50 L ${e.x} ${e.y}`}
                                stroke={activeEngineId === e.id ? "#22d3ee" : "#1e3a8a"}
                                strokeWidth={activeEngineId === e.id ? 0.5 : 0.25}
                                strokeDasharray="2 2"
                                className={activeEngineId === e.id || status === 'merged' ? 'animate-flow' : ''}
                            />
                        ))}
                    </svg>

                    <div className="relative transform-style-preserve-3d" style={{ transform: 'translateZ(100px)' }}>
                        <div className={`absolute inset-0 bg-cyan-500/20 rounded-full blur-[60px] animate-pulse pointer-events-none ${status === 'error' ? 'bg-red-500/20' : ''}`} />
                        
                        <button 
                            onClick={initiateSync}
                            className={`w-56 h-56 rounded-full border-2 bg-black/90 flex flex-col items-center justify-center transition-all duration-1000 relative group
                                ${status === 'merged' ? 'border-cyan-400 shadow-[0_0_80px_rgba(34,211,238,0.4)] scale-110' : 
                                  status === 'syncing' ? 'border-amber-400 animate-pulse' : 
                                  status === 'error' ? 'border-red-500 shadow-[0_0_40px_rgba(239,68,68,0.4)]' :
                                  'border-cyan-900/50 hover:border-cyan-400 hover:shadow-[0_0_40px_rgba(34,211,238,0.2)]'}
                            `}
                        >
                            <BrainCircuitIcon className={`w-20 h-20 transition-all duration-1000 ${status === 'merged' ? 'text-white drop-shadow-[0_0_15px_cyan]' : status === 'error' ? 'text-red-500' : 'text-cyan-900'}`} />
                            <p className={`mt-4 text-[10px] font-black tracking-[0.4em] uppercase group-hover:text-cyan-300 ${status === 'error' ? 'text-red-500' : 'text-cyan-600'}`}>
                                {status === 'merged' ? 'Singularity' : status === 'syncing' ? 'Synthesizing' : status === 'error' ? 'RETRY' : 'Initiate'}
                            </p>
                            
                            <div className="absolute -inset-4 border-t border-cyan-400/20 animate-ring-1 rounded-full pointer-events-none" />
                            <div className="absolute -inset-10 border-b border-cyan-400/10 animate-ring-2 rounded-full pointer-events-none" />
                        </button>

                        {cognitionEngines.map((engine) => (
                            <button
                                key={engine.id}
                                onClick={() => onPanelSelect?.(engine.targetPanelId)}
                                className={`absolute w-14 h-14 rounded-xl border-2 bg-black/80 flex flex-col items-center justify-center transition-all duration-500 backdrop-blur-md group/sat
                                    ${activeEngineId === engine.id ? 'border-cyan-400 scale-125 shadow-lg z-50' : 'border-cyan-900/50 opacity-40 hover:opacity-100 hover:border-cyan-500'}
                                `}
                                style={{
                                    left: `${engine.x}%`,
                                    top: `${engine.y}%`,
                                    transform: 'translate(-50%, -50%)',
                                }}
                            >
                                <engine.icon className={`w-6 h-6 ${activeEngineId === engine.id ? 'text-cyan-300' : 'text-cyan-800'} group-hover/sat:text-cyan-400 transition-colors`} />
                                <span className="text-[7px] font-black text-cyan-900 uppercase group-hover/sat:text-cyan-400 mt-1">{engine.label}</span>
                            </button>
                        ))}
                    </div>
                </div>

                <div className="col-span-3 h-full flex flex-col gap-4 transform-style-preserve-3d z-40" style={{ transform: 'translateZ(40px)' }}>
                    <div className="bg-black/40 border border-cyan-900/40 p-4 rounded-xl backdrop-blur-md">
                        <div className="flex justify-between items-center mb-3">
                            <h3 className="text-[10px] font-black text-cyan-500 uppercase tracking-widest flex items-center gap-2">
                                <LayersIcon className="w-4 h-4" /> Cognitive HUD
                            </h3>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                            <div className="bg-cyan-950/20 p-2 rounded border border-cyan-800/30 text-center">
                                <p className="text-[8px] font-black text-cyan-600 uppercase">Cycle</p>
                                <p className="text-xl font-mono text-white font-black">{epoch}</p>
                            </div>
                            <div className="bg-cyan-950/20 p-2 rounded border border-cyan-800/30 text-center">
                                <p className="text-[8px] font-black text-cyan-600 uppercase">Threads</p>
                                <p className="text-xl font-mono text-white font-black">2048</p>
                            </div>
                        </div>
                    </div>

                    <div className="bg-black/40 border border-cyan-900/40 rounded-xl backdrop-blur-md flex-grow min-h-0 overflow-hidden">
                        <QubitStateVisualizer qubitStability={qubitStability} />
                    </div>
                </div>
            </div>

            <div className="h-32 grid grid-cols-12 gap-6 z-50 flex-shrink-0 transform-style-preserve-3d" style={{ transform: 'translateZ(50px)' }}>
                <div className="col-span-9 bg-black/80 border border-cyan-900/50 rounded-xl p-4 font-mono text-[10px] overflow-hidden flex flex-col shadow-inner backdrop-blur-2xl">
                    <div className="flex items-center justify-between text-cyan-500 mb-2 border-b border-cyan-900/30 pb-2">
                        <div className="flex items-center gap-3 font-black uppercase tracking-[0.2em]">
                            <TerminalIcon className="w-4 h-4" /> topology_alignment_stream
                        </div>
                        <span className="animate-pulse text-green-500 font-black">● LIVE_FEED</span>
                    </div>
                    <div className="flex-grow overflow-y-auto space-y-1.5 custom-scrollbar pr-3">
                        {logs.map((log, i) => (
                            <div key={i} className="text-cyan-200/50 animate-fade-in flex gap-2">
                                <span className="text-cyan-600 font-black tracking-tighter">{">>>"}</span>
                                <span className="flex-grow leading-relaxed">{log}</span>
                            </div>
                        ))}
                        <div ref={logEndRef} />
                    </div>
                </div>

                <div className="col-span-3 flex flex-col justify-center">
                    <button 
                        onClick={initiateSync}
                        disabled={status === 'syncing'}
                        className={`w-full h-full py-6 rounded-xl font-black text-[13px] transition-all duration-500 border-2 uppercase tracking-[0.4em] flex flex-col items-center justify-center gap-2
                            ${status === 'syncing' ? 'bg-amber-900/10 border-amber-500/30 text-amber-300 cursor-wait' : 
                            status === 'merged' ? 'bg-rose-600/10 border-rose-500/40 text-rose-100 hover:bg-rose-600/20' :
                            status === 'error' ? 'bg-red-900/20 border-red-500 text-red-200 hover:bg-red-900/40' :
                            'bg-cyan-600/10 border-cyan-500/40 text-white hover:bg-cyan-600/20 hover:scale-[1.02] hover:shadow-[0_0_40px_rgba(34,211,238,0.3)]'}
                        `}
                    >
                        {status === 'syncing' ? (
                            <><LoaderIcon className="w-6 h-6 animate-spin" /> SYNCHRONIZING</>
                        ) : status === 'merged' ? (
                            'DE-LINK_MANIFOLD'
                        ) : status === 'error' ? (
                            'SYSTEM FAILURE - RETRY'
                        ) : (
                            'IGNITE_CORE'
                        )}
                        <span className="text-[7px] tracking-[0.1em] opacity-40">
                            {status === 'error' ? 'MANIFOLD_COLLAPSED' : '12D_MAPPING_READY'}
                        </span>
                    </button>
                </div>
            </div>

            {status === 'merged' && !isExternalProcessing && (
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-[100] overflow-hidden">
                     <div className="w-full h-[2px] bg-cyan-400/50 animate-ping shadow-[0_0_20px_rgba(34,211,238,0.5)]"></div>
                     <div className="absolute inset-0 bg-cyan-400/5 opacity-[0.03] animate-pulse"></div>
                </div>
            )}
        </div>
    );
};

export default AGISingularityInterface;
