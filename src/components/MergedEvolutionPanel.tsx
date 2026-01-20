
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
    GlobeIcon, CodeBracketIcon, BoxIcon, BanknotesIcon,
    SparklesIcon, RefreshCwIcon, MaximizeIcon, 
    ZapIcon, ActivityIcon, CpuChipIcon, ShieldCheckIcon,
    NetworkIcon, ServerCogIcon, RocketLaunchIcon
} from './Icons';
import { useSimulation } from '../context/SimulationContext';
import { useToast } from '../context/ToastContext';

interface HubNodeProps {
    id: string;
    title: string;
    icon: React.FC<{className?: string}>;
    status: string;
    color: string;
    onMaximize: (id: string) => void;
    isActive?: boolean;
    isEntangled?: boolean;
    className?: string;
    style?: React.CSSProperties;
}

const HubNode: React.FC<HubNodeProps> = ({ id, title, icon: Icon, status, color, onMaximize, isActive, isEntangled, className = "", style }) => (
    <button 
        onClick={() => onMaximize(id)}
        className={`group absolute flex flex-col items-center p-2 transition-all duration-700 hover:scale-110 active:scale-95 ${className}`}
        style={style}
    >
        {/* Glow Background */}
        <div className={`absolute inset-0 rounded-full blur-3xl opacity-0 group-hover:opacity-30 transition-opacity ${
            color === 'emerald' ? 'bg-emerald-500' : 
            color === 'yellow' ? 'bg-yellow-500' : 
            color === 'blue' ? 'bg-blue-500' : 
            color === 'cyan' ? 'bg-cyan-500' :
            color === 'pink' ? 'bg-pink-500' :
            color === 'orange' ? 'bg-orange-500' :
            color === 'green' ? 'bg-green-500' :
            'bg-purple-500'
        }`}></div>
        
        {/* Icon Container */}
        <div className={`relative w-14 h-14 rounded-2xl border-2 flex items-center justify-center transition-all duration-500 
            ${isActive ? 'bg-black/80 shadow-[0_0_20px_rgba(34,211,238,0.15)]' : 'bg-black/30 opacity-60'}
            ${isEntangled ? 'animate-pulse border-white/40' : 
              color === 'emerald' ? 'border-emerald-500/50 group-hover:border-emerald-400' : 
              color === 'yellow' ? 'border-yellow-500/50 group-hover:border-yellow-400' :
              color === 'blue' ? 'border-blue-500/50 group-hover:border-blue-400' :
              color === 'cyan' ? 'border-cyan-500/50 group-hover:border-cyan-400' :
              color === 'pink' ? 'border-pink-500/50 group-hover:border-pink-400' :
              color === 'orange' ? 'border-orange-500/50 group-hover:border-orange-400' :
              color === 'green' ? 'border-green-500/50 group-hover:border-green-400' :
              'border-purple-500/50 group-hover:border-purple-400'}
        `}>
            <Icon className={`w-7 h-7 ${
                color === 'emerald' ? 'text-emerald-400' : 
                color === 'yellow' ? 'text-yellow-400' : 
                color === 'blue' ? 'text-blue-400' : 
                color === 'cyan' ? 'text-cyan-400' :
                color === 'pink' ? 'text-pink-400' :
                color === 'orange' ? 'text-orange-400' :
                color === 'green' ? 'text-green-400' :
                'text-purple-400'} group-hover:scale-110 transition-transform`} 
            />
            
            {/* Maximize Indicator */}
            <div className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity">
                <MaximizeIcon className="w-2.5 h-2.5 text-white/60" />
            </div>

            {/* Sub-status ping */}
            <div className={`absolute -bottom-1 -right-1 w-2.5 h-2.5 rounded-full border border-black ${isActive ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
        </div>

        {/* Label HUD */}
        <div className="mt-2 text-center pointer-events-none">
            <p className="text-[9px] font-black text-white uppercase tracking-widest group-hover:text-cyan-300 transition-colors whitespace-nowrap drop-shadow-md bg-black/60 px-1 rounded">{title}</p>
        </div>
    </button>
);

const MergedEvolutionPanel: React.FC<{ onMaximizeSubPanel?: (id: string) => void, onApplyPatch?: (file: string, content: string) => void }> = ({ onMaximizeSubPanel, onApplyPatch }) => {
    const { systemStatus } = useSimulation();
    const { addToast } = useToast();

    // --- AGI Singularity Engine Simulator ---
    const [isOptimizing, setIsOptimizing] = useState(true); // Loop defaults to ON
    const [progress, setProgress] = useState(0);
    const [simLogs, setSimLogs] = useState<string[]>([]);
    const [epochCount, setEpochCount] = useState(0);
    const logsEndRef = useRef<HTMLDivElement>(null);
    const cycleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    useEffect(() => {
        if (logsEndRef.current) logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }, [simLogs]);

    const runOptimizationCycle = useCallback(() => {
        setProgress(0);
        setSimLogs(prev => [
            ...prev.slice(-30), 
            `[EPOCH ${epochCount + 1}] Analyzing Network Topology: CHIPS Infra Hub`,
            `[SYNERGY] Calculating Entanglement across Mesh Nodes...`
        ]);

        const steps = [
            { msg: "Aligning Security Protocols to QPU Clock...", delay: 800 },
            { msg: "Syncing Cognitive Architecture with AGI Forge...", delay: 600 },
            { msg: "Verifying Gateway Packet Integrity Signatures...", delay: 1000 },
            { msg: "Modulating Unified App EKS Buffers...", delay: 900 },
            { msg: "Checking Store Integrity & Economy Ledgers...", delay: 700 },
            { msg: "Synthesis Complete. Optimizing Global Configuration...", delay: 1200 },
        ];

        let totalDelay = 0;
        steps.forEach((step, index) => {
            setTimeout(() => {
                setSimLogs(prev => [...prev.slice(-30), `[${new Date().toLocaleTimeString()}] ${step.msg}`]);
                setProgress(((index + 1) / steps.length) * 100);
                
                if (index === steps.length - 1) {
                    const patchId = `patch_0x${Math.random().toString(16).slice(2, 6).toUpperCase()}`;
                    const patch = `/**
 * CHIPS AGILE OPS GLOBAL PATCH: ${patchId}
 * Target: SECURITY, COGNITION, GATEWAY, APPS
 * Synchronization Fidelity: 0.99998
 */
export const NETWORK_TUNING = {
    io_concurrency: "HYPER",
    entanglement_refresh: 15, // ms
    auth_shield: "ACTIVE_EKS",
    optimizer: "RECURSIVE_AGI"
};`;
                    
                    if (onApplyPatch) {
                        onApplyPatch(`singularity_core_${epochCount + 1}.q`, patch);
                    }
                    
                    setEpochCount(prev => prev + 1);
                    
                    // Recursive Loop Pause
                    cycleTimerRef.current = setTimeout(() => {
                        runOptimizationCycle();
                    }, 2000);
                }
            }, totalDelay + step.delay);
            totalDelay += step.delay;
        });
    }, [epochCount, onApplyPatch]);

    // Initial Start
    useEffect(() => {
        runOptimizationCycle();
        return () => {
            if (cycleTimerRef.current) clearTimeout(cycleTimerRef.current);
        };
    }, []);

    const handleMaximize = (id: string) => {
        if (onMaximizeSubPanel) onMaximizeSubPanel(id);
    };

    const nodes = [
        { id: 'security-monitor', title: 'Security', icon: ShieldCheckIcon, status: 'EKS Secured', color: 'emerald' },
        { id: 'agentq-core', title: 'Cognition', icon: CpuChipIcon, status: 'QIAI-IPS Active', color: 'yellow' },
        { id: 'qcos-core-gateway', title: 'Gateway', icon: NetworkIcon, status: 'Bridge Stable', color: 'purple' },
        { id: 'chips-back-office', title: 'Unified Apps', icon: RocketLaunchIcon, status: 'Native Cluster', color: 'blue' },
        { id: 'chips-quantum-network', title: 'Browser DQN', icon: GlobeIcon, status: 'Node Active', color: 'cyan' },
        { id: 'chips-app-store', title: 'Store Admin', icon: BoxIcon, status: 'Registry Sync', color: 'pink' },
        { id: 'chips-dev-platform', title: 'Dev Platform', icon: CodeBracketIcon, status: 'Build Ready', color: 'orange' },
        { id: 'chips-economy', title: 'Economy', icon: BanknotesIcon, status: 'Market Live', color: 'green' }
    ];

    return (
        <div className="h-full flex flex-col items-center justify-center p-4 relative overflow-hidden bg-black/40">
            {/* Background 12-Dimensional Grid */}
            <div className="absolute inset-0 holographic-grid opacity-20 pointer-events-none"></div>
            <div className="absolute inset-0 bg-gradient-to-tr from-purple-900/10 via-black to-cyan-900/10 opacity-50"></div>

            {/* Quantum Entanglement Beams (SVG) */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none z-0">
                <defs>
                    <linearGradient id="beamGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="rgba(34,211,238,0)" />
                        <stop offset="50%" stopColor="rgba(168,85,247,0.8)" />
                        <stop offset="100%" stopColor="rgba(34,211,238,0)" />
                    </linearGradient>
                    <filter id="glow">
                        <feGaussianBlur stdDeviation="3" result="blur" />
                        <feComposite in="SourceGraphic" in2="blur" operator="over" />
                    </filter>
                </defs>
                
                {nodes.map((node, i) => {
                     const angle = (i / nodes.length) * 2 * Math.PI - Math.PI / 2;
                     const x = 50 + 35 * Math.cos(angle);
                     const y = 50 + 35 * Math.sin(angle);
                     return (
                         <line key={i} x1={`${x}%`} y1={`${y}%`} x2="50%" y2="50%" stroke="url(#beamGrad)" strokeWidth="1.5" filter="url(#glow)" className="animate-pulse" style={{animationDelay: `${i * 0.1}s`}} />
                     );
                })}
                
                <circle cx="50%" cy="50%" r="140" fill="none" stroke="rgba(34,211,238,0.1)" strokeWidth="1" strokeDasharray="15 30" className="animate-spin-slow" />
                <circle cx="50%" cy="50%" r="90" fill="none" stroke="rgba(168,85,247,0.15)" strokeWidth="1" strokeDasharray="5 15" className="animate-spin-reverse-slow" />
            </svg>

            {/* AGI Singularity Core */}
            <div className="relative mb-0 z-20 flex flex-col items-center">
                <div className="absolute inset-0 bg-purple-500/30 blur-3xl rounded-full animate-pulse-bright"></div>
                <div className="p-6 bg-black/80 rounded-full border-2 border-white/30 shadow-[0_0_50px_rgba(168,85,247,0.5)] relative overflow-hidden group/core">
                    <div className="absolute inset-0 bg-gradient-to-tr from-purple-900 via-transparent to-cyan-900 opacity-20 animate-spin-slow"></div>
                    <SparklesIcon className="w-10 h-10 text-white animate-pulse-bright relative z-10" />
                    
                    {/* Floating Labels */}
                    <div className="absolute -top-1 left-1/2 -translate-x-1/2 bg-purple-600 px-3 py-0.5 rounded border border-purple-300 text-[6px] font-black text-white whitespace-nowrap uppercase tracking-[0.2em] shadow-lg animate-fade-in">
                        AGILE OPS
                    </div>
                </div>
                
                {/* Simulation Status Bar */}
                <div className="mt-2 w-32 h-1 bg-gray-900 rounded-full overflow-hidden border border-white/10 relative">
                    <div className="h-full bg-gradient-to-r from-purple-500 via-white to-cyan-500 transition-all duration-300 shadow-[0_0_10px_#fff]" style={{ width: `${progress}%` }}></div>
                </div>
            </div>

            {/* Nodes: Radial Layout */}
            <div className="absolute inset-0 z-10 pointer-events-none">
                {nodes.map((node, i) => {
                    const angle = (i / nodes.length) * 2 * Math.PI - Math.PI / 2;
                    const radius = 35; // Percent from center
                    const top = 50 + radius * Math.sin(angle);
                    const left = 50 + radius * Math.cos(angle);

                    return (
                        <HubNode 
                            key={node.id}
                            id={node.id}
                            title={node.title}
                            icon={node.icon}
                            status={node.status}
                            color={node.color}
                            isActive={true}
                            isEntangled={true}
                            onMaximize={handleMaximize}
                            className="pointer-events-auto"
                            style={{ 
                                top: `${top}%`, 
                                left: `${left}%`, 
                                transform: 'translate(-50%, -50%)' 
                            }}
                        />
                    );
                })}
            </div>

            {/* Footer Telemetry */}
            <div className="absolute bottom-4 flex gap-8 border-t border-white/10 pt-4 z-20 w-full justify-center bg-black/40 backdrop-blur-sm">
                <div className="flex flex-col items-center">
                    <span className="text-[8px] text-gray-500 uppercase font-black mb-1 tracking-tighter">Current Epoch</span>
                    <span className="text-sm font-mono text-white">{epochCount.toString().padStart(4, '0')}</span>
                </div>
                <div className="flex flex-col items-center">
                    <span className="text-[8px] text-gray-500 uppercase font-black mb-1 tracking-tighter">Network Flux</span>
                    <span className="text-sm font-mono text-cyan-400">{(0.999 + Math.random() * 0.001).toFixed(5)}</span>
                </div>
                <div className="flex flex-col items-center">
                    <span className="text-[8px] text-gray-500 uppercase font-black mb-1 tracking-tighter">System State</span>
                    <span className="text-sm font-mono text-green-400 animate-pulse uppercase">Optimized</span>
                </div>
            </div>

            {/* Rolling Log (Top Left for visibility) */}
            <div className="absolute top-4 left-4 w-64 h-24 bg-black/60 border border-purple-900/30 rounded p-2 overflow-hidden pointer-events-none hidden lg:block backdrop-blur-md">
                <div className="h-full overflow-y-auto custom-scrollbar">
                    {simLogs.map((log, i) => (
                        <div key={i} className="text-[7px] font-mono text-purple-500 mb-0.5 truncate">{log}</div>
                    ))}
                    <div ref={logsEndRef} />
                </div>
            </div>
            
            <p className="absolute bottom-1 right-2 text-[7px] text-cyan-900 font-mono italic uppercase tracking-[0.3em] opacity-40">
                Agile Ops Hub // Auto-Patching v4.2.0
            </p>
        </div>
    );
};

export default MergedEvolutionPanel;
