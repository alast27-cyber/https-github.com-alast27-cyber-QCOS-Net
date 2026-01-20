
import React, { useState, useMemo, useEffect, useRef } from 'react';
import { 
    ServerCogIcon, NetworkIcon, GlobeIcon, BoxIcon, 
    CpuChipIcon, ActivityIcon, ShieldCheckIcon, LinkIcon,
    HardDriveIcon, CloudServerIcon, ArrowRightIcon,
    ChartBarIcon, UsersIcon, SearchIcon, RocketLaunchIcon,
    CheckCircle2Icon, AlertTriangleIcon, ClockIcon, LockIcon,
    ArrowTopRightOnSquareIcon, TerminalIcon, StopIcon, PlayIcon,
    RefreshCwIcon, ZapIcon, ScaleIcon, TrashIcon
} from './Icons';
import { URIAssignment, AppDefinition } from '../types';
import CHIPSStoreAdmin from './CHIPSStoreAdmin';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

interface CHIPSBackOfficeProps {
  uriAssignments: URIAssignment[];
  marketApps: AppDefinition[];
}

// --- 1. Decentralized Quantum Nodes (DQN) View ---
const NodeOperationsView = () => {
    const nodes = [
        { id: 'DQN-Alpha', region: 'US-East', status: 'Entangled', latency: '12ms', qubits: 128, load: 45 },
        { id: 'DQN-Beta', region: 'EU-West', status: 'Entangled', latency: '24ms', qubits: 64, load: 78 },
        { id: 'DQN-Gamma', region: 'Asia-Pac', status: 'Syncing', latency: '140ms', qubits: 256, load: 12 },
        { id: 'DQN-Local', region: 'Localhost', status: 'Active', latency: '0ms', qubits: 32, load: 5 },
    ];

    return (
        <div className="h-full flex flex-col space-y-4 animate-fade-in overflow-hidden">
            <div className="grid grid-cols-3 gap-4">
                <div className="bg-black/20 p-4 rounded-lg border border-cyan-800/50 flex items-center justify-between">
                    <div>
                        <p className="text-xs text-cyan-500 uppercase tracking-widest">Active Nodes</p>
                        <p className="text-2xl font-mono text-white">4</p>
                    </div>
                    <CpuChipIcon className="w-8 h-8 text-cyan-900" />
                </div>
                <div className="bg-black/20 p-4 rounded-lg border border-cyan-800/50 flex items-center justify-between">
                    <div>
                        <p className="text-xs text-green-500 uppercase tracking-widest">Mesh Health</p>
                        <p className="text-2xl font-mono text-green-400">99.9%</p>
                    </div>
                    <ActivityIcon className="w-8 h-8 text-green-900" />
                </div>
                <div className="bg-black/20 p-4 rounded-lg border border-cyan-800/50 flex items-center justify-between">
                    <div>
                        <p className="text-xs text-purple-500 uppercase tracking-widest">Total Qubits</p>
                        <p className="text-2xl font-mono text-purple-400">480</p>
                    </div>
                    <NetworkIcon className="w-8 h-8 text-purple-900" />
                </div>
            </div>

            <div className="flex-grow bg-black/30 border border-cyan-900/50 rounded-lg overflow-hidden flex flex-col">
                <div className="p-3 bg-cyan-950/30 border-b border-cyan-800/50 flex justify-between items-center">
                    <h3 className="font-bold text-sm text-cyan-200 flex items-center">
                        <ServerCogIcon className="w-4 h-4 mr-2" /> Node Registry
                    </h3>
                    <div className="flex gap-2">
                         <span className="text-[10px] text-cyan-600 font-mono">PROTOCOL: EKS-V2</span>
                    </div>
                </div>
                {/* Prevent scrolling on Node Registry table */}
                <div className="flex-grow overflow-hidden">
                    <table className="w-full text-xs text-left">
                        <thead className="bg-cyan-950/20 text-cyan-500 font-mono uppercase sticky top-0 z-10">
                            <tr>
                                <th className="p-3">Node ID</th>
                                <th className="p-3">Region</th>
                                <th className="p-3">Status</th>
                                <th className="p-3">Qubits</th>
                                <th className="p-3 text-right">Load</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-cyan-900/30">
                            {nodes.map(node => (
                                <tr key={node.id} className="hover:bg-cyan-900/10 transition-colors">
                                    <td className="p-3 font-medium text-white flex items-center gap-2">
                                        <div className={`w-2 h-2 rounded-full ${node.status === 'Entangled' || node.status === 'Active' ? 'bg-green-500 shadow-[0_0_5px_#22c55e]' : 'bg-yellow-500 animate-pulse'}`}></div>
                                        {node.id}
                                    </td>
                                    <td className="p-3 text-gray-400">{node.region}</td>
                                    <td className="p-3">
                                        <span className={`px-2 py-0.5 rounded border text-[10px] uppercase ${node.status.includes('Sync') ? 'bg-yellow-900/20 border-yellow-800 text-yellow-300' : 'bg-green-900/20 border-green-800 text-green-300'}`}>
                                            {node.status}
                                        </span>
                                    </td>
                                    <td className="p-3 font-mono text-purple-300">{node.qubits}Q</td>
                                    <td className="p-3 text-right">
                                        <div className="flex items-center justify-end gap-2">
                                            <span className="text-cyan-300">{node.load}%</span>
                                            <div className="w-16 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                                                <div className={`h-full ${node.load > 80 ? 'bg-red-500' : 'bg-cyan-500'}`} style={{width: `${node.load}%`}}></div>
                                            </div>
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

// --- 2. Quantum to Web Gateway & Hosting View ---
const GatewayHostingView = ({ apps }: { apps: AppDefinition[] }) => {
    const pods = useMemo(() => {
        const basePods = [
            { id: 'pod-q-core', app: 'System Kernel', type: 'Q-State Cache', uptime: '100%', bandwidth: '4.5 GB/s', requests: '45k/s', status: 'Active' },
            { id: 'pod-bridge-1', app: 'Gateway Root', type: 'Hybrid-Bridge', uptime: '99.99%', bandwidth: '1.2 GB/s', requests: '12k/s', status: 'Active' },
        ];
        
        const appPods = apps.filter(a => a.status === 'installed').map((app, i) => ({
            id: `pod-app-${i + 1}`,
            app: app.name,
            type: 'Application Instance',
            uptime: '99.95%',
            bandwidth: `${(Math.random() * 0.5 + 0.5).toFixed(2)} GB/s`,
            requests: `${Math.floor(Math.random() * 5 + 1)}k/s`,
            status: 'Provisioned'
        }));

        return [...basePods, ...appPods];
    }, [apps]);

    return (
        <div className="h-full flex flex-col space-y-4 animate-fade-in overflow-hidden">
            <div className="bg-gradient-to-r from-purple-900/20 to-blue-900/20 p-4 rounded-lg border border-purple-500/30 flex items-center justify-between">
                <div>
                    <h3 className="text-lg font-bold text-white flex items-center gap-2">
                        <CloudServerIcon className="w-6 h-6 text-purple-400" /> Hybrid Infrastructure Provisioning
                    </h3>
                    <p className="text-xs text-purple-200">Auto-allocated bridge pods translating Q-Lang packets to HTTPS responses.</p>
                </div>
                <div className="text-right">
                    <p className="text-xs text-purple-400 uppercase tracking-widest">Active Hybrid Pods</p>
                    <p className="text-2xl font-mono text-white">{pods.length}</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-grow min-h-0 overflow-hidden">
                <div className="bg-black/20 rounded-lg border border-cyan-800/50 p-3 flex flex-col overflow-hidden">
                    <h4 className="text-sm font-bold text-cyan-300 mb-3 flex items-center border-b border-cyan-900/50 pb-2">
                        <HardDriveIcon className="w-4 h-4 mr-2" /> Server Pod Clusters
                    </h4>
                    {/* Prevent scrolling on pods list */}
                    <div className="space-y-2 overflow-hidden pr-1">
                        {pods.map(pod => (
                            <div key={pod.id} className="bg-cyan-950/20 p-2 rounded border border-cyan-900 flex justify-between items-center group hover:bg-cyan-900/30 transition-all">
                                <div>
                                    <div className="font-bold text-white text-xs flex items-center gap-2">
                                        {pod.id} 
                                        <span className="text-[9px] text-cyan-600 font-normal">({pod.app})</span>
                                    </div>
                                    <div className="text-[10px] text-gray-500">{pod.type}</div>
                                </div>
                                <div className="text-right">
                                    <div className={`text-xs font-mono ${pod.status === 'Active' ? 'text-green-400' : 'text-cyan-400 animate-pulse'}`}>{pod.status}</div>
                                    <div className="text-[10px] text-cyan-600">{pod.bandwidth}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="bg-black/20 rounded-lg border border-cyan-800/50 p-3 flex flex-col overflow-hidden">
                    <h4 className="text-sm font-bold text-cyan-300 mb-3 flex items-center border-b border-cyan-900/50 pb-2">
                        <ShieldCheckIcon className="w-4 h-4 mr-2" /> Security Policy Enforcer
                    </h4>
                    <div className="space-y-4 text-xs">
                        <div className="bg-green-950/20 p-2 rounded border border-green-900/40 flex justify-between items-center">
                            <div className="flex items-center gap-2">
                                <CheckCircle2Icon className="w-4 h-4 text-green-500" />
                                <span className="text-gray-300">DDoS Protection</span>
                            </div>
                            <span className="text-green-400 font-bold uppercase">Active</span>
                        </div>
                        <div className="bg-green-950/20 p-2 rounded border border-green-900/40 flex justify-between items-center">
                            <div className="flex items-center gap-2">
                                <CheckCircle2Icon className="w-4 h-4 text-green-500" />
                                <span className="text-gray-300">Q-Packet Verification</span>
                            </div>
                            <span className="text-green-400 font-bold uppercase">Enforced</span>
                        </div>
                        
                        <div className="pt-2 border-t border-cyan-900/30">
                            <p className="text-cyan-500 mb-2 uppercase font-bold tracking-tighter">Cipher Stack</p>
                            <div className="grid grid-cols-2 gap-2">
                                <div className="bg-black/40 p-2 rounded border border-purple-500/30 text-purple-300 font-mono text-center text-[9px]">
                                    Kyber-1024
                                </div>
                                <div className="bg-black/40 p-2 rounded border border-purple-500/30 text-purple-300 font-mono text-center text-[9px]">
                                    Dilithium-5
                                </div>
                            </div>
                        </div>
                        <div className="bg-yellow-900/10 p-2 rounded border border-yellow-800/30 text-[10px] text-yellow-500/70 italic text-center mt-auto">
                            Auto-Scaling Protocol: ACTIVE (Based on Neural Load)
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

// --- 3. Domain Name Registry View ---
const DomainRegistryView = ({ uriAssignments }: { uriAssignments: URIAssignment[] }) => {
    return (
        <div className="h-full flex flex-col space-y-4 animate-fade-in overflow-hidden">
             <div className="flex items-center justify-between bg-black/20 p-4 rounded-lg border border-cyan-800/50">
                <div className="flex items-center gap-3">
                    <GlobeIcon className="w-8 h-8 text-blue-400" />
                    <div>
                        <h3 className="text-lg font-bold text-white">Quantum Name Service (QNS)</h3>
                        <p className="text-xs text-blue-300">Mapping immutable Q-URIs to classical DNS via Q-DNS Bridge.</p>
                    </div>
                </div>
                <button className="holographic-button px-4 py-2 text-xs flex items-center gap-2">
                    <LinkIcon className="w-4 h-4" /> Register New Mapping
                </button>
            </div>

            <div className="flex-grow bg-black/30 border border-cyan-900/50 rounded-lg overflow-hidden flex flex-col">
                <table className="w-full text-xs text-left">
                    <thead className="bg-cyan-950/30 text-cyan-500 font-mono uppercase sticky top-0 z-10">
                        <tr>
                            <th className="p-3">Web Domain</th>
                            <th className="p-3">Target Q-URI</th>
                            <th className="p-3">Security</th>
                            <th className="p-3 text-right">Traffic</th>
                            <th className="p-3 text-center">Status</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-cyan-900/30 overflow-hidden">
                        {uriAssignments.map((ua, idx) => (
                            <tr key={idx} className="hover:bg-cyan-900/10 transition-colors animate-fade-in-right" style={{animationDelay: `${idx * 0.1}s`}}>
                                <td className="p-3 font-bold text-white flex items-center gap-2">
                                    <ArrowTopRightOnSquareIcon className="w-3 h-3 text-blue-500" />
                                    {ua.https_url.replace('https://', '')}
                                </td>
                                <td className="p-3 font-mono text-cyan-600 truncate max-w-[200px]">{ua.q_uri}</td>
                                <td className="p-3 text-green-300 flex items-center gap-1">
                                    <LockIcon className="w-3 h-3" /> EKS-Certified
                                </td>
                                <td className="p-3 text-right font-mono text-gray-400">Live</td>
                                <td className="p-3 text-center">
                                    <span className="inline-block w-2 h-2 rounded-full bg-green-500 shadow-[0_0_5px_#22c55e]"></span>
                                </td>
                            </tr>
                        ))}
                        {uriAssignments.length === 0 && (
                            <tr>
                                <td colSpan={5} className="p-8 text-center text-gray-500 italic">No external domains registered. Deploy an app to initiate.</td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

// --- 5. Unified App Back Office View (Live & Dynamic) ---
const UnifiedAppBackOffice = ({ uriAssignments, marketApps }: { uriAssignments: URIAssignment[]; marketApps: AppDefinition[] }) => {
    const activeApps = useMemo(() => {
        const installed = marketApps.filter(app => app.status === 'installed').map(app => {
            const assignment = uriAssignments.find(ua => ua.appName === app.name);
            return {
                id: app.id,
                appName: app.name,
                q_uri: app.q_uri || assignment?.q_uri || `CHIPS://${app.id}.apps`,
                https_url: app.https_url || assignment?.https_url || `https://qcos.apps.web/${app.id}`,
                icon: app.icon
            };
        });
        return installed;
    }, [uriAssignments, marketApps]);

    const [selectedApp, setSelectedApp] = useState<any | null>(null);
    const [liveMetrics, setLiveMetrics] = useState({ users: 0, requests: 0, cpu: 0, memory: 0, replicas: 2 });
    const [liveLogs, setLiveLogs] = useState<string[]>([]);
    const [isAppRunning, setIsAppRunning] = useState(true);

    useEffect(() => {
        if (!selectedApp && activeApps.length > 0) {
            setSelectedApp(activeApps[0]);
        }
    }, [activeApps, selectedApp]);

    useEffect(() => {
        if (!selectedApp || !isAppRunning) return;

        setLiveLogs([`[SYSTEM] Connected to instance ${selectedApp.id}...`, `[INFO] Container health: EXCELLENT`]);
        
        const interval = setInterval(() => {
            setLiveMetrics(prev => ({
                ...prev,
                users: Math.max(10, Math.floor(prev.users + (Math.random() - 0.5) * 50)),
                requests: Math.max(5, Math.floor(prev.requests + (Math.random() - 0.5) * 20)),
                cpu: Math.min(100, Math.max(5, prev.cpu + (Math.random() - 0.5) * 8)),
                memory: Math.min(100, Math.max(15, prev.memory + (Math.random() - 0.5) * 4)),
            }));

            if (Math.random() > 0.6) {
                const logTemplates = [
                    `[INFO] Ingress traffic routed via DQN-Alpha`,
                    `[DEBUG] Q-Packet verified: EKS-Signature Valid`,
                    `[SUCCESS] State vector synchronized across 3 nodes`,
                    `[WARN] Slight jitter detected in US-East-1 bridge`,
                    `[INFO] Auto-scaling trigger: Current load optimal`,
                    `[SUCCESS] EKS Key rotated for app session`,
                    `[DEBUG] Memory garbage collection complete`
                ];
                const newLog = logTemplates[Math.floor(Math.random() * logTemplates.length)];
                setLiveLogs(prevLogs => [`[${new Date().toLocaleTimeString()}] ${newLog}`, ...prevLogs.slice(0, 10)]); // Truncate logs
            }
        }, 1500);

        return () => clearInterval(interval);
    }, [selectedApp, isAppRunning]);

    const handleRestart = () => {
        setIsAppRunning(false);
        setLiveLogs(prev => [`[${new Date().toLocaleTimeString()}] [CRITICAL] RESTART SIGNAL RECEIVED`, ...prev]);
        setTimeout(() => {
            setIsAppRunning(true);
            setLiveLogs(prev => [`[${new Date().toLocaleTimeString()}] [SYSTEM] APPLICATION BOOT SUCCESSFUL`, ...prev]);
        }, 2000);
    };

    return (
        <div className="h-full flex gap-4 animate-fade-in relative overflow-hidden">
            <div className="absolute inset-0 holographic-grid opacity-10 pointer-events-none"></div>

            {/* App List Sidebar - Fixed Height, No Scroll */}
            <div className="w-1/3 bg-black/40 border border-cyan-900/50 rounded-lg flex flex-col overflow-hidden relative z-10">
                <div className="p-3 bg-cyan-950/40 border-b border-cyan-900/50 flex items-center justify-between">
                    <h3 className="text-xs font-bold text-cyan-200 flex items-center">
                        <BoxIcon className="w-4 h-4 mr-2" /> Deployed Instances
                    </h3>
                    <span className="bg-cyan-900/50 text-cyan-500 px-1.5 rounded text-[9px] font-mono">{activeApps.length}</span>
                </div>
                <div className="flex-grow overflow-hidden p-2 space-y-1">
                    {activeApps.length === 0 ? (
                        <div className="text-gray-600 text-[10px] p-8 text-center italic">No active deployments found.</div>
                    ) : (
                        activeApps.slice(0, 8).map((app) => ( // Show limited instances
                            <button 
                                key={app.id}
                                onClick={() => { setSelectedApp(app); setIsAppRunning(true); }}
                                className={`w-full text-left p-3 rounded-lg border transition-all flex justify-between items-center group ${selectedApp?.id === app.id ? 'bg-cyan-900/40 border-cyan-500 shadow-[0_0_15px_rgba(6,182,212,0.2)]' : 'bg-black/20 border-cyan-900/30 hover:bg-white/5 hover:border-cyan-700/50'}`}
                            >
                                <div className="min-w-0">
                                    <p className={`font-bold text-xs truncate ${selectedApp?.id === app.id ? 'text-white' : 'text-cyan-100 group-hover:text-white'}`}>{app.appName}</p>
                                    <p className="text-[8px] text-cyan-700 truncate mt-0.5 font-mono">ID: {app.id}</p>
                                </div>
                                <div className={`w-2 h-2 rounded-full ${selectedApp?.id === app.id ? 'bg-green-400 shadow-[0_0_8px_#4ade80]' : 'bg-gray-700'}`}></div>
                            </button>
                        ))
                    )}
                </div>
            </div>

            {/* App Detail View - Strictly Overflow Hidden */}
            <div className="flex-grow bg-black/40 border border-cyan-800/50 rounded-lg p-4 flex flex-col min-w-0 relative z-10 overflow-hidden">
                {selectedApp ? (
                    <>
                        {/* Header Area */}
                        <div className="flex justify-between items-start mb-6 pb-4 border-b border-cyan-900/50 flex-shrink-0">
                            <div>
                                <div className="flex items-center gap-3">
                                    <div className="p-2 bg-cyan-900/30 rounded-lg border border-cyan-500/30">
                                        <selectedApp.icon className="w-6 h-6 text-cyan-300" />
                                    </div>
                                    <div>
                                        <h2 className="text-xl font-black text-white tracking-wider flex items-center gap-3">
                                            {selectedApp.appName}
                                            <span className={`text-[9px] px-2 py-0.5 rounded border-2 font-black uppercase ${isAppRunning ? 'border-green-600 bg-green-900/30 text-green-400 shadow-[0_0_10px_rgba(34,197,94,0.3)]' : 'border-red-600 bg-red-900/30 text-red-400'}`}>
                                                {isAppRunning ? 'STABLE' : 'HALTED'}
                                            </span>
                                        </h2>
                                        <div className="flex gap-4 mt-1.5 text-[10px] font-mono text-cyan-600">
                                            <span className="flex items-center gap-1"><LinkIcon className="w-3 h-3"/> {selectedApp.q_uri}</span>
                                            <span className="flex items-center gap-1"><GlobeIcon className="w-3 h-3"/> {selectedApp.https_url}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div className="flex gap-2">
                                <button onClick={handleRestart} className="p-2 bg-yellow-900/20 border border-yellow-500/50 rounded text-yellow-400 hover:bg-yellow-900/40 transition-colors" title="Reboot Instance"><RefreshCwIcon className="w-4 h-4"/></button>
                                <button 
                                    onClick={() => setIsAppRunning(!isAppRunning)}
                                    className={`holographic-button px-4 py-2 text-[10px] font-black flex items-center gap-2 border-2 ${isAppRunning ? 'bg-red-600/20 border-red-500 text-red-300 hover:bg-red-600/30' : 'bg-green-600/20 border-green-500 text-green-300 hover:bg-green-600/30'}`}
                                >
                                    {isAppRunning ? <StopIcon className="w-3 h-3" /> : <PlayIcon className="w-3 h-3" />}
                                    {isAppRunning ? 'SHUTDOWN' : 'ACTIVATE'}
                                </button>
                            </div>
                        </div>

                        {/* Telemetry Matrix */}
                        <div className="grid grid-cols-2 lg:grid-cols-5 gap-3 mb-6 flex-shrink-0">
                            <div className="bg-black/60 p-3 rounded-lg border border-cyan-900/50 flex flex-col items-center">
                                <p className="text-[8px] text-gray-500 uppercase font-black tracking-tighter">Live Sessions</p>
                                <p className="text-xl font-mono text-white mt-1">{isAppRunning ? liveMetrics.users.toLocaleString() : 0}</p>
                            </div>
                            <div className="bg-black/60 p-3 rounded-lg border border-cyan-900/50 flex flex-col items-center">
                                <p className="text-[8px] text-gray-500 uppercase font-black tracking-tighter">Inbound / sec</p>
                                <p className="text-xl font-mono text-cyan-300 mt-1">{isAppRunning ? liveMetrics.requests : 0}</p>
                            </div>
                            <div className="bg-black/60 p-3 rounded-lg border border-cyan-900/50 flex flex-col items-center">
                                <p className="text-[8px] text-gray-500 uppercase font-black tracking-tighter">Q-Core Load</p>
                                <p className={`text-xl font-mono mt-1 ${liveMetrics.cpu > 80 ? 'text-red-400' : 'text-green-400'}`}>
                                    {isAppRunning ? liveMetrics.cpu : 0}%
                                </p>
                            </div>
                            <div className="bg-black/60 p-3 rounded-lg border border-cyan-900/50 flex flex-col items-center">
                                <p className="text-[8px] text-gray-500 uppercase font-black tracking-tighter">Entangled RAM</p>
                                <p className="text-xl font-mono text-purple-300 mt-1">{isAppRunning ? liveMetrics.memory : 0}%</p>
                            </div>
                            <div className="bg-cyan-900/20 p-3 rounded-lg border border-cyan-500/50 flex flex-col items-center group cursor-pointer hover:bg-cyan-900/40 transition-colors">
                                <p className="text-[8px] text-cyan-400 uppercase font-black tracking-tighter flex items-center gap-1">Scaling: Active <ScaleIcon className="w-2 h-2"/></p>
                                <p className="text-xl font-mono text-white mt-1">{liveMetrics.replicas}</p>
                                <span className="text-[7px] text-cyan-600 mt-0.5">Pod Cluster</span>
                            </div>
                        </div>

                        {/* Management Controls & Logs - Fixed Heights */}
                        <div className="flex-grow grid grid-cols-1 lg:grid-cols-2 gap-4 min-h-0 overflow-hidden">
                            {/* Command Terminal - Fixed Height, No Scroll */}
                            <div className="flex flex-col bg-black/60 border border-cyan-900/30 rounded-lg overflow-hidden relative">
                                <div className="absolute inset-0 bg-cyan-500/5 pointer-events-none"></div>
                                <div className="p-2 bg-cyan-950/40 border-b border-cyan-900/50 text-[10px] font-black text-cyan-500 uppercase flex justify-between items-center">
                                    <span className="flex items-center gap-2"><TerminalIcon className="w-3 h-3"/> Runtime Output Stream</span>
                                    <span className="text-gray-600 hover:text-white cursor-pointer" onClick={() => setLiveLogs([])}>Purge Log</span>
                                </div>
                                <div className="flex-grow overflow-hidden p-3 font-mono text-[9px]">
                                    {liveLogs.map((log, i) => (
                                        <div key={i} className="mb-1 text-cyan-100 flex gap-2 animate-fade-in-right">
                                            <span className="text-cyan-800 flex-shrink-0">[{i}]</span>
                                            <span className="break-all truncate">{log}</span>
                                        </div>
                                    ))}
                                    {isAppRunning && <div className="text-cyan-500 animate-pulse mt-1">_</div>}
                                </div>
                            </div>

                            {/* Infrastructure & Governance */}
                            <div className="flex flex-col gap-4 overflow-hidden">
                                <div className="bg-black/40 border border-cyan-900/50 rounded-lg p-3 flex-shrink-0">
                                    <h4 className="text-[10px] font-black text-cyan-500 uppercase tracking-widest mb-3 border-b border-cyan-900/30 pb-1">Container Governance</h4>
                                    <div className="space-y-3">
                                        <div className="flex justify-between items-center text-xs">
                                            <span className="text-gray-400">Resource Isolation</span>
                                            <span className="text-green-400 font-bold bg-green-900/20 px-2 rounded border border-green-800/50">Lvl-4 Secured</span>
                                        </div>
                                        <div className="flex justify-between items-center text-xs">
                                            <span className="text-gray-400">Auto-Scaling Threshold</span>
                                            <span className="text-cyan-300 font-mono">85% Neural Load</span>
                                        </div>
                                        <div className="flex justify-between items-center text-xs">
                                            <span className="text-gray-400">Redundancy Factor</span>
                                            <span className="text-purple-300 font-mono">3x Geographical</span>
                                        </div>
                                    </div>
                                </div>

                                <div className="flex-grow bg-gradient-to-br from-cyan-900/20 to-purple-900/10 border border-cyan-800/50 rounded-lg p-4 flex flex-col justify-center items-center text-center group overflow-hidden relative">
                                    <div className="absolute inset-0 bg-white/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                                    <ShieldCheckIcon className="w-10 h-10 text-cyan-400 mb-3 group-hover:scale-110 transition-transform duration-500" />
                                    <h4 className="text-xs font-bold text-white uppercase tracking-widest mb-1">EKS Encryption Map</h4>
                                    <p className="text-[9px] text-cyan-600 max-w-[180px]">Active Entangled Key State verified across all 15 Gateway Nodes.</p>
                                    <button className="mt-4 px-4 py-1.5 bg-black/60 border border-cyan-700 rounded text-[9px] font-bold text-cyan-300 hover:border-cyan-300 transition-all uppercase tracking-widest">Rotate Global Key</button>
                                </div>
                            </div>
                        </div>
                    </>
                ) : (
                    <div className="h-full flex flex-col items-center justify-center text-cyan-800 text-center gap-4">
                        <div className="relative">
                            <div className="absolute inset-0 bg-cyan-500/10 blur-3xl animate-pulse"></div>
                            <ServerCogIcon className="w-16 h-16 opacity-30 relative z-10" />
                        </div>
                        <div>
                            <h3 className="text-lg font-black text-white uppercase tracking-[0.2em]">Application Selection Required</h3>
                            <p className="text-xs max-w-xs mx-auto">Please select a deployed application instance from the instance list to access its dedicated Back Office terminal.</p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};


// --- Main Layout Component ---
const CHIPSBackOffice: React.FC<CHIPSBackOfficeProps> = ({ uriAssignments, marketApps }) => {
    const [activeTab, setActiveTab] = useState<number>(5);

    const tabs = [
        { id: 1, label: 'Decentralized Nodes', icon: ServerCogIcon },
        { id: 2, label: 'Gateway & Hosting', icon: CloudServerIcon },
        { id: 3, label: 'Domain Registry', icon: GlobeIcon },
        { id: 4, label: 'App Store Admin', icon: BoxIcon },
        { id: 5, label: 'Unified App Back Office', icon: RocketLaunchIcon },
    ];

    return (
        <div className="flex flex-col h-full space-y-4 overflow-hidden">
            {/* Main Title Bar */}
            <div className="flex items-center justify-between pb-2 border-b border-cyan-800/30 flex-shrink-0">
                <h2 className="text-lg font-bold text-white tracking-widest flex items-center">
                    <NetworkIcon className="w-5 h-5 mr-2 text-red-400" /> Chips Back Office
                </h2>
                <div className="flex items-center gap-2 text-xs text-cyan-500">
                    <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                    MESH SECURE
                </div>
            </div>

            {/* Navigation Tabs */}
            <div className="flex space-x-1 bg-black/40 rounded p-1 flex-shrink-0 overflow-x-auto no-scrollbar border border-cyan-900/50">
                {tabs.map(tab => (
                    <button 
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`px-3 py-2 text-[10px] font-bold rounded transition-colors flex items-center gap-2 whitespace-nowrap uppercase tracking-tighter ${
                            activeTab === tab.id 
                                ? 'bg-cyan-700 text-white shadow-[0_0_10px_theme(colors.cyan.900)]' 
                                : 'text-cyan-500 hover:text-cyan-300 hover:bg-white/5'
                        }`}
                    >
                        <tab.icon className="w-4 h-4" />
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Content Area - STRICTLY OVERFLOW HIDDEN */}
            <div className="flex-grow min-h-0 overflow-hidden relative p-1">
                {activeTab === 1 && <NodeOperationsView />}
                {activeTab === 2 && <GatewayHostingView apps={marketApps} />}
                {activeTab === 3 && <DomainRegistryView uriAssignments={uriAssignments} />}
                {activeTab === 4 && <CHIPSStoreAdmin liveApps={marketApps.filter(a => a.status === 'installed')} />}
                {activeTab === 5 && <UnifiedAppBackOffice uriAssignments={uriAssignments} marketApps={marketApps} />}
            </div>
        </div>
    );
};

export default CHIPSBackOffice;
