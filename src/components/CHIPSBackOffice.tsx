
import React, { useState, useMemo, useEffect, useRef, Suspense } from 'react';
import { 
    ServerCogIcon, NetworkIcon, GlobeIcon, BoxIcon, 
    CpuChipIcon, ActivityIcon, ShieldCheckIcon, LinkIcon,
    HardDriveIcon, CloudServerIcon, ArrowRightIcon,
    ChartBarIcon, UsersIcon, SearchIcon, RocketLaunchIcon,
    CheckCircle2Icon, AlertTriangleIcon, ClockIcon, LockIcon,
    ArrowTopRightOnSquareIcon, TerminalIcon, StopIcon, PlayIcon,
    RefreshCwIcon, ZapIcon, ScaleIcon, TrashIcon,
    CodeBracketIcon, BanknotesIcon, BrainCircuitIcon
} from './Icons';
import { URIAssignment, AppDefinition, UIStructure } from '../types';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import GlassPanel from './GlassPanel';

// Local Component Imports via Relative Paths
import CHIPSStoreAdmin from './CHIPSStoreAdmin';
import MergedEvolutionPanel from './MergedEvolutionPanel';
import ChipsEconomy from './ChipsEconomy';
import SecurityMonitorAndSimulator from './SecurityMonitorAndSimulator';
import QuantumCognitiveArchitecture from './QuantumCognitiveArchitecture';
import CHIPSGatewayAdmin from './CHIPSGatewayAdmin';

// Lazy load ChipsDevPlatform
const ChipsDevPlatform = React.lazy(() => import('./ChipsDevPlatform'));

interface CHIPSBackOfficeProps {
  uriAssignments: URIAssignment[];
  marketApps: AppDefinition[];
  onApplyPatch?: (file: string, content: string) => void;
  onMaximizeSubPanel?: (id: string) => void;
  onAiAssist?: (currentCode: string, instruction: string) => Promise<string>;
  onDeploy?: (details: { name: string; description: string; code: string; uiStructure?: UIStructure }) => void;
  systemHealth?: any;
}

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
                <div className="flex-grow overflow-auto custom-scrollbar">
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

const UnifiedAppBackOffice = ({ uriAssignments, marketApps }: { uriAssignments: URIAssignment[]; marketApps: AppDefinition[] }) => {
    const nativeApps = marketApps.filter(a => a.status === 'installed').slice(0, 9);

    return (
        <div className="h-full flex flex-col gap-4 animate-fade-in overflow-hidden">
             <div className="bg-gradient-to-r from-purple-900/20 to-cyan-900/20 p-4 rounded-lg border border-cyan-500/30 flex items-center justify-between">
                <div>
                    <h3 className="text-lg font-bold text-white flex items-center gap-2">
                        <RocketLaunchIcon className="w-6 h-6 text-purple-400" /> Unified Quantum Apps Back Office
                    </h3>
                    <p className="text-xs text-cyan-200">Centralized management for native QCOS administration tools.</p>
                </div>
                <div className="text-right">
                    <p className="text-xs text-purple-400 uppercase tracking-widest">Active Native Apps</p>
                    <p className="text-2xl font-mono text-white">{nativeApps.length}</p>
                </div>
            </div>

            <div className="flex-grow min-h-0 bg-black/20 rounded-lg border border-cyan-800/50 p-4 overflow-y-auto custom-scrollbar">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {nativeApps.map((app, index) => (
                        <div key={app.id} className="bg-black/40 border border-cyan-900/50 p-3 rounded-lg hover:bg-cyan-900/20 transition-all group">
                            <div className="flex items-center gap-3 mb-2">
                                <div className="p-2 bg-cyan-950/40 rounded border border-cyan-800 group-hover:border-cyan-500 transition-colors">
                                    <app.icon className="w-5 h-5 text-cyan-400" />
                                </div>
                                <div className="min-w-0">
                                    <h4 className="font-bold text-sm text-white truncate">{app.name}</h4>
                                    <p className="text-[10px] text-gray-500 uppercase tracking-wider">Native Module {index + 1}</p>
                                </div>
                            </div>
                            <div className="flex justify-between items-center text-[10px] mt-2 pt-2 border-t border-cyan-900/30">
                                <span className="text-green-400 flex items-center gap-1"><div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"/> Online</span>
                                <button className="text-cyan-500 hover:text-white">Manage</button>
                            </div>
                        </div>
                    ))}
                    {nativeApps.length === 0 && (
                        <div className="col-span-full text-center text-gray-500 py-8">
                            No native apps active. Check App Store or Deployment status.
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

// --- Main Layout Component ---
const CHIPSBackOffice: React.FC<CHIPSBackOfficeProps> = ({ 
    uriAssignments, 
    marketApps, 
    onApplyPatch, 
    onMaximizeSubPanel,
    onAiAssist,
    onDeploy,
    systemHealth
}) => {
    const [activeTab, setActiveTab] = useState<string>('agile-ops');

    const tabs = [
        { id: 'agile-ops', label: 'Agile Ops', icon: RocketLaunchIcon },
        { id: 'unified-apps', label: 'Unified Apps', icon: BoxIcon },
        { id: 'browser-dqn', label: 'Browser DQN', icon: GlobeIcon },
        { id: 'app-store', label: 'Store Admin', icon: ShieldCheckIcon },
        { id: 'dev-platform', label: 'Dev Platform', icon: CodeBracketIcon },
        { id: 'economy', label: 'Economy', icon: BanknotesIcon },
        { id: 'security', label: 'Security', icon: ShieldCheckIcon },
        { id: 'cognitive', label: 'Cognitive', icon: BrainCircuitIcon },
        { id: 'gateway', label: 'Gateway', icon: NetworkIcon },
    ];

    return (
        <div className="flex flex-col h-full space-y-4 overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between pb-2 border-b border-cyan-800/30 flex-shrink-0">
                <h2 className="text-lg font-bold text-white tracking-widest flex items-center">
                    <ServerCogIcon className="w-5 h-5 mr-2 text-purple-400" /> Chips Back Office
                </h2>
                <div className="flex items-center gap-2 text-xs text-cyan-500">
                    <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                    SYSTEM ONLINE
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
                                ? 'bg-purple-700 text-white shadow-[0_0_10px_theme(colors.purple.900)]' 
                                : 'text-cyan-500 hover:text-cyan-300 hover:bg-white/5'
                        }`}
                    >
                        <tab.icon className="w-4 h-4" />
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Content Area - STRICTLY OVERFLOW HIDDEN */}
            <div className="flex-grow min-h-0 overflow-hidden relative p-1 bg-black/20 rounded-lg border border-cyan-900/30">
                {activeTab === 'agile-ops' && (
                    <MergedEvolutionPanel onApplyPatch={onApplyPatch} onMaximizeSubPanel={onMaximizeSubPanel} />
                )}
                {activeTab === 'unified-apps' && (
                    <UnifiedAppBackOffice uriAssignments={uriAssignments} marketApps={marketApps} />
                )}
                {activeTab === 'browser-dqn' && (
                    <NodeOperationsView />
                )}
                {activeTab === 'app-store' && (
                    <CHIPSStoreAdmin liveApps={marketApps.filter(a => a.status === 'installed')} />
                )}
                {activeTab === 'dev-platform' && (
                    <Suspense fallback={<div className="h-full flex items-center justify-center text-cyan-500">Loading Dev Platform...</div>}>
                         <ChipsDevPlatform onAiAssist={onAiAssist} onDeploy={onDeploy} />
                    </Suspense>
                )}
                {activeTab === 'economy' && (
                    <ChipsEconomy />
                )}
                {activeTab === 'security' && (
                    <SecurityMonitorAndSimulator onMaximize={() => onMaximizeSubPanel?.('security-monitor')} />
                )}
                {activeTab === 'cognitive' && (
                    <QuantumCognitiveArchitecture />
                )}
                {activeTab === 'gateway' && (
                    <CHIPSGatewayAdmin uriAssignments={uriAssignments} />
                )}
            </div>
        </div>
    );
};

export default CHIPSBackOffice;
