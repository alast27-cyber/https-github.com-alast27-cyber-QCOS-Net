
import React, { useState } from 'react';
import GlassPanel from './GlassPanel';
import { 
    ServerStackIcon, 
    GlobeIcon, 
    BoxIcon, 
    ActivityIcon, 
    RocketLaunchIcon, 
    RefreshCwIcon, 
    ShieldCheckIcon, 
    CpuChipIcon, 
    ChartBarIcon, 
    ServerCogIcon, 
    DownloadCloudIcon, 
    MaximizeIcon, 
    ChevronLeftIcon, 
    CheckCircle2Icon
} from './Icons';
import { URIAssignment } from '../types';

interface ChipsQuantumInternetPanelProps {
    uriAssignments: URIAssignment[];
}

const ItemCard: React.FC<{ 
    id: string;
    title: string; 
    icon: React.FC<{className?: string}>; 
    desc: string; 
    status?: string; 
    action?: string;
    color?: string;
    onAction?: (id: string) => void;
}> = ({ id, title, icon: Icon, desc, status, action, color = 'cyan', onAction }) => (
    <div className={`p-3 rounded-lg border border-${color}-500/30 bg-black/40 hover:bg-${color}-900/20 transition-all duration-200 group flex flex-col h-full relative overflow-hidden`}>
        <div className={`absolute top-0 right-0 p-2 opacity-0 group-hover:opacity-100 transition-opacity`}>
             <MaximizeIcon className={`w-4 h-4 text-${color}-400`} />
        </div>
        <div className="flex justify-between items-start mb-2">
            <div className={`p-2 rounded-md bg-${color}-900/50 text-${color}-300`}>
                <Icon className="w-5 h-5" />
            </div>
            {status && <span className="text-[10px] uppercase font-bold tracking-wider text-green-400 bg-green-900/30 px-2 py-0.5 rounded-full">{status}</span>}
        </div>
        <h4 className="text-sm font-bold text-white mb-1">{title}</h4>
        <p className="text-xs text-gray-400 mb-3 flex-grow">{desc}</p>
        {action && (
            <button 
                onClick={() => onAction && onAction(id)}
                className={`w-full py-1.5 rounded text-xs font-bold bg-${color}-500/20 text-${color}-200 border border-${color}-500/50 hover:bg-${color}-500/40 transition-colors flex items-center justify-center gap-2`}
            >
                {action} <MaximizeIcon className="w-3 h-3" />
            </button>
        )}
    </div>
);

// --- Sub-Panel Content Placeholders ---

const SubPanelContent: React.FC<{ id: string; onBack: () => void; color?: string }> = ({ id, onBack, color = 'cyan' }) => {
    const [installCounts] = useState(() => Array.from({ length: 7 }, () => Math.floor(Math.random() * 500)));
    // Determine content based on ID
    let content = <div className="text-center text-gray-500 mt-10">Content Module Loading...</div>;
    let title = "Sub-Panel";

    if (id === 'dqn') {
        title = "Chips Decentralized Quantum Network";
        content = (
            <div className="space-y-4 animate-fade-in">
                <div className="bg-black/40 border border-cyan-900/50 p-4 rounded-lg h-48 relative overflow-hidden flex items-center justify-center">
                    <div className="absolute inset-0 holographic-grid opacity-20"></div>
                    <ServerStackIcon className="w-24 h-24 text-cyan-800/50 animate-pulse" />
                    <div className="absolute inset-0 flex items-center justify-center">
                        <p className="text-cyan-400 font-mono text-xs bg-black/60 px-2 rounded">VISUALIZING 8,432 NODES</p>
                    </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                    <div className="bg-cyan-950/30 p-3 rounded border border-cyan-800">
                        <p className="text-xs text-cyan-500">Active Nodes</p>
                        <p className="text-xl font-bold text-white">8,432</p>
                    </div>
                    <div className="bg-cyan-950/30 p-3 rounded border border-cyan-800">
                        <p className="text-xs text-cyan-500">Network Hashrate</p>
                        <p className="text-xl font-bold text-white">42.8 P-QFLOPS</p>
                    </div>
                </div>
                <div className="text-xs text-gray-400 font-mono">
                    <p>{">"} Initiating handshake with neighbor nodes...</p>
                    <p>{">"} Syncing ledger state...</p>
                    <p className="text-green-400">{">"} Connection Secured (Quantum Entanglement Layer)</p>
                </div>
            </div>
        );
    } else if (id === 'browser') {
        title = "Chimera Browser Download";
        content = (
            <div className="flex flex-col items-center justify-center h-full space-y-6 animate-fade-in">
                <GlobeIcon className="w-20 h-20 text-blue-400 animate-spin-slow" />
                <div className="text-center">
                    <h3 className="text-xl font-bold text-white">Chimera v4.2</h3>
                    <p className="text-sm text-blue-300">The gateway to the quantum web.</p>
                </div>
                <div className="space-y-2 w-full max-w-xs">
                    <button className="w-full py-2 bg-blue-600/40 border border-blue-500 hover:bg-blue-600/60 rounded text-white font-bold flex items-center justify-center gap-2">
                        <DownloadCloudIcon className="w-4 h-4" /> Download for Windows
                    </button>
                    <button className="w-full py-2 bg-blue-600/20 border border-blue-500/50 hover:bg-blue-600/40 rounded text-white font-bold flex items-center justify-center gap-2">
                        <DownloadCloudIcon className="w-4 h-4" /> Download for macOS
                    </button>
                    <button className="w-full py-2 bg-blue-600/20 border border-blue-500/50 hover:bg-blue-600/40 rounded text-white font-bold flex items-center justify-center gap-2">
                        <DownloadCloudIcon className="w-4 h-4" /> Download for Linux
                    </button>
                </div>
            </div>
        );
    } else if (id === 'store') {
        title = "Quantum App Store";
        content = (
            <div className="grid grid-cols-2 gap-3 animate-fade-in">
                {[1, 2, 3, 4, 5, 6].map(i => (
                    <div key={i} className="p-2 border border-purple-500/30 bg-purple-900/10 rounded flex gap-2 items-center hover:bg-purple-900/30 cursor-pointer">
                        <BoxIcon className="w-8 h-8 text-purple-400" />
                        <div>
                            <p className="text-xs font-bold text-white">Q-App {i}</p>
                            <p className="text-[10px] text-gray-400">Utility • {installCounts[i]} installs</p>
                        </div>
                    </div>
                ))}
                <div className="col-span-2 text-center text-xs text-purple-400 mt-4">
                    Browsing Decentralized Registry...
                </div>
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col">
            <div className="flex items-center gap-2 mb-4 border-b border-white/10 pb-2">
                <button onClick={onBack} className="p-1 rounded hover:bg-white/10 text-cyan-400">
                    <ChevronLeftIcon className="w-5 h-5" />
                </button>
                <h3 className="text-base font-bold text-white">{title}</h3>
            </div>
            <div className="flex-grow overflow-y-auto pr-1">
                {content}
            </div>
        </div>
    );
};


const ChipsQuantumInternetPanel: React.FC<ChipsQuantumInternetPanelProps> = ({ uriAssignments }) => {
    const [isFlipped, setIsFlipped] = useState(false);
    const [activeSubPanel, setActiveSubPanel] = useState<string | null>(null);

    const toggleSide = () => setIsFlipped(!isFlipped);

    const handleOpenSubPanel = (id: string) => {
        setActiveSubPanel(id);
    };

    const handleBack = () => {
        setActiveSubPanel(null);
    };

    if (activeSubPanel) {
        return (
            <GlassPanel title={<div className="flex items-center"><CpuChipIcon className="w-5 h-5 mr-2 text-cyan-400" /> Chips Quantum Internet</div>}>
                <div className="p-2 h-full">
                    <SubPanelContent id={activeSubPanel} onBack={handleBack} />
                </div>
            </GlassPanel>
        );
    }

    return (
        <GlassPanel 
            title={
                <div className="flex justify-between items-center w-full">
                    <div className="flex items-center">
                        {isFlipped ? <GlobeIcon className="w-5 h-5 mr-2 text-purple-400" /> : <CpuChipIcon className="w-5 h-5 mr-2 text-cyan-400" />}
                        <span>{isFlipped ? 'Chips Back Office' : 'Chips Quantum Internet'}</span>
                    </div>
                    <button 
                        onClick={toggleSide} 
                        className="text-xs flex items-center gap-1 bg-white/5 hover:bg-white/10 px-2 py-1 rounded border border-white/10 transition-colors"
                        title={isFlipped ? "Switch to Public View" : "Switch to Back Office"}
                    >
                        <RefreshCwIcon className="w-3 h-3" />
                        {isFlipped ? 'Public View' : 'Back Office'}
                    </button>
                </div>
            }
        >
            <div className="relative h-full overflow-hidden p-2">
                {/* Front Side: Chips Quantum Internet */}
                <div className={`absolute inset-0 p-2 transition-all duration-500 transform ${isFlipped ? 'translate-x-full opacity-0 pointer-events-none' : 'translate-x-0 opacity-100'}`}>
                    <div className="grid grid-cols-2 gap-3 h-full overflow-y-auto pb-4">
                        <div className="col-span-2 pb-2 border-b border-cyan-800/50 mb-1">
                            <p className="text-xs text-cyan-300">Decentralized ecosystem services and public access nodes.</p>
                        </div>

                        <ItemCard 
                            id="dqn"
                            title="Chips DQN" 
                            icon={ServerStackIcon} 
                            desc="Decentralized Quantum Nodes providing distributed compute power."
                            status="8,432 Nodes"
                            action="View Network"
                            onAction={handleOpenSubPanel}
                        />
                        
                        <ItemCard 
                            id="browser"
                            title="Chimera Browser" 
                            icon={GlobeIcon} 
                            desc="Quantum-native browsing client installers (Win/Mac/Linux)."
                            action="Download v4.2"
                            color="blue"
                            onAction={handleOpenSubPanel}
                        />

                        <ItemCard 
                            id="store"
                            title="Quantum App Store" 
                            icon={BoxIcon} 
                            desc="Marketplace for Q-Lang applications and algorithms."
                            status={`${uriAssignments.length + 12} Apps`}
                            action="Browse Store"
                            color="purple"
                            onAction={handleOpenSubPanel}
                        />

                        <ItemCard 
                            id="economy"
                            title="Chips Economy" 
                            icon={ActivityIcon} 
                            desc="Q-Credit exchange rates, staking, and resource allocation."
                            status="1 QC = 1.05 USD"
                            action="Open Wallet"
                            color="yellow"
                            onAction={handleOpenSubPanel}
                        />

                        <div className="col-span-2">
                            <ItemCard 
                                id="deployment"
                                title="Chips Deployment" 
                                icon={RocketLaunchIcon} 
                                desc="CI/CD pipelines for deploying Q-Apps to the CHIPS network."
                                action="Manage Deployments"
                                color="green"
                                onAction={handleOpenSubPanel}
                            />
                        </div>
                    </div>
                </div>

                {/* Back Side: Chips Back Office */}
                <div className={`absolute inset-0 p-2 transition-all duration-500 transform ${isFlipped ? 'translate-x-0 opacity-100' : '-translate-x-full opacity-0 pointer-events-none'}`}>
                    <div className="grid grid-cols-2 gap-3 h-full overflow-y-auto pb-4">
                        <div className="col-span-2 pb-2 border-b border-purple-800/50 mb-1">
                            <p className="text-xs text-purple-300">Administrative controls and network telemetry.</p>
                        </div>

                        <ItemCard 
                            id="backoffice-diag"
                            title="DQN Back Office" 
                            icon={CpuChipIcon} 
                            desc="Node diagnostics, peering protocols, and hardware telemetry."
                            color="purple"
                            action="Diagnostics"
                            onAction={handleOpenSubPanel}
                        />

                        <ItemCard 
                            id="backoffice-store"
                            title="CQA Store Admin" 
                            icon={ShieldCheckIcon} 
                            desc="App verification, developer console, and revenue metrics."
                            color="purple"
                            action="Manage Store"
                            onAction={handleOpenSubPanel}
                        />

                        <ItemCard 
                            id="backoffice-econ"
                            title="Economy Admin" 
                            icon={ChartBarIcon} 
                            desc="Minting controls, ledger audit, and liquidity pools."
                            color="purple"
                            action="Audit Ledger"
                            onAction={handleOpenSubPanel}
                        />

                        <ItemCard 
                            id="backoffice-deploy"
                            title="Deployment Admin" 
                            icon={ServerCogIcon} 
                            desc="Routing tables, gateway load balancing, and version control."
                            color="purple"
                            action="Configure Gateways"
                            onAction={handleOpenSubPanel}
                        />
                        
                        <div className="col-span-2 bg-purple-900/20 p-3 rounded-lg border border-purple-800/50 mt-2">
                            <div className="flex items-center gap-2 mb-2 text-purple-300">
                                <ActivityIcon className="w-4 h-4 animate-pulse" />
                                <span className="text-xs font-bold uppercase">Network Health</span>
                            </div>
                            <div className="w-full h-1 bg-gray-800 rounded-full overflow-hidden">
                                <div className="h-full bg-purple-500 w-[98%] shadow-[0_0_10px_theme(colors.purple.500)]" />
                            </div>
                            <div className="flex justify-between text-[10px] text-gray-400 mt-1 font-mono">
                                <span>UPTIME: 99.999%</span>
                                <span>LATENCY: 12ms</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </GlassPanel>
    );
};

export default ChipsQuantumInternetPanel;
