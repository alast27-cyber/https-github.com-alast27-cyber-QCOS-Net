
import React, { useState, useMemo, useEffect } from 'react';
import { 
    CloudServerIcon, GlobeIcon, RocketLaunchIcon, HardDriveIcon, 
    CheckCircle2Icon, AlertTriangleIcon, ActivityIcon, LinkIcon,
    RefreshCwIcon, LockIcon, ShieldCheckIcon, ServerCogIcon,
    ArrowTopRightOnSquareIcon, UploadCloudIcon, ChartBarIcon, HeartIcon,
    ClockIcon, ToggleLeftIcon, ToggleRightIcon, LoaderIcon, NetworkIcon,
    GitBranchIcon, KeyIcon, PlayIcon, BrainCircuitIcon, SparklesIcon,
    CubeTransparentIcon, FileCodeIcon, Share2Icon, CpuChipIcon,
    CodeBracketIcon, CommandIcon
} from './Icons';
import { URIAssignment } from '../types';
import GlassPanel from './GlassPanel';
import LoadingSkeleton from './LoadingSkeleton';

interface CHIPSGatewayAdminProps {
    uriAssignments: URIAssignment[];
}

type Tab = 'hosting' | 'domains' | 'deployments' | 'protocol' | 'network';

// --- Type Definitions ---
type AppStatus = 'Active' | 'Pending' | 'Error';

interface GatewayApp {
    id: string;
    name: string;
    chipsAddress: string;
    publicUrl: string;
    status: AppStatus;
}

// --- CHIPS Protocol Types (Based on PDF Spec) ---
interface CHIPSPacketHeader {
    protocol: 'CHIPS://';
    sourceId: string; // e.g., QAN-ROOT-001
    targetScope: string; // e.g., SCOPE::GEO::REG
    version: string; // V2.0
}

interface CHIPSControlBlock {
    timestamp: number;
    eksReference: string; // Entangled Key State ID
    integrityHash: string; // Token Verification Hash
}

interface CHIPSPacket {
    id: string;
    header: CHIPSPacketHeader;
    control: CHIPSControlBlock;
    payload: string; // Encrypted Q-Lang script
    status: 'Created' | 'Broadcasting' | 'Filtering' | 'Delivered' | 'Verifying' | 'Executed';
}

// --- Reusable Sub-components ---
const Section: React.FC<{ title: string; icon: React.FC<{className?: string}>; children: React.ReactNode }> = ({ title, icon: Icon, children }) => (
    <div className="bg-black/20 p-3 rounded-lg border border-cyan-800/50">
        <h3 className="text-base font-semibold text-cyan-200 flex items-center mb-3">
            <Icon className="w-5 h-5 mr-2" /> {title}
        </h3>
        {children}
    </div>
);

const Kpi: React.FC<{ label: string; value: string; icon: React.FC<{className?: string}> }> = ({ label, value, icon: Icon }) => (
    <div className="flex items-center">
        <Icon className="w-4 h-4 text-cyan-400 mr-2" />
        <span className="text-cyan-400 text-xs">{label}:</span>
        <span className="font-mono text-white ml-auto">{value}</span>
    </div>
);

const StatusIndicator: React.FC<{ status: AppStatus }> = ({ status }) => {
    const config = {
        Active: { text: 'Active', color: 'text-green-400', icon: <CheckCircle2Icon className="w-4 h-4" /> },
        Pending: { text: 'Pending', color: 'text-yellow-400', icon: <ClockIcon className="w-4 h-4" /> },
        Error: { text: 'Error', color: 'text-red-400', icon: <AlertTriangleIcon className="w-4 h-4" /> },
    };
    const current = config[status];
    return (
        <div className={`flex items-center gap-1 text-xs font-semibold ${current.color}`}>
            {current.icon}
            {current.text}
        </div>
    );
};

// --- Mock Data (Optimized) ---
const hostingPods = [
    { id: 'Node-172', region: 'Local LAN', load: 12, type: 'Phys-Bridge', status: 'Active', version: 'v3.2.0', ip: '172.16.1.170' },
    { id: 'pod-01', region: 'US-East', load: 45, type: 'Hybrid-Bridge', status: 'Active', version: 'v3.2.0' },
    { id: 'pod-02', region: 'EU-Central', load: 42, type: 'Hybrid-Bridge', status: 'Active', version: 'v3.2.0' },
    { id: 'pod-04', region: 'US-West', load: 41, type: 'Hybrid-Bridge', status: 'Active', version: 'v3.2.0' },
];

const initialAppsData: GatewayApp[] = [
  { id: 'app1', name: 'Global Abundance Engine', chipsAddress: 'CHIPS://gae.qcos.apps/main', publicUrl: 'https://qcos.apps.web/abundance', status: 'Active' },
  { id: 'app2', name: 'QMC: Finance', chipsAddress: 'CHIPS://qmc-finance.qcos.apps/main', publicUrl: 'https://qcos.apps.web/qmc-finance', status: 'Active' },
  { id: 'app3', name: 'Molecular Simulator', chipsAddress: 'CHIPS://mol-sim.qcos.apps/main', publicUrl: 'https://qcos.apps.web/mol-sim', status: 'Active' },
  { id: 'app4', name: 'Quantum Network Visualizer', chipsAddress: 'CHIPS://qnet-viz.qcos.apps/main', publicUrl: 'https://qcos.apps.web/qnet-viz', status: 'Active' },
  { id: 'app5', name: 'Quantum Voice Chat (Q-VOX)', chipsAddress: 'CHIPS://q-vox.qcos.apps/main', publicUrl: 'https://qcos.apps.web/q-vox', status: 'Active' },
];

const initialDataSources = [
    { id: 'ds1', name: 'Google Scholar API', url: 'https://scholar.google.com/api', schedule: 'Daily', dataTypes: ['text', 'pdf'], status: 'healthy' },
    { id: 'ds2', name: 'arXiv Pre-prints', url: 'https://arxiv.org/list/quant-ph/new', schedule: 'Hourly', dataTypes: ['text', 'pdf'], status: 'healthy' },
    { id: 'ds3', name: 'CERN Open Data', url: 'https://opendata.cern.ch', schedule: 'Weekly', dataTypes: ['csv', 'binary'], status: 'healthy' },
];

const CHIPSGatewayAdmin: React.FC<CHIPSGatewayAdminProps> = ({ uriAssignments }) => {
    const [activeTab, setActiveTab] = useState<Tab>('hosting');
    const [apps, setApps] = useState(initialAppsData);
    const [dataSources, setDataSources] = useState(initialDataSources);
    const [newApp, setNewApp] = useState({ name: '', chipsAddress: '' });
    const [newSource, setNewSource] = useState({ name: '', url: '', schedule: 'Daily' });
    const [isRegistering, setIsRegistering] = useState(false);
    const [ipWhitelisting, setIpWhitelisting] = useState(true); // Default to secure
    
    // Background Service States
    const [qcosBg, setQcosBg] = useState(true);

    // Custom Domain / Deployment States
    const [domainForm, setDomainForm] = useState({
        name: '',
        target: '',
        ssl: 'EKS-Secured'
    });
    const [deploymentLogs, setDeploymentLogs] = useState<string[]>([]);
    const [isDeployingDomain, setIsDeployingDomain] = useState(false);
    const [customDomains, setCustomDomains] = useState<{ id: string, domain: string, target: string, dsr: string, ssl: string, status: string }[]>([]);

    // --- Protocol Simulation State ---
    const [activePacket, setActivePacket] = useState<CHIPSPacket | null>(null);
    const [simLogs, setSimLogs] = useState<string[]>([]);
    const [simStep, setSimStep] = useState(0);

    // --- Network Sim State ---
    const [netUrl, setNetUrl] = useState('https://api.qcos.network/v1/quantum-state');
    const [netMethod, setNetMethod] = useState('GET');
    const [netLogs, setNetLogs] = useState<string[]>([]);
    const [netResponse, setNetResponse] = useState<string>('');
    const [netStatus, setNetStatus] = useState<'IDLE' | 'CONNECTING' | 'TRANSFER' | 'COMPLETE'>('IDLE');
    const [oscilloscopeOffsets] = useState(() => [Math.random() * 20, Math.random() * 20]);

    // Merge live URI assignments with static system domains
    const domainRecords = useMemo(() => {
        const staticDomains = [
            { id: 'sys-1', q_uri: 'chips://qmc-finance', web_domain: 'finance.qcos.io', ssl: 'EKS-Secured', traffic: '1.2M/day' },
            { id: 'sys-2', q_uri: 'chips://global-swine', web_domain: 'swine-intel.ag', ssl: 'EKS-Secured', traffic: '850k/day' },
            { id: 'sys-3', q_uri: 'chips://mol-sim', web_domain: 'molsim.science', ssl: 'EKS-Secured', traffic: '42k/day' },
            { id: 'sys-4', q_uri: 'chips://agent-q', web_domain: 'chat.agentq.ai', ssl: 'Quantum-TLS', traffic: '5.6M/day' },
        ];

        const liveDomains = uriAssignments.map((ua, index) => ({
            id: `user-${index}`,
            q_uri: ua.q_uri,
            web_domain: ua.https_url.replace('https://', ''),
            ssl: 'EKS-Secured',
            traffic: 'Live (New)'
        }));
        
        // Map custom domains to the table format
        const customMapped = customDomains.map((cd, index) => ({
            id: cd.id,
            q_uri: cd.target,
            web_domain: cd.domain,
            ssl: cd.ssl,
            traffic: cd.status === 'Active' ? 'Initializing...' : 'Pending'
        }));

        return [...customMapped, ...liveDomains, ...staticDomains];
    }, [uriAssignments, customDomains]);

    // Merge live deployments with static history
    const deployments = useMemo(() => {
        const staticDeployments = [
            { id: 'dep-x99', app: 'QMC Finance', version: '2.4.0', time: '10 mins ago', status: 'Success', stage: 'Production', method: 'GitHub/Netlify' },
            { id: 'dep-x98', app: 'Swine Foresight', version: '1.1.5', time: '1 hour ago', status: 'Success', stage: 'Production', method: 'Direct' },
            { id: 'dep-x97', app: 'Mol-Sim Toolkit', version: '0.9.1', time: '3 hours ago', status: 'Success', stage: 'Production', method: 'GitHub/Netlify' }, // Fixed status
        ];

        const liveDeployments = uriAssignments.map((ua, index) => ({
            id: `dep-live-${index}`,
            app: ua.appName,
            version: '1.0.0',
            time: ua.timestamp,
            status: 'Success',
            stage: 'Production',
            method: 'Netlify'
        }));

        return [...liveDeployments.reverse(), ...staticDeployments];
    }, [uriAssignments]);

    const handleRegisterApp = () => {
        if (newApp.name && newApp.chipsAddress.startsWith('CHIPS://') && !isRegistering) {
            setIsRegistering(true);
            const newId = `app${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            const appSlug = newApp.name.toLowerCase().replace(/[^a-z0-9-]/g, '-');
            const newAppEntry: GatewayApp = {
                id: newId,
                name: newApp.name,
                chipsAddress: newApp.chipsAddress,
                publicUrl: `https://qcos.apps.web/${appSlug}`,
                status: 'Pending'
            };
            
            setApps(prev => [...prev, newAppEntry]);
            setNewApp({ name: '', chipsAddress: '' });

            setTimeout(() => {
                setApps(prev => prev.map(app => 
                    app.id === newId ? { ...app, status: 'Active' } : app
                ));
                setIsRegistering(false);
            }, 3000);
        } else {
            alert("Please provide a valid App Name and a CHIPS Address starting with 'CHIPS://'");
        }
    };

    const handleAddDataSource = () => {
        if (newSource.name && newSource.url) {
            setDataSources(prev => [...prev, {
                id: `ds-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                name: newSource.name,
                url: newSource.url,
                schedule: newSource.schedule,
                dataTypes: ['stream'],
                status: 'healthy'
            }]);
            setNewSource({ name: '', url: '', schedule: 'Daily' });
        }
    };

    const handleDeployCustomDomain = () => {
        if (!domainForm.name.trim() || !domainForm.target.trim()) {
            alert("Please provide both a Domain Name and a Target CHIPS Address.");
            return;
        }
        setIsDeployingDomain(true);
        setDeploymentLogs([]);
        
        const steps = [
            `Initiating deployment pipeline for '${domainForm.name}'...`,
            `Binding Target: Resolving '${domainForm.target}'...`,
            `DSR: Registering Decentralized Service Record...`,
            `DSR: Record Created [ID: DSR-${Math.floor(Math.random()*10000)}]`,
            `Security: Provisioning ${domainForm.ssl} Certificate...`,
            `Security: Certificate Issued (Dilithium-3 Signed)`,
            `Routing: Propagating routes to QCOS Edge Nodes...`,
            `Success: ${domainForm.name} is now ACTIVE and pointing to ${domainForm.target}.`
        ];

        let delay = 0;
        steps.forEach((step, index) => {
            delay += 800; // Accelerated deployment
            setTimeout(() => {
                setDeploymentLogs(prev => [...prev, step]);
                if (index === steps.length - 1) {
                    setIsDeployingDomain(false);
                    setCustomDomains(prev => [...prev, {
                        id: `cust-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                        domain: domainForm.name,
                        target: domainForm.target,
                        dsr: 'Verified',
                        ssl: domainForm.ssl,
                        status: 'Active'
                    }]);
                    setDomainForm(prev => ({ ...prev, name: '', target: '' })); // Keep SSL preference
                }
            }, delay);
        });
    };

    // --- CHIPS Protocol Simulation Logic ---
    const generatePacket = () => {
        const id = Math.random().toString(36).substring(2, 11).toUpperCase();
        setSimLogs([`[QAN] Generating CHIPS Packet ${id}...`]);
        setSimStep(1);
        setActivePacket({
            id,
            header: {
                protocol: 'CHIPS://',
                sourceId: 'QAN-ROOT-001',
                targetScope: 'SCOPE::GEO::REG',
                version: 'V2.0'
            },
            control: {
                timestamp: Date.now(),
                eksReference: `EKS-${Math.floor(Math.random() * 9000) + 1000}-SECURE`,
                integrityHash: `0x${Math.random().toString(16).substring(2, 42)}`
            },
            payload: `[ENCRYPTED_QLANG_BLOB_${Math.floor(Math.random() * 1000)}]`,
            status: 'Created'
        });
    };

    useEffect(() => {
        if (activeTab === 'protocol' && activePacket) {
            const steps = [
                { s: 'Broadcasting', msg: 'QAN Broadcast: Sending packet to Regional Gateways...', delay: 1000 },
                { s: 'Filtering', msg: 'Gateway Filtering: Matching TARGET_SCOPE (SCOPE::GEO::REG)... Match Found.', delay: 2500 },
                { s: 'Delivered', msg: 'DQN Reception: Packet received at Subnet Node.', delay: 4000 },
                { s: 'Verifying', msg: 'QEP Acceptance: Checking EKS_REFERENCE against local state...', delay: 5500 },
                { s: 'Executed', msg: 'Integrity Verified. Decrypting Q-Lang payload. Execution started.', delay: 7000 }
            ];

            let timers: ReturnType<typeof setTimeout>[] = [];

            steps.forEach((step, index) => {
                const timer = setTimeout(() => {
                    setActivePacket(prev => prev ? { ...prev, status: step.s as any } : null);
                    setSimLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${step.msg}`]);
                    setSimStep(index + 2); // Steps start at 1 (Created)
                }, step.delay);
                timers.push(timer);
            });

            return () => timers.forEach(clearTimeout);
        }
    }, [activePacket?.id, activeTab]);

    // --- Network Sim Logic ---
    const addNetLog = (msg: string) => setNetLogs(prev => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev]);

    const simulateFetch = async () => {
        setNetStatus('CONNECTING');
        setNetLogs([]);
        setNetResponse('');
        
        addNetLog(`FETCH: Starting request to ${netUrl}`);
        
        setTimeout(() => {
            addNetLog('FETCH: Resolving DNS (QNS)... Resolved to 10.244.0.5');
            
            setTimeout(() => {
                addNetLog('FETCH: Establishing Secure EKS Tunnel...');
                
                setTimeout(() => {
                    setNetStatus('TRANSFER');
                    addNetLog('FETCH: Request Sent. Waiting for response...');
                    
                    setTimeout(() => {
                        addNetLog('FETCH: Response received. Status: 200 OK');
                        setNetResponse(JSON.stringify({ status: 'active', qubits: 240, coherence: 0.99 }, null, 2));
                        setNetStatus('COMPLETE');
                    }, 1200);
                }, 800);
            }, 800);
        }, 500);
    };

    const simulateXHR = () => {
         setNetStatus('CONNECTING');
         setNetLogs([]);
         setNetResponse('');
         
         addNetLog(`XHR: Initializing XMLHttpRequest...`);
         addNetLog(`XHR: Opening ${netMethod} ${netUrl}`);
         
         setTimeout(() => {
             addNetLog('XHR: readyState 1 (OPENED)');
             
             setTimeout(() => {
                 addNetLog('XHR: Sending request...');
                 addNetLog('XHR: readyState 2 (HEADERS_RECEIVED)');
                 
                 setTimeout(() => {
                     setNetStatus('TRANSFER');
                     addNetLog('XHR: readyState 3 (LOADING)');
                     
                     setTimeout(() => {
                         addNetLog('XHR: readyState 4 (DONE)');
                         addNetLog('XHR: Status 200');
                         setNetResponse(JSON.stringify({ data: "Stream buffer complete", size: "45kb" }, null, 2));
                         setNetStatus('COMPLETE');
                     }, 1000);
                 }, 800);
             }, 800);
         }, 500);
    };

    return (
        <GlassPanel title={
            <div className="flex items-center justify-between w-full">
                <div className="flex items-center">
                    <NetworkIcon className="w-5 h-5 mr-2 text-red-400" />
                    <span>Quantum-to-Web Gateway Admin</span>
                </div>
                <div className="flex items-center gap-2 mr-2">
                    <span className="text-[10px] uppercase font-bold text-purple-400 flex items-center bg-purple-900/20 px-2 py-0.5 rounded border border-purple-500/30">
                        <SparklesIcon className="w-3 h-3 mr-1" />
                        Evolutionary Optimization: Active
                    </span>
                </div>
            </div>
        }>
            <div className="flex flex-col h-full animate-fade-in">
                {/* Sub-Navigation */}
                <div className="flex space-x-1 mb-4 bg-black/20 p-1 rounded-lg border border-cyan-900/50 w-fit mx-auto overflow-x-auto no-scrollbar">
                    <button 
                        onClick={() => setActiveTab('hosting')}
                        className={`px-4 py-1.5 text-xs font-bold rounded transition-colors flex items-center gap-2 whitespace-nowrap ${activeTab === 'hosting' ? 'bg-purple-600/40 text-white border border-purple-500/50' : 'text-cyan-500 hover:bg-white/5'}`}
                    >
                        <CloudServerIcon className="w-4 h-4" /> Hybrid Hosting
                    </button>
                    <button 
                        onClick={() => setActiveTab('domains')}
                        className={`px-4 py-1.5 text-xs font-bold rounded transition-colors flex items-center gap-2 whitespace-nowrap ${activeTab === 'domains' ? 'bg-purple-600/40 text-white border border-purple-500/50' : 'text-cyan-500 hover:bg-white/5'}`}
                    >
                        <GlobeIcon className="w-4 h-4" /> Domain Registry
                    </button>
                    <button 
                        onClick={() => setActiveTab('deployments')}
                        className={`px-4 py-1.5 text-xs font-bold rounded transition-colors flex items-center gap-2 whitespace-nowrap ${activeTab === 'deployments' ? 'bg-purple-600/40 text-white border border-purple-500/50' : 'text-cyan-500 hover:bg-white/5'}`}
                    >
                        <RocketLaunchIcon className="w-4 h-4" /> Deployments
                    </button>
                    <button 
                        onClick={() => setActiveTab('protocol')}
                        className={`px-4 py-1.5 text-xs font-bold rounded transition-colors flex items-center gap-2 whitespace-nowrap ${activeTab === 'protocol' ? 'bg-purple-600/40 text-white border border-purple-500/50' : 'text-cyan-500 hover:bg-white/5'}`}
                    >
                        <CubeTransparentIcon className="w-4 h-4" /> CHIPS Protocol
                    </button>
                    <button 
                        onClick={() => setActiveTab('network')}
                        className={`px-4 py-1.5 text-xs font-bold rounded transition-colors flex items-center gap-2 whitespace-nowrap ${activeTab === 'network' ? 'bg-purple-600/40 text-white border border-purple-500/50' : 'text-cyan-500 hover:bg-white/5'}`}
                    >
                        <CommandIcon className="w-4 h-4" /> Network Sim
                    </button>
                </div>

                {/* Content Area */}
                <div className="flex-grow overflow-y-auto pr-2 min-h-0 space-y-4">
                    
                    {/* --- HOSTING TAB --- */}
                    {activeTab === 'hosting' && (
                        <div className="space-y-4">
                            {/* Background Services Control */}
                            <div className="bg-gradient-to-r from-slate-900 to-black p-4 rounded-lg border border-cyan-800/50 shadow-md">
                                <h3 className="text-sm font-bold text-white mb-3 flex items-center">
                                    <ServerCogIcon className="w-4 h-4 mr-2 text-cyan-400" /> Background Services
                                </h3>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div className="flex items-center justify-between bg-black/40 p-3 rounded border border-cyan-900/30">
                                        <div className="flex items-center gap-3">
                                            <ActivityIcon className={`w-5 h-5 ${qcosBg ? 'text-green-400 animate-pulse' : 'text-gray-500'}`} />
                                            <div>
                                                <p className="text-sm font-semibold text-gray-200">QCOS Dashboard Daemon</p>
                                                <p className="text-[10px] text-gray-500">Maintains global state & simulation</p>
                                            </div>
                                        </div>
                                        <button onClick={() => setQcosBg(!qcosBg)} className={`text-2xl ${qcosBg ? 'text-green-500' : 'text-gray-600'}`}>
                                            {qcosBg ? <ToggleRightIcon className="w-8 h-8" /> : <ToggleLeftIcon className="w-8 h-8" />}
                                        </button>
                                    </div>
                                    <div className="flex items-center justify-between bg-black/40 p-3 rounded border border-green-900/30">
                                        <div className="flex items-center gap-3">
                                            <BrainCircuitIcon className="w-5 h-5 text-green-400" />
                                            <div>
                                                <p className="text-sm font-semibold text-gray-200">Neural Evolution Core</p>
                                                <p className="text-sm text-gray-500">Self-Optimizing Heuristics Active</p>
                                            </div>
                                        </div>
                                        <div className="text-green-500 font-bold text-xs uppercase tracking-wider">Online</div>
                                    </div>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                <div className="bg-black/20 p-3 rounded-lg border border-purple-900/50 text-center">
                                    <p className="text-xs text-purple-400 uppercase tracking-widest mb-1">Gateway Nodes</p>
                                    <p className="text-2xl font-mono text-white">15</p>
                                </div>
                                <div className="bg-black/20 p-3 rounded-lg border border-purple-900/50 text-center">
                                    <p className="text-xs text-purple-400 uppercase tracking-widest mb-1">Global Latency</p>
                                    <p className="text-2xl font-mono text-green-400">18ms</p>
                                </div>
                                <div className="bg-black/20 p-3 rounded-lg border border-purple-900/50 text-center">
                                    <p className="text-xs text-purple-400 uppercase tracking-widest mb-1">Q-Web Requests</p>
                                    <p className="text-2xl font-mono text-cyan-300">8.2M</p>
                                </div>
                                <div className="bg-black/20 p-3 rounded-lg border border-purple-900/50 text-center">
                                    <p className="text-xs text-purple-400 uppercase tracking-widest mb-1">Efficiency</p>
                                    <p className="text-2xl font-mono text-white">99.9%</p>
                                </div>
                            </div>

                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                                {/* Left Column: Apps & Security */}
                                <div className="flex flex-col gap-4">
                                    <Section title="Application Registrations" icon={LinkIcon}>
                                        <div className="space-y-3 max-h-64 overflow-y-auto pr-2 -mr-2">
                                            {apps.map(app => (
                                                <div key={app.id} className="bg-cyan-950/30 p-2 rounded-md">
                                                    <p className="font-bold text-white">{app.name}</p>
                                                    <p className="text-xs text-cyan-500 font-mono break-all">{app.chipsAddress}</p>
                                                    <div className="flex items-center justify-between mt-2">
                                                        <StatusIndicator status={app.status} />
                                                        {app.status === 'Active' && <a href={app.publicUrl} target="_blank" rel="noopener noreferrer" className="text-xs text-blue-300 hover:underline break-all block">{app.publicUrl}</a>}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                        <div className="border-t border-cyan-800/50 pt-3 mt-3">
                                            <h4 className="text-sm font-semibold text-cyan-300 flex items-center mb-2">
                                                <RocketLaunchIcon className="w-4 h-4 mr-2" /> Register New Application
                                            </h4>
                                            <div className="space-y-2">
                                                <input type="text" placeholder="Application Name" value={newApp.name} onChange={e => setNewApp(s => ({...s, name: e.target.value}))} disabled={isRegistering} className="w-full p-1 bg-black/50 border border-cyan-800 rounded-md text-white text-xs" />
                                                <input type="text" placeholder="CHIPS Address (e.g., CHIPS://...)" value={newApp.chipsAddress} onChange={e => setNewApp(s => ({...s, chipsAddress: e.target.value}))} disabled={isRegistering} className="w-full p-1 bg-black/50 border border-cyan-800 rounded-md text-white text-xs" />
                                                <button onClick={handleRegisterApp} disabled={isRegistering} title="Register and deploy a new application to the gateway." className="holographic-button text-xs px-3 py-1.5 rounded w-full flex items-center justify-center gap-2">
                                                    {isRegistering ? <><LoaderIcon className="w-4 h-4 animate-spin"/> Registering...</> : 'Register & Deploy'}
                                                </button>
                                            </div>
                                        </div>
                                    </Section>

                                    <Section title="Security & Access Control" icon={ShieldCheckIcon}>
                                        <div className="space-y-3">
                                            <div className="flex items-center justify-between">
                                                <span className="text-cyan-300">Enable IP Whitelisting</span>
                                                <button onClick={() => setIpWhitelisting(p => !p)} title="Toggle IP whitelisting for enhanced security" className={ipWhitelisting ? 'text-green-400' : 'text-gray-400'}>
                                                    {ipWhitelisting ? <ToggleRightIcon className="w-10 h-10 -m-2" /> : <ToggleLeftIcon className="w-10 h-10 -m-2" />}
                                                </button>
                                            </div>
                                            {ipWhitelisting && <input type="text" value="10.0.0.0/8 (Internal Only)" disabled className="w-full p-1 bg-black/50 border border-cyan-800 rounded-md text-white text-xs" />}
                                            <p className="text-xs text-cyan-600 text-center pt-2 border-t border-cyan-900">All access is governed by QCOS authentication protocols.</p>
                                        </div>
                                    </Section>
                                </div>

                                {/* Right Column: Pods & Metrics */}
                                <div className="flex flex-col gap-4">
                                    <h3 className="text-sm font-semibold text-cyan-300 border-b border-cyan-900 pb-2">Active Bridge Pods</h3>
                                    <div className="space-y-3">
                                        {hostingPods.map(pod => (
                                            <div key={pod.id} className="bg-cyan-950/20 p-3 rounded-lg border border-cyan-800/50 flex items-center justify-between">
                                                <div className="flex items-center gap-3">
                                                    <div className={`p-2 rounded-full ${pod.status === 'Active' ? 'bg-green-900/30 text-green-400' : pod.status === 'Warning' ? 'bg-red-900/30 text-red-400' : 'bg-yellow-900/30 text-yellow-400'}`}>
                                                        <HardDriveIcon className="w-5 h-5" />
                                                    </div>
                                                    <div>
                                                        <p className="font-bold text-sm text-white">{pod.id}</p>
                                                        <p className="text-xs text-cyan-500">{pod.region} • {pod.type}</p>
                                                        {(pod as any).ip && <p className="text-[10px] text-green-300 font-mono mt-0.5">{(pod as any).ip}</p>}
                                                    </div>
                                                </div>
                                                <div className="text-right">
                                                    <div className="flex items-center justify-end gap-1 mb-1">
                                                        <ActivityIcon className="w-3 h-3 text-cyan-600" />
                                                        <span className={`text-xs font-mono ${pod.load > 90 ? 'text-red-400' : 'text-cyan-300'}`}>{pod.load}% Load</span>
                                                    </div>
                                                    <span className={`text-[10px] px-2 py-0.5 rounded-full border ${pod.status === 'Active' ? 'border-green-800 bg-green-900/20 text-green-400' : pod.status === 'Warning' ? 'border-red-800 bg-red-900/20 text-red-400' : 'border-yellow-800 bg-yellow-900/20 text-yellow-400'}`}>
                                                        {pod.status}
                                                    </span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>

                                    <Section title="Performance Metrics" icon={ChartBarIcon}>
                                        <div className="space-y-2">
                                            <Kpi label="Data Ingested" value="1.2 TB/day" icon={UploadCloudIcon} />
                                            <Kpi label="Public App Requests" value="~1.5k/min" icon={GlobeIcon} />
                                            <Kpi label="Average Latency" value="18 ms" icon={ClockIcon} />
                                        </div>
                                    </Section>
                                    <Section title="Data Source Health" icon={HeartIcon}>
                                        <div className="space-y-2">
                                            <div className="flex items-center justify-between text-green-400"><p>Active Sources</p><p className="font-mono font-bold">{dataSources.filter(d=>d.status === 'healthy').length}</p></div>
                                            <div className="flex items-center justify-between text-red-400"><p>Sources with Errors</p><p className="font-mono font-bold">{dataSources.filter(d=>d.status === 'error').length}</p></div>
                                        </div>
                                    </Section>

                                    <Section title="AGI Data Source Configuration" icon={UploadCloudIcon}>
                                         <div className="grid grid-cols-2 gap-2 mb-2">
                                             <input type="text" placeholder="Source Name" value={newSource.name} onChange={e => setNewSource(s => ({...s, name: e.target.value}))} className="w-full p-1 bg-black/50 border border-cyan-800 rounded-md text-white text-xs" />
                                             <input type="text" placeholder="URL / Endpoint" value={newSource.url} onChange={e => setNewSource(s => ({...s, url: e.target.value}))} className="w-full p-1 bg-black/50 border border-cyan-800 rounded-md text-white text-xs" />
                                         </div>
                                         <div className="flex gap-2 mb-2">
                                            <select value={newSource.schedule} onChange={e => setNewSource(s => ({...s, schedule: e.target.value}))} className="w-1/3 p-1 bg-black/50 border border-cyan-800 rounded-md text-white text-xs">
                                                <option>Real-time</option>
                                                <option>Continuous</option>
                                                <option>Hourly</option>
                                                <option>Daily</option>
                                                <option>Weekly</option>
                                            </select>
                                            <button onClick={handleAddDataSource} title="Add and connect a new data source for AGI training." className="holographic-button text-xs px-3 py-1 rounded flex-grow flex items-center justify-center gap-2">
                                                <UploadCloudIcon className="w-3 h-3" /> Connect Source
                                            </button>
                                         </div>
                                         
                                         <div className="mt-3 space-y-2 max-h-32 overflow-y-auto pr-1 custom-scrollbar bg-black/20 p-1 rounded">
                                             {dataSources.map(ds => (
                                                <div key={ds.id} className="flex items-center justify-between text-xs bg-cyan-950/30 p-1.5 rounded-md border border-cyan-900/30">
                                                    <div className="flex items-center gap-2 overflow-hidden">
                                                        <div className={`w-2 h-2 rounded-full flex-shrink-0 ${ds.status === 'healthy' ? 'bg-green-400' : ds.status === 'warning' ? 'bg-yellow-400' : 'bg-red-400'}`} title={ds.status}></div>
                                                        <div className="flex flex-col min-w-0">
                                                            <span className="text-white truncate font-bold">{ds.name}</span>
                                                            <span className="text-cyan-600 truncate text-[9px]">{ds.url}</span>
                                                        </div>
                                                    </div>
                                                    <span className="text-cyan-500 capitalize bg-black/30 px-1.5 rounded text-[9px]">{ds.schedule}</span>
                                                </div>
                                             ))}
                                         </div>
                                    </Section>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* --- DOMAINS TAB --- */}
                    {activeTab === 'domains' && (
                        <div className="space-y-4">
                            
                            {/* Domain Registration & Deployment Wizard */}
                            <div className="bg-gradient-to-r from-blue-900/20 to-purple-900/20 p-4 rounded-lg border border-blue-500/30 flex flex-col gap-3">
                                <div className="flex items-center gap-3">
                                    <div className="p-2 bg-blue-500/20 rounded-full text-blue-300"><GlobeIcon className="w-6 h-6" /></div>
                                    <div>
                                        <h4 className="text-sm font-bold text-white">Register External Domain</h4>
                                        <p className="text-xs text-blue-200">Deploy QCOS Apps to custom domains (e.g., Chips.iai) with DSR Verification & SSL.</p>
                                    </div>
                                </div>
                                
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-2">
                                    <div className="flex flex-col">
                                        <label className="text-[10px] text-blue-300 mb-1">Domain Name</label>
                                        <input 
                                            type="text" 
                                            placeholder="e.g. myapp.qcos.io" 
                                            value={domainForm.name}
                                            onChange={(e) => setDomainForm(p => ({ ...p, name: e.target.value }))}
                                            className="w-full bg-black/50 border border-blue-700/50 rounded p-2 pl-3 text-sm text-white focus:border-blue-400 focus:outline-none placeholder-blue-700/50"
                                            disabled={isDeployingDomain}
                                        />
                                    </div>
                                    <div className="flex flex-col">
                                        <label className="text-[10px] text-blue-300 mb-1">Target CHIPS Address</label>
                                        <input 
                                            type="text" 
                                            placeholder="CHIPS://app.qcos/main" 
                                            value={domainForm.target}
                                            onChange={(e) => setDomainForm(p => ({ ...p, target: e.target.value }))}
                                            className="w-full bg-black/50 border border-blue-700/50 rounded p-2 pl-3 text-sm text-white focus:border-blue-400 focus:outline-none placeholder-blue-700/50"
                                            disabled={isDeployingDomain}
                                        />
                                    </div>
                                    <div className="flex flex-col">
                                        <label className="text-[10px] text-blue-300 mb-1">SSL Security</label>
                                        <select 
                                            value={domainForm.ssl}
                                            onChange={(e) => setDomainForm(p => ({ ...p, ssl: e.target.value }))}
                                            className="w-full bg-black/50 border border-blue-700/50 rounded p-2 text-sm text-white focus:border-blue-400 focus:outline-none"
                                            disabled={isDeployingDomain}
                                        >
                                            <option>EKS-Secured</option>
                                            <option>Quantum-TLS</option>
                                            <option>Standard HTTPS</option>
                                        </select>
                                    </div>
                                </div>
                                <button 
                                    onClick={handleDeployCustomDomain}
                                    disabled={isDeployingDomain || !domainForm.name || !domainForm.target}
                                    className="holographic-button px-4 py-2 bg-blue-600/30 border-blue-500/50 text-blue-200 text-xs font-bold rounded flex items-center justify-center gap-2 hover:bg-blue-600/50 disabled:opacity-50"
                                >
                                    {isDeployingDomain ? <LoaderIcon className="w-4 h-4 animate-spin"/> : <RocketLaunchIcon className="w-4 h-4"/>}
                                    {isDeployingDomain ? "Deploying & Verifying..." : "Deploy Custom Domain"}
                                </button>

                                {/* Deployment Logs */}
                                {deploymentLogs.length > 0 && (
                                    <div className="mt-2 bg-black/60 rounded border border-blue-900/50 p-3 h-32 overflow-y-auto font-mono text-[10px] text-gray-300 custom-scrollbar">
                                        {deploymentLogs.map((log, i) => (
                                            <div key={i} className="mb-1 animate-fade-in flex gap-2">
                                                <span className="text-blue-500">[{new Date().toLocaleTimeString()}]</span>
                                                <span>{log}</span>
                                            </div>
                                        ))}
                                        {isDeployingDomain && <div className="animate-pulse text-blue-400">_</div>}
                                    </div>
                                )}
                            </div>

                            <div className="bg-purple-900/10 p-3 rounded-lg border border-purple-500/30 flex items-start gap-3">
                                <ShieldCheckIcon className="w-6 h-6 text-purple-400 flex-shrink-0" />
                                <div>
                                    <h4 className="text-sm font-bold text-white">Quantum Name Service (QNS)</h4>
                                    <p className="text-xs text-purple-200 mt-1">
                                        QNS maps immutable CHIPS addresses to traditional DNS, secured by Entangled Key State (EKS) certificates.
                                        This creates a verified "Bridge" between the Quantum and Classical web.
                                    </p>
                                </div>
                            </div>

                            <div className="overflow-x-auto bg-black/20 rounded-lg border border-cyan-800/50">
                                <table className="w-full text-xs text-left">
                                    <thead className="text-cyan-500 bg-cyan-950/30 uppercase font-mono">
                                        <tr>
                                            <th className="p-3">Web Domain (DNS)</th>
                                            <th className="p-3">Target CHIPS URI</th>
                                            <th className="p-3">Security</th>
                                            <th className="p-3 text-right">Traffic</th>
                                            <th className="p-3 text-center">Action</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-cyan-900/30">
                                        {domainRecords.map(dom => (
                                            <tr key={dom.id} className="hover:bg-cyan-900/10 transition-colors">
                                                <td className="p-3 font-medium text-white flex items-center gap-2">
                                                    <GlobeIcon className="w-3 h-3 text-purple-400" />
                                                    {dom.web_domain}
                                                </td>
                                                <td className="p-3 text-cyan-300 font-mono">{dom.q_uri}</td>
                                                <td className="p-3">
                                                    <span className="flex items-center gap-1 text-green-400 bg-green-900/20 px-2 py-0.5 rounded border border-green-800/50 w-fit">
                                                        <LockIcon className="w-3 h-3" /> {dom.ssl}
                                                    </span>
                                                </td>
                                                <td className="p-3 text-right text-cyan-200 font-mono">{dom.traffic}</td>
                                                <td className="p-3 text-center">
                                                    <div className="flex justify-center gap-2">
                                                        <a 
                                                            href={`https://${dom.web_domain}`} 
                                                            target="_blank" 
                                                            rel="noopener noreferrer"
                                                            className="text-cyan-500 hover:text-white transition-colors"
                                                            title="Open Site"
                                                        >
                                                            <ArrowTopRightOnSquareIcon className="w-4 h-4" />
                                                        </a>
                                                    </div>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {/* --- DEPLOYMENTS TAB --- */}
                    {activeTab === 'deployments' && (
                        <div className="space-y-4">
                            <div className="flex justify-between items-center pb-2 border-b border-cyan-800/50">
                                <h3 className="text-sm font-semibold text-cyan-300">Pipeline Activity</h3>
                                <button className="holographic-button px-3 py-1 text-xs flex items-center gap-2 bg-green-600/20 text-green-300 border-green-500/50">
                                    <RefreshCwIcon className="w-3 h-3" /> Trigger Build
                                </button>
                            </div>

                            <div className="space-y-3">
                                {deployments.map(dep => (
                                    <div key={dep.id} className="relative bg-black/30 p-3 rounded-lg border border-cyan-800/50 flex flex-col md:flex-row md:items-center justify-between gap-3 overflow-hidden">
                                        {dep.status === 'Failed' && <div className="absolute left-0 top-0 bottom-0 w-1 bg-red-500"></div>}
                                        {dep.status === 'Success' && <div className="absolute left-0 top-0 bottom-0 w-1 bg-green-500"></div>}
                                        
                                        <div className="flex items-center gap-3 pl-2">
                                            <div className="bg-cyan-900/50 p-2 rounded-md">
                                                <ServerCogIcon className="w-5 h-5 text-cyan-300" />
                                            </div>
                                            <div>
                                                <p className="font-bold text-sm text-white">{dep.app}</p>
                                                <p className="text-xs text-cyan-500 font-mono">Ver: {dep.version} • ID: {dep.id}</p>
                                            </div>
                                        </div>

                                        {/* Pipeline Visual */}
                                        <div className="flex items-center gap-2 text-[10px] text-cyan-600 hidden md:flex">
                                            <span className="text-green-500">Build</span> 
                                            <span className="w-4 h-px bg-cyan-800"></span> 
                                            <span className="text-green-500">Test</span> 
                                            <span className="w-4 h-px bg-cyan-800"></span>
                                            <span className={dep.status === 'Failed' ? 'text-red-500 font-bold' : 'text-green-500'}>Sign</span> 
                                            <span className="w-4 h-px bg-cyan-800"></span>
                                            <span className={dep.status === 'Success' ? 'text-white font-bold' : 'text-gray-500'}>{dep.stage}</span>
                                        </div>

                                        <div className="text-right pl-2">
                                            <div className={`flex items-center justify-end gap-1 text-xs font-bold ${dep.status === 'Success' ? 'text-green-400' : 'text-red-400'}`}>
                                                {dep.status === 'Success' ? <CheckCircle2Icon className="w-3 h-3"/> : <AlertTriangleIcon className="w-3 h-3"/>}
                                                {dep.status}
                                            </div>
                                            <p className="text-[10px] text-gray-500">{dep.time}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* --- PROTOCOL TAB --- */}
                    {activeTab === 'protocol' && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 h-full">
                            {/* Packet Inspector (Left) */}
                            <div className="flex flex-col gap-4">
                                <Section title="CHIPS Packet Inspector" icon={FileCodeIcon}>
                                    <div className="flex justify-between items-center mb-2">
                                        <button onClick={generatePacket} className="holographic-button px-3 py-1 text-xs flex items-center gap-2 bg-blue-600/20 text-blue-300 border-blue-500/50">
                                            <PlayIcon className="w-3 h-3" /> Generate Test Packet
                                        </button>
                                        <span className={`text-xs font-mono px-2 py-0.5 rounded border ${activePacket ? 'bg-green-900/30 border-green-500 text-green-400' : 'bg-gray-800 border-gray-600 text-gray-500'}`}>
                                            Status: {activePacket?.status || 'Idle'}
                                        </span>
                                    </div>
                                    
                                    {activePacket ? (
                                        <div className="font-mono text-[10px] space-y-2 bg-black/40 p-2 rounded border border-cyan-900/30 shadow-inner">
                                            {/* Header */}
                                            <div className="border-l-2 border-purple-500 pl-2">
                                                <p className="text-purple-400 font-bold mb-1">A. Packet Header</p>
                                                <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-gray-300">
                                                    <span>PROTOCOL:</span> <span className="text-white">{activePacket.header.protocol}</span>
                                                    <span>SOURCE_ID:</span> <span className="text-white">{activePacket.header.sourceId}</span>
                                                    <span>TARGET_SCOPE:</span> <span className="text-yellow-300">{activePacket.header.targetScope}</span>
                                                    <span>VERSION:</span> <span className="text-white">{activePacket.header.version}</span>
                                                </div>
                                            </div>

                                            {/* Control Block */}
                                            <div className="border-l-2 border-cyan-500 pl-2">
                                                <p className="text-cyan-400 font-bold mb-1">B. Control Block (EKS)</p>
                                                <div className="space-y-1 text-gray-300">
                                                    <div className="flex justify-between">
                                                        <span>TIMESTAMP:</span> <span className="text-white">{activePacket.control.timestamp}</span>
                                                    </div>
                                                    <div className="flex justify-between">
                                                        <span>EKS_REF:</span> <span className="text-green-400">{activePacket.control.eksReference}</span>
                                                    </div>
                                                    <div className="flex justify-between">
                                                        <span>HASH:</span> <span className="text-gray-500 truncate w-32" title={activePacket.control.integrityHash}>{activePacket.control.integrityHash?.substring(0, 16)}...</span>
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Payload */}
                                            <div className="border-l-2 border-red-500 pl-2">
                                                <p className="text-red-400 font-bold mb-1">C. Secured Payload</p>
                                                <div className="p-2 bg-red-900/10 rounded border border-red-900/30 text-red-200/50 break-all">
                                                    {activePacket.payload}
                                                </div>
                                                <p className="text-gray-500 mt-1 italic text-[9px]">Encrypted via EKS OTP</p>
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="h-48 flex items-center justify-center text-gray-500 text-xs italic border border-dashed border-gray-700 rounded">
                                            No active packet. Click Generate to simulate.
                                        </div>
                                    )}
                                </Section>
                            </div>

                            {/* Routing Visualization (Right) */}
                            <div className="flex flex-col gap-4">
                                <Section title="Mesh Routing Simulator" icon={Share2Icon}>
                                    <div className="relative h-64 bg-black/40 rounded border border-cyan-900/30 p-4 flex flex-col justify-between">
                                        {/* Background Grid */}
                                        <div className="absolute inset-0 bg-[linear-gradient(rgba(0,255,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(0,255,255,0.05)_1px,transparent_1px)] bg-[size:20px_20px]"></div>

                                        {/* Nodes */}
                                        <div className="flex justify-between items-center relative z-10">
                                            {/* QAN */}
                                            <div className={`flex flex-col items-center p-2 rounded border transition-all duration-500 ${simStep >= 1 ? 'bg-purple-900/30 border-purple-500 scale-110 shadow-[0_0_15px_rgba(168,85,247,0.4)]' : 'bg-gray-900/50 border-gray-700 opacity-50'}`}>
                                                <ServerCogIcon className={`w-6 h-6 ${simStep >= 1 ? 'text-purple-400' : 'text-gray-500'}`} />
                                                <span className="text-[9px] font-bold mt-1 text-white">QAN (Source)</span>
                                            </div>

                                            {/* Arrow 1 */}
                                            <div className="flex-grow h-0.5 bg-gray-700 mx-2 relative">
                                                <div className={`absolute top-0 left-0 h-full bg-cyan-400 transition-all duration-1000 ${simStep >= 2 ? 'w-full' : 'w-0'}`}></div>
                                            </div>

                                            {/* Gateway */}
                                            <div className={`flex flex-col items-center p-2 rounded border transition-all duration-500 ${simStep >= 3 ? 'bg-blue-900/30 border-blue-500 scale-110 shadow-[0_0_15px_rgba(59,130,246,0.4)]' : 'bg-gray-900/50 border-gray-700 opacity-50'}`}>
                                                <NetworkIcon className={`w-6 h-6 ${simStep >= 3 ? 'text-blue-400' : 'text-gray-500'}`} />
                                                <span className="text-[9px] font-bold mt-1 text-white">Regional GW</span>
                                            </div>

                                            {/* Arrow 2 */}
                                            <div className="flex-grow h-0.5 bg-gray-700 mx-2 relative">
                                                <div className={`absolute top-0 left-0 h-full bg-cyan-400 transition-all duration-1000 ${simStep >= 4 ? 'w-full' : 'w-0'}`}></div>
                                            </div>

                                            {/* DQN */}
                                            <div className={`flex flex-col items-center p-2 rounded border transition-all duration-500 ${simStep >= 5 ? 'bg-green-900/30 border-green-500 scale-110 shadow-[0_0_15px_rgba(34,197,94,0.4)]' : 'bg-gray-900/50 border-gray-700 opacity-50'}`}>
                                                <CpuChipIcon className={`w-6 h-6 ${simStep >= 5 ? 'text-green-400' : 'text-gray-500'}`} />
                                                <span className="text-[9px] font-bold mt-1 text-white">Target DQN</span>
                                            </div>
                                        </div>

                                        {/* Verification Steps Overlay */}
                                        <div className="mt-4 bg-black/60 rounded p-2 text-[10px] font-mono border border-cyan-900/50 h-24 overflow-y-auto custom-scrollbar">
                                            {simLogs.map((log, i) => (
                                                <div key={i} className="mb-1 text-cyan-300 border-b border-cyan-900/30 pb-1 last:border-0">{log}</div>
                                            ))}
                                        </div>
                                    </div>
                                </Section>
                            </div>
                        </div>
                    )}
                    
                    {/* --- NETWORK SIM TAB --- */}
                    {activeTab === 'network' && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 h-full">
                            <Section title="Network Request Simulator" icon={ActivityIcon}>
                                <div className="space-y-4">
                                    <div className="flex gap-2">
                                        <select 
                                            value={netMethod} 
                                            onChange={(e) => setNetMethod(e.target.value)}
                                            className="bg-black/50 border border-cyan-800 rounded px-2 py-1 text-xs text-white"
                                        >
                                            <option value="GET">GET</option>
                                            <option value="POST">POST</option>
                                            <option value="PUT">PUT</option>
                                            <option value="DELETE">DELETE</option>
                                        </select>
                                        <input 
                                            value={netUrl} 
                                            onChange={(e) => setNetUrl(e.target.value)}
                                            className="flex-grow bg-black/50 border border-cyan-800 rounded px-2 py-1 text-xs text-white font-mono"
                                        />
                                    </div>
                                    
                                    <div className="flex gap-2">
                                        <button 
                                            onClick={simulateFetch}
                                            disabled={netStatus === 'CONNECTING' || netStatus === 'TRANSFER'}
                                            className="flex-1 holographic-button py-2 bg-blue-600/30 border-blue-500 text-blue-200 text-xs font-bold rounded flex items-center justify-center gap-2 hover:bg-blue-600/50 disabled:opacity-50"
                                        >
                                            {netStatus === 'CONNECTING' ? <LoaderIcon className="w-3 h-3 animate-spin"/> : <GlobeIcon className="w-3 h-3"/>}
                                            Simulate Fetch
                                        </button>
                                        <button 
                                            onClick={simulateXHR}
                                            disabled={netStatus === 'CONNECTING' || netStatus === 'TRANSFER'}
                                            className="flex-1 holographic-button py-2 bg-purple-600/30 border-purple-500 text-purple-200 text-xs font-bold rounded flex items-center justify-center gap-2 hover:bg-purple-600/50 disabled:opacity-50"
                                        >
                                            {netStatus === 'CONNECTING' ? <LoaderIcon className="w-3 h-3 animate-spin"/> : <RefreshCwIcon className="w-3 h-3"/>}
                                            Simulate XHR
                                        </button>
                                    </div>
                                    
                                    {netStatus === 'CONNECTING' || netStatus === 'TRANSFER' ? (
                                         <div className="bg-black/40 border border-cyan-900/30 rounded p-2 h-32 flex items-center justify-center">
                                             <LoadingSkeleton className="h-2 w-3/4 mb-2" />
                                             <LoadingSkeleton className="h-2 w-1/2" />
                                         </div>
                                    ) : (
                                        <div className="bg-black/40 border border-cyan-900/30 rounded p-2 text-xs h-32 overflow-y-auto font-mono custom-scrollbar">
                                            {netLogs.map((log, i) => (
                                                <div key={i} className="text-cyan-200 mb-0.5">{log}</div>
                                            ))}
                                        </div>
                                    )}
                                    
                                    {netResponse && (
                                        <div className="bg-green-900/20 border border-green-800/30 rounded p-2 text-xs font-mono text-green-300 overflow-x-auto">
                                            <pre>{netResponse}</pre>
                                        </div>
                                    )}
                                </div>
                            </Section>
                            
                            <Section title="Quantum Network Telemetry" icon={ActivityIcon}>
                                <div className="space-y-3">
                                    <Kpi label="EKS Tunnel Fidelity" value="99.98%" icon={ShieldCheckIcon} />
                                    <Kpi label="Quantum Latency" value="0.04 ms" icon={ClockIcon} />
                                    <Kpi label="Q-DNS Resolution" value="0.01 ms" icon={GlobeIcon} />
                                    
                                    <div className="bg-black/30 p-2 rounded border border-cyan-900/30 mt-4 relative h-40 overflow-hidden">
                                        <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-20">
                                            <div className="w-full h-full bg-[linear-gradient(rgba(0,255,255,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(0,255,255,0.1)_1px,transparent_1px)] bg-[size:10px_10px]"></div>
                                        </div>
                                        {/* Simple Oscilloscope Visual */}
                                        <svg className="w-full h-full" preserveAspectRatio="none">
                                            <path d={`M 0 50 Q 25 ${50 + oscilloscopeOffsets[0]} 50 50 T 100 50 T 150 50 T 200 50 T 250 50 T 300 50`} fill="none" stroke="#22d3ee" strokeWidth="2" className="animate-pulse" />
                                            <path d={`M 0 50 Q 25 ${50 - oscilloscopeOffsets[1]} 50 50 T 100 50 T 150 50 T 200 50 T 250 50 T 300 50`} fill="none" stroke="#a855f7" strokeWidth="2" className="animate-pulse" style={{ animationDelay: '0.5s' }} />
                                        </svg>
                                    </div>
                                </div>
                            </Section>
                        </div>
                    )}

                </div>
            </div>
        </GlassPanel>
    );
};

export default CHIPSGatewayAdmin;
