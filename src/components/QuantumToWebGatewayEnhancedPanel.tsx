
import React, { useState, useMemo } from 'react';
import GlassPanel from './GlassPanel';
import { 
    LinkIcon, GlobeIcon, UploadCloudIcon, CalendarDaysIcon, 
    ShieldCheckIcon, KeyIcon, ChartBarIcon, ClockIcon, HeartIcon, AlertTriangleIcon,
    RocketLaunchIcon, LoaderIcon, CheckCircle2Icon, ToggleLeftIcon, ToggleRightIcon, NetworkIcon,
    CloudServerIcon, HardDriveIcon, ActivityIcon, LockIcon, ArrowTopRightOnSquareIcon, RefreshCwIcon, ServerCogIcon
} from './Icons';
import { URIAssignment } from '../types';

// --- Type Definitions ---
type Tab = 'hosting' | 'domains' | 'deployments';
type AppStatus = 'Active' | 'Pending' | 'Error';

interface GatewayApp {
    id: string;
    name: string;
    chipsAddress: string;
    publicUrl: string;
    status: AppStatus;
}

interface QuantumToWebGatewayEnhancedPanelProps {
    onConnect: (sourceName: string) => void;
    uriAssignments: URIAssignment[];
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


// --- Mock Data ---
const hostingPods = [
    { id: 'pod-01', region: 'US-East', load: 45, type: 'Hybrid-Bridge', status: 'Active', version: 'v3.1.2' },
    { id: 'pod-02', region: 'EU-Central', load: 72, type: 'Hybrid-Bridge', status: 'Active', version: 'v3.1.2' },
    { id: 'pod-03', region: 'Asia-South', load: 12, type: 'Q-Cache', status: 'Syncing', version: 'v3.1.1' },
    { id: 'pod-04', region: 'US-West', load: 98, type: 'Hybrid-Bridge', status: 'Warning', version: 'v3.1.2' },
];

const initialAppsData: GatewayApp[] = [
  { id: 'app1', name: 'Global Abundance Engine', chipsAddress: 'CHIPS://gae.qcos.apps/main', publicUrl: 'https://qcos.apps.web/abundance', status: 'Active' },
  { id: 'app2', name: 'QMC: Finance', chipsAddress: 'CHIPS://qmc-finance.qcos.apps/main', publicUrl: 'https://qcos.apps.web/qmc-finance', status: 'Active' },
  { id: 'app3', name: 'Molecular Simulator', chipsAddress: 'CHIPS://mol-sim.qcos.apps/main', publicUrl: 'https://qcos.apps.web/mol-sim', status: 'Pending' },
  { id: 'app4', name: 'Quantum Network Visualizer', chipsAddress: 'CHIPS://qnet-viz.qcos.apps/main', publicUrl: 'https://qcos.apps.web/qnet-viz', status: 'Error' },
];

const initialDataSources = [
    { id: 'ds1', name: 'Google Scholar API', url: 'https://scholar.google.com/api', schedule: 'Daily', dataTypes: ['text', 'pdf'], status: 'healthy' },
    { id: 'ds2', name: 'arXiv Pre-prints (quant-ph)', url: 'https://arxiv.org/list/quant-ph/new', schedule: 'Hourly', dataTypes: ['text', 'pdf'], status: 'healthy' },
    { id: 'ds3', name: 'CERN Open Data Portal', url: 'https://opendata.cern.ch', schedule: 'Weekly', dataTypes: ['csv', 'binary'], status: 'error' },
    { id: 'ds4', name: 'Global Weather Arrays', url: 'https://weather-api.global/v2/stream', schedule: 'Continuous', dataTypes: ['json', 'binary'], status: 'healthy' },
    { id: 'ds5', name: 'Financial Tickers (Crypto/Forex)', url: 'wss://market-stream.finance', schedule: 'Real-time', dataTypes: ['stream'], status: 'healthy' },
    { id: 'ds6', name: 'Bio-Medical Research DB', url: 'https://pubmed.ncbi.nlm.nih.gov/api', schedule: 'Daily', dataTypes: ['xml', 'text'], status: 'healthy' },
    { id: 'ds7', name: 'LHC Collision Data', url: 'https://lhc.cern.ch/stream', schedule: 'Real-time', dataTypes: ['binary'], status: 'healthy' },
    { id: 'ds8', name: 'Global HFT Feed', url: 'wss://hft.global/feed', schedule: 'Real-time', dataTypes: ['stream'], status: 'healthy' },
    { id: 'ds9', name: 'USGS Seismic', url: 'https://earthquake.usgs.gov/fdsnws', schedule: 'Continuous', dataTypes: ['json'], status: 'warning' },
    { id: 'ds10', name: 'NASA Exoplanet Archive', url: 'https://exoplanetarchive.ipac.caltech.edu/api', schedule: 'Weekly', dataTypes: ['csv'], status: 'healthy' },
    { id: 'ds11', name: 'Global Quantum Material Lattice (GQML)', url: 'wss://materials.qcos.network/stream', schedule: 'Real-time', dataTypes: ['binary', 'json'], status: 'healthy' },
];

const QuantumToWebGatewayEnhancedPanel: React.FC<QuantumToWebGatewayEnhancedPanelProps> = ({ uriAssignments, onConnect }) => {
    const [activeTab, setActiveTab] = useState<Tab>('hosting');
    const [apps, setApps] = useState(initialAppsData);
    const [dataSources, setDataSources] = useState(initialDataSources);
    const [newSource, setNewSource] = useState({ name: '', url: '', schedule: 'daily' });
    const [ipWhitelisting, setIpWhitelisting] = useState(false);
    const [newApp, setNewApp] = useState({ name: '', chipsAddress: '' });
    const [isRegistering, setIsRegistering] = useState(false);

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

        return [...liveDomains, ...staticDomains];
    }, [uriAssignments]);

    // Merge live deployments with static history
    const deployments = useMemo(() => {
        const staticDeployments = [
            { id: 'dep-x99', app: 'QMC Finance', version: '2.4.0', time: '10 mins ago', status: 'Success', stage: 'Production' },
            { id: 'dep-x98', app: 'Swine Foresight', version: '1.1.5', time: '1 hour ago', status: 'Success', stage: 'Production' },
            { id: 'dep-x97', app: 'Mol-Sim Toolkit', version: '0.9.0-beta', time: '3 hours ago', status: 'Failed', stage: 'Staging' },
        ];

        const liveDeployments = uriAssignments.map((ua, index) => ({
            id: `dep-live-${index}`,
            app: ua.appName,
            version: '1.0.0',
            time: ua.timestamp,
            status: 'Success',
            stage: 'Production'
        }));

        return [...liveDeployments.reverse(), ...staticDeployments];
    }, [uriAssignments]);

    const handleAddDataSource = () => {
        if (newSource.name && newSource.url) {
            const newSourceName = newSource.name;
            setDataSources([...dataSources, { ...newSource, id: `ds${Date.now()}`, dataTypes: ['text'], status: 'healthy' }]);
            setNewSource({ name: '', url: '', schedule: 'daily' });
            onConnect(newSourceName);
        }
    };
    
    const handleRegisterApp = () => {
        if (newApp.name && newApp.chipsAddress.startsWith('CHIPS://') && !isRegistering) {
            setIsRegistering(true);
            const newId = `app${Date.now()}`;
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


    return (
        <GlassPanel title={
            <div className="flex items-center">
                <NetworkIcon className="w-5 h-5 mr-2 text-red-400" />
                <span>Quantum-to-Web Gateway Admin</span>
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
                </div>

                {/* Content Area */}
                <div className="flex-grow overflow-y-auto pr-2 min-h-0 space-y-4">
                    
                    {/* --- HOSTING TAB --- */}
                    {activeTab === 'hosting' && (
                        <div className="space-y-4">
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                <div className="bg-black/20 p-3 rounded-lg border border-purple-900/50 text-center">
                                    <p className="text-xs text-purple-400 uppercase tracking-widest mb-1">Gateway Nodes</p>
                                    <p className="text-2xl font-mono text-white">14</p>
                                </div>
                                <div className="bg-black/20 p-3 rounded-lg border border-purple-900/50 text-center">
                                    <p className="text-xs text-purple-400 uppercase tracking-widest mb-1">Global Latency</p>
                                    <p className="text-2xl font-mono text-green-400">24ms</p>
                                </div>
                                <div className="bg-black/20 p-3 rounded-lg border border-purple-900/50 text-center">
                                    <p className="text-xs text-purple-400 uppercase tracking-widest mb-1">Q-Web Requests</p>
                                    <p className="text-2xl font-mono text-cyan-300">8.2M</p>
                                </div>
                                <div className="bg-black/20 p-3 rounded-lg border border-purple-900/50 text-center">
                                    <p className="text-xs text-purple-400 uppercase tracking-widest mb-1">Storage Used</p>
                                    <p className="text-2xl font-mono text-white">45%</p>
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
                                            {ipWhitelisting && <input type="text" placeholder="e.g., 192.168.1.0/24" className="w-full p-1 bg-black/50 border border-cyan-800 rounded-md text-white text-xs" />}
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
                                            <Kpi label="Average Latency" value="42 ms" icon={ClockIcon} />
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
                                             <input type="text" placeholder="URL" value={newSource.url} onChange={e => setNewSource(s => ({...s, url: e.target.value}))} className="w-full p-1 bg-black/50 border border-cyan-800 rounded-md text-white text-xs" />
                                         </div>
                                         <button onClick={handleAddDataSource} title="Add and connect a new data source for AGI training." className="holographic-button text-xs px-3 py-1.5 rounded w-full">Add Data Source</button>
                                         <div className="mt-3 space-y-2 max-h-24 overflow-y-auto pr-1">
                                             {dataSources.map(ds => (
                                                <div key={ds.id} className="flex items-center justify-between text-xs bg-cyan-950/30 p-1.5 rounded-md">
                                                    <div className="flex items-center gap-2">
                                                        <div className={`w-2 h-2 rounded-full ${ds.status === 'healthy' ? 'bg-green-400' : 'bg-red-400'}`} title={ds.status}></div>
                                                        <span className="text-white truncate">{ds.name}</span>
                                                    </div>
                                                    <span className="text-cyan-500 capitalize">{ds.schedule}</span>
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

                </div>
            </div>
        </GlassPanel>
    );
};

export default QuantumToWebGatewayEnhancedPanel;
