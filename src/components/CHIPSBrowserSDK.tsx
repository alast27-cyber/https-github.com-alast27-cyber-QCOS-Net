
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
    GlobeIcon, ArrowRightIcon, MessageSquareIcon, CpuChipIcon, 
    PlusIcon, XIcon, ChevronLeftIcon, ChevronRightIcon, ArrowPathIcon,
    HomeIcon, ShieldCheckIcon, AlertTriangleIcon, LoaderIcon, ServerCogIcon, 
    KeyIcon, ActivityIcon, SparklesIcon, BrainCircuitIcon, SearchIcon, LightBulbIcon, BoxIcon,
    StarIcon
} from './Icons';
import { AppDefinition } from '../types';
import ChimeraCoreStatus from './ChimeraCoreStatus';
import CHIPSAppStore from './CHIPSAppStore';
import LivePreviewFrame from './LivePreviewFrame';
import AgenticAddressBar from './AgenticAddressBar';
import LoadingSkeleton from './LoadingSkeleton';

interface CHIPSBrowserSDKProps {
    initialApp?: AppDefinition;
    onToggleAgentQ: () => void;
    apps: AppDefinition[];
    onInstallApp: (id: string) => void;
}

interface BrowserTab {
    id: string;
    title: string;
    uri: string;
    history: string[];
    historyIndex: number;
    isLoading: boolean;
    aiContextSummary?: string; // AI-Native feature: Summary of current content
    icon?: React.FC<{ className?: string }>;
}

const NEW_TAB_URI = 'chips://newtab';
const STORE_URI_CHIPS = 'chips://store';
const STORE_URI_WEB = 'https://store.qcos';

// --- Sub-component: DQN Status Bar ---
const DQNStatusBar = () => (
    <div className="bg-cyan-950/90 border-t border-cyan-800/50 p-1 px-3 flex items-center justify-between text-[10px] font-mono select-none backdrop-blur-md z-20">
        <div className="flex items-center gap-4">
            <span className="flex items-center text-cyan-400" title="This Browser is acting as a Decentralized Quantum Node">
                <ServerCogIcon className="w-3 h-3 mr-1" />
                NODE: <span className="text-white ml-1">DQN-LOCAL-BROWSER</span>
            </span>
            <span className="flex items-center text-green-400">
                <ActivityIcon className="w-3 h-3 mr-1" />
                STATUS: <span className="text-white ml-1">ACTIVE</span>
            </span>
            <span className="flex items-center text-yellow-400" title="Entangled Key State Synchronized">
                <KeyIcon className="w-3 h-3 mr-1" />
                EKS: <span className="text-white ml-1">SYNCED (Dilithium-2)</span>
            </span>
        </div>
        <div className="flex items-center gap-4 hidden sm:flex">
            <span className="text-cyan-600">Qubit Capacity: <span className="text-cyan-300">128</span></span>
            <span className="text-cyan-600">Avg Fidelity: <span className="text-cyan-300">99.80%</span></span>
        </div>
    </div>
);

// --- Sub-component: AI-Native Smart New Tab ---
const SmartNewTab: React.FC<{ apps: AppDefinition[], onNavigate: (uri: string) => void }> = ({ apps, onNavigate }) => {
    const installedApps = apps.filter(a => a.status === 'installed');
    const [isConnecting, setIsConnecting] = useState(true);

    useEffect(() => {
        const timer = setTimeout(() => setIsConnecting(false), 800);
        return () => clearTimeout(timer);
    }, []);

    if (isConnecting) {
        return (
            <div className="h-full flex flex-col items-center justify-center p-8 bg-black">
                <div className="flex flex-col items-center gap-4">
                    <div className="relative w-16 h-16">
                        <div className="absolute inset-0 border-t-2 border-cyan-500 rounded-full animate-spin"></div>
                        <div className="absolute inset-2 border-r-2 border-purple-500 rounded-full animate-spin-reverse-slow"></div>
                        <div className="absolute inset-0 flex items-center justify-center">
                            <CpuChipIcon className="w-6 h-6 text-cyan-700" />
                        </div>
                    </div>
                    <div className="text-center">
                        <p className="text-xs font-mono text-cyan-500 uppercase tracking-widest animate-pulse">Establishing Mesh Link...</p>
                        <p className="text-[10px] text-gray-600 font-mono mt-1">Handshaking with QAN-ROOT-01</p>
                    </div>
                </div>
            </div>
        );
    }
    
    return (
        <div className="h-full flex flex-col items-center justify-center p-8 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-slate-900/80 via-black to-black animate-fade-in">
            <div className="relative mb-8 group cursor-default">
                <div className="absolute inset-0 bg-cyan-500/10 blur-3xl rounded-full animate-pulse-slow"></div>
                <BrainCircuitIcon className="w-24 h-24 text-cyan-400 relative z-10 drop-shadow-[0_0_25px_rgba(6,182,212,0.6)] animate-float" />
                <div className="absolute -bottom-4 left-1/2 -translate-x-1/2 text-[10px] text-cyan-600 font-mono tracking-widest bg-black/60 px-2 rounded border border-cyan-900/50 whitespace-nowrap">
                    AI-NATIVE INTERFACE ONLINE
                </div>
            </div>

            <h1 className="text-3xl font-bold text-white tracking-[0.2em] mb-6 font-mono text-center">
                CHIPS <span className="text-cyan-400">BROWSER</span>
            </h1>

            {/* Predictive Suggestions / "Speed Dial" */}
            <div className="w-full max-w-4xl">
                <div className="flex items-center justify-between mb-4 px-2">
                    <p className="text-cyan-500 text-xs uppercase tracking-widest">Predicted Workflows</p>
                    <button onClick={() => onNavigate(STORE_URI_CHIPS)} className="text-xs text-cyan-400 hover:text-white flex items-center gap-1 group">
                        <BoxIcon className="w-3 h-3 group-hover:text-cyan-300" /> Open App Store
                    </button>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    {/* Store Shortcut */}
                    <button 
                        onClick={() => onNavigate(STORE_URI_CHIPS)}
                        className="group relative flex flex-col items-center p-4 rounded-xl bg-cyan-900/30 border border-cyan-700/50 hover:border-cyan-400 hover:bg-cyan-800/40 transition-all duration-300"
                    >
                        <BoxIcon className="w-8 h-8 text-cyan-300 mb-2 group-hover:scale-110 transition-transform" />
                        <span className="text-xs font-bold text-cyan-100">App Store</span>
                    </button>

                    {installedApps.slice(0, 3).map(app => (
                        <button 
                            key={app.id}
                            onClick={() => onNavigate(app.q_uri || `chips://app/${app.id}`)}
                            className="group relative flex flex-col items-center p-4 rounded-xl bg-cyan-950/20 border border-cyan-800/30 hover:border-cyan-400 hover:bg-cyan-900/40 transition-all duration-300"
                        >
                            <app.icon className="w-8 h-8 text-cyan-300 mb-2 group-hover:scale-110 transition-transform" />
                            <span className="text-xs font-bold text-cyan-100">{app.name}</span>
                            <div className="absolute inset-0 border border-cyan-400/0 group-hover:border-cyan-400/50 rounded-xl transition-all duration-500 scale-95 group-hover:scale-100"></div>
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );
};

// --- Sub-component: AI Sidebar ---
const AIContextSidebar: React.FC<{ context: string | undefined, isLoading: boolean }> = ({ context, isLoading }) => (
    <div className="w-64 bg-black/60 border-l border-cyan-800/50 p-4 flex flex-col h-full backdrop-blur-md absolute right-0 top-0 bottom-0 z-10 transition-transform duration-300 transform translate-x-0">
        <div className="flex items-center gap-2 mb-4 text-cyan-300 border-b border-cyan-800/50 pb-2">
            <SparklesIcon className="w-4 h-4" />
            <span className="text-xs font-bold uppercase tracking-wider">Neural Context</span>
        </div>
        <div className="flex-grow text-xs text-cyan-100 space-y-3 font-mono">
            {isLoading ? (
                <div className="space-y-4">
                     <div className="flex items-center gap-2 text-cyan-500 mb-2">
                        <LoaderIcon className="w-4 h-4 animate-spin" />
                        <span>Scanning Context...</span>
                     </div>
                     <LoadingSkeleton className="h-20 w-full" />
                     <div className="space-y-2 pt-4">
                        <LoadingSkeleton className="h-4 w-3/4" />
                        <LoadingSkeleton className="h-4 w-1/2" />
                        <LoadingSkeleton className="h-4 w-5/6" />
                     </div>
                </div>
            ) : (
                <>
                    <div className="bg-cyan-950/40 p-2 rounded border border-cyan-900/50 animate-fade-in">
                        <p className="text-cyan-500 mb-1 text-[10px] uppercase">Summary</p>
                        <p>{context || "No active content to analyze."}</p>
                    </div>
                    <div className="bg-cyan-950/40 p-2 rounded border border-cyan-900/50 animate-fade-in">
                        <p className="text-cyan-500 mb-1 text-[10px] uppercase">Key Entities</p>
                        <ul className="list-disc list-inside text-cyan-200">
                            <li>Quantum State</li>
                            <li>DQN Node</li>
                            <li>EKS Protocol</li>
                        </ul>
                    </div>
                    <div className="mt-4 animate-fade-in">
                        <p className="text-cyan-500 mb-2 text-[10px] uppercase">Suggested Actions</p>
                        <button className="w-full text-left p-2 hover:bg-cyan-900/30 rounded text-cyan-300 flex items-center gap-2 transition-colors">
                            <LightBulbIcon className="w-3 h-3" /> Save to Memory
                        </button>
                        <button className="w-full text-left p-2 hover:bg-cyan-900/30 rounded text-cyan-300 flex items-center gap-2 transition-colors">
                            <ArrowPathIcon className="w-3 h-3" /> Cross-Reference
                        </button>
                    </div>
                </>
            )}
        </div>
    </div>
);

const CHIPSBrowserSDK: React.FC<CHIPSBrowserSDKProps> = ({ initialApp, onToggleAgentQ, apps, onInstallApp }) => {
    const [tabs, setTabs] = useState<BrowserTab[]>(() => {
        if (initialApp) {
            const startUri = initialApp?.q_uri || NEW_TAB_URI;
            const newTab: BrowserTab = {
                id: `tab-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                title: initialApp ? initialApp.name : 'New Tab',
                uri: startUri,
                history: [startUri],
                historyIndex: 0,
                isLoading: false,
                icon: initialApp?.icon,
                aiContextSummary: "System ready. Awaiting quantum intent."
            };
            return [newTab];
        }
        return [{
            id: 'tab-0',
            title: 'New Tab',
            uri: NEW_TAB_URI,
            history: [NEW_TAB_URI],
            historyIndex: 0,
            isLoading: false,
            aiContextSummary: "System ready. Awaiting quantum intent."
        }];
    });
    const [activeTabId, setActiveTabId] = useState<string | null>(null);
    const [intentInput, setIntentInput] = useState('');
    const [showAISidebar, setShowAISidebar] = useState(false);
    const [isCoreStatusOpen, setIsCoreStatusOpen] = useState(false);
    const [bookmarks, setBookmarks] = useState<{ uri: string, title: string }[]>([
        { uri: STORE_URI_CHIPS, title: 'App Store' },
        { uri: 'chips://qmc-finance', title: 'QMC Finance' }
    ]);



    // Update address bar when active tab changes
    useEffect(() => {
        const activeTab = tabs.find(t => t.id === activeTabId);
        if (activeTab && activeTab.uri !== intentInput) {
            setTimeout(() => setIntentInput(activeTab.uri === NEW_TAB_URI ? '' : activeTab.uri), 0);
        }
    }, [activeTabId, tabs, intentInput]);

    const createTab = useCallback(() => {
        const newTab: BrowserTab = {
            id: `tab-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            title: 'New Tab',
            uri: NEW_TAB_URI,
            history: [NEW_TAB_URI],
            historyIndex: 0,
            isLoading: false,
            aiContextSummary: "Awaiting input..."
        };
        setTabs(prev => [...prev, newTab]);
        setActiveTabId(newTab.id);
        setIntentInput('');
    }, []);

    const closeTab = useCallback((id: string, e: React.MouseEvent) => {
        e.stopPropagation();
        setTabs(prev => {
            const newTabs = prev.filter(t => t.id !== id);
            if (newTabs.length === 0) {
                const emptyTab: BrowserTab = {
                    id: `tab-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                    title: 'New Tab',
                    uri: NEW_TAB_URI,
                    history: [NEW_TAB_URI],
                    historyIndex: 0,
                    isLoading: false
                };
                setActiveTabId(emptyTab.id);
                return [emptyTab];
            }
            if (activeTabId === id) {
                setActiveTabId(newTabs[newTabs.length - 1].id);
            }
            return newTabs;
        });
    }, [activeTabId]);

    // --- AI-Native Intent Resolver ---
    const resolveIntent = (input: string): { uri: string, title: string, icon?: any } => {
        const lowerInput = input.toLowerCase().trim();
        
        // Direct Store Access
        if (lowerInput === 'store' || lowerInput === 'app store' || lowerInput === 'qcos store' || lowerInput === 'store.qcos' || lowerInput.includes('chips://store')) {
            return { uri: STORE_URI_CHIPS, title: 'QCOS Store', icon: BoxIcon };
        }

        // Direct URI
        if (lowerInput.startsWith('chips://') || lowerInput.startsWith('http')) {
            return { uri: input, title: input };
        }

        // Semantic Mapping (Simulating QSC)
        if (lowerInput.includes('pig') || lowerInput.includes('swine') || lowerInput.includes('pork')) {
            if (lowerInput.includes('consumer') || lowerInput.includes('buy')) return { uri: 'chips://pighaven-consumer-trust', title: 'PigHaven', icon: undefined };
            if (lowerInput.includes('philippine') || lowerInput.includes('farm')) return { uri: 'chips://philippine-swine-resilience', title: 'PH Resilience', icon: undefined };
            return { uri: 'chips://global-swine-foresight', title: 'Global Foresight', icon: undefined };
        }
        if (lowerInput.includes('drug') || lowerInput.includes('molecule') || lowerInput.includes('bio')) {
            return { uri: 'chips://q-biomed', title: 'Q-BioMed', icon: undefined };
        }
        if (lowerInput.includes('finance') || lowerInput.includes('market') || lowerInput.includes('stock')) {
            return { uri: 'chips://qmc-finance', title: 'QMC Finance', icon: undefined };
        }

        // Default Fallback
        return { uri: `chips://search?q=${encodeURIComponent(input)}`, title: `Search: ${input}` };
    };

    const navigateTab = useCallback((tabId: string, input: string) => {
        const { uri, title, icon } = resolveIntent(input);

        setTabs(prev => prev.map(tab => {
            if (tab.id !== tabId) return tab;

            const newHistory = tab.history.slice(0, tab.historyIndex + 1);
            newHistory.push(uri);

            // Attempt to find app definition for better icon/title if it's a known app URI
            const app = apps.find(a => 
                a.q_uri === uri || 
                a.https_url === uri || 
                `chips://app/${a.id}` === uri || 
                (uri.startsWith('chips://') && a.q_uri?.includes(uri.split('chips://')[1]))
            );

            return {
                ...tab,
                uri,
                history: newHistory,
                historyIndex: newHistory.length - 1,
                title: app ? app.name : title,
                icon: app ? app.icon : icon,
                isLoading: true,
                aiContextSummary: "Analyzing new vector..."
            };
        }));

        setIntentInput(uri);

        // Simulate Network/AI Load
        setTimeout(() => {
            setTabs(prev => prev.map(t => t.id === tabId ? { 
                ...t, 
                isLoading: false, 
                aiContextSummary: `Content loaded from ${uri}. Node fidelity: 99.9%.` 
            } : t));
        }, 1500); // 1.5s simulated load time
    }, [apps]);

    const renderContent = (tab: BrowserTab) => {
        if (tab.isLoading) {
            return (
                <div className="h-full flex flex-col items-center justify-center p-8 bg-black/40">
                    <div className="w-full max-w-sm space-y-6">
                         <div className="text-center">
                            <LoaderIcon className="w-12 h-12 text-cyan-500 animate-spin mx-auto mb-4" />
                            <p className="text-sm font-mono text-cyan-400 uppercase tracking-widest animate-pulse">Resolving Quantum Intent...</p>
                         </div>
                         <div className="space-y-3">
                             <div className="flex justify-between text-xs text-cyan-700 font-mono">
                                 <span>QSC Validation</span>
                                 <span>EKS Signature</span>
                             </div>
                             <LoadingSkeleton className="h-2 w-full" />
                             <LoadingSkeleton className="h-2 w-3/4" />
                             <LoadingSkeleton className="h-2 w-5/6" />
                         </div>
                    </div>
                </div>
            );
        }

        // --- App Store Rendering ---
        if (tab.uri === STORE_URI_CHIPS || tab.uri === STORE_URI_WEB) {
            return (
                <CHIPSAppStore 
                    apps={apps} 
                    onInstall={onInstallApp}
                    onLaunch={(id) => navigateTab(tab.id, apps.find(a => a.id === id)?.q_uri || `chips://app/${id}`)}
                />
            );
        }

        if (tab.uri === NEW_TAB_URI) {
            return <SmartNewTab apps={apps} onNavigate={(uri) => navigateTab(tab.id, uri)} />;
        }

        // Loose matching for demo purposes
        const app = apps.find(a => 
            a.q_uri === tab.uri || 
            a.https_url === tab.uri || 
            `chips://app/${a.id}` === tab.uri ||
            (tab.uri.startsWith('chips://') && a.q_uri && tab.uri.includes(a.q_uri.replace('CHIPS://', '').split('/')[0]))
        );
        
        if (app) {
            if (app.status === 'installed') {
                 // Update for Custom App Logic
                 if (app.isCustom && app.code) {
                    const appFiles = { 'App.tsx': app.code };
                    return (
                        <div className="relative h-full w-full bg-slate-900">
                            <LivePreviewFrame code={app.code} files={appFiles} />
                        </div>
                    );
                }
                return (
                    <div className="relative h-full w-full">
                        {app.component}
                    </div>
                );
            } else {
                return (
                    <div className="h-full flex flex-col items-center justify-center text-center p-8">
                        <div className="w-16 h-16 rounded-full bg-cyan-900/30 flex items-center justify-center mb-4 border border-cyan-700">
                            <app.icon className="w-8 h-8 text-cyan-300" />
                        </div>
                        <h3 className="text-xl font-bold text-white mb-2">{app.name} is not installed locally.</h3>
                        <p className="text-cyan-400 mb-6 max-w-sm">{app.description}</p>
                        <div className="flex gap-4">
                            <button 
                                onClick={() => navigateTab(tab.id, STORE_URI_CHIPS)} 
                                className="px-6 py-2 bg-transparent border border-cyan-500 text-cyan-400 font-bold rounded-md hover:bg-cyan-900/20"
                            >
                                View in Store
                            </button>
                            <button 
                                onClick={() => onInstallApp(app.id)} 
                                className="holographic-button px-6 py-2 bg-cyan-600/30 border-cyan-500/50 hover:bg-cyan-600/50 text-white font-bold rounded-md"
                            >
                                Download Packet & Install
                            </button>
                        </div>
                    </div>
                );
            }
        }

        return (
            <div className="h-full flex flex-col items-center justify-center text-center text-gray-400">
                <AlertTriangleIcon className="w-16 h-16 text-gray-600 mb-4" />
                <h3 className="text-lg font-mono text-gray-300">404: Quantum State Not Found</h3>
                <p className="text-sm mt-2 font-mono text-gray-500 max-w-md break-all">
                    The Q-URI <span className="text-cyan-700">{tab.uri}</span> could not be resolved by the Quantum Authority Node.
                </p>
                <button onClick={() => navigateTab(tab.id, NEW_TAB_URI)} className="mt-6 text-cyan-500 hover:text-cyan-300 underline text-sm">Return Home</button>
            </div>
        );
    };

    const activeTab = tabs.find(t => t.id === activeTabId);
    const isBookmarked = activeTab && bookmarks.some(b => b.uri === activeTab.uri);

    const toggleBookmark = () => {
        if (!activeTab) return;
        if (isBookmarked) {
            setBookmarks(prev => prev.filter(b => b.uri !== activeTab.uri));
        } else {
            setBookmarks(prev => [...prev, { uri: activeTab.uri, title: activeTab.title }]);
        }
    };

    return (
        <div className="h-full flex flex-col bg-black/40 rounded-lg overflow-hidden border border-cyan-900/50 shadow-2xl relative backdrop-blur-md">
            
            {/* 1. Tab Bar */}
            <div className="flex items-end px-2 pt-2 bg-black/60 border-b border-cyan-800/50 gap-1 overflow-x-auto no-scrollbar">
                {tabs.map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTabId(tab.id)}
                        className={`
                            group relative flex items-center min-w-[140px] max-w-[220px] h-9 px-3 rounded-t-lg border-t border-l border-r transition-all duration-200 text-xs font-medium select-none
                            ${activeTabId === tab.id 
                                ? 'bg-cyan-950/80 border-cyan-600 text-white z-10 shadow-[0_-5px_10px_rgba(0,0,0,0.5)]' 
                                : 'bg-transparent border-transparent text-cyan-600 hover:bg-white/5 hover:text-cyan-300'}
                        `}
                    >
                        {tab.icon ? <tab.icon className="w-3 h-3 mr-2 flex-shrink-0" /> : <GlobeIcon className="w-3 h-3 mr-2 flex-shrink-0" />}
                        <span className="truncate flex-grow text-left">{tab.title}</span>
                        <div 
                            onClick={(e) => closeTab(tab.id, e)}
                            className={`ml-2 p-0.5 rounded-full hover:bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity ${activeTabId === tab.id ? 'text-cyan-200' : 'text-cyan-700'}`}
                        >
                            <XIcon className="w-3 h-3" />
                        </div>
                        {activeTabId === tab.id && <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-cyan-400 shadow-[0_0_5px_cyan]"></div>}
                    </button>
                ))}
                <button 
                    onClick={createTab}
                    className="h-8 w-8 flex items-center justify-center rounded-lg hover:bg-white/10 text-cyan-600 hover:text-cyan-300 transition-colors ml-1"
                    title="New Tab"
                >
                    <PlusIcon className="w-5 h-5" />
                </button>
            </div>

            {/* 2. Navigation & Intent Bar */}
            <div className="flex flex-col bg-black/40 border-b border-cyan-800/30 backdrop-blur-sm relative z-30">
                <div className="flex items-center gap-2 p-2">
                    <div className="flex items-center gap-1">
                        <button onClick={() => activeTabId && navigateTab(activeTabId, activeTab?.uri || NEW_TAB_URI)} className="p-1.5 rounded-md hover:bg-white/10 text-cyan-400 transition-colors">
                            <ArrowPathIcon className={`w-4 h-4 ${activeTab?.isLoading ? 'animate-spin' : ''}`} />
                        </button>
                        <button onClick={() => activeTabId && navigateTab(activeTabId, NEW_TAB_URI)} className="p-1.5 rounded-md hover:bg-white/10 text-cyan-400 transition-colors">
                            <HomeIcon className="w-4 h-4" />
                        </button>
                    </div>

                    {/* Agentic Address Bar */}
                    <div className="flex-grow w-full max-w-4xl mx-auto">
                        <AgenticAddressBar 
                            value={intentInput}
                            onChange={setIntentInput}
                            onNavigate={(val) => activeTabId && navigateTab(activeTabId, val)}
                            isLoading={activeTab?.isLoading}
                        />
                    </div>

                    <div className="flex items-center gap-1 ml-2">
                        <button 
                            onClick={() => setShowAISidebar(v => !v)} 
                            className={`p-1.5 rounded-full border transition-colors ${showAISidebar ? 'bg-cyan-800 border-cyan-400 text-white' : 'bg-cyan-900/30 border-cyan-700/50 text-cyan-300 hover:bg-cyan-800'}`} 
                            title="Toggle AI Context"
                        >
                            <SparklesIcon className="w-4 h-4" />
                        </button>
                        <div className="relative">
                            <button onClick={() => setIsCoreStatusOpen(v => !v)} className={`p-1.5 rounded-full border transition-colors ${isCoreStatusOpen ? 'bg-cyan-800 border-cyan-400 text-white' : 'bg-transparent border-transparent text-cyan-600 hover:bg-white/5 hover:text-cyan-300'}`} title="Chimera Core Status">
                                <CpuChipIcon className="w-4 h-4" />
                            </button>
                            {isCoreStatusOpen && (
                                <div className="absolute top-full right-0 mt-2 z-50 animate-fade-in-up">
                                    <ChimeraCoreStatus />
                                </div>
                            )}
                        </div>
                    </div>
                </div>
                
                {/* Bookmarks Bar */}
                <div className="flex items-center gap-1 px-2 py-1 bg-black/30 border-t border-cyan-900/30 text-xs overflow-x-auto no-scrollbar">
                    {bookmarks.map(b => {
                        const app = apps.find(a => a.q_uri === b.uri || `chips://app/${a.id}` === b.uri);
                        const Icon = app ? app.icon : (b.uri.includes('store') ? BoxIcon : GlobeIcon);
                        return (
                            <button 
                                key={b.uri} 
                                onClick={() => activeTabId && navigateTab(activeTabId, b.uri)}
                                className="flex items-center gap-1 px-2 py-1 rounded hover:bg-white/10 text-cyan-400 hover:text-cyan-200 transition-colors whitespace-nowrap"
                                title={b.uri}
                            >
                                <Icon className="w-3 h-3" />
                                <span className="max-w-[120px] truncate">{b.title}</span>
                            </button>
                        );
                    })}
                </div>
            </div>

            {/* 3. Main Content Area & Sidebar */}
            <main className="flex-grow bg-slate-950/30 relative overflow-hidden flex flex-row">
                <div className="flex-grow flex flex-col relative overflow-hidden">
                    {activeTab ? renderContent(activeTab) : (
                        <div className="h-full flex items-center justify-center text-cyan-700">
                            <p>No tabs open.</p>
                            <button onClick={createTab} className="ml-2 text-cyan-400 underline">Open New Tab</button>
                        </div>
                    )}
                </div>
                
                {showAISidebar && activeTab && (
                    <AIContextSidebar context={activeTab.aiContextSummary} isLoading={activeTab.isLoading} />
                )}
            </main>

            {/* 4. DQN Status Bar */}
            <DQNStatusBar />
        </div>
    );
};

export default CHIPSBrowserSDK;
