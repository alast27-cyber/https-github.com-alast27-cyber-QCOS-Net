
import React, { useState, useEffect, useCallback, Suspense } from 'react';
import { 
    GlobeIcon, ArrowRightIcon, MessageSquareIcon, CpuChipIcon, 
    PlusIcon, XIcon, ChevronLeftIcon, ChevronRightIcon, ArrowPathIcon,
    HomeIcon, ShieldCheckIcon, AlertTriangleIcon, LoaderIcon, ServerCogIcon, 
    KeyIcon, ActivityIcon, SparklesIcon, BrainCircuitIcon, SearchIcon, LightBulbIcon, BoxIcon,
    StarIcon, FileCodeIcon, PlayIcon, AtomIcon, CodeBracketIcon, BanknotesIcon, MicIcon,
    TerminalIcon, DatabaseIcon, StopIcon
} from './Icons';
import { AppDefinition, UIStructure } from '../types';

// Local Imports - Need to points these to the right place or mock them
// For standalone, some of these might need to be mocked or copied
const ChimeraCoreStatus = () => <div className="p-4 bg-slate-900 border border-cyan-500/20 rounded-lg text-cyan-400 font-mono text-xs">CHIMERA CORE STATUS: NOMINAL (Isolated)</div>;
const CHIPSAppStore = ({ apps, onInstall }: any) => (
    <div className="p-8 flex flex-col items-center justify-center text-cyan-500">
        <h2 className="text-2xl font-bold mb-4">Quantum App Registry</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {apps.map((app: any) => (
                <div key={app.id} className="p-4 border border-cyan-500/20 rounded bg-black/40">
                    <h3 className="font-bold">{app.name}</h3>
                    <p className="text-xs opacity-60 mb-2">{app.description}</p>
                    <span className="text-[10px] px-2 py-0.5 bg-cyan-900/40 rounded">{app.status}</span>
                </div>
            ))}
        </div>
    </div>
);
const ChipsEconomy = () => <div className="p-8 text-green-500 font-mono">QUANTUM ECONOMY: ACCESS DENIED (Requires Root Entanglement)</div>;
const QuantumVoiceChat = () => <div className="p-8 text-purple-500 font-mono">VOICE CHAT: INITIALIZING Q-VOX...</div>;
const MonacoEditorWrapper = ({ code }: any) => <pre className="p-4 bg-black text-green-500 overflow-auto h-full text-xs">{code}</pre>;
const LivePreviewFrame = () => <div className="h-full flex items-center justify-center text-cyan-500 italic">Live Preview in Standalone Manifold</div>;
const AgenticAddressBar = ({ value, onChange, onNavigate, isLoading }: any) => (
    <div className="flex-grow flex items-center bg-black/60 border border-cyan-800/50 rounded-lg overflow-hidden h-9">
        <div className="px-3 text-cyan-700"><GlobeIcon className="w-4 h-4" /></div>
        <input 
            type="text" 
            value={value} 
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && onNavigate(value)}
            className="flex-grow bg-transparent border-none outline-none text-cyan-100 text-sm font-mono placeholder:text-cyan-900"
            placeholder="Enter Quantum Intent or Q-URI..."
        />
        {isLoading && <div className="px-3"><LoaderIcon className="w-4 h-4 animate-spin text-cyan-500" /></div>}
    </div>
);

// Lazy load ChipsDevPlatform
const ChipsDevPlatform = React.lazy(() => Promise.resolve({ default: () => <div className="p-8 text-yellow-500">Dev Platform: Inaccessible in Standalone Mode.</div> }));

// Mock Component for QuantumProtocolLibrary since it's simple
const QuantumProtocolLibrary: React.FC = () => (
    <div className="h-full flex flex-col items-center justify-center text-cyan-600">
        <AtomIcon className="w-16 h-16 opacity-50 mb-4" />
        <h3 className="text-xl font-bold">Protocol Library</h3>
        <p className="text-sm">Access compiled Q-Lang algorithms.</p>
    </div>
);

interface CHIPSBrowserProps {
    initialApp?: AppDefinition;
    onToggleAgentQ: () => void;
    apps: AppDefinition[];
    onInstallApp: (id: string) => void;
    onDeployApp?: (details: { name: string; description: string; code: string; uiStructure?: UIStructure }) => void;
    isFullScreen?: boolean;
    onToggleFullScreen?: () => void;
}

interface AIContextData {
    summary: string;
    entities: string[];
    actions: string[];
    confidence: number;
}

interface BrowserTab {
    id: string;
    title: string;
    uri: string;
    history: string[];
    historyIndex: number;
    isLoading: boolean;
    aiContext: AIContextData;
    icon?: React.FC<{ className?: string }>;
    codeContent?: string; 
    codeLanguage?: string;
}

const NEW_TAB_URI = 'chips://newtab';
const STORE_URI_CHIPS = 'chips://store';
const DEV_URI = 'chips://dev';
const ECONOMY_URI = 'chips://economy';
const VOICE_URI = 'chips://q-vox.qcos.apps/main';

const analyzeQuantumContext = (uri: string, title: string): AIContextData => {
    return {
        summary: `Analyzing ${title}... Component isolated in standalone manifold.`,
        entities: ["Standalone Instance", "Quantum State"],
        actions: ["Refresh Connection"],
        confidence: 95
    };
};

const SmartNewTab: React.FC<{ apps: AppDefinition[], onNavigate: (uri: string) => void }> = ({ apps, onNavigate }) => {
    return (
        <div className="h-full flex flex-col items-center justify-center p-8 bg-slate-950">
            <BrainCircuitIcon className="w-24 h-24 text-cyan-400 mb-8" />
            <h1 className="text-3xl font-bold text-white tracking-[0.2em] mb-6 font-mono text-center">
                CHIPS <span className="text-cyan-400">BROWSER</span>
            </h1>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 w-full max-w-2xl">
                <button onClick={() => onNavigate(STORE_URI_CHIPS)} className="p-4 rounded border border-cyan-800 hover:border-cyan-400 bg-cyan-900/10 transition-colors flex flex-col items-center">
                    <BoxIcon className="w-8 h-8 text-cyan-300 mb-2" />
                    <span className="text-xs font-bold">App Store</span>
                </button>
                <button onClick={() => onNavigate(DEV_URI)} className="p-4 rounded border border-yellow-800 hover:border-yellow-400 bg-yellow-900/10 transition-colors flex flex-col items-center">
                    <CodeBracketIcon className="w-8 h-8 text-yellow-300 mb-2" />
                    <span className="text-xs font-bold">Dev Hub</span>
                </button>
                <button onClick={() => onNavigate(VOICE_URI)} className="p-4 rounded border border-purple-800 hover:border-purple-400 bg-purple-900/10 transition-colors flex flex-col items-center">
                    <MicIcon className="w-8 h-8 text-purple-300 mb-2" />
                    <span className="text-xs font-bold">Voice Chat</span>
                </button>
                <button onClick={() => onNavigate(ECONOMY_URI)} className="p-4 rounded border border-green-800 hover:border-green-400 bg-green-900/10 transition-colors flex flex-col items-center">
                    <BanknotesIcon className="w-8 h-8 text-green-300 mb-2" />
                    <span className="text-xs font-bold">Economy</span>
                </button>
            </div>
        </div>
    );
};

const AIContextSidebar: React.FC<{ context: AIContextData | undefined, isLoading: boolean }> = ({ context, isLoading }) => (
    <div className="w-64 bg-black/80 border-l border-cyan-800/50 p-4 flex flex-col h-full backdrop-blur-md absolute right-0 top-0 bottom-0 z-10">
        <div className="flex items-center gap-2 mb-4 text-cyan-300 border-b border-cyan-800/50 pb-2">
            <SparklesIcon className="w-4 h-4" />
            <span className="text-xs font-bold uppercase tracking-wider">Neural Context</span>
        </div>
        <div className="flex-grow text-xs text-cyan-100 space-y-3 font-mono">
           <p>{context?.summary}</p>
        </div>
    </div>
);

const CHIPSBrowser: React.FC<CHIPSBrowserProps> = ({ initialApp, onToggleAgentQ, apps, onInstallApp, onDeployApp, isFullScreen = false, onToggleFullScreen }) => {
    const [tabs, setTabs] = useState<BrowserTab[]>(() => {
        if (initialApp) {
            const startUri = initialApp.q_uri || initialApp.https_url || '';
            const initContext = analyzeQuantumContext(startUri, initialApp.name);
            return [{
                id: 'tab-0',
                title: initialApp.name,
                uri: startUri,
                history: [startUri],
                historyIndex: 0,
                isLoading: false,
                icon: initialApp.icon,
                aiContext: initContext,
                codeContent: initialApp.code
            }];
        }
        return [{
            id: 'tab-0',
            title: 'New Tab',
            uri: NEW_TAB_URI,
            history: [NEW_TAB_URI],
            historyIndex: 0,
            isLoading: false,
            aiContext: { summary: "System ready.", entities: [], actions: [], confidence: 100 }
        }];
    });
    const [activeTabId, setActiveTabId] = useState<string | null>(tabs[0].id);
    const [intentInput, setIntentInput] = useState('');
    const [showAISidebar, setShowAISidebar] = useState(false);

    const activeTab = tabs.find(t => t.id === activeTabId);

    const createTab = useCallback(() => {
        const newTab = {
            id: `tab-${Date.now()}`,
            title: 'New Tab',
            uri: NEW_TAB_URI,
            history: [NEW_TAB_URI],
            historyIndex: 0,
            isLoading: false,
            aiContext: { summary: "New tab ready.", entities: [], actions: [], confidence: 100 }
        };
        setTabs(prev => [...prev, newTab]);
        setActiveTabId(newTab.id);
    }, []);

    const navigateTab = useCallback((tabId: string, input: string) => {
        setTabs(prev => prev.map(tab => {
            if (tab.id !== tabId) return tab;
            return {
                ...tab,
                uri: input,
                title: input,
                history: [...tab.history, input],
                historyIndex: tab.historyIndex + 1
            };
        }));
        setIntentInput(input);
    }, []);

    const closeTab = (id: string, e: React.MouseEvent) => {
        e.stopPropagation();
        setTabs(prev => prev.filter(t => t.id !== id));
    };

    const renderContent = (tab: BrowserTab) => {
        if (tab.uri === NEW_TAB_URI) return <SmartNewTab apps={apps} onNavigate={(uri) => navigateTab(tab.id, uri)} />;
        if (tab.uri === STORE_URI_CHIPS) return <CHIPSAppStore apps={apps} onInstall={onInstallApp} />;
        
        // Simple fallback
        return (
            <div className="h-full flex flex-col items-center justify-center text-cyan-600 bg-black/40">
                <GlobeIcon className="w-12 h-12 mb-4 opacity-50" />
                <h3 className="text-xl font-bold">{tab.title}</h3>
                <p className="text-sm font-mono opacity-60">{tab.uri}</p>
                <div className="mt-8 p-4 border border-cyan-500/20 rounded bg-slate-900/50 max-w-lg w-full">
                     <p className="text-xs text-cyan-400">Quantum Content Buffer:</p>
                     <div className="mt-2 text-[10px] text-cyan-700 break-all font-mono">
                        {tab.codeContent || "No binary data found for this Q-URI in local cache."}
                     </div>
                </div>
            </div>
        );
    };

    return (
        <div className={`h-full flex flex-col bg-slate-950 text-cyan-100 overflow-hidden ${isFullScreen ? '' : 'rounded-xl border border-cyan-500/20'}`}>
            {/* Tab Bar */}
            <div className="flex items-center px-4 pt-2 bg-black/40 border-b border-cyan-900/50 overflow-x-auto no-scrollbar gap-1">
                {tabs.map(tab => (
                    <button 
                        key={tab.id}
                        onClick={() => setActiveTabId(tab.id)}
                        className={`flex items-center gap-2 px-4 py-2 text-xs font-mono rounded-t-lg transition-colors border-t border-l border-r ${activeTabId === tab.id ? 'bg-cyan-900/20 border-cyan-500/40 text-cyan-200' : 'bg-transparent border-transparent text-cyan-800'}`}
                    >
                        <span className="truncate max-w-[100px]">{tab.title}</span>
                        <XIcon className="w-3 h-3 hover:text-red-500" onClick={(e) => closeTab(tab.id, e)} />
                    </button>
                ))}
                <button onClick={createTab} className="p-2 text-cyan-800 hover:text-cyan-400"><PlusIcon className="w-4 h-4" /></button>
            </div>

            {/* Toolbar */}
            <div className="flex items-center gap-2 p-2 bg-black/20 border-b border-cyan-900/30">
                <div className="flex items-center gap-1 pr-2">
                    <button className="p-1.5 hover:bg-cyan-900/30 rounded text-cyan-600"><ChevronLeftIcon className="w-4 h-4" /></button>
                    <button className="p-1.5 hover:bg-cyan-900/30 rounded text-cyan-600"><ChevronRightIcon className="w-4 h-4" /></button>
                    <button onClick={() => activeTabId && navigateTab(activeTabId, activeTab?.uri || '')} className="p-1.5 hover:bg-cyan-900/30 rounded text-cyan-600"><ArrowPathIcon className="w-4 h-4" /></button>
                </div>
                <AgenticAddressBar 
                    value={intentInput}
                    onChange={setIntentInput}
                    onNavigate={(val: string) => activeTabId && navigateTab(activeTabId, val)}
                    isLoading={activeTab?.isLoading}
                />
                <button onClick={() => setShowAISidebar(!showAISidebar)} className={`p-1.5 rounded-full ${showAISidebar ? 'bg-cyan-500 text-black' : 'text-cyan-600 hover:bg-cyan-900/30'}`}>
                    <SparklesIcon className="w-4 h-4" />
                </button>
            </div>

            {/* Content Area */}
            <div className="flex-grow relative overflow-hidden flex">
                <div className="flex-grow relative">
                    {activeTab && renderContent(activeTab)}
                </div>
                {showAISidebar && <AIContextSidebar context={activeTab?.aiContext} isLoading={activeTab?.isLoading || false} />}
            </div>
            
            {/* Status Bar */}
            <div className="p-1 px-3 bg-black/60 border-t border-cyan-900/50 flex justify-between text-[10px] font-mono text-cyan-800">
                <div className="flex gap-4">
                    <span>NODE: DQN-STANDALONE</span>
                    <span>FIDELITY: 99.98%</span>
                </div>
                <div>QCOS V4.5.0-STANDALONE</div>
            </div>
        </div>
    );
};

export default CHIPSBrowser;
