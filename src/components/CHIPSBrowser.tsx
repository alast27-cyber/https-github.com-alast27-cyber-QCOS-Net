
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

// Local Imports
import ChimeraCoreStatus from './ChimeraCoreStatus';
import CHIPSAppStore from './CHIPSAppStore';
import ChipsEconomy from './ChipsEconomy';
import QuantumVoiceChat from './QuantumVoiceChat';
import MonacoEditorWrapper from './MonacoEditorWrapper';
import DeployedAppWrapper from './DeployedAppWrapper';
import LivePreviewFrame from './LivePreviewFrame';
import AgenticAddressBar from './AgenticAddressBar';

// Lazy load ChipsDevPlatform
const ChipsDevPlatform = React.lazy(() => import('./ChipsDevPlatform'));

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
const STORE_URI_WEB = 'https://store.qcos';
const PROTOCOLS_URI = 'chips://protocols';
const DEV_URI = 'chips://dev';
const ECONOMY_URI = 'chips://economy';
const VOICE_URI = 'chips://q-vox.qcos.apps/main';

const analyzeQuantumContext = (uri: string, title: string): AIContextData => {
    const lowerUri = uri.toLowerCase();
    
    if (lowerUri.includes('store')) {
        return {
            summary: "Decentralized Registry Access. Analyzing 15 new Q-App submissions. Network trust score for this node is 99.8%.",
            entities: ["Registry Contract", "DQN-Manifest", "Verification-Sig"],
            actions: ["Scan for Updates", "Verify Signatures"],
            confidence: 99.9
        };
    }
    if (lowerUri.includes('economy') || lowerUri.includes('finance')) {
        return {
            summary: "Streaming QMC financial data. Volatility vectors detected in Sector 7. Recommendation: Hedging via Smart Contract.",
            entities: ["Q-Credits", "Liquidity Pool", "Risk Vector"],
            actions: ["Run Risk Sim", "Export Ledger"],
            confidence: 98.4
        };
    }
    if (lowerUri.includes('protocols') || lowerUri.includes('dev')) {
        return {
            summary: "Development Environment Active. Q-Lang compiler v4.2 standing by. Zero syntax errors detected in local cache.",
            entities: ["Compiler", "Q-Lang", "Debugger"],
            actions: ["Compile Source", "Debug Stream"],
            confidence: 100
        };
    }
    if (lowerUri.includes('.py') || lowerUri.includes('.rs') || lowerUri.includes('.cpp') || lowerUri.includes('.q')) {
        return {
            summary: "Polyglot Source Detected. Analyzing syntax tree for logical coherence and quantum-compatibility.",
            entities: ["Source Code", "AST", "Runtime Env"],
            actions: ["Execute", "Lint", "Optimize"],
            confidence: 99.5
        };
    }
    return {
        summary: `Analyzing semantic content of ${title}... Content vector mapped to 12-D Hilbert Space.`,
        entities: ["Unknown Content", "Raw Data"],
        actions: ["Deep Scan", "Save to Memory"],
        confidence: 85.0
    };
};

const DQNStatusBar = ({ eksStatus, fidelity }: { eksStatus: string, fidelity: number }) => (
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
            <span className={`flex items-center ${eksStatus === 'SYNCED' ? 'text-yellow-400' : 'text-red-400 animate-pulse'}`} title="Entangled Key State Synchronized">
                <KeyIcon className="w-3 h-3 mr-1" />
                EKS: <span className="text-white ml-1">{eksStatus} (Dilithium-2)</span>
            </span>
        </div>
        <div className="flex items-center gap-4 hidden sm:flex">
            <span className="text-cyan-600">Qubit Capacity: <span className="text-cyan-300">128</span></span>
            <span className="text-cyan-600">Avg Fidelity: <span className="text-cyan-300">{(fidelity || 0).toFixed(2)}%</span></span>
        </div>
    </div>
);

const CodeRuntimeEnv: React.FC<{ content: string; language: string; fileName: string }> = ({ content, language, fileName }) => {
    const [output, setOutput] = useState<string[]>([]);
    const [isRunning, setIsRunning] = useState(false);
    const [editorValue, setEditorValue] = useState(content);

    const handleRun = () => {
        setIsRunning(true);
        setOutput(['Initializing Polyglot Runtime...', `Loading ${language.toUpperCase()} kernel...`]);
        
        setTimeout(() => {
            let logs: string[] = [];
            if (language === 'python') {
                logs = ['> Executing Python script via Pyodide bridge...', 'Importing modules...', '>> Output: Analysis Complete. Variance: 0.042'];
            } else if (language === 'rust') {
                logs = ['> Compiling (Cargo --release)...', 'Optimizing LLVM IR...', 'Running target/release/bin...', '>> System operational. Throughput: 45GB/s'];
            } else if (language === 'q-lang') {
                logs = ['> Connecting to QPU...', 'Allocating 12 logical qubits...', 'Applying Hadamard Gates...', '>> Measurement Result: |01101> (Probability: 0.98)'];
            } else {
                logs = ['> Interpreting source...', '>> Execution finished with exit code 0.'];
            }

            let delay = 0;
            logs.forEach((log, i) => {
                delay += 800;
                setTimeout(() => {
                    setOutput(prev => [...prev, log]);
                    if (i === logs.length - 1) setIsRunning(false);
                }, delay);
            });
        }, 1000);
    };

    return (
        <div className="h-full flex flex-col bg-slate-900">
            <div className="flex items-center justify-between p-2 bg-black/40 border-b border-cyan-900/50">
                <div className="flex items-center gap-2">
                    <div className="flex items-center px-2 py-1 bg-cyan-900/30 border border-cyan-800 rounded text-xs text-cyan-300">
                        <FileCodeIcon className="w-3.5 h-3.5 mr-2" />
                        {fileName}
                    </div>
                    <span className="text-[10px] text-gray-500 uppercase font-mono">{language}</span>
                </div>
                <div className="flex items-center gap-2">
                     <span className="text-[10px] text-green-500 font-mono hidden md:block">PARADIGM: {language === 'q-lang' ? 'QUANTUM' : 'CLASSICAL'}</span>
                    <button 
                        onClick={handleRun}
                        disabled={isRunning}
                        className={`holographic-button px-4 py-1 text-xs font-bold flex items-center gap-2 rounded ${isRunning ? 'bg-gray-800 text-gray-500' : 'bg-green-600/30 border-green-500 text-green-300 hover:bg-green-600/50'}`}
                    >
                        {isRunning ? <LoaderIcon className="w-3.5 h-3.5 animate-spin"/> : <PlayIcon className="w-3.5 h-3.5" />}
                        {isRunning ? 'Running...' : 'Execute'}
                    </button>
                </div>
            </div>

            <div className="flex-grow flex flex-col md:flex-row overflow-hidden">
                <div className="flex-grow md:w-2/3 h-1/2 md:h-full relative border-b md:border-b-0 md:border-r border-cyan-900/30">
                     <MonacoEditorWrapper 
                        code={editorValue} 
                        onChange={(v) => setEditorValue(v || "")} 
                        language={language} 
                        theme="qcos-dark"
                    />
                </div>
                <div className="md:w-1/3 h-1/2 md:h-full bg-black flex flex-col font-mono text-xs">
                    <div className="p-2 bg-gray-900 border-b border-gray-800 text-gray-400 flex items-center gap-2">
                        <TerminalIcon className="w-3.5 h-3.5" /> Output Channel
                    </div>
                    <div className="flex-grow p-3 space-y-1 overflow-y-auto text-green-400/90 font-mono">
                        {output.length === 0 && <span className="text-gray-600 italic">Ready to execute.</span>}
                        {output.map((line, i) => (
                            <div key={i} className="break-words">{line}</div>
                        ))}
                        {isRunning && <div className="animate-pulse">_</div>}
                    </div>
                </div>
            </div>
        </div>
    );
};

const SmartNewTab: React.FC<{ apps: AppDefinition[], onNavigate: (uri: string) => void }> = ({ apps, onNavigate }) => {
    return (
        <div className="h-full flex flex-col items-center justify-center p-8 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-slate-900/80 via-black to-black">
            <div className="relative mb-8 group cursor-default">
                <div className="absolute inset-0 bg-cyan-500/10 blur-3xl rounded-full animate-pulse-slow"></div>
                <BrainCircuitIcon className="w-24 h-24 text-cyan-400 relative z-10 drop-shadow-[0_0_25px_rgba(6,182,212,0.6)] animate-float" />
                <div className="absolute -bottom-4 left-1/2 -translate-x-1/2 text-[10px] text-cyan-600 font-mono tracking-widest bg-black/60 px-2 rounded border border-cyan-900/50 whitespace-nowrap">
                    VIRTUAL QUANTUM MACHINE: ACTIVE
                </div>
            </div>

            <h1 className="text-3xl font-bold text-white tracking-[0.2em] mb-6 font-mono text-center">
                CHIPS <span className="text-cyan-400">BROWSER</span>
            </h1>

            <div className="w-full max-w-4xl">
                <div className="flex items-center justify-between mb-4 px-2">
                    <p className="text-cyan-500 text-xs uppercase tracking-widest">Featured Quantum Sites</p>
                    <button onClick={() => onNavigate(STORE_URI_CHIPS)} className="text-xs text-cyan-400 hover:text-white flex items-center gap-1 group">
                        <BoxIcon className="w-3 h-3 group-hover:text-cyan-300" /> Open App Store
                    </button>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <button onClick={() => onNavigate(STORE_URI_CHIPS)} className="group relative flex flex-col items-center p-4 rounded-xl bg-cyan-900/30 border border-cyan-700/50 hover:border-cyan-400 hover:bg-cyan-800/40 transition-all duration-300">
                        <BoxIcon className="w-8 h-8 text-cyan-300 mb-2 group-hover:scale-110 transition-transform" />
                        <span className="text-xs font-bold text-cyan-100">App Store</span>
                    </button>
                    <button onClick={() => onNavigate(DEV_URI)} className="group relative flex flex-col items-center p-4 rounded-xl bg-yellow-900/20 border border-yellow-700/50 hover:border-yellow-400 hover:bg-yellow-800/30 transition-all duration-300">
                        <CodeBracketIcon className="w-8 h-8 text-yellow-300 mb-2 group-hover:scale-110 transition-transform" />
                        <span className="text-xs font-bold text-yellow-100">ChipsDev</span>
                    </button>
                    <button onClick={() => onNavigate(VOICE_URI)} className="group relative flex flex-col items-center p-4 rounded-xl bg-purple-900/20 border border-purple-700/50 hover:border-purple-400 hover:bg-purple-800/30 transition-all duration-300">
                        <MicIcon className="w-8 h-8 text-purple-300 mb-2 group-hover:scale-110 transition-transform" />
                        <span className="text-xs font-bold text-purple-100">Quantum Voice</span>
                    </button>
                    <button onClick={() => onNavigate(ECONOMY_URI)} className="group relative flex flex-col items-center p-4 rounded-xl bg-green-900/20 border border-green-700/50 hover:border-green-400 hover:bg-green-800/30 transition-all duration-300">
                        <BanknotesIcon className="w-8 h-8 text-green-300 mb-2 group-hover:scale-110 transition-transform" />
                        <span className="text-xs font-bold text-green-100">Economy</span>
                    </button>
                </div>
            </div>
        </div>
    );
};

const AIContextSidebar: React.FC<{ context: AIContextData | undefined, isLoading: boolean }> = ({ context, isLoading }) => (
    <div className="w-64 bg-black/60 border-l border-cyan-800/50 p-4 flex flex-col h-full backdrop-blur-md absolute right-0 top-0 bottom-0 z-10 transition-transform duration-300 transform translate-x-0">
        <div className="flex items-center gap-2 mb-4 text-cyan-300 border-b border-cyan-800/50 pb-2">
            <SparklesIcon className="w-4 h-4" />
            <span className="text-xs font-bold uppercase tracking-wider">Neural Context</span>
            <span className="ml-auto text-[9px] font-mono text-cyan-600">{context?.confidence.toFixed(1)}% CONF</span>
        </div>
        <div className="flex-grow text-xs text-cyan-100 space-y-3 font-mono">
            {isLoading ? (
                <div className="flex flex-col items-center justify-center h-40 text-cyan-500">
                    <LoaderIcon className="w-6 h-6 animate-spin mb-2" />
                    <span>Processing Vector...</span>
                </div>
            ) : (
                <>
                    <div className="bg-cyan-950/40 p-2 rounded border border-cyan-900/50 shadow-inner">
                        <p className="text-cyan-500 mb-1 text-[10px] uppercase font-bold flex items-center gap-2">
                             <BrainCircuitIcon className="w-3 h-3" /> Cognitive Summary
                        </p>
                        <p className="leading-relaxed">{context?.summary || "No active context."}</p>
                    </div>
                    <div className="bg-cyan-950/40 p-2 rounded border border-cyan-900/50">
                        <p className="text-cyan-500 mb-1 text-[10px] uppercase font-bold flex items-center gap-2">
                            <DatabaseIcon className="w-3 h-3" /> Recognized Entities
                        </p>
                        <ul className="list-disc list-inside text-cyan-200">
                            {context?.entities.map((e, i) => (
                                <li key={i}>{e}</li>
                            )) || <li>None identified.</li>}
                        </ul>
                    </div>
                    <div className="mt-4">
                        <p className="text-cyan-500 mb-2 text-[10px] uppercase font-bold flex items-center gap-2">
                            <TerminalIcon className="w-3 h-3" /> Recommended Actions
                        </p>
                        <div className="space-y-2">
                            {context?.actions.map((act, i) => (
                                <button key={i} className="w-full text-left p-2 hover:bg-cyan-900/30 rounded text-cyan-300 flex items-center gap-2 transition-colors border border-transparent hover:border-cyan-800/50">
                                    <ArrowRightIcon className="w-3 h-3" /> {act}
                                </button>
                            )) || <p className="text-gray-500 italic">No actions available.</p>}
                        </div>
                    </div>
                </>
            )}
        </div>
    </div>
);

const CHIPSBrowser: React.FC<CHIPSBrowserProps> = ({ initialApp, onToggleAgentQ, apps, onInstallApp, onDeployApp }) => {
    const [tabs, setTabs] = useState<BrowserTab[]>([]);
    const [activeTabId, setActiveTabId] = useState<string | null>(null);
    const [intentInput, setIntentInput] = useState('');
    const [showAISidebar, setShowAISidebar] = useState(false);
    const [isCoreStatusOpen, setIsCoreStatusOpen] = useState(false);
    const [bookmarks, setBookmarks] = useState<{ uri: string, title: string }[]>([
        { uri: 'chips://qmc-finance', title: 'QMC Finance' }
    ]);

    const [eksDetails, setEksDetails] = useState({
        status: 'SYNCED',
        fidelity: 99.99,
    });

    useEffect(() => {
        const interval = setInterval(() => {
            if (Math.random() > 0.7) {
                setEksDetails(prev => ({ ...prev, status: 'ROTATING', fidelity: 85 }));
                setTimeout(() => {
                    setEksDetails({ status: 'SYNCED', fidelity: 99.99 });
                }, 1500);
            }
        }, 5000);
        return () => clearInterval(interval);
    }, []);

    useEffect(() => {
        if (tabs.length === 0) {
            const startUri = initialApp?.q_uri || NEW_TAB_URI;
            const initContext = analyzeQuantumContext(startUri, initialApp ? initialApp.name : 'New Tab');
            
            const newTab: BrowserTab = {
                id: `tab-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                title: initialApp ? initialApp.name : 'New Tab',
                uri: startUri,
                history: [startUri],
                historyIndex: 0,
                isLoading: false,
                icon: initialApp?.icon,
                aiContext: initContext
            };
            setTabs([newTab]);
            setActiveTabId(newTab.id);
            setIntentInput(startUri === NEW_TAB_URI ? '' : startUri);
        }
    }, [initialApp]);

    useEffect(() => {
        const activeTab = tabs.find(t => t.id === activeTabId);
        if (activeTab) {
            setIntentInput(activeTab.uri === NEW_TAB_URI ? '' : activeTab.uri);
        }
    }, [activeTabId, tabs]);

    const createTab = useCallback(() => {
        const startUri = NEW_TAB_URI;
        const initContext = analyzeQuantumContext(startUri, 'New Tab');
        
        const newTab: BrowserTab = {
            id: `tab-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            title: 'New Tab',
            uri: startUri,
            history: [startUri],
            historyIndex: 0,
            isLoading: false,
            aiContext: initContext
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
                const initContext = analyzeQuantumContext(NEW_TAB_URI, 'New Tab');
                const emptyTab: BrowserTab = {
                    id: `tab-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                    title: 'New Tab',
                    uri: NEW_TAB_URI,
                    history: [NEW_TAB_URI],
                    historyIndex: 0,
                    isLoading: false,
                    aiContext: initContext
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

    const resolveIntent = (input: string): { uri: string, title: string, icon?: any, isCode?: boolean, language?: string } => {
        const lowerInput = input.toLowerCase().trim();
        
        if (lowerInput.endsWith('.py')) return { uri: input, title: input, icon: FileCodeIcon, isCode: true, language: 'python' };
        if (lowerInput.endsWith('.rs')) return { uri: input, title: input, icon: FileCodeIcon, isCode: true, language: 'rust' };
        if (lowerInput.endsWith('.cpp')) return { uri: input, title: input, icon: FileCodeIcon, isCode: true, language: 'cpp' };
        if (lowerInput.endsWith('.q')) return { uri: input, title: input, icon: AtomIcon, isCode: true, language: 'q-lang' };
        if (lowerInput.endsWith('.js') || lowerInput.endsWith('.ts')) return { uri: input, title: input, icon: FileCodeIcon, isCode: true, language: 'typescript' };

        if (lowerInput === 'store' || lowerInput.includes('chips://store')) return { uri: STORE_URI_CHIPS, title: 'Chips Store', icon: BoxIcon };
        if (lowerInput === 'protocols' || lowerInput.includes('chips://protocols')) return { uri: PROTOCOLS_URI, title: 'Protocols', icon: FileCodeIcon };
        if (lowerInput === 'dev' || lowerInput.includes('chips://dev')) return { uri: DEV_URI, title: 'ChipsDev', icon: CodeBracketIcon };
        if (lowerInput === 'economy' || lowerInput.includes('chips://economy')) return { uri: ECONOMY_URI, title: 'Economy', icon: BanknotesIcon };

        if (lowerInput.startsWith('chips://') || lowerInput.startsWith('http')) return { uri: input, title: input };

        return { uri: `chips://search?q=${encodeURIComponent(input)}`, title: `Search: ${input}` };
    };

    const navigateTab = useCallback((tabId: string, input: string) => {
        const { uri, title, icon, isCode, language } = resolveIntent(input);

        setTabs(prev => prev.map(tab => {
            if (tab.id !== tabId) return tab;

            const newHistory = tab.history.slice(0, tab.historyIndex + 1);
            newHistory.push(uri);

            let mockContent = "";
            if (isCode) {
                if (language === 'python') mockContent = "import qcos\nimport numpy as np\n\ndef main():\n    print('Initializing Neural Lattice...')\n    lattice = qcos.QuantumLattice(12)\n    print(f'Stability: {lattice.stability}')\n\nmain()";
                else if (language === 'rust') mockContent = "fn main() {\n    println!(\"Starting QCOS Kernel Daemon...\");\n    let qpu = QuantumProcessingUnit::new();\n    qpu.calibrate();\n}";
                else if (language === 'q-lang') mockContent = "QREG q[4];\nCREG c[4];\n\nOP::H q[0];\nOP::CNOT q[0], q[1];\n\nMEASURE q[0] -> c[0];\nMEASURE q[1] -> c[1];";
                else mockContent = "// Source file loaded from distributed storage";
            }

            return {
                ...tab,
                uri,
                history: newHistory,
                historyIndex: newHistory.length - 1,
                title,
                icon: icon || tab.icon,
                isLoading: true,
                aiContext: { summary: "Loading...", entities: [], actions: [], confidence: 0 },
                codeContent: isCode ? mockContent : undefined,
                codeLanguage: language
            };
        }));

        setIntentInput(uri);

        setTimeout(() => {
            setTabs(prev => prev.map(t => {
                if (t.id === tabId) {
                    const analysis = analyzeQuantumContext(uri, t.title);
                    return { ...t, isLoading: false, aiContext: analysis };
                }
                return t;
            }));
        }, 800);
    }, [apps]);

    const handleIntentSubmit = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter' && activeTabId) {
            navigateTab(activeTabId, intentInput);
        }
    };

    const renderContent = (tab: BrowserTab) => {
        if (tab.isLoading) {
            return (
                <div className="h-full flex flex-col items-center justify-center text-cyan-400">
                    <LoaderIcon className="w-16 h-16 animate-spin mb-4" />
                    <p className="font-mono text-sm animate-pulse uppercase tracking-widest">Resolving Quantum Intent...</p>
                </div>
            );
        }

        if (tab.codeContent && tab.codeLanguage) {
            return (
                <CodeRuntimeEnv 
                    content={tab.codeContent} 
                    language={tab.codeLanguage} 
                    fileName={tab.title}
                />
            );
        }

        if (tab.uri === STORE_URI_CHIPS || tab.uri === STORE_URI_WEB) {
            return <CHIPSAppStore apps={apps} onInstall={onInstallApp} onLaunch={(id) => navigateTab(tab.id, apps.find(a => a.id === id)?.q_uri || `chips://app/${id}`)} />;
        }
        if (tab.uri === PROTOCOLS_URI) return <QuantumProtocolLibrary />;
        if (tab.uri === DEV_URI) return (
            <Suspense fallback={<div className="p-4 text-cyan-500">Loading Dev Platform...</div>}>
                 <ChipsDevPlatform 
                    onAiAssist={async () => ""} 
                    onDeploy={onDeployApp ? onDeployApp : () => alert("Deployment service unavailable in this context.")} 
                 />
            </Suspense>
        );
        if (tab.uri === ECONOMY_URI) return <ChipsEconomy />;
        if (tab.uri === NEW_TAB_URI) return <SmartNewTab apps={apps} onNavigate={(uri) => navigateTab(tab.id, uri)} />;
        if (tab.uri.includes('q-vox')) return <QuantumVoiceChat />;

        const app = apps.find(a => 
            a.q_uri === tab.uri || 
            a.https_url === tab.uri || 
            `chips://app/${a.id}` === tab.uri ||
            (tab.uri.startsWith('chips://') && a.q_uri && tab.uri.includes(a.q_uri.replace('CHIPS://', '').split('/')[0]))
        );
        
        if (app && app.status === 'installed') {
            // Enhanced Rendering for Custom Apps
            if (app.isCustom && app.code) {
                // If the app code is a JSON bundle (multi-file), parse it
                let files: { [key: string]: string } = {};
                let entryCode = app.code;
                
                try {
                    const bundle = JSON.parse(app.code);
                    if (bundle.files && bundle.entry) {
                        files = bundle.files;
                        entryCode = bundle.files[bundle.entry];
                    } else if (bundle.files) {
                         // Fallback if no entry specified
                         files = bundle.files;
                         entryCode = Object.values(bundle.files)[0] as string;
                    } else {
                        // Single file fallback
                        files = { 'App.tsx': app.code };
                    }
                } catch {
                     // Not a bundle, single file
                     files = { 'App.tsx': app.code };
                }

                // If 'App.tsx' is not in files, ensure it is set for LivePreviewFrame to pick up imports correctly
                if (!files['App.tsx']) {
                    files['App.tsx'] = entryCode;
                }

                return (
                    <div className="relative h-full w-full bg-slate-900">
                        <LivePreviewFrame code={entryCode} files={files} />
                    </div>
                );
            }
            
            // If component exists, render it (static apps)
            if (app.component) {
                 return <div className="relative h-full w-full">{app.component}</div>;
            }

            // Fallback for custom apps with missing code (Data Corruption)
            return (
                <div className="h-full flex flex-col items-center justify-center text-center text-gray-400 bg-black/40">
                    <AlertTriangleIcon className="w-16 h-16 text-red-500 mb-4" />
                    <h3 className="text-lg font-mono text-red-300">Quantum Data Corruption</h3>
                    <p className="text-sm mt-2 font-mono text-gray-500 max-w-md break-all">
                        Application binary for <span className="text-cyan-700">{app.name}</span> is missing from local storage.
                    </p>
                    <button onClick={() => onInstallApp(app.id)} className="mt-6 text-cyan-500 hover:text-cyan-300 underline text-sm">Re-download Packet</button>
                </div>
            );
        }

        // Fallback for system modules without explicit component mapping
        if (!app && tab.uri.startsWith('chips://')) {
             return (
                <div className="h-full flex flex-col items-center justify-center text-center text-gray-400">
                    <AlertTriangleIcon className="w-16 h-16 text-gray-600 mb-4" />
                    <h3 className="text-lg font-mono text-gray-300">404: Quantum State Not Found</h3>
                    <p className="text-sm mt-2 font-mono text-gray-500 max-w-md break-all">
                        The Q-URI <span className="text-cyan-700">{tab.uri}</span> could not be resolved.
                    </p>
                    <button onClick={() => navigateTab(tab.id, NEW_TAB_URI)} className="mt-6 text-cyan-500 hover:text-cyan-300 underline text-sm">Return Home</button>
                </div>
            );
        }

        return (
            <div className="h-full flex flex-col items-center justify-center text-center text-gray-400">
                <AlertTriangleIcon className="w-16 h-16 text-gray-600 mb-4" />
                <h3 className="text-lg font-mono text-gray-300">404: Quantum State Not Found</h3>
                <p className="text-sm mt-2 font-mono text-gray-500 max-w-md break-all">
                    The Q-URI <span className="text-cyan-700">{tab.uri}</span> could not be resolved.
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
                    <AIContextSidebar context={activeTab.aiContext} isLoading={activeTab.isLoading} />
                )}
            </main>

            <DQNStatusBar eksStatus={eksDetails.status} fidelity={eksDetails.fidelity} />
        </div>
    );
};

export default CHIPSBrowser;
