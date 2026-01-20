
import React, { useState, useEffect } from 'react';
import { 
    SparklesIcon, CodeBracketIcon, CubeIcon, LoaderIcon, BugAntIcon, 
    Share2Icon, RocketLaunchIcon, XIcon, CheckCircle2Icon, AlertTriangleIcon, 
    FileCodeIcon, DocumentArrowUpIcon, LightBulbIcon, EyeIcon, 
    CpuChipIcon, ArrowRightIcon, GlobeIcon, LinkIcon, ClipboardIcon,
    BoxIcon, ShieldCheckIcon, KeyIcon
} from './Icons';
import SyntaxHighlighter from './SyntaxHighlighter';
import { UIStructure } from '../types';
import HolographicPreviewRenderer from './HolographicPreviewRenderer';

interface AgentQAppCreatorProps {
    onDeployApp: (details: { name: string; description: string; code: string; uiStructure?: UIStructure }) => void;
    onGenerateApp: (description: string) => Promise<{ files: { [path: string]: string; }; uiStructure: UIStructure | null; }>;
    onUpdateApp: (files: { [path: string]: string; }) => Promise<{ updatedFiles: { [path: string]: string; }; summary: string; }>;
    onDebugApp: (files: { [path: string]: string; }) => Promise<{ fixedFiles: { [path: string]: string; }; summary: string; uiStructure: UIStructure | null; }>;
    onClose: () => void;
}

type CreatorStage = 'ideation' | 'blueprint' | 'studio' | 'deployment';
type SkillMode = 'visionary' | 'architect';

const SUGGESTIONS = [
    { label: "Quantum Secure Chat", prompt: "Create a secure chat application using QKD protocols with a dark holographic theme." },
    { label: "Market Predictor", prompt: "Build a dashboard showing real-time stock trends with a Quantum Monte Carlo simulation chart." },
    { label: "Entanglement Visualizer", prompt: "A visual tool to demonstrate Bell states with interactive qubits." },
    { label: "Task Board", prompt: "A kanban-style task manager optimized for quantum dev teams." }
];

const AgentQAppCreator: React.FC<AgentQAppCreatorProps> = ({ onGenerateApp, onDeployApp, onUpdateApp, onDebugApp, onClose }) => {
    // --- State Management ---
    const [stage, setStage] = useState<CreatorStage>('ideation');
    const [skillMode, setSkillMode] = useState<SkillMode>('visionary');
    
    const [textInput, setTextInput] = useState<string>('');
    const [files, setFiles] = useState<{ [path: string]: string }>({});
    const [uiStructure, setUiStructure] = useState<UIStructure | null>(null);
    const [previewState, setPreviewState] = useState<{ [key: string]: any }>({});
    
    const [isProcessing, setIsProcessing] = useState(false);
    const [statusMessage, setStatusMessage] = useState<string>('');
    const [error, setError] = useState<string | null>(null);
    
    // Deployment State
    const [deployLogs, setDeployLogs] = useState<string[]>([]);
    const [deployComplete, setDeployComplete] = useState(false);
    const [generatedUris, setGeneratedUris] = useState<{ chips: string, public: string } | null>(null);
    const [deploymentStep, setDeploymentStep] = useState<number>(0); // 0: Idle, 1: Package, 2: Sign, 3: Route, 4: Done
    
    const [activeFileTab, setActiveFileTab] = useState<string>('App.tsx');
    const [tweakInput, setTweakInput] = useState('');

    // --- Lifecycle Hooks ---
    useEffect(() => {
        if (Object.keys(files).length > 0 && !files[activeFileTab]) {
            setActiveFileTab(Object.keys(files)[0]);
        }
    }, [files, activeFileTab]);

    // --- Helpers ---
    const extractStateFromCode = (code: string) => {
        const initialStateRegex = /const \[(\w+), set\w+\] = useState\((.*?)\);/g;
        let match;
        const newInitialState: { [key: string]: any } = {};
        while ((match = initialStateRegex.exec(code)) !== null) {
            const stateVar = match[1]; const rawInitialValue = match[2];
            try { newInitialState[stateVar!] = JSON.parse(rawInitialValue); } catch (e) { newInitialState[stateVar!] = rawInitialValue.replace(/['"]/g, ''); }
        }
        return newInitialState;
    };

    const handlePreviewAction = (handlerName: string) => {
        const reactCode = files['App.tsx'];
        if (!reactCode) return;

        // Simple state update simulation for preview
        const functionRegex = new RegExp(`const ${handlerName} = \\(\\) =>\\s*\\{([\\s\\S]*?)\\}`, 'm');
        const match = reactCode.match(functionRegex);
        const functionBody = match ? match[1].trim() : '';
        
        // Basic heuristics for state toggling/incrementing
        const incrementRegex = /set(\w+)\(\s*(\w+)\s*\+\s*1\s*\)/;
        const incrementRegexPrev = /set(\w+)\(\s*prev\w+\s*=>\s*prev\w+\s*\+\s*1\s*\)/;
        const toggleRegex = /set(\w+)\(\s*!(\w+)\s*\)/;
        const toggleRegexPrev = /set(\w+)\(\s*prev\w+\s*=>\s*!prev\w+\s*\)/;

        let incMatch = functionBody.match(incrementRegex) || functionBody.match(incrementRegexPrev);
        let togMatch = functionBody.match(toggleRegex) || functionBody.match(toggleRegexPrev);

        if (incMatch) {
            const varName = incMatch[1].toLowerCase().replace('prev', '');
            const stateKey = Object.keys(previewState).find(k => k.toLowerCase() === varName || varName.includes(k.toLowerCase()));
            if (stateKey) {
                setPreviewState(p => ({ ...p, [stateKey]: (typeof p[stateKey] === 'number' ? p[stateKey] + 1 : 1) }));
            }
        } else if (togMatch) {
            const varName = togMatch[1].toLowerCase().replace('prev', '');
            const stateKey = Object.keys(previewState).find(k => k.toLowerCase() === varName || varName.includes(k.toLowerCase()));
            if (stateKey) {
                setPreviewState(p => ({ ...p, [stateKey]: !p[stateKey] }));
            }
        }
    };

    // --- Core Actions ---

    const handleGenerate = async () => {
        if (!textInput.trim() || isProcessing) return;
        setIsProcessing(true);
        setStatusMessage('Agent Q is dreaming up the architecture...');
        setError(null);

        try {
            await new Promise(r => setTimeout(r, 1500));
            setStatusMessage('Constructing holographic lattice...');
            
            const { files: newFiles, uiStructure: newUiStructure } = await onGenerateApp(textInput);
            
            setFiles(newFiles);
            setUiStructure(newUiStructure);
            
            if (newFiles['App.tsx']) {
                setPreviewState(extractStateFromCode(newFiles['App.tsx']));
            }
            
            setStage('blueprint');
        } catch (e: any) { 
            setError(`Generation failed: ${e.message}`); 
        } finally { 
            setIsProcessing(false); 
        }
    };

    const handleTweak = async () => {
        if (!tweakInput.trim() || isProcessing) return;
        setIsProcessing(true);
        setStatusMessage('Refining quantum matrix...');
        
        try {
            const filesWithInstruction = { ...files };
            if (filesWithInstruction['App.tsx']) {
                filesWithInstruction['App.tsx'] = `// INSTRUCTION: ${tweakInput}\n` + filesWithInstruction['App.tsx'];
            }

            const { updatedFiles, summary } = await onUpdateApp(filesWithInstruction);
            setFiles(updatedFiles);
            if (updatedFiles['App.tsx']) {
                const newState = extractStateFromCode(updatedFiles['App.tsx']);
                setPreviewState(prev => ({ ...newState, ...prev }));
            }
            setStatusMessage(`Update complete: ${summary}`);
            setTweakInput('');
        } catch (e: any) {
            setError(`Update failed: ${e.message}`);
        } finally {
            setIsProcessing(false);
        }
    };

    const handleDebug = async () => {
        setIsProcessing(true);
        setStatusMessage('Analyzing logic gates for coherence...');
        try {
            const { fixedFiles, summary, uiStructure: newUiStructure } = await onDebugApp(files);
            setFiles(fixedFiles);
            if (newUiStructure) setUiStructure(newUiStructure);
            setStatusMessage(`Debug complete: ${summary}`);
        } catch (e: any) {
            setError(`Debug failed: ${e.message}`);
        } finally {
            setIsProcessing(false);
        }
    };

    const handleDeploy = () => {
        setStage('deployment');
        setDeployLogs([]);
        setDeployComplete(false);
        setDeploymentStep(1); // Start Packaging
        
        const manifestContent = files['manifest.json'];
        const appTsxContent = files['App.tsx'];
        
        let appName = "New Quantum App";
        let appDesc = "Generated by Agent Q";

        if (manifestContent) {
            try {
                const manifest = JSON.parse(manifestContent);
                if(manifest.name) appName = manifest.name;
                if(manifest.description) appDesc = manifest.description;
            } catch(e) {
                console.warn("Manifest parsing error", e);
            }
        }

        const appSlug = appName.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');
        const q_uri = `CHIPS://${appSlug}.qcos.apps/main`;
        const https_url = `https://qcos.apps.web/${appSlug}`;
        setGeneratedUris({ chips: q_uri, public: https_url });

        const sequence = [
            { msg: `Analyzing manifest for '${appName}'...`, delay: 800, step: 1 },
            { msg: "Packaging application assets into CHIPS bundle...", delay: 1500, step: 1 },
            { msg: "Calculating Entangled Key State (EKS) Signature...", delay: 1000, step: 2 },
            { msg: "EKS Signature verified. Identity confirmed.", delay: 800, step: 2 },
            { msg: "Registering with Decentralized Quantum Node...", delay: 1200, step: 3 },
            { msg: `Assigning CHIPS Address: ${q_uri}`, delay: 800, step: 3 },
            { msg: "Provisioning Quantum-to-Web Gateway...", delay: 1000, step: 3 },
            { msg: "Deployment Complete. Propagating to edge nodes.", delay: 500, step: 4 }
        ];

        let totalDelay = 0;
        sequence.forEach((item, index) => {
            totalDelay += item.delay;
            setTimeout(() => {
                setDeployLogs(prev => [...prev, item.msg]);
                setDeploymentStep(item.step);
                if (index === sequence.length - 1) {
                    setDeployComplete(true);
                    onDeployApp({
                        name: appName,
                        description: appDesc,
                        code: appTsxContent || "",
                        uiStructure: uiStructure || undefined
                    });
                }
            }, totalDelay);
        });
    };

    // --- Renderers ---

    const renderIdeation = () => (
        <div className="flex flex-col items-center justify-center h-full max-w-2xl mx-auto space-y-8 animate-fade-in-up">
            <div className="text-center space-y-2">
                <div className="inline-flex items-center justify-center p-3 rounded-full bg-cyan-900/30 border border-cyan-500/50 mb-2 shadow-[0_0_20px_theme(colors.cyan.500/30%)]">
                    <SparklesIcon className="w-8 h-8 text-cyan-300 animate-pulse" />
                </div>
                <h2 className="text-2xl font-bold text-white tracking-wide">What shall we build today?</h2>
                <p className="text-cyan-400">Agent Q is ready to architect your vision.</p>
            </div>

            <div className="w-full relative group">
                <div className="absolute -inset-0.5 bg-gradient-to-r from-cyan-500 to-purple-600 rounded-lg blur opacity-30 group-hover:opacity-60 transition duration-500"></div>
                <textarea 
                    className="relative w-full p-4 bg-black rounded-lg border border-cyan-800 text-white placeholder-gray-600 focus:outline-none focus:border-cyan-500 font-mono text-sm resize-none h-32"
                    placeholder="Describe your app... e.g., 'A secure dashboard for monitoring qubit coherence'"
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleGenerate()}
                />
                <button 
                    onClick={handleGenerate}
                    disabled={!textInput.trim() || isProcessing}
                    className="absolute bottom-3 right-3 p-2 bg-cyan-700 hover:bg-cyan-600 rounded-md text-white disabled:opacity-50 transition-colors"
                >
                    {isProcessing ? <LoaderIcon className="w-5 h-5 animate-spin" /> : <ArrowRightIcon className="w-5 h-5" />}
                </button>
            </div>

            <div className="w-full">
                <p className="text-xs text-cyan-600 uppercase tracking-widest mb-3 font-bold">Inspiration</p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {SUGGESTIONS.map((s, i) => (
                        <button 
                            key={i} 
                            onClick={() => setTextInput(s.prompt)}
                            className="text-left p-3 rounded bg-cyan-950/20 border border-cyan-900/50 hover:bg-cyan-900/40 hover:border-cyan-500/50 transition-all text-xs text-cyan-300"
                        >
                            <span className="block font-bold text-white mb-1">{s.label}</span>
                            {s.prompt}
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );

    const renderBlueprint = () => (
        <div className="flex flex-col h-full animate-fade-in">
            <div className="text-center mb-6">
                <h3 className="text-xl font-bold text-white">Blueprint Generated</h3>
                <p className="text-sm text-cyan-400">Agent Q has architected the following structure based on your vision.</p>
            </div>

            <div className="flex-grow flex items-center justify-center mb-6">
                {/* Visual Representation of the UI Tree */}
                <div className="relative w-full max-w-lg aspect-video bg-black/40 border border-cyan-500/30 rounded-lg p-6 flex flex-col items-center justify-center shadow-[0_0_30px_rgba(0,255,255,0.1)_inset]">
                    <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-cyan-500 to-transparent opacity-50"></div>
                    
                    {/* Abstract Tree Nodes */}
                    <div className="flex flex-col items-center gap-4">
                        <div className="px-4 py-2 bg-cyan-900/50 border border-cyan-500 rounded text-white font-mono text-sm animate-pulse-border">
                            {uiStructure?.component || 'App Container'}
                        </div>
                        <div className="h-8 w-px bg-cyan-700/50"></div>
                        <div className="flex gap-4">
                            {(uiStructure?.children || Array.from({length:3})).slice(0,3).map((c: any, i: number) => (
                                <div key={i} className="px-3 py-1 bg-cyan-950/50 border border-cyan-800 rounded text-cyan-300 text-xs">
                                    {c?.component || 'Component'}
                                </div>
                            ))}
                            {(uiStructure?.children?.length || 0) > 3 && (
                                <div className="px-2 py-1 text-cyan-600 text-xs">...</div>
                            )}
                        </div>
                    </div>
                    
                    <div className="absolute bottom-2 right-2 text-[10px] text-cyan-700 font-mono">Q-LANG v4.2 COMPATIBLE</div>
                </div>
            </div>

            <div className="flex justify-center gap-4">
                <button onClick={() => setStage('ideation')} className="px-6 py-2 rounded border border-slate-600 text-slate-400 hover:text-white hover:border-white transition-colors">
                    Back
                </button>
                <button onClick={() => setStage('studio')} className="holographic-button px-8 py-3 bg-cyan-600/30 border-cyan-500 text-white font-bold rounded flex items-center gap-2">
                    <CubeIcon className="w-5 h-5" /> Enter Studio
                </button>
            </div>
        </div>
    );

    const renderStudio = () => (
        <div className="flex flex-col h-full animate-fade-in">
            {/* Studio Header */}
            <div className="flex items-center justify-between p-2 border-b border-cyan-800/50 mb-2">
                <div className="flex items-center gap-2">
                    <button onClick={() => setStage('blueprint')} className="text-cyan-500 hover:text-white"><ArrowRightIcon className="w-4 h-4 rotate-180" /></button>
                    <span className="text-sm font-bold text-white">Studio</span>
                    <span className="text-xs text-cyan-600 font-mono">/ {skillMode.toUpperCase()} MODE</span>
                </div>
                <div className="flex bg-black/40 rounded p-0.5 border border-cyan-900">
                    <button onClick={() => setSkillMode('visionary')} className={`px-3 py-1 text-[10px] uppercase font-bold rounded transition-colors ${skillMode === 'visionary' ? 'bg-cyan-700 text-white' : 'text-cyan-500 hover:text-cyan-300'}`}>Visionary</button>
                    <button onClick={() => setSkillMode('architect')} className={`px-3 py-1 text-[10px] uppercase font-bold rounded transition-colors ${skillMode === 'architect' ? 'bg-purple-700 text-white' : 'text-purple-500 hover:text-purple-300'}`}>Architect</button>
                </div>
            </div>

            <div className="flex-grow flex gap-2 min-h-0 relative">
                {/* Main Preview Area */}
                <div className={`flex-grow bg-black/20 border border-cyan-800/50 rounded-lg overflow-hidden flex flex-col relative transition-all duration-300 ${skillMode === 'architect' ? 'w-1/2' : 'w-full'}`}>
                    <div className="absolute top-0 left-0 right-0 p-2 flex justify-between pointer-events-none z-10">
                        <span className="bg-black/60 text-cyan-300 text-[10px] px-2 py-1 rounded backdrop-blur-sm border border-cyan-900 shadow-sm">Holographic Live Preview</span>
                    </div>
                    
                    <div className="flex-grow overflow-auto p-4 holographic-blueprint-bg relative">
                        <div className="absolute inset-0 holographic-grid opacity-20 pointer-events-none"></div>
                        
                        <div className="relative z-10">
                            {uiStructure ? (
                                <HolographicPreviewRenderer structure={uiStructure} state={previewState} onAction={handlePreviewAction} />
                            ) : (
                                <div className="flex items-center justify-center h-full text-red-400 mt-20">
                                    <AlertTriangleIcon className="w-6 h-6 mr-2" /> Structure Lost
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Visionary Controls (Bottom Overlay) */}
                    <div className="p-3 bg-black/80 backdrop-blur-md border-t border-cyan-800 flex gap-2">
                        <div className="relative flex-grow">
                            <input 
                                type="text" 
                                value={tweakInput}
                                onChange={(e) => setTweakInput(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && handleTweak()}
                                disabled={isProcessing}
                                placeholder="Describe a change (e.g., 'Make buttons blue', 'Add a header')..."
                                className="w-full bg-cyan-900/20 border border-cyan-700/50 rounded pl-9 pr-2 py-2 text-sm text-white placeholder-cyan-700 focus:outline-none focus:border-cyan-400"
                            />
                            <SparklesIcon className="w-4 h-4 text-cyan-400 absolute left-3 top-2.5" />
                        </div>
                        <button onClick={handleTweak} disabled={isProcessing} className="p-2 bg-cyan-700 hover:bg-cyan-600 rounded text-white disabled:opacity-50 transition-colors">
                            {isProcessing ? <LoaderIcon className="w-4 h-4 animate-spin"/> : <ArrowRightIcon className="w-4 h-4"/>}
                        </button>
                    </div>
                </div>

                {/* Architect Code View */}
                {skillMode === 'architect' && (
                    <div className="w-1/2 bg-black/40 border border-purple-900/50 rounded-lg flex flex-col overflow-hidden animate-fade-in-right">
                        <div className="flex items-center bg-black/40 border-b border-purple-900/30 overflow-x-auto no-scrollbar">
                            {Object.keys(files).map(file => (
                                <button key={file} onClick={() => setActiveFileTab(file)} className={`px-3 py-2 text-xs font-mono border-r border-purple-900/30 whitespace-nowrap ${activeFileTab === file ? 'bg-purple-900/20 text-purple-200 border-b-2 border-b-purple-500' : 'text-purple-500 hover:bg-white/5'}`}>
                                    {file}
                                </button>
                            ))}
                        </div>
                        <div className="flex-grow overflow-auto relative">
                            <SyntaxHighlighter code={files[activeFileTab] || ''} language="tsx" />
                        </div>
                        <div className="p-2 bg-black/60 border-t border-purple-900/30 flex justify-end gap-2">
                            <button onClick={handleDebug} disabled={isProcessing} className="px-3 py-1 bg-yellow-600/20 text-yellow-300 text-xs rounded border border-yellow-600/50 flex items-center gap-1 hover:bg-yellow-600/30">
                                <BugAntIcon className="w-3 h-3" /> Auto-Fix
                            </button>
                        </div>
                    </div>
                )}
            </div>

            {/* Global Actions */}
            <div className="flex justify-between items-center mt-2 pt-2 border-t border-cyan-800/30">
                <span className="text-xs text-cyan-600">{statusMessage}</span>
                <button onClick={handleDeploy} className="holographic-button px-6 py-2 bg-green-600/20 border-green-500 text-green-300 font-bold rounded flex items-center gap-2 hover:bg-green-600/40">
                    <RocketLaunchIcon className="w-4 h-4" /> Deploy App
                </button>
            </div>
        </div>
    );

    const renderDeployment = () => (
        <div className="flex flex-col h-full animate-fade-in p-2 relative gap-4">
            
            <div className="flex items-center justify-between mb-2">
                <h2 className="text-xl font-bold text-white flex items-center">
                    <RocketLaunchIcon className={`w-6 h-6 mr-2 ${deployComplete ? 'text-green-400' : 'text-cyan-400 animate-pulse'}`} />
                    {deployComplete ? "Deployment Successful" : "Deploying to CHIPS Network..."}
                </h2>
                {deployComplete && <button onClick={onClose} className="holographic-button px-4 py-1.5 text-xs">Close</button>}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 flex-grow min-h-0">
                {/* Left Column: Status Cards */}
                <div className="space-y-4">
                    {/* Step 1: Packaging */}
                    <div className={`p-4 rounded-lg border transition-all duration-500 ${deploymentStep >= 1 ? 'bg-cyan-950/40 border-cyan-500/50' : 'bg-black/20 border-gray-800 opacity-50'}`}>
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-bold text-cyan-200 flex items-center"><BoxIcon className="w-4 h-4 mr-2"/> Application Packaging</span>
                            {deploymentStep > 1 ? <CheckCircle2Icon className="w-4 h-4 text-green-400"/> : deploymentStep === 1 ? <LoaderIcon className="w-4 h-4 text-cyan-400 animate-spin"/> : null}
                        </div>
                        {deploymentStep >= 1 && (
                            <div className="text-xs text-cyan-500 font-mono space-y-1">
                                <p>Manifest: Verified</p>
                                <p>Assets: Compressed (Quantum-Safe)</p>
                                <p>Bundle Size: 1.2MB</p>
                            </div>
                        )}
                    </div>

                    {/* Step 2: Signing */}
                    <div className={`p-4 rounded-lg border transition-all duration-500 ${deploymentStep >= 2 ? 'bg-purple-900/20 border-purple-500/50' : 'bg-black/20 border-gray-800 opacity-50'}`}>
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-bold text-purple-200 flex items-center"><ShieldCheckIcon className="w-4 h-4 mr-2"/> Security & Signing</span>
                            {deploymentStep > 2 ? <CheckCircle2Icon className="w-4 h-4 text-green-400"/> : deploymentStep === 2 ? <LoaderIcon className="w-4 h-4 text-purple-400 animate-spin"/> : null}
                        </div>
                        {deploymentStep >= 2 && (
                            <div className="text-xs text-purple-400 font-mono space-y-1">
                                <p>EKS Signature: <span className="text-white">Valid</span></p>
                                <p>Hash: 0x7F...9A2B</p>
                                <p>Audit: Passed</p>
                            </div>
                        )}
                    </div>

                    {/* Step 3: Routing */}
                    <div className={`p-4 rounded-lg border transition-all duration-500 ${deploymentStep >= 3 ? 'bg-green-900/20 border-green-500/50' : 'bg-black/20 border-gray-800 opacity-50'}`}>
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-bold text-green-200 flex items-center"><GlobeIcon className="w-4 h-4 mr-2"/> Network Routing</span>
                            {deploymentStep > 3 ? <CheckCircle2Icon className="w-4 h-4 text-green-400"/> : deploymentStep === 3 ? <LoaderIcon className="w-4 h-4 text-green-400 animate-spin"/> : null}
                        </div>
                        {deploymentStep >= 3 && (
                            <div className="text-xs text-green-400 font-mono space-y-1">
                                <p>DQN Registration: Complete</p>
                                <p>Gateway: Active</p>
                                <p>Propagation: 100%</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Right Column: Console & Output */}
                <div className="flex flex-col gap-4">
                    {/* Console */}
                    <div className="flex-grow bg-black/60 border border-cyan-900 rounded-lg p-3 font-mono text-[10px] overflow-y-auto shadow-inner h-48">
                        {deployLogs.map((log, i) => (
                            <div key={i} className="mb-1 animate-fade-in break-words">
                                <span className="text-cyan-600 mr-2">[{new Date().toLocaleTimeString()}]</span>
                                <span className="text-cyan-100">{log}</span>
                            </div>
                        ))}
                        {!deployComplete && <div className="animate-pulse text-cyan-500 mt-1">_</div>}
                    </div>

                    {/* Final Output */}
                    {deployComplete && generatedUris && (
                        <div className="bg-cyan-950/30 border border-cyan-800 rounded-lg p-4 animate-fade-in-up">
                            <div className="mb-3">
                                <label className="text-[10px] text-cyan-500 uppercase tracking-wider block mb-1">CHIPS Address</label>
                                <div className="flex items-center bg-black/40 rounded p-2 border border-cyan-900">
                                    <CpuChipIcon className="w-4 h-4 text-cyan-600 mr-2" />
                                    <span className="text-cyan-300 font-mono text-xs truncate flex-grow">{generatedUris.chips}</span>
                                </div>
                            </div>
                            <div className="mb-3">
                                <label className="text-[10px] text-green-500 uppercase tracking-wider block mb-1">Public Web Gateway</label>
                                <div className="flex items-center bg-black/40 rounded p-2 border border-green-900/50">
                                    <GlobeIcon className="w-4 h-4 text-green-600 mr-2" />
                                    <span className="text-green-300 font-mono text-xs truncate flex-grow">{generatedUris.public}</span>
                                    <button 
                                        onClick={() => navigator.clipboard.writeText(generatedUris.public)}
                                        className="ml-2 text-cyan-500 hover:text-white"
                                        title="Copy URL"
                                    >
                                        <ClipboardIcon className="w-4 h-4" />
                                    </button>
                                    <a 
                                        href={generatedUris.public} 
                                        target="_blank" 
                                        rel="noopener noreferrer"
                                        className="ml-2 text-cyan-500 hover:text-white"
                                        title="Open Link"
                                    >
                                        <LinkIcon className="w-4 h-4" />
                                    </a>
                                </div>
                            </div>
                            <button className="w-full py-2 bg-cyan-800/30 hover:bg-cyan-800/50 rounded border border-cyan-700 text-xs text-cyan-200 flex items-center justify-center gap-2 transition-colors">
                                <FileCodeIcon className="w-4 h-4" /> Download Source Bundle
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );

    return (
        <div className="h-full flex flex-col text-cyan-200 relative">
            <button onClick={onClose} className="absolute top-0 right-0 p-2 text-cyan-600 hover:text-white z-50">
                <XIcon className="w-6 h-6" />
            </button>

            {/* Error Toast */}
            {error && (
                <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-red-900/90 border border-red-500 text-white px-4 py-2 rounded shadow-lg flex items-center gap-3 z-50 animate-fade-in-up">
                    <AlertTriangleIcon className="w-5 h-5" />
                    <span className="text-sm">{error}</span>
                    <button onClick={() => setError(null)}><XIcon className="w-4 h-4"/></button>
                </div>
            )}

            <div className="flex-grow p-4 overflow-hidden h-full">
                {stage === 'ideation' && renderIdeation()}
                {stage === 'blueprint' && renderBlueprint()}
                {stage === 'studio' && renderStudio()}
                {stage === 'deployment' && renderDeployment()}
            </div>
        </div>
    );
};

export default AgentQAppCreator;
