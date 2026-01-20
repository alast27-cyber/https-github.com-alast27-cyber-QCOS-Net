
import React, { useState, useEffect, useRef } from 'react';
import GlassPanel from './GlassPanel';
import { 
    CpuChipIcon, DownloadCloudIcon, TerminalIcon, 
    AlertTriangleIcon, CheckCircle2Icon, LoaderIcon, 
    ShieldCheckIcon, BoxIcon, ChevronRightIcon 
} from './Icons';

interface ChipsBrowserInstallerProps {
    onInstallComplete: () => void;
}

const ChipsBrowserInstaller: React.FC<ChipsBrowserInstallerProps> = ({ onInstallComplete }) => {
    const [step, setStep] = useState(0);
    const [progress, setProgress] = useState(0);
    const [density, setDensity] = useState(0); // Simulated MB
    const [logs, setLogs] = useState<string[]>([]);
    const [isComplete, setIsComplete] = useState(false);
    const logEndRef = useRef<HTMLDivElement>(null);

    const addLog = (msg: string) => {
        setLogs(prev => [...prev, `> ${msg}`]);
    };

    useEffect(() => {
        if (logEndRef.current) {
            logEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [logs]);

    useEffect(() => {
        const runInstaller = async () => {
            // Stage 1: Analysis
            if (step === 0) {
                addLog("Scanning local filesystem...");
                await new Promise(r => setTimeout(r, 800));
                addLog("Found: ChipsBrowser.exe");
                await new Promise(r => setTimeout(r, 600));
                addLog("ERROR: File integrity check failed. Size: 12 bytes.");
                addLog("DIAGNOSIS: Quantum Symlink collapsed. QPU missing.");
                await new Promise(r => setTimeout(r, 1000));
                addLog("INITIATING BYPASS PROTOCOL...");
                setStep(1);
            }
            
            // Stage 2: Core Compiling
            if (step === 1) {
                const interval = setInterval(() => {
                    setProgress(prev => {
                        const next = prev + 1;
                        if (next >= 100) {
                            clearInterval(interval);
                            setStep(2);
                        }
                        return next;
                    });
                    setDensity(prev => prev + (Math.random() * 2.5));
                }, 50);
                return () => clearInterval(interval);
            }

            // Stage 3: Linking & Handshake
            if (step === 2) {
                addLog("Core compiled. Size: 142MB.");
                addLog("Linking Q-Lang Libraries...");
                await new Promise(r => setTimeout(r, 800));
                addLog("Establishing Entangled Key State (EKS)...");
                await new Promise(r => setTimeout(r, 1200));
                addLog("EKS: SYNCED [Dilithium-2]");
                setStep(3);
            }

            // Stage 4: Finalize
            if (step === 3) {
                addLog("[QCOS::SETUP] Agent Q is now active on this machine.");
                addLog("[QCOS::SETUP] Bypass Successful.");
                setIsComplete(true);
            }
        };

        runInstaller();
    }, [step]);

    return (
        <div className="fixed inset-0 z-[200] bg-black flex items-center justify-center font-mono text-cyan-100">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-cyan-900/20 via-black to-black pointer-events-none"></div>
            
            <div className="w-full max-w-2xl bg-black/80 border border-cyan-900 rounded-lg shadow-2xl overflow-hidden relative">
                {/* Header */}
                <div className="bg-cyan-950/50 p-3 border-b border-cyan-800 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <CpuChipIcon className="w-5 h-5 text-cyan-400" />
                        <span className="font-bold text-sm tracking-widest">QCOS INSTALLER WIZARD</span>
                    </div>
                    <div className="flex gap-2">
                        <div className="w-3 h-3 rounded-full bg-red-500/50"></div>
                        <div className="w-3 h-3 rounded-full bg-yellow-500/50"></div>
                        <div className="w-3 h-3 rounded-full bg-green-500/50"></div>
                    </div>
                </div>

                {/* Content */}
                <div className="p-6 flex flex-col gap-6">
                    
                    {/* Status Visualizer */}
                    <div className="flex items-center justify-center py-4">
                        <div className="relative">
                            <div className={`w-32 h-32 rounded-full border-4 flex items-center justify-center transition-all duration-500 ${isComplete ? 'border-green-500 shadow-[0_0_30px_#22c55e]' : 'border-cyan-500/30'}`}>
                                {isComplete ? (
                                    <CheckCircle2Icon className="w-16 h-16 text-green-400 animate-fade-in" />
                                ) : (
                                    <DownloadCloudIcon className="w-12 h-12 text-cyan-600 animate-pulse" />
                                )}
                            </div>
                            {!isComplete && (
                                <div className="absolute inset-0 rounded-full border-t-4 border-cyan-400 animate-spin"></div>
                            )}
                        </div>
                    </div>

                    {/* Progress Bars */}
                    <div className="space-y-4">
                        <div className="flex justify-between text-xs uppercase tracking-wider">
                            <span>Compilation Density</span>
                            <span className="text-cyan-400">{density.toFixed(1)} MB / 150 MB</span>
                        </div>
                        <div className="h-2 w-full bg-gray-800 rounded-full overflow-hidden">
                            <div 
                                className="h-full bg-cyan-500 transition-all duration-75 ease-linear" 
                                style={{ width: `${Math.min(100, (density / 150) * 100)}%` }}
                            ></div>
                        </div>

                        <div className="flex justify-between text-xs uppercase tracking-wider">
                            <span>Installation Progress</span>
                            <span className="text-green-400">{progress}%</span>
                        </div>
                        <div className="h-2 w-full bg-gray-800 rounded-full overflow-hidden">
                            <div 
                                className={`h-full transition-all duration-300 ${isComplete ? 'bg-green-500' : 'bg-purple-500'}`} 
                                style={{ width: `${progress}%` }}
                            ></div>
                        </div>
                    </div>

                    {/* Terminal Output */}
                    <div className="h-48 bg-black/50 border border-cyan-900/30 rounded p-3 overflow-y-auto font-mono text-xs text-gray-300 custom-scrollbar shadow-inner">
                        {logs.map((log, i) => (
                            <div key={i} className="mb-1">{log}</div>
                        ))}
                        <div ref={logEndRef} />
                        {!isComplete && <div className="animate-pulse text-cyan-500">_</div>}
                    </div>

                    {/* Action Button */}
                    <div className="flex justify-end">
                        <button 
                            onClick={onInstallComplete}
                            disabled={!isComplete}
                            className={`px-8 py-3 rounded font-bold text-sm flex items-center gap-2 transition-all duration-300 ${
                                isComplete 
                                ? 'bg-green-600 hover:bg-green-500 text-white shadow-[0_0_20px_rgba(34,197,94,0.4)]' 
                                : 'bg-gray-800 text-gray-500 cursor-not-allowed'
                            }`}
                        >
                            {isComplete ? 'LAUNCH CHIPS BROWSER' : 'INSTALLING...'} 
                            {isComplete && <ChevronRightIcon className="w-4 h-4" />}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ChipsBrowserInstaller;
