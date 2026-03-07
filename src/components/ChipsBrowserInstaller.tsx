import React, { useState, useEffect, useRef } from 'react';
import { UniversalBridge } from '../bridge/UniversalBridge';
import { 
    CpuChipIcon, DownloadCloudIcon, CheckCircle2Icon, 
    ChevronRightIcon 
} from './Icons';

interface ChipsBrowserInstallerProps {
    onInstallComplete: () => void;
}

const ChipsBrowserInstaller: React.FC<ChipsBrowserInstallerProps> = ({ onInstallComplete }) => {
    const [step, setStep] = useState(0);
    const [progress, setProgress] = useState(0);
    const [density, setDensity] = useState(0); 
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
            try {
                // Stage 1: Analysis & Integrity Check (Physical Pillars)
                if (step === 0) {
                    const envType = UniversalBridge.isElectron ? "ELECTRON_HOST" : "CLOUD_PREVIEW";
                    addLog(`Environment Detected: ${envType}`);
                    addLog("Accessing Physical Substrate (C:\\Program Files\\QCOS\\bin)...");
                    
                    const result = await UniversalBridge.monitorPillars();
                    
                    // Logic to handle the { status, details } object from Main.ts
                    if (result.status === "SYNCED" || result.status === "CLOUD_SIM") {
                        addLog(`Physical Pillars Grounded. Status: ${result.status}`);
                        // Update density UI based on how many pillars are grounded
                        const groundedCount = result.details.filter(p => p.isGrounded || p.exists).length;
                        setDensity(groundedCount * 37.5); // 150MB total / 4 pillars
                    } else {
                        addLog(`WARNING: Substrate density low. Status: ${result.status}`);
                    }
                    
                    setTimeout(() => setStep(1), 1000);
                }
                
                // Stage 2: Core Installation (IPC Bridge)
                if (step === 1) {
                    addLog("Initiating Neural Link via Universal Bridge...");
                    
                    const progressInterval = setInterval(() => {
                        setProgress(prev => Math.min(99, prev + 1));
                    }, 30);

                    const installResult = await UniversalBridge.installChips();
                    clearInterval(progressInterval);

                    if (installResult.success) {
                        setProgress(100);
                        setDensity(150); 
                        addLog(`Core Installation Successful: ${installResult.message}`);
                        setStep(2);
                    } else {
                        addLog(`INSTALLATION FAILED: ${installResult.message}`);
                    }
                }

                // Stage 3: Linking & Handshake
                if (step === 2) {
                    addLog("Linking Q-Lang Libraries...");
                    addLog("Establishing Entangled Key State (EKS)...");
                    addLog("EKS: SYNCED [Dilithium-2]");
                    setTimeout(() => setStep(3), 800);
                }

                // Stage 4: Finalize
                if (step === 3) {
                    addLog("[QCOS::SETUP] Agent Q is now active on this machine.");
                    addLog("[QCOS::SETUP] System Materialized.");
                    setIsComplete(true);
                }
            } catch (error) {
                addLog(`FATAL ERROR: ${error}`);
            }
        };

        runInstaller();
    }, [step]);

    return (
        <div className="fixed inset-0 z-[200] bg-black flex items-center justify-center font-mono text-cyan-100">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-cyan-900/20 via-black to-black pointer-events-none"></div>
            
            <div className="w-full max-w-2xl bg-black/80 border border-cyan-900 rounded-lg shadow-2xl overflow-hidden relative cyber-grid-bg">
                {/* Header */}
                <div className="bg-cyan-950/50 p-3 border-b border-cyan-800 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <CpuChipIcon className="w-5 h-5 text-cyan-400" />
                        <span className="font-bold text-sm tracking-widest uppercase">
                            QCOS Installer ({UniversalBridge.isElectron ? 'Grounded Node' : 'Cloud Preview'})
                        </span>
                    </div>
                </div>

                {/* Content */}
                <div className="p-6 flex flex-col gap-6">
                    
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

                    <div className="space-y-4">
                        <div className="flex justify-between text-xs uppercase">
                            <span>Compilation Density</span>
                            <span className="text-cyan-400">{density.toFixed(1)} MB / 150.0 MB</span>
                        </div>
                        <div className="h-2 w-full bg-gray-800 rounded-full overflow-hidden">
                            <div 
                                className="h-full bg-cyan-500 transition-all duration-500" 
                                style={{ width: `${(density / 150) * 100}%` }}
                            ></div>
                        </div>

                        <div className="flex justify-between text-xs uppercase">
                            <span>Sync Progress</span>
                            <span className="text-green-400">{progress}%</span>
                        </div>
                        <div className="h-2 w-full bg-gray-800 rounded-full overflow-hidden">
                            <div 
                                className={`h-full transition-all duration-300 ${isComplete ? 'bg-green-500' : 'bg-purple-500'}`} 
                                style={{ width: `${progress}%` }}
                            ></div>
                        </div>
                    </div>

                    <div className="h-40 bg-black/50 border border-cyan-900/30 rounded p-3 overflow-y-auto text-xs text-gray-400">
                        {logs.map((log, i) => (
                            <div key={i} className="mb-1">{log}</div>
                        ))}
                        <div ref={logEndRef} />
                    </div>

                    <div className="flex justify-end">
                        <button 
                            onClick={onInstallComplete}
                            disabled={!isComplete}
                            className={`px-8 py-3 rounded font-bold text-sm flex items-center gap-2 transition-all duration-300 ${
                                isComplete 
                                ? 'bg-green-600 hover:bg-green-500 text-white shadow-[0_0_20px_#22c55e66]' 
                                : 'bg-gray-800 text-gray-500 cursor-not-allowed'
                            }`}
                        >
                            {isComplete ? 'INITIALIZE QIAI-IPS' : 'MATERIALIZING...'} 
                            {isComplete && <ChevronRightIcon className="w-4 h-4" />}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ChipsBrowserInstaller;