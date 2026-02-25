
import React, { useState, useEffect, useRef } from 'react';
import { BrainCircuitIcon, ActivityIcon, LockIcon, ArrowRightIcon, TerminalIcon, CheckCircle2Icon, CpuChipIcon } from './Icons';

interface OnboardingFlowProps {
    onComplete: () => void;
}

const OnboardingFlow: React.FC<OnboardingFlowProps> = ({ onComplete }) => {
    const [stage, setStage] = useState<0 | 1 | 2>(0);
    const [logs, setLogs] = useState<string[]>([]);
    const [handshakeProgress, setHandshakeProgress] = useState(0);

    // Stage 1 Logic: EKS Handshake Simulation
    useEffect(() => {
        if (stage === 1) {
            const sequence = [
                { msg: "[INIT] Scanning local hardware substrate...", delay: 500 },
                { msg: "[OK] Local Node ID: 0x882B9...", delay: 1200 },
                { msg: "Generating local Bell State (|Φ⁺⟩)...", delay: 2000 },
                { msg: "[SYNCING] Q_Verify state reconstruction...", delay: 3500 },
                { msg: "Binding UUID to Entangled Key State (EKS)...", delay: 5000 },
                { msg: "[SUCCESS] Entanglement Integrity Hash: 1.000", delay: 6500 },
            ];

            const timeouts: ReturnType<typeof setTimeout>[] = [];

            sequence.forEach(({ msg, delay }) => {
                const t = setTimeout(() => {
                    setLogs(prev => [...prev, msg]);
                }, delay);
                timeouts.push(t);
            });

            // Progress bar
            const interval = setInterval(() => {
                setHandshakeProgress(prev => Math.min(prev + 1, 100));
            }, 70);

            const finishT = setTimeout(() => {
                clearInterval(interval);
                setStage(2);
            }, 7500);
            timeouts.push(finishT);

            return () => {
                timeouts.forEach(clearTimeout);
                clearInterval(interval);
            };
        }
    }, [stage]);

    return (
        <div className="fixed inset-0 z-[200] bg-slate-950 flex flex-col items-center justify-center text-cyan-100 font-mono overflow-hidden">
            {/* Background Ambient */}
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-cyan-900/10 via-black to-black pointer-events-none"></div>
            
            {/* Stage 0: The Awakening */}
            {stage === 0 && (
                <div className="relative z-10 flex flex-col items-center max-w-lg text-center p-6 animate-fade-in-up">
                    <div className="w-32 h-32 mb-8 relative">
                         <div className="absolute inset-0 bg-cyan-500/20 blur-3xl rounded-full animate-pulse"></div>
                         <BrainCircuitIcon className="w-full h-full text-cyan-400 relative z-10 animate-pulse-slow" />
                    </div>
                    
                    <h1 className="text-3xl font-black tracking-[0.2em] text-white mb-6 uppercase">System Heartbeat Detected</h1>
                    
                    <p className="text-lg text-cyan-200 mb-8 leading-relaxed">
                        Initializing... I am <span className="text-cyan-400 font-bold">Agent Q</span>. 
                        I am not just your browser; I am the neural bridge to the QCOS mesh. 
                        To begin, we must entangle your hardware with the global authority.
                    </p>

                    <button 
                        onClick={() => setStage(1)}
                        className="group relative px-8 py-4 bg-cyan-900/30 border border-cyan-500 text-cyan-300 font-bold text-lg rounded-lg overflow-hidden transition-all hover:bg-cyan-800/50 hover:shadow-[0_0_30px_rgba(6,182,212,0.4)]"
                    >
                        <span className="relative z-10 flex items-center gap-3">
                            Begin Entanglement <ArrowRightIcon className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                        </span>
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-500/10 to-transparent -translate-x-full group-hover:animate-shimmer"></div>
                    </button>
                </div>
            )}

            {/* Stage 1: EKS Handshake */}
            {stage === 1 && (
                <div className="relative z-10 w-full max-w-2xl p-6 flex flex-col items-center">
                     <div className="flex items-center justify-between w-full mb-8 px-12">
                        <div className="flex flex-col items-center">
                            <div className="w-16 h-16 rounded-full border-2 border-cyan-500 bg-cyan-900/20 flex items-center justify-center mb-2 shadow-[0_0_20px_cyan]">
                                <CpuChipIcon className="w-8 h-8 text-cyan-300" />
                            </div>
                            <span className="text-xs uppercase tracking-widest text-cyan-600">Local Node</span>
                        </div>
                        
                        <div className="flex-grow mx-4 relative h-1 bg-gray-800 rounded-full overflow-hidden">
                            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 via-white to-purple-500 w-1/2 animate-flow-right blur-sm"></div>
                            <div className="absolute inset-0 bg-cyan-500 h-full transition-all duration-100" style={{width: `${handshakeProgress}%`}}></div>
                        </div>

                        <div className="flex flex-col items-center">
                            <div className="w-16 h-16 rounded-full border-2 border-purple-500 bg-purple-900/20 flex items-center justify-center mb-2 shadow-[0_0_20px_purple]">
                                <ActivityIcon className="w-8 h-8 text-purple-300" />
                            </div>
                            <span className="text-xs uppercase tracking-widest text-purple-600">Global QAN</span>
                        </div>
                     </div>

                     <div className="text-center mb-8 max-w-md">
                        <p className="text-cyan-200 text-sm italic">
                            "Stay still. I am generating a local Bell State. Binding your UUID to a unique Entangled Key State (EKS). Every packet signed by the laws of physics."
                        </p>
                     </div>

                     <div className="w-full bg-black/60 border border-cyan-900 rounded-lg p-4 font-mono text-xs h-48 overflow-hidden flex flex-col shadow-inner">
                        <div className="flex items-center gap-2 border-b border-cyan-900/50 pb-2 mb-2 text-cyan-500">
                            <TerminalIcon className="w-4 h-4" />
                            <span>HANDSHAKE_PROTOCOL_V4.2</span>
                        </div>
                        <div className="flex-grow flex flex-col justify-end space-y-1">
                            {logs.map((log, i) => (
                                <div key={i} className="animate-fade-in-right text-cyan-100/80">
                                    <span className="text-cyan-600 mr-2">{'>'}</span>{log}
                                </div>
                            ))}
                             <div className="animate-pulse text-cyan-500">_</div>
                        </div>
                     </div>
                </div>
            )}

            {/* Stage 2: AI-Native Realization */}
            {stage === 2 && (
                <div className="relative z-10 flex flex-col items-center max-w-xl text-center p-6 animate-fade-in">
                    <div className="relative mb-8">
                         <div className="absolute inset-0 bg-purple-600/30 blur-3xl rounded-full animate-pulse"></div>
                         <div className="w-40 h-40 rounded-full border-2 border-purple-500 flex items-center justify-center bg-black/50 backdrop-blur-md shadow-[0_0_50px_rgba(168,85,247,0.5)]">
                             <BrainCircuitIcon className="w-20 h-20 text-purple-300" />
                         </div>
                         <div className="absolute -bottom-3 left-1/2 -translate-x-1/2 bg-purple-900/80 text-purple-200 text-[10px] px-3 py-1 rounded-full border border-purple-500 uppercase tracking-widest">
                            Bridge Stable
                         </div>
                    </div>

                    <h2 className="text-2xl font-bold text-white mb-4">The link is stable.</h2>
                    <p className="text-purple-200 mb-8 leading-relaxed">
                        I can now 'think' ahead of your clicks. I will be resolving your Q-URIs and pre-running simulations in the background. 
                        <br/><br/>
                        <span className="text-white font-bold">My mind is now your mind.</span>
                    </p>

                    <div className="w-full relative group">
                        <input 
                            type="text" 
                            placeholder="Enter a CHIPS:// destination or ask me a question..."
                            className="w-full bg-black/50 border border-purple-500/50 rounded-lg py-4 pl-6 pr-12 text-white placeholder-purple-700/50 focus:outline-none focus:border-purple-400 focus:shadow-[0_0_20px_rgba(168,85,247,0.2)] transition-all"
                            onKeyDown={(e) => {
                                if (e.key === 'Enter') onComplete();
                            }}
                            autoFocus
                        />
                        <button 
                            onClick={onComplete}
                            className="absolute right-2 top-2 bottom-2 aspect-square bg-purple-600 hover:bg-purple-500 rounded-md flex items-center justify-center text-white transition-colors"
                        >
                            <ArrowRightIcon className="w-5 h-5" />
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default OnboardingFlow;
