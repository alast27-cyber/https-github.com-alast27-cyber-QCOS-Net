import React, { useState, useEffect } from 'react';
import { BrainCircuitIcon, GalaxyIcon, ActivityIcon, SparklesIcon, FileIcon } from './Icons';
import { useSimulation } from '../context/SimulationContext';
import AgiTrainingSimulationRoadmap from './AgiTrainingSimulationRoadmap';

const QcosDashboard: React.FC = () => {
    const { qiaiIps, universeConnections } = useSimulation();
    const [orbitingFiles, setOrbitingFiles] = useState<{ id: number; name: string; angle: number; radius: number; speed: number }[]>([]);
    const [predictedActions, setPredictedActions] = useState<string[]>([
        "Optimize Workspace Memory",
        "Draft Quantum-Encryption Script",
        "Visualize System Data Traffic"
    ]);

    useEffect(() => {
        // Initialize orbiting files for Semantic Gravity Well
        const files = [
            'quantum_architecture.q',
            'qiai_ips_core.rs',
            'grand_universe_sim.py',
            'qcos_kernel.ts',
            'neural_link_config.json'
        ];
        
        const initialOrbiters = files.map((name, i) => ({
            id: i,
            name,
            angle: Math.random() * Math.PI * 2,
            radius: 80 + Math.random() * 60,
            speed: 0.005 + Math.random() * 0.01
        }));
        
        setTimeout(() => setOrbitingFiles(initialOrbiters), 0);

        // Animation loop for orbiting files
        let animationFrameId: number;
        const animate = () => {
            setOrbitingFiles(prev => prev.map(file => ({
                ...file,
                angle: file.angle + file.speed
            })));
            animationFrameId = requestAnimationFrame(animate);
        };
        animate();

        return () => cancelAnimationFrame(animationFrameId);
    }, []);

    return (
        <div className="h-full flex flex-col p-6 bg-black/40 rounded-xl border-2 border-cyan-500/30 text-cyan-100 font-mono relative overflow-hidden shadow-[0_0_50px_rgba(6,182,212,0.15)]">
            {/* Security & Containment Null-Field Border */}
            <div className="absolute inset-0 border-[4px] border-black pointer-events-none z-50"></div>
            
            {/* Ambient Lighting Shift (QLLM Empathic Resonance Matrix) */}
            <div className="absolute inset-0 bg-gradient-to-br from-cyan-900/20 via-transparent to-purple-900/20 pointer-events-none animate-pulse" style={{ animationDuration: '8s' }}></div>

            <div className="flex justify-between items-start z-10 mb-6">
                <div>
                    <h2 className="text-2xl font-black uppercase tracking-[0.3em] text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500 flex items-center gap-3">
                        <BrainCircuitIcon className="w-8 h-8 text-cyan-400" />
                        Q-DASH: OMNI-STATE
                    </h2>
                    <div className="flex items-center gap-2 mt-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-ping"></div>
                        <span className="text-xs text-cyan-600 font-bold tracking-widest">Q-NATIVE PRIME INTEGRATION: 100%</span>
                    </div>
                </div>
                
                {/* Quantum Core Status (QIAI_IPS Health) */}
                <div className="bg-black/60 border border-cyan-500/30 p-3 rounded-lg flex flex-col gap-2">
                    <div className="text-[10px] text-cyan-500 font-bold uppercase tracking-widest mb-1">QIAI_IPS Coherence</div>
                    <div className="flex items-center gap-3 text-xs">
                        <span className="w-8 text-right">ILL</span>
                        <div className="w-24 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                            <div className="h-full bg-cyan-400 animate-pulse" style={{ width: `${qiaiIps.qil.coherence * 100}%` }}></div>
                        </div>
                        <span className="text-cyan-300">{(qiaiIps.qil.coherence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex items-center gap-3 text-xs">
                        <span className="w-8 text-right">IPS</span>
                        <div className="w-24 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                            <div className="h-full bg-purple-400 animate-pulse" style={{ width: `${qiaiIps.qips.coherence * 100}%` }}></div>
                        </div>
                        <span className="text-purple-300">{(qiaiIps.qips.coherence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex items-center gap-3 text-xs">
                        <span className="w-8 text-right">CLL</span>
                        <div className="w-24 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                            <div className="h-full bg-green-400 animate-pulse" style={{ width: `${qiaiIps.qcl.coherence * 100}%` }}></div>
                        </div>
                        <span className="text-green-300">{(qiaiIps.qcl.coherence * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>

            <div className="flex-grow flex flex-col gap-4 z-10 overflow-hidden">
                {/* Top Section: Gravity Well & Sidebar */}
                <div className="flex gap-6 h-1/2 min-h-[300px]">
                    {/* The Semantic Gravity Well */}
                    <div className="flex-grow bg-black/50 border border-cyan-900/50 rounded-xl relative overflow-hidden flex items-center justify-center">
                        <div className="absolute top-4 left-4 text-xs font-bold text-cyan-600 uppercase tracking-widest flex items-center gap-2">
                            <ActivityIcon className="w-4 h-4" /> Semantic Gravity Well
                        </div>
                        
                        {/* Central Core */}
                        <div className="relative w-24 h-24 rounded-full bg-cyan-900/30 border border-cyan-400/50 flex items-center justify-center shadow-[0_0_30px_rgba(6,182,212,0.4)] z-20">
                            <div className="absolute inset-0 rounded-full bg-cyan-400/20 animate-ping"></div>
                            <BrainCircuitIcon className="w-10 h-10 text-cyan-300" />
                        </div>

                        {/* Orbiting Files */}
                        {orbitingFiles.map(file => {
                            const x = Math.cos(file.angle) * file.radius;
                            const y = Math.sin(file.angle) * file.radius;
                            
                            return (
                                <div 
                                    key={file.id}
                                    className="absolute flex items-center gap-2 bg-black/80 border border-cyan-800/50 px-3 py-1.5 rounded-full text-[10px] text-cyan-300 whitespace-nowrap transition-all hover:scale-110 hover:border-cyan-400 hover:text-white cursor-pointer z-30 shadow-lg"
                                    style={{ 
                                        transform: `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`,
                                        left: '50%',
                                        top: '50%'
                                    }}
                                >
                                    <FileIcon className="w-3 h-3 text-cyan-500" />
                                    {file.name}
                                </div>
                            );
                        })}

                        {/* Gravity Rings */}
                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[200px] h-[200px] rounded-full border border-cyan-900/30 border-dashed animate-[spin_20s_linear_infinite] z-10"></div>
                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[300px] h-[300px] rounded-full border border-cyan-900/20 border-dashed animate-[spin_30s_linear_infinite_reverse] z-10"></div>
                    </div>

                    {/* Right Sidebar */}
                    <div className="w-80 flex flex-col gap-6">
                        {/* The Grand Universe Simulator Feed */}
                        <div className="h-48 bg-black/60 border border-purple-500/30 rounded-xl p-4 relative overflow-hidden group">
                            <div className="absolute inset-0 bg-[url('https://picsum.photos/seed/cosmos/400/300')] bg-cover bg-center opacity-30 group-hover:opacity-50 transition-opacity duration-700 mix-blend-screen"></div>
                            <div className="relative z-10 h-full flex flex-col">
                                <div className="text-[10px] font-bold text-purple-400 uppercase tracking-widest flex items-center gap-2 mb-2">
                                    <GalaxyIcon className="w-4 h-4" /> Grand Universe Feed
                                </div>
                                <div className="flex-grow flex items-end">
                                    <div className="bg-black/80 backdrop-blur-sm p-2 rounded border border-purple-900/50 w-full">
                                        <div className="text-[9px] text-purple-300 font-mono">SECTOR 7G: Stellar Nucleosynthesis</div>
                                        <div className="text-[8px] text-gray-500 mt-1">Simulating 10^24 permutations...</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* ILL Intent Predictor */}
                        <div className="flex-grow bg-black/60 border border-cyan-500/30 rounded-xl p-4 flex flex-col">
                            <div className="text-[10px] font-bold text-cyan-400 uppercase tracking-widest flex items-center gap-2 mb-4">
                                <SparklesIcon className="w-4 h-4" /> ILL Intent Predictor
                            </div>
                            <div className="text-xs text-gray-400 mb-3 italic">Pre-cognitive actions ready for execution:</div>
                            <div className="flex flex-col gap-2">
                                {predictedActions.map((action, idx) => (
                                    <button 
                                        key={idx}
                                        className="text-left px-3 py-2 bg-cyan-950/30 border border-cyan-800/50 rounded hover:bg-cyan-900/50 hover:border-cyan-400 transition-all text-xs text-cyan-200 group flex items-center justify-between"
                                    >
                                        {action}
                                        <ActivityIcon className="w-3 h-3 opacity-0 group-hover:opacity-100 transition-opacity text-cyan-400" />
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Bottom Section: Training Roadmap */}
                <div className="h-1/2 min-h-[250px]">
                    <AgiTrainingSimulationRoadmap />
                </div>
            </div>
        </div>
    );
};

export default QcosDashboard;
