import React from 'react';
import { CpuChipIcon, ActivityIcon, GlobeIcon, ServerCogIcon, ShieldCheckIcon, ZapIcon, SettingsIcon } from './Icons';
import CHIPSBackOffice from './CHIPSBackOffice';
import { useQuantumApps } from '../hooks/useQuantumApps';

const ChipsDash: React.FC = () => {
    const [barHeights] = React.useState(() => Array.from({ length: 20 }).map(() => 20 + Math.random() * 80));
    
    // Dummy functions for useQuantumApps
    const addLog = () => {};
    const handlePanelSelect = () => {};
    
    const { marketApps, uriAssignments } = useQuantumApps(addLog, handlePanelSelect);

    return (
        <div className="h-full flex flex-col p-6 bg-black/40 rounded-xl border-2 border-emerald-500/30 text-emerald-100 font-mono relative overflow-hidden shadow-[0_0_50px_rgba(16,185,129,0.15)]">
            {/* Security & Containment Null-Field Border */}
            <div className="absolute inset-0 border-[4px] border-black pointer-events-none z-50"></div>

            {/* Ambient Lighting Shift */}
            <div className="absolute inset-0 bg-gradient-to-br from-emerald-900/20 via-transparent to-cyan-900/20 pointer-events-none animate-pulse" style={{ animationDuration: '8s' }}></div>

            <div className="flex justify-between items-start z-10 mb-6">
                <div>
                    <h2 className="text-2xl font-black uppercase tracking-[0.3em] text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-500 flex items-center gap-3">
                        <CpuChipIcon className="w-8 h-8 text-emerald-400" />
                        CHIPS - DASH
                    </h2>
                    <div className="flex items-center gap-4 mt-2">
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-ping"></div>
                            <span className="text-xs text-emerald-600 font-bold tracking-widest">SYSTEM OPTIMIZED</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-cyan-500 rounded-full"></div>
                            <span className="text-xs text-cyan-600 font-bold tracking-widest">QUANTUM LINK: ACTIVE</span>
                        </div>
                    </div>
                </div>
                
                <div className="bg-black/60 border border-emerald-500/30 p-3 rounded-lg flex flex-col gap-2">
                    <div className="text-[10px] text-emerald-500 font-bold uppercase tracking-widest mb-1">System Resources</div>
                    <div className="flex items-center gap-3 text-xs">
                        <span className="w-12 text-right text-emerald-700">CPU</span>
                        <div className="w-24 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                            <div className="h-full bg-emerald-400 animate-pulse" style={{ width: '45%' }}></div>
                        </div>
                        <span className="text-emerald-300">45%</span>
                    </div>
                    <div className="flex items-center gap-3 text-xs">
                        <span className="w-12 text-right text-emerald-700">MEM</span>
                        <div className="w-24 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                            <div className="h-full bg-cyan-400 animate-pulse" style={{ width: '62%' }}></div>
                        </div>
                        <span className="text-cyan-300">62%</span>
                    </div>
                </div>
            </div>

            <div className="flex-grow grid grid-cols-1 lg:grid-cols-3 gap-6 z-10 overflow-y-auto pr-2 custom-scrollbar">
                {/* Main Activity Panel */}
                <div className="lg:col-span-2 flex flex-col gap-6">
                    <div className="bg-black/50 border border-emerald-900/50 rounded-xl p-4 flex-grow min-h-[300px] relative overflow-hidden group">
                        <div className="absolute inset-0 bg-[url('https://picsum.photos/seed/tech/800/600')] bg-cover bg-center opacity-10 group-hover:opacity-20 transition-opacity duration-700"></div>
                        <h3 className="text-emerald-400 font-bold uppercase tracking-widest mb-4 flex items-center gap-2 relative z-10">
                            <ActivityIcon className="w-4 h-4" /> Network Activity Stream
                        </h3>
                        <div className="relative z-10 h-full flex flex-col justify-center items-center text-emerald-800/50 font-bold tracking-widest">
                            <div className="w-full h-32 flex items-end justify-center gap-1 mb-4">
                                {barHeights.map((height, i) => (
                                    <div 
                                        key={i} 
                                        className="w-3 bg-emerald-500/30 hover:bg-emerald-400/60 transition-all duration-300"
                                        style={{ height: `${height}%` }}
                                    ></div>
                                ))}
                            </div>
                            [REAL-TIME DATA VISUALIZATION]
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-6 h-48">
                        <div className="bg-black/50 border border-emerald-900/50 rounded-xl p-4 relative overflow-hidden">
                            <h3 className="text-emerald-400 font-bold uppercase tracking-widest mb-2 flex items-center gap-2 text-xs">
                                <ShieldCheckIcon className="w-3 h-3" /> Security Protocols
                            </h3>
                            <div className="space-y-2 mt-4">
                                <div className="flex justify-between items-center text-xs border-b border-emerald-900/30 pb-1">
                                    <span className="text-emerald-200">Firewall</span>
                                    <span className="text-emerald-500 font-bold">ACTIVE</span>
                                </div>
                                <div className="flex justify-between items-center text-xs border-b border-emerald-900/30 pb-1">
                                    <span className="text-emerald-200">Encryption</span>
                                    <span className="text-emerald-500 font-bold">AES-256-GCM</span>
                                </div>
                                <div className="flex justify-between items-center text-xs border-b border-emerald-900/30 pb-1">
                                    <span className="text-emerald-200">Intrusion Detection</span>
                                    <span className="text-emerald-500 font-bold">MONITORING</span>
                                </div>
                            </div>
                        </div>
                        <div className="bg-black/50 border border-emerald-900/50 rounded-xl p-4 relative overflow-hidden">
                             <h3 className="text-emerald-400 font-bold uppercase tracking-widest mb-2 flex items-center gap-2 text-xs">
                                <ZapIcon className="w-3 h-3" /> Power Distribution
                            </h3>
                            <div className="flex items-center justify-center h-full pb-4">
                                <div className="relative w-24 h-24 rounded-full border-4 border-emerald-900/30 flex items-center justify-center">
                                    <div className="absolute inset-0 rounded-full border-t-4 border-emerald-500 animate-spin"></div>
                                    <div className="text-center">
                                        <div className="text-xl font-bold text-white">98%</div>
                                        <div className="text-[8px] text-emerald-500 uppercase">Efficiency</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Sidebar */}
                <div className="flex flex-col gap-6">
                    <div className="bg-black/50 border border-emerald-900/50 rounded-xl p-4 h-1/2 min-h-[200px]">
                        <h3 className="text-emerald-400 font-bold uppercase tracking-widest mb-4 flex items-center gap-2">
                            <GlobeIcon className="w-4 h-4" /> Global Nodes
                        </h3>
                        <div className="space-y-3 overflow-y-auto max-h-[200px] pr-2 custom-scrollbar">
                            {[1, 2, 3, 4, 5].map((i) => (
                                <div key={i} className="bg-emerald-950/20 p-2 rounded border border-emerald-900/30 flex items-center justify-between group hover:bg-emerald-900/30 transition-colors cursor-pointer">
                                    <div className="flex items-center gap-2">
                                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></div>
                                        <div>
                                            <div className="text-xs text-emerald-200 font-bold">NODE-0{i}</div>
                                            <div className="text-[8px] text-emerald-600">US-EAST-{i}</div>
                                        </div>
                                    </div>
                                    <div className="text-xs text-emerald-500 font-mono">{10 + i}ms</div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="bg-black/50 border border-emerald-900/50 rounded-xl p-4 h-1/3 min-h-[150px]">
                        <h3 className="text-emerald-400 font-bold uppercase tracking-widest mb-4 flex items-center gap-2">
                            <SettingsIcon className="w-4 h-4" /> Backoffice
                        </h3>
                        <CHIPSBackOffice uriAssignments={uriAssignments} marketApps={marketApps} />
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ChipsDash;
