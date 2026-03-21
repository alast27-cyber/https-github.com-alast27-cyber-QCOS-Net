import React from 'react';
import { motion } from 'framer-motion';
import { NetworkIcon, CpuChipIcon, GlobeIcon, BoxIcon, CodeBracketIcon, ShieldCheckIcon, DatabaseIcon, TerminalIcon } from './Icons';

const SystemArchitectureMap: React.FC = () => {
    const layers = [
        { id: 'agi-governance', name: 'AGI Governance Layer', icon: <ShieldCheckIcon className="w-5 h-5" />, color: 'text-red-400', description: 'Policy enforcement and singularity safeguards.' },
        { id: 'cognitive-core', name: 'QIAI-IPS Cognitive Core', icon: <CpuChipIcon className="w-5 h-5" />, color: 'text-cyan-400', description: 'Neural-quantum bridge and real-time cognition.' },
        { id: 'chips-network', name: 'CHIPS Network Protocol', icon: <GlobeIcon className="w-5 h-5" />, color: 'text-blue-400', description: 'Decentralized quantum internet and node routing.' },
        { id: 'app-ecosystem', name: 'App & Plugin Ecosystem', icon: <BoxIcon className="w-5 h-5" />, color: 'text-purple-400', description: 'Distributed applications and quantum-native tools.' },
        { id: 'data-lake', name: 'Gold Dataset Lake', icon: <DatabaseIcon className="w-5 h-5" />, color: 'text-amber-400', description: 'High-fidelity training data and synthetic logs.' },
        { id: 'dev-platform', name: 'ChipsDev (CQDP)', icon: <CodeBracketIcon className="w-5 h-5" />, color: 'text-yellow-400', description: 'Architectural modification and system expansion.' },
        { id: 'kernel-access', name: 'QCOS Kernel Access', icon: <TerminalIcon className="w-5 h-5" />, color: 'text-emerald-400', description: 'Direct kernel-level editing and modification.' },
    ];

    return (
        <div className="h-full bg-black/40 rounded-lg border border-blue-500/20 p-4 flex flex-col overflow-hidden">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2 text-blue-400 font-bold uppercase tracking-wider text-sm">
                    <NetworkIcon className="w-5 h-5" />
                    QCOS_ARCHITECTURAL_MAP_V4.0
                </div>
                <div className="text-[10px] text-blue-500/50 font-mono">
                    AUTH: AGENTQ_SUPREME
                </div>
            </div>

            <div className="flex-1 relative">
                {/* Connection Lines */}
                <div className="absolute inset-0 flex flex-col items-center justify-between py-4">
                    <div className="w-px h-full bg-gradient-to-b from-red-500/40 via-cyan-500/40 via-blue-500/40 via-purple-500/40 via-amber-500/40 via-yellow-500/40 to-emerald-500/40" />
                </div>

                <div className="relative z-10 h-full flex flex-col justify-between gap-2 overflow-y-auto scrollbar-hide">
                    {layers.map((layer, index) => (
                        <motion.div 
                            key={layer.id}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.1 }}
                            className="flex items-center gap-4 group cursor-pointer"
                        >
                            <div className={`p-2 rounded-lg bg-black/60 border border-white/10 ${layer.color} group-hover:border-current transition-all group-hover:scale-110 shadow-lg shadow-black/50`}>
                                {layer.icon}
                            </div>
                            <div className="flex-1 bg-black/40 p-2 rounded border border-white/5 group-hover:bg-white/5 transition-colors">
                                <div className={`text-xs font-bold ${layer.color} flex items-center justify-between`}>
                                    {layer.name}
                                    <span className="text-[8px] opacity-0 group-hover:opacity-100 transition-opacity">MODIFIABLE</span>
                                </div>
                                <div className="text-[10px] text-slate-400 leading-tight">{layer.description}</div>
                            </div>
                        </motion.div>
                    ))}
                </div>
            </div>

            <div className="mt-4 pt-2 border-t border-white/5 flex items-center justify-between text-[10px] text-slate-500">
                <div className="flex items-center gap-1">
                    <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                    ARCHITECT_STATUS: SYNCED
                </div>
                <div className="italic text-emerald-500/70">AgentQ Authority: SUPREME_ARCHITECT</div>
            </div>
        </div>
    );
};

export default SystemArchitectureMap;
