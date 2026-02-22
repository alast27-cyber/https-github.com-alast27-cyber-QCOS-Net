import React, { useState } from 'react';
import GlassPanel from './GlassPanel';
import { 
    BoxIcon, SearchIcon, StarIcon, DownloadCloudIcon, 
    CpuChipIcon, GlobeIcon, BeakerIcon, ShieldCheckIcon,
    CurrencyDollarIcon, ServerCogIcon, LayoutGridIcon
} from './Icons';
import { AppDefinition, URIAssignment, UIStructure } from '../types';

interface QuantumAppExchangeProps {
    apps?: AppDefinition[];
    onInstall?: (id: string) => void;
    onLaunch?: (id: string) => void;
    onDeployApp?: (details: { name: string; description: string; code: string; uiStructure?: UIStructure }) => void;
    uriAssignments?: URIAssignment[];
    onGenerateApp?: (description: string) => Promise<{ files: { [path: string]: string }, uiStructure: UIStructure | null }>;
    onUpdateApp?: (files: { [path: string]: string }) => Promise<{ updatedFiles: { [path: string]: string }, summary: string }>;
    onDebugApp?: (files: { [path: string]: string }) => Promise<{ fixedFiles: { [path: string]: string }, summary: string, uiStructure: UIStructure | null }>;
    onSimulate?: (id: string) => void;
}

const categories = ['All', 'Finance', 'Science', 'Security', 'Utilities'];

const QuantumAppExchange: React.FC<QuantumAppExchangeProps> = ({ apps, onInstall, onLaunch }) => {
    const [activeCategory, setActiveCategory] = useState('All');
    const [searchTerm, setSearchTerm] = useState('');

    const filteredApps = (apps || []).filter(app => {
        const matchesSearch = app.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
                              app.description.toLowerCase().includes(searchTerm.toLowerCase());
        
        if (!matchesSearch) return false;
        if (activeCategory === 'All') return true;
        
        const desc = app.description.toLowerCase();
        if (activeCategory === 'Finance' && (desc.includes('finance') || desc.includes('market'))) return true;
        if (activeCategory === 'Science' && (desc.includes('bio') || desc.includes('simulation') || desc.includes('molecular'))) return true;
        if (activeCategory === 'Security' && (desc.includes('security') || desc.includes('qkd'))) return true;
        if (activeCategory === 'Utilities' && (desc.includes('browser') || desc.includes('visualizer'))) return true;
        
        return false;
    });

    return (
        <GlassPanel title={<div className="flex items-center"><BoxIcon className="w-5 h-5 mr-2 text-purple-400" /> Quantum App Exchange</div>}>
            <div className="h-full flex flex-col bg-slate-950/50">
                {/* Header / Nav */}
                <div className="p-4 border-b border-cyan-900/50 flex flex-col space-y-4">
                    <div className="relative">
                        <input 
                            type="text" 
                            placeholder="Search quantum apps..." 
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="w-full bg-black/40 border border-cyan-800 rounded-lg py-2 pl-9 pr-4 text-sm text-cyan-100 focus:outline-none focus:border-cyan-500 transition-colors"
                        />
                        <SearchIcon className="w-4 h-4 text-cyan-600 absolute left-3 top-1/2 -translate-y-1/2" />
                    </div>
                    
                    <div className="flex items-center gap-2 overflow-x-auto pb-1 no-scrollbar">
                        {categories.map(cat => (
                            <button
                                key={cat}
                                onClick={() => setActiveCategory(cat)}
                                className={`px-3 py-1 rounded-full text-xs font-semibold whitespace-nowrap transition-colors border ${
                                    activeCategory === cat 
                                        ? 'bg-purple-900/40 border-purple-500 text-purple-200' 
                                        : 'bg-transparent border-transparent text-slate-400 hover:text-cyan-200 hover:bg-white/5'
                                }`}
                            >
                                {cat}
                            </button>
                        ))}
                    </div>
                </div>

                {/* App Grid */}
                <div className="p-4 overflow-y-auto custom-scrollbar flex-grow grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {filteredApps.map(app => (
                        <div key={app.id} className="bg-black/40 border border-cyan-900/30 rounded-xl p-4 hover:bg-cyan-900/10 hover:border-cyan-700/50 transition-all duration-200 group flex flex-col">
                            <div className="flex items-start justify-between mb-3">
                                <div className="w-10 h-10 bg-black/60 rounded-lg flex items-center justify-center border border-cyan-900/50 group-hover:border-cyan-500/30 transition-colors">
                                    <app.icon className="w-6 h-6 text-cyan-500 group-hover:text-cyan-300" />
                                </div>
                                {app.status === 'installed' && (
                                    <span className="text-[10px] bg-green-900/30 text-green-400 px-2 py-0.5 rounded border border-green-800/50">INSTALLED</span>
                                )}
                            </div>
                            
                            <h3 className="font-bold text-slate-200 text-sm mb-1 group-hover:text-white transition-colors">{app.name}</h3>
                            <p className="text-[10px] text-slate-500 line-clamp-2 mb-4 flex-grow">{app.description}</p>
                            
                            <div className="flex items-center justify-between mt-auto">
                                <div className="flex items-center gap-0.5 text-[9px] text-slate-500">
                                    <StarIcon className="w-2.5 h-2.5 text-yellow-600" />
                                    <span>4.8</span>
                                </div>

                                <button
                                    onClick={() => app.status === 'installed' ? onLaunch?.(app.id) : onInstall?.(app.id)}
                                    className={`px-3 py-1.5 rounded-lg text-[10px] font-bold transition-all duration-200 flex items-center gap-1.5
                                        ${app.status === 'installed' 
                                            ? 'bg-slate-800 text-cyan-400 hover:bg-slate-700' 
                                            : 'bg-purple-600/20 text-purple-300 border border-purple-500/30 hover:bg-purple-600/40 hover:border-purple-500'
                                        }`}
                                >
                                    {app.status === 'installed' ? 'OPEN' : <><DownloadCloudIcon className="w-3 h-3" /> GET</>}
                                </button>
                            </div>
                        </div>
                    ))}
                    
                    {filteredApps.length === 0 && (
                        <div className="col-span-full flex flex-col items-center justify-center text-gray-500 py-8">
                            <BoxIcon className="w-8 h-8 opacity-20 mb-2" />
                            <p className="text-xs">No apps found.</p>
                        </div>
                    )}
                </div>
            </div>
        </GlassPanel>
    );
};

export default QuantumAppExchange;