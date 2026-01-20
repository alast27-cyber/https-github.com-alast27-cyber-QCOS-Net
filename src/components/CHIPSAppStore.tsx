
import React, { useState } from 'react';
import { 
    SearchIcon, StarIcon, DownloadCloudIcon, CheckCircle2Icon, 
    CpuChipIcon, GlobeIcon, ShieldCheckIcon, BeakerIcon, 
    ArrowRightIcon, BoxIcon, LayoutGridIcon, CurrencyDollarIcon,
    ServerCogIcon, PuzzlePieceIcon
} from './Icons';
import { AppDefinition } from '../types';

interface CHIPSAppStoreProps {
    apps: AppDefinition[];
    onInstall: (id: string) => void;
    onLaunch: (id: string) => void;
}

const categories = ['All', 'Installed', 'Games', 'Finance', 'Science', 'Security', 'Utilities'];

const categoryIcons: { [key: string]: React.FC<{ className?: string }> } = {
    'All': LayoutGridIcon,
    'Installed': CheckCircle2Icon,
    'Games': PuzzlePieceIcon,
    'Finance': CurrencyDollarIcon,
    'Science': BeakerIcon, 
    'Security': ShieldCheckIcon,
    'Utilities': ServerCogIcon,
};

const CHIPSAppStore: React.FC<CHIPSAppStoreProps> = ({ apps, onInstall, onLaunch }) => {
    const [activeCategory, setActiveCategory] = useState('All');
    const [searchTerm, setSearchTerm] = useState('');

    const filteredApps = apps.filter(app => {
        const matchesSearch = app.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
                              app.description.toLowerCase().includes(searchTerm.toLowerCase());
        
        if (!matchesSearch) return false;
        if (activeCategory === 'All') return true;
        
        const desc = app.description.toLowerCase();
        const name = app.name.toLowerCase();
        
        if (activeCategory === 'Installed') return app.status === 'installed';
        
        if (activeCategory === 'Games' && (
            desc.includes('game') || desc.includes('play') || desc.includes('snake') || 
            desc.includes('chess') || desc.includes('arcade') || name.includes('snake')
        )) return true;

        if (activeCategory === 'Finance' && (desc.includes('finance') || desc.includes('market') || desc.includes('economy'))) return true;
        if (activeCategory === 'Science' && (desc.includes('bio') || desc.includes('molecular') || desc.includes('simulation') || desc.includes('swine'))) return true;
        if (activeCategory === 'Security' && (desc.includes('security') || desc.includes('qkd') || desc.includes('encryption'))) return true;
        if (activeCategory === 'Utilities' && (desc.includes('browser') || desc.includes('visualizer') || desc.includes('solver'))) return true;
        
        return false;
    });

    // Determine featured app (Dynamic if new apps installed)
    const installedApps = apps.filter(a => a.status === 'installed');
    const newestApp = installedApps.length > 0 ? installedApps[0] : null;
    const featuredApp = newestApp || apps.find(a => a.id === 'global-swine-foresight') || apps[0];

    const handleAppAction = (app: AppDefinition) => {
        if (app.status === 'installed') {
            onLaunch(app.id);
            return;
        }

        if (app.id === 'chips-browser-sdk') {
            const sdkContent = `
================================================================
CHIPS BROWSER SDK - QUANTUM DEVELOPMENT KIT
================================================================
Version: 4.2.0-Quantum
Build: 2025.10.31-Release
Architecture: Universal (Classical + QPU)

CONTENTS:
- /bin/qcos-browser-engine
- /lib/quantum-networking
- /src/CHIPSBrowserSDK.tsx
- /docs/api-reference.q

INSTRUCTIONS:
1. Run this installer on your local machine.
2. Ensure you have a valid QCOS Identity Key.
3. Use 'qcos deploy' to publish your custom browser extensions.

[SYSTEM MESSAGE]
This file serves as a bootstrap for the local development environment.
Connecting to QCOS Repository...
Download verified.
            `;
            
            const blob = new Blob([sdkContent], { type: 'text/plain' });
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', 'ChipsBrowserSDK_Installer_v4.2.q');
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.URL.revokeObjectURL(url);
        }
        
        onInstall(app.id);
    };

    return (
        <div className="h-full flex flex-col bg-slate-950 overflow-hidden">
            {/* Header / Nav */}
            <div className="bg-slate-950/80 backdrop-blur-md border-b border-cyan-900/50 px-6 py-4 flex items-center justify-between flex-shrink-0">
                <div className="flex items-center gap-2">
                    <BoxIcon className="w-6 h-6 text-cyan-400" />
                    <h1 className="text-lg font-bold text-white tracking-wide">Chips Quantum App Store</h1>
                </div>
                <div className="relative w-64">
                    <input 
                        type="text" 
                        placeholder="Search quantum apps..." 
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full bg-slate-900 border border-cyan-800 rounded-full py-1.5 pl-9 pr-4 text-sm text-cyan-100 focus:outline-none focus:border-cyan-500 transition-colors"
                    />
                    <SearchIcon className="w-4 h-4 text-cyan-600 absolute left-3 top-1/2 -translate-y-1/2" />
                </div>
            </div>

            {/* Content Area - Fixed, No internal parent scroll */}
            <div className="p-6 max-w-6xl mx-auto w-full space-y-8 flex-grow overflow-hidden flex flex-col">
                
                {/* Featured Hero - Only show on 'All' or if it matches current filter */}
                {featuredApp && !searchTerm && activeCategory === 'All' && (
                    <div className="relative rounded-2xl overflow-hidden border border-cyan-800/50 group cursor-pointer flex-shrink-0" onClick={() => handleAppAction(featuredApp)}>
                        <div className="absolute inset-0 bg-gradient-to-r from-cyan-950 to-blue-950/50 z-0"></div>
                        <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-10 z-0"></div>
                        <div className="relative z-10 p-6 flex flex-col md:flex-row items-center gap-8">
                            <div className="flex-shrink-0 w-24 h-24 bg-black/40 rounded-2xl border border-cyan-500/30 flex items-center justify-center shadow-[0_0_30px_rgba(6,182,212,0.2)] group-hover:scale-105 transition-transform duration-500">
                                <featuredApp.icon className="w-12 h-12 text-cyan-300" />
                            </div>
                            <div className="flex-grow text-center md:text-left">
                                <span className="inline-block px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 text-[10px] font-bold uppercase tracking-wider mb-2">
                                    {featuredApp.isCustom ? 'New Release' : "Editor's Choice"}
                                </span>
                                <h2 className="text-2xl font-bold text-white mb-1">{featuredApp.name}</h2>
                                <p className="text-cyan-200/80 text-xs mb-3 max-w-xl truncate">{featuredApp.description}</p>
                                <button 
                                    className={`px-6 py-2 rounded-full font-bold text-xs transition-all duration-200 flex items-center gap-2
                                        ${featuredApp.status === 'installed' 
                                            ? 'bg-green-500/20 border border-green-500 text-green-300 hover:bg-green-500/30' 
                                            : 'bg-cyan-500 text-black hover:bg-cyan-400 shadow-[0_0_15px_rgba(6,182,212,0.4)]'
                                        }`}
                                    onClick={(e) => { e.stopPropagation(); handleAppAction(featuredApp); }}
                                >
                                    {featuredApp.status === 'installed' ? 'OPEN' : 'GET'}
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {/* Categories Bar */}
                <div className="flex items-center gap-2 overflow-x-auto pb-2 no-scrollbar flex-shrink-0">
                    {categories.map(cat => {
                        const Icon = categoryIcons[cat] || BoxIcon;
                        return (
                            <button
                                key={cat}
                                onClick={() => setActiveCategory(cat)}
                                className={`px-4 py-1.5 rounded-full text-[10px] font-semibold whitespace-nowrap transition-colors border flex items-center gap-2 ${
                                    activeCategory === cat 
                                        ? 'bg-cyan-500/20 border-cyan-500 text-cyan-300' 
                                        : 'bg-transparent border-transparent text-slate-400 hover:text-cyan-200 hover:bg-white/5'
                                }`}
                            >
                                <Icon className="w-3 h-3" />
                                {cat}
                            </button>
                        );
                    })}
                </div>

                {/* App Grid - Local scroll if needed */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 overflow-y-auto custom-scrollbar pr-2 flex-grow content-start">
                    {filteredApps.map(app => (
                        <div key={app.id} className="bg-slate-900/50 border border-cyan-900/30 rounded-xl p-4 flex items-start gap-4 hover:bg-cyan-900/10 hover:border-cyan-700/50 transition-all duration-200 group h-32">
                            <div className="flex-shrink-0 w-12 h-12 bg-black/40 rounded-xl flex items-center justify-center border border-slate-800 group-hover:border-cyan-600/30 transition-colors">
                                <app.icon className="w-6 h-6 text-slate-400 group-hover:text-cyan-300 transition-colors" />
                            </div>
                            <div className="flex-grow min-w-0 flex flex-col h-full">
                                <div className="flex justify-between items-start">
                                    <h3 className="font-bold text-slate-200 truncate pr-2 text-xs group-hover:text-white transition-colors">{app.name}</h3>
                                </div>
                                <p className="text-[10px] text-slate-500 line-clamp-2 mt-1 mb-2 flex-grow">{app.description}</p>
                                
                                <div className="flex items-center justify-between mt-auto">
                                    <div className="flex items-center gap-0.5 text-[9px] text-slate-400">
                                        <StarIcon className="w-2.5 h-2.5 text-yellow-500" />
                                        <span>4.8</span>
                                    </div>

                                    <button
                                        onClick={() => handleAppAction(app)}
                                        disabled={app.status === 'installing' || app.status === 'downloading'}
                                        className={`px-3 py-1 rounded-full text-[10px] font-bold transition-all duration-200 flex items-center
                                            ${app.status === 'installed' 
                                                ? 'bg-green-900/20 text-green-400 border border-green-800 hover:bg-green-900/40' 
                                                : 'bg-cyan-900/30 text-cyan-400 border border-cyan-700/50 hover:bg-cyan-500 hover:text-black hover:border-cyan-500'
                                            }`}
                                    >
                                        {app.status === 'installed' ? 'OPEN' : 
                                         app.status === 'downloading' ? '...' : 
                                         app.status === 'installing' ? '...' : 'GET'}
                                    </button>
                                </div>
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
        </div>
    );
};

export default CHIPSAppStore;
