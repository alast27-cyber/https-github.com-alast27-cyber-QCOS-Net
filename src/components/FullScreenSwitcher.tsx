
import React from 'react';
import { XIcon, GridIcon } from './Icons';

interface PanelOption {
    id: string;
    title: string;
    icon: React.FC<{ className?: string }>;
}

interface FullScreenSwitcherProps {
    isOpen: boolean;
    onToggle: () => void;
    onPanelSelect: (panelId: string) => void;
    corePanels: PanelOption[];
    appPanels: PanelOption[];
    className?: string;
}

const FullScreenSwitcher: React.FC<FullScreenSwitcherProps> = ({ 
    isOpen, 
    onToggle, 
    onPanelSelect, 
    corePanels, 
    appPanels,
    className = ""
}) => {
    if (!isOpen) {
        return (
            <div className={className}>
                <button 
                    onClick={onToggle}
                    className="p-3 bg-black/60 backdrop-blur-md border border-cyan-500/30 rounded-full text-cyan-400 hover:bg-cyan-900/40 hover:text-white transition-all hover:scale-110 shadow-[0_0_15px_rgba(6,182,212,0.3)]"
                    title="Open Panel Switcher"
                >
                    <GridIcon className="w-6 h-6" />
                </button>
            </div>
        );
    }

    return (
        <div className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-xl animate-fade-in flex flex-col items-center justify-center">
            <button 
                onClick={onToggle}
                className="absolute top-8 right-8 p-2 text-gray-500 hover:text-white transition-colors"
            >
                <XIcon className="w-8 h-8" />
            </button>

            <div className="w-full max-w-5xl px-8">
                <h2 className="text-3xl font-black text-white tracking-[0.2em] mb-12 text-center uppercase border-b border-white/10 pb-6">
                    <span className="text-cyan-400">QCOS</span> Interface Matrix
                </h2>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
                    {/* Core Systems */}
                    <div className="space-y-6">
                        <h3 className="text-sm font-bold text-cyan-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                            <div className="w-2 h-2 bg-cyan-500 rounded-full animate-pulse"></div>
                            Core Systems
                        </h3>
                        <div className="grid grid-cols-2 gap-4">
                            {corePanels.map(panel => (
                                <button
                                    key={panel.id}
                                    onClick={() => onPanelSelect(panel.id)}
                                    className="group relative flex flex-col items-center p-6 bg-black/40 border border-cyan-900/50 rounded-xl hover:bg-cyan-900/20 hover:border-cyan-500/50 transition-all duration-300"
                                >
                                    <div className="p-4 bg-cyan-950/30 rounded-full mb-4 group-hover:scale-110 transition-transform shadow-[0_0_20px_rgba(6,182,212,0.1)]">
                                        <panel.icon className="w-8 h-8 text-cyan-300" />
                                    </div>
                                    <span className="text-sm font-bold text-gray-200 group-hover:text-white tracking-wide">{panel.title}</span>
                                    <div className="absolute inset-0 border border-cyan-500/0 group-hover:border-cyan-500/30 rounded-xl transition-all scale-95 group-hover:scale-100 pointer-events-none"></div>
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Installed Applications */}
                    <div className="space-y-6">
                        <h3 className="text-sm font-bold text-purple-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                            <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                            Active Applications
                        </h3>
                        <div className="grid grid-cols-3 gap-4">
                            {appPanels.map(app => (
                                <button
                                    key={app.id}
                                    onClick={() => onPanelSelect(app.id)}
                                    className="group flex flex-col items-center p-4 bg-black/40 border border-purple-900/30 rounded-xl hover:bg-purple-900/10 hover:border-purple-500/50 transition-all duration-300"
                                >
                                    <div className="w-12 h-12 bg-black/60 rounded-lg flex items-center justify-center mb-3 border border-purple-900/50 group-hover:border-purple-500/50 shadow-lg">
                                        <app.icon className="w-6 h-6 text-purple-400 group-hover:text-purple-200" />
                                    </div>
                                    <span className="text-xs font-medium text-gray-400 group-hover:text-white text-center line-clamp-2">{app.title}</span>
                                </button>
                            ))}
                            {appPanels.length === 0 && (
                                <div className="col-span-3 py-12 text-center text-gray-600 italic border border-dashed border-gray-800 rounded-xl">
                                    No active applications found.
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default FullScreenSwitcher;
