import React, { useState } from 'react';
import NeuralProgrammingPanel from './NeuralProgrammingPanel';
import PhiSyncWidget from './PhiSyncWidget';
import NeuralSignalTranslator from './NeuralSignalTranslator';
import ChipsDevPlatform from './ChipsDevPlatform';

const NeuralDash: React.FC = () => {
    const [view, setView] = useState<'main' | 'translator'>('main');
    const [devFullscreen, setDevFullscreen] = useState(false);

    return (
        <div className="h-full flex flex-col p-6 bg-black/40 rounded-xl border-2 border-orange-500/30 text-orange-100 font-mono relative overflow-hidden shadow-[0_0_50px_rgba(249,115,22,0.15)]">
            <div className="absolute inset-0 border-[4px] border-black pointer-events-none z-50"></div>
            <div className="absolute inset-0 bg-gradient-to-br from-orange-900/20 via-transparent to-purple-900/20 pointer-events-none animate-pulse" style={{ animationDuration: '8s' }}></div>
            
            <div className="flex items-center justify-between mb-6 z-10">
                <h2 className="text-2xl font-black uppercase tracking-[0.3em] text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-purple-500 flex items-center gap-3">
                    NEURAL - DASH
                </h2>
                <button 
                    onClick={() => setView(view === 'main' ? 'translator' : 'main')}
                    className="px-4 py-2 bg-orange-600/30 border border-orange-500 rounded font-bold text-xs uppercase hover:bg-orange-600/50 transition-all"
                >
                    {view === 'main' ? 'Go to Translator' : 'Back to Main'}
                </button>
            </div>
            
            <div className="flex-grow z-10 overflow-auto">
                {view === 'main' ? (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <NeuralProgrammingPanel />
                        <PhiSyncWidget />
                        <div className={devFullscreen ? 'fixed inset-4 z-[100] flex flex-col bg-black/95 border-2 border-orange-500/50 rounded-xl p-4 shadow-2xl' : 'lg:col-span-2 bg-black/50 border border-orange-900/50 rounded-xl p-4 flex flex-col min-h-[600px]'}>
                            <div className="flex justify-between items-center mb-4 flex-shrink-0">
                                <h3 className="text-orange-400 font-bold uppercase tracking-widest flex items-center gap-2">
                                    CHIPS Dev Platform
                                </h3>
                                <button
                                    onClick={() => setDevFullscreen(!devFullscreen)}
                                    className="text-orange-500 hover:text-orange-300 transition-colors px-3 py-1 border border-orange-500/30 rounded text-xs font-bold uppercase"
                                >
                                    {devFullscreen ? 'Collapse' : 'Full Screen'}
                                </button>
                            </div>
                            <div className="flex-grow overflow-hidden relative">
                                <ChipsDevPlatform />
                            </div>
                        </div>
                    </div>
                ) : (
                    <NeuralSignalTranslator />
                )}
            </div>
        </div>
    );
};

export default NeuralDash;
