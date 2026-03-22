import React from 'react';
import NeuralProgrammingPanel from './NeuralProgrammingPanel';
import PhiSyncWidget from './PhiSyncWidget';

const NeuralDash: React.FC = () => {
    return (
        <div className="h-full flex flex-col p-6 bg-black/40 rounded-xl border-2 border-orange-500/30 text-orange-100 font-mono relative overflow-hidden shadow-[0_0_50px_rgba(249,115,22,0.15)]">
            <div className="absolute inset-0 border-[4px] border-black pointer-events-none z-50"></div>
            <div className="absolute inset-0 bg-gradient-to-br from-orange-900/20 via-transparent to-purple-900/20 pointer-events-none animate-pulse" style={{ animationDuration: '8s' }}></div>
            
            <h2 className="text-2xl font-black uppercase tracking-[0.3em] text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-purple-500 flex items-center gap-3 mb-6 z-10">
                NEURAL - DASH
            </h2>
            
            <div className="flex-grow z-10 overflow-auto grid grid-cols-1 lg:grid-cols-2 gap-6">
                <NeuralProgrammingPanel />
                <PhiSyncWidget />
            </div>
        </div>
    );
};

export default NeuralDash;
