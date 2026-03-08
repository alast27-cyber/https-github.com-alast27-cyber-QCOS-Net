import React from 'react';
import { GalaxyIcon } from './Icons';
import GrandUniverseSimulator from './GrandUniverseSimulator';
import QuantumLargeLanguageModel from './QuantumLargeLanguageModel';
import QuantumMachineLearning from './QuantumMachineLearning';
import QuantumReinforcementLearning from './QuantumReinforcementLearning';
import QuantumDeepLearning from './QuantumDeepLearning';
import QuantumGenerativeLearningModel from './QuantumGenerativeLearningModel';

const GusDash: React.FC = () => {
    return (
        <div className="h-full flex flex-col p-6 bg-black/40 rounded-xl border-2 border-purple-500/30 text-purple-100 font-mono relative overflow-hidden shadow-[0_0_50px_rgba(168,85,247,0.15)]">
            {/* Security & Containment Null-Field Border */}
            <div className="absolute inset-0 border-[4px] border-black pointer-events-none z-50"></div>
            
            {/* Ambient Lighting Shift */}
            <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 via-transparent to-cyan-900/20 pointer-events-none animate-pulse" style={{ animationDuration: '8s' }}></div>

            <div className="flex justify-between items-start z-10 mb-6">
                <div>
                    <h2 className="text-2xl font-black uppercase tracking-[0.3em] text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-cyan-500 flex items-center gap-3">
                        <GalaxyIcon className="w-8 h-8 text-purple-400" />
                        GUS - DASH
                    </h2>
                    <div className="flex items-center gap-4 mt-2">
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-purple-500 rounded-full animate-ping"></div>
                            <span className="text-xs text-purple-600 font-bold tracking-widest">GRAND UNIVERSE SIMULATOR</span>
                        </div>
                    </div>
                </div>
            </div>

            <div className="flex-grow z-10 overflow-y-auto relative grid grid-cols-1 xl:grid-cols-2 gap-6">
                <div className="xl:col-span-2 h-[500px]">
                    <GrandUniverseSimulator embedded={true} />
                </div>
                <div className="h-[400px]">
                    <QuantumLargeLanguageModel />
                </div>
                <div className="h-[400px]">
                    <QuantumMachineLearning />
                </div>
                <div className="h-[400px]">
                    <QuantumReinforcementLearning />
                </div>
                <div className="h-[400px]">
                    <QuantumDeepLearning />
                </div>
                <div className="h-[400px]">
                    <QuantumGenerativeLearningModel />
                </div>
            </div>
        </div>
    );
};

export default GusDash;
