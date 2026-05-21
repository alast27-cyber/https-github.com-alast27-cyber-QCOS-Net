import React from 'react';
import GrandUniverseSimulator from './src/components/GrandUniverseSimulator';
import QuantumCognitiveArchitecture from '../src/components/QuantumCognitiveArchitecture';
import { SimulationProvider } from './src/context/SimulationContext';
import { ToastProvider } from './src/context/ToastContext';
import ToastContainer from './src/components/ToastContainer';

const StandaloneContent: React.FC = () => {
    return (
        <div className="w-full h-screen bg-slate-950 flex flex-col overflow-hidden relative">
            <div className="fixed inset-0 pointer-events-none z-0">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(168,85,247,0.08),transparent_70%)]" />
                <div className="absolute top-0 left-0 w-full h-full bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-[0.03] brightness-50 contrast-150" />
            </div>
            
            <div className="flex flex-row flex-grow z-10 relative">
                <div className="w-2/3 h-full p-2 pr-1">
                    <GrandUniverseSimulator embedded={true} qubitCount={1024} />
                </div>
                <div className="w-1/3 h-full p-2 pl-1">
                    <QuantumCognitiveArchitecture />
                </div>
            </div>
            
            <div className="z-20 p-0.5 px-3 bg-black/80 border-t border-purple-900/40 flex justify-between text-[9px] font-mono text-purple-900/70">
                <div className="flex gap-4">
                    <span>NODE: GUS-STANDALONE</span>
                    <span>ENTANGLEMENT: ACTIVE</span>
                    <span>Q-STATE: COHERENT</span>
                </div>
                <div>GUS v2.0.0</div>
            </div>

            <ToastContainer />
        </div>
    );
};

const StandaloneApp: React.FC = () => {
    return (
        <ToastProvider>
            <SimulationProvider>
               <StandaloneContent />
            </SimulationProvider>
        </ToastProvider>
    );
};

export default StandaloneApp;
