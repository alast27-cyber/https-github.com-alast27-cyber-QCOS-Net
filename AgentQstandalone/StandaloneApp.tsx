import React from 'react';
import AgentQ from './ai-core/AgentQ';
import { useAgentQ } from './ai-core/useAgentQ';
import { SimulationProvider } from './shared/context/SimulationContext';
import { ToastProvider } from './shared/context/ToastContext';
import { AuthProvider } from './shared/context/AuthContext';
import QAPI from './QAPI';
import { useEffect } from 'react';

/**
 * StandaloneApp: The isolated UI entry point for AgentQ.
 * 
 * This component wraps AgentQ in the minimal necessary context to function
 * as a standalone application while maintaining quantum entanglement with
 * the QCOS ecosystem.
 */
const StandaloneApp: React.FC = () => {
    const { agentQProps } = useAgentQ({
        focusedPanelId: 'standalone-quantum-core',
        panelInfoMap: { 
            'standalone-quantum-core': { 
                title: 'STANDALONE AGENTQ CORE', 
                description: 'AgentQ operating in a dedicated high-dimensional manifold.' 
            } 
        },
        qcosVersion: 4.5,
        systemHealth: {
            neuralLoad: 12,
            activeThreads: 8,
            dataThroughput: 1500,
            ipsThroughput: 2400,
            cognitiveEfficiency: 0.999,
            decoherenceFactor: 0.001,
            semanticIntegrity: 1.0,
            powerEfficiency: 0.98,
            processingSpeed: 0.95,
            qpuTempEfficiency: 0.92,
            qubitStability: 0.99
        },
        onDashboardControl: (action) => console.log("[STANDALONE] Control Command Received:", action)
    });

    useEffect(() => {
        QAPI.entangle('standalone-agentq-interface');
        QAPI.resolveDecoherence();
    }, []);

    return (
        <AuthProvider>
            <SimulationProvider>
                <ToastProvider>
                    <div className="w-full h-screen bg-slate-950 flex items-center justify-center overflow-hidden">
                        {/* Background Effect */}
                        <div className="fixed inset-0 pointer-events-none">
                            <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(6,182,212,0.1),transparent_70%)]" />
                            <div className="absolute top-0 left-0 w-full h-full bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 brightness-50" />
                        </div>
                        
                        <div className="w-full h-full max-w-5xl max-h-[90vh] shadow-[0_0_100px_rgba(6,182,212,0.2)] rounded-3xl overflow-hidden border border-cyan-500/20">
                            <AgentQ 
                                {...agentQProps}
                                isOpen={true}
                                onToggleOpen={() => {}} // Standalone is persistent
                                fullScreen={false}
                                embedded={true}
                            />
                        </div>
                    </div>
                </ToastProvider>
            </SimulationProvider>
        </AuthProvider>
    );
};

export default StandaloneApp;
