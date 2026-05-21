
import React from 'react';
import SecurityMonitorAndSimulator from './SecurityMonitorAndSimulator';
import QPUHealth from './QPUHealth';
import SystemDiagnostic from './SystemDiagnostic';
import QuantumDataIngestion from './QuantumDataIngestion';
import { useSimulation } from '../context/SimulationContext';
import { ShieldCheckIcon, CpuChipIcon, ServerCogIcon, DatabaseIcon } from './Icons';

const UtilityHubPanel: React.FC<{ onMaximizeSubPanel?: (id: string) => void }> = ({ onMaximizeSubPanel }) => {
    const { systemStatus } = useSimulation();

    return (
        <div className="h-full flex flex-col gap-2 p-2 overflow-hidden bg-black/20">
            {/* Header Telemetry Bar */}
            <div className="grid grid-cols-4 gap-2 mb-1 flex-shrink-0">
                <div className="bg-black/40 border border-green-500/20 p-2 rounded flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <ShieldCheckIcon className="w-3.5 h-3.5 text-green-400" />
                        <span className="text-[9px] font-bold text-gray-400 uppercase hidden sm:inline">Sec-Shield</span>
                    </div>
                    <span className="text-[10px] font-mono text-green-300">ACTIVE</span>
                </div>
                <div className="bg-black/40 border border-cyan-500/20 p-2 rounded flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <CpuChipIcon className="w-3.5 h-3.5 text-cyan-400" />
                        <span className="text-[9px] font-bold text-gray-400 uppercase hidden sm:inline">Q-Stability</span>
                    </div>
                    <span className="text-[10px] font-mono text-cyan-300">99.9%</span>
                </div>
                <div className="bg-black/40 border border-yellow-500/20 p-2 rounded flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <ServerCogIcon className="w-3.5 h-3.5 text-yellow-400" />
                        <span className="text-[9px] font-bold text-gray-400 uppercase hidden sm:inline">I/O Health</span>
                    </div>
                    <span className="text-[10px] font-mono text-yellow-300">NOMINAL</span>
                </div>
                <div className="bg-black/40 border border-blue-500/20 p-2 rounded flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <DatabaseIcon className="w-3.5 h-3.5 text-blue-400" />
                        <span className="text-[9px] font-bold text-gray-400 uppercase hidden sm:inline">Ingest</span>
                    </div>
                    <span className="text-[10px] font-mono text-blue-300">{((systemStatus?.ipsThroughput || 0) / 10).toFixed(0)} PB/s</span>
                </div>
            </div>

            {/* Main Content Grid - 2x2 Layout for density */}
            <div className="flex-grow grid grid-cols-1 md:grid-cols-2 grid-rows-2 gap-2 min-h-0">
                
                {/* Top Left: Security */}
                <div className="row-span-1 md:row-span-1 overflow-hidden rounded-lg bg-black/20 border border-cyan-900/30">
                    <SecurityMonitorAndSimulator onMaximize={() => onMaximizeSubPanel?.('security-monitor')} />
                </div>
                
                {/* Top Right: QPU Health */}
                <div className="row-span-1 md:row-span-1 overflow-hidden rounded-lg bg-black/20 border border-cyan-900/30">
                    <QPUHealth 
                        systemHealth={{ ...systemStatus, cognitiveEfficiency: 0.99, semanticIntegrity: 0.99, dataThroughput: 100, ipsThroughput: 100, powerEfficiency: 1, decoherenceFactor: 0.001, processingSpeed: 10, qpuTempEfficiency: 1, qubitStability: 200, neuralLoad: 35, activeThreads: 128 }} 
                        onMaximize={() => onMaximizeSubPanel?.('qpu-health')}
                    />
                </div>

                {/* Bottom Left: Data Ingestion (Real-time Streams) */}
                <div className="row-span-1 md:row-span-1 overflow-hidden rounded-lg bg-black/20 border border-cyan-900/30 relative">
                    <QuantumDataIngestion onMaximize={() => onMaximizeSubPanel?.('data-ingestion')} />
                </div>

                {/* Bottom Right: Diagnostics */}
                <div className="row-span-1 md:row-span-1 overflow-hidden rounded-lg bg-black/20 border border-cyan-900/30">
                    <SystemDiagnostic 
                        onClose={() => onMaximizeSubPanel?.(null as any)}
                        onMaximize={() => onMaximizeSubPanel?.('system-diagnostic')}
                    />
                </div>
            </div>
            
            <p className="text-center text-[8px] font-mono text-cyan-800 uppercase tracking-[0.4em] opacity-40 mt-1 flex-shrink-0">
                Utility Service Mesh // Vitals Control v4.2
            </p>
        </div>
    );
};

export default UtilityHubPanel;
