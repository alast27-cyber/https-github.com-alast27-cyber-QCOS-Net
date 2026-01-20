
import React from 'react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import GlassPanel from './GlassPanel';
import { Share2Icon, SparklesIcon, CpuChipIcon, AlertTriangleIcon, CheckCircle2Icon, LoaderIcon } from './Icons';
import { SystemHealth } from '../types';
import { useSimulation } from '../context/SimulationContext';

interface QCOSSystemEvolutionInterfaceProps {
    systemHealth: SystemHealth;
}

const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-900/80 p-2 border border-cyan-400 text-white rounded-md text-sm">
          <p className="label">{`Time: ${label}s`}</p>
          <p className="text-purple-400">{`Cognitive Efficiency: ${((payload[0]?.value || 0) * 100).toFixed(2)}%`}</p>
          {payload[1] && <p className="text-green-400">{`Semantic Integrity: ${((payload[1]?.value || 0) * 100).toFixed(2)}%`}</p>}
        </div>
      );
    }
    return null;
  };

const QCOSSystemEvolutionInterface: React.FC<QCOSSystemEvolutionInterfaceProps> = ({ systemHealth }) => {
    const { evolution, toggleEvolution } = useSimulation();

    const getOverallStatus = () => {
        if (!evolution.isActive && evolution.dataPoints.length === 0) return 'Idle';
        if (evolution.isActive) return 'Optimizing...';
        const lastData = evolution.dataPoints[evolution.dataPoints.length - 1];
        if (lastData && lastData.cognitiveEfficiency > 0.95 && lastData.semanticIntegrity > 0.98) {
            return 'Optimal';
        }
        if (lastData && (lastData.cognitiveEfficiency < 0.9 || lastData.semanticIntegrity < 0.95)) {
            return 'Warning';
        }
        return 'Stable';
    };

    const status = getOverallStatus();

    const statusIcon = {
        Idle: <CpuChipIcon className="w-5 h-5 mr-2 text-cyan-400" />,
        'Optimizing...': <LoaderIcon className="w-5 h-5 mr-2 text-yellow-400 animate-spin" />,
        Optimal: <CheckCircle2Icon className="w-5 h-5 mr-2 text-green-400" />,
        Warning: <AlertTriangleIcon className="w-5 h-5 mr-2 text-red-400" />,
        Stable: <SparklesIcon className="w-5 h-5 mr-2 text-blue-400" />,
    }[status];

    return (
        <GlassPanel title="QCOS System Evolution Interface">
            <div className="p-4 space-y-4 text-sm text-cyan-200 h-full flex flex-col">

                {/* Controls */}
                <div className="flex-shrink-0 grid grid-cols-3 gap-2 bg-black/20 p-2 rounded-lg border border-cyan-800/50">
                    <button onClick={toggleEvolution} className={`holographic-button py-2 rounded-md font-bold text-xs flex items-center justify-center ${evolution.isActive ? 'bg-red-600/30 border-red-500/50 text-red-200' : 'bg-cyan-600/30 border-cyan-500/50 text-cyan-200'}`}>
                        {evolution.isActive ? 'Pause Sim' : 'Start Sim'}
                    </button>
                    <button onClick={() => {}} className="holographic-button py-2 rounded-md font-bold text-xs flex items-center justify-center bg-slate-600/30 border-slate-500/50 text-slate-200 opacity-50 cursor-not-allowed">
                        Reset Sim
                    </button>
                    <div className="flex items-center justify-center bg-black/30 border border-cyan-900/50 rounded-md p-1">
                        {statusIcon}
                        <span className="text-cyan-300">{status}</span>
                    </div>
                </div>

                {/* Live Metrics & Chart */}
                <div className="flex-grow grid grid-rows-[1fr_0.5fr] gap-4 min-h-0">
                    {/* Chart */}
                    <div className="bg-black/20 p-2 rounded-lg border border-cyan-800/50 flex flex-col">
                        <h3 className="text-sm font-semibold text-cyan-300 mb-1">QNN Evolution Metrics</h3>
                        <div className="flex-grow">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={evolution.dataPoints} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                                    <CartesianGrid strokeDasharray="1 1" stroke="rgba(0, 255, 255, 0.1)" />
                                    <XAxis dataKey="time" stroke="rgba(0, 255, 255, 0.7)" tick={{ fontSize: 10 }} />
                                    <YAxis stroke="rgba(0, 255, 255, 0.7)" domain={[0.8, 1]} tick={{ fontSize: 10 }} tickFormatter={(val) => `${(val * 100).toFixed(0)}%`} />
                                    <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'cyan', strokeWidth: 1 }} />
                                    <Area type="monotone" dataKey="cognitiveEfficiency" stroke="#a78bfa" fill="#a78bfa" fillOpacity={0.3} name="Cognitive Efficiency" />
                                    <Area type="monotone" dataKey="semanticIntegrity" stroke="#4ade80" fill="#4ade80" fillOpacity={0.3} name="Semantic Integrity" />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Simulation Log */}
                    <div className="bg-black/20 p-2 rounded-lg border border-cyan-800/50 flex flex-col">
                        <h3 className="text-sm font-semibold text-cyan-300 mb-1">Simulation Log</h3>
                        <div className="flex-grow overflow-y-auto pr-2 -mr-2 text-xs font-mono">
                            {evolution.logs.map((log, index) => (
                                <p key={index} className="text-gray-300">{log}</p>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Insights */}
                <div className="flex-shrink-0 bg-black/20 p-3 rounded-lg border border-cyan-800/50">
                    <h3 className="flex items-center text-base font-semibold text-cyan-300 mb-2">
                        <Share2Icon className="w-5 h-5 mr-2" /> Current Insights
                    </h3>
                    <p className="text-xs text-cyan-400">
                        The system is currently operating with a cognitive efficiency of <span className="font-bold text-white">{((systemHealth.cognitiveEfficiency || 0) * 100).toFixed(2)}%</span> and semantic integrity of <span className="font-bold text-white">{((systemHealth.semanticIntegrity || 0) * 100).toFixed(2)}%</span>. Running simulations helps predict optimal QNN tuning parameters.
                    </p>
                </div>
            </div>
        </GlassPanel>
    );
};

export default QCOSSystemEvolutionInterface;
