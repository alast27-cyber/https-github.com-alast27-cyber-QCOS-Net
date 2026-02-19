
import React, { useState, useEffect } from 'react';
import { CpuChipIcon, ShieldCheckIcon, SparklesIcon } from './Icons';

const ChimeraCoreStatus: React.FC = () => {
    const [metrics, setMetrics] = useState({ load: 35, threats: 0, optimizations: 3 });

    useEffect(() => {
        const interval = setInterval(() => {
            setMetrics({
                load: 20 + Math.random() * 30,
                threats: Math.random() > 0.95 ? 1 : 0,
                optimizations: 3 + Math.floor(Math.random() * 2)
            });
        }, 3000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="p-3 bg-slate-900/90 border border-cyan-700 rounded-lg shadow-lg w-64 space-y-3 font-mono text-xs backdrop-blur-sm">
            <div className="flex items-center gap-2">
                <CpuChipIcon className="w-5 h-5 text-cyan-300 animate-pulse"/>
                <h4 className="font-bold text-cyan-200">Chimera AI Core Status</h4>
            </div>
            <div>
                <p className="flex justify-between text-cyan-400"><span>Cognitive Load</span> <span className="text-white">{(metrics.load || 0).toFixed(1)}%</span></p>
                <div className="w-full h-1 bg-cyan-900/50 rounded-full mt-1"><div className="h-1 bg-cyan-400 rounded-full" style={{width: `${metrics.load}%`}}></div></div>
            </div>
            <div>
                <p className="flex justify-between items-center text-cyan-400">
                    <span>Threat Analysis</span> 
                    <span className={metrics.threats > 0 ? 'text-red-400 font-bold' : 'text-green-400'}>{metrics.threats > 0 ? 'ANOMALY' : 'NOMINAL'}</span>
                </p>
                <div className="w-full h-1 bg-cyan-900/50 rounded-full mt-1"><div className={`h-1 rounded-full ${metrics.threats > 0 ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`} style={{width: `100%`}}></div></div>
            </div>
            <div>
                <p className="flex justify-between text-cyan-400"><span>Active Optimizations</span> <span className="text-white">{metrics.optimizations}</span></p>
                 <div className="w-full h-1 bg-cyan-900/50 rounded-full mt-1"><div className="h-1 bg-purple-400 rounded-full" style={{width: `${(metrics.optimizations / 5) * 100}%`}}></div></div>
            </div>
        </div>
    );
};

export default ChimeraCoreStatus;
