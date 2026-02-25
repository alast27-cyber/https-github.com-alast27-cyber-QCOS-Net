
import React, { useState, useEffect } from 'react';
import GlassPanel from './GlassPanel';
import { 
    DatabaseIcon, ActivityIcon, LinkIcon, GlobeIcon, 
    ServerStackIcon, RefreshCwIcon, PlayIcon, StopIcon,
    ZapIcon, CheckCircle2Icon, Share2Icon, ArrowRightIcon
} from './Icons';
import { useSimulation, DataSource } from '../context/SimulationContext';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

const SourceRow: React.FC<{ source: DataSource, onToggle: (id: string) => void }> = ({ source, onToggle }) => {
    return (
        <div className={`p-2 rounded border mb-2 transition-all flex items-center justify-between group ${source.isEntangled ? 'bg-purple-900/30 border-purple-500 shadow-[0_0_10px_rgba(168,85,247,0.2)]' : 'bg-black/30 border-cyan-900/30'}`}>
            <div className="flex items-center gap-3 overflow-hidden">
                <div className={`p-1.5 rounded-full ${source.isEntangled ? 'bg-purple-600 text-white animate-pulse' : 'bg-gray-800 text-gray-500'}`}>
                    <LinkIcon className="w-3 h-3" />
                </div>
                <div className="min-w-0">
                    <p className={`text-[10px] font-bold truncate ${source.isEntangled ? 'text-white' : 'text-cyan-200'}`}>{source.name}</p>
                    <div className="flex items-center gap-2 text-[8px] font-mono text-gray-400">
                        <span>{(source.throughput || 0).toFixed(1)} PB/s</span>
                        <span className="w-px h-2 bg-gray-700"></span>
                        <span className={source.fidelity > 99 ? 'text-green-400' : 'text-yellow-400'}>{(source.fidelity || 0).toFixed(2)}% Fid</span>
                        {source.isEntangled && <span className="text-purple-300 ml-1">↔ SIM_LINKED</span>}
                    </div>
                </div>
            </div>
            
            <button 
                onClick={() => onToggle(source.id)}
                className={`p-1.5 rounded transition-all ${source.isEntangled ? 'text-purple-300 hover:text-white bg-purple-500/20' : 'text-gray-600 hover:text-cyan-400 hover:bg-white/5'}`}
                title={source.isEntangled ? "Decouple Entanglement" : "Entangle Stream"}
            >
                <ZapIcon className={`w-4 h-4 ${source.isEntangled ? 'fill-current animate-pulse' : ''}`} />
            </button>
        </div>
    );
};

const QuantumDataIngestion: React.FC<{ onMaximize?: () => void }> = ({ onMaximize }) => {
    const { dataIngestion, toggleSourceEntanglement } = useSimulation();
    const [chartHistory, setChartHistory] = useState<{time: number, val: number}[]>([]);
    const [tick, setTick] = useState(0);

    // Calculate aggregate metrics
    const totalThroughput = dataIngestion.reduce((acc, s) => acc + (s.status === 'ACTIVE' ? (s.throughput || 0) : 0), 0);
    const avgFidelity = dataIngestion.reduce((acc, s) => acc + (s.fidelity || 0), 0) / (dataIngestion.length || 1);
    const entangledStreams = dataIngestion.filter(s => s.isEntangled).length;

    // Update chart history
    useEffect(() => {
        setTick(t => t + 1);
        setChartHistory(prev => {
            const newData = [...prev, { time: tick, val: totalThroughput }];
            if (newData.length > 30) newData.shift();
            return newData;
        });
    }, [totalThroughput]); 

    return (
        <GlassPanel 
            onMaximize={onMaximize}
            title={
                <div className="flex items-center justify-between w-full">
                    <div className="flex items-center">
                        <DatabaseIcon className="w-5 h-5 mr-2 text-blue-400" />
                        <span>Live Data Ingestion</span>
                    </div>
                    <div className="flex items-center gap-2 text-[9px] font-mono bg-black/40 px-2 py-0.5 rounded border border-cyan-900 mr-6">
                        <span className="text-cyan-500">NET:</span>
                        <span className="text-white font-bold">{(totalThroughput / 1000).toFixed(2)} EB/s</span>
                    </div>
                </div>
            }
        >
            <div className="flex flex-col h-full gap-2 p-2 overflow-hidden">
                
                {/* Seamless Uplink Indicator */}
                <div className="flex justify-between items-center bg-purple-900/20 border border-purple-500/30 p-2 rounded mb-1 animate-fade-in">
                    <div className="flex items-center gap-2">
                         <Share2Icon className="w-4 h-4 text-purple-400 animate-pulse" />
                         <div>
                             <p className="text-[9px] font-bold text-white uppercase tracking-widest">Quantum-Entangled Uplink: ACTIVE</p>
                             <p className="text-[8px] text-purple-300">Feeding Grand Universe Simulator & QML Engine</p>
                         </div>
                    </div>
                    <button 
                        onClick={() => toggleSourceEntanglement('ALL')}
                        className="holographic-button px-2 py-1 text-[8px] font-bold bg-purple-600/30 border border-purple-500 text-white rounded flex items-center gap-1 hover:bg-purple-600/50 transition-all"
                    >
                        <RefreshCwIcon className="w-3 h-3" /> Force Sync All
                    </button>
                </div>

                {/* Top: Chart & Buffer Visual */}
                <div className="flex h-24 gap-2 flex-shrink-0">
                    <div className="w-2/3 bg-black/20 rounded border border-cyan-900/30 relative flex flex-col overflow-hidden">
                        <div className="absolute top-1 left-2 text-[8px] text-cyan-600 font-bold uppercase tracking-widest z-10">Throughput Vector</div>
                        <div className="flex-grow w-full h-full">
                             <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={chartHistory}>
                                    <defs>
                                        <linearGradient id="colorThroughput" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                                    <XAxis dataKey="time" hide />
                                    <YAxis hide domain={['auto', 'auto']} />
                                    <Area type="monotone" dataKey="val" stroke="#3b82f6" strokeWidth={2} fill="url(#colorThroughput)" isAnimationActive={true} animationDuration={300} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    <div className="w-1/3 bg-black/20 rounded border border-purple-900/30 flex flex-col items-center justify-center relative overflow-hidden">
                        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-purple-900/20 via-transparent to-transparent"></div>
                        <div className="relative z-10 text-center">
                            <div className="w-10 h-10 rounded-full border-2 border-purple-500/50 flex items-center justify-center mx-auto mb-1 shadow-[0_0_15px_rgba(168,85,247,0.3)] animate-pulse-slow">
                                <div className="w-6 h-6 rounded-full bg-purple-500/20 flex items-center justify-center animate-spin-reverse-slow">
                                    <LinkIcon className="w-3 h-3 text-purple-300" />
                                </div>
                            </div>
                            <p className="text-[8px] text-purple-400 uppercase font-bold">Quantum Buffer</p>
                            <p className="text-[10px] text-white font-mono">{(avgFidelity || 0).toFixed(2)}%</p>
                        </div>
                    </div>
                </div>

                {/* Bottom: Source List */}
                <div className="flex-grow overflow-y-auto custom-scrollbar pr-1 min-h-0">
                    <p className="text-[9px] text-gray-500 uppercase font-bold mb-2 sticky top-0 bg-black/80 p-1 z-10 backdrop-blur-sm flex justify-between border-b border-white/5">
                        <span>Entangled Streams ({entangledStreams}/{dataIngestion.length})</span>
                        <span className="text-purple-400 flex items-center gap-1"><ArrowRightIcon className="w-3 h-3"/> SIMULATOR</span>
                    </p>
                    <div className="space-y-1">
                        {dataIngestion.map(source => (
                            <SourceRow key={source.id} source={source} onToggle={toggleSourceEntanglement} />
                        ))}
                    </div>
                </div>

            </div>
        </GlassPanel>
    );
};

export default QuantumDataIngestion;
