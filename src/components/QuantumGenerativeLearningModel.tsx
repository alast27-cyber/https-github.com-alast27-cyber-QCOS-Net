import React, { useState, useEffect } from 'react';
import { 
    SparklesIcon, PlayIcon, StopIcon, 
    LayersIcon, ActivityIcon 
} from './Icons';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import GlassPanel from './GlassPanel';

const QuantumGenerativeLearningModel: React.FC = () => {
    const [isActive, setIsActive] = useState(false);
    const [creativity, setCreativity] = useState(0.7);
    const [metrics, setMetrics] = useState<{step: number, gLoss: number, dLoss: number}[]>([]);
    const [generatedArtifacts, setGeneratedArtifacts] = useState<string[]>([]);
    
    // Simulation Loop
    useEffect(() => {
        let interval: ReturnType<typeof setInterval>;
        if (isActive) {
            interval = setInterval(() => {
                setMetrics(prev => {
                    const step = (prev.length > 0 ? prev[prev.length - 1].step : 0) + 1;
                    const gLoss = Math.max(0.1, Math.random() * 2 * creativity);
                    const dLoss = Math.max(0.1, Math.random());
                    return [...prev, { step, gLoss, dLoss }].slice(-50);
                });

                // Generate artifacts occasionally
                if (Math.random() > 0.8) {
                    const colors = ['#a855f7', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#8b5cf6'];
                    const newArtifact = colors[Math.floor(Math.random() * colors.length)];
                    setGeneratedArtifacts(prev => [newArtifact, ...prev].slice(0, 9));
                }
            }, 500);
        }
        return () => clearInterval(interval);
    }, [isActive, creativity]);

    return (
        <GlassPanel title={
            <div className="flex items-center gap-2">
                <SparklesIcon className="w-5 h-5 text-pink-400" />
                <span>Quantum Generative Learning (QGLM)</span>
            </div>
        }>
            <div className="flex flex-col h-full gap-4 p-4">
                {/* Controls */}
                <div className="flex justify-between items-center bg-black/30 p-3 rounded-lg border border-pink-900/50">
                    <div className="flex gap-4 items-center">
                        <div className="flex flex-col">
                            <span className="text-[10px] text-gray-400 uppercase">Creativity (Temp)</span>
                            <input 
                                type="range" 
                                min="0" max="1" step="0.1" 
                                value={creativity} 
                                onChange={(e) => setCreativity(parseFloat(e.target.value))}
                                className="w-24 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                            />
                        </div>
                        <span className="font-mono text-pink-300 font-bold">{creativity.toFixed(1)}</span>
                    </div>
                    <button 
                        onClick={() => setIsActive(!isActive)}
                        className={`px-4 py-1.5 text-xs font-bold rounded flex items-center gap-2 transition-all ${isActive ? 'bg-red-600/20 border border-red-500 text-red-300' : 'bg-green-600/20 border border-green-500 text-green-200'}`}
                    >
                        {isActive ? <StopIcon className="w-3 h-3" /> : <PlayIcon className="w-3 h-3" />}
                        {isActive ? 'Halt Generation' : 'Start Synthesis'}
                    </button>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full min-h-0">
                    {/* Metrics Chart */}
                    <div className="bg-black/30 border border-pink-800/30 rounded-lg p-4 flex flex-col">
                        <h4 className="text-pink-300 font-bold text-xs uppercase mb-2 flex items-center gap-2">
                            <ActivityIcon className="w-4 h-4" /> Adversarial Loss
                        </h4>
                        <div className="flex-grow min-h-[150px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={metrics}>
                                    <defs>
                                        <linearGradient id="colorGLoss" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#ec4899" stopOpacity={0.8}/>
                                            <stop offset="95%" stopColor="#ec4899" stopOpacity={0}/>
                                        </linearGradient>
                                        <linearGradient id="colorDLoss" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.8}/>
                                            <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(236, 72, 153, 0.1)" vertical={false} />
                                    <XAxis dataKey="step" hide />
                                    <YAxis hide />
                                    <Tooltip contentStyle={{backgroundColor: 'rgba(0,0,0,0.9)', borderColor: '#ec4899', color: '#fff'}} itemStyle={{fontSize: '10px'}} />
                                    <Area type="monotone" dataKey="gLoss" stroke="#ec4899" fillOpacity={1} fill="url(#colorGLoss)" />
                                    <Area type="monotone" dataKey="dLoss" stroke="#06b6d4" fillOpacity={1} fill="url(#colorDLoss)" />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Generated Artifacts */}
                    <div className="bg-black/30 border border-pink-800/30 rounded-lg p-4 flex flex-col">
                        <h4 className="text-pink-300 font-bold text-xs uppercase mb-2 flex items-center gap-2">
                            <LayersIcon className="w-4 h-4" /> Latent Space Artifacts
                        </h4>
                        <div className="grid grid-cols-3 gap-2 flex-grow">
                            {generatedArtifacts.map((color, i) => (
                                <div key={i} className="rounded border border-white/10 animate-fade-in relative overflow-hidden group aspect-square">
                                    <div className="absolute inset-0 opacity-50" style={{ backgroundColor: color }}></div>
                                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent"></div>
                                    <div className="absolute bottom-1 left-1 text-[8px] font-mono text-white opacity-0 group-hover:opacity-100 transition-opacity">
                                        GEN-{Math.floor(Math.random() * 1000)}
                                    </div>
                                </div>
                            ))}
                            {generatedArtifacts.length === 0 && (
                                <div className="col-span-3 flex items-center justify-center text-gray-600 text-xs italic">
                                    Initialize synthesis...
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </GlassPanel>
    );
};

export default QuantumGenerativeLearningModel;
