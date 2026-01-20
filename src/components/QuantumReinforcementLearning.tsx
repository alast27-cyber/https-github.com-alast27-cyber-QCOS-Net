
import React, { useState } from 'react';
import GlassPanel from './GlassPanel';
import { RocketLaunchIcon, ActivityIcon, PlayIcon, StopIcon } from './Icons';
import { useSimulation } from '../context/SimulationContext';
import { ResponsiveContainer, AreaChart, Area, CartesianGrid, XAxis, YAxis, Tooltip } from 'recharts';

const QuantumReinforcementLearning: React.FC = () => {
    const { qrlEngine, startQRLTraining, stopQRLTraining } = useSimulation();

    return (
        <GlassPanel title={<div className="flex items-center"><RocketLaunchIcon className="w-5 h-5 mr-2 text-orange-400" /> Quantum Reinforcement Learning</div>}>
            <div className="flex flex-col h-full p-4 gap-4">
                <div className="flex justify-between items-center bg-black/30 p-3 rounded-lg border border-orange-900/50">
                    <div>
                        <p className="text-xs text-orange-500 font-bold uppercase">Reward Signal</p>
                        <p className="text-2xl font-mono text-white">{(qrlEngine.avgReward || 0).toFixed(3)}</p>
                    </div>
                    <div>
                         <p className="text-xs text-cyan-500 font-bold uppercase text-right">Epsilon (Exploration)</p>
                         <p className="text-2xl font-mono text-cyan-300 text-right">{(qrlEngine.epsilon || 0).toFixed(3)}</p>
                    </div>
                </div>

                <div className="flex-grow bg-black/40 border border-orange-900/30 rounded-lg overflow-hidden flex flex-col">
                     <div className="p-2 border-b border-orange-900/30 text-[10px] text-orange-400 font-bold uppercase flex justify-between">
                         <span>Training Episode: {qrlEngine.currentEpisode}</span>
                         <span>{qrlEngine.status}</span>
                     </div>
                     <div className="flex-grow relative">
                         <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={qrlEngine.cumulativeRewards.slice(-50)}>
                                <defs>
                                    <linearGradient id="colorReward" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#f97316" stopOpacity={0.8}/>
                                        <stop offset="95%" stopColor="#f97316" stopOpacity={0}/>
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="episode" hide />
                                <YAxis hide />
                                <Tooltip contentStyle={{backgroundColor: '#000', borderColor: '#f97316', fontSize: '10px'}} itemStyle={{color: '#fff'}} />
                                <Area type="monotone" dataKey="value" stroke="#f97316" fillOpacity={1} fill="url(#colorReward)" />
                            </AreaChart>
                         </ResponsiveContainer>
                     </div>
                </div>

                <div className="flex gap-2">
                    {qrlEngine.status === 'IDLE' ? (
                        <button onClick={() => startQRLTraining()} className="flex-1 holographic-button py-2 bg-green-600/20 border-green-500 text-green-200 text-xs font-bold rounded flex items-center justify-center gap-2">
                            <PlayIcon className="w-3 h-3" /> Start QRL Agent
                        </button>
                    ) : (
                         <button onClick={() => stopQRLTraining()} className="flex-1 holographic-button py-2 bg-red-600/20 border-red-500 text-red-200 text-xs font-bold rounded flex items-center justify-center gap-2">
                            <StopIcon className="w-3 h-3" /> Stop Training
                        </button>
                    )}
                </div>
            </div>
        </GlassPanel>
    );
};

export default QuantumReinforcementLearning;
