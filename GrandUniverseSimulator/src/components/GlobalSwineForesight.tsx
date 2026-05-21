
import React, { useState } from 'react';
import GlassPanel from './GlassPanel';
import { GlobeIcon, ArrowTrendingUpIcon, ActivityIcon, AlertTriangleIcon, MapPinIcon } from './Icons';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid, BarChart, Bar, Cell } from 'recharts';

const commodityData = [
  { month: 'Jan', feed: 120, pork: 180 },
  { month: 'Feb', feed: 125, pork: 182 },
  { month: 'Mar', feed: 130, pork: 195 },
  { month: 'Apr', feed: 128, pork: 200 },
  { month: 'May', feed: 135, pork: 210 },
  { month: 'Jun', feed: 140, pork: 205 },
];

const riskRegions = [
  { region: 'East Asia', risk: 85, status: 'Critical' },
  { region: 'Eastern Europe', risk: 65, status: 'High' },
  { region: 'Americas', risk: 25, status: 'Low' },
  { region: 'Southeast Asia', risk: 78, status: 'High' },
];

const GlobalSwineForesight: React.FC = () => {
    const [activeView, setActiveView] = useState<'market' | 'bio'>('market');

    return (
        <GlassPanel title={<div className="flex items-center"><GlobeIcon className="w-5 h-5 mr-2 text-pink-400" /> Global Swine Foresight</div>}>
            <div className="flex flex-col h-full p-4 gap-4">
                
                {/* Navigation */}
                <div className="flex space-x-2 border-b border-pink-900/50 pb-2">
                    <button 
                        onClick={() => setActiveView('market')} 
                        className={`px-4 py-2 rounded text-xs font-bold transition-all ${activeView === 'market' ? 'bg-pink-900/40 text-pink-300 border border-pink-500/50' : 'text-gray-400 hover:text-white'}`}
                    >
                        Commodity Markets
                    </button>
                    <button 
                        onClick={() => setActiveView('bio')} 
                        className={`px-4 py-2 rounded text-xs font-bold transition-all ${activeView === 'bio' ? 'bg-red-900/40 text-red-300 border border-red-500/50' : 'text-gray-400 hover:text-white'}`}
                    >
                        Biosecurity Radar
                    </button>
                </div>

                {activeView === 'market' ? (
                    <div className="flex-grow flex flex-col gap-4 animate-fade-in">
                        <div className="grid grid-cols-3 gap-4">
                            <div className="bg-black/30 p-3 rounded border border-pink-900/30">
                                <p className="text-[10px] text-pink-400 uppercase">Global Lean Hog Index</p>
                                <p className="text-xl font-mono text-white">$92.40</p>
                                <span className="text-[9px] text-green-400">+1.2% (24h)</span>
                            </div>
                            <div className="bg-black/30 p-3 rounded border border-pink-900/30">
                                <p className="text-[10px] text-pink-400 uppercase">Soybean Meal Futures</p>
                                <p className="text-xl font-mono text-white">$340.5</p>
                                <span className="text-[9px] text-red-400">-0.5% (24h)</span>
                            </div>
                            <div className="bg-black/30 p-3 rounded border border-pink-900/30">
                                <p className="text-[10px] text-pink-400 uppercase">Q-Forecast (Q3)</p>
                                <p className="text-xl font-mono text-cyan-300">Bullish</p>
                                <span className="text-[9px] text-cyan-500">Confidence: 94%</span>
                            </div>
                        </div>

                        <div className="flex-grow bg-black/20 border border-pink-900/30 rounded-lg p-2 min-h-[200px]">
                            <h4 className="text-xs font-bold text-pink-300 mb-2 px-2 flex items-center"><ArrowTrendingUpIcon className="w-4 h-4 mr-2"/> Price Correlation Analysis</h4>
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={commodityData}>
                                    <defs>
                                        <linearGradient id="colorPork" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#f472b6" stopOpacity={0.3}/>
                                            <stop offset="95%" stopColor="#f472b6" stopOpacity={0}/>
                                        </linearGradient>
                                        <linearGradient id="colorFeed" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#fbbf24" stopOpacity={0.3}/>
                                            <stop offset="95%" stopColor="#fbbf24" stopOpacity={0}/>
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                    <XAxis dataKey="month" stroke="#666" fontSize={10} />
                                    <YAxis stroke="#666" fontSize={10} />
                                    <Tooltip contentStyle={{backgroundColor: '#000', borderColor: '#f472b6'}} itemStyle={{fontSize:'10px'}} />
                                    <Area type="monotone" dataKey="pork" stroke="#f472b6" fillOpacity={1} fill="url(#colorPork)" name="Pork Index" />
                                    <Area type="monotone" dataKey="feed" stroke="#fbbf24" fillOpacity={1} fill="url(#colorFeed)" name="Feed Cost" />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                ) : (
                    <div className="flex-grow flex flex-col gap-4 animate-fade-in">
                        <div className="bg-red-900/20 border border-red-500/30 p-3 rounded-lg flex items-center gap-3">
                            <AlertTriangleIcon className="w-8 h-8 text-red-500 animate-pulse" />
                            <div>
                                <h4 className="text-sm font-bold text-white">Pandemic Watch: Strain Q-H5</h4>
                                <p className="text-xs text-red-200">Quantum sensors detected elevated viral RNA traces in regional logistics hubs (Region IV-A). Recommended lockdown of Sector 7.</p>
                            </div>
                        </div>

                        <div className="flex-grow grid grid-cols-2 gap-4">
                            <div className="bg-black/20 border border-red-900/30 rounded-lg p-2 flex flex-col">
                                <h4 className="text-xs font-bold text-red-400 mb-2 px-1">Regional Risk Levels</h4>
                                <div className="flex-grow">
                                     <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={riskRegions} layout="vertical">
                                            <XAxis type="number" hide />
                                            <YAxis dataKey="region" type="category" width={90} tick={{fontSize: 9, fill: '#aaa'}} />
                                            <Tooltip cursor={{fill: 'transparent'}} contentStyle={{backgroundColor: '#000', borderColor: '#ef4444', fontSize:'10px'}} />
                                            <Bar dataKey="risk" radius={[0, 4, 4, 0]}>
                                                {riskRegions.map((entry, index) => (
                                                    <Cell key={`cell-${index}`} fill={entry.risk > 80 ? '#ef4444' : entry.risk > 50 ? '#f59e0b' : '#22c55e'} />
                                                ))}
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                            <div className="bg-black/20 border border-red-900/30 rounded-lg p-3 overflow-y-auto custom-scrollbar">
                                <h4 className="text-xs font-bold text-red-400 mb-2">Live Biosecurity Alerts</h4>
                                <ul className="space-y-2">
                                    {[
                                        { time: '10:42 AM', msg: 'Anomalous transport vector identified in Batangas.' },
                                        { time: '09:15 AM', msg: 'Feed shipment flagged for contamination check.' },
                                        { time: 'Yesterday', msg: 'Vaccine efficacy simulation updated: 94% success.' },
                                        { time: 'Yesterday', msg: 'New ASF variant pattern matching completed.' }
                                    ].map((alert, i) => (
                                        <li key={i} className="text-[10px] border-b border-white/5 pb-1 mb-1">
                                            <span className="text-gray-500 font-mono block">{alert.time}</span>
                                            <span className="text-gray-300">{alert.msg}</span>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </GlassPanel>
    );
};

export default GlobalSwineForesight;
