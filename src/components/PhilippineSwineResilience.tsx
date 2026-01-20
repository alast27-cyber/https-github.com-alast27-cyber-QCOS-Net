
import React from 'react';
import GlassPanel from './GlassPanel';
import { BuildingFarmIcon, DNAIcon, ChartBarIcon, TruckIcon, ShieldCheckIcon, LightBulbIcon, ArrowTrendingUpIcon } from './Icons';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell, LineChart, Line, CartesianGrid } from 'recharts';

// Mock data
const feedData = [
  { name: 'Corn', value: 45, color: '#facc15' },
  { name: 'Soy', value: 25, color: '#86efac' },
  { name: 'Vitamins', value: 10, color: '#a78bfa' },
  { name: 'Other', value: 20, color: '#60a5fa' },
];

const demandData = [
    { week: -4, demand: 85 }, { week: -3, demand: 88 }, { week: -2, demand: 87 },
    { week: -1, demand: 90 }, { week: 0, demand: 92 }, { week: 1, demand: 95 },
    { week: 2, demand: 94 },
];

const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-900/80 p-2 border border-cyan-400 text-white rounded-md text-sm">
          <p>{`${label}: ${payload[0].value}%`}</p>
        </div>
      );
    }
    return null;
  };

const PhilippineSwineResilience: React.FC = () => {
  return (
    <GlassPanel title={<div className="flex items-center"><BuildingFarmIcon className="w-6 h-6 mr-2" />Philippine Swine Resilience</div>}>
      <div className="p-4 h-full overflow-y-auto grid grid-cols-1 md:grid-cols-3 gap-4 text-cyan-200">
        
        {/* Left Column */}
        <div className="md:col-span-1 flex flex-col gap-4">
            {/* QML Breeding */}
            <div className="bg-black/20 p-3 rounded-lg border border-cyan-900 flex-1 flex flex-col">
                <h4 className="flex items-center text-sm font-semibold text-cyan-300 mb-2"><DNAIcon className="w-4 h-4 mr-2"/>QML-Enhanced Breeding</h4>
                <div className="flex-grow bg-cyan-950/50 p-3 rounded-md text-center flex flex-col justify-center">
                    <p className="text-xs text-cyan-400">Optimal Breeding Pair Identified</p>
                    <p className="text-lg font-mono text-white">Sire: <span className="text-yellow-300">LD-481</span></p>
                    <p className="text-lg font-mono text-white">Dam: <span className="text-purple-300">YK-229</span></p>
                    <p className="text-xs text-green-400 mt-1">+15% Disease Resistance | +8% Growth Rate</p>
                </div>
            </div>
             {/* Creditworthiness */}
            <div className="bg-black/20 p-3 rounded-lg border border-cyan-900 flex-1 flex flex-col">
                <h4 className="flex items-center text-sm font-semibold text-cyan-300 mb-2"><ShieldCheckIcon className="w-4 h-4 mr-2"/>Farm Health & Credit Score</h4>
                <div className="flex-grow bg-cyan-950/50 p-3 rounded-md text-center flex flex-col justify-center">
                     <p className="text-xs text-cyan-400">QNN-Generated Score</p>
                     <p className="text-5xl font-mono text-green-300 my-1">92.5</p>
                     <p className="text-xs text-cyan-400">Creditworthiness: <span className="font-bold">Excellent</span></p>
                </div>
            </div>
        </div>

        {/* Middle Column */}
        <div className="md:col-span-1 flex flex-col gap-4">
             {/* Feed Optimization */}
            <div className="bg-black/20 p-3 rounded-lg border border-cyan-900 flex-1 flex flex-col">
                <h4 className="flex items-center text-sm font-semibold text-cyan-300 mb-2"><ChartBarIcon className="w-4 h-4 mr-2"/>Quantum Feed Optimization</h4>
                <p className="text-xs text-cyan-500 mb-2">Real-time low-cost formulation for grower stage.</p>
                <div className="flex-grow">
                     <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={feedData} layout="vertical" margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                            <XAxis type="number" hide />
                            <YAxis type="category" dataKey="name" stroke="rgba(0, 255, 255, 0.7)" tick={{ fontSize: 10 }} width={50} />
                            <Tooltip content={<CustomTooltip />} cursor={{fill: 'rgba(0, 255, 255, 0.1)'}}/>
                            <Bar dataKey="value" name="Percentage">
                                {feedData.map((entry, index) => <Cell key={`cell-${index}`} fill={entry.color} opacity={0.8}/>)}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
                <p className="text-xs text-center text-green-400 font-bold mt-2">Cost Savings: 12.7%</p>
            </div>
             {/* Demand Forecast */}
             <div className="bg-black/20 p-3 rounded-lg border border-cyan-900 flex-1 flex flex-col">
                <h4 className="flex items-center text-sm font-semibold text-cyan-300 mb-2"><ArrowTrendingUpIcon className="w-4 h-4 mr-2"/>Local Demand Forecast</h4>
                 <div className="flex-grow">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={demandData} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="1 1" stroke="rgba(0, 255, 255, 0.2)" />
                            <XAxis dataKey="week" stroke="rgba(0, 255, 255, 0.7)" tick={{ fontSize: 10 }} unit="w" />
                            <YAxis stroke="rgba(0, 255, 255, 0.7)" tick={{ fontSize: 10 }} domain={[80, 100]} />
                            <Tooltip content={<CustomTooltip />} />
                            <Line type="monotone" dataKey="demand" name="Demand" stroke="#00FFFF" strokeWidth={2} dot={{r: 2}} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>

        {/* Right Column */}
        <div className="md:col-span-1 flex flex-col gap-4">
            {/* Supply Chain */}
            <div className="bg-black/20 p-3 rounded-lg border border-cyan-900 flex-1 flex flex-col">
                <h4 className="flex items-center text-sm font-semibold text-cyan-300 mb-2"><TruckIcon className="w-4 h-4 mr-2"/>Inter-Island Logistics</h4>
                <div className="flex-grow bg-cyan-950/30 rounded-lg p-2 flex items-center justify-center text-center">
                    <p className="text-xs text-cyan-600">Stylized map of the Philippines with optimized quantum routes would be rendered here.</p>
                </div>
            </div>
             {/* Insights */}
            <div className="bg-black/20 p-3 rounded-lg border border-cyan-900 flex-1 flex flex-col">
                <h4 className="flex items-center text-sm font-semibold text-cyan-300 mb-2"><LightBulbIcon className="w-4 h-4 mr-2"/>Quantum Farm Consultant</h4>
                <ul className="text-xs space-y-2 text-cyan-300 list-disc list-inside">
                    <li><span className="font-bold text-yellow-300">Alert:</span> High ASF risk detected in Region IV-A. Increase biosecurity protocols.</li>
                    <li><span className="font-bold text-green-300">Action:</span> Shift to finisher feed ration for Batch G-12.</li>
                    <li><span className="font-bold text-blue-300">Opportunity:</span> High demand projected in Cebu. Consider rerouting next shipment.</li>
                </ul>
            </div>
        </div>
      </div>
    </GlassPanel>
  );
};

export default PhilippineSwineResilience;
