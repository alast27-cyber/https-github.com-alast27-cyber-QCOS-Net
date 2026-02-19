
import React from 'react';
import GlassPanel from './GlassPanel';
import { UsersIcon, ShieldCheckIcon, MapPinIcon, CurrencyDollarIcon, InformationCircleIcon, BuildingFarmIcon, TruckIcon, ShoppingCartIcon } from './Icons';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

// Mock data
const priceData = [
    { day: -14, price: 180 }, { day: -7, price: 185 }, { day: 0, price: 182 },
    { day: 7, price: 188 }, { day: 14, price: 190 },
];
const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-900/80 p-2 border border-cyan-400 text-white rounded-md text-sm">
          <p>{`Day ${label}: ₱${payload[0].value.toFixed(2)}/kg`}</p>
        </div>
      );
    }
    return null;
  };

const PigHavenConsumerTrust: React.FC = () => {
    return (
        <GlassPanel title={<div className="flex items-center"><UsersIcon className="w-6 h-6 mr-2" />PigHaven Consumer Trust</div>}>
            <div className="p-4 h-full overflow-y-auto grid grid-cols-1 md:grid-cols-2 gap-4 text-cyan-200">
                
                {/* Left Column: Traceability */}
                <div className="flex flex-col gap-4">
                    <div className="bg-black/20 p-4 rounded-lg border border-cyan-900">
                        <h4 className="flex items-center text-base font-semibold text-cyan-300 mb-3"><ShieldCheckIcon className="w-5 h-5 mr-2"/>Quantum-Secured Traceability</h4>
                        <p className="text-xs text-cyan-400 mb-2">Scan a product's QR code or enter its ID to view its journey, secured on a quantum-resistant ledger.</p>
                        <div className="flex gap-2">
                             <input type="text" defaultValue="PHTR-Q-8A4B1Z9C" className="flex-grow bg-black/40 border border-blue-500/50 rounded-md p-2 text-white placeholder:text-gray-500 font-mono text-sm"/>
                             <button className="holographic-button px-4 py-2 bg-cyan-500/30 border border-cyan-500/50 text-cyan-200 font-bold rounded">Trace</button>
                        </div>
                    </div>
                    <div className="bg-black/20 p-4 rounded-lg border border-cyan-900 flex-grow">
                        <h5 className="font-semibold text-white mb-2">Product Journey: PHTR-Q-8A4B1Z9C</h5>
                        <div className="space-y-3">
                            <div className="flex items-center gap-3 text-sm">
                                <BuildingFarmIcon className="w-6 h-6 text-green-400"/>
                                <div><p className="font-bold">Origin</p><p className="text-xs text-cyan-400">PigHaven Farms, Batangas</p></div>
                            </div>
                            <div className="h-6 w-0.5 ml-3 bg-cyan-700/50"/>
                             <div className="flex items-center gap-3 text-sm">
                                <TruckIcon className="w-6 h-6 text-blue-400"/>
                                <div><p className="font-bold">Logistics</p><p className="text-xs text-cyan-400">Quantum-Optimized Route to Manila</p></div>
                            </div>
                             <div className="h-6 w-0.5 ml-3 bg-cyan-700/50"/>
                             <div className="flex items-center gap-3 text-sm">
                                <ShoppingCartIcon className="w-6 h-6 text-purple-400"/>
                                <div><p className="font-bold">Retail</p><p className="text-xs text-cyan-400">MegaMarket, Quezon City</p></div>
                            </div>
                        </div>
                    </div>
                </div>

                 {/* Right Column: Market Info */}
                <div className="flex flex-col gap-4">
                    <div className="bg-black/20 p-4 rounded-lg border border-cyan-900">
                        <h4 className="flex items-center text-base font-semibold text-cyan-300 mb-3"><MapPinIcon className="w-5 h-5 mr-2"/>Market Availability & Price</h4>
                        <div className="grid grid-cols-2 gap-3 text-center">
                            <div className="bg-cyan-950/50 p-2 rounded">
                                <p className="text-xs text-cyan-400">Metro Manila Availability</p>
                                <p className="text-2xl font-bold text-green-300">High</p>
                            </div>
                            <div className="bg-cyan-950/50 p-2 rounded">
                                <p className="text-xs text-cyan-400">Average Price/kg</p>
                                <p className="text-2xl font-bold text-white">₱182.50</p>
                            </div>
                        </div>
                    </div>
                     <div className="bg-black/20 p-4 rounded-lg border border-cyan-900 flex-grow flex flex-col">
                        <h5 className="font-semibold text-white mb-2 flex items-center"><CurrencyDollarIcon className="w-5 h-5 mr-2"/>Price Forecast (14-Day)</h5>
                         <div className="flex-grow">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={priceData} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                                    <CartesianGrid strokeDasharray="1 1" stroke="rgba(0, 255, 255, 0.2)" />
                                    <XAxis dataKey="day" stroke="rgba(0, 255, 255, 0.7)" tick={{ fontSize: 10 }} />
                                    <YAxis stroke="rgba(0, 255, 255, 0.7)" tick={{ fontSize: 10 }} domain={[170, 200]} />
                                    <Tooltip content={<CustomTooltip />} />
                                    <Line type="monotone" dataKey="price" name="Price" stroke="#86efac" strokeWidth={2} dot={{r: 2}} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                    <div className="bg-black/20 p-4 rounded-lg border border-cyan-900">
                         <h4 className="flex items-center text-base font-semibold text-cyan-300 mb-2"><InformationCircleIcon className="w-5 h-5 mr-2"/>How Quantum Helps</h4>
                         <p className="text-xs text-cyan-400">Our quantum-optimized supply chain reduces waste and logistics costs, leading to more stable, affordable prices and a secure food supply for your community.</p>
                    </div>
                </div>

            </div>
        </GlassPanel>
    );
};

export default PigHavenConsumerTrust;
