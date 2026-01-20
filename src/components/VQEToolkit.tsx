
import React, { useState, useEffect } from 'react';
import GlassPanel from './GlassPanel';
import { GitBranchIcon, AtomIcon, PlayIcon, LoaderIcon } from './Icons';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

const VQEToolkit: React.FC = () => {
    const [molecule, setMolecule] = useState('H2');
    const [isOptimizing, setIsOptimizing] = useState(false);
    const [iteration, setIteration] = useState(0);
    const [energyHistory, setEnergyHistory] = useState<{iter: number, energy: number}[]>([]);
    const [currentEnergy, setCurrentEnergy] = useState(0);
    
    // Theoretical ground states for demo
    const targets: {[key: string]: number} = { 'H2': -1.137, 'LiH': -7.882, 'BeH2': -15.595 };

    const startVQE = () => {
        setIsOptimizing(true);
        setIteration(0);
        setEnergyHistory([]);
        setCurrentEnergy(0); // Start high
    };

    useEffect(() => {
        if (!isOptimizing) return;
        
        const target = targets[molecule];
        let currentIter = iteration;
        let currentE = target + Math.random() * 2; // Initial guess

        const interval = setInterval(() => {
            currentIter++;
            setIteration(currentIter);
            
            // Gradient Descent simulation
            const diff = currentE - target;
            currentE = currentE - (diff * 0.2) + (Math.random() - 0.5) * 0.05;
            
            setCurrentEnergy(currentE);
            setEnergyHistory(prev => [...prev, { iter: currentIter, energy: currentE }]);

            if (currentIter > 50 || Math.abs(currentE - target) < 0.01) {
                setIsOptimizing(false);
            }
        }, 100);

        return () => clearInterval(interval);
    }, [isOptimizing, molecule, iteration]);

    return (
        <GlassPanel title={<div className="flex items-center"><GitBranchIcon className="w-6 h-6 mr-2 text-pink-400" /> VQE Molecular Solver</div>}>
            <div className="flex flex-col h-full p-4 gap-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Controls */}
                    <div className="bg-black/30 p-4 rounded-lg border border-cyan-800/50">
                        <label className="text-xs font-bold text-cyan-500 uppercase block mb-2">Target Molecule</label>
                        <select 
                            value={molecule}
                            onChange={(e) => setMolecule(e.target.value)}
                            disabled={isOptimizing}
                            className="w-full bg-black/50 border border-cyan-700 text-white rounded p-2 mb-4"
                        >
                            <option value="H2">Hydrogen (H2)</option>
                            <option value="LiH">Lithium Hydride (LiH)</option>
                            <option value="BeH2">Beryllium Hydride (BeH2)</option>
                        </select>
                        <button 
                            onClick={startVQE}
                            disabled={isOptimizing}
                            className="holographic-button w-full py-2 flex items-center justify-center gap-2 font-bold text-sm bg-pink-600/20 border-pink-500 text-pink-200 hover:bg-pink-600/40 disabled:opacity-50"
                        >
                            {isOptimizing ? <LoaderIcon className="w-4 h-4 animate-spin"/> : <PlayIcon className="w-4 h-4"/>}
                            {isOptimizing ? 'Minimizing...' : 'Calculate Ground State'}
                        </button>
                    </div>

                    {/* Stats */}
                    <div className="bg-black/30 p-4 rounded-lg border border-cyan-800/50 flex flex-col justify-center text-center">
                        <p className="text-xs text-gray-400 uppercase">Current Energy</p>
                        <p className="text-3xl font-mono text-white my-2">{currentEnergy.toFixed(5)} <span className="text-sm text-gray-500">Ha</span></p>
                        <div className="flex justify-between text-xs px-4 mt-2">
                             <span className="text-cyan-400">Iter: {iteration}</span>
                             <span className="text-green-400">Target: {targets[molecule]}</span>
                        </div>
                    </div>
                </div>

                {/* Chart */}
                <div className="flex-grow bg-black/20 border border-cyan-900/50 rounded-lg p-2 min-h-0 flex flex-col">
                    <div className="flex items-center gap-2 mb-2 px-2">
                        <AtomIcon className="w-4 h-4 text-cyan-500" />
                        <span className="text-xs font-bold text-cyan-300">Optimization Landscape</span>
                    </div>
                    <div className="flex-grow min-h-0">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={energyHistory}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                <XAxis dataKey="iter" stroke="gray" tick={{fontSize: 10}} />
                                <YAxis stroke="gray" domain={['auto', 'auto']} tick={{fontSize: 10}} />
                                <Tooltip contentStyle={{backgroundColor: '#000', borderColor: '#ec4899'}} itemStyle={{color: '#fff'}} />
                                <Line type="monotone" dataKey="energy" stroke="#ec4899" strokeWidth={2} dot={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>
        </GlassPanel>
    );
};

export default VQEToolkit;
