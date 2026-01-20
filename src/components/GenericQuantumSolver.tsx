
import React, { useState } from 'react';
import GlassPanel from './GlassPanel';
import { PuzzlePieceIcon, PlayIcon, RefreshCwIcon, LoaderIcon, CheckCircle2Icon, CodeBracketIcon } from './Icons';

const GenericQuantumSolver: React.FC = () => {
    const [problemType, setProblemType] = useState('Optimization (QUBO)');
    const [variables, setVariables] = useState(10);
    const [isSolving, setIsSolving] = useState(false);
    const [solution, setSolution] = useState<string | null>(null);
    const [logs, setLogs] = useState<string[]>([]);

    const handleSolve = () => {
        setIsSolving(true);
        setSolution(null);
        setLogs([`Initializing ${problemType} solver...`, `Allocating ${variables} logical qubits...`]);

        setTimeout(() => {
            setLogs(prev => [...prev, "Constructing Hamiltonian..."]);
            setTimeout(() => {
                setLogs(prev => [...prev, "Running QAOA circuit (p=3)...", "Minimizing energy state..."]);
                setTimeout(() => {
                    const resultBitstring = Array.from({length: variables}, () => Math.round(Math.random())).join('');
                    const energy = -(variables * 0.8 + Math.random() * 2).toFixed(4);
                    setSolution(`Optimal State: |${resultBitstring}⟩\nGround Energy: ${energy} Ha`);
                    setIsSolving(false);
                    setLogs(prev => [...prev, "Convergence reached."]);
                }, 1500);
            }, 1000);
        }, 800);
    };

    return (
        <GlassPanel title={<div className="flex items-center"><PuzzlePieceIcon className="w-5 h-5 mr-2 text-yellow-400" /> General Quantum Solver</div>}>
            <div className="flex flex-col h-full p-4 gap-4">
                
                {/* Configuration */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-black/20 p-4 rounded-lg border border-yellow-800/50">
                    <div>
                        <label className="text-xs text-yellow-500 font-bold uppercase block mb-1">Problem Class</label>
                        <select 
                            value={problemType} 
                            onChange={(e) => setProblemType(e.target.value)}
                            className="w-full bg-black/50 border border-yellow-700 text-white text-sm rounded p-2 focus:border-yellow-400 outline-none"
                            disabled={isSolving}
                        >
                            <option>Optimization (QUBO)</option>
                            <option>Constraint Satisfaction (SAT)</option>
                            <option>Graph Partitioning</option>
                            <option>Traveling Salesman (TSP)</option>
                        </select>
                    </div>
                    <div>
                        <label className="text-xs text-yellow-500 font-bold uppercase block mb-1">Variables (Qubits)</label>
                        <input 
                            type="number" 
                            min="2" max="100" 
                            value={variables}
                            onChange={(e) => setVariables(parseInt(e.target.value))}
                            className="w-full bg-black/50 border border-yellow-700 text-white text-sm rounded p-2 focus:border-yellow-400 outline-none"
                            disabled={isSolving}
                        />
                    </div>
                </div>

                {/* Execution & Logs */}
                <div className="flex-grow flex flex-col bg-black/40 border border-cyan-900/30 rounded-lg overflow-hidden relative">
                    <div className="p-2 bg-black/60 border-b border-cyan-900/30 flex justify-between items-center">
                        <span className="text-[10px] text-cyan-400 uppercase font-bold tracking-widest flex items-center gap-2">
                            <CodeBracketIcon className="w-3 h-3" /> Solver Log
                        </span>
                        {isSolving && <LoaderIcon className="w-3 h-3 text-yellow-400 animate-spin" />}
                    </div>
                    <div className="flex-grow p-3 font-mono text-xs overflow-y-auto custom-scrollbar space-y-1">
                        {logs.map((log, i) => (
                            <div key={i} className="text-cyan-100/80">
                                <span className="text-cyan-700 mr-2">{`>`}</span>{log}
                            </div>
                        ))}
                    </div>
                    {solution && (
                        <div className="p-4 bg-green-900/20 border-t border-green-500/30 animate-fade-in-up">
                            <div className="flex items-center gap-2 mb-2 text-green-400 font-bold text-sm">
                                <CheckCircle2Icon className="w-5 h-5" /> Solution Found
                            </div>
                            <pre className="text-white font-mono text-xs whitespace-pre-wrap bg-black/40 p-2 rounded border border-green-800/50">
                                {solution}
                            </pre>
                        </div>
                    )}
                </div>

                {/* Action Bar */}
                <button 
                    onClick={handleSolve}
                    disabled={isSolving}
                    className={`holographic-button w-full py-3 rounded-lg font-bold text-sm flex items-center justify-center gap-2 transition-all ${isSolving ? 'bg-gray-800 text-gray-500 cursor-not-allowed' : 'bg-yellow-600/20 border-yellow-500 text-yellow-200 hover:bg-yellow-600/40'}`}
                >
                    {isSolving ? 'Annealing...' : <><PlayIcon className="w-4 h-4" /> Execute Solver</>}
                </button>
            </div>
        </GlassPanel>
    );
};

export default GenericQuantumSolver;
