
import React, { useState, useEffect, useRef } from 'react';
import GlassPanel from './GlassPanel';
import { 
    CpuChipIcon, 
    ServerStackIcon, 
    ArrowPathIcon, 
    ActivityIcon, 
    PlayIcon, 
    StopIcon, 
    LayersIcon,
    GitBranchIcon,
    ChartBarIcon,
    AlertTriangleIcon,
    CheckCircle2Icon
} from './Icons';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import QuantumMemoryDataMatrix from './QuantumMemoryDataMatrix';

// --- Types ---
type QDLPhase = 'idle' | 'input_encoding' | 'convolution' | 'pooling' | 'entanglement' | 'measurement' | 'backprop';
type TrainingMode = 'standard' | 'layer_wise';

const QuantumDeepLearning: React.FC = () => {
    // --- State ---
    const [phase, setPhase] = useState<QDLPhase>('idle');
    const [isAutoRun, setIsAutoRun] = useState(false);
    const [instruction, setInstruction] = useState('');
    const [epoch, setEpoch] = useState(0);
    const [lossData, setLossData] = useState<{ epoch: number; loss: number }[]>([]);
    const [trainingMode, setTrainingMode] = useState<TrainingMode>('standard');
    const [activeLayer, setActiveLayer] = useState<number | null>(null); // 0: Input, 1: Conv, 2: Pool, 3: Dense
    const [logs, setLogs] = useState<string[]>(["QDL Engine Initialized. 240-Qubit PQC Stack Ready."]);
    
    // Qubit Visualization State
    const [qubitStates, setQubitStates] = useState<number[]>(Array(8).fill(0));

    const logEndRef = useRef<HTMLDivElement>(null);

    // --- Helpers ---
    const addLog = (msg: string) => setLogs(prev => [...prev.slice(-4), msg]);

    useEffect(() => {
        logEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [logs]);

    // --- Simulation Logic ---
    const runTrainingCycle = () => {
        if (phase !== 'idle' && phase !== 'backprop') return;

        const task = instruction.trim() ? `"${instruction}"` : "Features";

        // 1. Input Encoding
        setPhase('input_encoding');
        setActiveLayer(0);
        setQubitStates(prev => prev.map(() => Math.random())); // Randomize input state
        addLog(`[Epoch ${epoch + 1}] Encoding ${task} into 240-dim Hilbert Space...`);

        setTimeout(() => {
            // 2. Quantum Convolution (QCNN)
            setPhase('convolution');
            setActiveLayer(1);
            addLog("Applying 240-Qubit Quantum Convolution Filter...");

            setTimeout(() => {
                // 3. Pooling (Collapse)
                setPhase('pooling');
                setActiveLayer(2);
                // Collapse half the qubits
                setQubitStates(prev => prev.map((val, i) => i % 2 === 0 ? val : 0));
                addLog("Pooling Layer: Collapsing entanglement pairs...");

                setTimeout(() => {
                    // 4. Fully Connected (Entanglement)
                    setPhase('entanglement');
                    setActiveLayer(3);
                    addLog("Dense PQC: Spreading info via Full Entanglement...");

                    setTimeout(() => {
                        // 5. Measurement & Backprop
                        setPhase('backprop');
                        setActiveLayer(null);
                        
                        // Calculate Loss based on mode
                        let currentLoss = 0.8; // Default high loss
                        if (lossData.length > 0) currentLoss = lossData[lossData.length - 1].loss;

                        if (trainingMode === 'standard') {
                            // Barren Plateau: Loss fluctuates but doesn't drop
                            currentLoss = Math.max(0.75, Math.min(0.85, currentLoss + (Math.random() - 0.5) * 0.05));
                            addLog("Gradient Descent: Gradients vanishing (Barren Plateau detected).");
                        } else {
                            // Layer-wise: Loss drops significantly
                            currentLoss = Math.max(0.05, currentLoss * 0.85); // Decay
                            addLog("Gradient Descent: Layer-wise optimization effective.");
                        }

                        setLossData(prev => [...prev.slice(-30), { epoch: epoch + 1, loss: currentLoss }]);
                        setEpoch(prev => prev + 1);

                        // Auto-switch mode logic
                        if (trainingMode === 'standard' && epoch > 10) {
                            setTrainingMode('layer_wise');
                            addLog(">>> SWITCHING PROTOCOL: Layer-wise Training Activated <<<");
                        }

                        setTimeout(() => {
                            if (isAutoRun) {
                                runTrainingCycle();
                            } else {
                                setPhase('idle');
                            }
                        }, 500);
                    }, 1000);
                }, 1000);
            }, 1000);
        }, 1000);
    };

    useEffect(() => {
        if (isAutoRun && phase === 'idle') {
            runTrainingCycle();
        }
    }, [isAutoRun]);

    const resetSimulation = () => {
        setIsAutoRun(false);
        setPhase('idle');
        setEpoch(0);
        setLossData([]);
        setTrainingMode('standard');
        setLogs(["Simulation Reset."]);
        setQubitStates(Array(8).fill(0));
    };

    // --- Render Helpers ---
    const renderLayer = (index: number, name: string, type: 'input' | 'conv' | 'pool' | 'dense') => {
        const isActive = activeLayer === index;
        const baseClass = `relative w-full p-2 mb-2 rounded border transition-all duration-300 flex flex-col justify-between`;
        const activeClass = isActive 
            ? 'bg-cyan-900/40 border-cyan-400 shadow-[0_0_10px_theme(colors.cyan.500/30%)] scale-105' 
            : 'bg-black/40 border-cyan-900/30 opacity-70';

        return (
            <div className={`${baseClass} ${activeClass}`}>
                <div className="flex items-center gap-2 mb-1">
                    {type === 'input' && <ActivityIcon className="w-4 h-4 text-blue-400" />}
                    {type === 'conv' && <LayersIcon className="w-4 h-4 text-purple-400" />}
                    {type === 'pool' && <ArrowPathIcon className="w-4 h-4 text-red-400" />}
                    {type === 'dense' && <ServerStackIcon className="w-4 h-4 text-green-400" />}
                    <span className="text-xs font-bold text-white">{name}</span>
                </div>
                {/* Visuals inside layer */}
                <div className="flex gap-1 items-center">
                    {type === 'input' && (
                        <>
                            {qubitStates.map((v, i) => (
                                <div key={i} className="w-1 h-3 rounded-full bg-blue-500" style={{ opacity: v + 0.2 }} />
                            ))}
                            <span className="text-[9px] text-blue-300 ml-1">...240q</span>
                        </>
                    )}
                    {type === 'conv' && <div className="text-[10px] text-purple-300 font-mono animate-pulse">Filter: [Rx, Ry] on 240q</div>}
                    {type === 'pool' && <div className="text-[10px] text-red-300 font-mono">Collapse 240 -> 120</div>}
                    {type === 'dense' && (
                        <div className="w-full">
                            <QuantumMemoryDataMatrix 
                                label="Entangled Weights" 
                                colorBase="green" 
                                rows={2} cols={6} 
                                className="w-full"
                            />
                        </div>
                    )}
                </div>
            </div>
        );
    };

    return (
        <GlassPanel title={<div className="flex items-center"><ServerStackIcon className="w-6 h-6 mr-2 text-purple-400"/> Quantum Deep Learning (QDL)</div>}>
            <div className="flex flex-col h-full p-3 gap-3">
                
                {/* Top Section: Architecture & Metrics */}
                <div className="flex-grow grid grid-cols-1 md:grid-cols-2 gap-3 min-h-0">
                    
                    {/* Left: The Quantum Neural Stack */}
                    <div className="flex flex-col relative overflow-hidden p-2 rounded-lg border border-cyan-900/30 bg-black/20">
                        <div className="absolute top-2 right-2 text-[10px] text-cyan-600 font-mono">240-QUBIT PQC ARCHITECTURE</div>
                        <div className="flex-grow flex flex-col justify-center">
                            {renderLayer(0, "Input Encoding", 'input')}
                            <div className="h-4 w-0.5 bg-cyan-800 mx-auto" />
                            {renderLayer(1, "Quantum Conv (QCNN)", 'conv')}
                            <div className="h-4 w-0.5 bg-cyan-800 mx-auto" />
                            {renderLayer(2, "Pooling Layer", 'pool')}
                            <div className="h-4 w-0.5 bg-cyan-800 mx-auto" />
                            {renderLayer(3, "Dense Entanglement", 'dense')}
                        </div>
                    </div>

                    {/* Right: Training Landscape */}
                    <div className="flex flex-col gap-2">
                        <div className="flex justify-between items-center text-xs text-white">
                            <span className="flex items-center gap-1"><ChartBarIcon className="w-3 h-3"/> Loss Landscape</span>
                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${trainingMode === 'standard' ? 'bg-red-900/50 text-red-300' : 'bg-green-900/50 text-green-300'}`}>
                                {trainingMode === 'standard' ? 'BARREN PLATEAU' : 'LAYER-WISE OPT'}
                            </span>
                        </div>
                        <div className="flex-grow min-h-[120px] bg-black/20 rounded border border-cyan-900/30 relative">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={lossData}>
                                    <defs>
                                        <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.8}/>
                                            <stop offset="95%" stopColor="#f43f5e" stopOpacity={0}/>
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                                    <XAxis dataKey="epoch" hide />
                                    <YAxis domain={[0, 1]} hide />
                                    <Tooltip 
                                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', fontSize: '10px' }}
                                        itemStyle={{ color: '#f43f5e' }}
                                    />
                                    <Area type="monotone" dataKey="loss" stroke="#f43f5e" fillOpacity={1} fill="url(#colorLoss)" isAnimationActive={false} />
                                </AreaChart>
                            </ResponsiveContainer>
                            {lossData.length === 0 && <div className="absolute inset-0 flex items-center justify-center text-xs text-gray-600">No training data</div>}
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                            <div className="bg-black/30 p-2 rounded border border-cyan-900/50">
                                <p className="text-gray-400">Current Loss</p>
                                <p className="text-lg font-mono text-white">{lossData.length > 0 ? lossData[lossData.length-1].loss.toFixed(4) : '---'}</p>
                            </div>
                            <div className="bg-black/30 p-2 rounded border border-cyan-900/50">
                                <p className="text-gray-400">Total Epochs</p>
                                <p className="text-lg font-mono text-cyan-400">{epoch}</p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Bottom: Controls & Logs */}
                <div className="flex-shrink-0 grid grid-cols-3 gap-3 h-36">
                    <div className="col-span-2 bg-black/30 rounded-lg border border-cyan-900/50 p-2 overflow-y-auto font-mono text-[10px] flex flex-col">
                        <div className="text-purple-400 mb-1 sticky top-0 bg-black/50 backdrop-blur-sm w-full font-bold border-b border-purple-900/50 flex justify-between">
                            <span>Training Log:</span>
                            <span className="text-gray-500">{phase.toUpperCase()}</span>
                        </div>
                        <div className="flex-grow">
                            {logs.map((l, i) => (
                                <div key={i} className="mb-0.5 text-cyan-100/80"><span className="text-purple-500 mr-1">{`>`}</span>{l}</div>
                            ))}
                            <div ref={logEndRef} />
                        </div>
                    </div>
                    
                    <div className="col-span-1 flex flex-col justify-end space-y-2">
                        <input 
                            type="text" 
                            value={instruction}
                            onChange={(e) => setInstruction(e.target.value)}
                            placeholder="QDL Task..."
                            className="w-full bg-black/40 border border-cyan-800 rounded px-2 py-1.5 text-xs text-white focus:border-cyan-500 outline-none placeholder:text-gray-600"
                            disabled={isAutoRun || (phase !== 'idle' && phase !== 'backprop')}
                        />
                        <button 
                            onClick={runTrainingCycle} 
                            disabled={isAutoRun || phase !== 'idle'}
                            className={`holographic-button w-full py-2 rounded-md flex items-center justify-center text-xs font-bold ${phase === 'idle' && !isAutoRun ? 'bg-cyan-600/30 text-white' : 'bg-gray-800/30 text-gray-500 cursor-not-allowed'}`}
                        >
                            <PlayIcon className="w-3 h-3 mr-2" /> Step
                        </button>
                        <button 
                            onClick={() => setIsAutoRun(!isAutoRun)} 
                            className={`holographic-button w-full py-2 rounded-md flex items-center justify-center text-xs font-bold ${isAutoRun ? 'bg-red-500/30 text-red-200 border-red-500/50' : 'bg-green-600/30 text-green-200 border-green-500/50'}`}
                        >
                            {isAutoRun ? <><StopIcon className="w-3 h-3 mr-2" /> Pause</> : <><ArrowPathIcon className="w-3 h-3 mr-2" /> Auto-Train</>}
                        </button>
                        <button 
                            onClick={resetSimulation} 
                            className="holographic-button w-full py-1 rounded-md flex items-center justify-center text-[10px] bg-slate-700/30 text-slate-300 border-slate-600"
                        >
                            Reset
                        </button>
                    </div>
                </div>
            </div>
        </GlassPanel>
    );
};

export default QuantumDeepLearning;
