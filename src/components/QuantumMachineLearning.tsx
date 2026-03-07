import React, { useState, useEffect, useRef } from 'react';
import GlassPanel from './GlassPanel';
import { 
    BrainCircuitIcon, PlayIcon, StopIcon, SettingsIcon, 
    ChartBarIcon, RefreshCwIcon, LoaderIcon, CheckCircle2Icon,
    GitBranchIcon, Share2Icon, CpuChipIcon, FastForwardIcon,
    CircleStackIcon, DatabaseIcon, GalaxyIcon, ZapIcon, ArrowRightIcon,
    LinkIcon, SparklesIcon
} from './Icons';
import { useSimulation } from '../context/SimulationContext';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

// --- Visualization Canvas ---
const QMLTopologyVisualizer: React.FC<{ active: boolean, progress: number, depth: number, width: number, isLinked: boolean }> = ({ active, progress, depth, width, isLinked }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let frameId: number;
        let t = 0;

        const nodes: {x: number, y: number, layer: number}[] = [];
        const resize = () => {
            const parent = canvas.parentElement;
            if(parent) {
                canvas.width = parent.clientWidth;
                canvas.height = parent.clientHeight;
                initNodes(canvas.width, canvas.height);
            }
        };

        const initNodes = (w: number, h: number) => {
            nodes.length = 0;
            const layerSpacing = w / (depth + 1);
            for(let l=0; l<depth; l++) {
                const nodesInLayer = l === 0 || l === depth -1 ? 4 : 6;
                const nodeSpacing = h / (nodesInLayer + 1);
                for(let n=0; n<nodesInLayer; n++) {
                    nodes.push({
                        x: layerSpacing * (l + 1),
                        y: nodeSpacing * (n + 1),
                        layer: l
                    });
                }
            }
        };

        resize();
        window.addEventListener('resize', resize);

        const draw = () => {
            t += 0.05;
            const w = canvas.width;
            const h = canvas.height;
            ctx.clearRect(0, 0, w, h);

            // Draw Connections
            ctx.lineWidth = isLinked ? 2 : 1;
            nodes.forEach(node => {
                const nextLayerNodes = nodes.filter(n => n.layer === node.layer + 1);
                nextLayerNodes.forEach(next => {
                    const isActiveLine = active && Math.random() > 0.9;
                    ctx.beginPath();
                    ctx.moveTo(node.x, node.y);
                    ctx.lineTo(next.x, next.y);
                    ctx.strokeStyle = isLinked 
                        ? `rgba(168, 85, 247, ${0.2 + Math.random() * 0.1})` 
                        : isActiveLine 
                            ? `rgba(0, 255, 255, ${0.5 + Math.sin(t * 5) * 0.5})` 
                            : `rgba(6, 182, 212, ${0.1})`;
                    ctx.stroke();
                });
            });

            // Draw Nodes
            nodes.forEach(node => {
                ctx.beginPath();
                ctx.arc(node.x, node.y, isLinked ? 5 : 4, 0, Math.PI * 2);
                
                // Color based on active/progress
                let color = 'rgba(100, 116, 139, 0.5)';
                if (active) {
                    const layerProgress = (progress / 100) * depth;
                    if (node.layer <= layerProgress) {
                        color = isLinked ? '#a855f7' : '#22d3ee'; // Purple if linked
                        ctx.shadowBlur = isLinked ? 15 : 10;
                        ctx.shadowColor = color;
                    } else if (node.layer <= layerProgress + 1) {
                         color = '#facc15'; // Yellow for current processing
                         ctx.shadowBlur = 5;
                         ctx.shadowColor = '#facc15';
                    }
                }
                
                ctx.fillStyle = color;
                ctx.fill();
                ctx.shadowBlur = 0;
            });
            
            // "Pulse" packet during training
            if (active) {
                 const packetX = (w * (progress / 100));
                 ctx.beginPath();
                 ctx.fillStyle = isLinked ? '#a855f7' : '#ffffff';
                 ctx.arc(packetX, h/2 + Math.sin(t)*50, 3, 0, Math.PI*2);
                 ctx.fill();
            }

            frameId = requestAnimationFrame(draw);
        };
        draw();

        return () => {
            window.removeEventListener('resize', resize);
            cancelAnimationFrame(frameId);
        };
    }, [active, progress, depth, isLinked]);

    return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />;
};


const QuantumMachineLearning: React.FC<{ embedded?: boolean }> = ({ embedded = false }) => {
    const { qmlEngine, qrlEngine, startQMLTraining, stopQMLTraining, integrateQMLModel, toggleAutoEvolution, entanglementMesh } = useSimulation();
    const isLinked = entanglementMesh.isQRLtoQNNLinked;
    
    // Local UI State for Configuration
    const [selectedModel, setSelectedModel] = useState<'QNN' | 'BOLTZMANN' | 'GAN' | 'TRANSFORMER'>('QNN');
    const [learningRate, setLearningRate] = useState(0.01);
    const [circuitDepth, setCircuitDepth] = useState(4);
    const [qubits, setQubits] = useState(240); 
    const [lossHistory, setLossHistory] = useState<{epoch: number, loss: number, accuracy: number}[]>([]);
    const [showLibrary, setShowLibrary] = useState(false);

    // Sync History with Engine
    useEffect(() => {
        if (qmlEngine.status === 'TRAINING' || qmlEngine.status === 'CONVERGED') {
            setTimeout(() => {
                setLossHistory(prev => {
                    if (prev.length === 0 || prev[prev.length-1].epoch !== qmlEngine.currentEpoch) {
                        return [...prev, { epoch: qmlEngine.currentEpoch, loss: qmlEngine.loss, accuracy: qmlEngine.accuracy * 100 }].slice(-50);
                    }
                    return prev;
                });
            }, 0);
        }
        if (qmlEngine.status === 'IDLE' && lossHistory.length > 0 && qmlEngine.currentEpoch === 0) {
            setTimeout(() => setLossHistory([]), 0); 
        }
    }, [qmlEngine.currentEpoch, qmlEngine.status, qmlEngine.loss, qmlEngine.accuracy]);

    // Update local state when auto-evolution changes engine hyperparameters
    useEffect(() => {
        if (qmlEngine.autoEvolution.isActive) {
            setTimeout(() => {
                setQubits(qmlEngine.hyperparameters.qubitCount);
                setCircuitDepth(qmlEngine.hyperparameters.circuitDepth);
                setLearningRate(qmlEngine.hyperparameters.learningRate);
                setSelectedModel(qmlEngine.modelType as any);
            }, 0);
        }
    }, [qmlEngine.hyperparameters, qmlEngine.modelType, qmlEngine.autoEvolution.isActive]);

    const handleStart = () => {
        startQMLTraining({ learningRate, circuitDepth, qubitCount: qubits }, selectedModel as any);
    };

    const content = (
        <div className="flex flex-col h-full gap-4 p-4 overflow-hidden relative">
            
            {/* Configuration Panel */}
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4 bg-black/20 p-3 rounded-lg border border-purple-900/30 flex-shrink-0 relative">
                 {/* Overlay for Auto-Evolution */}
                 {qmlEngine.autoEvolution.isActive && (
                    <div className="absolute inset-0 z-10 bg-black/60 backdrop-blur-[1px] flex items-center justify-center rounded-lg border border-green-500/30 shadow-[inset_0_0_20px_rgba(34,197,94,0.1)]">
                        <div className="text-center">
                            <LoaderIcon className="w-8 h-8 text-green-400 animate-spin mx-auto mb-2" />
                            <p className="text-green-300 font-bold text-sm uppercase tracking-widest animate-pulse">Autonomous Evolution Cycle Active</p>
                            <p className="text-[10px] text-gray-400 font-mono mt-1">Stage {qmlEngine.autoEvolution.currentStage + 1}: Seeking Global Minima...</p>
                        </div>
                    </div>
                 )}

                <div>
                    <label className="text-[10px] text-cyan-500 uppercase font-bold block mb-1">Model Architecture</label>
                    <select 
                        value={selectedModel} 
                        onChange={(e) => setSelectedModel(e.target.value as any)}
                        disabled={qmlEngine.status !== 'IDLE'}
                        className="w-full bg-black/50 border border-cyan-800 text-xs text-white rounded p-1.5 outline-none focus:border-purple-500 disabled:opacity-50"
                    >
                        <option value="QNN">Quantum Neural (QNN)</option>
                        <option value="BOLTZMANN">Quantum Boltzmann</option>
                        <option value="GAN">Quantum GAN</option>
                        <option value="TRANSFORMER">Q-Transformer</option>
                    </select>
                </div>
                <div>
                    <label className="text-[10px] text-cyan-500 uppercase font-bold block mb-1">Learning Rate: {learningRate.toFixed(3)}</label>
                    <input 
                        type="range" min="0.001" max="0.1" step="0.001" 
                        value={learningRate} 
                        onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                        disabled={qmlEngine.status !== 'IDLE'}
                        className="w-full h-1.5 bg-cyan-900 rounded-lg appearance-none cursor-pointer" 
                    />
                </div>
                <div>
                    <label className="text-[10px] text-cyan-500 uppercase font-bold block mb-1">Qubits: {qubits}</label>
                    <input 
                        type="range" min="2" max="1024" step="1" 
                        value={qubits} 
                        onChange={(e) => setQubits(parseInt(e.target.value))}
                        disabled={qmlEngine.status !== 'IDLE'}
                        className="w-full h-1.5 bg-cyan-900 rounded-lg appearance-none cursor-pointer" 
                    />
                </div>
                <div className="flex items-end gap-2 md:col-span-2">
                     {qmlEngine.status === 'IDLE' || qmlEngine.status === 'CONVERGED' || qmlEngine.status === 'FAILED' ? (
                        <>
                            <button onClick={handleStart} className="flex-1 holographic-button py-2 text-xs font-bold bg-green-600/20 border-green-500 text-green-300 flex items-center justify-center gap-2 hover:bg-green-600/40">
                                <PlayIcon className="w-3 h-3" /> Train
                            </button>
                            <button 
                                onClick={toggleAutoEvolution} 
                                className={`flex-1 holographic-button py-2 text-xs font-bold flex items-center justify-center gap-2 
                                    ${qmlEngine.autoEvolution.isActive ? 'bg-red-600/20 border-red-500 text-red-300 hover:bg-red-600/40' : 'bg-purple-600/30 border-purple-500 text-purple-300 hover:bg-purple-600/50 animate-pulse-border'}`}
                                title="Auto-update engine configuration"
                            >
                                <FastForwardIcon className="w-3 h-3" /> {qmlEngine.autoEvolution.isActive ? 'Stop Evolution' : 'Auto-Evolve'}
                            </button>
                        </>
                     ) : (
                         <button onClick={stopQMLTraining} className="w-full holographic-button py-2 text-xs font-bold bg-red-900/30 border-red-500 text-red-200 flex items-center justify-center gap-2 hover:bg-red-900/40">
                             <StopIcon className="w-3 h-3" /> Stop Training
                         </button>
                     )}
                </div>
            </div>

            {/* Main Content Area */}
            <div className="flex-grow flex gap-4 min-h-0 relative">
                
                {/* Visualizer Area */}
                <div className={`flex-grow flex flex-col transition-all duration-300 ${showLibrary ? 'md:w-2/3' : 'w-full'}`}>
                     
                     <div className={`flex-grow bg-black/40 rounded-lg border transition-all duration-700 relative overflow-hidden flex flex-col ${isLinked ? 'border-purple-500 shadow-[0_0_20px_rgba(168,85,247,0.2)]' : 'border-cyan-800/50'}`}>
                        <div className="absolute top-2 left-2 z-10 flex gap-2">
                            <span className="text-[10px] text-cyan-300 font-mono bg-black/50 px-2 rounded border border-cyan-900">
                                Epoch: {qmlEngine.currentEpoch}/{qmlEngine.totalEpochs}
                            </span>
                            {isLinked && (
                                <span className="text-[10px] bg-purple-900/50 text-purple-300 px-2 rounded border border-purple-500 flex items-center gap-1 animate-pulse">
                                    <LinkIcon className="w-2 h-2" /> POLICY_STREAM: ACTIVE
                                </span>
                            )}
                        </div>

                        {/* Entangled Stream HUD */}
                        {isLinked && (
                            <div className="absolute top-10 left-2 z-10 w-48 bg-black/60 border border-purple-800/40 rounded p-2 animate-fade-in-right">
                                <p className="text-[8px] text-purple-400 uppercase font-black mb-1 tracking-widest">Incoming Strategy Vector</p>
                                <div className="flex gap-0.5 h-6 items-center">
                                    {qrlEngine.policyDistribution.map((p, i) => (
                                        <div key={i} className="flex-1 bg-purple-900/50 border-t border-purple-500" style={{ height: `${p * 100}%` }}></div>
                                    ))}
                                </div>
                                <div className="flex justify-between mt-1 text-[7px] text-gray-500 font-mono uppercase">
                                    <span>Reward: {(qrlEngine.reward || 0).toFixed(1)}</span>
                                    <span className="text-green-400">Linked</span>
                                </div>
                            </div>
                        )}

                        <QMLTopologyVisualizer 
                            active={qmlEngine.status === 'TRAINING'} 
                            progress={qmlEngine.progress}
                            depth={qmlEngine.hyperparameters.circuitDepth}
                            width={100}
                            isLinked={isLinked}
                        />
                        {qmlEngine.status === 'CONVERGED' && !qmlEngine.autoEvolution.isActive && (
                            <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm z-20 flex-col animate-fade-in">
                                <CheckCircle2Icon className="w-12 h-12 text-green-400 mb-2" />
                                <h3 className="text-xl font-bold text-white">Model Converged</h3>
                                <p className="text-sm text-green-300">Accuracy: {((qmlEngine.accuracy || 0) * 100).toFixed(4)}%</p>
                                <button 
                                    onClick={integrateQMLModel}
                                    className="mt-4 holographic-button px-6 py-2 bg-blue-600/30 border-blue-500 text-blue-200 font-bold rounded flex items-center gap-2"
                                >
                                    <Share2Icon className="w-4 h-4" /> Integrate to System
                                </button>
                            </div>
                        )}
                     </div>

                    {/* Performance Graphs (Lower half) */}
                    <div className="h-40 grid grid-cols-2 gap-4 mt-4">
                        <div className="bg-black/30 rounded-lg border border-cyan-800/50 p-2 relative">
                            <p className="text-[9px] text-cyan-500 uppercase font-bold absolute top-2 left-2 z-10">Convergence Delta Trace</p>
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={lossHistory}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                                    <XAxis dataKey="epoch" hide />
                                    <YAxis yAxisId="left" domain={[0, 'auto']} hide />
                                    <YAxis yAxisId="right" orientation="right" domain={[0, 100]} hide />
                                    <Tooltip 
                                        contentStyle={{backgroundColor: '#000', borderColor: '#a855f7'}} 
                                        itemStyle={{fontSize: '9px'}} 
                                        labelStyle={{display: 'none'}}
                                    />
                                    <Line yAxisId="left" type="monotone" dataKey="loss" stroke={isLinked ? "#a855f7" : "#ef4444"} dot={false} strokeWidth={2} isAnimationActive={false} />
                                    <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#22c55e" dot={false} strokeWidth={1.5} isAnimationActive={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="bg-black/30 rounded-lg border border-cyan-800/50 p-2 overflow-y-auto font-mono text-[9px] custom-scrollbar">
                            {qmlEngine.logs.map((log, i) => (
                                <div key={i} className="mb-0.5 text-gray-300 border-b border-white/5 pb-0.5 last:border-0 hover:text-cyan-200 transition-colors">
                                    <span className="text-cyan-700 mr-2">[{new Date().toLocaleTimeString()}]</span>
                                    {log}
                                </div>
                            ))}
                            {qmlEngine.status === 'TRAINING' && <div className="text-cyan-500 animate-pulse mt-1">&gt; SEEKING GLOBAL MINIMA...</div>}
                        </div>
                    </div>
                </div>

                {/* Model Library Sidebar */}
                {showLibrary && (
                    <div className="w-1/3 bg-black/30 rounded-lg border border-cyan-900/50 flex flex-col overflow-hidden animate-fade-in-right">
                         <div className="p-2 border-b border-cyan-900/50 bg-cyan-950/40 text-xs font-bold text-cyan-300 flex items-center justify-between">
                             <span>Evolved Models Registry</span>
                             <span className="bg-cyan-900/50 px-1.5 rounded text-[10px] font-mono">{qmlEngine.modelLibrary.length}</span>
                         </div>
                         <div className="flex-grow overflow-y-auto p-2 space-y-2 custom-scrollbar">
                             {qmlEngine.modelLibrary.length === 0 && (
                                 <div className="text-center text-gray-500 text-[10px] italic p-4">
                                     No converged artifacts saved.
                                 </div>
                             )}
                             {qmlEngine.modelLibrary.map((model) => (
                                 <div key={model.id} className="bg-black/40 border border-cyan-800/30 rounded p-2 hover:bg-cyan-900/10 transition-colors group">
                                     <div className="flex justify-between items-start mb-1">
                                         <span className="text-[10px] font-bold text-white group-hover:text-cyan-400 transition-colors font-mono">{model.type}</span>
                                         <span className="text-[9px] text-green-400 font-mono font-bold">{((model.accuracy || 0) * 100).toFixed(2)}%</span>
                                     </div>
                                     <div className="grid grid-cols-2 gap-1 text-[9px] text-gray-400 font-mono mb-1">
                                         <span>Qubits: {model.qubits}</span>
                                         <span>Depth: {model.depth}</span>
                                     </div>
                                     <div className="text-[8px] text-cyan-600 text-right font-mono uppercase">{model.timestamp}</div>
                                 </div>
                             ))}
                         </div>
                    </div>
                )}

            </div>
        </div>
    );

    if (embedded) return content;

    return (
        <GlassPanel title={
            <div className="flex items-center justify-between w-full">
                <div className="flex items-center">
                    <BrainCircuitIcon className="w-5 h-5 mr-2 text-purple-400" />
                    <span>Quantum Machine Learning Engine</span>
                </div>
                <div className="flex items-center gap-2">
                    {isLinked && (
                        <div className="flex items-center gap-1.5 text-[8px] font-black text-purple-200 bg-purple-600/40 px-2 py-0.5 rounded border border-purple-400 animate-pulse">
                            <SparklesIcon className="w-2.5 h-2.5" /> ADAPTIVE_SYNC
                        </div>
                    )}
                    {qmlEngine.autoEvolution.isActive && (
                         <span className="text-[10px] bg-green-900/30 text-green-300 px-2 py-0.5 rounded border border-green-600 animate-pulse flex items-center">
                            <FastForwardIcon className="w-3 h-3 mr-1" />
                            AUTO-EVOLVING
                        </span>
                    )}
                    <button 
                        onClick={() => setShowLibrary(!showLibrary)} 
                        className={`text-[10px] px-2 py-0.5 rounded border flex items-center gap-1 transition-colors ${showLibrary ? 'bg-cyan-700 text-white' : 'bg-black/40 text-cyan-400 hover:text-white'}`}
                    >
                        <DatabaseIcon className="w-3 h-3" /> Library
                    </button>
                </div>
            </div>
        }>
            {content}
        </GlassPanel>
    );
};

export default QuantumMachineLearning;