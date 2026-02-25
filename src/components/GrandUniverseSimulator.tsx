import React, { useState, useEffect, useRef } from 'react';
import GlassPanel from './GlassPanel';
import { 
    GalaxyIcon, PlayIcon, StopIcon, RefreshCwIcon, 
    SettingsIcon, CpuChipIcon, SparklesIcon,
    NetworkIcon, BrainCircuitIcon, LoaderIcon, LinkIcon,
    LayersIcon, AtomIcon, ActivityIcon, EyeIcon
} from './Icons';
import { AppDefinition } from '../types';
import { useSimulation } from '../context/SimulationContext';
import { useToast } from '../context/ToastContext';
import { GoogleGenAI } from '@google/genai';
import { generateContentWithRetry } from '../utils/gemini';

interface GrandUniverseSimulatorProps {
    qubitCount?: number;
    onApplyPatch?: (file: string, content: string) => void;
    onExportToCreator?: (prompt: string) => void;
    connectedApp?: Omit<AppDefinition, 'component'> | null;
    embedded?: boolean;
}

interface Universe {
    id: number;
    stability: number;
    active: boolean;
    type: 'standard' | 'optimized' | 'qnd_mirror';
}

const GrandUniverseSimulator: React.FC<GrandUniverseSimulatorProps> = ({ 
    qubitCount = 240, 
    onApplyPatch, 
    onExportToCreator,
    connectedApp,
    embedded = false
}) => {
    const { 
        inquiry, updateInquiry, simConfig,
        entanglementMesh, toggleUniverseToKernel, toggleUniverseToAgentQ, universeConnections
    } = useSimulation();
    const { addToast } = useToast();

    const [isSimulationRunning, setIsSimulationRunning] = useState(simConfig.isUniverseActive);
    const [timeRemaining, setTimeRemaining] = useState('CALCULATING...');
    const [entropy, setEntropy] = useState(0.45);
    const [timelineDivergence, setTimelineDivergence] = useState(0);
    const [universes, setUniverses] = useState<Universe[]>(() => {
        const count = embedded ? 8 : 16;
        return Array.from({length: count}, (_, i) => ({
            id: i,
            stability: 0.8 + Math.random() * 0.2,
            active: true,
            type: i % 3 === 0 ? 'qnd_mirror' : 'standard'
        }));
    });

    const [simulationMode, setSimulationMode] = useState<'universes' | 'qul_ledger' | 'chronos_engine' | 'psn_navigator' | 'parallel_cognitive'>('universes');
    const [neuralResonance, setNeuralResonance] = useState(0);

    // 3 Concurrent Simulation Inputs
    const [simInputs, setSimInputs] = useState<string[]>(['', '', '']);
    const [activeSims, setActiveSims] = useState<boolean[]>([false, false, false]);

    const [optScore, setOptScore] = useState(0);
    const [optLogs, setOptLogs] = useState<string[]>([]);
    const [showSettings, setShowSettings] = useState(false);
    
    const [cognitiveEngines, setCognitiveEngines] = useState([
        { id: 'QLLM', name: 'Quantum LLM', strength: 0.85, active: false, result: 0 },
        { id: 'QRL', name: 'Quantum RL', strength: 0.92, active: false, result: 0 },
        { id: 'QML', name: 'Quantum ML', strength: 0.88, active: false, result: 0 },
        { id: 'QGL', name: 'Quantum GL', strength: 0.75, active: false, result: 0 },
        { id: 'QDL', name: 'Quantum DL', strength: 0.95, active: false, result: 0 },
    ]);
    const [parallelSimActive, setParallelSimActive] = useState(false);
    const [consensusResult, setConsensusResult] = useState<number | null>(null);

    const apiResultRef = useRef<string | null>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    const runParallelSimulation = () => {
        if (parallelSimActive) return;
        setParallelSimActive(true);
        setSimulationMode('parallel_cognitive');
        setIsSimulationRunning(true);
        setConsensusResult(null);
        addToast("Parallel Cognitive Simulation Initiated.", "info");
        
        // Reset active state
        setCognitiveEngines(prev => prev.map(e => ({ ...e, active: false, result: 0 })));

        // Activate engines one by one
        cognitiveEngines.forEach((engine, idx) => {
            setTimeout(() => {
                setCognitiveEngines(prev => prev.map(e => e.id === engine.id ? { ...e, active: true, result: Math.random() * 100 } : e));
                setOptLogs(prev => [...prev, `[PARALLEL] ${engine.name} processing timeline data...`]);
            }, idx * 800);
        });

        // Calculate consensus
        setTimeout(() => {
            setCognitiveEngines(prev => {
                const results = prev.map(e => e.result * e.strength);
                const totalStrength = prev.reduce((sum, e) => sum + e.strength, 0);
                const consensus = results.reduce((sum, r) => sum + r, 0) / totalStrength;
                setConsensusResult(consensus);
                return prev;
            });
            setOptLogs(prev => [...prev, `[PARALLEL] Consensus Reached. Most accurate timeline predicted.`]);
            setParallelSimActive(false);
            setIsSimulationRunning(false);
        }, cognitiveEngines.length * 800 + 1000);
    };

    const handleInputChange = (index: number, value: string) => {
        const newInputs = [...simInputs];
        newInputs[index] = value;
        setSimInputs(newInputs);
    };

    const handleRunSimSlot = (index: number) => {
        if (!simInputs[index].trim()) return;
        
        const newActive = [...activeSims];
        newActive[index] = true;
        setActiveSims(newActive);

        const cmd = simInputs[index].toLowerCase();
        
        if (cmd.includes('qul') || cmd.includes('ledger') || cmd.includes('mirror')) {
            setSimulationMode('qul_ledger');
            setIsSimulationRunning(true);
            addToast("Quantum Universal Ledger: QND Measurement Active.", "info");
            setTimeout(() => {
                setOptLogs(prev => [...prev, `[SLOT ${index+1}] Scribe Layer: Ingesting Entangled States...`]);
            }, 1000);
        } else if (cmd.includes('chronos') || cmd.includes('oracle') || cmd.includes('predict')) {
            setSimulationMode('chronos_engine');
            setIsSimulationRunning(true);
            addToast("Chronos Engine: Probabilistic Forking Initiated.", "info");
            setTimeout(() => {
                setOptLogs(prev => [...prev, `[SLOT ${index+1}] MToE: Resolving Undecidable Phenomena...`]);
            }, 1000);
        } else if (cmd.includes('psn') || cmd.includes('navigator') || cmd.includes('anomaly')) {
            setSimulationMode('psn_navigator');
            setIsSimulationRunning(true);
            addToast("Probability Space Navigator: Scanning for High-Deviation Futures.", "warning");
            setTimeout(() => {
                setOptLogs(prev => [...prev, `[SLOT ${index+1}] PSN: Constructive Interference Applied to Query.`]);
            }, 1000);
        } else {
            addToast("Grand Simulator: Processing Query...", "info");
            setTimeout(() => {
                setOptLogs(prev => [...prev, `[SLOT ${index+1}] AgentQ: Applying Hamiltonian Energy Minimization...`]);
            }, 1000);
        }

        setTimeout(() => {
             const newActiveComplete = [...activeSims];
             newActiveComplete[index] = false;
             setActiveSims(newActiveComplete); 
        }, 3000);
    };

    useEffect(() => {
        let interval: ReturnType<typeof setInterval>;
        if (isSimulationRunning && simulationMode === 'universes') {
            interval = setInterval(() => {
                setEntropy(prev => Math.min(1, Math.max(0, prev + (Math.random() - 0.5) * 0.05)));
                setTimelineDivergence(prev => prev + 0.1);
                
                const secs = Math.max(0, 120 - Math.floor(timelineDivergence));
                const mins = Math.floor(secs / 60);
                const remainderSecs = secs % 60;
                setTimeRemaining(`${mins.toString().padStart(2, '0')}:${remainderSecs.toString().padStart(2, '0')}`);

                setUniverses(prev => prev.map(u => ({
                    ...u,
                    stability: Math.max(0, Math.min(1, u.stability + (u.type === 'qnd_mirror' ? (Math.random() * 0.02) : (Math.random() - 0.5) * 0.1)))
                })));

            }, 1000);
        }
        return () => clearInterval(interval);
    }, [isSimulationRunning, timelineDivergence, simulationMode]);

    useEffect(() => {
        if (simulationMode === 'universes') return;

        let animationFrameId: number;
        let phase = 0;

        const render = () => {
            const canvas = canvasRef.current;
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;

            const parentNode = canvas.parentElement;
            if (parentNode) {
                canvas.width = parentNode.clientWidth;
                canvas.height = parentNode.clientHeight;
            }

            const w = canvas.width;
            const h = canvas.height;
            const cx = w / 2;
            const cy = h / 2;

            ctx.clearRect(0, 0, w, h);
            phase += 0.05;

            if (simulationMode === 'qul_ledger') {
                // Draw QUL Mirror Effect (Hilbert Hotel & 1:1 Entanglement)
                const numParticles = 40;
                
                // Draw Central Ledger Core
                ctx.beginPath();
                ctx.arc(cx, cy, 40, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(168, 85, 247, ${isSimulationRunning ? 0.2 + Math.sin(phase)*0.1 : 0.1})`;
                ctx.fill();
                ctx.strokeStyle = '#a855f7';
                ctx.lineWidth = 2;
                ctx.stroke();
                
                ctx.fillStyle = '#d8b4fe';
                ctx.font = 'bold 10px monospace';
                ctx.textAlign = 'center';
                ctx.fillText('QUL CORE', cx, cy + 4);

                // Draw entangled pairs (Universe <-> Ledger)
                for (let i = 0; i < numParticles; i++) {
                    const angle = (i / numParticles) * Math.PI * 2 + phase * 0.2;
                    const dist = 120 + Math.sin(phase * 2 + i) * 20;
                    
                    const ux = cx + Math.cos(angle) * dist; // Universe particle
                    const uy = cy + Math.sin(angle) * dist;
                    
                    const lx = cx + Math.cos(angle) * 40; // Ledger particle (mapped to surface of core)
                    const ly = cy + Math.sin(angle) * 40;

                    // Draw Entanglement Link (QND Measurement)
                    ctx.beginPath();
                    ctx.moveTo(lx, ly);
                    ctx.lineTo(ux, uy);
                    ctx.strokeStyle = `rgba(168, 85, 247, ${0.1 + Math.random() * 0.3})`;
                    ctx.lineWidth = 1;
                    if (isSimulationRunning && Math.random() > 0.8) {
                        ctx.strokeStyle = '#c084fc';
                        ctx.lineWidth = 2;
                    }
                    ctx.stroke();

                    // Universe Particle
                    ctx.beginPath();
                    ctx.arc(ux, uy, 3, 0, Math.PI * 2);
                    ctx.fillStyle = '#38bdf8'; // Cyan/Blue for physical universe
                    ctx.fill();

                    // Ledger Particle (Mirror)
                    ctx.beginPath();
                    ctx.arc(lx, ly, 2, 0, Math.PI * 2);
                    ctx.fillStyle = '#f0abfc'; // Pink/Purple for synthetic ledger
                    ctx.fill();
                }

                // Hilbert Hotel Mapping Rings
                ctx.beginPath();
                ctx.strokeStyle = `rgba(168, 85, 247, 0.2)`;
                ctx.lineWidth = 1;
                ctx.setLineDash([4, 4]);
                ctx.arc(cx, cy, 120, 0, Math.PI * 2);
                ctx.stroke();
                ctx.arc(cx, cy, 140, 0, Math.PI * 2);
                ctx.stroke();
                ctx.setLineDash([]);
                
                ctx.fillStyle = '#a855f7';
                ctx.font = '12px monospace';
                ctx.fillText('1:1 ENTANGLEMENT MAPPING (QND)', cx, cy + 180);
            } else if (simulationMode === 'chronos_engine') {
                // Draw Chronos Engine Probability Space
                const radius = 80 + Math.sin(phase) * 10;
                ctx.beginPath();
                ctx.arc(cx, cy, radius, 0, Math.PI * 2);
                ctx.strokeStyle = '#22d3ee';
                ctx.lineWidth = 3;
                ctx.stroke();
                
                for(let i=0; i<12; i++) {
                    const angle = (i / 12) * Math.PI * 2 + phase;
                    ctx.beginPath();
                    ctx.moveTo(cx, cy);
                    ctx.lineTo(cx + Math.cos(angle) * radius * 1.5, cy + Math.sin(angle) * radius * 1.5);
                    ctx.strokeStyle = `rgba(34, 211, 238, ${0.2 + Math.random() * 0.5})`;
                    ctx.stroke();
                }
                ctx.fillStyle = '#22d3ee';
                ctx.fillText('MToE FORKING', cx, cy + radius + 20);
            } else if (simulationMode === 'psn_navigator') {
                // Draw Probability Space Navigator
                ctx.beginPath();
                ctx.moveTo(0, cy);
                for(let x=0; x<w; x+=10) {
                    const y = cy + Math.sin(x * 0.05 + phase) * 40 * Math.sin(x * 0.01);
                    ctx.lineTo(x, y);
                }
                ctx.strokeStyle = '#facc15';
                ctx.lineWidth = 2;
                ctx.stroke();
                
                // Anomaly detection point
                const anomalyX = (phase * 50) % w;
                const anomalyY = cy + Math.sin(anomalyX * 0.05 + phase) * 40 * Math.sin(anomalyX * 0.01);
                ctx.beginPath();
                ctx.arc(anomalyX, anomalyY, 8, 0, Math.PI * 2);
                ctx.fillStyle = '#ef4444';
                ctx.fill();
                ctx.fillStyle = '#facc15';
                ctx.fillText('HIGH-DEVIATION FUTURE DETECTED', cx, cy - 60);
            } else if (simulationMode === 'parallel_cognitive') {
                // Draw Parallel Cognitive Engines
                const radius = 100;
                const numEngines = 5;
                
                // Draw central node
                ctx.beginPath();
                ctx.arc(cx, cy, 30 + Math.sin(phase * 2) * 5, 0, Math.PI * 2);
                ctx.fillStyle = consensusResult !== null ? '#10b981' : '#3b82f6';
                ctx.fill();
                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 14px sans-serif';
                ctx.fillText(consensusResult !== null ? `${consensusResult.toFixed(1)}%` : 'SYNC', cx, cy + 5);

                // Draw engines and connections
                for(let i=0; i<numEngines; i++) {
                    const angle = (i / numEngines) * Math.PI * 2 - Math.PI / 2;
                    const ex = cx + Math.cos(angle) * radius;
                    const ey = cy + Math.sin(angle) * radius;
                    
                    const engine = cognitiveEngines[i];
                    
                    // Connection line
                    ctx.beginPath();
                    ctx.moveTo(cx, cy);
                    ctx.lineTo(ex, ey);
                    ctx.strokeStyle = engine.active ? '#06b6d4' : '#374151';
                    ctx.lineWidth = engine.active ? 3 : 1;
                    if (engine.active) {
                        ctx.setLineDash([5, 5]);
                        ctx.lineDashOffset = -phase * 20;
                    } else {
                        ctx.setLineDash([]);
                    }
                    ctx.stroke();
                    ctx.setLineDash([]);

                    // Engine node
                    ctx.beginPath();
                    ctx.arc(ex, ey, 20, 0, Math.PI * 2);
                    ctx.fillStyle = engine.active ? '#0891b2' : '#1f2937';
                    ctx.fill();
                    ctx.strokeStyle = engine.active ? '#22d3ee' : '#4b5563';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    
                    // Engine label
                    ctx.fillStyle = engine.active ? '#cffafe' : '#9ca3af';
                    ctx.font = '10px monospace';
                    ctx.fillText(engine.id, ex, ey + 4);
                    
                    if (engine.result > 0) {
                        ctx.fillStyle = '#a5b4fc';
                        ctx.fillText(`${engine.result.toFixed(0)}%`, ex, ey - 25);
                    }
                }
            }

            animationFrameId = requestAnimationFrame(render);
        };
        render();
        return () => { cancelAnimationFrame(animationFrameId); };
    }, [simulationMode, isSimulationRunning]);

    const toggleSimulation = () => setIsSimulationRunning(!isSimulationRunning);

    const handleReset = () => {
        setIsSimulationRunning(false);
        setTimelineDivergence(0);
        setEntropy(0.45);
        setTimeRemaining('02:00');
        setUniverses(prev => prev.map(u => ({ ...u, type: 'standard', stability: 0.8 + Math.random() * 0.2 })));
        setSimulationMode('universes');
    };

    const content = (
        <div className={`flex flex-col h-full gap-2 overflow-hidden relative ${embedded ? 'p-1' : 'p-4'}`}>
            <div className={`flex justify-between items-center bg-black/30 rounded-lg border border-purple-800/50 flex-shrink-0 ${embedded ? 'p-1.5' : 'p-3'}`}>
                <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1.5">
                        <AtomIcon className={`${embedded ? 'w-4 h-4' : 'w-5 h-5'} text-purple-400`} />
                        <span className={`${embedded ? 'text-[10px]' : 'text-sm'} font-bold text-purple-200`}>
                            {simulationMode === 'qul_ledger' ? 'QUL LEDGER' : 
                             simulationMode === 'chronos_engine' ? 'CHRONOS ENGINE' : 
                             simulationMode === 'psn_navigator' ? 'PSN NAVIGATOR' : 
                             simulationMode === 'parallel_cognitive' ? 'PARALLEL COGNITIVE' : 'GRAND UNIVERSE'}
                        </span>
                    </div>
                    {entanglementMesh.isUniverseLinkedToQLang && (
                        <div className="flex items-center gap-1.5 ml-2 px-2 py-0.5 rounded-full bg-purple-900/40 border border-purple-400 text-[8px] font-black text-white animate-pulse">
                            <LinkIcon className="w-2.5 h-2.5" /> LINKED
                        </div>
                    )}
                </div>
                <div className="flex gap-1.5">
                    <button 
                        onClick={toggleSimulation}
                        className={`holographic-button px-3 py-1 rounded flex items-center gap-1.5 ${embedded ? 'text-[9px]' : 'text-xs'} font-bold ${isSimulationRunning ? 'bg-red-600/30 border-red-500 text-red-200' : 'bg-green-600/30 border-green-500 text-green-200'}`}
                    >
                        {isSimulationRunning ? <StopIcon className="w-3 h-3" /> : <PlayIcon className="w-3 h-3" />}
                        {isSimulationRunning ? 'Stop' : 'Run'}
                    </button>
                    <button onClick={handleReset} className="holographic-button px-2 py-1 rounded bg-slate-600/30 border-slate-500 text-slate-300">
                        <RefreshCwIcon className="w-3 h-3" />
                    </button>
                </div>
            </div>

            <div className={`grid grid-cols-1 ${embedded ? 'gap-2' : 'lg:grid-cols-3 gap-4'} h-full min-h-0`}>
                {!embedded && (
                    <div className="bg-black/30 border border-purple-800/50 rounded-lg p-4 flex flex-col gap-4 overflow-y-auto">
                        <h4 className="text-purple-300 font-bold flex items-center justify-between border-b border-purple-800 pb-2">
                            <LayersIcon className="w-4 h-4" /> Grand Simulator Layers
                        </h4>
                        <div className="space-y-3">
                            {/* Input Slot 1 */}
                            <div className="bg-purple-950/20 p-2 rounded border border-purple-900/50">
                                <label className="text-[10px] text-purple-500 uppercase font-bold mb-1 block">Layer I: The Scribe (QUL)</label>
                                <div className="flex gap-2">
                                    <input 
                                        className="w-full bg-black/50 border border-purple-800 rounded px-2 py-1 text-xs text-white" 
                                        placeholder="e.g., Init QND Mirror..."
                                        value={simInputs[0]}
                                        onChange={(e) => handleInputChange(0, e.target.value)}
                                    />
                                    <button onClick={() => handleRunSimSlot(0)} className={`p-1.5 rounded border ${activeSims[0] ? 'bg-green-600 border-green-400' : 'bg-purple-900 border-purple-700'}`}>
                                        {activeSims[0] ? <LoaderIcon className="w-3 h-3 animate-spin text-white"/> : <PlayIcon className="w-3 h-3 text-purple-400"/>}
                                    </button>
                                </div>
                            </div>
                             {/* Input Slot 2 */}
                             <div className="bg-cyan-950/20 p-2 rounded border border-cyan-900/50">
                                <label className="text-[10px] text-cyan-500 uppercase font-bold mb-1 block">Layer II: Chronos Engine</label>
                                <div className="flex gap-2">
                                    <input 
                                        className="w-full bg-black/50 border border-cyan-800 rounded px-2 py-1 text-xs text-white" 
                                        placeholder="e.g., Predict MToE Forking..."
                                        value={simInputs[1]}
                                        onChange={(e) => handleInputChange(1, e.target.value)}
                                    />
                                    <button onClick={() => handleRunSimSlot(1)} className={`p-1.5 rounded border ${activeSims[1] ? 'bg-green-600 border-green-400' : 'bg-cyan-900 border-cyan-700'}`}>
                                        {activeSims[1] ? <LoaderIcon className="w-3 h-3 animate-spin text-white"/> : <PlayIcon className="w-3 h-3 text-cyan-400"/>}
                                    </button>
                                </div>
                            </div>
                            {/* Input Slot 3 */}
                            <div className="bg-yellow-950/20 p-2 rounded border border-yellow-900/50">
                                <label className="text-[10px] text-yellow-500 uppercase font-bold mb-1 block">Layer III: PSN Navigator</label>
                                <div className="flex gap-2">
                                    <input 
                                        className="w-full bg-black/50 border border-yellow-800 rounded px-2 py-1 text-xs text-white" 
                                        placeholder="e.g., Scan Anomalies..."
                                        value={simInputs[2]}
                                        onChange={(e) => handleInputChange(2, e.target.value)}
                                    />
                                    <button onClick={() => handleRunSimSlot(2)} className={`p-1.5 rounded border ${activeSims[2] ? 'bg-green-600 border-green-400' : 'bg-yellow-900 border-yellow-700'}`}>
                                        {activeSims[2] ? <LoaderIcon className="w-3 h-3 animate-spin text-white"/> : <PlayIcon className="w-3 h-3 text-yellow-400"/>}
                                    </button>
                                </div>
                            </div>
                        </div>

                        {/* Entanglement Bridge Controls */}
                        <div className="mt-4 pt-4 border-t border-purple-800/50">
                             <h4 className="text-purple-300 font-bold flex items-center justify-between mb-3 text-xs uppercase tracking-widest">
                                <NetworkIcon className="w-4 h-4 mr-2" /> MQ-AGI Integration
                            </h4>
                            <div className="space-y-3">
                                <button 
                                    onClick={() => toggleUniverseToKernel(!universeConnections.kernel)}
                                    className={`w-full p-2 rounded border transition-all flex items-center justify-between ${universeConnections.kernel ? 'bg-blue-900/40 border-blue-400 shadow-[0_0_10px_blue]' : 'bg-black/40 border-gray-700'}`}
                                >
                                    <span className="text-xs font-bold text-white flex items-center gap-2">
                                        <CpuChipIcon className="w-4 h-4 text-blue-400" /> QCOS Kernel Uplink
                                    </span>
                                    <div className={`w-2 h-2 rounded-full ${universeConnections.kernel ? 'bg-blue-400 animate-pulse' : 'bg-gray-600'}`}></div>
                                </button>

                                 <button 
                                    onClick={() => toggleUniverseToAgentQ(!universeConnections.agentQ)}
                                    className={`w-full p-2 rounded border transition-all flex items-center justify-between ${universeConnections.agentQ ? 'bg-purple-900/40 border-purple-400 shadow-[0_0_10px_purple]' : 'bg-black/40 border-gray-700'}`}
                                >
                                    <span className="text-xs font-bold text-white flex items-center gap-2">
                                        <BrainCircuitIcon className="w-4 h-4 text-purple-400" /> AgentQ (QIAI-IPS)
                                    </span>
                                    <div className={`w-2 h-2 rounded-full ${universeConnections.agentQ ? 'bg-purple-400 animate-pulse' : 'bg-gray-600'}`}></div>
                                </button>
                            </div>
                        </div>

                        {/* Cognitive Engines Parallel Link */}
                        <div className="mt-4 pt-4 border-t border-purple-800/50">
                            <h4 className="text-cyan-300 font-bold flex items-center justify-between mb-3 text-xs uppercase tracking-widest">
                                <BrainCircuitIcon className="w-4 h-4 mr-2" /> Cognitive Engines Parallel Link
                            </h4>
                            <div className="grid grid-cols-2 gap-2">
                                {cognitiveEngines.map(engine => (
                                    <div key={engine.id} className={`p-1.5 rounded border text-[9px] flex items-center justify-between ${engine.active ? 'bg-cyan-900/40 border-cyan-400 shadow-[0_0_5px_cyan]' : 'bg-black/40 border-gray-700'}`}>
                                        <span className="text-white font-bold">{engine.id}</span>
                                        <span className="text-cyan-300">{(engine.strength * 100).toFixed(0)}% STR</span>
                                    </div>
                                ))}
                            </div>
                            <button 
                                onClick={runParallelSimulation}
                                className={`mt-2 w-full p-2 rounded border transition-all flex items-center justify-center gap-2 ${parallelSimActive ? 'bg-cyan-600 border-cyan-400 text-white' : 'bg-cyan-900/40 border-cyan-700 text-cyan-200'}`}
                            >
                                {parallelSimActive ? <LoaderIcon className="w-4 h-4 animate-spin" /> : <PlayIcon className="w-4 h-4" />}
                                <span className="text-xs font-bold">Run Parallel Prediction</span>
                            </button>
                        </div>

                        {/* Realtime System Patches */}
                        <div className="mt-4 pt-4 border-t border-purple-800/50">
                            <h4 className="text-emerald-400 font-bold flex items-center justify-between mb-3 text-xs uppercase tracking-widest">
                                <ActivityIcon className="w-4 h-4 mr-2" /> Realtime System Patches
                            </h4>
                            <div className="flex gap-2">
                                <button 
                                    onClick={() => {
                                        addToast("Patching AgentQ Core with latest QUL state...", "info");
                                        setTimeout(() => addToast("AgentQ Core patched successfully.", "success"), 1500);
                                    }}
                                    className="flex-1 p-2 bg-emerald-900/30 border border-emerald-700 rounded text-[10px] font-bold text-emerald-200 hover:bg-emerald-800/50 transition-colors"
                                >
                                    Update AgentQ Core
                                </button>
                                <button 
                                    onClick={() => {
                                        addToast("Patching QCOS Core with latest QUL state...", "info");
                                        setTimeout(() => addToast("QCOS Core patched successfully.", "success"), 1500);
                                    }}
                                    className="flex-1 p-2 bg-emerald-900/30 border border-emerald-700 rounded text-[10px] font-bold text-emerald-200 hover:bg-emerald-800/50 transition-colors"
                                >
                                    Update QCOS Core
                                </button>
                            </div>
                        </div>
                        
                        <div className="mt-auto pt-4 border-t border-purple-800/50">
                             <div className="flex-grow bg-black/60 rounded p-2 text-[9px] font-mono text-purple-100 space-y-1 overflow-y-auto custom-scrollbar h-24">
                                {optLogs.map((log, i) => (
                                    <div key={i} className="flex gap-2 animate-fade-in-right">
                                        <span className="text-purple-800 flex-shrink-0">[{i}]</span>
                                        <span>{log}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                )}

                <div className={`${embedded ? 'h-full' : 'lg:col-span-2'} bg-black/50 border border-purple-800/50 rounded-lg relative overflow-hidden flex items-center justify-center`}>
                    <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-purple-900/10 via-black to-black"></div>
                    
                    {simulationMode === 'universes' && (
                        <div className={`grid ${embedded ? 'grid-cols-4 gap-2 p-2' : 'grid-cols-4 gap-4 p-4'} relative z-10 w-full h-full content-center`}>
                            {universes.map(u => (
                                <div 
                                    key={u.id}
                                    className="aspect-square rounded-full border relative transition-all duration-1000 group"
                                    style={{
                                        borderColor: u.type === 'qnd_mirror' ? `rgba(168, 85, 247, ${u.stability})` : `rgba(6, 182, 212, ${u.stability})`,
                                        transform: `scale(${0.8 + u.stability * 0.4})`,
                                        background: activeSims.some(s => s) ? 'rgba(255,255,255,0.05)' : 'transparent'
                                    }}
                                >
                                    <div className="absolute inset-0 flex items-center justify-center">
                                        <SparklesIcon className={`${embedded ? 'w-3 h-3' : 'w-4 h-4'} ${u.type === 'qnd_mirror' ? 'text-purple-300 animate-spin-slow' : 'text-white opacity-50 animate-pulse'}`} />
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {simulationMode !== 'universes' && (
                        <div className="relative w-full h-full flex flex-col">
                            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full z-10" />
                        </div>
                    )}
                </div>
            </div>
        </div>
    );

    if (embedded) return content;

    return (
        <GlassPanel title={
            <div className="flex items-center justify-between w-full">
                <div className="flex items-center">
                    <GalaxyIcon className="w-5 h-5 mr-2 animate-spin-slow text-purple-400" />
                    <span className="text-purple-100">Grand Universe Simulator</span>
                </div>
                <button onClick={() => setShowSettings(!showSettings)} className="p-1 rounded transition-colors text-purple-500 hover:text-purple-300">
                    <SettingsIcon className="w-4 h-4" />
                </button>
            </div>
        }>
            {content}
        </GlassPanel>
    );
};

export default GrandUniverseSimulator;
