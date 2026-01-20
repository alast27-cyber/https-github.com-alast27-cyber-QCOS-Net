
import React, { useState, useEffect, useCallback } from 'react';
import GlassPanel from './GlassPanel';
import { 
    CodeBracketIcon, PlayIcon, StopIcon, SettingsIcon, 
    Share2Icon, GitBranchIcon, CpuChipIcon, RocketLaunchIcon,
    ArrowRightIcon, FastForwardIcon, SparklesIcon, LoaderIcon, CheckCircle2Icon,
    BrainCircuitIcon, RefreshCwIcon
} from './Icons';
import { useSimulation } from '../context/SimulationContext';
import { useToast } from '../context/ToastContext';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { GoogleGenAI, Type } from '@google/genai';
import { generateContentWithRetry } from '../utils/gemini';

const QuantumLargeLanguageModel: React.FC<{ embedded?: boolean }> = ({ embedded = false }) => {
    const { qllm, toggleQLLM, setQLLMTraining, updateQLLMConfig, toggleQLLMAutoTopology, systemStatus } = useSimulation();
    const { addToast } = useToast();

    // Local UI State
    const [qllmOutputFormat, setQllmOutputFormat] = useState<'probability' | 'optimization' | 'superposition'>('probability');
    const [optimizationResult, setOptimizationResult] = useState<any>(null);
    const [isOptimizing, setIsOptimizing] = useState(false);
    const [qllmLossHistory, setQllmLossHistory] = useState<{step: number, loss: number}[]>([]);
    const [qllmTrainingStep, setQllmTrainingStep] = useState(0);

    // QLLM Training Logic
    useEffect(() => {
        let interval: ReturnType<typeof setInterval>;
        if (qllm.isTraining && qllm.isActive) {
            interval = setInterval(() => {
                setQllmTrainingStep(prev => {
                    const next = prev + 1;
                    const newLoss = Math.max(0.01, (qllm.loss || 2.5) * 0.98 + (Math.random() * 0.1 - 0.05));
                    updateQLLMConfig({ loss: newLoss });
                    setQllmLossHistory(h => [...h, { step: next, loss: newLoss }].slice(-50));
                    return next;
                });
            }, 200);
        }
        return () => clearInterval(interval);
    }, [qllm.isTraining, qllm.isActive, qllm.loss, updateQLLMConfig]);

    const handleTopologyOptimization = useCallback(async () => {
        if (isOptimizing) return;
        setIsOptimizing(true);
        setQllmOutputFormat('optimization');
        
        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
            const systemInstruction = "You are the QCOS Topology Optimizer. Analyze the current OS state and suggest a 12-dimensional holographic configuration. Return ONLY a valid JSON object. Do not include extra text. Ensure high semantic density in your rationale.";
            
            const schema = {
                type: Type.OBJECT,
                properties: {
                    rationale: { type: Type.STRING },
                    layout_vector: { 
                        type: Type.ARRAY, 
                        items: { 
                            type: Type.OBJECT,
                            properties: {
                                panel: { type: Type.STRING },
                                priority: { type: Type.NUMBER },
                                dimensions: { type: Type.STRING }
                            }
                        } 
                    },
                    performance_gain: { type: Type.NUMBER }
                },
                required: ["rationale", "layout_vector", "performance_gain"]
            };

            const response = await generateContentWithRetry(ai, {
                model: 'gemini-3-flash-preview',
                contents: "Execute recursive topology optimization. Seek global minima for AGI Singularity Forge performance.",
                config: { 
                    systemInstruction, 
                    responseMimeType: 'application/json',
                    responseSchema: schema 
                }
            });

            // ROBUST JSON EXTRACTION: Find the JSON block within possible markdown noise
            const jsonText = response.text || '';
            const jsonMatch = jsonText.match(/\{[\s\S]*\}/);
            if (!jsonMatch) throw new Error("Result returned invalid content structure.");
            
            // Safe parse with fallback for trailing chars
            let cleaned = jsonMatch[0];
            const lastBrace = cleaned.lastIndexOf('}');
            if (lastBrace !== -1) {
                cleaned = cleaned.substring(0, lastBrace + 1);
            }

            const data = JSON.parse(cleaned);
            setOptimizationResult(data);
            
            // Auto-Implement Notification
            if (qllm.isAutoTopology) {
                addToast(`Implemented Optimized Vector: ${data.performance_gain}% Synergy Increase`, 'success');
            }
        } catch (e) {
            console.error("Topology Optimization Error:", e);
            addToast("Failed to sync topology optimization vector. Attempting re-alignment...", "error");
        } finally {
            setIsOptimizing(false);
        }
    }, [isOptimizing, qllm.isAutoTopology, addToast]);

    // Auto-Evolution Loop
    useEffect(() => {
        let timer: ReturnType<typeof setTimeout>;
        if (qllm.isAutoTopology && !isOptimizing && qllm.isActive) {
            // Trigger optimization every 15-20 seconds in auto-evolve mode (throttled)
            timer = setTimeout(() => {
                handleTopologyOptimization();
            }, 15000 + Math.random() * 5000);
        }
        return () => clearTimeout(timer);
    }, [qllm.isAutoTopology, isOptimizing, qllm.isActive, handleTopologyOptimization]);

    const content = (
        <div className="flex flex-col h-full gap-4 p-4 overflow-hidden relative">
            <div className="flex justify-between items-start bg-black/30 p-3 rounded-lg border border-purple-900/50">
                <div className="flex gap-4">
                     <div className="flex flex-col">
                        <span className="text-[10px] text-gray-400 uppercase">Context Window</span>
                        <span className="font-mono text-cyan-300 font-bold">{qllm.contextWindow.toLocaleString()} Tok</span>
                     </div>
                     <div className="flex flex-col">
                        <span className="text-[10px] text-gray-400 uppercase">System Boost</span>
                        <span className="font-mono text-green-400 font-bold">{qllm.isActive ? `+${((qllm.efficiencyBoost - 1) * 100).toFixed(0)}%` : '0%'}</span>
                     </div>
                </div>
                {!embedded && (
                    <button 
                        onClick={() => toggleQLLM(!qllm.isActive)}
                        className={`holographic-button px-4 py-1.5 text-xs font-bold rounded flex items-center gap-2 ${qllm.isActive ? 'bg-red-600/20 border-red-500 text-red-300' : 'bg-green-600/20 border-green-500 text-green-200'}`}
                    >
                        <CpuChipIcon className="w-3 h-3" /> {qllm.isActive ? 'Deactivate Core' : 'Activate Core'}
                    </button>
                )}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 h-full min-h-0">
                <div className="col-span-1 bg-black/30 border border-purple-800/50 rounded-lg p-4 flex flex-col gap-4 overflow-y-auto">
                    <h4 className="text-purple-300 font-bold flex items-center border-b border-purple-800 pb-2 text-sm uppercase tracking-widest">
                        <SettingsIcon className="w-4 h-4 mr-2" /> Hyper-Objectives
                    </h4>
                    
                    <div className="space-y-3">
                         <button 
                            onClick={toggleQLLMAutoTopology}
                            disabled={!qllm.isActive}
                            className={`w-full holographic-button py-3 text-[10px] font-black rounded flex items-center justify-center gap-2 transition-all ${qllm.isAutoTopology ? 'bg-purple-600 text-white shadow-[0_0_15px_rgba(168,85,247,0.4)] animate-pulse' : 'bg-blue-900/30 border-blue-500/50 text-blue-200'}`}
                        >
                            {qllm.isAutoTopology ? <RefreshCwIcon className="w-4 h-4 animate-spin" /> : <FastForwardIcon className="w-4 h-4" />}
                            {qllm.isAutoTopology ? 'AUTO-EVOLVE: ACTIVE' : 'AUTO-EVOLVE TOPOLOGY'}
                        </button>

                        <button 
                            onClick={() => setQLLMTraining(!qllm.isTraining)}
                            disabled={!qllm.isActive}
                            className={`w-full holographic-button py-3 text-[10px] font-bold rounded flex items-center justify-center gap-2 ${qllm.isTraining ? 'bg-red-600/30 border-red-500 text-red-200' : 'bg-purple-600/30 border-purple-500 text-purple-200'}`}
                        >
                            {qllm.isTraining ? <StopIcon className="w-4 h-4" /> : <PlayIcon className="w-4 h-4" />}
                            {qllm.isTraining ? 'Halt Weights' : 'Retrain LLM Lattice'}
                        </button>
                    </div>

                    {qllm.isAutoTopology && (
                        <div className="bg-purple-900/20 p-3 rounded border border-purple-500/30 animate-fade-in">
                            <p className="text-[10px] text-purple-200 font-bold mb-1 uppercase tracking-tighter flex items-center gap-2">
                                <SparklesIcon className="w-3 h-3 animate-spin-slow" /> Implementation Queue
                            </p>
                            <p className="text-[9px] text-gray-500 italic leading-relaxed">Continuous recursive topology synthesis enabled. Configurations are implemented upon convergence via QNN evolution bridge.</p>
                        </div>
                    )}

                    <div className="mt-auto pt-4 border-t border-purple-900/30 text-[9px] text-gray-500">
                        <p className="flex items-center gap-2">
                            <CheckCircle2Icon className="w-3 h-3 text-green-500" />
                            Global Entanglement Ready
                        </p>
                    </div>
                </div>

                <div className="col-span-1 lg:col-span-2 flex flex-col gap-4">
                    <div className="flex-grow bg-black/40 rounded border border-purple-800/50 p-4 relative overflow-hidden">
                        {qllmOutputFormat === 'optimization' ? (
                            <div className="h-full flex flex-col animate-fade-in">
                                <h5 className="text-cyan-300 font-bold text-xs uppercase mb-3 flex items-center gap-2">
                                    <RocketLaunchIcon className="w-4 h-4" /> Evolutionary Topology State
                                </h5>
                                {isOptimizing ? (
                                    <div className="flex-grow flex flex-col items-center justify-center">
                                        <div className="relative w-20 h-20 mb-4">
                                            <div className="absolute inset-0 border-2 border-dashed border-cyan-500 rounded-full animate-spin"></div>
                                            <BrainCircuitIcon className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-10 h-10 text-cyan-400 animate-pulse" />
                                        </div>
                                        <p className="text-cyan-500 font-mono text-[10px] animate-pulse">Running Monte Carlo Topology Pass...</p>
                                    </div>
                                ) : optimizationResult ? (
                                    <div className="space-y-4">
                                        <div className="p-3 bg-blue-900/20 border border-blue-500/30 rounded-lg">
                                            <p className="text-[10px] text-blue-400 font-bold uppercase mb-1">AI Rationale</p>
                                            <p className="text-xs text-white leading-relaxed">{optimizationResult.rationale}</p>
                                        </div>
                                        <div className="grid grid-cols-2 gap-3">
                                            {optimizationResult.layout_vector.map((lv: any, i: number) => (
                                                <div key={i} className="bg-black/60 p-2 rounded border border-cyan-900/50 flex justify-between items-center group hover:border-cyan-400 transition-colors">
                                                    <span className="text-[10px] font-bold text-cyan-200">{lv.panel}</span>
                                                    <span className="text-[10px] font-mono text-cyan-600">{lv.dimensions}</span>
                                                </div>
                                            ))}
                                        </div>
                                        <div className="flex justify-between items-center bg-green-900/20 p-2 rounded border border-green-500/30">
                                            <span className="text-[10px] text-green-400 font-bold">Predicted Performance Gain</span>
                                            <span className="text-xl font-black font-mono text-green-300">+{optimizationResult.performance_gain}%</span>
                                        </div>
                                        {qllm.isAutoTopology && (
                                            <div className="flex items-center gap-2 text-[9px] text-green-500 font-black uppercase">
                                                <CheckCircle2Icon className="w-3 h-3" /> Auto-Implementation Success: Config applied to kernel.
                                            </div>
                                        )}
                                    </div>
                                ) : (
                                    <div className="h-full flex flex-col items-center justify-center text-gray-700 italic text-sm gap-2">
                                        <SparklesIcon className="w-8 h-8 opacity-20" />
                                        Awaiting first evolutionary pass...
                                    </div>
                                )}
                            </div>
                        ) : (
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={qllmLossHistory}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(168, 85, 247, 0.1)" vertical={false} />
                                    <XAxis dataKey="step" hide />
                                    <YAxis domain={[0, 3]} hide />
                                    <Tooltip contentStyle={{backgroundColor: 'rgba(0,0,0,0.9)', borderColor: '#a855f7', color: '#fff'}} itemStyle={{fontSize: '10px'}} />
                                    <Line type="monotone" dataKey="loss" stroke="#a855f7" strokeWidth={2} dot={false} isAnimationActive={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );

    if (embedded) return content;

    return (
        <GlassPanel title={
            <div className="flex items-center justify-between w-full">
                <div className="flex items-center">
                    <CodeBracketIcon className="w-5 h-5 mr-2 text-purple-400" />
                    <span>Quantum Large Language Model (QLLM)</span>
                </div>
                <button 
                    onClick={() => updateQLLMConfig({ efficiencyBoost: qllm.efficiencyBoost > 1 ? 1 : 4.5 })}
                    className={`text-[10px] px-2 py-0.5 rounded border transition-all duration-500 ${qllm.efficiencyBoost > 1 ? 'bg-cyan-600 text-white shadow-[0_0_15px_cyan] border-white' : 'bg-gray-800 text-gray-500 border-gray-700'}`}
                >
                    <Share2Icon className={`w-3 h-3 mr-1 inline ${qllm.efficiencyBoost > 1 ? 'animate-spin-slow' : ''}`} />
                    GLOBAL ENTANGLEMENT
                </button>
            </div>
        }>
            {content}
        </GlassPanel>
    );
};

export default QuantumLargeLanguageModel;
