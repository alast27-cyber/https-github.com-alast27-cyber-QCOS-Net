
import React, { useState, useEffect, useRef, useCallback } from 'react';
import GlassPanel from './GlassPanel';
import SyntaxHighlighter from './SyntaxHighlighter';
import { 
    BrainCircuitIcon, ActivityIcon, WifiIcon, SettingsIcon, 
    PlayIcon, StopIcon, ZapIcon, LockIcon, CheckCircle2Icon,
    RefreshCwIcon, CpuChipIcon, SparklesIcon, Share2Icon,
    ToggleLeftIcon, ToggleRightIcon, ChartBarIcon, CodeBracketIcon,
    ServerStackIcon, FastForwardIcon, ArrowRightIcon, LinkIcon,
    TruckIcon, HomeIcon, LoaderIcon, UsersIcon, HeartIcon, SearchIcon, ClipboardIcon,
    FingerPrintIcon, ServerCogIcon, AlertTriangleIcon, MessageSquareIcon, LayersIcon,
    GridIcon, RadioIcon
} from './Icons';
import { useSimulation } from '../context/SimulationContext';
import { useToast } from '../context/ToastContext';
import { ResponsiveContainer, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Legend, BarChart, Bar, XAxis, YAxis, Tooltip, AreaChart, Area } from 'recharts';
import { GoogleGenAI, Type } from "@google/genai";
import { generateContentWithRetry } from '../utils/gemini';

type HardwareType = 'EEG' | 'fNIRS' | 'INVASIVE_LACE' | 'RF_WIFI' | 'BLUETOOTH_6' | 'QUANTUM_ENTANGLEMENT';
type ConnectionState = 'DISCONNECTED' | 'SCANNING' | 'CALIBRATING' | 'CONNECTED';
type PanelMode = 'LIVE' | 'SIMULATOR' | 'PROGRAM' | 'CLINICAL' | 'QUANTUM_OPS';
type VisualMode = 'WAVEFORM' | 'MANIFOLD';

interface NeuralScript {
    id: string;
    name: string;
    category: 'Remote Control' | 'Automation' | 'Security';
    targetType: string;
    complexity: 'Low' | 'Medium' | 'High';
    code: string;
}

interface Patient {
    id: string;
    name: string;
    condition: string;
    neuralStability: number;
    bioSignature: string;
    status: 'Stable' | 'Critical' | 'Monitoring';
}

const MOCK_PATIENTS: Patient[] = [
    { id: 'PT-1092', name: 'Subject Alpha', condition: 'Acute Anxiety / Hyper-Arousal', neuralStability: 45, bioSignature: '0x7F...9A', status: 'Critical' },
    { id: 'PT-3321', name: 'Subject Beta', condition: 'Depressive Synaptic Dormancy', neuralStability: 62, bioSignature: '0xA1...4B', status: 'Monitoring' },
    { id: 'PT-0045', name: 'Subject Gamma', condition: 'PTSD - Memory Loop', neuralStability: 30, bioSignature: '0xC4...1D', status: 'Critical' },
    { id: 'PT-8812', name: 'Subject Delta', condition: 'Baseline / Healthy Control', neuralStability: 98, bioSignature: '0x00...00', status: 'Stable' },
];

const INITIAL_NEURAL_SCRIPTS: NeuralScript[] = [
    {
        id: 'rc_drone',
        name: 'Teleoperation: UAV-X',
        category: 'Remote Control',
        targetType: 'Drone Swarm',
        complexity: 'High',
        code: `// Neural-Motor Mapping for UAV\nIMPORT MOTOR_CORTEX_MAP;\nTARGET = "UAV-X-77";\n\nFUNCTION ON_THOUGHT(intent_vector) {\n  pitch = intent_vector.frontal_lobe.y;\n  yaw = intent_vector.motor_cortex.z;\n  \n  // Entangle command byte\n  Q_CMD = ENTANGLE(pitch, yaw);\n  TRANSMIT(TARGET, Q_CMD);\n}`
    },
    {
        id: 'bio_auth',
        name: 'Quantum Bio-Key',
        category: 'Security',
        targetType: 'Secure Vault',
        complexity: 'Medium',
        code: `// Unique Bio-Signature Hash\nCONST BIO_SEED = MEASURE(HIPPOCAMPUS.pattern);\nKEY = SHA3_QUANTUM(BIO_SEED);\n\nIF (VERIFY(KEY, TARGET.lock)) {\n    UNLOCK();\n} ELSE {\n    ALERT("Unauthorized Neural Pattern");\n}`
    }
];

// --- Sub-component: Quantum Operations Interface ---
const QuantumOpsInterface: React.FC = () => {
    // State for animations
    const [pulseData, setPulseData] = useState<{t: number, v: number}[]>([]);
    const vectorRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        let animationFrameId: number;
        const animate = () => {
            if (vectorRef.current) {
                vectorRef.current.style.transform = `rotate(${Date.now()/50}deg) rotateX(45deg)`;
            }
            animationFrameId = requestAnimationFrame(animate);
        };
        animate();
        return () => cancelAnimationFrame(animationFrameId);
    }, []);
    const [crosstalk, setCrosstalk] = useState<number[][]>([]);
    const [calibration, setCalibration] = useState<number[]>([0,0,0,0]);
    const [qstFidelity, setQstFidelity] = useState(0.85);

    // Init data
    useEffect(() => {
        // Init Crosstalk
        const initialMatrix = Array(8).fill(0).map(() => Array(8).fill(0).map(() => Math.random() * 0.2));
        setCrosstalk(initialMatrix);
    }, []);

    // Simulation Loop
    useEffect(() => {
        const interval = setInterval(() => {
            // Pulse Update
            const t = Date.now() / 1000;
            const newPulse = Array.from({length: 20}, (_, i) => ({
                t: i,
                v: Math.sin(t * 5 + i * 0.5) * Math.exp(-i * 0.1) + (Math.random() * 0.1)
            }));
            setPulseData(newPulse);

            // Crosstalk Update (RL Agent reducing noise)
            setCrosstalk(prev => prev.map(row => row.map(val => Math.max(0.01, val * 0.98 + (Math.random() - 0.5) * 0.01))));

            // Calibration Progress (MARL)
            setCalibration(prev => prev.map(c => c >= 100 ? 0 : c + Math.random() * 5));

            // QST Fidelity
            setQstFidelity(f => Math.min(0.9999, Math.max(0.8, f + (Math.random() - 0.5) * 0.001)));

        }, 100);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 grid-rows-2 gap-4 h-full p-2 overflow-hidden animate-fade-in text-xs">
            
            {/* 1. Low-Latency FPGA & Pulse Shaping */}
            <div className="col-span-1 lg:col-span-2 bg-black/40 border border-cyan-900/50 rounded-xl p-3 flex flex-col relative overflow-hidden">
                <div className="flex justify-between items-center mb-2 border-b border-cyan-800/30 pb-2">
                    <div className="flex items-center gap-2 text-cyan-300 font-bold">
                        <CpuChipIcon className="w-4 h-4" /> 
                        <span>FPGA Inference & Pulse Optimization</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="bg-green-900/30 text-green-400 px-2 py-0.5 rounded border border-green-800 text-[9px] font-mono">DMA: ACTIVE</span>
                        <span className="bg-blue-900/30 text-blue-400 px-2 py-0.5 rounded border border-blue-800 text-[9px] font-mono">LATENCY: 12ns</span>
                    </div>
                </div>
                <div className="flex-grow flex gap-4 min-h-0">
                    <div className="w-1/3 flex flex-col gap-2">
                         <div className="bg-cyan-950/20 p-2 rounded border border-cyan-900/50">
                             <p className="text-[9px] text-gray-400 uppercase">Optimization Strategy</p>
                             <p className="text-cyan-200 font-bold">GRAPE / CRAB Hybrid</p>
                         </div>
                         <div className="bg-purple-950/20 p-2 rounded border border-purple-900/50">
                             <p className="text-[9px] text-gray-400 uppercase">Gradient Descent</p>
                             <div className="w-full h-1 bg-gray-800 rounded mt-1 overflow-hidden">
                                 <div className="h-full bg-purple-500 animate-pulse" style={{width: '85%'}}></div>
                             </div>
                         </div>
                    </div>
                    <div className="w-2/3 relative bg-black/20 rounded border border-cyan-900/30 overflow-hidden">
                         <ResponsiveContainer width="100%" height="100%">
                             <AreaChart data={pulseData}>
                                 <Area type="monotone" dataKey="v" stroke="#22d3ee" fill="rgba(34, 211, 238, 0.1)" strokeWidth={2} isAnimationActive={false} />
                             </AreaChart>
                         </ResponsiveContainer>
                         <p className="absolute top-1 left-1 text-[8px] text-cyan-600 font-mono">CONTROL_FIELD_AMPLITUDE</p>
                    </div>
                </div>
            </div>

            {/* 3. Crosstalk Mitigation Matrix */}
            <div className="col-span-1 row-span-2 bg-black/40 border border-red-900/30 rounded-xl p-3 flex flex-col">
                <div className="flex items-center gap-2 text-red-300 font-bold mb-2 border-b border-red-900/30 pb-2">
                    <GridIcon className="w-4 h-4" />
                    <span>Crosstalk Matrix (ZZ)</span>
                </div>
                <div className="flex-grow grid grid-cols-8 gap-1 auto-rows-fr aspect-square">
                    {crosstalk.map((row, r) => row.map((val, c) => (
                        <div 
                            key={`${r}-${c}`} 
                            className="rounded-sm transition-colors duration-300"
                            style={{ 
                                backgroundColor: `rgba(248, 113, 113, ${val * 3})`,
                                border: r===c ? '1px solid rgba(255,255,255,0.2)' : 'none'
                            }}
                            title={`Q${r}-Q${c}: ${(val*100).toFixed(1)}%`}
                        ></div>
                    )))}
                </div>
                <div className="mt-2 text-[9px] text-gray-400 font-mono">
                    <p className="flex justify-between"><span>RL AGENT:</span> <span className="text-green-400">COMPENSATING</span></p>
                    <p className="flex justify-between"><span>PARASITIC COUPLING:</span> <span className="text-yellow-400">LOW</span></p>
                </div>
            </div>

            {/* 4. Stochastic Hamiltonian & Multi-Agent */}
            <div className="col-span-1 bg-black/40 border border-orange-900/30 rounded-xl p-3 flex flex-col gap-2">
                <div className="flex items-center gap-2 text-orange-300 font-bold mb-1">
                    <ServerStackIcon className="w-4 h-4" />
                    <span>Shadow Sim & Calibration</span>
                </div>
                <div className="space-y-2 overflow-y-auto custom-scrollbar flex-grow">
                    {calibration.map((prog, i) => (
                        <div key={i} className="flex items-center gap-2">
                             <div className="w-16 text-[9px] text-gray-400">Agent-0{i+1}</div>
                             <div className="flex-grow h-1.5 bg-gray-800 rounded-full overflow-hidden">
                                 <div className="h-full bg-orange-500 transition-all duration-300" style={{width: `${prog}%`}}></div>
                             </div>
                             <div className="w-8 text-[9px] text-orange-400 text-right">{prog.toFixed(0)}%</div>
                        </div>
                    ))}
                </div>
                <div className="bg-orange-950/20 p-2 rounded border border-orange-900/40 text-[9px] font-mono text-orange-200 truncate">
                    H = Σωᵢσᶻᵢ + ΣJᵢⱼ(σ⁺ᵢσ⁻ⱼ + σ⁻ᵢσ⁺ⱼ)
                </div>
            </div>

            {/* 6. Quantum State Tomography */}
            <div className="col-span-1 bg-black/40 border border-purple-900/30 rounded-xl p-3 flex flex-col relative overflow-hidden">
                <div className="flex items-center gap-2 text-purple-300 font-bold mb-2 z-10">
                    <ActivityIcon className="w-4 h-4" />
                    <span>State Tomography (QST)</span>
                </div>
                
                {/* Visualizer Placeholder */}
                <div className="flex-grow relative flex items-center justify-center">
                    <div className="absolute inset-0 bg-purple-500/5 rounded-full blur-xl animate-pulse"></div>
                    {/* Simplified Bloch Sphere Representation */}
                    <div className="w-24 h-24 rounded-full border border-purple-500/30 relative flex items-center justify-center animate-spin-slow">
                        <div className="w-full h-px bg-purple-500/30 absolute"></div>
                        <div className="h-full w-px bg-purple-500/30 absolute"></div>
                        <div className="w-16 h-16 rounded-full border border-purple-400/20"></div>
                        {/* State Vector */}
                        <div ref={vectorRef} className="absolute w-1 h-12 bg-gradient-to-t from-transparent to-white origin-bottom bottom-1/2 left-1/2 -ml-0.5"></div>
                    </div>
                </div>

                <div className="mt-2 flex justify-between items-end z-10">
                    <div>
                        <p className="text-[9px] text-gray-500 uppercase">Est. Purity</p>
                        <p className="text-lg font-mono text-white">{(qstFidelity * 100).toFixed(2)}%</p>
                    </div>
                    <div className="text-right">
                         <p className="text-[9px] text-gray-500 uppercase">Inference</p>
                         <p className="text-xs font-mono text-purple-400">Bayesian NN</p>
                    </div>
                </div>
            </div>

        </div>
    );
};

// --- Sub-component: Clinical Intelligence ---
const ClinicalBridgeInterface: React.FC = () => {
    // ... (Existing Clinical Code - omitted for brevity but logic remains same if not changed)
    const { addToast } = useToast();
    const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [diagnosticReport, setDiagnosticReport] = useState<string | null>(null);
    const [telemetry, setTelemetry] = useState({ dopamine: 50, serotonin: 50, cortisol: 50, coherence: 85 });

    const handleAnalyze = async () => {
        if (!selectedPatient) return;
        setIsAnalyzing(true);
        setDiagnosticReport(null);

        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
            const prompt = `Act as a Neuro-Diagnostic AI. Analyze the following patient data and telemetry:
            Patient: ${selectedPatient.name}
            Condition: ${selectedPatient.condition}
            Stability: ${selectedPatient.neuralStability}%
            Cortisol: ${telemetry.cortisol}
            Dopamine: ${telemetry.dopamine}
            Coherence: ${telemetry.coherence}%
            Provide a professional, concise clinical assessment and suggested neural-tuning parameters.`;

            const response = await generateContentWithRetry(ai, {
                model: 'gemini-3-flash-preview',
                contents: prompt
            });

            setDiagnosticReport(response.text || "Analysis failed to converge.");
            addToast("Diagnostic Report Synthesized", "success");
        } catch (e) {
            addToast("Failed to connect to Neural Diagnostic core.", "error");
        } finally {
            setIsAnalyzing(false);
        }
    };

    return (
        <div className="flex flex-col md:flex-row gap-4 h-full p-2 overflow-hidden animate-fade-in">
            <div className="w-full md:w-1/3 bg-black/30 border border-cyan-900/50 rounded-lg flex flex-col overflow-hidden">
                <div className="p-3 border-b border-cyan-900/50 bg-cyan-950/20">
                    <h3 className="text-xs font-black text-cyan-300 flex items-center gap-2 uppercase tracking-widest"><UsersIcon className="w-4 h-4" /> BCI Registry</h3>
                </div>
                <div className="flex-grow overflow-y-auto p-2 space-y-2 custom-scrollbar">
                    {MOCK_PATIENTS.map(patient => (
                        <button 
                            key={patient.id} 
                            onClick={() => { setSelectedPatient(patient); setDiagnosticReport(null); }} 
                            className={`w-full p-3 rounded-lg border text-left transition-all group ${selectedPatient?.id === patient.id ? 'bg-cyan-900/40 border-cyan-500 shadow-[0_0_15px_cyan]' : 'bg-black/40 border-transparent hover:bg-white/5'}`}
                        >
                            <div className="flex justify-between items-start mb-1">
                                <span className="text-[10px] font-black text-white uppercase tracking-tighter">{patient.name}</span>
                                <span className={`text-[8px] px-1.5 rounded-full border ${patient.status === 'Critical' ? 'bg-red-900/40 border-red-500 text-red-400' : 'bg-green-900/40 border-green-500 text-green-400'}`}>{patient.status}</span>
                            </div>
                            <p className="text-[9px] text-gray-400 truncate italic">"{patient.condition}"</p>
                        </button>
                    ))}
                </div>
            </div>

            <div className="flex-grow flex flex-col gap-4 min-h-0">
                {selectedPatient ? (
                    <div className="flex flex-col h-full gap-4">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 flex-shrink-0">
                            {[
                                { label: 'Stability', val: `${selectedPatient.neuralStability}%`, color: 'text-cyan-400' },
                                { label: 'Signature', val: selectedPatient.bioSignature, color: 'text-purple-400' },
                                { label: 'Coherence', val: '0.88v', color: 'text-green-400' },
                                { label: 'Node', val: 'DQN-12', color: 'text-amber-400' }
                            ].map((stat, i) => (
                                <div key={i} className="bg-black/40 border border-white/10 p-2 rounded-lg text-center">
                                    <p className="text-[8px] text-gray-500 uppercase font-black">{stat.label}</p>
                                    <p className={`text-xs font-mono font-bold ${stat.color}`}>{stat.val}</p>
                                </div>
                            ))}
                        </div>

                        <div className="flex-grow bg-black/40 border border-cyan-900/30 rounded-xl p-4 overflow-y-auto custom-scrollbar relative">
                            {isAnalyzing ? (
                                <div className="flex flex-col items-center justify-center h-full">
                                    <LoaderIcon className="w-12 h-12 text-cyan-500 animate-spin" />
                                    <p className="text-cyan-400 text-xs font-mono animate-pulse mt-4">RUNNING NEURAL DIAGNOSTICS...</p>
                                </div>
                            ) : diagnosticReport ? (
                                <div className="animate-fade-in space-y-4">
                                    <h4 className="text-sm font-black text-white uppercase flex items-center gap-2 border-b border-white/10 pb-2">
                                        <HeartIcon className="w-4 h-4 text-red-500" /> Diagnostic Assessment
                                    </h4>
                                    <p className="text-xs text-cyan-100 leading-relaxed font-mono whitespace-pre-wrap">{diagnosticReport}</p>
                                </div>
                            ) : (
                                <div className="flex flex-col items-center justify-center h-full text-center space-y-4 opacity-50">
                                    <BrainCircuitIcon className="w-16 h-16 text-cyan-900" />
                                    <p className="text-xs text-cyan-700 max-w-xs">Awaiting neural load analysis. Click the button below to initiate Gemini-powered clinical assessment.</p>
                                </div>
                            )}
                        </div>

                        <button 
                            onClick={handleAnalyze} 
                            disabled={isAnalyzing}
                            className="w-full py-3 bg-cyan-600/20 border-2 border-cyan-500 text-cyan-200 font-black uppercase tracking-widest rounded-xl hover:bg-cyan-600/40 transition-all flex items-center justify-center gap-3 disabled:opacity-50"
                        >
                            <ActivityIcon className="w-5 h-5" /> Start Neural Analysis
                        </button>
                    </div>
                ) : (
                    <div className="flex flex-col items-center justify-center h-full text-center opacity-30">
                        <UsersIcon className="w-20 h-20 text-cyan-900 mb-4" />
                        <p className="text-sm font-black uppercase tracking-[0.2em]">Select BCI Subject to proceed</p>
                    </div>
                )}
            </div>
        </div>
    );
};

// --- Sub-component: Neural Script Editor ---
const EntanglementScriptForge: React.FC = () => {
    // ... (Existing Script Forge Code)
    const { addToast } = useToast();
    const [scripts, setScripts] = useState<NeuralScript[]>(INITIAL_NEURAL_SCRIPTS);
    const [selectedScript, setSelectedScript] = useState<NeuralScript>(INITIAL_NEURAL_SCRIPTS[0]);
    const [isGenerating, setIsGenerating] = useState(false);
    const [prompt, setPrompt] = useState('');

    const handleGenerateScript = async () => {
        if (!prompt.trim()) return;
        setIsGenerating(true);
        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
            const response = await generateContentWithRetry(ai, {
                model: 'gemini-3-pro-preview',
                contents: `Create a BCI Neural Script (Q-Lang style) for: ${prompt}. Focus on synaptic mapping and quantum entanglement triggers.`
            });

            const newCode = response.text || '// Generation failed';
            const newScript: NeuralScript = {
                id: `script-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                name: prompt.substring(0, 20) + '...',
                category: 'Remote Control',
                targetType: 'Custom Node',
                complexity: 'Medium',
                code: newCode
            };

            setScripts(prev => [newScript, ...prev]);
            setSelectedScript(newScript);
            setPrompt('');
            addToast("Neural Logic Synthesized", "success");
        } catch (e) {
            addToast("Agent Q failed to compile neural intent.", "error");
        } finally {
            setIsGenerating(false);
        }
    };

    return (
        <div className="flex flex-col md:flex-row gap-4 h-full p-2 overflow-hidden animate-fade-in">
            <div className="w-full md:w-1/3 bg-black/30 border border-purple-900/50 rounded-lg flex flex-col overflow-hidden">
                <div className="p-3 border-b border-purple-900/50 bg-purple-950/20">
                    <h3 className="text-xs font-black text-purple-300 flex items-center gap-2 uppercase tracking-widest"><CodeBracketIcon className="w-4 h-4" /> Script Forge</h3>
                </div>
                <div className="flex-grow overflow-y-auto p-2 space-y-2 custom-scrollbar">
                    {scripts.map(script => (
                        <button key={script.id} onClick={() => setSelectedScript(script)} className={`w-full p-3 rounded-lg border text-left transition-all ${selectedScript.id === script.id ? 'bg-purple-900/40 border-purple-500 shadow-[0_0_15px_rgba(168,85,247,0.4)]' : 'bg-black/40 border-transparent hover:bg-white/5'}`}>
                            <span className="text-[10px] font-black text-white uppercase">{script.name}</span>
                        </button>
                    ))}
                </div>
            </div>
            <div className="flex-grow flex flex-col gap-4 min-h-0">
                <div className="flex-grow relative flex flex-col bg-black/40 border border-purple-800/30 rounded-xl overflow-hidden">
                    <div className="p-2 bg-purple-950/40 border-b border-purple-800/30 flex justify-between items-center">
                        <span className="text-[10px] font-mono text-purple-300 uppercase">{selectedScript.name}.q</span>
                        <span className="text-[8px] text-gray-500">READ_ONLY</span>
                    </div>
                    <div className="flex-grow p-4 overflow-auto custom-scrollbar">
                        <SyntaxHighlighter code={selectedScript.code} language="q-lang" />
                    </div>
                </div>

                <div className="p-4 bg-black/60 border border-cyan-800/50 rounded-xl flex gap-3">
                    <input 
                        type="text" 
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="Describe a new neural behavior..."
                        className="flex-grow bg-black/50 border border-cyan-700 text-white rounded-lg px-4 py-2 text-sm outline-none focus:border-cyan-400"
                        onKeyDown={(e) => e.key === 'Enter' && handleGenerateScript()}
                    />
                    <button 
                        onClick={handleGenerateScript}
                        disabled={isGenerating || !prompt.trim()}
                        className="px-6 py-2 bg-purple-600/30 border border-purple-500 text-purple-200 font-bold rounded-lg flex items-center gap-2 hover:bg-purple-600/50 disabled:opacity-50"
                    >
                        {isGenerating ? <LoaderIcon className="w-4 h-4 animate-spin"/> : <SparklesIcon className="w-4 h-4"/>}
                        Forge
                    </button>
                </div>
            </div>
        </div>
    );
};

// --- Main Component ---
const NeuralProgrammingPanel: React.FC = () => {
    const { neuralInterface, connectNeuralInterface, disconnectNeuralInterface } = useSimulation();
    const { addToast } = useToast();
    const [mode, setMode] = useState<PanelMode>('LIVE');
    const [visualMode, setVisualMode] = useState<VisualMode>('WAVEFORM');
    const [hardware, setHardware] = useState<HardwareType>('EEG');
    const [connectionState, setConnectionState] = useState<ConnectionState>('DISCONNECTED');
    const [isDecoding, setIsDecoding] = useState(false);
    const [decodedIntent, setDecodedIntent] = useState<string | null>(null);
    
    // Optimization State
    const [entropy, setEntropy] = useState(0.5);
    const [safetyCap, setSafetyCap] = useState(true);
    const [fidelityThreshold, setFidelityThreshold] = useState(0.85);
    const [liveFidelity, setLiveFidelity] = useState(0.99);
    const [coherenceT2, setCoherenceT2] = useState(120); // microseconds
    
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const manifoldNodes = useRef<{x: number, y: number, z: number, w: number}[]>([]);

    const handleConnect = () => {
        setConnectionState('SCANNING');
        setTimeout(() => {
            setConnectionState('CALIBRATING');
            setTimeout(() => {
                connectNeuralInterface(hardware);
                setConnectionState('CONNECTED');
                addToast('Neural Interface Hardware Connected', 'success');
            }, 2000);
        }, 1500);
    };

    const handleDecodeIntent = async () => {
        if (connectionState !== 'CONNECTED') return;
        setIsDecoding(true);
        setDecodedIntent(null);

        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
            // Simulate neural data extraction
            const mockWaveform = Array.from({length: 10}, () => Math.random().toFixed(4));
            const prompt = `Translate this real-time neural waveform into a human intent: [${mockWaveform.join(', ')}]. 
            Connection Hardware: ${hardware}. 
            Identify the primary cognitive intent (e.g., 'Moving robotic arm left', 'Increasing system focus', 'Initiating encrypted logout').`;

            const response = await generateContentWithRetry(ai, {
                model: 'gemini-3-flash-preview',
                contents: prompt
            });

            setDecodedIntent(response.text || "Indeterminate signal.");
            addToast("Neural Intent Decoded", "info");
        } catch (e) {
            addToast("Failed to resolve neural harmonics.", "error");
        } finally {
            setIsDecoding(false);
        }
    };

    // Initialize Manifold Nodes
    useEffect(() => {
        if (manifoldNodes.current.length === 0) {
            for(let i=0; i<60; i++) {
                manifoldNodes.current.push({
                    x: (Math.random() - 0.5) * 2,
                    y: (Math.random() - 0.5) * 2,
                    z: (Math.random() - 0.5) * 2,
                    w: Math.random() // Synaptic Weight
                });
            }
        }
    }, []);

    // Physics Loop for Metrics
    useEffect(() => {
        if (connectionState !== 'CONNECTED') return;
        const interval = setInterval(() => {
            // Update T2 (fluctuate slightly)
            setCoherenceT2(prev => Math.max(10, prev + (Math.random() - 0.5) * 5));
            
            // Update Fidelity
            setLiveFidelity(prev => {
                const noise = (Math.random() - 0.5) * (safetyCap ? 0.01 : 0.05);
                const drift = entropy > 0.8 ? -0.01 : 0;
                return Math.max(0, Math.min(1, prev + noise + drift));
            });
        }, 800);
        return () => clearInterval(interval);
    }, [connectionState, safetyCap, entropy]);

    // Canvas Render Loop
    useEffect(() => {
        if (mode !== 'LIVE' || connectionState !== 'CONNECTED') return;
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        
        let animationFrameId: number;
        let t = 0;

        const render = () => {
            if (canvas.parentElement) {
                canvas.width = canvas.parentElement.clientWidth;
                canvas.height = canvas.parentElement.clientHeight;
            }
            const w = canvas.width;
            const h = canvas.height;
            const cx = w / 2;
            const cy = h / 2;
            t += 0.01 + (entropy * 0.05); // Speed based on entropy

            ctx.clearRect(0, 0, w, h);

            if (visualMode === 'WAVEFORM') {
                // Background grid
                ctx.strokeStyle = 'rgba(6, 182, 212, 0.05)';
                ctx.lineWidth = 1;
                for(let i=0; i<w; i+=40) { ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, h); ctx.stroke(); }
                for(let i=0; i<h; i+=40) { ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(w, i); ctx.stroke(); }

                // Waves
                const colors = ['#22d3ee', '#a855f7', '#14b8a6'];
                colors.forEach((color, i) => {
                    ctx.beginPath();
                    ctx.strokeStyle = color;
                    ctx.lineWidth = i === 0 ? 3 : 1;
                    ctx.globalAlpha = i === 0 ? 1 : 0.4;
                    if(i === 0) { ctx.shadowBlur = 15; ctx.shadowColor = color; }

                    for (let x = 0; x < w; x++) {
                        const freq = (hardware === 'INVASIVE_LACE' ? 0.2 : 0.05) * (1 + i * 0.5);
                        const amp = 40 / (i + 1);
                        const y = cy + Math.sin(x * freq + t * 5 + i) * amp + (Math.random() - 0.5) * 2;
                        ctx.lineTo(x, y);
                    }
                    ctx.stroke();
                    ctx.shadowBlur = 0;
                });
            } else {
                // MANIFOLD 4D PROJECTION
                const fov = 300;
                
                // Sort nodes by depth for basic occlusion logic
                const projectedNodes = manifoldNodes.current.map(node => {
                    // Rotate Y
                    let x = node.x * Math.cos(t) - node.z * Math.sin(t);
                    let z = node.z * Math.cos(t) + node.x * Math.sin(t);
                    // Rotate X
                    let y = node.y * Math.cos(t*0.5) - z * Math.sin(t*0.5);
                    z = z * Math.cos(t*0.5) + node.y * Math.sin(t*0.5);
                    
                    const scale = fov / (fov + (z * 100));
                    return { x: x * 100 * scale + cx, y: y * 100 * scale + cy, scale, w: node.w };
                });

                // Connections
                ctx.lineWidth = 0.5;
                ctx.strokeStyle = 'rgba(168, 85, 247, 0.2)';
                projectedNodes.forEach((n1, i) => {
                    projectedNodes.forEach((n2, j) => {
                        if (i !== j) {
                            const dist = Math.sqrt(Math.pow(n1.x-n2.x, 2) + Math.pow(n1.y-n2.y, 2));
                            if (dist < 60) {
                                ctx.beginPath();
                                ctx.moveTo(n1.x, n1.y);
                                ctx.lineTo(n2.x, n2.y);
                                ctx.stroke();
                            }
                        }
                    });
                });

                // Nodes (Heatmap Color)
                projectedNodes.forEach(n => {
                    // Heatmap: Blue -> Cyan -> Green -> Yellow -> Red
                    let color = '#22d3ee';
                    if (n.w > 0.8) color = '#ef4444';
                    else if (n.w > 0.6) color = '#eab308';
                    else if (n.w > 0.4) color = '#22c55e';
                    else if (n.w > 0.2) color = '#06b6d4';
                    else color = '#3b82f6';

                    ctx.beginPath();
                    ctx.fillStyle = color;
                    ctx.shadowColor = color;
                    ctx.shadowBlur = 10 * n.w;
                    ctx.arc(n.x, n.y, 3 * n.scale * (0.5 + n.w), 0, Math.PI * 2);
                    ctx.fill();
                    ctx.shadowBlur = 0;
                    
                    // Jitter weights slightly
                    manifoldNodes.current.forEach(node => {
                        if (Math.random() > 0.95) node.w = Math.max(0, Math.min(1, node.w + (Math.random()-0.5)*0.1));
                    });
                });
            }

            animationFrameId = requestAnimationFrame(render);
        };
        render();
        return () => cancelAnimationFrame(animationFrameId);
    }, [mode, connectionState, hardware, visualMode, entropy]);

    return (
        <GlassPanel title={
            <div className="flex items-center justify-between w-full pr-2">
                <div className="flex items-center gap-2">
                    <BrainCircuitIcon className="w-5 h-5 text-cyan-400 animate-pulse" />
                    <span className="tracking-[0.2em] uppercase font-black text-sm">Neural Interface Platform</span>
                </div>
                <div className="flex items-center gap-4">
                    <div className="flex bg-black/60 rounded-lg p-1 border border-cyan-900/50 shadow-inner">
                        <button onClick={() => setMode('LIVE')} className={`px-4 py-1 text-[10px] font-black uppercase rounded-md transition-all flex items-center gap-2 ${mode === 'LIVE' ? 'bg-cyan-700 text-white shadow-lg shadow-cyan-900' : 'text-cyan-600 hover:text-cyan-300'}`}><ActivityIcon className="w-3.5 h-3.5" /> Live</button>
                        <button onClick={() => setMode('PROGRAM')} className={`px-4 py-1 text-[10px] font-black uppercase rounded-md transition-all flex items-center gap-2 ${mode === 'PROGRAM' ? 'bg-purple-700 text-white shadow-lg shadow-purple-900' : 'text-purple-600 hover:text-purple-300'}`}><CodeBracketIcon className="w-3.5 h-3.5" /> Forge</button>
                        <button onClick={() => setMode('CLINICAL')} className={`px-4 py-1 text-[10px] font-black uppercase rounded-md transition-all flex items-center gap-2 ${mode === 'CLINICAL' ? 'bg-teal-700 text-white shadow-lg shadow-teal-900' : 'text-teal-600 hover:text-teal-300'}`}><HeartIcon className="w-3.5 h-3.5" /> Clinical</button>
                        <button onClick={() => setMode('QUANTUM_OPS')} className={`px-4 py-1 text-[10px] font-black uppercase rounded-md transition-all flex items-center gap-2 ${mode === 'QUANTUM_OPS' ? 'bg-orange-700 text-white shadow-lg shadow-orange-900' : 'text-orange-600 hover:text-orange-300'}`}><ZapIcon className="w-3.5 h-3.5" /> Q-Ops</button>
                    </div>
                </div>
            </div>
        }>
            <div className="h-full relative overflow-hidden bg-black/20">
                {mode === 'LIVE' ? (
                    <div className="flex flex-col h-full gap-4 p-4 overflow-hidden animate-fade-in">
                        
                        {/* Hardware Connection Header */}
                        <div className="flex flex-col md:flex-row gap-4 bg-black/50 p-4 rounded-2xl border border-cyan-800/50 shadow-xl relative overflow-hidden">
                             <div className="absolute inset-0 bg-cyan-500/5 opacity-50 pointer-events-none"></div>
                             <div className="flex-grow relative z-10">
                                <label className="text-[10px] text-cyan-500 uppercase font-black mb-2 block tracking-widest">Hardware Substrate</label>
                                <div className="flex gap-2">
                                    <select value={hardware} onChange={(e) => setHardware(e.target.value as HardwareType)} disabled={connectionState !== 'DISCONNECTED'} className="flex-grow bg-black/60 border-2 border-cyan-900 text-white text-xs font-bold rounded-xl px-4 py-2.5 outline-none focus:border-cyan-400 disabled:opacity-50 transition-all">
                                        <option value="EEG">EEG-CAP (32-CH NON-INVASIVE)</option>
                                        <option value="INVASIVE_LACE">Q-LACE (INVASIVE HI-FIDELITY)</option>
                                        <option value="QUANTUM_ENTANGLEMENT">NON-LOCAL ENTANGLEMENT BRIDGE</option>
                                    </select>
                                    {connectionState === 'CONNECTED' ? (
                                        <button onClick={disconnectNeuralInterface} className="px-6 py-2.5 bg-red-600/30 border-2 border-red-500 text-red-200 font-black rounded-xl text-xs uppercase hover:bg-red-600/50 transition-all">Detach</button>
                                    ) : (
                                        <button onClick={handleConnect} disabled={connectionState !== 'DISCONNECTED'} className="px-6 py-2.5 bg-green-600/30 border-2 border-green-500 text-green-200 font-black rounded-xl text-xs uppercase hover:bg-green-600/50 transition-all disabled:opacity-50">
                                            {connectionState === 'SCANNING' ? 'Scanning...' : connectionState === 'CALIBRATING' ? 'Calibrating...' : 'Connect'}
                                        </button>
                                    )}
                                </div>
                             </div>
                        </div>

                        {/* Quantum Metrics HUD */}
                        <div className="grid grid-cols-2 gap-4">
                             <div className="bg-black/60 p-2 rounded-lg border border-purple-900/50 flex justify-between items-center">
                                 <div className="flex items-center gap-2">
                                     <CpuChipIcon className="w-4 h-4 text-purple-400" />
                                     <span className="text-[10px] text-purple-200 font-bold uppercase">Coherence (T2)</span>
                                 </div>
                                 <span className="text-sm font-mono text-white">{coherenceT2.toFixed(1)} <span className="text-[10px] text-gray-500">μs</span></span>
                             </div>
                             <div className={`bg-black/60 p-2 rounded-lg border flex justify-between items-center ${liveFidelity < fidelityThreshold ? 'border-red-500 animate-pulse' : 'border-green-900/50'}`}>
                                 <div className="flex items-center gap-2">
                                     {liveFidelity < fidelityThreshold ? <AlertTriangleIcon className="w-4 h-4 text-red-500"/> : <CheckCircle2Icon className="w-4 h-4 text-green-400" />}
                                     <span className={`text-[10px] font-bold uppercase ${liveFidelity < fidelityThreshold ? 'text-red-400' : 'text-green-200'}`}>
                                         State Fidelity
                                     </span>
                                 </div>
                                 <span className="text-sm font-mono text-white">{liveFidelity.toFixed(4)}</span>
                             </div>
                        </div>
                        {liveFidelity < fidelityThreshold && (
                             <div className="text-center text-[10px] text-red-500 font-black uppercase tracking-[0.2em] bg-red-950/50 border border-red-900 p-1 rounded animate-pulse">
                                 ⚠ CRITICAL DECOHERENCE WARNING ⚠
                             </div>
                        )}

                        {/* Visualization Area */}
                        <div className="flex-grow relative bg-black/60 rounded-2xl border border-cyan-900/50 overflow-hidden shadow-[inset_0_0_40px_rgba(0,0,0,0.8)] flex flex-col">
                            {connectionState === 'CONNECTED' ? (
                                <>
                                    <div className="absolute top-4 right-4 z-20 flex bg-black/80 rounded p-1 border border-gray-700">
                                        <button onClick={() => setVisualMode('WAVEFORM')} className={`p-1 rounded ${visualMode === 'WAVEFORM' ? 'bg-cyan-700 text-white' : 'text-gray-500'}`} title="Waveform"><ActivityIcon className="w-4 h-4"/></button>
                                        <button onClick={() => setVisualMode('MANIFOLD')} className={`p-1 rounded ${visualMode === 'MANIFOLD' ? 'bg-purple-700 text-white' : 'text-gray-500'}`} title="Latent Manifold"><LayersIcon className="w-4 h-4"/></button>
                                    </div>
                                    <canvas ref={canvasRef} className="absolute inset-0 w-full h-full opacity-80" />
                                    
                                    {visualMode === 'MANIFOLD' && (
                                        <div className="absolute bottom-4 right-4 z-20 flex flex-col gap-1 items-end">
                                            <span className="text-[8px] text-gray-500 uppercase font-bold">Synaptic Weight Heatmap</span>
                                            <div className="w-32 h-2 bg-gradient-to-r from-blue-500 via-green-500 to-red-500 rounded-full border border-white/20"></div>
                                        </div>
                                    )}

                                    <div className="absolute top-4 left-4 z-20 flex gap-2">
                                        <div className="px-3 py-1 bg-black/80 rounded-full border border-cyan-500/50 text-[10px] font-black text-cyan-400 flex items-center gap-2">
                                            <div className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse"></div>
                                            LINK_STABLE
                                        </div>
                                    </div>
                                </>
                            ) : (
                                <div className="m-auto text-center space-y-6 animate-pulse">
                                    <FingerPrintIcon className="w-24 h-24 text-cyan-900 mx-auto" />
                                    <div>
                                        <h4 className="text-xl font-black text-cyan-800 uppercase tracking-widest">Awaiting BCI Link</h4>
                                        <p className="text-xs text-gray-700 font-mono">BIOMETRIC_HANDSHAKE_PENDING</p>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Optimization Controls */}
                        <div className="grid grid-cols-3 gap-3 bg-black/50 p-3 rounded-xl border border-gray-800">
                             <div className="col-span-1">
                                 <label className="text-[8px] text-purple-400 uppercase font-bold block mb-1">Entanglement Entropy</label>
                                 <input 
                                     type="range" min="0" max="1" step="0.1" 
                                     value={entropy} onChange={(e) => setEntropy(parseFloat(e.target.value))}
                                     className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                                 />
                             </div>
                             <div className="col-span-1 flex flex-col justify-between">
                                 <label className="text-[8px] text-green-400 uppercase font-bold block">Hebbian Safety Cap</label>
                                 <div className="flex items-center gap-2">
                                     <button onClick={() => setSafetyCap(!safetyCap)} className={`w-8 h-4 rounded-full p-0.5 transition-colors ${safetyCap ? 'bg-green-600' : 'bg-red-600'}`}>
                                         <div className={`w-3 h-3 bg-white rounded-full transition-transform ${safetyCap ? 'translate-x-4' : ''}`}></div>
                                     </button>
                                     <span className="text-[9px] text-white">{safetyCap ? "ENABLED" : "RISK_MODE"}</span>
                                 </div>
                             </div>
                             <div className="col-span-1">
                                 <label className="text-[8px] text-yellow-400 uppercase font-bold block mb-1">Fidelity Threshold</label>
                                 <input 
                                     type="number" min="0.5" max="0.99" step="0.01" 
                                     value={fidelityThreshold} onChange={(e) => setFidelityThreshold(parseFloat(e.target.value))}
                                     className="w-full bg-black/50 border border-yellow-800 text-[10px] text-white p-1 rounded"
                                 />
                             </div>
                        </div>

                    </div>
                ) : mode === 'PROGRAM' ? (
                    <EntanglementScriptForge />
                ) : mode === 'QUANTUM_OPS' ? (
                    <QuantumOpsInterface />
                ) : (
                    <ClinicalBridgeInterface />
                )}
            </div>
        </GlassPanel>
    );
};

export default NeuralProgrammingPanel;
