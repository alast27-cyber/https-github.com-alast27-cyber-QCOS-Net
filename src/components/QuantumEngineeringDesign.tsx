
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
    AtomIcon, PlayIcon, ZapIcon,
    ChartBarIcon, ActivityIcon, CheckCircle2Icon, Share2Icon,
    DownloadCloudIcon, LayersIcon, CpuChipIcon, StopIcon,
    MaximizeIcon, XIcon, ZoomInIcon, ZoomOutIcon, BoxIcon,
    FileCodeIcon, GridIcon, EyeIcon, MoveIcon, FileIcon,
    RefreshCwIcon, SettingsIcon, CodeBracketIcon, RulerIcon,
    SparklesIcon, LoaderIcon
} from './Icons';
import { useSimulation } from '../context/SimulationContext';
import { useToast } from '../context/ToastContext';
import { ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts';

// --- Types ---
type PipelineStage = 'IDLE' | 'SEMANTIC_PARSING' | 'ENGINE_CONFIG' | 'QUANTUM_OPT' | 'GENERATIVE_CAD' | 'PHYSICS_VALIDATION' | 'COMPLETE';
type ViewerMode = 'WIREFRAME' | 'SOLID' | 'THERMAL' | 'STRESS';
type GeometryType = 'TORUS' | 'CONE' | 'LATTICE' | 'PLANAR' | 'SPHERE';
type CADMode = 'DESIGN' | 'PHYSICS' | 'BLUEPRINT';

interface EngineWeight {
    subject: string;
    A: number; // Current Weight
    fullMark: number;
}

interface GeneratedFile {
    name: string;
    type: 'model' | 'blueprint' | 'report' | 'data';
    size: string;
    geometry?: GeometryType; 
    content?: string; 
}

interface Vertex { x: number, y: number, z: number }

// --- QUANTUM CAD STUDIO (Back Panel) ---
const QuantumCADStudio: React.FC<{ onClose: () => void }> = ({ onClose }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [mode, setMode] = useState<CADMode>('DESIGN');
    const [prompt, setPrompt] = useState('');
    const [isGenerating, setIsGenerating] = useState(false);
    const [physicsState, setPhysicsState] = useState({ temp: 300, stress: 0, flux: 0 });
    const [mesh, setMesh] = useState<Vertex[]>([]);
    const [rotation, setRotation] = useState({ x: 0.5, y: 0.5 });
    
    const generateMesh = (type: string) => {
        const vertices: Vertex[] = [];
        const steps = 24;
        const radius = 80;

        for (let i = 0; i < steps; i++) {
            const theta = (i / steps) * Math.PI * 2;
            for (let j = 0; j < steps; j++) {
                const phi = (j / steps) * Math.PI;
                
                let x, y, z;
                
                if (type.includes('cube')) {
                    x = (Math.random() - 0.5) * 150;
                    y = (Math.random() - 0.5) * 150;
                    z = (Math.random() - 0.5) * 150;
                } else if (type.includes('engine') || type.includes('thruster')) {
                    const r = radius * (1 - j/steps);
                    x = r * Math.cos(theta);
                    z = r * Math.sin(theta);
                    y = (j / steps) * 200 - 100;
                } else {
                    // Torus/Sphere hybrid
                    const R = 60; const r = 30;
                    x = (R + r * Math.cos(phi)) * Math.cos(theta);
                    y = (R + r * Math.cos(phi)) * Math.sin(theta);
                    z = r * Math.sin(phi);
                }
                vertices.push({ x, y, z });
            }
        }
        setMesh(vertices);
    };

    // Initialize standard mesh
    useEffect(() => {
        setTimeout(() => generateMesh('sphere'), 0);
    }, []);

    const handleGenerate = () => {
        if (!prompt) return;
        setIsGenerating(true);
        setTimeout(() => {
            generateMesh(prompt.toLowerCase());
            setIsGenerating(false);
        }, 1500);
    };

    // Physics Loop
    useEffect(() => {
        if (mode !== 'PHYSICS') return;
        const interval = setInterval(() => {
            setPhysicsState(prev => ({
                temp: 300 + Math.random() * 500,
                stress: Math.random() * 100,
                flux: Math.random()
            }));
        }, 100);
        return () => clearInterval(interval);
    }, [mode]);

    // Render Loop
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let frameId: number;
        let time = 0;

        const render = () => {
            time += 0.02;
            
            // Auto-rotate if not interacting
            const currentRotX = rotation.x + time * 0.1;
            const currentRotY = rotation.y + time * 0.2;

            if (canvas.parentElement) {
                canvas.width = canvas.parentElement.clientWidth;
                canvas.height = canvas.parentElement.clientHeight;
            }
            const w = canvas.width;
            const h = canvas.height;
            const cx = w / 2;
            const cy = h / 2;

            // Background
            if (mode === 'BLUEPRINT') {
                ctx.fillStyle = '#0f172a'; // Slate-900
                ctx.fillRect(0, 0, w, h);
                // Grid
                ctx.strokeStyle = 'rgba(56, 189, 248, 0.2)';
                ctx.lineWidth = 1;
                const gridSize = 40;
                for (let x = 0; x < w; x += gridSize) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
                for (let y = 0; y < h; y += gridSize) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
            } else {
                ctx.clearRect(0, 0, w, h);
            }

            // Project and Draw Vertices
            const focalLength = 500;

            mesh.forEach((v, i) => {
                // Clone to avoid modifying state directly during physics sim
                let px = v.x;
                let py = v.y;
                let pz = v.z;

                // Physics Perturbation
                if (mode === 'PHYSICS') {
                    const vibration = Math.sin(time * 20 + i) * (physicsState.temp / 200);
                    px += vibration;
                    py += vibration;
                    pz += vibration;
                }

                // Rotation
                const y = py * Math.cos(currentRotX) - pz * Math.sin(currentRotX);
                let z = py * Math.sin(currentRotX) + pz * Math.cos(currentRotX);
                const x = px * Math.cos(currentRotY) - z * Math.sin(currentRotY);
                z = px * Math.sin(currentRotY) + z * Math.cos(currentRotY);

                // Projection
                const scale = focalLength / (focalLength + z);
                const x2d = x * scale + cx;
                const y2d = y * scale + cy;

                // Drawing Style
                if (mode === 'BLUEPRINT') {
                    ctx.fillStyle = 'white';
                    ctx.fillRect(x2d, y2d, 1.5, 1.5);
                    // Minimal lines
                    if (i % 2 === 0) {
                         // Connect to "previous" roughly
                         ctx.strokeStyle = 'rgba(255,255,255,0.1)';
                         ctx.beginPath();
                         ctx.moveTo(cx, cy); // Schematic lines
                         ctx.lineTo(x2d, y2d);
                         ctx.stroke();
                    }
                } else if (mode === 'PHYSICS') {
                    // Heatmap coloring
                    const heatColor = `hsl(${200 - (physicsState.temp / 10)}, 100%, 50%)`;
                    ctx.fillStyle = heatColor;
                    ctx.beginPath();
                    ctx.arc(x2d, y2d, scale * 2, 0, Math.PI * 2);
                    ctx.fill();
                    
                    // Stress lines
                    if (Math.random() > 0.95) {
                        ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
                        ctx.beginPath();
                        ctx.moveTo(x2d, y2d);
                        ctx.lineTo(x2d + (Math.random()-0.5)*20, y2d + (Math.random()-0.5)*20);
                        ctx.stroke();
                    }

                } else {
                    // Design Mode (Protruding / Volumetric feel)
                    const depthAlpha = Math.max(0.1, Math.min(1, (z + 100) / 200));
                    ctx.fillStyle = `rgba(34, 211, 238, ${depthAlpha})`;
                    ctx.fillRect(x2d, y2d, scale * 2, scale * 2);
                    
                    if (i % 10 === 0) {
                         ctx.strokeStyle = `rgba(34, 211, 238, ${depthAlpha * 0.3})`;
                         ctx.beginPath();
                         ctx.moveTo(cx, cy);
                         ctx.lineTo(x2d, y2d);
                         ctx.stroke();
                    }
                }
            });
            
            // Blueprint Dimensions
            if (mode === 'BLUEPRINT') {
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 1;
                
                // Draw Dimension Lines
                ctx.beginPath();
                ctx.moveTo(cx - 100, cy + 150);
                ctx.lineTo(cx + 100, cy + 150);
                ctx.moveTo(cx - 100, cy + 145); ctx.lineTo(cx - 100, cy + 155);
                ctx.moveTo(cx + 100, cy + 145); ctx.lineTo(cx + 100, cy + 155);
                ctx.stroke();
                
                ctx.fillStyle = '#fff';
                ctx.font = '10px monospace';
                ctx.fillText('240.5 mm', cx - 20, cy + 165);
                
                ctx.fillStyle = 'cyan';
                ctx.font = '12px monospace';
                ctx.fillText('FIG 1.0 - QUANTUM ASSEMBLY', 20, 30);
            }

            frameId = requestAnimationFrame(render);
        };
        render();
        return () => cancelAnimationFrame(frameId);
    }, [mode, mesh, physicsState, rotation]);

    return (
        <div className="h-full flex flex-col bg-slate-950 rounded-xl overflow-hidden border border-cyan-500/20 shadow-2xl relative">
            {/* Studio Toolbar */}
            <div className="flex justify-between items-center bg-black/60 border-b border-cyan-800/50 p-2 backdrop-blur-md">
                <div className="flex items-center gap-2">
                    <AtomIcon className="w-5 h-5 text-purple-400" />
                    <span className="font-bold text-white text-xs tracking-wider">Q-CAD STUDIO</span>
                </div>
                <div className="flex bg-black/60 rounded p-0.5 border border-cyan-900">
                    <button onClick={() => setMode('DESIGN')} className={`px-3 py-1 text-[10px] font-bold rounded transition-colors ${mode === 'DESIGN' ? 'bg-cyan-700 text-white' : 'text-gray-400 hover:text-white'}`}>DESIGN</button>
                    <button onClick={() => setMode('PHYSICS')} className={`px-3 py-1 text-[10px] font-bold rounded transition-colors ${mode === 'PHYSICS' ? 'bg-red-700 text-white' : 'text-gray-400 hover:text-white'}`}>PHYSICS</button>
                    <button onClick={() => setMode('BLUEPRINT')} className={`px-3 py-1 text-[10px] font-bold rounded transition-colors ${mode === 'BLUEPRINT' ? 'bg-blue-700 text-white' : 'text-gray-400 hover:text-white'}`}>BLUEPRINT</button>
                </div>
                <button 
                    onClick={onClose} 
                    className="p-1.5 hover:bg-cyan-500/20 rounded-md border border-cyan-800 text-cyan-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"
                    title="Return to Pipeline View"
                >
                    <RefreshCwIcon className="w-4 h-4" />
                </button>
            </div>

            <div className="flex-grow flex gap-2 min-h-0 relative p-2">
                {/* Canvas Viewport */}
                <div className="flex-grow relative bg-black rounded border border-cyan-900/30 overflow-hidden cursor-crosshair">
                     <canvas ref={canvasRef} className="absolute inset-0 w-full h-full block" />
                     
                     {/* Overlay HUD */}
                     <div className="absolute top-2 left-2 flex flex-col gap-1 pointer-events-none">
                         <div className="bg-black/60 px-2 py-1 rounded border border-cyan-900/50 text-[9px] text-cyan-400 font-mono">
                             VERTICES: {mesh.length}
                         </div>
                         {mode === 'PHYSICS' && (
                             <div className="bg-black/60 px-2 py-1 rounded border border-red-900/50 text-[9px] text-red-400 font-mono">
                                 TEMP: {physicsState.temp.toFixed(1)}K <br/>
                                 STRESS: {physicsState.stress.toFixed(2)} MPa
                             </div>
                         )}
                     </div>

                     {/* Prompt Input Overlay */}
                     <div className="absolute bottom-4 left-4 right-4 flex gap-2">
                         <div className="relative flex-grow">
                             <div className="absolute inset-y-0 left-2 flex items-center pointer-events-none">
                                 <SparklesIcon className="w-4 h-4 text-purple-400 animate-pulse" />
                             </div>
                             <input 
                                type="text" 
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                placeholder="Describe shape to generate (e.g. 'Hyper-drive casing', 'Flux cube')..."
                                className="w-full bg-black/80 backdrop-blur-md border border-purple-500/50 rounded pl-8 pr-4 py-2 text-xs text-white focus:border-purple-400 outline-none shadow-lg"
                                onKeyDown={(e) => e.key === 'Enter' && handleGenerate()}
                             />
                         </div>
                         <button 
                            onClick={handleGenerate}
                            disabled={isGenerating}
                            className="bg-purple-600/80 hover:bg-purple-500 text-white text-xs font-bold px-4 py-2 rounded border border-purple-400 shadow-[0_0_15px_rgba(168,85,247,0.4)] flex items-center gap-2 transition-all"
                         >
                             {isGenerating ? <RefreshCwIcon className="w-4 h-4 animate-spin"/> : <ZapIcon className="w-4 h-4"/>}
                             {isGenerating ? 'GEN...' : 'GENERATE'}
                         </button>
                     </div>
                </div>

                {/* Properties Sidebar (Right) */}
                <div className="w-40 flex-shrink-0 bg-black/40 border border-cyan-900/50 rounded flex flex-col p-2 gap-2">
                    <div className="text-[10px] font-bold text-cyan-500 uppercase border-b border-cyan-900/50 pb-1">Properties</div>
                    <div className="space-y-2">
                        <div className="bg-cyan-950/30 p-1.5 rounded border border-cyan-900">
                            <span className="block text-[8px] text-gray-500 uppercase">Material</span>
                            <span className="block text-[10px] text-white">Graphene-Aerogel</span>
                        </div>
                        <div className="bg-cyan-950/30 p-1.5 rounded border border-cyan-900">
                            <span className="block text-[8px] text-gray-500 uppercase">Mass</span>
                            <span className="block text-[10px] text-white">4.2 kg</span>
                        </div>
                        <div className="bg-cyan-950/30 p-1.5 rounded border border-cyan-900">
                             <span className="block text-[8px] text-gray-500 uppercase">Topology</span>
                             <span className="block text-[10px] text-white">Non-Euclidean</span>
                        </div>
                    </div>
                    
                    <div className="text-[10px] font-bold text-cyan-500 uppercase border-b border-cyan-900/50 pb-1 mt-2">Tools</div>
                    <div className="space-y-1">
                        <button onClick={() => setMode('PHYSICS')} className={`w-full text-left px-2 py-1.5 rounded text-[10px] border flex items-center gap-2 transition-colors ${mode === 'PHYSICS' ? 'bg-red-900/20 border-red-500 text-red-200' : 'border-gray-800 text-gray-400 hover:text-white hover:bg-white/5'}`}>
                            <ActivityIcon className="w-3 h-3" /> Stress Test
                        </button>
                        <button className="w-full text-left px-2 py-1.5 rounded text-[10px] border border-gray-800 text-gray-400 hover:text-white hover:bg-white/5 transition-colors flex items-center gap-2">
                            <MoveIcon className="w-3 h-3" /> Fluid Dynamics
                        </button>
                    </div>

                    <div className="mt-auto">
                        <button className="w-full py-2 bg-blue-600/20 border border-blue-500 text-blue-200 text-[10px] font-bold rounded flex items-center justify-center gap-2 hover:bg-blue-600/40 transition-colors">
                             <DownloadCloudIcon className="w-3 h-3" /> EXPORT CAD
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};


// --- QUANTUM CAD VIEWER (Overlay - Kept for legacy compatibility) ---
const QuantumCADViewer: React.FC<{ file: GeneratedFile; onClose: () => void }> = ({ file, onClose }) => {
    return (
         <div className="absolute inset-0 z-50 bg-black/90 backdrop-blur-xl flex flex-col animate-fade-in">
             <div className="flex justify-between p-2 border-b border-cyan-800 bg-black/40">
                 <div className="flex items-center gap-2">
                    <FileIcon className="w-4 h-4 text-cyan-400" />
                    <span className="text-white text-xs font-bold">{file.name}</span>
                 </div>
                 <button onClick={onClose} className="hover:text-white text-gray-500"><XIcon className="w-4 h-4" /></button>
             </div>
             <div className="flex-grow flex items-center justify-center text-cyan-500 flex-col gap-2">
                 <BoxIcon className="w-16 h-16 opacity-50" />
                 <p className="text-sm font-mono">Quick Preview Mode</p>
                 <p className="text-xs text-gray-500">File Size: {file.size}</p>
             </div>
         </div>
    );
};

// --- Main Component ---
const QuantumEngineeringDesign: React.FC = () => {
    const { addToast } = useToast();
    const { qiaiIps, updateQIAIIPS } = useSimulation();
    const [viewMode, setViewMode] = useState<'PIPELINE' | 'STUDIO'>('PIPELINE');

    // --- State ---
    const [inputPrompt, setInputPrompt] = useState("");
    const [stage, setStage] = useState<PipelineStage>('IDLE');
    const [logs, setLogs] = useState<string[]>([]);
    const [engineWeights, setEngineWeights] = useState<EngineWeight[]>([
        { subject: 'QLLM (Semantic)', A: 20, fullMark: 100 },
        { subject: 'QML (Pattern)', A: 20, fullMark: 100 },
        { subject: 'QRL (Strategy)', A: 20, fullMark: 100 },
        { subject: 'QGL (Creative)', A: 20, fullMark: 100 },
        { subject: 'QDL (Physics)', A: 20, fullMark: 100 },
    ]);
    const [optimizationData, setOptimizationData] = useState<{iter: number, energy: number}[]>([]);
    const [generatedArtifacts, setGeneratedArtifacts] = useState<GeneratedFile[]>([]);
    const [activeFile, setActiveFile] = useState<GeneratedFile | null>(null);

    const addLog = (msg: string) => setLogs(prev => [...prev.slice(-10), `[${new Date().toLocaleTimeString()}] ${msg}`]);

    const runPipeline = () => {
        if (!inputPrompt.trim()) {
            addToast("Please enter a design prompt.", "warning");
            return;
        }
        if (stage !== 'IDLE' && stage !== 'COMPLETE') return;
        
        setStage('SEMANTIC_PARSING');
        setLogs(["Initializing Project Quantum-Design...", "Agent Q: Parsing intent vector..."]);
        setGeneratedArtifacts([]);
        setOptimizationData([]);
        
        // ... (Existing pipeline logic preserved) ...
        setTimeout(() => {
            setStage('ENGINE_CONFIG');
            addLog(`Intent Detected: GENERATOR`);
            
            updateQIAIIPS({
                qil: { ...qiaiIps.qil, status: 'INGESTING', load: 75 },
                qips: { ...qiaiIps.qips, status: 'SOLVING', load: 92 }
            });
        }, 1000);

        setTimeout(() => {
            setStage('QUANTUM_OPT');
            let iter = 0;
            const optInterval = setInterval(() => {
                iter++;
                setOptimizationData(prev => [...prev, { iter, energy: Math.random() * 10 }]);
                if (iter > 10) {
                    clearInterval(optInterval);
                    addLog("Convergence Reached.");
                    setStage('COMPLETE');
                    setGeneratedArtifacts([{ name: 'design_v1.step', type: 'model', size: '12MB', geometry: 'TORUS' }]);
                }
            }, 200);
        }, 2000);
    };

    return (
        <div className="relative w-full h-full overflow-hidden group">
            
            {/* Viewer Overlay */}
            {activeFile && <QuantumCADViewer file={activeFile} onClose={() => setActiveFile(null)} />}

            {/* --- FRONT PANEL: PIPELINE VIEW --- */}
            <div className={`absolute inset-0 flex flex-col gap-4 p-4 transition-all duration-500 ease-in-out ${viewMode === 'PIPELINE' ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-full pointer-events-none'}`}>
                {/* Header Row */}
                <div className="flex items-center justify-between">
                     <div className="flex items-center text-cyan-200 font-bold text-sm">
                        <AtomIcon className="w-5 h-5 mr-2 text-blue-400" />
                        <span>Quantum Engineering Design Suite</span>
                     </div>
                     <div className="flex items-center gap-2">
                         <span className={`text-[10px] px-2 py-0.5 rounded border font-mono ${stage !== 'IDLE' && stage !== 'COMPLETE' ? 'bg-blue-900/30 border-blue-500 text-blue-300 animate-pulse' : 'bg-gray-800 border-gray-600 text-gray-500'}`}>
                             {stage === 'IDLE' ? 'READY' : stage}
                         </span>
                         <button 
                             onClick={() => setViewMode('STUDIO')}
                             className="p-1.5 hover:bg-purple-500/20 rounded-md border border-purple-800 text-purple-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"
                             title="Switch to Q-CAD Studio"
                         >
                             <RefreshCwIcon className="w-4 h-4" />
                         </button>
                     </div>
                </div>

                {/* Top: Interaction */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 flex-shrink-0">
                    <div className="col-span-2 bg-black/40 border border-cyan-800/50 rounded-lg p-3 flex flex-col gap-2">
                        <div className="flex items-center justify-between">
                            <h4 className="text-xs font-bold text-cyan-300 uppercase flex items-center gap-2">
                                <Share2Icon className="w-4 h-4" /> Semantic Translation Layer
                            </h4>
                            <span className="text-[9px] text-gray-500">Agent Q v4.2</span>
                        </div>
                        <div className="flex gap-2">
                            <input 
                                className="flex-grow bg-black/50 border border-cyan-900 rounded px-3 py-2 text-sm text-white placeholder-cyan-700/50 focus:border-cyan-400 outline-none"
                                value={inputPrompt}
                                onChange={(e) => setInputPrompt(e.target.value)}
                                placeholder="Describe engineering requirement..."
                            />
                            <button 
                                onClick={runPipeline}
                                disabled={stage !== 'IDLE' && stage !== 'COMPLETE'}
                                className={`holographic-button px-4 py-2 text-xs font-bold rounded flex items-center gap-2 ${stage !== 'IDLE' && stage !== 'COMPLETE' ? 'opacity-50 cursor-not-allowed' : 'bg-blue-600/30 border-blue-500 text-blue-200'}`}
                            >
                                {stage !== 'IDLE' && stage !== 'COMPLETE' ? <StopIcon className="w-3 h-3"/> : <PlayIcon className="w-3 h-3" />}
                                Generate
                            </button>
                        </div>
                    </div>

                    <div className="col-span-1 bg-black/40 border border-purple-900/50 rounded-lg p-2 relative flex flex-col items-center">
                        <div className="absolute top-2 left-2 text-[9px] font-bold text-purple-400 uppercase">Cognitive Topology</div>
                        <div className="h-24 w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={engineWeights}>
                                    <PolarGrid stroke="#334155" />
                                    <PolarAngleAxis dataKey="subject" tick={{fontSize: 8, fill: '#94a3b8'}} />
                                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                                    <Radar name="Config" dataKey="A" stroke="#a855f7" fill="#a855f7" fillOpacity={0.4} />
                                </RadarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>

                {/* Middle: Visualization Placeholder */}
                <div className="flex-grow bg-black/50 border border-cyan-800/50 rounded-lg relative overflow-hidden flex flex-col items-center justify-center min-h-[100px]">
                    <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-cyan-900/10 via-black to-black opacity-50"></div>
                    {stage === 'GENERATIVE_CAD' ? (
                        <div className="text-center">
                            <LoaderIcon className="w-12 h-12 text-cyan-400 animate-spin mx-auto mb-4" />
                            <p className="text-cyan-300 font-mono text-sm uppercase tracking-widest animate-pulse">Synthesizing Geometry...</p>
                        </div>
                    ) : (
                        <div className="text-center opacity-50">
                            <BoxIcon className="w-16 h-16 text-cyan-800 mx-auto mb-2" />
                            <p className="text-xs text-cyan-600 font-mono">Awaiting Input or Switch to Studio Mode</p>
                        </div>
                    )}
                </div>

                {/* Bottom: Logs */}
                <div className="h-32 bg-black/30 border border-cyan-900/50 rounded-lg p-3 flex flex-col font-mono text-[10px] relative overflow-hidden">
                    <h4 className="text-xs font-bold text-cyan-500 uppercase mb-2 flex items-center gap-2">
                        <LayersIcon className="w-3 h-3" /> System Feed
                    </h4>
                    <div className="flex-grow overflow-y-auto custom-scrollbar space-y-1">
                        {logs.map((log, i) => (
                            <div key={i} className="text-cyan-200/80 border-b border-white/5 pb-0.5">{log}</div>
                        ))}
                    </div>
                    {stage === 'COMPLETE' && (
                        <div className="absolute bottom-2 right-2 flex gap-2">
                            {generatedArtifacts.map((art, i) => (
                                <button key={i} onClick={() => setActiveFile(art)} className="px-3 py-1 bg-green-900/40 border border-green-500 text-green-300 rounded text-[9px] font-bold hover:bg-green-900/60">
                                    View {art.name}
                                </button>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* --- BACK PANEL: Q-CAD STUDIO --- */}
            <div className={`absolute inset-0 transition-all duration-500 ease-in-out ${viewMode === 'STUDIO' ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-full pointer-events-none'}`}>
                 <QuantumCADStudio onClose={() => setViewMode('PIPELINE')} />
            </div>

        </div>
    );
};

export default QuantumEngineeringDesign;
