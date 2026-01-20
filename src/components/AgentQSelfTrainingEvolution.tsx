
import React, { useRef, useEffect, useState } from 'react';
import {
  BrainCircuitIcon,
  PlayIcon,
  StopIcon,
  LoaderIcon,
  CheckCircle2Icon,
  SparklesIcon,
  FileCodeIcon,
  UploadCloudIcon,
  GitBranchIcon,
  AtomIcon,
  GalaxyIcon,
  CpuChipIcon,
  ActivityIcon,
  RocketLaunchIcon,
  CodeBracketIcon,
  ZapIcon,
  LinkIcon,
  TimelineIcon
} from './Icons';
import { SystemHealth } from '../types';
import { useToast } from '../context/ToastContext';
import { useSimulation } from '../context/SimulationContext';
import GlassPanel from './GlassPanel';

interface AgentQSelfTrainingEvolutionProps {
    isRecalibrating: boolean;
    isUpgrading: boolean;
    systemHealth: SystemHealth;
    activeDataStreams: string[];
}

// Satellite Node Component for the Visualizer
const SatelliteNode: React.FC<{ 
    title: string; 
    icon: React.FC<{className?: string}>; 
    isActive: boolean; 
    sync: number;
    color: string;
    position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' | 'bottom-center' | 'mid-left' | 'mid-right' | 'top-center';
    composite?: boolean;
    isEntangled?: boolean;
}> = ({ title, icon: Icon, isActive, sync, color, position, composite = false, isEntangled = false }) => {
    // Positioning logic via Tailwind classes
    const posClass = {
        'top-left': 'top-4 left-4',
        'top-right': 'top-4 right-4',
        'top-center': 'top-4 left-1/2 -translate-x-1/2',
        'mid-left': 'top-1/2 left-4 -translate-y-1/2',
        'mid-right': 'top-1/2 right-4 -translate-y-1/2',
        'bottom-left': 'bottom-4 left-4',
        'bottom-right': 'bottom-4 right-4',
        'bottom-center': 'bottom-4 left-1/2 -translate-x-1/2'
    }[position];

    // Dynamic coloring based on prop
    const getColorClasses = (c: string, active: boolean, entangled: boolean) => {
        if (entangled) return 'border-white bg-white/20 text-white shadow-[0_0_20px_white]';
        if (!active) return 'border-gray-700 bg-black/50 text-gray-500';
        switch (c) {
            case 'cyan': return 'border-cyan-400 bg-cyan-900/30 text-cyan-300 shadow-[0_0_15px_theme(colors.cyan.500)]';
            case 'purple': return 'border-purple-400 bg-purple-900/30 text-purple-300 shadow-[0_0_15px_theme(colors.purple.500)]';
            case 'green': return 'border-green-400 bg-green-900/30 text-green-300 shadow-[0_0_15px_theme(colors.green.500)]';
            case 'yellow': return 'border-yellow-400 bg-yellow-900/30 text-yellow-300 shadow-[0_0_15px_theme(colors.yellow.500)]';
            case 'pink': return 'border-pink-400 bg-pink-900/30 text-pink-300 shadow-[0_0_15px_theme(colors.pink.500)]';
            case 'orange': return 'border-orange-400 bg-orange-900/30 text-orange-300 shadow-[0_0_15px_theme(colors.orange.500)]';
            case 'blue': return 'border-blue-400 bg-blue-900/30 text-blue-300 shadow-[0_0_15px_theme(colors.blue.500)]';
            case 'white': return 'border-white bg-cyan-950/40 text-white shadow-[0_0_15px_white]';
            default: return 'border-gray-700 bg-black/50 text-gray-500';
        }
    };

    const colorClasses = getColorClasses(color, isActive, isEntangled);

    return (
        <div className={`absolute ${posClass} flex flex-col items-center transition-all duration-700 z-20 ${isActive || isEntangled ? 'scale-100' : 'opacity-60 scale-90 grayscale'}`}>
            <div className={`relative p-2 rounded-full border-2 ${colorClasses} ${composite ? 'animate-border-shimmer' : ''}`}>
                <Icon className={`w-5 h-5 ${isActive || isEntangled ? 'animate-pulse' : ''}`} />
                {(isActive || isEntangled) && (
                    <div className="absolute inset-0 rounded-full animate-ping opacity-10 bg-white"></div>
                )}
                {isEntangled && (
                    <div className="absolute -top-1 -right-1 bg-white rounded-full p-0.5 border border-black shadow-lg">
                        <LinkIcon className="w-2.5 h-2.5 text-black" />
                    </div>
                )}
            </div>
            <div className="mt-1 text-center bg-black/60 px-2 rounded backdrop-blur-sm border border-black/50">
                <span className={`text-[9px] font-bold uppercase tracking-wider text-white truncate block w-full`}>{title}</span>
                {(isActive || isEntangled) && (
                    <div className={`text-[7px] font-mono text-cyan-200 uppercase`}>{isEntangled ? 'ENTANGLED' : `SYNC: ${(sync || 0).toFixed(0)}%`}</div>
                )}
            </div>
        </div>
    );
};

const AgentQSelfTrainingEvolution: React.FC<AgentQSelfTrainingEvolutionProps> = ({ isRecalibrating }) => {
    const { addToast } = useToast();
    const { 
        training, evolution, toggleEvolution, entanglementMesh, 
        setTrainingPatch, qllm, qrlEngine, qdlEngine, 
        neuralInterface, singularityBoost, qmlEngine 
    } = useSimulation();
    
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // --- Particle & Visual Effects Loop ---
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let frameId: number;
        let t = 0;

        // Neural Particles State
        const particles: {x: number, y: number, vx: number, vy: number, age: number, life: number}[] = [];

        const draw = () => {
            // Handle DPI Scaling
            if (canvas.parentElement) {
                const dpr = window.devicePixelRatio || 1;
                const rect = canvas.parentElement.getBoundingClientRect();
                
                if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
                    canvas.width = rect.width * dpr;
                    canvas.height = rect.height * dpr;
                    canvas.style.width = `${rect.width}px`;
                    canvas.style.height = `${rect.height}px`;
                    ctx.scale(dpr, dpr);
                }
            }

            // Logic Dimensions
            const w = canvas.width / (window.devicePixelRatio || 1);
            const h = canvas.height / (window.devicePixelRatio || 1);
            const cx = w / 2;
            const cy = h / 2;
            
            t += 0.03;

            // Fade Effect (Trails)
            ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
            ctx.fillRect(0, 0, w, h);

            // Node Positions (Must align with SatelliteNode CSS positions roughly)
            const nodePos = {
                sim: { x: 50, y: 50 },
                qml: { x: w - 50, y: 50 },
                neural: { x: 50, y: cy },
                qdl: { x: w - 50, y: cy },
                qrl: { x: cx, y: h - 50 },
                universe: { x: cx, y: 50 },
                center: { x: cx, y: cy }
            };

            const nodes = [
                { id: 'sim', active: qllm.isActive, ...nodePos.sim, color: 'rgba(6, 182, 212, 0.6)' },
                { id: 'qml', active: qmlEngine.status !== 'IDLE', ...nodePos.qml, color: 'rgba(168, 85, 247, 0.6)' },
                { id: 'neural', active: neuralInterface.isActive, ...nodePos.neural, color: 'rgba(250, 204, 21, 0.6)' },
                { id: 'qdl', active: qdlEngine.status === 'TRAINING', ...nodePos.qdl, color: 'rgba(59, 130, 246, 0.6)' },
                { id: 'qrl', active: qrlEngine.status !== 'IDLE', ...nodePos.qrl, color: 'rgba(249, 115, 22, 0.6)' },
                { id: 'universe', active: true, ...nodePos.universe, color: 'rgba(255, 255, 255, 0.8)' }
            ];

            // 1. Draw "Neural Fog" (Background Fluidity)
            for (let i = 0; i < 5; i++) {
                ctx.beginPath();
                ctx.lineWidth = 20;
                ctx.strokeStyle = `rgba(168, 85, 247, 0.03)`;
                ctx.moveTo(0, h/2 + Math.sin(t + i) * 100);
                ctx.bezierCurveTo(w/3, h/2 + Math.cos(t * 1.5) * 150, w*2/3, h/2 - Math.cos(t) * 150, w, h/2 + Math.sin(t + i) * 100);
                ctx.stroke();
            }

            // 2. Full Mesh Entanglement Lines & Particle Generation
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const n1 = nodes[i];
                    const n2 = nodes[j];
                    
                    const bothActive = n1.active && n2.active;
                    
                    ctx.beginPath();
                    ctx.lineWidth = bothActive ? 1.5 : 0.5;
                    ctx.strokeStyle = bothActive 
                        ? `rgba(255, 255, 255, ${0.1 + Math.sin(t * 3 + i + j) * 0.1})` 
                        : `rgba(255, 255, 255, 0.05)`;
                    
                    ctx.moveTo(n1.x, n1.y);
                    // Curve lines slightly towards center for "Gravitational Singularity" feel
                    const cpX = cx + (n1.x - cx) * 0.2 + (n2.x - cx) * 0.2;
                    const cpY = cy + (n1.y - cy) * 0.2 + (n2.y - cy) * 0.2;
                    ctx.quadraticCurveTo(cx, cy, n2.x, n2.y);
                    ctx.stroke();

                    // Spawn Particles on active connections
                    if (bothActive && Math.random() > 0.92) {
                        particles.push({
                            x: n1.x, y: n1.y,
                            vx: (n2.x - n1.x) * 0.02,
                            vy: (n2.y - n1.y) * 0.02,
                            age: 0, life: 50 + Math.random() * 20
                        });
                    }
                }
            }

            // 3. Render & Update Particles
            for (let i = particles.length - 1; i >= 0; i--) {
                const p = particles[i];
                p.x += p.vx;
                p.y += p.vy;
                p.age++;

                // Gravitational Pull to Center Singularity
                p.x += (cx - p.x) * 0.005;
                p.y += (cy - p.y) * 0.005;

                const alpha = 1 - (p.age / p.life);
                ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
                ctx.beginPath();
                ctx.arc(p.x, p.y, 1.5, 0, Math.PI * 2);
                ctx.fill();

                if (p.age > p.life) particles.splice(i, 1);
            }

            // 4. Central Singularity Core (Pulsating)
            ctx.beginPath();
            const corePulse = (25 + singularityBoost/3) + Math.sin(t * 2) * (3 + singularityBoost/15);
            const coreColor = singularityBoost > 40 ? 'rgba(255, 255, 255, 0.95)' : (singularityBoost > 20 ? 'rgba(168, 85, 247, 0.8)' : 'rgba(100, 116, 139, 0.5)');
            
            // Core Glow Gradient
            const gradient = ctx.createRadialGradient(cx, cy, 5, cx, cy, corePulse);
            gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
            gradient.addColorStop(0.5, coreColor);
            gradient.addColorStop(1, 'rgba(0,0,0,0)');
            
            ctx.fillStyle = gradient;
            ctx.arc(cx, cy, corePulse, 0, Math.PI * 2);
            ctx.fill();

            // 12-Dimensional Rings
            ctx.lineWidth = 1;
            for (let r = 0; r < 3; r++) {
                ctx.beginPath();
                ctx.strokeStyle = `rgba(34, 211, 238, ${0.3 / (r + 1)})`;
                ctx.ellipse(cx, cy, corePulse + 10 + r * 15, (corePulse + 10 + r * 8) * 0.4, t * (0.5 + r * 0.2), 0, Math.PI * 2);
                ctx.stroke();
            }

            frameId = requestAnimationFrame(draw);
        };
        draw();

        return () => {
            cancelAnimationFrame(frameId);
        };
    }, [qllm.isActive, qmlEngine.status, neuralInterface.isActive, qdlEngine.status, qrlEngine.status, singularityBoost, entanglementMesh.isQRLtoQNNLinked]);

    return (
        <GlassPanel title={<div className="flex items-center"><SparklesIcon className="w-5 h-5 mr-2 text-purple-400" />QNN Evolution</div>}>
            <div className="flex flex-col h-full space-y-2 p-1 overflow-hidden">
                {/* Header Info */}
                <div className="flex justify-between items-center bg-black/40 p-2 rounded-lg border border-cyan-800/30 flex-shrink-0">
                    <div>
                        <h3 className="text-xs font-bold text-white flex items-center">
                            AGI Singularity Forge: Hyper-Entangled
                        </h3>
                        <div className="flex items-center gap-2">
                            <p className="text-[7px] text-cyan-600 font-mono uppercase">Topology: Full-Mesh Non-Local</p>
                            {singularityBoost > 0 && (
                                <span className="text-[7px] bg-cyan-900/40 text-cyan-300 px-1 rounded border border-cyan-700 animate-pulse uppercase">Dimension Shift Active</span>
                            )}
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="flex flex-col items-end">
                            <span className="text-[6px] text-gray-500 uppercase font-bold tracking-tighter">Singularity Alignment</span>
                            <div className="w-16 h-0.5 bg-gray-800 rounded-full overflow-hidden mt-0.5 border border-white/5">
                                <div className="h-full bg-gradient-to-r from-cyan-500 via-purple-500 to-white transition-all duration-1000" style={{ width: `${Math.min(100, 40 + singularityBoost)}%` }}></div>
                            </div>
                        </div>
                        <button onClick={toggleEvolution} className="holographic-button px-2 py-0.5 text-[8px] font-bold flex items-center gap-1 bg-purple-600/30 text-purple-200 border-purple-500/50">
                            {evolution.isActive ? <StopIcon className="w-2 h-2" /> : <PlayIcon className="w-2 h-2" />}
                            {evolution.isActive ? 'HALT' : 'EVOLVE'}
                        </button>
                    </div>
                </div>

                {/* Main Visualizer Area */}
                <div className="flex-grow relative bg-black/60 rounded-lg border border-cyan-900/20 overflow-hidden flex items-center justify-center shadow-[inset_0_0_30px_rgba(0,0,0,0.8)]">
                    <canvas ref={canvasRef} className="absolute inset-0 w-full h-full z-10" />
                    
                    {/* Center Anchor Icon */}
                    <div className="absolute z-30 pointer-events-none top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                        <div className="relative w-12 h-12 flex items-center justify-center">
                            <div className="absolute inset-0 border border-white/20 rounded-full animate-ping"></div>
                            <CpuChipIcon className="w-6 h-6 text-white animate-pulse drop-shadow-[0_0_10px_white]" />
                        </div>
                    </div>

                    {/* Nodes Overlay */}
                    <SatelliteNode title="Sim + QLLM" icon={GalaxyIcon} position="top-left" color="cyan" isActive={qllm.isActive} sync={92} composite={true} />
                    <SatelliteNode title="Predictor Core" icon={TimelineIcon} position="top-center" color="white" isActive={true} sync={100} isEntangled={true} />
                    <SatelliteNode title="QML Forge" icon={BrainCircuitIcon} position="top-right" color="purple" isActive={qmlEngine.status !== 'IDLE'} sync={qmlEngine.progress} composite={true} />
                    <SatelliteNode title="Neural Bridge" icon={ActivityIcon} position="mid-left" color="yellow" isActive={neuralInterface.isActive} sync={neuralInterface.coherence} />
                    <SatelliteNode title="Deep hierarchy" icon={AtomIcon} position="mid-right" color="blue" isActive={qdlEngine.status === 'TRAINING'} sync={qdlEngine.accuracy * 100} />
                    <SatelliteNode title="Strategy Loop" icon={RocketLaunchIcon} position="bottom-center" color="orange" isActive={qrlEngine.status !== 'IDLE'} sync={qrlEngine.avgReward > 0 ? Math.min(100, qrlEngine.avgReward * 10) : 0} />
                </div>

                {/* Notification Area */}
                {training.generatedPatch && (
                    <div className="bg-cyan-900/30 border border-cyan-500/50 p-2 rounded-lg animate-fade-in-up flex items-center justify-between flex-shrink-0">
                        <div className="flex items-center gap-2">
                            <FileCodeIcon className="w-4 h-4 text-cyan-300" />
                            <span className="text-[8px] font-bold text-white uppercase">Neural Artifact Generated</span>
                        </div>
                        <button 
                            onClick={() => { setTrainingPatch(null); addToast("Evolved Logic Committed.", "success"); }}
                            className="px-3 py-1 text-[8px] font-bold bg-cyan-600/40 border border-cyan-500 text-white rounded hover:bg-cyan-600/60 flex items-center gap-1"
                        >
                            <UploadCloudIcon className="w-2 h-2" /> Integrate
                        </button>
                    </div>
                )}
            </div>
        </GlassPanel>
    );
};

export default AgentQSelfTrainingEvolution;
