
import React, { useState, useEffect, useRef } from 'react';
import GlassPanel from './GlassPanel';
import { CpuChipIcon, ActivityIcon, PlayIcon, StopIcon, RefreshCwIcon, ZapIcon, GridIcon } from './Icons';

interface Qubit {
    id: number;
    state: '0' | '1' | '+' | '-';
    phase: number;
    probability0: number; 
}

interface QuantumSystemSimulatorProps {
    embedded?: boolean;
}

const QuantumSystemSimulator: React.FC<QuantumSystemSimulatorProps> = ({ embedded = false }) => {
    const [qubits, setQubits] = useState<Qubit[]>(Array.from({ length: 16 }, (_, i) => ({ 
        id: i, 
        state: '0', 
        phase: 0,
        probability0: 1 
    })));
    const [isRunning, setIsRunning] = useState(true);
    const [coherence, setCoherence] = useState(100);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // --- Optimization: Simulation Logic Loop (Decoupled from Render) ---
    useEffect(() => {
        let interval: ReturnType<typeof setInterval>;
        if (isRunning) {
            interval = setInterval(() => {
                setQubits(prev => prev.map(q => {
                    const drift = (Math.random() - 0.5) * 0.15;
                    const phaseDrift = (Math.random() - 0.5) * 0.2;
                    
                    let newProb = q.probability0 + drift;
                    newProb = Math.max(0, Math.min(1, newProb));
                    
                    let newPhase = (q.phase + phaseDrift) % (2 * Math.PI);
                    
                    return { ...q, probability0: newProb, phase: newPhase };
                }));

                setCoherence(prev => Math.max(85, prev + (Math.random() - 0.5)));
            }, 50); // 20 ticks per second logic update
        }
        return () => clearInterval(interval);
    }, [isRunning]);

    // --- Advanced Rendering Engine ---
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let animationFrameId: number;
        let t = 0;

        const render = () => {
            t += 0.02;
            
            // Auto-resize handling for responsiveness
            if (canvas.parentElement) {
                const rect = canvas.parentElement.getBoundingClientRect();
                const dpr = window.devicePixelRatio || 1;
                if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
                    canvas.width = rect.width * dpr;
                    canvas.height = rect.height * dpr;
                    ctx.scale(dpr, dpr);
                    canvas.style.width = `${rect.width}px`;
                    canvas.style.height = `${rect.height}px`;
                }
            }
            
            // Logical dimensions
            const width = canvas.width / (window.devicePixelRatio || 1);
            const height = canvas.height / (window.devicePixelRatio || 1);
            
            ctx.clearRect(0, 0, width, height);

            // 1. Background: Quantum Interference Field
            if (isRunning) {
                const gradient = ctx.createLinearGradient(0, 0, width, height);
                gradient.addColorStop(0, "rgba(6, 182, 212, 0.02)");
                gradient.addColorStop(0.5, "rgba(168, 85, 247, 0.05)");
                gradient.addColorStop(1, "rgba(6, 182, 212, 0.02)");
                ctx.fillStyle = gradient;
                ctx.fillRect(0, 0, width, height);

                // Scanning interference line
                const scanY = (Math.sin(t * 0.5) + 1) / 2 * height;
                ctx.beginPath();
                ctx.moveTo(0, scanY);
                ctx.lineTo(width, scanY);
                ctx.strokeStyle = "rgba(6, 182, 212, 0.1)";
                ctx.lineWidth = 1;
                ctx.stroke();
            }

            // Grid Layout Calculation
            const cols = 4;
            // rows = 4 (implicit from length 16)
            const cellW = width / cols;
            const cellH = height / 4;

            // 2. Render Entanglement Beams (Background Layer)
            if (isRunning) {
                ctx.beginPath();
                ctx.strokeStyle = 'rgba(168, 85, 247, 0.15)'; // Purple-ish
                ctx.lineWidth = 1;
                for(let i=0; i<3; i++) {
                     // Dynamic random pairings based on time to simulate flux
                     const offset = Math.floor(t * 2) + i;
                     const q1Idx = offset % 16;
                     const q2Idx = (offset + 5) % 16;
                     
                     const c1 = q1Idx % cols; const r1 = Math.floor(q1Idx / cols);
                     const x1 = c1 * cellW + cellW / 2; const y1 = r1 * cellH + cellH / 2;
                     
                     const c2 = q2Idx % cols; const r2 = Math.floor(q2Idx / cols);
                     const x2 = c2 * cellW + cellW / 2; const y2 = r2 * cellH + cellH / 2;
                     
                     ctx.moveTo(x1, y1);
                     // Bezier curve for organic connection
                     const cpX = (x1 + x2) / 2 + Math.sin(t * 3 + i) * 30;
                     const cpY = (y1 + y2) / 2 + Math.cos(t * 3 + i) * 30;
                     ctx.quadraticCurveTo(cpX, cpY, x2, y2);
                }
                ctx.stroke();
            }

            // 3. Render Qubits (Bloch Spheres)
            qubits.forEach((q, i) => {
                const col = i % cols;
                const row = Math.floor(i / cols);
                const x = col * cellW + cellW / 2;
                const y = row * cellH + cellH / 2;
                const radius = Math.min(cellW, cellH) * 0.25;

                // Probability Visualization (Heatmap Color)
                const r = Math.floor(200 * (1 - q.probability0));
                const g = Math.floor(200 * (1 - Math.abs(0.5 - q.probability0) * 2));
                const b = Math.floor(255 * q.probability0);
                const color = `rgb(${r},${g},${b})`;

                // Outer Glow (Breathing effect)
                const pulse = 1 + Math.sin(t * 4 + i) * 0.1;
                
                // Sphere Gradient (3D Effect)
                const sphereGrad = ctx.createRadialGradient(x - radius/3, y - radius/3, 2, x, y, radius);
                sphereGrad.addColorStop(0, "rgba(255,255,255,0.9)");
                sphereGrad.addColorStop(0.3, color);
                sphereGrad.addColorStop(1, "rgba(0,0,0,0.4)");
                
                ctx.fillStyle = sphereGrad;
                ctx.beginPath();
                ctx.arc(x, y, radius * pulse, 0, 2 * Math.PI);
                ctx.fill();

                // Wireframe Orbital Ring
                ctx.strokeStyle = `rgba(255,255,255,0.2)`;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.ellipse(x, y, radius, radius * 0.4, t + i, 0, Math.PI * 2);
                ctx.stroke();

                // State Vector (Needle)
                const theta = (1 - q.probability0) * Math.PI; 
                const vecLen = radius * 1.2;
                // Visualise phase as rotation
                const vecX = x + vecLen * Math.sin(theta) * Math.sin(q.phase + t);
                const vecY = y - vecLen * Math.cos(theta);
                
                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(vecX, vecY);
                ctx.stroke();

                // Vector Head
                ctx.fillStyle = "#fff";
                ctx.beginPath();
                ctx.arc(vecX, vecY, 2, 0, Math.PI*2);
                ctx.fill();

                // ID Label (Only in full mode)
                if (!embedded) {
                    ctx.fillStyle = 'rgba(255,255,255,0.7)';
                    ctx.font = '10px "Share Tech Mono"';
                    ctx.textAlign = 'center';
                    ctx.fillText(`q${q.id}`, x, y + radius + 14);
                }
            });
            
            animationFrameId = requestAnimationFrame(render);
        };

        render();
        return () => cancelAnimationFrame(animationFrameId);
    }, [qubits, isRunning, embedded]);

    const applyHadamard = () => {
        setQubits(prev => prev.map(q => ({ ...q, probability0: 0.5, phase: 0 })));
    };

    const resetQubits = () => {
        setQubits(prev => prev.map(q => ({ ...q, probability0: 1, phase: 0 })));
        setIsRunning(false);
    };

    const content = (
        <div className="flex flex-col h-full p-2 gap-2 relative overflow-hidden">
             {embedded && (
                <div className="absolute top-2 right-2 z-10 flex gap-1 pointer-events-none">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <span className="text-[8px] text-green-400 font-mono">LIVE SIMULATION</span>
                </div>
            )}

            {/* Stats Bar (Hidden if embedded small view) */}
            {!embedded && (
                <div className="flex justify-between items-center bg-black/40 p-2 rounded border border-cyan-900/30">
                    <div className="flex items-center gap-2">
                        <ActivityIcon className="w-4 h-4 text-green-400" />
                        <span className="text-xs text-cyan-200">Coherence: {coherence.toFixed(2)}%</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <GridIcon className="w-4 h-4 text-purple-400" />
                        <span className="text-xs text-purple-200">Topology: Lattice (4x4)</span>
                    </div>
                </div>
            )}

            {/* Visualizer */}
            <div className={`flex-grow bg-black/20 border border-cyan-900/30 rounded-lg overflow-hidden relative ${embedded ? 'min-h-[120px]' : ''}`}>
                <canvas ref={canvasRef} className="w-full h-full block" />
                {!isRunning && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm pointer-events-none">
                        <p className="text-cyan-500 text-xs font-mono animate-pulse">SYSTEM IDLE</p>
                    </div>
                )}
            </div>

            {/* Controls */}
            {!embedded && (
                <div className="flex gap-2">
                    <button onClick={() => setIsRunning(!isRunning)} className={`flex-1 py-2 rounded text-xs font-bold flex items-center justify-center gap-2 transition-all ${isRunning ? 'bg-red-900/30 text-red-300 border border-red-500 hover:bg-red-900/50' : 'bg-green-900/30 text-green-300 border border-green-500 hover:bg-green-900/50'}`}>
                        {isRunning ? <StopIcon className="w-3 h-3"/> : <PlayIcon className="w-3 h-3"/>}
                        {isRunning ? 'Stop' : 'Run'}
                    </button>
                    <button onClick={applyHadamard} className="flex-1 py-2 rounded text-xs font-bold bg-blue-900/30 text-blue-300 border border-blue-500 hover:bg-blue-900/50 flex items-center justify-center gap-2">
                        <ZapIcon className="w-3 h-3" /> Apply H (All)
                    </button>
                    <button onClick={resetQubits} className="flex-1 py-2 rounded text-xs font-bold bg-gray-700/30 text-gray-300 border border-gray-500 hover:bg-gray-700/50 flex items-center justify-center gap-2">
                        <RefreshCwIcon className="w-3 h-3"/> Reset
                    </button>
                </div>
            )}
        </div>
    );

    if (embedded) return content;

    return (
        <GlassPanel title={<div className="flex items-center"><CpuChipIcon className="w-5 h-5 mr-2 text-cyan-400" /> Quantum System Simulator</div>}>
            {content}
        </GlassPanel>
    );
};

export default QuantumSystemSimulator;
