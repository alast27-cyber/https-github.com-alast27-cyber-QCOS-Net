
import React, { useState, useEffect, useRef } from 'react';
import GlassPanel from './GlassPanel';
import { CpuChipIcon, PlayIcon, StopIcon, RefreshCwIcon } from './Icons';

interface Qubit {
    id: number;
    state: '0' | '1' | '+' | '-';
    probability0: number; // Probability of measuring |0>
}

const QubitSimulator: React.FC = () => {
    const [qubits, setQubits] = useState<Qubit[]>(Array.from({ length: 5 }, (_, i) => ({ id: i, state: '0', probability0: 1 })));
    const [isRunning, setIsRunning] = useState(false);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Simulation Loop
    useEffect(() => {
        let interval: ReturnType<typeof setInterval>;
        if (isRunning) {
            interval = setInterval(() => {
                setQubits(prev => prev.map(q => {
                    // Random fluctuation in probability (decoherence simulation)
                    const drift = (Math.random() - 0.5) * 0.1;
                    let newProb = q.probability0 + drift;
                    newProb = Math.max(0, Math.min(1, newProb));
                    return { ...q, probability0: newProb };
                }));
            }, 100);
        }
        return () => clearInterval(interval);
    }, [isRunning]);

    // Canvas Rendering
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const render = () => {
            const width = canvas.width;
            const height = canvas.height;
            ctx.clearRect(0, 0, width, height);

            qubits.forEach((q, i) => {
                const x = (i + 1) * (width / (qubits.length + 1));
                const y = height / 2;
                const radius = 20;

                // Bloch Sphere Representation (Simplified 2D)
                ctx.beginPath();
                ctx.arc(x, y, radius, 0, 2 * Math.PI);
                ctx.strokeStyle = '#06b6d4'; // Cyan
                ctx.lineWidth = 2;
                ctx.stroke();

                // State Vector
                const angle = (1 - q.probability0) * Math.PI; // Map prob to angle (0 to PI)
                const vecX = x + radius * Math.sin(angle);
                const vecY = y - radius * Math.cos(angle);

                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(vecX, vecY);
                ctx.strokeStyle = '#facc15'; // Yellow
                ctx.lineWidth = 3;
                ctx.stroke();

                // Label
                ctx.fillStyle = '#fff';
                ctx.font = '10px monospace';
                ctx.textAlign = 'center';
                ctx.fillText(`q${q.id}`, x, y + radius + 15);
                ctx.fillText(`${(q.probability0 * 100).toFixed(0)}% |0>`, x, y + radius + 25);
            });
        };

        const animationId = requestAnimationFrame(render);
        return () => cancelAnimationFrame(animationId);
    }, [qubits]);

    const applyHadamard = () => {
        setQubits(prev => prev.map(q => ({ ...q, probability0: 0.5 })));
    };

    const resetQubits = () => {
        setQubits(prev => prev.map(q => ({ ...q, probability0: 1 })));
        setIsRunning(false);
    };

    return (
        <GlassPanel title={<div className="flex items-center"><CpuChipIcon className="w-5 h-5 mr-2 text-purple-400" /> Qubit Simulator</div>}>
            <div className="flex flex-col h-full p-4 gap-4">
                <div className="flex-grow bg-black/40 border border-cyan-900/30 rounded-lg overflow-hidden relative">
                    <canvas ref={canvasRef} width={400} height={200} className="w-full h-full" />
                </div>
                <div className="flex gap-2">
                    <button onClick={() => setIsRunning(!isRunning)} className={`flex-1 py-2 rounded text-xs font-bold flex items-center justify-center gap-2 ${isRunning ? 'bg-red-900/30 text-red-300 border border-red-500' : 'bg-green-900/30 text-green-300 border border-green-500'}`}>
                        {isRunning ? <StopIcon className="w-4 h-4"/> : <PlayIcon className="w-4 h-4"/>}
                        {isRunning ? 'Stop Noise' : 'Start Simulation'}
                    </button>
                    <button onClick={applyHadamard} className="flex-1 py-2 rounded text-xs font-bold bg-blue-900/30 text-blue-300 border border-blue-500">
                        Apply Hadamard (H)
                    </button>
                    <button onClick={resetQubits} className="flex-1 py-2 rounded text-xs font-bold bg-gray-700/30 text-gray-300 border border-gray-500 flex items-center justify-center gap-2">
                        <RefreshCwIcon className="w-4 h-4"/> Reset
                    </button>
                </div>
            </div>
        </GlassPanel>
    );
};

export default QubitSimulator;
