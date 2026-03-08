import React, { useState, useEffect, useRef } from 'react';
import { SparklesIcon, RefreshCwIcon, LayersIcon, CpuChipIcon } from './Icons';

const QuantumGenerativeModel: React.FC = () => {
    const [isGenerating, setIsGenerating] = useState(false);
    const [progress, setProgress] = useState(0);
    const [generationType, setGenerationType] = useState<'TEXT' | 'IMAGE' | 'AUDIO' | 'CODE'>('IMAGE');
    const [latentVector, setLatentVector] = useState<number[]>(Array(16).fill(0));
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Simulate latent space fluctuations
    useEffect(() => {
        const interval = setInterval(() => {
            if (!isGenerating) {
                setLatentVector(prev => prev.map(v => Math.max(0, Math.min(1, v + (Math.random() - 0.5) * 0.1))));
            }
        }, 100);
        return () => clearInterval(interval);
    }, [isGenerating]);

    // Simulate generation process
    useEffect(() => {
        if (isGenerating) {
            const interval = setInterval(() => {
                setProgress(prev => {
                    if (prev >= 100) {
                        setIsGenerating(false);
                        return 0;
                    }
                    return prev + 2;
                });
            }, 50);
            return () => clearInterval(interval);
        }
    }, [isGenerating]);

    // Visualizer for latent space
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const draw = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const w = canvas.width;
            const h = canvas.height;
            
            // Draw grid
            ctx.strokeStyle = 'rgba(168, 85, 247, 0.1)';
            ctx.beginPath();
            for(let i=0; i<w; i+=20) { ctx.moveTo(i,0); ctx.lineTo(i,h); }
            for(let i=0; i<h; i+=20) { ctx.moveTo(0,i); ctx.lineTo(w,i); }
            ctx.stroke();

            // Draw latent points
            latentVector.forEach((val, i) => {
                const x = (i / latentVector.length) * w + (w / latentVector.length) / 2;
                const y = h - (val * h);
                
                ctx.fillStyle = isGenerating ? '#22d3ee' : '#a855f7';
                ctx.beginPath();
                ctx.arc(x, y, isGenerating ? 4 : 2, 0, Math.PI * 2);
                ctx.fill();

                // Connect points
                if (i > 0) {
                    const prevX = ((i - 1) / latentVector.length) * w + (w / latentVector.length) / 2;
                    const prevY = h - (latentVector[i-1] * h);
                    ctx.strokeStyle = isGenerating ? 'rgba(34, 211, 238, 0.5)' : 'rgba(168, 85, 247, 0.3)';
                    ctx.beginPath();
                    ctx.moveTo(prevX, prevY);
                    ctx.lineTo(x, y);
                    ctx.stroke();
                }
            });
        };

        const anim = requestAnimationFrame(draw);
        return () => cancelAnimationFrame(anim);
    }, [latentVector, isGenerating]);

    return (
        <div className="h-full flex flex-col bg-black/40 border border-purple-500/30 rounded-xl p-4 relative overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-br from-purple-900/10 via-transparent to-cyan-900/10 opacity-0 group-hover:opacity-100 transition-opacity duration-700"></div>
            
            <div className="flex justify-between items-center mb-4 z-10">
                <div className="flex items-center gap-2">
                    <div className="p-1.5 bg-purple-500/20 rounded-lg border border-purple-500/50">
                        <SparklesIcon className="w-4 h-4 text-purple-400" />
                    </div>
                    <div>
                        <h3 className="text-sm font-bold text-purple-100 tracking-wider">QUANTUM GENERATIVE MODEL</h3>
                        <p className="text-[10px] text-purple-400/70 font-mono">Q-GAN ARCHITECTURE v4.2</p>
                    </div>
                </div>
                <div className="flex gap-2">
                    {['TEXT', 'IMAGE', 'AUDIO', 'CODE'].map((type) => (
                        <button 
                            key={type}
                            onClick={() => setGenerationType(type as any)}
                            className={`text-[8px] px-2 py-1 rounded border transition-all ${generationType === type ? 'bg-purple-500/30 border-purple-400 text-purple-200' : 'bg-black/30 border-gray-800 text-gray-500 hover:border-purple-500/30'}`}
                        >
                            {type}
                        </button>
                    ))}
                </div>
            </div>

            <div className="flex-grow relative z-10 border border-purple-500/20 rounded-lg bg-black/50 overflow-hidden mb-4">
                <canvas ref={canvasRef} width={400} height={200} className="w-full h-full opacity-80" />
                
                {isGenerating && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm">
                        <div className="text-center">
                            <RefreshCwIcon className="w-8 h-8 text-cyan-400 animate-spin mx-auto mb-2" />
                            <p className="text-xs font-mono text-cyan-300 animate-pulse">GENERATING {generationType}...</p>
                            <p className="text-[10px] text-cyan-500/70 mt-1">Collapsing Wave Function: {progress}%</p>
                        </div>
                    </div>
                )}
            </div>

            <div className="grid grid-cols-2 gap-3 z-10">
                <div className="bg-purple-900/10 p-2 rounded border border-purple-500/20">
                    <div className="flex justify-between items-center mb-1">
                        <span className="text-[10px] text-purple-400">FID SCORE</span>
                        <span className="text-xs font-mono text-purple-200">2.45</span>
                    </div>
                    <div className="h-1 bg-purple-900/30 rounded-full overflow-hidden">
                        <div className="h-full bg-purple-500 w-[92%]"></div>
                    </div>
                </div>
                <div className="bg-cyan-900/10 p-2 rounded border border-cyan-500/20">
                    <div className="flex justify-between items-center mb-1">
                        <span className="text-[10px] text-cyan-400">DIVERSITY</span>
                        <span className="text-xs font-mono text-cyan-200">0.98</span>
                    </div>
                    <div className="h-1 bg-cyan-900/30 rounded-full overflow-hidden">
                        <div className="h-full bg-cyan-500 w-[88%]"></div>
                    </div>
                </div>
            </div>

            <button 
                onClick={() => setIsGenerating(true)}
                disabled={isGenerating}
                className={`mt-4 w-full py-2 rounded border text-xs font-bold tracking-widest transition-all ${isGenerating ? 'bg-gray-800 border-gray-700 text-gray-500 cursor-not-allowed' : 'bg-purple-500/20 border-purple-500/50 text-purple-300 hover:bg-purple-500/30 hover:text-white hover:shadow-[0_0_15px_rgba(168,85,247,0.3)]'}`}
            >
                {isGenerating ? 'PROCESSING...' : 'INITIATE GENERATION SEQUENCE'}
            </button>
        </div>
    );
};

export default QuantumGenerativeModel;
