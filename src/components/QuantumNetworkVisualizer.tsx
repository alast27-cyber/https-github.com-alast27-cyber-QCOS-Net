
import React, { useEffect, useRef, useState } from 'react';
import GlassPanel from './GlassPanel';
import { Share2Icon, GlobeIcon, ServerCogIcon, ActivityIcon } from './Icons';

const QuantumNetworkVisualizer: React.FC = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [nodes, setNodes] = useState<{x: number, y: number, id: string, type: 'QAN' | 'DQN'}[]>([]);
    
    // Initialize random nodes
    useEffect(() => {
        const newNodes: {x: number, y: number, id: string, type: 'QAN' | 'DQN'}[] = [];
        // Center QAN
        newNodes.push({ x: 0.5, y: 0.5, id: 'QAN-ROOT', type: 'QAN' as const });
        // Satellite DQNs
        for(let i=0; i<8; i++) {
            newNodes.push({
                x: 0.1 + Math.random() * 0.8,
                y: 0.1 + Math.random() * 0.8,
                id: `DQN-${i}`,
                type: 'DQN' as const
            });
        }
        setTimeout(() => setNodes(newNodes), 0);
    }, []);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let frameId: number;
        let t = 0;

        const render = () => {
            if (canvas.parentElement) {
                canvas.width = canvas.parentElement.clientWidth;
                canvas.height = canvas.parentElement.clientHeight;
            }
            const w = canvas.width;
            const h = canvas.height;
            t += 0.02;

            ctx.clearRect(0, 0, w, h);

            // Draw Entanglement Links
            nodes.forEach((node, i) => {
                if (node.type === 'QAN') {
                    nodes.forEach((target, j) => {
                        if (i !== j) {
                            const nx = node.x * w;
                            const ny = node.y * h;
                            const tx = target.x * w;
                            const ty = target.y * h;

                            // Dynamic Line
                            ctx.beginPath();
                            ctx.moveTo(nx, ny);
                            ctx.quadraticCurveTo(
                                (nx + tx) / 2 + Math.sin(t + i) * 20,
                                (ny + ty) / 2 + Math.cos(t + j) * 20,
                                tx, ty
                            );
                            
                            const fidelity = 0.8 + Math.sin(t * 2 + j) * 0.1; // Pulsing fidelity
                            ctx.strokeStyle = `rgba(168, 85, 247, ${fidelity * 0.5})`;
                            ctx.lineWidth = 1;
                            ctx.stroke();

                            // Packet
                            if (Math.random() > 0.98) {
                                ctx.beginPath();
                                const p = (Math.sin(t * 5 + i) + 1) / 2;
                                const px = nx + (tx - nx) * p;
                                const py = ny + (ty - ny) * p;
                                ctx.fillStyle = '#fff';
                                ctx.arc(px, py, 2, 0, Math.PI * 2);
                                ctx.fill();
                            }
                        }
                    });
                }
            });

            // Draw Nodes
            nodes.forEach(node => {
                const x = node.x * w;
                const y = node.y * h;
                
                ctx.beginPath();
                ctx.arc(x, y, node.type === 'QAN' ? 15 : 8, 0, Math.PI * 2);
                ctx.fillStyle = node.type === 'QAN' ? '#22d3ee' : '#22c55e';
                ctx.fill();
                
                // Ring
                ctx.beginPath();
                ctx.arc(x, y, node.type === 'QAN' ? 20 : 12, 0, Math.PI * 2);
                ctx.strokeStyle = node.type === 'QAN' ? 'rgba(34, 211, 238, 0.5)' : 'rgba(34, 197, 94, 0.5)';
                ctx.lineWidth = 1;
                ctx.stroke();

                // Label
                ctx.fillStyle = '#fff';
                ctx.font = '10px monospace';
                ctx.textAlign = 'center';
                ctx.fillText(node.id, x, y + (node.type === 'QAN' ? 35 : 25));
            });

            frameId = requestAnimationFrame(render);
        };
        render();
        return () => cancelAnimationFrame(frameId);
    }, [nodes]);

    return (
        <GlassPanel title={<div className="flex items-center"><Share2Icon className="w-6 h-6 mr-2 text-purple-400" /> CHIPS Network Topology</div>}>
            <div className="flex flex-col h-full p-4 relative">
                <div className="absolute top-4 left-4 z-10 flex gap-4">
                    <div className="flex items-center gap-2 bg-black/60 px-3 py-1 rounded border border-cyan-800">
                        <GlobeIcon className="w-4 h-4 text-cyan-400" />
                        <span className="text-xs text-white font-bold">Global Mesh: Active</span>
                    </div>
                    <div className="flex items-center gap-2 bg-black/60 px-3 py-1 rounded border border-cyan-800">
                        <ActivityIcon className="w-4 h-4 text-purple-400" />
                        <span className="text-xs text-white font-bold">Entanglement: 94%</span>
                    </div>
                </div>
                <div className="flex-grow bg-black/40 border border-cyan-900/50 rounded-lg overflow-hidden relative">
                    <div className="absolute inset-0 holographic-grid opacity-20 pointer-events-none"></div>
                    <canvas ref={canvasRef} className="w-full h-full block" />
                </div>
                <div className="mt-4 flex gap-4 text-xs text-gray-400">
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-cyan-400"></div> QAN (Authority Node)
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-green-500"></div> DQN (Decentralized Node)
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-0.5 bg-purple-500"></div> EKS Link
                    </div>
                </div>
            </div>
        </GlassPanel>
    );
};

export default QuantumNetworkVisualizer;
