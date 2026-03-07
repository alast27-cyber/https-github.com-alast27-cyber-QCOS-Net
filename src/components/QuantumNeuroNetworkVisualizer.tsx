
import React, { useState, useEffect } from 'react';
import { BrainCircuitIcon, SparklesIcon } from './Icons';

interface Node {
    id: number;
    x: number;
    y: number;
    size: number;
}

const QuantumNeuroNetworkVisualizer: React.FC = () => {
    const [nodes, setNodes] = useState<Node[]>([]);

    useEffect(() => {
        setTimeout(() => {
            setNodes(Array.from({ length: 12 }).map((_, i) => ({
                id: i,
                x: 20 + Math.random() * 60,
                y: 20 + Math.random() * 60,
                size: 2 + Math.random() * 4
            })));
        }, 0);
    }, []);

    return (
        <div className="h-full flex flex-col p-4 bg-black/40 rounded-lg border border-purple-900/30 overflow-hidden relative">
            <div className="flex justify-between items-center mb-4 z-10">
                <div className="flex items-center gap-2">
                    <BrainCircuitIcon className="w-4 h-4 text-purple-400" />
                    <span className="text-xs font-bold uppercase tracking-widest text-purple-200">QNN Topology</span>
                </div>
                <div className="flex items-center gap-1 text-[9px] text-purple-500 font-mono">
                    <SparklesIcon className="w-3 h-3 animate-spin-slow" />
                    SYNAPTIC_FLUX: <span className="text-white">OPTIMAL</span>
                </div>
            </div>

            <div className="flex-grow relative border border-purple-900/20 rounded-lg bg-purple-950/10 overflow-hidden">
                <svg className="absolute inset-0 w-full h-full">
                    <defs>
                        <radialGradient id="nodeGlow" cx="50%" cy="50%" r="50%">
                            <stop offset="0%" stopColor="rgba(168, 85, 247, 0.6)" />
                            <stop offset="100%" stopColor="rgba(168, 85, 247, 0)" />
                        </radialGradient>
                    </defs>
                    
                    {/* Connections */}
                    {nodes.map((node, i) => (
                        nodes.slice(i + 1, i + 3).map((target, j) => (
                            <line 
                                key={`${i}-${j}`}
                                x1={`${node.x}%`} y1={`${node.y}%`}
                                x2={`${target.x}%`} y2={`${target.y}%`}
                                stroke="rgba(168, 85, 247, 0.2)"
                                strokeWidth="1"
                                className="animate-pulse"
                            />
                        ))
                    ))}

                    {/* Nodes */}
                    {nodes.map(node => (
                        <g key={node.id} className="animate-pulse" style={{ animationDelay: `${node.id * 0.1}s` }}>
                            <circle cx={`${node.x}%`} cy={`${node.y}%`} r={node.size + 4} fill="url(#nodeGlow)" />
                            <circle cx={`${node.x}%`} cy={`${node.y}%`} r={node.size} fill="#a855f7" />
                        </g>
                    ))}
                </svg>
                
                <div className="absolute bottom-2 left-2 right-2 flex justify-between text-[8px] font-mono text-purple-400/60 uppercase">
                    <span>Active Neurons: 12,288</span>
                    <span>Coherence: 99.4%</span>
                </div>
            </div>

            <div className="absolute inset-0 pointer-events-none bg-gradient-to-t from-purple-900/10 to-transparent"></div>
        </div>
    );
};

export default QuantumNeuroNetworkVisualizer;
