
import React, { useEffect, useState, useRef } from 'react';
import { CpuChipIcon, SparklesIcon, UsersIcon, ChartBarIcon } from './Icons';
import { Message } from '../utils/agentUtils';

const GRID_SIZE = 64; // 8x8 grid

interface MemoryBlock {
    id: number;
    type: 'user' | 'agent' | 'system' | 'empty';
    content: string;
    timestamp: number;
    activity: number; // 0-1 for visual intensity
}

interface MemoryMatrixProps {
    lastActivity: number;
    memorySummary: string | null;
    interactive?: boolean;
    messages?: Message[];
}

const MemoryMatrix: React.FC<MemoryMatrixProps> = ({ lastActivity, memorySummary, interactive = false, messages = [] }) => {
    const [blocks, setBlocks] = useState<MemoryBlock[]>([]);
    const [selectedBlockId, setSelectedBlockId] = useState<number | null>(null);
    const [decay, setDecay] = useState(0);

    const decayRef = useRef(0);
    useEffect(() => {
        decayRef.current = Math.random() * 0.01;
        setTimeout(() => setDecay(decayRef.current), 0);
    }, [selectedBlockId]);

    useEffect(() => {
        const newBlocks: MemoryBlock[] = Array.from({ length: GRID_SIZE }).map((_, i) => ({
            id: i,
            type: 'empty',
            content: '0x00',
            timestamp: 0,
            activity: 0
        }));

        const recentMessages = messages.slice(-GRID_SIZE);
        
        recentMessages.forEach((msg, i) => {
            const activity = Math.min(1, 0.4 + (msg.text.length / 200));
            
            newBlocks[i] = {
                id: i,
                type: msg.sender === 'user' ? 'user' : 'agent',
                content: msg.text,
                timestamp: Date.now(), 
                activity: activity
            };
        });

        for (let i = recentMessages.length; i < GRID_SIZE; i++) {
             if (Math.random() > 0.85) {
                 newBlocks[i] = {
                     id: i,
                     type: 'system',
                     content: `[SYSTEM_OP]: Optimizing neural pathway ${i}...`,
                     timestamp: Date.now(),
                     activity: Math.random() * 0.3
                 };
             }
        }

        // Wrap in setTimeout to avoid synchronous setState inside useEffect
        setTimeout(() => setBlocks(newBlocks), 0);
    }, [messages, lastActivity]);

    const getBlockColor = (type: string, activity: number) => {
        const alpha = 0.4 + (activity * 0.6);
        switch (type) {
            case 'user': return `rgba(168, 85, 247, ${alpha})`;
            case 'agent': return `rgba(6, 182, 212, ${alpha})`;
            case 'system': return `rgba(34, 197, 94, ${alpha})`;
            default: return `rgba(30, 41, 59, 0.2)`;
        }
    };

    const selectedBlock = blocks.find(b => b.id === selectedBlockId);

    return (
        <div className="flex flex-col h-full gap-2 overflow-hidden">
            <div className="flex justify-between items-center px-1">
                <span className="text-[10px] text-cyan-500 font-bold uppercase tracking-wider">Memory Allocation</span>
                <span className="text-[10px] text-gray-500 font-mono">{messages.length} Segments</span>
            </div>

            <div className="grid grid-cols-8 gap-1 p-1 bg-black/20 rounded border border-cyan-900/30 flex-shrink-0">
                {blocks.map(block => (
                    <button
                        key={block.id}
                        onClick={() => block.type !== 'empty' && setSelectedBlockId(block.id)}
                        disabled={block.type === 'empty'}
                        className={`
                            aspect-square rounded-[1px] transition-all duration-300 relative group overflow-hidden
                            ${selectedBlockId === block.id ? 'ring-1 ring-white z-10 shadow-[0_0_10px_rgba(255,255,255,0.5)]' : ''}
                            ${block.type === 'empty' ? 'cursor-default' : 'cursor-pointer hover:brightness-125'}
                        `}
                        style={{ backgroundColor: getBlockColor(block.type, block.activity) }}
                    >
                        {block.type !== 'empty' && (
                            <div className="absolute inset-0 bg-gradient-to-b from-transparent via-white/10 to-transparent translate-y-[-100%] group-hover:translate-y-[100%] transition-transform duration-1000" />
                        )}
                    </button>
                ))}
            </div>

            <div className="flex-grow bg-cyan-950/20 border border-cyan-900/50 rounded-lg p-3 relative overflow-hidden flex flex-col min-h-0">
                {selectedBlock ? (
                    <div className="animate-fade-in flex flex-col h-full">
                        <div className="flex justify-between items-start mb-2 border-b border-cyan-800/30 pb-2 flex-shrink-0">
                            <div className="flex items-center gap-2">
                                {selectedBlock.type === 'user' && <UsersIcon className="w-4 h-4 text-purple-400" />}
                                {selectedBlock.type === 'agent' && <SparklesIcon className="w-4 h-4 text-cyan-400" />}
                                {selectedBlock.type === 'system' && <CpuChipIcon className="w-4 h-4 text-green-400" />}
                                <span className={`text-xs font-bold uppercase ${selectedBlock.type === 'user' ? 'text-purple-300' : selectedBlock.type === 'agent' ? 'text-cyan-300' : 'text-green-300'}`}>
                                    {selectedBlock.type} MEMORY
                                </span>
                            </div>
                            <span className="text-[10px] font-mono text-cyan-600">ADDR: 0x{selectedBlock.id.toString(16).padStart(4, '0').toUpperCase()}</span>
                        </div>
                        
                        <div className="flex-grow overflow-y-auto custom-scrollbar">
                            <p className="text-xs font-mono text-cyan-100 whitespace-pre-wrap leading-relaxed">
                                {selectedBlock.content}
                            </p>
                        </div>
                    </div>
                ) : (
                    <div className="flex flex-col items-center justify-center h-full text-cyan-600/50">
                        <p className="text-[10px] font-mono uppercase tracking-wide">Select block to decode</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default MemoryMatrix;
