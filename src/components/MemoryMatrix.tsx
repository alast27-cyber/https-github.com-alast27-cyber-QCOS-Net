
import React, { useEffect, useState } from 'react';
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

    // Map messages to blocks whenever messages or activity changes
    useEffect(() => {
        // Initialize empty grid
        const newBlocks: MemoryBlock[] = Array.from({ length: GRID_SIZE }).map((_, i) => ({
            id: i,
            type: 'empty',
            content: '0x00',
            timestamp: 0,
            activity: 0
        }));

        // Fill with recent messages (reversed to show newest at top-left or sequential)
        // Let's fill sequentially from the start with recent messages
        // Take up to GRID_SIZE recent messages
        const recentMessages = messages.slice(-GRID_SIZE); // Get last 64
        
        recentMessages.forEach((msg, i) => {
            // Calculate a pseudo-random activity level based on message length and time
            const activity = Math.min(1, 0.4 + (msg.text.length / 200));
            
            newBlocks[i] = {
                id: i,
                type: msg.sender === 'user' ? 'user' : 'agent',
                content: msg.text,
                timestamp: Date.now(), 
                activity: activity
            };
        });

        // Fill some remaining spots with "System" memory artifacts (simulated)
        for (let i = recentMessages.length; i < GRID_SIZE; i++) {
             // Only some spots
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

        setBlocks(newBlocks);
    }, [messages, lastActivity]);

    const getBlockColor = (type: string, activity: number) => {
        const alpha = 0.4 + (activity * 0.6); // Base opacity + activity
        switch (type) {
            case 'user': return `rgba(168, 85, 247, ${alpha})`; // Purple (User)
            case 'agent': return `rgba(6, 182, 212, ${alpha})`; // Cyan (Agent)
            case 'system': return `rgba(34, 197, 94, ${alpha})`; // Green (System)
            default: return `rgba(30, 41, 59, 0.2)`; // Slate/Empty
        }
    };

    const selectedBlock = blocks.find(b => b.id === selectedBlockId);

    return (
        <div className="flex flex-col h-full gap-2 overflow-hidden">
            {/* Header / Stats */}
            <div className="flex justify-between items-center px-1">
                <span className="text-[10px] text-cyan-500 font-bold uppercase tracking-wider">Memory Allocation</span>
                <span className="text-[10px] text-gray-500 font-mono">{messages.length} Segments</span>
            </div>

            {/* The Matrix Grid */}
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
                        title={block.type !== 'empty' ? `${block.type.toUpperCase()}: ${block.content.substring(0, 20)}...` : 'Unallocated'}
                    >
                        {/* Scanline Effect on active blocks */}
                        {block.type !== 'empty' && (
                            <div className="absolute inset-0 bg-gradient-to-b from-transparent via-white/10 to-transparent translate-y-[-100%] group-hover:translate-y-[100%] transition-transform duration-1000" />
                        )}
                    </button>
                ))}
            </div>

            {/* Inspector View */}
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
                        
                        <div className="mt-2 pt-2 border-t border-cyan-900/30 grid grid-cols-2 gap-2 text-[10px] text-gray-400 font-mono flex-shrink-0">
                            <div className="flex items-center gap-1">
                                <ChartBarIcon className="w-3 h-3 text-cyan-600" />
                                <span>Coherence: {(selectedBlock.activity * 100).toFixed(1)}%</span>
                            </div>
                            <div className="text-right">
                                Decay: {(Math.random() * 0.01).toFixed(4)}/ms
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="flex flex-col items-center justify-center h-full text-cyan-600/50">
                        <div className="flex gap-1 mb-2 opacity-50">
                            <div className="w-1.5 h-1.5 bg-current rounded-full animate-bounce" style={{animationDelay:'0s'}}/>
                            <div className="w-1.5 h-1.5 bg-current rounded-full animate-bounce" style={{animationDelay:'0.1s'}}/>
                            <div className="w-1.5 h-1.5 bg-current rounded-full animate-bounce" style={{animationDelay:'0.2s'}}/>
                        </div>
                        <p className="text-[10px] font-mono uppercase tracking-wide">Select block to decode</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default MemoryMatrix;
