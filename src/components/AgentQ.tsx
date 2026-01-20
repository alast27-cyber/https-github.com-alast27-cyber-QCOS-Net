import React, { useState, useRef, useEffect, useMemo } from 'react';
import { MessageSquareIcon, XIcon, SendIcon, LoaderIcon, MicIcon, MicOffIcon, RefreshCwIcon, PaperclipIcon, FileIcon, BrainCircuitIcon, SearchIcon, CheckCircle2Icon, SparklesIcon, ActivityIcon, LinkIcon, CpuChipIcon, GalaxyIcon, AlertTriangleIcon, LayoutGridIcon, ArrowRightIcon, StopIcon, PlayIcon } from './Icons';
import { Message, ChartData } from '../utils/agentUtils';
import { 
    ResponsiveContainer, LineChart, Line, BarChart, Bar, AreaChart, Area, 
    XAxis, YAxis, CartesianGrid, Tooltip, Legend 
} from 'recharts';
import { useSimulation } from '../context/SimulationContext';
import MemoryMatrix from './MemoryMatrix';

interface AgentQProps {
  isOpen: boolean;
  onToggleOpen: () => void;
  messages: Message[];
  isLoading: boolean;
  onSendMessage: (message: string, file: File | null) => void;
  embedded?: boolean;
  fullScreen?: boolean;
  lastActivity?: number;
  isTtsEnabled?: boolean;
  onToggleTts?: () => void;
  memorySummary?: string | null;
  onClearMemory?: () => void;
  activeContext?: string | null;
  focusedPanelId?: string | null;
  activeActions?: string[];
  suggestedActions?: string[];
  triggerClassName?: string;
  onDeployApp?: (details: any) => void;
  currentContextName?: string;
}

const AgentQ: React.FC<AgentQProps> = ({ 
    isOpen, onToggleOpen, messages, isLoading, onSendMessage, 
    embedded = false, fullScreen = false, lastActivity,
    isTtsEnabled, onToggleTts, memorySummary, onClearMemory,
    activeContext, focusedPanelId, activeActions = [], 
    suggestedActions = [], triggerClassName, onDeployApp, currentContextName
}) => {
    // Safety Fallbacks for QLLM Status
    const simulation = useSimulation();
    const qllm = simulation?.qllm ?? { status: 'offline', efficiencyBoost: 1.0, isActive: false };
    const systemStatus = simulation?.systemStatus ?? { status: 'offline' };

    const [input, setInput] = useState('');
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Determines orb color based on QLLM status
    const orbStatusColor = useMemo(() => {
        if (qllm?.status === 'training') return 'border-purple-500 shadow-[0_0_30px_rgba(168,85,247,0.5)]';
        if (qllm?.status === 'synced') return 'border-cyan-400 shadow-[0_0_30px_rgba(34,211,238,0.5)]';
        return 'border-gray-600 shadow-none opacity-50';
    }, [qllm?.status]);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, isLoading]);

    const handleSend = () => {
        if (input.trim() || selectedFile) {
            onSendMessage(input, selectedFile);
            setInput('');
            setSelectedFile(null);
        }
    };

    if (embedded && !isOpen) return null;

    return (
        <div className={embedded ? "h-full w-full flex flex-col bg-black/40 rounded-xl border border-white/10 overflow-hidden" : 
            `fixed bottom-20 left-[24rem] w-[450px] max-h-[600px] flex flex-col bg-slate-950/90 backdrop-blur-2xl rounded-2xl border border-cyan-500/30 shadow-[0_0_50px_rgba(0,0,0,0.8)] z-50 transition-all duration-500 transform ${isOpen ? 'scale-100 translate-y-0 opacity-100' : 'scale-95 translate-y-10 opacity-0 pointer-events-none'}`
        }>
            {/* Header */}
            <div className="p-4 border-b border-white/10 flex items-center justify-between bg-gradient-to-r from-cyan-950/40 to-transparent">
                <div className="flex items-center gap-3">
                    <div className={`w-3 h-3 rounded-full animate-pulse ${qllm?.status === 'synced' ? 'bg-cyan-400' : 'bg-gray-600'}`} />
                    <div>
                        <h3 className="text-xs font-black text-white tracking-widest uppercase">Agent Q Intelligence</h3>
                        <p className="text-[9px] text-cyan-500/70 font-mono">STATUS: {qllm?.status?.toUpperCase() ?? 'DISCONNECTED'}</p>
                    </div>
                </div>
                {!embedded && (
                    <button onClick={onToggleOpen} className="p-1 hover:bg-white/10 rounded-lg transition-colors">
                        <XIcon className="w-5 h-5 text-gray-500" />
                    </button>
                )}
            </div>

            {/* Chat Body */}
            <div ref={scrollRef} className="flex-grow overflow-y-auto p-4 space-y-4 custom-scrollbar bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')]">
                {messages.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center opacity-40 space-y-4 py-10">
                        <BrainCircuitIcon className="w-16 h-16 text-cyan-400 animate-pulse" />
                        <p className="text-xs font-mono text-center max-w-[200px]">Waiting for Neural Link Input...</p>
                    </div>
                ) : (
                    messages.map((msg, i) => (
                        <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`max-w-[85%] p-3 rounded-2xl text-xs leading-relaxed ${
                                msg.role === 'user' 
                                ? 'bg-cyan-600/20 border border-cyan-500/30 text-cyan-50 text-right rounded-tr-none' 
                                : 'bg-white/5 border border-white/10 text-gray-300 rounded-tl-none'
                            }`}>
                                {msg.content}
                            </div>
                        </div>
                    ))
                )}
                {isLoading && (
                    <div className="flex justify-start">
                        <div className="bg-white/5 border border-white/10 p-3 rounded-2xl rounded-tl-none">
                            <LoaderIcon className="w-4 h-4 text-cyan-400 animate-spin" />
                        </div>
                    </div>
                )}
            </div>

            {/* Input Footer */}
            <div className="p-4 border-t border-white/10 bg-black/40">
                <div className="relative flex items-center gap-2">
                    <input 
                        type="text" 
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                        placeholder="Relay command to Q..."
                        className="w-full bg-white/5 border border-white/10 rounded-xl py-3 px-4 text-xs text-white placeholder:text-gray-600 focus:outline-none focus:border-cyan-500/50 transition-all"
                    />
                    <button 
                        onClick={handleSend}
                        disabled={!input.trim() && !selectedFile}
                        className="p-3 bg-cyan-600/20 border border-cyan-500/40 rounded-xl text-cyan-400 hover:bg-cyan-600/40 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                    >
                        <SendIcon className="w-4 h-4" />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default AgentQ;