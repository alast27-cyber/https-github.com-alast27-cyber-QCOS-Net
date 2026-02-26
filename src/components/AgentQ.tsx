import React, { useState, useRef, useEffect, useMemo } from 'react';
import { MessageSquareIcon, XIcon, SendIcon, LoaderIcon, MicIcon, MicOffIcon, RefreshCwIcon, PaperclipIcon, FileIcon, BrainCircuitIcon, SearchIcon, CheckCircle2Icon, SparklesIcon, ActivityIcon, LinkIcon, CpuChipIcon, GalaxyIcon, AlertTriangleIcon, LayoutGridIcon, ArrowRightIcon } from './Icons';
import { Message, ChartData } from '../utils/agentUtils';
import { 
    ResponsiveContainer, LineChart, Line, BarChart, Bar, AreaChart, Area, 
    XAxis, YAxis, CartesianGrid, Tooltip, Legend 
} from 'recharts';
import { useSimulation } from '../context/SimulationContext';

// Local Component Import
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
  isSpeaking?: boolean;
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
    isOpen, 
    onToggleOpen, 
    messages, 
    isLoading, 
    onSendMessage, 
    embedded = false,
    fullScreen = false,
    lastActivity,
    isTtsEnabled,
    onToggleTts,
    isSpeaking,
    onClearMemory,
    memorySummary,
    activeContext,
    focusedPanelId,
    activeActions = [],
    suggestedActions = [],
    triggerClassName = "fixed bottom-4 left-64 z-50",
    currentContextName
}) => {
  const { inquiry, systemStatus, universeConnections } = useSimulation();
  const [defaultLastActivity] = useState(() => Date.now());
  const [input, setInput] = useState('');
  const [attachedFile, setAttachedFile] = useState<File | null>(null);
  const [showMemory, setShowMemory] = useState(false);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const displayContext = activeContext || currentContextName;

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (isOpen && !showMemory && !searchQuery) {
      scrollToBottom();
    }
  }, [messages, isOpen, showMemory, searchQuery]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setAttachedFile(e.target.files[0]);
    }
  };

  const handleSend = () => {
    if ((input.trim() || attachedFile) && !isLoading) {
      onSendMessage(input, attachedFile);
      setInput('');
      setAttachedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSend();
    }
  };

  const filteredMessages = useMemo(() => {
    if (!searchQuery) return messages;
    return messages.filter(m => 
      m.text.toLowerCase().includes(searchQuery.toLowerCase()) ||
      m.attachment?.name.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [messages, searchQuery]);

  const highlightText = (text: string) => {
    if (!searchQuery) return text;
    const parts = text.split(new RegExp(`(${searchQuery})`, 'gi'));
    return parts.map((part, i) => 
        part.toLowerCase() === searchQuery.toLowerCase() 
            ? <span key={i} className="bg-yellow-500/40 text-white font-bold rounded px-0.5">{part}</span> 
            : part
    );
  };

  const renderChart = (chartData: ChartData) => {
      const commonProps = {
          data: chartData.data,
          margin: { top: 5, right: 5, left: -20, bottom: 0 }
      };

      const renderContent = () => {
          switch (chartData.type) {
              case 'line':
                  return (
                      <LineChart {...commonProps}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                          <XAxis dataKey="name" stroke="rgba(255,255,255,0.5)" tick={{fontSize: 10}} />
                          <YAxis stroke="rgba(255,255,255,0.5)" tick={{fontSize: 10}} />
                          <Tooltip contentStyle={{backgroundColor: '#000', border: '1px solid #06b6d4'}} itemStyle={{color: '#fff'}} />
                          <Legend />
                          <Line type="monotone" dataKey="value" name={chartData.dataKey} stroke={chartData.color || "#06b6d4"} strokeWidth={2} dot={{r: 3}} />
                      </LineChart>
                  );
              case 'bar':
                  return (
                      <BarChart {...commonProps}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                          <XAxis dataKey="name" stroke="rgba(255,255,255,0.5)" tick={{fontSize: 10}} />
                          <YAxis stroke="rgba(255,255,255,0.5)" tick={{fontSize: 10}} />
                          <Tooltip contentStyle={{backgroundColor: '#000', border: '1px solid #06b6d4'}} itemStyle={{color: '#fff'}} />
                          <Legend />
                          <Bar dataKey="value" name={chartData.dataKey} fill={chartData.color || "#06b6d4"} radius={[4, 4, 0, 0]} />
                      </BarChart>
                  );
              case 'area':
                  return (
                      <AreaChart {...commonProps}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                          <XAxis dataKey="name" stroke="rgba(255,255,255,0.5)" tick={{fontSize: 10}} />
                          <YAxis stroke="rgba(255,255,255,0.5)" tick={{fontSize: 10}} />
                          <Tooltip contentStyle={{backgroundColor: '#000', border: '1px solid #06b6d4'}} itemStyle={{color: '#fff'}} />
                          <Legend />
                          <Area type="monotone" dataKey="value" name={chartData.dataKey} stroke={chartData.color || "#06b6d4"} fill={chartData.color || "#06b6d4"} fillOpacity={0.3} />
                      </AreaChart>
                  );
              default:
                  return null;
          }
      };

      return (
          <div className="w-full h-48 mt-2 mb-2 bg-black/40 rounded-lg border border-cyan-900/50 p-2">
              <p className="text-xs text-cyan-400 font-bold mb-2 text-center uppercase tracking-wider">{chartData.title}</p>
              <ResponsiveContainer width="100%" height="85%">
                  {renderContent() || <div/>}
              </ResponsiveContainer>
          </div>
      );
  };

  const isDataLinked = inquiry.status === 'simulating' || inquiry.status === 'optimizing';

  const chatHistoryView = (
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden relative">
          <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-cyan-900 scrollbar-track-transparent">
              {filteredMessages.length === 0 && !isLoading && (
                  <div className="flex flex-col items-center justify-center h-full opacity-60 px-4">
                      <SparklesIcon className="w-12 h-12 text-cyan-500 mb-4 animate-pulse" />
                      <p className="text-sm text-cyan-100 text-center font-mono">Quantum Intelligence Active. Select an entry point or describe your intent.</p>
                      <div className="grid grid-cols-2 gap-3 w-full mt-8">
                         {suggestedActions.slice(0, 4).map((action, i) => (
                             <button 
                                key={i}
                                onClick={() => onSendMessage(action, null)}
                                className="p-3 bg-cyan-950/20 border border-cyan-900/50 rounded-xl text-left hover:bg-cyan-900/30 hover:border-cyan-500 transition-all group"
                             >
                                <div className="flex items-center gap-2 text-cyan-400 mb-1">
                                    <ArrowRightIcon className="w-3 h-3 group-hover:translate-x-1 transition-transform" />
                                    <span className="text-[10px] font-bold uppercase tracking-widest">Execute</span>
                                </div>
                                <span className="text-xs text-cyan-100">{action}</span>
                             </button>
                         ))}
                      </div>
                  </div>
              )}
              
              {filteredMessages.map((msg, idx) => {
                  const isSystem = msg.sender === 'system';
                  const isUser = msg.sender === 'user';
                  
                  return (
                    <div 
                        key={msg.id || idx} 
                        className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
                    >
                        <div 
                            className={`max-w-[90%] p-3 rounded-lg text-sm font-mono leading-relaxed ${
                                isUser 
                                    ? 'bg-cyan-900/40 text-cyan-50 border border-cyan-700/50 rounded-br-none' 
                                    : isSystem
                                        ? 'bg-red-900/20 text-red-200 border border-red-800 rounded-bl-none w-full animate-pulse-border'
                                        : 'bg-slate-900/60 text-slate-200 border border-slate-700/50 rounded-bl-none w-full'
                            }`}
                        >
                            {isSystem && (
                                <div className="flex items-center gap-2 mb-1 border-b border-red-800/50 pb-1">
                                    <AlertTriangleIcon className="w-4 h-4 text-red-400" />
                                    <span className="text-[10px] font-bold text-red-400 uppercase tracking-widest">System Alert</span>
                                </div>
                            )}
                            {msg.text && <div className="whitespace-pre-wrap">{highlightText(msg.text)}</div>}
                            {msg.chartData && renderChart(msg.chartData)}
                            
                            {msg.generatedImage && (
                               <div className="mt-2 rounded-lg overflow-hidden border border-cyan-500/50 shadow-[0_0_15px_rgba(6,182,212,0.3)]">
                                   <img src={msg.generatedImage} alt="Generated Visualization" className="w-full h-auto object-cover" />
                               </div>
                            )}
                            
                            {msg.attachment && (
                                <div className="mt-2 text-xs text-cyan-400 flex items-center bg-black/20 p-2 rounded border border-cyan-900/30">
                                    <FileIcon className="w-3 h-3 mr-2" /> 
                                    <span className="opacity-90">{highlightText(msg.attachment.name)}</span>
                                </div>
                            )}
                        </div>
                    </div>
                  );
              })}
              
              {isLoading && (
                  <div className="flex justify-start">
                      <div className="bg-slate-900/60 p-3 rounded-lg rounded-bl-none border border-slate-700/50 flex items-center space-x-2">
                          <LoaderIcon className="w-4 h-4 text-cyan-400 animate-spin" />
                          <span className="text-xs text-cyan-400 font-mono">
                              {inquiry.status === 'optimizing' ? 'Analyzing Data Streams...' : 'Processing...'}
                          </span>
                      </div>
                  </div>
              )}
              <div ref={messagesEndRef} />
          </div>

          <div className="flex-shrink-0">
              {!isLoading && suggestedActions && suggestedActions.length > 0 && filteredMessages.length > 0 && (
                <div className="flex gap-2 p-2 overflow-x-auto no-scrollbar bg-black/40 border-t border-cyan-900/30">
                  {suggestedActions.map((action, i) => (
                    <button 
                      key={i} 
                      onClick={() => onSendMessage(action, null)}
                      className="whitespace-nowrap px-3 py-1 rounded-full bg-cyan-900/20 border border-cyan-700/50 text-[10px] text-cyan-400 hover:bg-cyan-800/40 hover:text-white transition-all"
                    >
                      {action}
                    </button>
                  ))}
                </div>
              )}

              {attachedFile && (
                <div className="p-2 bg-slate-800/80 border-t border-cyan-900/50 flex items-center justify-between">
                  <div className="flex items-center gap-2 overflow-hidden">
                    <FileIcon className="w-4 h-4 text-cyan-400" />
                    <span className="text-xs text-white truncate">{attachedFile.name}</span>
                  </div>
                  <button onClick={() => setAttachedFile(null)} className="text-gray-400 hover:text-white">
                    <XIcon className="w-4 h-4" />
                  </button>
                </div>
              )}

              <div className="p-3 bg-slate-950 border-t border-cyan-900/50 flex items-center gap-2">
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleFileChange} 
                  className="hidden" 
                />
                <button 
                  onClick={() => fileInputRef.current?.click()}
                  className="p-2 text-cyan-600 hover:text-cyan-400 transition-colors"
                  title="Attach File"
                >
                  <PaperclipIcon className="w-5 h-5" />
                </button>
                
                <input 
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask Agent Q..."
                  className="flex-grow bg-black/50 border border-cyan-900/50 rounded-lg px-3 py-2 text-sm text-white placeholder-cyan-800 focus:outline-none focus:border-cyan-500"
                />

                <button 
                  onClick={handleSend}
                  disabled={isLoading || (!input.trim() && !attachedFile)}
                  className="p-2 bg-cyan-600 rounded-lg text-white disabled:opacity-30 disabled:cursor-not-allowed hover:bg-cyan-500 transition-colors"
                >
                  <SendIcon className="w-5 h-5" />
                </button>
              </div>
          </div>
      </div>
  );

  const memoryView = (
    <div className="flex-1 flex flex-col min-h-0 p-4">
        <MemoryMatrix 
          lastActivity={lastActivity || defaultLastActivity} 
          memorySummary={memorySummary || "Core Cognitive State"} 
          messages={messages}
          interactive={true}
        />
        <div className="mt-4 flex gap-2">
            <button 
                onClick={onClearMemory}
                className="flex-1 py-2 bg-red-900/20 border border-red-800 text-red-400 text-xs font-bold rounded"
            >
                FLUSH RECALL
            </button>
            <button 
                onClick={() => setShowMemory(false)}
                className="flex-1 py-2 bg-slate-800 border border-slate-700 text-white text-xs font-bold rounded"
            >
                RETURN TO CHAT
            </button>
        </div>
    </div>
  );

  if (embedded) {
    return (
        <div className="h-full flex flex-col bg-black/60 backdrop-blur-md overflow-hidden border border-cyan-900/30">
            <div className="p-3 border-b border-cyan-900/50 bg-cyan-950/20 flex justify-between items-center">
                <div className="flex items-center gap-2">
                    <BrainCircuitIcon className="w-4 h-4 text-cyan-400" />
                    <span className="text-[10px] font-bold text-white uppercase tracking-widest">Agent Q Interface</span>
                </div>
                <div className="flex items-center gap-2">
                    {universeConnections.agentQ && (
                         <div className="flex items-center gap-1 text-[8px] bg-purple-900/50 text-purple-200 px-2 py-0.5 rounded border border-purple-500 animate-pulse">
                            <GalaxyIcon className="w-2.5 h-2.5" />
                            UNIVERSE UPLINK
                         </div>
                    )}
                    <button onClick={() => setShowMemory(!showMemory)} className={`p-1 rounded transition-colors ${showMemory ? 'text-cyan-400 bg-cyan-900/30' : 'text-gray-500 hover:text-cyan-400'}`}>
                        <CpuChipIcon className="w-4 h-4" />
                    </button>
                </div>
            </div>
            {showMemory ? memoryView : chatHistoryView}
        </div>
    );
  }

  return (
    <>
      {isOpen && (
        <div className={`fixed z-[100] transition-all duration-500 ${fullScreen ? 'inset-4' : 'bottom-20 right-6 w-[450px] h-[650px] max-h-[80vh]'}`}>
          <div className="h-full bg-slate-950/90 backdrop-blur-2xl border border-cyan-500/30 rounded-2xl shadow-[0_0_50px_rgba(0,0,0,0.5)] flex flex-col overflow-hidden animate-fade-in-up">
            
            {/* Header */}
            <div className="flex-shrink-0 p-4 border-b border-cyan-500/50 bg-black/80 flex items-center justify-between relative overflow-hidden">
              {/* Q-Native Prime Background Effect - Closed Loop Singularity */}
              <div className="absolute inset-0 bg-[conic-gradient(from_0deg,transparent_0deg,rgba(6,182,212,0.1)_180deg,transparent_360deg)] animate-spin-slow pointer-events-none opacity-30" />
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(139,92,246,0.1),transparent_60%)] animate-pulse-slow pointer-events-none" />
              <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-cyan-500 to-transparent opacity-50" />
              
              <div className="flex items-center gap-4 relative z-10">
                <div className="relative group">
                  <div className={`absolute inset-0 bg-cyan-400/30 blur-xl rounded-full transition-all duration-500 ${isSpeaking ? 'scale-150 opacity-60' : 'scale-100 opacity-30 group-hover:opacity-50'}`}></div>
                  <div className="relative w-10 h-10 flex items-center justify-center bg-black/50 rounded-full border border-cyan-500/50 shadow-[0_0_15px_rgba(6,182,212,0.4)]">
                    <BrainCircuitIcon className={`w-6 h-6 text-cyan-400 ${isSpeaking ? 'animate-pulse' : ''}`} />
                  </div>
                  {/* Quantum Superposition Indicator */}
                  <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-purple-500 rounded-full border border-black animate-ping" />
                </div>
                
                <div>
                  <h3 className="text-sm font-black text-white uppercase tracking-[0.2em] flex items-center gap-2">
                    Agent Q
                    <span className="text-[8px] px-1.5 py-0.5 rounded bg-cyan-900/50 border border-cyan-500/30 text-cyan-300 font-mono tracking-normal shadow-[0_0_10px_rgba(6,182,212,0.2)]">PRIME</span>
                  </h3>
                  <div className="flex flex-col gap-0.5 mt-1">
                    <div className="flex items-center gap-1.5">
                      <div className={`w-1 h-1 rounded-full ${isSpeaking ? 'bg-cyan-400 animate-ping' : 'bg-green-500 animate-pulse'}`}></div>
                      <span className="text-[9px] text-cyan-500 font-mono tracking-wider">Q-NATIVE COGNITION</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <GalaxyIcon className="w-2.5 h-2.5 text-purple-400 animate-spin-slow" />
                      <span className="text-[8px] text-purple-400 font-mono tracking-wider">GRAND UNIVERSE LINKED</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2 relative z-10">
                <button 
                  onClick={onToggleTts}
                  className={`p-2 rounded-lg border transition-all ${isTtsEnabled ? 'bg-cyan-500/10 border-cyan-500/50 text-cyan-300' : 'bg-slate-900 border-slate-800 text-gray-600'}`}
                  title={isTtsEnabled ? "Mute Agent" : "Enable Voice"}
                >
                  {isTtsEnabled ? <MicIcon className="w-4 h-4" /> : <MicOffIcon className="w-4 h-4" />}
                </button>
                <button 
                  onClick={() => setShowMemory(!showMemory)}
                  className={`p-2 rounded-lg border transition-all ${showMemory ? 'bg-cyan-500/10 border-cyan-500/50 text-cyan-300' : 'bg-slate-900 border-slate-800 text-gray-600'}`}
                  title="View Cognitive Memory"
                >
                  <CpuChipIcon className="w-4 h-4" />
                </button>
                <button 
                  onClick={onToggleOpen}
                  className="p-2 text-gray-500 hover:text-white transition-colors"
                >
                  <XIcon className="w-6 h-6" />
                </button>
              </div>
            </div>

            {/* Cognitive Architecture Visualization Layer */}
            <div className="flex justify-between px-4 py-1.5 bg-black/60 border-b border-cyan-900/30 backdrop-blur-sm relative z-10">
                <div className="flex items-center gap-1.5 group cursor-help" title="Intuitive Logic Layer: Predictive Superposition Active">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse shadow-[0_0_5px_rgba(59,130,246,0.5)]"></div>
                    <span className="text-[8px] text-blue-400 font-mono tracking-wider group-hover:text-blue-300 transition-colors">ILL: PRE-COG</span>
                </div>
                <div className="flex items-center gap-1.5 group cursor-help" title="Integrated Processing Sphere: Autonomous Neuro-Plasticity Active">
                    <div className="w-1.5 h-1.5 rounded-full bg-purple-500 animate-pulse delay-75 shadow-[0_0_5px_rgba(168,85,247,0.5)]"></div>
                    <span className="text-[8px] text-purple-400 font-mono tracking-wider group-hover:text-purple-300 transition-colors">IPS: PLASTICITY</span>
                </div>
                <div className="flex items-center gap-1.5 group cursor-help" title="Contextual Linguistic Layer: Quantum Empathy Active">
                    <div className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse delay-150 shadow-[0_0_5px_rgba(244,63,94,0.5)]"></div>
                    <span className="text-[8px] text-rose-400 font-mono tracking-wider group-hover:text-rose-300 transition-colors">CLL: EMPATHY</span>
                </div>
            </div>

            {/* Content */}
            <div className="flex-grow flex flex-col min-h-0">
               {showMemory ? memoryView : chatHistoryView}
            </div>

            {/* Context Footer */}
            {!showMemory && displayContext && (
                <div className="px-4 py-1.5 bg-black/60 border-t border-cyan-900/30 flex justify-between items-center">
                    <span className="text-[9px] text-cyan-800 font-mono uppercase">Context: {displayContext}</span>
                    <span className="text-[9px] text-gray-600 font-mono uppercase">QCOS_V4.5</span>
                </div>
            )}
          </div>
        </div>
      )}

      {/* Floating Trigger Button (when not embedded) */}
      {!embedded && !isOpen && (
        <div className={triggerClassName}>
            <button 
                onClick={onToggleOpen}
                className="relative w-14 h-14 bg-cyan-950/80 border-2 border-cyan-400/50 rounded-full flex items-center justify-center shadow-[0_0_20px_rgba(6,182,212,0.3)] group hover:scale-110 transition-all duration-300"
            >
                <div className={`absolute inset-0 rounded-full bg-cyan-400 ${isSpeaking ? 'animate-ping opacity-40' : 'animate-ping opacity-20 group-hover:opacity-40'}`}></div>
                <BrainCircuitIcon className={`w-7 h-7 text-cyan-300 relative z-10 group-hover:text-white transition-colors ${isSpeaking ? 'animate-pulse' : ''}`} />
                
                {/* Notification indicator if active */}
                {isDataLinked && (
                    <div className="absolute -top-1 -right-1 w-4 h-4 bg-purple-500 rounded-full border border-black flex items-center justify-center">
                        <ActivityIcon className="w-2.5 h-2.5 text-white animate-pulse" />
                    </div>
                )}
            </button>
        </div>
      )}
    </>
  );
};

export default AgentQ;