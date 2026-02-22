
import React, { useState, useRef, useEffect } from 'react';
import { UsersIcon, XIcon, SendIcon, LoaderIcon, PaperclipIcon, FileIcon } from './Icons';
import GlassPanel from './GlassPanel';
import { Message } from '../utils/agentUtils';

interface AdminChatProps {
  isOpen: boolean;
  onToggle: () => void;
  mode: 'console' | 'chat';
  setMode: (mode: 'console' | 'chat') => void;
  consoleMessages: Message[];
  chatMessages: Message[];
  isLoading: boolean;
  onSendMessage: (input: string, file: File | null) => void;
  triggerClassName?: string;
}

const AdminChat: React.FC<AdminChatProps> = ({ 
    isOpen, 
    onToggle, 
    mode,
    setMode,
    consoleMessages,
    chatMessages,
    isLoading, 
    onSendMessage,
    triggerClassName = "fixed bottom-4 right-36 z-50 pointer-events-auto group"
}) => {
  const [input, setInput] = useState('');
  const [attachedFile, setAttachedFile] = useState<File | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [consoleMessages, chatMessages, mode]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files[0]) {
          setAttachedFile(e.target.files[0]);
      }
  };

  const handleSend = () => {
    onSendMessage(input, attachedFile);
    setInput('');
    setAttachedFile(null);
    if(fileInputRef.current) fileInputRef.current.value = "";
  };
  
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') handleSend();
  };
  
  const getTabClasses = (tabName: 'console' | 'chat') => {
    const base = "flex-1 p-2 text-sm font-bold text-center transition-colors cursor-pointer";
    const active = "bg-amber-500/20 text-amber-200 border-b-2 border-amber-400";
    const inactive = "text-amber-500 border-b-2 border-transparent hover:bg-amber-500/10";
    return `${base} ${mode === tabName ? active : inactive}`;
  };
  
  const messages = mode === 'console' ? consoleMessages : chatMessages;

  return (
    <>
      {isOpen && (
        <div className="fixed z-50 animate-fade-in-up pointer-events-auto bottom-4 right-4 w-[380px] h-[550px]">
          <GlassPanel title="Admin Comms">
            <div className="flex flex-col h-full">
              <div className="flex-shrink-0 flex border-b border-amber-500/30">
                  <button onClick={() => setMode('console')} className={getTabClasses('console')}>Console</button>
                  <button onClick={() => setMode('chat')} className={getTabClasses('chat')}>Chat</button>
              </div>
              <div className="flex-grow overflow-y-auto pr-2 space-y-3 p-2">
                {messages.map((msg, index) => {
                    const isUser = msg.sender === 'user';
                    const bubbleClasses = `max-w-[80%] p-2 rounded-lg text-sm ${isUser ? 'bg-amber-700/50 text-white' : (mode === 'console' ? 'bg-slate-700/50 text-amber-200' : 'bg-blue-700/50 text-blue-200')}`;
                    return (
                      <div key={msg.id || index} className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
                        <div className={bubbleClasses}>
                           {msg.text && <pre className="font-mono whitespace-pre-wrap">{msg.text}</pre>}
                           {msg.attachment && (
                              <div className="mt-2 p-2 rounded-md flex items-center gap-2 bg-black/20"><FileIcon className="w-5 h-5 text-cyan-300" /><span className="text-xs truncate">{msg.attachment.name}</span></div>
                          )}
                        </div>
                      </div>
                    )
                })}
                {isLoading && <div className="flex justify-start"><div className="p-2 rounded-lg bg-slate-700/50"><LoaderIcon className="w-5 h-5 animate-spin" /></div></div>}
                <div ref={messagesEndRef} />
              </div>
              {attachedFile && mode === 'chat' && <div className="mx-2 p-2 rounded-md animate-fade-in text-sm flex items-center justify-between bg-slate-800/60"><div className="flex items-center gap-2 overflow-hidden"><FileIcon className="w-4 h-4 text-amber-400" /><span className="text-white truncate">{attachedFile.name}</span></div><button onClick={() => { setAttachedFile(null); if(fileInputRef.current) fileInputRef.current.value = ""; }} className="p-1 rounded-full hover:bg-white/10"><XIcon className="w-4 h-4 text-amber-400" /></button></div>}
              <div className="flex-shrink-0 pt-2 border-t border-amber-500/20 flex items-center gap-2 p-2">
                 {mode === 'chat' && <><input ref={fileInputRef} type="file" onChange={handleFileChange} className="hidden"/><button onClick={() => fileInputRef.current?.click()} disabled={isLoading} className="w-10 h-10 flex-shrink-0 flex items-center justify-center rounded-md bg-amber-500/30 hover:bg-amber-500/50 border border-amber-500/50 text-amber-200 transition-colors disabled:opacity-50"><PaperclipIcon className="w-5 h-5"/></button></>}
                <input type="text" value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={handleKeyDown} placeholder={mode === 'console' ? "Enter command..." : "Chat..."} disabled={isLoading} className="flex-grow bg-black/30 border border-amber-500/50 rounded-md p-2 text-white placeholder:text-gray-500 focus:ring-1 focus:ring-amber-400 focus:outline-none"/>
                <button onClick={handleSend} disabled={isLoading || (!input.trim() && !attachedFile)} className="w-10 h-10 flex-shrink-0 flex items-center justify-center rounded-md bg-amber-500/30 hover:bg-amber-500/50 border border-amber-500/50 text-amber-200 transition-colors disabled:opacity-50"><SendIcon className="w-5 h-5"/></button>
              </div>
            </div>
          </GlassPanel>
          <button onClick={onToggle} className="absolute top-3 right-3 z-50 text-amber-400/70 hover:text-white p-2 rounded-full hover:bg-white/10"><XIcon className="w-5 h-5" /></button>
        </div>
      )}
      <div className={triggerClassName}>
        <button onClick={onToggle} className="relative w-12 h-12 bg-amber-900/50 border-2 border-amber-400/50 rounded-full flex items-center justify-center shadow-[0_0_15px_theme(colors.amber.400)] holographic-button" style={{filter: "drop-shadow(0 0 8px theme('colors.amber.400'))"}}>
            <UsersIcon className="w-6 h-6 text-amber-300 animate-pulse" />
            <div className="absolute inset-y-2 left-1/2 -translate-x-1/2 w-0.5 bg-red-500 rounded-full animate-pulse-bg shadow-[0_0_3px_#f00]"></div>
        </button>
      </div>
    </>
  );
};

export default AdminChat;
