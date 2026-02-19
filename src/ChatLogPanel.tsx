import React, { useState, useRef, useEffect } from 'react';
import { MessageSquareIcon, SendIcon, LoaderIcon, DocumentArrowUpIcon } from './Icons';
import { Message } from '../utils/agentUtils';

interface ChatLogPanelProps {
    messages: Message[];
    isLoading: boolean;
    onSendMessage: (input: string) => void;
}

const ChatLogPanel: React.FC<ChatLogPanelProps> = ({ messages, isLoading, onSendMessage }) => {
    const [input, setInput] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSend = () => {
        if (input.trim() && !isLoading) {
            onSendMessage(input.trim());
            setInput('');
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };
    
    const handleExport = () => {
        const timestamp = new Date().toLocaleString();
        let fileContent = `QCOS Agent Q Chat Log\nExported on: ${timestamp}\n\n`;

        messages.forEach(msg => {
            const sender = msg.sender === 'user' ? 'User' : 'Agent Q';
            fileContent += `[${sender}]\n${msg.text}\n\n`;
        });
        
        const blob = new Blob([fileContent], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'qcos-chat-history.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };


    const renderMessageContent = (text: string) => {
        // A simplified version without code highlighting for this panel
        const truncatedText = text.length > 300 ? text.substring(0, 300) + '...' : text;
        return <p>{truncatedText}</p>;
    };

    return (
        <div className="h-full flex flex-col">
             <div className="flex justify-between items-center mb-2 flex-shrink-0">
                <h3 className="text-cyan-400 text-sm font-semibold flex items-center">
                    <MessageSquareIcon className="w-4 h-4 mr-2" />
                    Chat Log
                </h3>
                <button 
                    onClick={handleExport}
                    disabled={messages.length === 0}
                    className="p-1 rounded-md text-cyan-400/70 hover:text-white hover:bg-white/10 disabled:opacity-50 disabled:cursor-not-allowed"
                    aria-label="Export chat history"
                    title="Export Chat History"
                >
                    <DocumentArrowUpIcon className="w-5 h-5" />
                </button>
            </div>
            <div className="flex-grow overflow-y-auto pr-2 space-y-3 mb-2 bg-black/20 p-2 rounded-md border border-cyan-900">
                {messages.length === 0 && (
                    <div className="h-full flex flex-col items-center justify-center p-4 text-center text-cyan-600">
                        <MessageSquareIcon className="w-8 h-8 mb-2" />
                        <p className="text-sm">Chat history with Agent Q will appear here.</p>
                    </div>
                )}
                {messages.map((msg, index) => (
                    <div key={index} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-[90%] p-2 rounded-lg text-xs ${msg.sender === 'user' ? 'bg-cyan-800/60 text-white' : 'bg-slate-700/60 text-cyan-200'}`}>
                            {renderMessageContent(msg.text)}
                        </div>
                    </div>
                ))}
                {isLoading && (
                     <div className="flex justify-start">
                        <div className="bg-slate-700/60 text-cyan-200 p-2 rounded-lg">
                            <LoaderIcon className="w-4 h-4 animate-spin" />
                        </div>
                     </div>
                  )}
                <div ref={messagesEndRef} />
            </div>
            <div className="flex-shrink-0 flex items-center gap-2">
                <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    rows={1}
                    placeholder="Chat with Agent Q..."
                    disabled={isLoading}
                    className="flex-grow bg-black/30 border border-blue-500/50 rounded-md p-2 text-white placeholder:text-gray-500 focus:ring-1 focus:ring-cyan-400 focus:outline-none text-sm resize-none"
                />
                <button 
                    onClick={handleSend}
                    disabled={isLoading || !input.trim()}
                    className="w-10 h-10 flex-shrink-0 flex items-center justify-center rounded-md bg-cyan-500/30 hover:bg-cyan-500/50 border border-cyan-500/50 text-cyan-200 transition-colors disabled:opacity-50"
                    aria-label="Send Message"
                >
                    <SendIcon className="w-5 h-5"/>
                </button>
            </div>
        </div>
    );
};

export default ChatLogPanel;