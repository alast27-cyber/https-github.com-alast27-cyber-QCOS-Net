
import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Message, Role } from '../types';
import { User, Bot, Sparkles, AlertCircle, ExternalLink } from 'lucide-react';

interface MessageBubbleProps {
  message: Message;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.role === Role.USER;
  const isError = message.isError;

  return (
    <div className={`group flex gap-4 w-full max-w-4xl mx-auto p-4 md:p-6 ${isUser ? 'bg-transparent' : 'bg-slate-800/30 rounded-2xl'}`}>
      <div className={`flex-shrink-0 mt-1 h-8 w-8 rounded-full flex items-center justify-center ${
        isUser ? 'bg-slate-700 text-slate-300' : isError ? 'bg-red-900/50 text-red-400' : 'bg-blue-600/20 text-blue-400'
      }`}>
        {isUser ? <User size={16} /> : isError ? <AlertCircle size={16} /> : <Bot size={16} />}
      </div>
      
      <div className="flex-1 min-w-0 overflow-hidden">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-sm font-semibold text-slate-200">
            {isUser ? 'You' : 'Nebula'}
          </span>
          <span className="text-xs text-slate-500">
            {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </span>
        </div>

        {/* Attachments (User Images) */}
        {message.attachments && message.attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {message.attachments.map((att, i) => (
              <img 
                key={i}
                src={att.previewUrl} 
                alt="attachment" 
                className="max-h-64 rounded-lg border border-slate-700 object-cover"
              />
            ))}
          </div>
        )}

        {/* Text Content */}
        <div className={`prose prose-invert prose-slate max-w-none text-slate-300 leading-7 ${isError ? 'text-red-300' : ''}`}>
           <ReactMarkdown
             components={{
               img: ({node, ...props}) => (
                 <img {...props} className="rounded-lg max-h-96 border border-slate-700" alt={props.alt || 'generated content'} />
               ),
               a: ({node, ...props}) => (
                <a {...props} className="text-blue-400 hover:text-blue-300 hover:underline" target="_blank" rel="noopener noreferrer" />
               ),
               code: ({ node, className, children, ...props }) => {
                 const match = /language-(\w+)/.exec(className || '')
                 return match ? (
                   <div className="relative group">
                     <pre className="bg-slate-950 p-4 rounded-lg overflow-x-auto border border-slate-800 my-4">
                       <code className={className} {...props}>
                         {children}
                       </code>
                     </pre>
                   </div>
                 ) : (
                   <code className="bg-slate-800 px-1.5 py-0.5 rounded text-sm text-slate-200" {...props}>
                     {children}
                   </code>
                 )
               }
             }}
           >
             {message.text}
           </ReactMarkdown>
        </div>
        
        {/* Grounding Sources */}
        {message.groundingMetadata?.groundingChunks && message.groundingMetadata.groundingChunks.length > 0 && (
           <div className="mt-4 pt-3 border-t border-slate-700/50">
             <p className="text-xs text-slate-500 font-medium mb-2 uppercase tracking-wider flex items-center gap-1">
               <Sparkles size={10} /> Sources
             </p>
             <div className="flex flex-wrap gap-2">
                {message.groundingMetadata.groundingChunks.map((chunk, idx) => {
                  if (!chunk.web?.uri) return null;
                  return (
                    <a 
                      key={idx}
                      href={chunk.web.uri}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-blue-300 px-2 py-1.5 rounded-md transition-colors border border-slate-700"
                    >
                      <ExternalLink size={10} />
                      <span className="truncate max-w-[150px]">{chunk.web.title || new URL(chunk.web.uri).hostname}</span>
                    </a>
                  );
                })}
             </div>
           </div>
        )}
      </div>
    </div>
  );
};
