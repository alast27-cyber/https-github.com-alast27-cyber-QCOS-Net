
import React from 'react';
import { MaximizeIcon, MinimizeIcon } from './Icons';

interface GlassPanelProps {
  title: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
  onMaximize?: () => void;
  isMaximized?: boolean;
}

const GlassPanel: React.FC<GlassPanelProps> = ({ title, children, className = '', style, onMaximize, isMaximized = false }) => {
  return (
    <div 
      className={`
        bg-black/60 border border-cyan-500/20 rounded-xl
        flex flex-col h-full transition-all duration-500
        group hover:border-cyan-400/50 hover:shadow-[0_0_40px_rgba(0,255,255,0.15)]
        ${className}
      `}
      style={style}
    >
      <header className="relative z-10 px-4 py-2 flex items-center justify-between border-b border-cyan-500/10 bg-cyan-950/20 backdrop-blur-xl rounded-t-xl overflow-hidden">
        <h2 className="text-[11px] sm:text-xs font-black text-cyan-200 tracking-[0.2em] uppercase flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-pulse"></span>
          <span>{title}</span>
        </h2>
        
        {onMaximize && (
            <button 
                onClick={(e) => {
                    e.stopPropagation();
                    onMaximize();
                }}
                className="flex items-center justify-center p-1.5 rounded-lg hover:bg-cyan-500/20 transition-all text-cyan-500"
            >
                {isMaximized ? (
                    <MinimizeIcon className="w-3.5 h-3.5" />
                ) : (
                    <MaximizeIcon className="w-3.5 h-3.5" />
                )}
            </button>
        )}
      </header>
      <div className="relative z-10 p-3 flex-grow overflow-hidden flex flex-col min-h-0 bg-gradient-to-b from-transparent to-black/20">
        {children}
      </div>
    </div>
  );
};

export default GlassPanel;
