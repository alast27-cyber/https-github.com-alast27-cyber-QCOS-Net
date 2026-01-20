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
        holographic-projection rounded-xl
        flex flex-col h-full transition-all duration-500
        group hover:border-cyan-400/50 hover:shadow-[0_0_40px_rgba(0,255,255,0.15)]
        ${className}
      `}
      style={style}
    >
      {/* 12-D Refractive Edges */}
      <div className="refractive-edge rounded-xl"></div>
      
      {/* Scanline Texture Layer */}
      <div className="scanlines-v2 rounded-xl"></div>

      {/* Holographic HUD Corners */}
      <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-cyan-500/40 rounded-tl-xl pointer-events-none group-hover:border-cyan-400"></div>
      <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-cyan-500/40 rounded-tr-xl pointer-events-none group-hover:border-cyan-400"></div>
      <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-cyan-500/40 rounded-bl-xl pointer-events-none group-hover:border-cyan-400"></div>
      <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-cyan-500/40 rounded-br-xl pointer-events-none group-hover:border-cyan-400"></div>

      {/* Header - Advanced Holographic Ribbon */}
      <header className="relative z-10 px-4 py-2 flex items-center justify-between border-b border-cyan-500/10 bg-cyan-950/20 backdrop-blur-xl rounded-t-xl overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/5 to-transparent pointer-events-none"></div>
        <h2 className="text-[11px] sm:text-xs font-black text-cyan-200 tracking-[0.2em] uppercase flex items-center gap-2 drop-shadow-[0_0_5px_rgba(0,255,255,0.3)]">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-pulse"></span>
          <span className="chromatic-text">{title}</span>
        </h2>
        
        {onMaximize && (
            <button 
                onClick={(e) => {
                    e.stopPropagation();
                    onMaximize();
                }}
                className="flex items-center justify-center p-1.5 rounded-lg hover:bg-cyan-500/20 transition-all group/btn text-cyan-500"
                title={isMaximized ? "Minimize Layer" : "Project Full Dimension"}
            >
                {isMaximized ? (
                    <MinimizeIcon className="w-3.5 h-3.5" />
                ) : (
                    <MaximizeIcon className="w-3.5 h-3.5 group-hover/btn:scale-110 transition-transform" />
                )}
            </button>
        )}
      </header>

      {/* Content - 12-D Depth Container */}
      <div className="relative z-10 p-3 flex-grow overflow-hidden custom-scrollbar flex flex-col min-h-0 bg-gradient-to-b from-transparent to-black/20">
        {children}
      </div>

      {/* Subtle Bottom Ambient Glow */}
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-1/2 h-1 bg-cyan-500/20 blur-md pointer-events-none"></div>
    </div>
  );
};

export default GlassPanel;