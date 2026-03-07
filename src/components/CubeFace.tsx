
import React from 'react';
import GlassPanel from './GlassPanel';
import { FaceData } from '../utils/dashboardConfig';
import { MaximizeIcon } from './Icons';

interface CubeFaceProps {
  index: number;
  active: boolean;
  transform: string;
  content: FaceData;
  onMaximize?: (panelId: string) => void;
}

const CubeFace: React.FC<CubeFaceProps> = ({ index, active, transform, content, onMaximize }) => {
  return (
    <div 
      className={`absolute inset-0 backface-hidden transition-all duration-1000 ${active ? 'opacity-100 pointer-events-auto' : 'opacity-10 pointer-events-none'}`}
      style={{ 
        transform: transform,
        // Ensure faces don't intercept clicks when inactive/background
        zIndex: active ? 10 : 0 
      }}
    >
      <div className={`w-full h-full grid gap-4 p-4 ${content.layout}`}>
        {content.panels.map(panel => (
          <div key={panel.id} className={`relative group ${panel.className || ''}`}>
             {/* Hover Maximize Button */}
             {onMaximize && (
                <button 
                    onClick={() => onMaximize(panel.id)}
                    className="absolute -top-2 -right-2 z-50 p-1.5 bg-black/80 text-cyan-400 border border-cyan-600 rounded-full opacity-0 group-hover:opacity-100 transition-opacity hover:scale-110 cursor-pointer shadow-lg"
                    title="Maximize Panel"
                >
                    <MaximizeIcon className="w-4 h-4" />
                </button>
             )}
             
             {/* Render Content */}
             {panel.content ? (
                 <GlassPanel title={panel.title}>
                     {panel.content}
                 </GlassPanel>
             ) : (
                 <GlassPanel title={panel.title}>
                    <div className="flex flex-col h-full justify-center items-center text-center p-4">
                        <p className="text-sm text-cyan-200">{panel.description}</p>
                        {panel.minAdminLevel > 1 && (
                            <span className="mt-2 text-[10px] text-red-400 border border-red-900/50 px-2 py-0.5 rounded bg-red-900/10 uppercase tracking-widest">
                                Admin Level {panel.minAdminLevel}+
                            </span>
                        )}
                    </div>
                 </GlassPanel>
             )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default CubeFace;
