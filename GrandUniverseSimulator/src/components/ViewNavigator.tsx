
import React from 'react';
import { ChevronUpIcon, ChevronDownIcon, ChevronLeftIcon, ChevronRightIcon, CubeIcon } from './Icons';

interface ViewNavigatorProps {
  onNavigate: (direction: 'up' | 'down' | 'left' | 'right') => void;
  currentFace: number;
}

const ViewNavigator: React.FC<ViewNavigatorProps> = ({ onNavigate, currentFace }) => {
  const faceNames = ['CORE', 'SYSTEM', 'DEV', 'HARDWARE', 'CONFIG', 'VITALS'];

  return (
    <div className="fixed bottom-8 right-8 z-50 flex flex-col items-center gap-2 pointer-events-auto animate-fade-in">
      <button 
        onClick={() => onNavigate('up')}
        className="p-2 rounded-full bg-cyan-900/40 border border-cyan-500/30 hover:bg-cyan-500/20 text-cyan-300 transition-colors"
        title="Rotate Up"
      >
        <ChevronUpIcon className="w-6 h-6" />
      </button>
      
      <div className="flex items-center gap-2">
        <button 
            onClick={() => onNavigate('left')}
            className="p-2 rounded-full bg-cyan-900/40 border border-cyan-500/30 hover:bg-cyan-500/20 text-cyan-300 transition-colors"
            title="Rotate Left"
        >
            <ChevronLeftIcon className="w-6 h-6" />
        </button>
        
        <div className="w-16 h-16 flex flex-col items-center justify-center bg-black/60 backdrop-blur-md rounded-lg border border-cyan-500/50 shadow-[0_0_15px_theme(colors.cyan.900)]">
            <CubeIcon className="w-6 h-6 text-cyan-400 mb-1" />
            <span className="text-[10px] font-bold text-white tracking-wider">{faceNames[currentFace]}</span>
        </div>

        <button 
            onClick={() => onNavigate('right')}
            className="p-2 rounded-full bg-cyan-900/40 border border-cyan-500/30 hover:bg-cyan-500/20 text-cyan-300 transition-colors"
            title="Rotate Right"
        >
            <ChevronRightIcon className="w-6 h-6" />
        </button>
      </div>

      <button 
        onClick={() => onNavigate('down')}
        className="p-2 rounded-full bg-cyan-900/40 border border-cyan-500/30 hover:bg-cyan-500/20 text-cyan-300 transition-colors"
        title="Rotate Down"
      >
        <ChevronDownIcon className="w-6 h-6" />
      </button>
    </div>
  );
};

export default ViewNavigator;
