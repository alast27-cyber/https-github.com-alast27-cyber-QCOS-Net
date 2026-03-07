
import React from 'react';

const LiveIndicator: React.FC = () => (
    <div className="flex items-center space-x-2 ml-auto">
      <div className="relative flex h-2 w-2">
        <div className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></div>
        <div className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></div>
      </div>
      <span className="text-red-400 text-xs font-bold tracking-widest">LIVE</span>
    </div>
);

export default LiveIndicator;
