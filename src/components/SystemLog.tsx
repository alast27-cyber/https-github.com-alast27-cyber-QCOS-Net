import React, { useRef, useEffect } from 'react';
import { LogEntry } from '../types';

interface SystemLogProps {
  logs: LogEntry[];
}

const levelStyles = {
    INFO: 'text-blue-400',
    WARN: 'text-yellow-400',
    ERROR: 'text-red-400',
    CMD: 'text-purple-400',
    SUCCESS: 'text-green-400',
};

const LiveIndicator = () => (
    <div className="flex items-center space-x-2 ml-auto">
        <div className="relative flex h-2 w-2">
            <div className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></div>
            <div className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></div>
        </div>
        <span className="text-red-400 text-xs font-bold tracking-widest">LIVE</span>
    </div>
);
LiveIndicator.displayName = 'LiveIndicator';


const SystemLog: React.FC<SystemLogProps> = ({ logs }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [logs]);

  return (
    <div className="h-full flex flex-col">
       <div className="flex items-center justify-between w-full mb-1">
            <label className="block text-cyan-400 text-sm font-semibold">
                System Log
            </label>
            <LiveIndicator />
        </div>
      <div ref={scrollRef} className="overflow-y-auto flex-grow text-xs space-y-1 pr-2 h-full flex flex-col-reverse bg-black/30 border border-blue-500/50 rounded-md p-2 holographic-grid">
        <div>
          {logs.map((log, index) => (
            <div key={log.id} className="flex animate-fade-in font-mono">
              <span className={`text-cyan-500 w-16 flex-shrink-0 ${index === 0 ? 'animate-pulse-bright' : ''}`}>{log.time}</span>
              <span className={`w-14 flex-shrink-0 font-bold ${levelStyles[log.level]}`}>{`[${log.level}]`}</span>
              <span className="text-white">{log.msg}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
export default SystemLog;