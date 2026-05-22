import React from 'react';
import { MicIcon, MicOffIcon, LogOutIcon } from './Icons';
import { ListeningState } from '../hooks/useVoiceCommands';
import { useAuth } from '../context/AuthContext';

interface ResourceStewardProps {
  listeningState: ListeningState;
  onToggleListen: () => void;
  isVoiceSupported: boolean;
}

const ResourceSteward: React.FC<ResourceStewardProps> = ({ listeningState, onToggleListen, isVoiceSupported }) => {
  const { isAuthenticated, logout, currentUser, adminLevel } = useAuth();

  const handleClick = () => {
    if (isVoiceSupported) {
      onToggleListen();
    }
  };
  
  const getStatus = () => {
      if (!isVoiceSupported) return 'unsupported';
      if (listeningState === 'permission_denied') return 'permission_denied';
      if (listeningState === 'error') return 'error';
      if (listeningState === 'listening') return 'listening';
      return 'idle';
  }
  
  const status = getStatus();

  const statusConfig = {
    idle: {
      buttonClasses: 'bg-cyan-500/20 border-cyan-500/50 holographic-button',
      icon: <MicIcon className="w-5 h-5 text-cyan-300"/>,
      statusText: 'text-green-400',
      statusLabel: 'Online',
      ping: <div className="absolute inset-0 rounded-full bg-cyan-400 animate-ping opacity-50"></div>,
    },
    listening: {
      buttonClasses: 'bg-cyan-400/40 border-cyan-400 scale-105 holographic-button',
      icon: <MicIcon className="w-5 h-5 text-white"/>,
      statusText: 'text-cyan-300',
      statusLabel: 'Listening...',
      ping: <div className="absolute inset-0 rounded-full bg-cyan-300 animate-ping opacity-90"></div>,
    },
    unsupported: {
      buttonClasses: 'bg-red-500/20 border-red-500/50 cursor-not-allowed',
      icon: <MicIcon className="w-5 h-5 text-red-400/70"/>,
      statusText: 'text-red-400',
      statusLabel: 'Voice N/A',
      ping: null,
    },
    error: {
      buttonClasses: 'bg-red-500/20 border-red-500/50',
      icon: <MicOffIcon className="w-5 h-5 text-red-300"/>,
      statusText: 'text-red-400',
      statusLabel: 'No Speech',
      ping: null,
    },
    permission_denied: {
      buttonClasses: 'bg-yellow-500/20 border-yellow-500/50 cursor-not-allowed',
      icon: <MicOffIcon className="w-5 h-5 text-yellow-300"/>,
      statusText: 'text-yellow-400',
      statusLabel: 'Mic Blocked',
      ping: null,
    },
  };

  const currentConfig = statusConfig[status as keyof typeof statusConfig];

  return (
    <div className="flex items-center space-x-3 relative">
      {listeningState === 'permission_denied' && (
        <div className="absolute right-full top-1/2 -translate-y-1/2 mr-4 w-max max-w-xs bg-slate-800 border border-yellow-500/50 p-3 rounded-md shadow-lg animate-fade-in-right">
          <p className="text-sm text-yellow-200 font-semibold">Permission Denied</p>
          <p className="text-xs text-yellow-300 mt-1">To use voice commands, grant microphone access. Check the mic/camera icon in your browser's address bar.</p>
        </div>
      )}
      <div className="text-right hidden sm:block">
        <p className="text-sm text-white font-semibold">{isAuthenticated ? currentUser : 'Resource Steward'}</p>
        <p className={`text-xs ${currentConfig.statusText} transition-colors`}>{isAuthenticated ? `Admin Level: ${adminLevel}` : currentConfig.statusLabel}</p>
      </div>
       <button 
        onClick={handleClick}
        disabled={status === 'unsupported' || status === 'permission_denied'}
        className={`relative w-10 h-10 flex items-center justify-center rounded-full transition-all duration-300 ${currentConfig.buttonClasses}`}
        aria-label={`Resource Steward: ${currentConfig.statusLabel}`}
        title={`Resource Steward: ${currentConfig.statusLabel}`}
      >
        {currentConfig.icon}
        {currentConfig.ping}
      </button>
      {isAuthenticated && (
        <button 
            onClick={logout}
            className="relative w-10 h-10 flex items-center justify-center rounded-full bg-red-500/20 border-red-500/50 holographic-button"
            aria-label="Log Out"
            title="Log Out"
        >
            <LogOutIcon className="w-5 h-5 text-red-300"/>
        </button>
      )}
    </div>
  );
};

export default ResourceSteward;