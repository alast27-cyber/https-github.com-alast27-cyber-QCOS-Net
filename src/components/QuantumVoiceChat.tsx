
import React, { useState, useEffect } from 'react';
import GlassPanel from './GlassPanel';
import { MicIcon, MicOffIcon, KeyIcon, ShieldCheckIcon, ActivityIcon } from './Icons';

// --- QKD Simulation Service ---
const establishSecureKey = async () => {
    return new Promise<string | null>(resolve => {
        setTimeout(() => {
            // Simulate BB84 Key Exchange
            const key = Array.from({ length: 64 }, () => Math.random() > 0.5 ? '1' : '0').join('');
            resolve(key);
        }, 1500);
    });
};

// --- Sub-components ---

const StatusBar: React.FC<{ status: string; keyFragment: string }> = ({ status, keyFragment }) => (
    <div className="flex items-center justify-between bg-black/40 border border-cyan-900/50 rounded-lg p-3 mb-4">
        <div className="flex items-center gap-3">
            <div className={`w-3 h-3 rounded-full ${status === 'Connected' ? 'bg-green-500 shadow-[0_0_10px_#22c55e]' : 'bg-yellow-500 animate-pulse'}`}></div>
            <span className={`font-mono text-sm ${status === 'Connected' ? 'text-green-400' : 'text-yellow-400'}`}>{status}</span>
        </div>
        {keyFragment && (
            <div className="flex items-center gap-2 text-xs font-mono text-cyan-600 bg-black/30 px-2 py-1 rounded border border-cyan-900/30">
                <KeyIcon className="w-3 h-3" />
                <span>EKS: {keyFragment}...</span>
            </div>
        )}
    </div>
);

const VoiceInterface: React.FC<{ status: string }> = ({ status }) => {
    const [isMuted, setIsMuted] = useState(false);
    const [volume, setVolume] = useState(0);

    // Simulate audio visualizer
    useEffect(() => {
        if (status !== 'Connected' || isMuted) {
            setTimeout(() => setVolume(0), 0);
            return;
        }
        const interval = setInterval(() => {
            setVolume(Math.random() * 100);
        }, 100);
        return () => clearInterval(interval);
    }, [status, isMuted]);

    return (
        <div className="flex flex-col items-center justify-center flex-grow p-8 bg-gradient-to-b from-cyan-900/10 to-transparent rounded-xl border border-cyan-800/30 relative overflow-hidden">
            {/* Audio Visualizer Ring */}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-30">
                 <div className={`rounded-full border border-cyan-500 transition-all duration-75`} style={{ width: `${200 + volume}px`, height: `${200 + volume}px`, opacity: volume / 150 }}></div>
                 <div className={`absolute rounded-full border border-purple-500 transition-all duration-100`} style={{ width: `${180 + volume * 0.8}px`, height: `${180 + volume * 0.8}px`, opacity: volume / 200 }}></div>
            </div>

            <div className="w-32 h-32 rounded-full bg-gradient-to-br from-cyan-500 to-purple-600 p-1 mb-6 shadow-[0_0_30px_rgba(6,182,212,0.4)] relative z-10">
                <div className="w-full h-full rounded-full bg-black flex items-center justify-center relative overflow-hidden">
                    <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')] opacity-20"></div>
                    <span className="text-4xl font-bold text-white font-mono">Q</span>
                </div>
                {status === 'Connected' && <div className="absolute bottom-1 right-1 w-6 h-6 bg-green-500 border-2 border-black rounded-full"></div>}
            </div>

            <h2 className="text-2xl font-bold text-white mb-2">Quantum User</h2>
            <p className="text-cyan-500 text-sm mb-8 font-mono tracking-widest">
                {status === 'Connected' ? (isMuted ? 'MIC MUTED' : 'SECURE CHANNEL ACTIVE') : 'ESTABLISHING LINK...'}
            </p>

            <button 
                onClick={() => setIsMuted(!isMuted)}
                disabled={status !== 'Connected'}
                className={`w-16 h-16 rounded-full flex items-center justify-center transition-all duration-300 ${
                    status !== 'Connected' ? 'bg-gray-800 text-gray-500 cursor-not-allowed' :
                    isMuted ? 'bg-red-900/50 text-red-400 border border-red-500 hover:bg-red-900/80' : 
                    'bg-cyan-600 text-white shadow-[0_0_20px_cyan] hover:scale-110 hover:bg-cyan-500'
                }`}
            >
                {isMuted ? <MicOffIcon className="w-8 h-8" /> : <MicIcon className="w-8 h-8" />}
            </button>
        </div>
    );
};

// --- Main Component ---

const QuantumVoiceChat: React.FC = () => {
    const [status, setStatus] = useState('Initializing...');
    const [sharedKey, setSharedKey] = useState<string | null>(null);

    useEffect(() => {
        const connect = async () => {
            setStatus('Entangling Qubits...');
            const key = await establishSecureKey();
            if (key) {
                setSharedKey(key);
                setStatus('Connected');
            } else {
                setStatus('Connection Failed');
            }
        };
        connect();
    }, []);

    return (
        <GlassPanel title={<div className="flex items-center"><MicIcon className="w-5 h-5 mr-2 text-cyan-400" /> Q-VOX: Quantum Voice</div>}>
            <div className="flex flex-col h-full p-4">
                <StatusBar status={status} keyFragment={sharedKey ? sharedKey.substring(0, 16) : ''} />
                <VoiceInterface status={status} />
                
                {/* Security Footer */}
                <div className="mt-4 pt-3 border-t border-cyan-900/50 flex justify-between items-center text-xs text-gray-500 font-mono">
                    <div className="flex items-center gap-2">
                        <ShieldCheckIcon className="w-4 h-4 text-green-500" />
                        <span>E2E QUANTUM ENCRYPTION</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <ActivityIcon className="w-4 h-4 text-purple-500" />
                        <span>LATENCY: 0.04ms</span>
                    </div>
                </div>
            </div>
        </GlassPanel>
    );
};

export default QuantumVoiceChat;
