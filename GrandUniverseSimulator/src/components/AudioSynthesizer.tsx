
import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ActivityIcon, Volume2Icon, VolumeXIcon, SparklesIcon, ZapIcon } from './Icons';

interface AudioSynthesizerProps {
    isActive: boolean;
    isSpeaking?: boolean;
    intensity?: number;
}

const AudioSynthesizer: React.FC<AudioSynthesizerProps> = ({ isActive, isSpeaking = false, intensity = 0.5 }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const oscillatorRef = useRef<OscillatorNode | null>(null);
    const gainNodeRef = useRef<GainNode | null>(null);
    const animationFrameRef = useRef<number | null>(null);
    
    const [isMuted, setIsMuted] = useState(false);
    const [audioInitialized, setAudioInitialized] = useState(false);

    const initAudio = () => {
        if (audioContextRef.current) return;

        try {
            const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
            const ctx = new AudioContextClass();
            const analyser = ctx.createAnalyser();
            analyser.fftSize = 256;
            
            const gainNode = ctx.createGain();
            gainNode.gain.value = 0; // Start silent
            
            gainNode.connect(analyser);
            analyser.connect(ctx.destination);
            
            audioContextRef.current = ctx;
            analyserRef.current = analyser;
            gainNodeRef.current = gainNode;
            setAudioInitialized(true);
        } catch (e) {
            console.error("Failed to initialize AudioContext:", e);
        }
    };

    const startSynth = () => {
        if (!audioContextRef.current || !gainNodeRef.current) return;
        
        if (oscillatorRef.current) {
            oscillatorRef.current.stop();
        }

        const osc = audioContextRef.current.createOscillator();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(110, audioContextRef.current.currentTime); // Low drone
        
        const filter = audioContextRef.current.createBiquadFilter();
        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(400, audioContextRef.current.currentTime);
        
        osc.connect(filter);
        filter.connect(gainNodeRef.current);
        
        osc.start();
        oscillatorRef.current = osc;
    };

    useEffect(() => {
        if (isActive && audioInitialized && !isMuted) {
            startSynth();
        } else {
            if (oscillatorRef.current) {
                oscillatorRef.current.stop();
                oscillatorRef.current = null;
            }
        }
        
        return () => {
            if (oscillatorRef.current) {
                oscillatorRef.current.stop();
            }
        };
    }, [isActive, audioInitialized, isMuted]);

    useEffect(() => {
        if (!gainNodeRef.current || !audioContextRef.current) return;
        
        const targetGain = (isActive && !isMuted) ? (isSpeaking ? 0.15 : 0.05) : 0;
        gainNodeRef.current.gain.setTargetAtTime(targetGain, audioContextRef.current.currentTime, 0.1);
        
        if (oscillatorRef.current) {
            const freq = isSpeaking ? 220 + Math.random() * 20 : 110;
            oscillatorRef.current.frequency.setTargetAtTime(freq, audioContextRef.current.currentTime, 0.2);
        }
    }, [isActive, isSpeaking, isMuted]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const render = () => {
            const width = canvas.width;
            const height = canvas.height;
            ctx.clearRect(0, 0, width, height);

            if (analyserRef.current && isActive) {
                const bufferLength = analyserRef.current.frequencyBinCount;
                const dataArray = new Uint8Array(bufferLength);
                analyserRef.current.getByteFrequencyData(dataArray);

                const barWidth = (width / bufferLength) * 2.5;
                let x = 0;

                for (let i = 0; i < bufferLength; i++) {
                    const barHeight = (dataArray[i] / 255) * height;
                    
                    const r = 6;
                    const g = 182;
                    const b = 212;
                    
                    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${isSpeaking ? 0.8 : 0.4})`;
                    ctx.fillRect(x, height - barHeight, barWidth, barHeight);

                    x += barWidth + 1;
                }
                
                // Draw a center line
                ctx.beginPath();
                ctx.moveTo(0, height / 2);
                ctx.lineTo(width, height / 2);
                ctx.strokeStyle = 'rgba(6, 182, 212, 0.2)';
                ctx.stroke();
            } else {
                // Idle line
                ctx.beginPath();
                ctx.moveTo(0, height / 2);
                ctx.lineTo(width, height / 2);
                ctx.strokeStyle = 'rgba(6, 182, 212, 0.1)';
                ctx.stroke();
            }

            animationFrameRef.current = requestAnimationFrame(render);
        };

        render();
        return () => {
            if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
        };
    }, [isActive, isSpeaking]);

    return (
        <div className="bg-black/40 border border-cyan-900/30 rounded-lg p-3 overflow-hidden relative group">
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                    <ActivityIcon className={`w-3 h-3 ${isActive ? 'text-cyan-400' : 'text-gray-600'}`} />
                    <span className="text-[10px] font-bold text-cyan-500 uppercase tracking-widest">Live Audio Synth</span>
                </div>
                <div className="flex items-center gap-2">
                    {!audioInitialized ? (
                        <button 
                            onClick={initAudio}
                            className="text-[8px] bg-cyan-500/20 text-cyan-300 px-2 py-0.5 rounded border border-cyan-500/50 hover:bg-cyan-500/40 transition-all"
                        >
                            INITIALIZE
                        </button>
                    ) : (
                        <button 
                            onClick={() => setIsMuted(!isMuted)}
                            className="text-cyan-600 hover:text-cyan-400 transition-colors"
                        >
                            {isMuted ? <VolumeXIcon className="w-3 h-3" /> : <Volume2Icon className="w-3 h-3" />}
                        </button>
                    )}
                </div>
            </div>
            
            <div className="h-12 w-full relative">
                <canvas 
                    ref={canvasRef} 
                    width={300} 
                    height={48} 
                    className="w-full h-full"
                />
                
                <AnimatePresence>
                    {isSpeaking && (
                        <motion.div 
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="absolute inset-0 pointer-events-none flex items-center justify-center"
                        >
                            <div className="w-full h-[1px] bg-cyan-400/30 blur-sm animate-pulse"></div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
            
            <div className="mt-2 flex justify-between items-center">
                <div className="flex gap-1">
                    <div className={`w-1 h-1 rounded-full ${isActive ? 'bg-cyan-500 animate-pulse' : 'bg-gray-800'}`}></div>
                    <div className={`w-1 h-1 rounded-full ${isSpeaking ? 'bg-purple-500 animate-ping' : 'bg-gray-800'}`}></div>
                    <div className={`w-1 h-1 rounded-full ${audioInitialized ? 'bg-green-500' : 'bg-gray-800'}`}></div>
                </div>
                <div className="flex items-center gap-1 text-[8px] text-cyan-700 font-mono">
                    <ZapIcon className="w-2 h-2" />
                    <span>Q-MODULATION: {isSpeaking ? 'HIGH' : 'LOW'}</span>
                </div>
            </div>
        </div>
    );
};

export default AudioSynthesizer;
