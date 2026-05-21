
import React, { useEffect, useRef, useState } from 'react';
import { ActivityIcon, Volume2Icon, VolumeXIcon, ZapIcon } from './Icons';

interface AudioSynthesizerProps {
    isActive: boolean;
    isSpeaking?: boolean;
    intensity?: number;
    minimal?: boolean;
}

const AudioSynthesizer: React.FC<AudioSynthesizerProps> = ({ isActive, isSpeaking = false, intensity = 0.5, minimal = false }) => {
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
            gainNode.gain.value = 0;
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
        if (oscillatorRef.current) oscillatorRef.current.stop();
        const osc = audioContextRef.current.createOscillator();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(110, audioContextRef.current.currentTime);
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
                    ctx.fillStyle = `rgba(6, 182, 212, ${isSpeaking ? 0.8 : 0.4})`;
                    ctx.fillRect(x, height - barHeight, barWidth, barHeight);
                    x += barWidth + 1;
                }
            }
            animationFrameRef.current = requestAnimationFrame(render);
        };
        render();
        return () => {
            if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
        };
    }, [isActive, isSpeaking]);

    if (minimal) {
        return (
            <div className="h-4 w-full flex items-center gap-2 overflow-hidden">
                <canvas ref={canvasRef} width={200} height={16} className="h-full flex-grow opacity-60" />
                {!audioInitialized && (
                    <button onClick={initAudio} className="text-[7px] text-cyan-500 font-mono underline bg-transparent border-none p-0">INIT_AUDIO</button>
                )}
            </div>
        );
    }

    return (
        <div className="bg-black/40 border border-cyan-900/30 rounded-lg p-3 overflow-hidden relative group">
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                    <ActivityIcon className={`w-3 h-3 ${isActive ? 'text-cyan-400' : 'text-gray-600'}`} />
                    <span className="text-[10px] font-bold text-cyan-500 uppercase tracking-widest">Live Audio Synth</span>
                </div>
                <div className="flex items-center gap-2">
                    {!audioInitialized ? (
                        <button onClick={initAudio} className="text-[8px] bg-cyan-500/20 text-cyan-300 px-2 py-0.5 rounded border border-cyan-500/50">INITIALIZE</button>
                    ) : (
                        <button onClick={() => setIsMuted(!isMuted)} className="text-cyan-600 hover:text-cyan-400">
                            {isMuted ? <VolumeXIcon className="w-3 h-3" /> : <Volume2Icon className="w-3 h-3" />}
                        </button>
                    )}
                </div>
            </div>
            <div className="h-10 w-full">
                <canvas ref={canvasRef} width={300} height={40} className="w-full h-full" />
            </div>
        </div>
    );
};

export default AudioSynthesizer;
