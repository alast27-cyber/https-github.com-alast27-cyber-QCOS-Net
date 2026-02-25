
import React, { useState, useEffect } from 'react';
import GlassPanel from './GlassPanel';
import { ActivityIcon, PlayIcon, RefreshCwIcon, ZapIcon } from './Icons';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts';

const QDeNoiseProcessor: React.FC = () => {
    const [isProcessing, setIsProcessing] = useState(false);
    const [data, setData] = useState<{t: number, original: number, noise: number, clean: number | null}[]>([]);
    const [fidelity, setFidelity] = useState(0);

    const generateSignal = () => {
        const newData = [];
        for(let i=0; i<50; i++) {
            const signal = Math.sin(i * 0.2) * 50 + 50;
            const noise = signal + (Math.random() - 0.5) * 40;
            newData.push({ t: i, original: signal, noise: noise, clean: null });
        }
        setData(newData);
        setFidelity(0);
        setIsProcessing(false);
    };

    useEffect(() => {
        setTimeout(() => generateSignal(), 0);
    }, []);

    const processSignal = () => {
        setIsProcessing(true);
        let progress = 0;
        
        const interval = setInterval(() => {
            progress += 2;
            setData(prev => prev.map((pt, i) => {
                if (i < progress) {
                    // Quantum Error Correction Smoothing
                    return { ...pt, clean: pt.original + (Math.random() - 0.5) * 5 }; // Close to original
                }
                return pt;
            }));

            if (progress >= 50) {
                clearInterval(interval);
                setIsProcessing(false);
                setFidelity(96.4 + Math.random() * 2);
            }
        }, 50);
    };

    return (
        <GlassPanel title={<div className="flex items-center"><ActivityIcon className="w-5 h-5 mr-2 text-blue-400" /> Q-DeNoise Signal Processor</div>}>
            <div className="flex flex-col h-full p-4 gap-4">
                
                {/* Stats */}
                <div className="flex justify-between items-center bg-black/30 p-3 rounded border border-blue-900/50">
                    <div>
                        <p className="text-[10px] text-gray-400 uppercase font-bold">Signal-to-Noise (SNR)</p>
                        <p className="text-xl font-mono text-white">{fidelity > 0 ? '+24dB' : '12dB'}</p>
                    </div>
                    <div className="text-right">
                         <p className="text-[10px] text-gray-400 uppercase font-bold">Signal Fidelity</p>
                         <p className={`text-xl font-mono ${fidelity > 0 ? 'text-green-400' : 'text-yellow-400'}`}>
                             {fidelity > 0 ? `${fidelity.toFixed(2)}%` : '---'}
                         </p>
                    </div>
                </div>

                {/* Chart */}
                <div className="flex-grow bg-black/40 border border-blue-900/30 rounded-lg p-2 min-h-[150px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={data}>
                            <XAxis dataKey="t" hide />
                            <YAxis domain={[0, 120]} hide />
                            <Tooltip contentStyle={{backgroundColor: '#000', fontSize:'10px'}} itemStyle={{padding:0}} />
                            <Line type="monotone" dataKey="noise" stroke="#ef4444" strokeWidth={1} dot={false} strokeOpacity={0.5} name="Noisy Input" />
                            {fidelity > 0 || isProcessing ? (
                                <Line type="monotone" dataKey="clean" stroke="#22d3ee" strokeWidth={2} dot={false} name="QEC Output" />
                            ) : null}
                            <Line type="monotone" dataKey="original" stroke="#4ade80" strokeWidth={1} strokeDasharray="3 3" dot={false} name="Ground Truth" />
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                {/* Controls */}
                <div className="grid grid-cols-2 gap-3">
                    <button 
                        onClick={generateSignal} 
                        disabled={isProcessing}
                        className="holographic-button py-2 rounded bg-gray-700/30 border-gray-600 text-gray-300 text-xs font-bold flex items-center justify-center gap-2"
                    >
                        <RefreshCwIcon className="w-4 h-4"/> New Signal
                    </button>
                    <button 
                        onClick={processSignal} 
                        disabled={isProcessing || fidelity > 0}
                        className={`holographic-button py-2 rounded text-xs font-bold flex items-center justify-center gap-2 ${isProcessing ? 'bg-blue-600/50 text-white cursor-wait' : 'bg-blue-600/20 border-blue-500 text-blue-300 hover:bg-blue-600/40'}`}
                    >
                        {isProcessing ? <ZapIcon className="w-4 h-4 animate-pulse"/> : <PlayIcon className="w-4 h-4"/>}
                        {isProcessing ? 'Denoising...' : 'Run QEC Filter'}
                    </button>
                </div>
            </div>
        </GlassPanel>
    );
};

export default QDeNoiseProcessor;
