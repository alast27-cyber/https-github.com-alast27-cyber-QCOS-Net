
import React, { useState } from 'react';
import GlassPanel from './GlassPanel';
import { EyeIcon, UploadCloudIcon, LoaderIcon } from './Icons';

const ImageAnalysis: React.FC = () => {
    const [image, setImage] = useState<string | null>(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [result, setResult] = useState<string | null>(null);

    const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const reader = new FileReader();
            reader.onload = (ev) => {
                setImage(ev.target?.result as string);
                setResult(null);
            };
            reader.readAsDataURL(e.target.files[0]);
        }
    };

    const handleAnalyze = () => {
        if (!image) return;
        setAnalyzing(true);
        setResult(null);
        setTimeout(() => {
            setAnalyzing(false);
            setResult("Analysis Complete: Quantum pattern coherence detected (98.2%). No structural anomalies found.");
        }, 2000);
    };

    return (
        <GlassPanel title={<div className="flex items-center"><EyeIcon className="w-5 h-5 mr-2 text-cyan-400" /> Quantum Image Analysis</div>}>
            <div className="p-4 h-full flex flex-col gap-4">
                <div className="flex-grow bg-black/40 border border-cyan-900/50 rounded-lg flex items-center justify-center relative overflow-hidden">
                    {image ? (
                        <img src={image} alt="Analysis Target" className="max-h-full max-w-full object-contain" />
                    ) : (
                        <div className="text-center text-cyan-600">
                            <UploadCloudIcon className="w-12 h-12 mx-auto mb-2 opacity-50" />
                            <p className="text-xs">Upload image for quantum scanning</p>
                        </div>
                    )}
                    <input type="file" accept="image/*" onChange={handleUpload} className="absolute inset-0 opacity-0 cursor-pointer" />
                    
                    {analyzing && (
                        <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center z-10 backdrop-blur-sm">
                            <LoaderIcon className="w-10 h-10 text-cyan-400 animate-spin mb-3" />
                            <div className="w-48 h-1 bg-gray-800 rounded-full overflow-hidden">
                                <div className="h-full bg-cyan-400 w-1/2 animate-pulse rounded-full"></div>
                            </div>
                            <p className="text-cyan-300 text-xs mt-2 font-mono animate-pulse">Scanning Quantum State...</p>
                        </div>
                    )}
                </div>
                
                {result && !analyzing && (
                    <div className="p-3 bg-green-900/20 border border-green-500/50 rounded text-xs text-green-300 animate-fade-in">
                        {result}
                    </div>
                )}

                <button 
                    onClick={handleAnalyze} 
                    disabled={!image || analyzing}
                    className="holographic-button w-full py-2 bg-cyan-600/30 border-cyan-500 text-cyan-200 font-bold rounded flex items-center justify-center gap-2 disabled:opacity-50"
                >
                    {analyzing ? <LoaderIcon className="w-4 h-4 animate-spin" /> : <EyeIcon className="w-4 h-4" />}
                    {analyzing ? 'Scanning...' : 'Analyze Pattern'}
                </button>
            </div>
        </GlassPanel>
    );
};

export default ImageAnalysis;
