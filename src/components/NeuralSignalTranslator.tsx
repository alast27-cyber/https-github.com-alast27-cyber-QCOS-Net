import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Brain, Zap, Code, RefreshCw } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer } from 'recharts';
import { InfonEntanglementProtocol } from '../qcos/entanglement';
import { GoogleGenAI } from '@google/genai';

const protocol = new InfonEntanglementProtocol();

const NeuralSignalTranslator: React.FC = () => {
    const [isBridged, setIsBridged] = useState(false);
    const [intent, setIntent] = useState('');
    const [signalData, setSignalData] = useState<{ time: number, val: number }[]>([]);
    const [generatedCode, setGeneratedCode] = useState('');
    const [isTranslating, setIsTranslating] = useState(false);
    const tickRef = useRef(0);

    const handleBridge = () => {
        protocol.bridgeCognitionToNeural('HUMAN_COGNITION', 'NEURAL_INTERFACE_PLATFORM');
        setIsBridged(true);
    };

    const handleExecuteIntent = async () => {
        if (!isBridged) return;
        setIsTranslating(true);
        setGeneratedCode('');

        // Simulate signal recording
        const newData = [];
        for (let i = 0; i < 50; i++) {
            newData.push({ time: tickRef.current++, val: Math.random() * 100 });
        }
        setSignalData(newData);

        // AgentQ Translation (using Gemini)
        try {
            const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });
            const response = await ai.models.generateContent({
                model: 'gemini-3-flash-preview',
                contents: `Translate the following neural signal pattern and user intent into a code algorithm. Intent: "${intent}". Neural signal pattern: ${JSON.stringify(newData.slice(0, 10))}. Output only the code algorithm.`,
            });
            setGeneratedCode(response.text || '// No code generated');
        } catch (error) {
            console.error('Translation error:', error);
            setGeneratedCode('// Error generating code');
        } finally {
            setIsTranslating(false);
        }
    };

    return (
        <div className="p-6 bg-black/60 rounded-xl border border-orange-500/30 text-orange-100 font-mono">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Brain className="text-orange-400" /> Neural Signal Translator
            </h3>

            <div className="flex gap-4 mb-6">
                <button 
                    onClick={handleBridge}
                    className={`px-4 py-2 rounded font-bold text-xs uppercase ${isBridged ? 'bg-green-600/30 border border-green-500' : 'bg-orange-600/30 border border-orange-500'}`}
                >
                    {isBridged ? 'Bridge Active' : 'Execute Neural Bridge'}
                </button>
            </div>

            {isBridged && (
                <div className="space-y-4">
                    <input 
                        type="text" 
                        value={intent} 
                        onChange={(e) => setIntent(e.target.value)}
                        placeholder="Execute intention through thought..."
                        className="w-full p-2 bg-black/50 border border-orange-500/30 rounded text-sm"
                    />
                    <button 
                        onClick={handleExecuteIntent}
                        disabled={isTranslating || !intent}
                        className="px-4 py-2 bg-purple-600/30 border border-purple-500 rounded font-bold text-xs uppercase disabled:opacity-50"
                    >
                        {isTranslating ? 'Translating...' : 'Execute & Translate'}
                    </button>
                </div>
            )}

            {signalData.length > 0 && (
                <div className="mt-6 h-40 bg-black/40 rounded p-2 border border-cyan-900/30">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={signalData}>
                            <XAxis dataKey="time" hide />
                            <YAxis hide />
                            <Area type="monotone" dataKey="val" stroke="#f97316" fill="#f97316" fillOpacity={0.2} />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            )}

            {generatedCode && (
                <div className="mt-6 p-4 bg-black/80 rounded border border-purple-500/30">
                    <h4 className="text-xs text-purple-400 mb-2 flex items-center gap-2"><Code size={12}/> AgentQ Algorithm:</h4>
                    <pre className="text-xs text-green-300 overflow-auto">{generatedCode}</pre>
                </div>
            )}
        </div>
    );
};

export default NeuralSignalTranslator;
