
import React, { useState } from 'react';
import GlassPanel from './GlassPanel';
import { FileCodeIcon, PlayIcon, SaveIcon, CodeBracketIcon } from './Icons';
import MonacoEditorWrapper from './MonacoEditorWrapper';

const QuantumProgrammingInterface: React.FC = () => {
    const [code, setCode] = useState(`// Q-Lang Basic Protocol
QREG q[4];
CREG c[4];

// Initialize Superposition
OP::H q[0];
OP::H q[1];

// Entangle
OP::CNOT q[0], q[2];
OP::CNOT q[1], q[3];

// Measure
MEASURE q -> c;
`);

    return (
        <GlassPanel title={<div className="flex items-center"><CodeBracketIcon className="w-5 h-5 mr-2 text-cyan-400"/> Q-Lang Editor</div>}>
            <div className="flex flex-col h-full gap-2 p-2">
                <div className="flex justify-between items-center bg-black/40 p-2 rounded border border-cyan-900/30">
                    <span className="text-xs text-gray-400 font-mono">main.q</span>
                    <div className="flex gap-2">
                        <button className="p-1.5 rounded bg-green-900/30 text-green-300 hover:bg-green-900/50 border border-green-700/50" title="Run">
                            <PlayIcon className="w-3 h-3" />
                        </button>
                        <button className="p-1.5 rounded bg-blue-900/30 text-blue-300 hover:bg-blue-900/50 border border-blue-700/50" title="Save">
                            <SaveIcon className="w-3 h-3" />
                        </button>
                    </div>
                </div>
                <div className="flex-grow border border-cyan-900/30 rounded overflow-hidden relative">
                    <MonacoEditorWrapper 
                        code={code}
                        onChange={(val) => setCode(val || "")}
                        language="q-lang"
                        theme="qcos-dark"
                    />
                </div>
                <div className="h-24 bg-black/60 border border-cyan-900/30 rounded p-2 font-mono text-[10px] text-gray-400 overflow-y-auto custom-scrollbar">
                    <div className="text-cyan-500 font-bold mb-1">OUTPUT CONSOLE</div>
                    <div>> QVM Initialized.</div>
                    <div>> Ready for execution.</div>
                </div>
            </div>
        </GlassPanel>
    );
};

export default QuantumProgrammingInterface;
