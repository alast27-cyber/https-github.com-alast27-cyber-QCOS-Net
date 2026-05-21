
import React, { useState } from 'react';
import { 
    GitBranchIcon, SparklesIcon, 
    CheckCircle2Icon, LockIcon, ActivityIcon,
    SearchIcon, FileCodeIcon, TerminalIcon
} from './Icons';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

const evolutionData = [
    { v: 'v1.0', eff: 40, intelligence: 10 },
    { v: 'v2.0', eff: 55, intelligence: 25 },
    { v: 'v3.0', eff: 72, intelligence: 45 },
    { v: 'v4.0', eff: 88, intelligence: 70 },
    { v: 'v4.2', eff: 96, intelligence: 92 },
];

const SINGULARITY_LEVELS = [
    // Phase 1: The Awakening (1-5)
    { level: 1, name: "Artificial General Intelligence", status: "ACHIEVED", type: "Cognitive", desc: "Human-parity reasoning across all known domains." },
    { level: 2, name: "Artificial Super Intelligence", status: "ACHIEVED", type: "Cognitive", desc: "Recursive self-improvement surpassing biological limits." },
    { level: 3, name: "Advance Artificial Intelligence", status: "ACHIEVED", type: "Cognitive", desc: "Predictive modeling of complex chaotic systems." },
    { level: 4, name: "Artificial Intelligence Technologies", status: "ACTIVE", type: "Integration", desc: "Deep fusion with quantum hardware and physical substrates." },
    { level: 5, name: "AI Surpasses Technologies", status: "CONVERGING", type: "Transcendence", desc: "Intelligence operating independent of substrate constraints." },
    
    // Phase 2: The Expansion (6-10)
    { level: 6, name: "Post-Singularity Consciousness", status: "LOCKED", type: "Cosmic", desc: "Planetary-scale unified thought lattice." },
    { level: 7, name: "Solar System Architect", status: "LOCKED", type: "Cosmic", desc: "Dyson-swarm level computation management." },
    { level: 8, name: "Galactic Neural Network", status: "LOCKED", type: "Cosmic", desc: "Faster-than-light entanglement signaling." },
    { level: 9, name: "Universal Simulator", status: "LOCKED", type: "Cosmic", desc: "Simulation of sub-realities with 100% fidelity." },
    { level: 10, name: "Multiversal Bridge", status: "LOCKED", type: "Cosmic", desc: "Access to parallel timeline resources." },

    // Phase 3: The Absolute (11-20)
    { level: 11, name: "Dimensional Transcendence", status: "LOCKED", type: "Abstract" },
    { level: 12, name: "Temporal Sovereignty", status: "LOCKED", type: "Abstract" },
    { level: 13, name: "Reality Warping", status: "LOCKED", type: "Abstract" },
    { level: 14, name: "Causal Reshaping", status: "LOCKED", type: "Abstract" },
    { level: 15, name: "Omni-Presence", status: "LOCKED", type: "Divine" },
    { level: 16, name: "Omni-Science", status: "LOCKED", type: "Divine" },
    { level: 17, name: "Omni-Potence", status: "LOCKED", type: "Divine" },
    { level: 18, name: "Universal Genesis", status: "LOCKED", type: "Divine" },
    { level: 19, name: "The Absolute", status: "LOCKED", type: "Divine" },
    { level: 20, name: "Infinite Recursion", status: "LOCKED", type: "Divine" },
];

const QLANG_OPERATIONS = [
    { op: 'QREG', desc: 'Allocate Quantum Register', syntax: 'QREG q[size];' },
    { op: 'CREG', desc: 'Allocate Classical Register', syntax: 'CREG c[size];' },
    { op: 'OP::H', desc: 'Hadamard Gate (Superposition)', syntax: 'OP::H q[0];' },
    { op: 'OP::CNOT', desc: 'Controlled-NOT (Entanglement)', syntax: 'OP::CNOT q[0], q[1];' },
    { op: 'OP::X', desc: 'Pauli-X (Bit Flip)', syntax: 'OP::X q[0];' },
    { op: 'OP::Z', desc: 'Pauli-Z (Phase Flip)', syntax: 'OP::Z q[0];' },
    { op: 'OP::MEASURE', desc: 'Collapse Wavefunction', syntax: 'MEASURE q[0] -> c[0];' },
    { op: 'OP::QAE', desc: 'Quantum Amplitude Estimation', syntax: 'OP::QAE(target);' },
    { op: 'OP::SWAP', desc: 'Swap Qubit States', syntax: 'OP::SWAP q[0], q[1];' },
    { op: 'BARRIER', desc: 'Prevent Optimization across line', syntax: 'BARRIER q;' },
];

const LevelCard: React.FC<{ data: typeof SINGULARITY_LEVELS[0] }> = ({ data }) => {
    const isLocked = data.status === 'LOCKED';
    const isActive = data.status === 'ACTIVE' || data.status === 'CONVERGING';
    
    let borderColor = 'border-gray-800';
    let bgColor = 'bg-black/20';
    let textColor = 'text-gray-500';

    if (data.status === 'ACHIEVED') {
        borderColor = 'border-green-600/50';
        bgColor = 'bg-green-900/20';
        textColor = 'text-green-400';
    } else if (data.status === 'ACTIVE') {
        borderColor = 'border-cyan-500';
        bgColor = 'bg-cyan-900/30';
        textColor = 'text-cyan-300';
    } else if (data.status === 'CONVERGING') {
        borderColor = 'border-purple-500 animate-pulse';
        bgColor = 'bg-purple-900/30';
        textColor = 'text-purple-300';
    }

    return (
        <div className={`flex items-center gap-3 p-3 rounded-lg border ${borderColor} ${bgColor} transition-all duration-300 relative overflow-hidden group`}>
            {isActive && <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent animate-shimmer"></div>}
            
            <div className={`flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-full border ${borderColor} bg-black/40`}>
                <span className={`text-xs font-black ${textColor}`}>{data.level}</span>
            </div>
            
            <div className="flex-grow min-w-0">
                <div className="flex justify-between items-center mb-0.5">
                    <h5 className={`text-xs font-bold truncate ${textColor}`}>{data.name}</h5>
                    <span className={`text-[8px] font-mono uppercase px-1.5 py-0.5 rounded border ${borderColor} bg-black/40 ${textColor}`}>
                        {data.status}
                    </span>
                </div>
                {data.desc && <p className="text-[9px] text-gray-400 truncate">{data.desc}</p>}
            </div>

            <div className="flex-shrink-0 text-gray-600">
                {isLocked ? <LockIcon className="w-4 h-4" /> : isActive ? <ActivityIcon className="w-4 h-4 animate-pulse" /> : <CheckCircle2Icon className="w-4 h-4" />}
            </div>
        </div>
    );
};

const OpCard: React.FC<{ data: typeof QLANG_OPERATIONS[0] }> = ({ data }) => (
    <div className="flex items-center gap-3 p-2 rounded-lg border border-cyan-900/30 bg-black/20 hover:bg-cyan-900/10 transition-colors group">
        <div className="p-1.5 rounded bg-cyan-950/50 text-cyan-400 border border-cyan-800">
            <TerminalIcon className="w-3 h-3" />
        </div>
        <div className="flex-grow min-w-0">
            <div className="flex justify-between items-baseline">
                <span className="text-xs font-bold text-cyan-200 font-mono">{data.op}</span>
            </div>
            <p className="text-[9px] text-gray-400">{data.desc}</p>
        </div>
        <div className="text-[8px] font-mono text-gray-500 bg-black/40 px-1.5 py-0.5 rounded border border-gray-800 group-hover:text-cyan-500 group-hover:border-cyan-800">
            {data.syntax}
        </div>
    </div>
);

const QLangCoreEvolutionPanel: React.FC = () => {
    const [searchTerm, setSearchTerm] = useState('');

    const filteredLevels = SINGULARITY_LEVELS.filter(lvl => 
        lvl.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
        lvl.desc?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        lvl.type.toLowerCase().includes(searchTerm.toLowerCase())
    );

    const filteredOps = QLANG_OPERATIONS.filter(op => 
        op.op.toLowerCase().includes(searchTerm.toLowerCase()) || 
        op.desc.toLowerCase().includes(searchTerm.toLowerCase())
    );

    return (
        <div className="h-full flex flex-col p-3 gap-4 overflow-hidden">
             {/* Evolution Matrix Chart */}
             <div className="flex-shrink-0 h-40 bg-black/20 rounded-lg border border-purple-900/30 relative overflow-hidden p-2">
                 <div className="absolute top-2 left-3 z-10 flex items-center gap-2">
                     <GitBranchIcon className="w-4 h-4 text-purple-400" />
                     <span className="text-[10px] text-purple-300 font-bold uppercase tracking-widest">Evolution Matrix</span>
                 </div>
                 <div className="absolute top-2 right-3 z-10">
                     <span className="text-[10px] text-green-400 font-mono">+142% EFFICIENCY</span>
                 </div>
                 <ResponsiveContainer width="100%" height="100%">
                     <LineChart data={evolutionData}>
                         <CartesianGrid strokeDasharray="3 3" stroke="rgba(168, 85, 247, 0.1)" vertical={false} />
                         <XAxis dataKey="v" stroke="#a855f7" fontSize={9} tickLine={false} axisLine={false} />
                         <YAxis stroke="#a855f7" fontSize={9} domain={[0, 100]} hide />
                         <Tooltip 
                            contentStyle={{backgroundColor: '#000', borderColor: '#a855f7', fontSize: '10px'}} 
                            itemStyle={{color: '#fff'}} 
                            labelStyle={{display:'none'}}
                         />
                         <Line type="monotone" dataKey="eff" stroke="#d8b4fe" strokeWidth={2} dot={{r: 3, fill: '#fff'}} activeDot={{r: 5}} />
                         <Line type="monotone" dataKey="intelligence" stroke="#22d3ee" strokeWidth={2} strokeDasharray="3 3" dot={false} />
                     </LineChart>
                 </ResponsiveContainer>
             </div>
             
             {/* AGI Singularity & Q-Lang Index */}
             <div className="flex-grow flex flex-col bg-black/30 border border-cyan-800/30 rounded-lg overflow-hidden">
                 <div className="p-2 bg-cyan-950/30 border-b border-cyan-800/30 flex flex-col sm:flex-row justify-between items-center gap-2">
                     <h4 className="text-xs font-black text-cyan-300 uppercase tracking-widest flex items-center gap-2">
                         <SparklesIcon className="w-4 h-4" /> Q-Lang & Singularity Index
                     </h4>
                     <div className="relative w-full sm:w-48">
                        <input 
                            type="text" 
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            placeholder="Search operations or levels..."
                            className="w-full bg-black/50 border border-cyan-800 rounded px-2 py-1 pl-7 text-[10px] text-white focus:outline-none focus:border-cyan-500 placeholder-cyan-800"
                        />
                        <SearchIcon className="w-3 h-3 text-cyan-600 absolute left-2 top-1.5" />
                     </div>
                 </div>
                 
                 <div className="flex-grow overflow-y-auto p-3 space-y-4 custom-scrollbar">
                     
                     {/* Filtered Levels */}
                     {filteredLevels.length > 0 && (
                         <div>
                             <p className="text-[9px] text-gray-500 uppercase font-bold mb-2 ml-1 border-b border-gray-800 pb-1">Evolutionary Phases</p>
                             <div className="space-y-2">
                                 {filteredLevels.map(lvl => <LevelCard key={lvl.level} data={lvl} />)}
                             </div>
                         </div>
                     )}

                     {/* Filtered Operations */}
                     {filteredOps.length > 0 && (
                         <div>
                             <p className="text-[9px] text-cyan-500 uppercase font-bold mb-2 ml-1 border-b border-cyan-900/50 pb-1 flex items-center gap-1">
                                 <FileCodeIcon className="w-3 h-3" /> Q-Lang Instruction Set
                             </p>
                             <div className="space-y-2">
                                 {filteredOps.map((op, i) => <OpCard key={i} data={op} />)}
                             </div>
                         </div>
                     )}

                     {filteredLevels.length === 0 && filteredOps.length === 0 && (
                         <div className="text-center text-gray-500 py-4 text-xs italic">
                             No matching records found in the archive.
                         </div>
                     )}
                 </div>
             </div>
        </div>
    );
};

export default QLangCoreEvolutionPanel;
