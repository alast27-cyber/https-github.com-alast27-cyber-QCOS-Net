
import React, { useState, useEffect, useRef } from 'react';
import { 
    SearchIcon, BrainCircuitIcon, CheckCircle2Icon, 
    AlertTriangleIcon, ActivityIcon, XCircleIcon, SparklesIcon 
} from './Icons';

interface AgenticAddressBarProps {
    value: string;
    onChange: (value: string) => void;
    onNavigate: (uri: string) => void;
    isLoading?: boolean;
}

const AgenticAddressBar: React.FC<AgenticAddressBarProps> = ({ value, onChange, onNavigate, isLoading }) => {
    const [isFocused, setIsFocused] = useState(false);
    const [agentThought, setAgentThought] = useState('');
    const [fidelity, setFidelity] = useState(0);
    const [validationError, setValidationError] = useState<string | null>(null);
    const [parseState, setParseState] = useState<{
        protocol: string;
        dqn: string;
        algo: string;
        task: string;
        valid: boolean;
    }>({ protocol: '', dqn: '', algo: '', task: '', valid: false });

    // AI-Native Input Analysis & Validation
    useEffect(() => {
        const input = value.trim();
        const lowerInput = input.toLowerCase();
        
        // 1. CHIPS Protocol Validation Logic
        if (lowerInput.startsWith('chips://')) {
            // Check for spaces (illegal in URIs)
            if (/\s/.test(input)) {
                setTimeout(() => {
                    setValidationError("Syntax Error: Spaces not allowed in Q-URI");
                    setFidelity(0);
                    setParseState({ protocol: 'CHIPS://', dqn: '', algo: '', task: '', valid: false });
                    setAgentThought("Error: Invalid URI formatting.");
                }, 0);
                return;
            }

            // Strict Structure Regex: chips://<NodeID>[.Domain][/Path]
            // Matches: chips://rigel, chips://node-1.finance, chips://app/main
            const chipsStructureRegex = /^(CHIPS:\/\/)([\w-]+)((?:\.[a-zA-Z0-9-]+)*)?(\/[a-zA-Z0-9-._~%!$&'()*+,;=]*)?(?:\?.*)?$/i;
            const match = input.match(chipsStructureRegex);

            if (!match) {
                setTimeout(() => {
                    setValidationError("Malformed URI: Invalid character or format");
                    setFidelity(0);
                    setParseState({ protocol: 'CHIPS://', dqn: '', algo: '', task: '', valid: false });
                    setAgentThought("Error: Address does not match CHIPS protocol.");
                }, 0);
                return;
            }

            const protocol = match[1];
            const dqn = match[2]; // Node ID
            const algo = match[3]; // Domain suffix
            const task = match[4]; // Path

            // Node ID Validation
            if (!dqn) {
                setTimeout(() => {
                    setValidationError("Missing Target Node Identifier");
                    setFidelity(0);
                    setParseState({ protocol, dqn: '', algo: '', task: '', valid: false });
                    setAgentThought("Error: Target node undefined.");
                }, 0);
                return;
            }

            setTimeout(() => {
                setParseState({
                    protocol: protocol || '',
                    dqn: dqn || '',
                    algo: algo || '',
                    task: task || '',
                    valid: true
                });
            }, 0);

            // Simulate Agent Reasoning based on Node Trust
            if (dqn) {
                if (['rigel', 'sirius', 'dqn-alpha', 'store', 'q-vox', 'qmc-finance', 'protocols'].includes(dqn.toLowerCase())) {
                    setTimeout(() => {
                        setAgentThought(`Node '${dqn}' verification passed. Latency: 12ms.`);
                        setFidelity(98);
                    }, 0);
                } else {
                    setTimeout(() => {
                        setAgentThought(`Resolving unknown DQN '${dqn}' via Distributed Registry...`);
                        setFidelity(45);
                    }, 0);
                }
            }

            if (algo) {
                setTimeout(() => setAgentThought(prev => prev + ` Loading context '${algo}'...`), 0);
            }

        } else if (lowerInput.startsWith('http')) {
            setTimeout(() => {
                setParseState({ protocol: 'HTTPS', dqn: '', algo: '', task: '', valid: true });
                setAgentThought("Legacy Web Protocol. Routing via Quantum Gateway...");
                setFidelity(100);
            }, 0);
        } else if (input.length > 0) {
            // Search Mode
            setTimeout(() => {
                setParseState({ protocol: '', dqn: '', algo: '', task: '', valid: false });
                setAgentThought(`Analyzing semantic intent: "${input}"...`);
                setFidelity(60);
            }, 0);
        } else {
            // Idle
            setTimeout(() => {
                setParseState({ protocol: '', dqn: '', algo: '', task: '', valid: false });
                setAgentThought("Agent Q Standing By. Enter Q-URI or Intent.");
                setFidelity(0);
            }, 0);
        }

    }, [value]);

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') {
            if (!validationError) {
                onNavigate(value);
            }
        }
    };

    // Reset validation error when input changes
    useEffect(() => {
        setTimeout(() => setValidationError(null), 0);
    }, [value]);

    return (
        <div className={`relative flex flex-col w-full transition-all duration-500 ${isFocused ? 'scale-[1.01]' : ''}`}>
            
            {/* Input Container */}
            <div className={`
                relative flex items-center w-full h-11 rounded-lg border backdrop-blur-md overflow-hidden transition-all duration-300
                ${validationError 
                    ? 'bg-red-950/30 border-red-500 shadow-[0_0_20px_rgba(239,68,68,0.2)]'
                    : isFocused 
                        ? 'bg-black/80 border-cyan-400 shadow-[0_0_20px_rgba(6,182,212,0.3)]' 
                        : 'bg-black/40 border-cyan-800/50 hover:border-cyan-600'}
            `}>
                {/* Left Icon (Dynamic based on state) */}
                <div className="pl-3 pr-2 flex-shrink-0">
                    {isLoading ? (
                        <ActivityIcon className="w-5 h-5 text-cyan-400 animate-spin" />
                    ) : validationError ? (
                        <XCircleIcon className="w-5 h-5 text-red-500 animate-pulse" />
                    ) : parseState.protocol.toUpperCase().startsWith('CHIPS') ? (
                        <div className="relative">
                            <BrainCircuitIcon className={`w-5 h-5 ${parseState.valid ? 'text-purple-400' : 'text-gray-400'}`} />
                            {parseState.valid && <div className="absolute inset-0 bg-purple-500 blur-md opacity-40 animate-pulse"></div>}
                        </div>
                    ) : (
                        <SearchIcon className="w-5 h-5 text-gray-500" />
                    )}
                </div>

                {/* The Input Field (Transparent Text Layering for Syntax Highlighting) */}
                <div className="relative flex-grow h-full font-mono text-sm overflow-hidden">
                    {/* Syntax Highlighter Layer (Behind Input) */}
                    <div className="absolute inset-0 flex items-center pointer-events-none px-1 whitespace-pre overflow-hidden" aria-hidden="true">
                        <span className={validationError ? "text-red-400" : "text-cyan-600"}>{parseState.protocol}</span>
                        <span className="text-yellow-400 font-bold">{parseState.dqn}</span>
                        <span className="text-purple-400">{parseState.algo}</span>
                        <span className="text-green-400">{parseState.task}</span>
                        {/* Render remaining text if regex didn't match perfectly or for non-chips input */}
                        {!parseState.protocol && <span className="text-transparent">{value}</span>}
                    </div>

                    <input
                        type="text"
                        value={value}
                        onChange={(e) => onChange(e.target.value)}
                        onKeyDown={handleKeyDown}
                        onFocus={() => setIsFocused(true)}
                        onBlur={() => setIsFocused(false)}
                        placeholder="Enter intent or CHIPS:// address..."
                        className={`
                            w-full h-full bg-transparent border-none outline-none px-1 text-white placeholder-gray-700
                            ${parseState.protocol.toUpperCase().startsWith('CHIPS') ? 'text-transparent caret-white selection:bg-cyan-500/30 selection:text-white' : 'text-white'}
                        `}
                        spellCheck={false}
                    />
                </div>

                {/* Right Side: Validation Msg or Fidelity */}
                <div className="pr-3 flex items-center gap-3 flex-shrink-0">
                    {validationError ? (
                        <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-red-900/40 border border-red-500/30">
                            <AlertTriangleIcon className="w-3 h-3 text-red-500" />
                            <span className="text-[10px] text-red-300 font-bold uppercase tracking-wider animate-fade-in whitespace-nowrap">
                                {validationError}
                            </span>
                        </div>
                    ) : (
                        value && (
                            <div className="flex flex-col items-end w-16">
                                <div className="flex items-center gap-1 mb-0.5">
                                    <span className="text-[8px] text-cyan-600 font-bold uppercase tracking-tighter">Fidelity</span>
                                    <span className={`text-[9px] font-mono ${fidelity > 80 ? 'text-green-400' : 'text-yellow-400'}`}>{fidelity}%</span>
                                </div>
                                <div className="w-full h-1 bg-gray-800 rounded-full overflow-hidden">
                                    <div 
                                        className={`h-full transition-all duration-500 ${fidelity > 90 ? 'bg-green-500 shadow-[0_0_5px_#22c55e]' : fidelity > 50 ? 'bg-yellow-500' : 'bg-red-500'}`} 
                                        style={{ width: `${fidelity}%` }}
                                    ></div>
                                </div>
                            </div>
                        )
                    )}
                    
                    {!validationError && parseState.valid && (
                        <CheckCircle2Icon className="w-4 h-4 text-green-500 animate-fade-in" />
                    )}
                </div>
            </div>

            {/* Agent Thought Stream (Ghost Text) */}
            <div className={`
                overflow-hidden transition-all duration-500 ease-out
                ${isFocused || value ? 'max-h-8 opacity-100 mt-1' : 'max-h-0 opacity-0'}
            `}>
                <div className="flex items-center gap-2 px-2 text-[10px] font-mono text-cyan-400/80">
                    <SparklesIcon className="w-3 h-3 animate-pulse" />
                    <span className="uppercase tracking-widest text-cyan-600 font-bold">AGENT Q:</span>
                    <span className="typing-effect truncate">{agentThought}</span>
                </div>
            </div>
            
            <style>{`
                .typing-effect {
                    display: inline-block;
                    overflow: hidden;
                    white-space: nowrap;
                    animation: typing 2s steps(40, end);
                }
                @keyframes typing {
                    from { width: 0 }
                    to { width: 100% }
                }
            `}</style>
        </div>
    );
};

export default AgenticAddressBar;
