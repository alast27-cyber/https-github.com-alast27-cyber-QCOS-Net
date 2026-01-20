import React from 'react';
import { CpuChipIcon } from './Icons';

// --- Types and Components for Circuit Visualization ---
export interface Gate {
    type: 'H' | 'X' | 'Y' | 'Z' | 'S' | 'T' | 'I' | 'RZ' | 'RY' | 'CNOT' | 'CZ' | 'MEASURE';
    target: number;
    control?: number;
    parameter?: string;
}

export interface ParsedCircuit {
    qubits: number;
    circuit: Gate[];
}

const GATE_VISUALS: { [key: string]: { color: string; symbol: string } } = {
    'H': { color: '#38bdf8', symbol: 'H' },   // blue-400
    'X': { color: '#a78bfa', symbol: 'X' },   // violet-400
    'Y': { color: '#4ade80', symbol: 'Y' },   // green-400
    'Z': { color: '#facc15', symbol: 'Z' },   // yellow-400
    'S': { color: '#2dd4bf', symbol: 'S' },   // teal-400
    'T': { color: '#fb923c', symbol: 'T' },   // orange-400
    'I': { color: '#9ca3af', symbol: 'I' },   // gray-400
    'RZ': { color: '#f472b6', symbol: 'RZ' }, // pink-400
    'RY': { color: '#f472b6', symbol: 'RY' }, // pink-400 (same as RZ)
};

const GATE_WIDTH = 25;
const GATE_PADDING = 35; // Increased padding for parameters
const QUBIT_SPACING = 40;

export const CircuitVisualizer: React.FC<{ circuitData: ParsedCircuit | null }> = ({ circuitData }) => {
    if (!circuitData) {
        return (
            <div className="w-full h-full flex flex-col items-center justify-center text-cyan-500 text-center">
                <CpuChipIcon className="w-10 h-10 mb-2" />
                <p>No valid Q-Lang file found to visualize.</p>
                <p className="text-xs">Generate an app or ensure a `.q` file exists in the editor.</p>
            </div>
        );
    }
    const { qubits, circuit } = circuitData;
    if (qubits === 0 && circuit.length === 0) {
        return (
             <div className="w-full h-full flex flex-col items-center justify-center text-cyan-500 text-center">
                <CpuChipIcon className="w-10 h-10 mb-2" />
                <p>Empty circuit.</p>
                <p className="text-xs">Define qubits (e.g., QREG q[2];) to begin.</p>
            </div>
        )
    }

    const height = qubits * QUBIT_SPACING + 20;
    const maxSteps = circuit.length > 0 ? circuit.length : 1;
    const width = (maxSteps + 1) * (GATE_WIDTH + GATE_PADDING);

    return (
        <div className="w-full h-full p-4 flex items-start justify-start overflow-auto">
            <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
                {Array.from({ length: qubits }).map((_, i) => (
                    <g key={`q-line-${i}`}>
                        <line x1="20" y1={20 + i * QUBIT_SPACING} x2={width - 20} y2={20 + i * QUBIT_SPACING} stroke="white" />
                        <text x="0" y={24 + i * QUBIT_SPACING} fill="cyan" fontSize="10">{`q${i}`}</text>
                    </g>
                ))}
                {circuit.map((gate, step) => {
                    const x = 30 + step * (GATE_WIDTH + GATE_PADDING);
                    const y = 10 + gate.target * QUBIT_SPACING;
                    const midX = x + GATE_WIDTH / 2;
                    
                    if (gate.type === 'CNOT' && gate.control !== undefined) {
                        const controlY = 20 + gate.control * QUBIT_SPACING;
                        const targetY = 20 + gate.target * QUBIT_SPACING;
                        return (
                            <g key={`gate-${step}`}>
                                <title>{`CNOT (Control: q${gate.control}, Target: q${gate.target})`}</title>
                                <line x1={midX} y1={controlY} x2={midX} y2={targetY} stroke="white" />
                                <circle cx={midX} cy={controlY} r="4" fill="white" />
                                <circle cx={midX} cy={targetY} r="8" stroke="white" strokeWidth="1.5" fill="none" />
                                <line x1={midX - 5} y1={targetY} x2={midX + 5} y2={targetY} stroke="white" strokeWidth="1.5" />
                                <line x1={midX} y1={targetY - 5} x2={midX} y2={targetY + 5} stroke="white" strokeWidth="1.5" />
                            </g>
                        );
                    } else if (gate.type === 'CZ' && gate.control !== undefined) {
                        const controlY = 20 + gate.control * QUBIT_SPACING;
                        const targetY = 20 + gate.target * QUBIT_SPACING;
                        return (
                            <g key={`gate-${step}`}>
                                <title>{`CZ (Control: q${gate.control}, Target: q${gate.target})`}</title>
                                <line x1={midX} y1={controlY} x2={midX} y2={targetY} stroke="white" />
                                <circle cx={midX} cy={controlY} r="4" fill="white" />
                                <circle cx={midX} cy={targetY} r="4" fill="white" />
                            </g>
                        );
                    } else if (gate.type === 'MEASURE') {
                        const midY = y + GATE_WIDTH / 2;
                        const pointerAngle = -Math.PI / 4;
                        const pointerLength = GATE_WIDTH * 0.3;
                        return (
                            <g key={`gate-${step}`}>
                                <title>{`Measure q${gate.target}`}</title>
                                <rect x={x} y={y} width={GATE_WIDTH} height={GATE_WIDTH} fill="#14b8a6" stroke="white" strokeWidth="1" rx="2" />
                                <path d={`M ${x + 4},${y + GATE_WIDTH - 4} A ${GATE_WIDTH / 2 - 4},${GATE_WIDTH / 2 - 4} 0 1 1 ${x + GATE_WIDTH - 4},${y + GATE_WIDTH - 4}`} stroke="black" strokeWidth="1.5" fill="none" />
                                <line x1={midX} y1={y + GATE_WIDTH - 4} x2={midX + Math.cos(pointerAngle) * pointerLength} y2={y + GATE_WIDTH - 4 + Math.sin(pointerAngle) * pointerLength} stroke="black" strokeWidth="1.5" />
                            </g>
                        );
                    } else if ((gate.type === 'RZ' || gate.type === 'RY') && gate.parameter) {
                        const visual = GATE_VISUALS[gate.type];
                        if (!visual) return null;
                        return (
                            <g key={`gate-${step}`} transform={`translate(${x}, ${y})`}>
                                <title>{`${gate.type}(${gate.parameter}) on q${gate.target}`}</title>
                                <rect width={GATE_WIDTH} height={GATE_WIDTH} fill={visual.color} stroke="white" strokeWidth="1" rx="2" />
                                <text x={GATE_WIDTH / 2} y={GATE_WIDTH / 2 + 5} fill="black" fontSize="12" fontWeight="bold" textAnchor="middle">{visual.symbol}</text>
                                <text x={GATE_WIDTH / 2} y={GATE_WIDTH + 12} fill="cyan" fontSize="8" textAnchor="middle">{`(${gate.parameter})`}</text>
                            </g>
                        );
                    } else {
                        const visual = GATE_VISUALS[gate.type as keyof typeof GATE_VISUALS];
                        if (!visual) return null;
                        const midY = y + GATE_WIDTH / 2;
                        return (
                            <g key={`gate-${step}`}>
                                <title>{`${gate.type} on q${gate.target}`}</title>
                                <rect x={x} y={y} width={GATE_WIDTH} height={GATE_WIDTH} fill={visual.color} stroke="white" strokeWidth="1" rx="2" />
                                <text x={midX} y={midY + 5} fill="black" fontSize="12" fontWeight="bold" textAnchor="middle">{visual.symbol}</text>
                            </g>
                        );
                    }
                })}
            </svg>
        </div>
    );
};