
import React, { useState } from 'react';

interface QuantumMemoryDataMatrixProps {
    label: string;
    colorBase: string;
    rows: number;
    cols: number;
    className?: string;
}

const QuantumMemoryDataMatrix: React.FC<QuantumMemoryDataMatrixProps> = ({ label, colorBase, rows, cols, className = "" }) => {
    const [opacities] = useState(() => Array.from({ length: rows * cols }).map(() => Math.random() * 0.8 + 0.2));
    const getColorClass = (base: string) => {
        switch (base) {
            case 'green': return 'bg-green-500';
            case 'blue': return 'bg-blue-500';
            case 'red': return 'bg-red-500';
            case 'purple': return 'bg-purple-500';
            default: return 'bg-cyan-500';
        }
    };

    return (
        <div className={`flex flex-col gap-1 ${className}`}>
            <span className="text-[8px] uppercase font-mono text-gray-500">{label}</span>
            <div 
                className="grid gap-0.5" 
                style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}
            >
                {Array.from({ length: rows * cols }).map((_, i) => (
                    <div 
                        key={i} 
                        className={`aspect-square rounded-[1px] ${getColorClass(colorBase)}`} 
                        style={{ opacity: opacities[i] }}
                    />
                ))}
            </div>
        </div>
    );
};

export default QuantumMemoryDataMatrix;
