import React, { createContext, useContext, useState, useCallback } from 'react';
import { useToast } from './ToastContext';

// --- Comprehensive Types ---
export interface SystemStatus {
    currentTask: string;
    neuralLoad: number;
    efficiencyBoost: number;
    isRepairing: boolean;
    isOptimized: boolean;
    status: 'online' | 'offline' | 'recalibrating';
}

interface TrainingState {
    isActive: boolean;
    epoch: number;
    loss: number;
    coherence: number;
    currentStage: number; // Fixes DistributedCognitiveArchitecture crash
    logs: string[];
}

interface SimulationContextType {
    systemStatus: SystemStatus;
    training: TrainingState;
    evolution: { isActive: boolean; logs: string[]; currentStage: number };
    qllm: { efficiencyBoost: number; status: string; isActive: boolean };
    entanglementMesh: { isUniverseLinkedToQLang: boolean };
    dataIngestion: any[];
    setSystemTask: (task: string, isRepairing?: boolean) => void;
    startAllSimulations: () => void;
}

const SimulationContext = createContext<SimulationContextType | undefined>(undefined);

export const SimulationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const { addToast } = useToast();

    const [systemStatus] = useState<SystemStatus>({
        currentTask: 'System Idle',
        neuralLoad: 12,
        efficiencyBoost: 1.8,
        isRepairing: false,
        isOptimized: false,
        status: 'online'
    });

    // We initialize EVERYTHING with defaults to prevent '.map' or '.status' crashes
    const [training] = useState<TrainingState>({
        isActive: false,
        epoch: 0,
        loss: 0.85,
        coherence: 65,
        currentStage: 0,
        logs: []
    });

    const value: SimulationContextType = {
        systemStatus,
        training,
        evolution: { isActive: false, logs: [], currentStage: 0 },
        qllm: { efficiencyBoost: 1.5, status: 'synced', isActive: true }, // Fixes AgentQ crash
        entanglementMesh: { isUniverseLinkedToQLang: true },
        dataIngestion: [],
        setSystemTask: (task) => console.log(`Task set: ${task}`),
        startAllSimulations: () => addToast("Quantum Mesh Synchronized", "success"),
    };

    return (
        <SimulationContext.Provider value={value}>
            {children}
        </SimulationContext.Provider>
    );
};

export const useSimulation = () => {
    const context = useContext(SimulationContext);
    if (!context) {
        // Fallback object to prevent crashes if used outside provider during HMR
        return {
            systemStatus: { status: 'offline' },
            training: { currentStage: 0, logs: [] },
            qllm: { status: 'offline' },
            evolution: { logs: [] }
        } as any;
    }
    return context;
};