import React, { useState, useCallback } from 'react';
import { BrainCircuitIcon, PlayIcon, ZapIcon, WindIcon } from './Icons';

// --- Mock Simulation Logic based on Python/C++ stubs ---

const LEARNING_MODES = ["ONION_PEELING", "DIALECTICS", "HISTORICAL", "PREDICTIVE"];

const calculateSiplMetrics = (complexity: number, energy: number) => {
    return { context: complexity, energy: energy };
};

const runIpsnnQnnInference = (features: { context: number, energy: number }) => {
    if (features.context > 0.8 && features.energy < 0.3) {
        return 0.1176;
    }
    return 0.0000;
};

const determinePolicy = (context: number, energy: number, v_score: number) => {
    if (energy > 0.60) return { policy: 0, reason: 'REFLEX VETO: Energy too high' };
    if (v_score > 0.10) return { policy: 1, reason: 'POLICY 1 (ACT)' };
    return { policy: 2, reason: 'POLICY 2 (GAMBLE)' };
};

const selectLearningMode = (probability_output: number) => {
    const mode_index = Math.min(Math.floor(probability_output * LEARNING_MODES.length), LEARNING_MODES.length - 1);
    return LEARNING_MODES[mode_index];
};

// --- Component --- 

const QcosDashboard: React.FC = () => {
    const [simulationLog, setSimulationLog] = useState<string[]>([]);
    const [isRunning, setIsRunning] = useState(false);

    const runSimulation = useCallback(() => {
        setIsRunning(true);
        setSimulationLog(['[INIT] Q-IAI Simulation Sequence Started...']);

        const complexity = 0.9;
        const energy = 0.2;
        let log: string[] = [];

        setTimeout(() => {
            log = [...log, `[INPUT] Simulated Task: Complexity=${complexity}, Energy=${energy}`];
            setSimulationLog(prev => [...prev, ...log.slice(prev.length -1)]);

            const qill_prob = Math.random();
            const learningMode = selectLearningMode(qill_prob);
            log = [...log, `   [Q-ILL] Mode Classified as: ${learningMode} (Confidence: ${qill_prob.toFixed(4)})`];
            setSimulationLog(prev => [...prev, ...log.slice(prev.length-1)]);
        }, 500);

        setTimeout(() => {
            const { context, energy: metricsEnergy } = calculateSiplMetrics(complexity, energy);
            const finalEnergy = metricsEnergy;
            log = [...log, `   [CLNN] Governance Decided: DEEPER_LEARNING structure`];
            setSimulationLog(prev => [...prev, ...log.slice(prev.length-1)]);

            const v_score = runIpsnnQnnInference({ context, energy: finalEnergy });
            log = [...log, `   [IPSNN] V-Score Generated: ${v_score.toFixed(4)}`];
            setSimulationLog(prev => [...prev, ...log.slice(prev.length-1)]);

            const { policy, reason } = determinePolicy(context, finalEnergy, v_score);
            log = [...log, `[OUTPUT] Final Policy: ${policy} - ${reason}`];
            setSimulationLog(prev => [...prev, ...log.slice(prev.length-1)]);

            setIsRunning(false);
        }, 1500);

    }, []);

    return (
        <div className="h-full flex flex-col p-4 bg-black/30 rounded-lg border border-cyan-500/20 text-cyan-100 font-mono">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold flex items-center"><BrainCircuitIcon className="w-5 h-5 mr-2" /> Q-IAI Simulation Panel</h3>
                <button 
                    onClick={runSimulation}
                    disabled={isRunning}
                    className="flex items-center px-4 py-2 bg-cyan-500/20 border border-cyan-500 rounded-md hover:bg-cyan-500/40 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300"
                >
                    {isRunning ? <WindIcon className="w-5 h-5 mr-2 animate-spin" /> : <PlayIcon className="w-5 h-5 mr-2" />}
                    {isRunning ? 'Running...' : 'Run Simulation'}
                </button>
            </div>
            <div className="flex-grow bg-black/50 p-4 rounded-md overflow-y-auto custom-scrollbar">
                {simulationLog.map((line, index) => (
                    <div key={index} className="text-sm mb-1 whitespace-pre-wrap">{`> ${line}`}</div>
                ))}
            </div>
        </div>
    );
};

export default QcosDashboard;
