
import React, { useState, useEffect, useRef } from 'react';
import { BugAntIcon, LoaderIcon, CheckCircle2Icon, AlertTriangleIcon, SparklesIcon, ServerCogIcon, GitBranchIcon, ShieldCheckIcon, LinkIcon, FileCodeIcon, CodeBracketIcon, CpuChipIcon, BrainCircuitIcon } from './Icons';
import { useSimulation } from '../context/SimulationContext';

type StageStatus = 'pending' | 'running' | 'success' | 'warning' | 'fixing' | 'fixed';

interface Stage {
  name: string;
  icon: React.FC<{className?: string}>;
  status: StageStatus;
  details?: string;
}

interface Phase {
    name: string;
    icon: React.FC<{className?: string}>;
    stages: Stage[];
}

interface SystemDiagnosticProps {
    onClose?: () => void;
    onMaximize?: () => void;
}

const initialPhases: Phase[] = [
    {
        name: 'Hardware Integrity',
        icon: CpuChipIcon,
        stages: [
            { name: 'QPU Core Temperature Scan', icon: ServerCogIcon, status: 'pending' },
            { name: 'Qubit Stability & Coherence Check', icon: GitBranchIcon, status: 'pending' },
            { name: 'Cryo-Cooling System Verification', icon: ServerCogIcon, status: 'pending' },
        ],
    },
    {
        name: 'Software & Network',
        icon: LinkIcon,
        stages: [
            { name: 'Quantum-to-Web Gateway Latency Test', icon: LinkIcon, status: 'pending' },
            { name: 'CHIPS Packet Routing Verification', icon: FileCodeIcon, status: 'pending' },
            { name: 'Core Kernel Integrity Scan', icon: CodeBracketIcon, status: 'pending' },
        ],
    },
    {
        name: 'Cognitive Core',
        icon: BrainCircuitIcon,
        stages: [
            { name: 'QNN Synapse Integrity Check', icon: GitBranchIcon, status: 'pending' },
            { name: 'Semantic Layer Drift Analysis', icon: ShieldCheckIcon, status: 'pending' },
            { name: 'Instinctive Problem Solving (IPS) Latency', icon: SparklesIcon || BrainCircuitIcon, status: 'pending' },
        ],
    },
];


const SystemDiagnostic: React.FC<SystemDiagnosticProps> = ({ onClose, onMaximize }) => {
    const { setSystemTask } = useSimulation();
    
    // Helper to deeply reset phases without losing Icon component references
    const getResetPhases = () => initialPhases.map(p => ({
        ...p,
        stages: p.stages.map(s => ({
            ...s,
            status: 'pending' as StageStatus,
            details: undefined
        }))
    }));

    const [phases, setPhases] = useState<Phase[]>(getResetPhases());
    const [overallStatus, setOverallStatus] = useState<'idle' | 'running' | 'fixing' | 'complete'>('idle');
    const [activeLog, setActiveLog] = useState<string[]>([]);
    const [report, setReport] = useState<{ summary: string; findings: string[]; actions: string[]; } | null>(null);
    const timeoutRefs = useRef<ReturnType<typeof setTimeout>[]>([]);
    const logRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        return () => timeoutRefs.current.forEach(clearTimeout);
    }, []);

    useEffect(() => {
        if (logRef.current) {
            logRef.current.scrollTop = logRef.current.scrollHeight;
        }
    }, [activeLog]);

    const addLog = (message: string) => {
        setActiveLog(prev => [...prev.slice(-20), `${new Date().toLocaleTimeString()} :: ${message}`]);
    };

    const startDiagnostic = () => {
        setOverallStatus('running');
        setSystemTask("Running Diagnostic: Hardware Integrity Scan", false); // Inform global context
        
        // Reset phases properly
        const freshPhases = getResetPhases();
        setPhases(freshPhases);
        
        setActiveLog(['System diagnostic sequence initiated...']);
        setReport(null);
        timeoutRefs.current.forEach(clearTimeout);
        timeoutRefs.current = [];

        let phaseIndex = 0;
        let stageIndex = 0;

        const runNext = () => {
            if (phaseIndex >= freshPhases.length) {
                // Check current state phases for warnings
                // We use a functional update here to ensure we check the latest state if needed, 
                // but for the logic flow, we know we just ran through them.
                // However, the `phases` variable in this closure is the `freshPhases` we started with, 
                // but we need to know if any *updates* caused warnings. 
                // Actually, the `freshPhases` variable isn't updated by setPhases. 
                // We need to track if we found anomalies.
                
                // For simplicity in this simulation, we hardcode the anomaly detection logic 
                // to match the specific step where we *set* the warning.
                
                // Re-evaluate based on the hardcoded logic we injected:
                // Anomaly occurs at phase 1, stage 0.
                const hasAnomaly = true; // In this deterministic sim, it always triggers.

                if (hasAnomaly) {
                    setOverallStatus('fixing');
                    setSystemTask("CRITICAL: Applying QNN Auto-Repair Patches", true); // Trigger high load mode
                    addLog('Anomalies detected. Initiating AI-driven repair sequence...');
                    const fixTimeout = setTimeout(() => {
                        setPhases(currentPhases => {
                            const fixedPhases = [...currentPhases];
                            // Deep copy not strictly needed for this specific update pattern but safer
                            const targetPhase = { ...fixedPhases[1] };
                            targetPhase.stages = [...targetPhase.stages];
                            const targetStage = { ...targetPhase.stages[0] };
                            
                            targetStage.status = 'fixed';
                            targetStage.details = 'Gateway rerouted through redundant node. Fixed.';
                            
                            targetPhase.stages[0] = targetStage;
                            fixedPhases[1] = targetPhase;
                            return fixedPhases;
                        });
                        addLog('QNN has applied corrective action to the Gateway.');
                        const completeTimeout = setTimeout(() => {
                            setOverallStatus('complete');
                            setSystemTask("System Diagnostic Complete. Nominal.", false); // Reset
                            setReport({
                                summary: 'System Health: Optimal. 1 minor anomaly detected and resolved by Agent Q.',
                                findings: ['High latency detected on primary Quantum-to-Web Gateway node.'],
                                actions: ['Agent Q rerouted traffic through a redundant, low-latency node.', 'Flagged primary node for maintenance.']
                            });
                        }, 2000);
                        timeoutRefs.current.push(completeTimeout);
                    }, 2000);
                    timeoutRefs.current.push(fixTimeout);
                } else {
                    setOverallStatus('complete');
                    setSystemTask("System Diagnostic Complete. Nominal.", false); // Reset
                    setReport({ summary: 'System Health: Optimal. No anomalies detected.', findings: [], actions: [] });
                }
                return;
            }
            
            const currentPhase = freshPhases[phaseIndex];
            if (!currentPhase) return; // Safety check
            const currentStage = currentPhase.stages[stageIndex];
            if (!currentStage) return; // Safety check

            // Update global task description
            setSystemTask(`Diagnostic: ${currentPhase.name} - ${currentStage.name}`, false);

            setPhases(prev =>
                prev.map((phase, pIdx) =>
                    pIdx === phaseIndex
                        ? {
                              ...phase,
                              stages: phase.stages.map((stage, sIdx) =>
                                  sIdx === stageIndex ? { ...stage, status: 'running' } : stage
                              ),
                          }
                        : phase
                )
            );
            addLog(`Running: ${currentStage.name}...`);

            const stageDuration = 500 + Math.random() * 300;
            const timeout = setTimeout(() => {
                const isAnomaly = phaseIndex === 1 && stageIndex === 0;
                const newStatus: StageStatus = isAnomaly ? 'warning' : 'success';
                const newDetails = isAnomaly ? 'High latency detected (>150ms).' : undefined;

                setPhases(prev =>
                    prev.map((phase, pIdx) =>
                        pIdx === phaseIndex
                            ? {
                                  ...phase,
                                  stages: phase.stages.map((stage, sIdx) =>
                                      sIdx === stageIndex
                                          ? { ...stage, status: newStatus, details: newDetails }
                                          : stage
                                  ),
                              }
                            : phase
                    )
                );

                if(newStatus === 'warning') addLog(`Warning: ${newDetails}`);

                stageIndex++;
                if (stageIndex >= freshPhases[phaseIndex].stages.length) {
                    phaseIndex++;
                    stageIndex = 0;
                }
                runNext();
            }, stageDuration);
            timeoutRefs.current.push(timeout);
        };
        
        runNext();
    };
    
    const getStatusIndicator = (status: StageStatus) => {
        switch (status) {
            case 'running': return <LoaderIcon className="w-4 h-4 text-cyan-300 animate-spin" />;
            case 'fixing': return <LoaderIcon className="w-4 h-4 text-yellow-300 animate-spin" />;
            case 'success': return <CheckCircle2Icon className="w-4 h-4 text-green-400" />;
            case 'fixed': return <CheckCircle2Icon className="w-4 h-4 text-green-400" />;
            case 'warning': return <AlertTriangleIcon className="w-4 h-4 text-yellow-400" />;
            case 'pending': default: return <div className="w-3 h-3 rounded-full bg-slate-700 border border-slate-500" />;
        }
    };

    if (overallStatus === 'idle') {
        return (
            <div className="text-center h-full flex flex-col items-center justify-center p-4">
                <BugAntIcon className="w-16 h-16 text-cyan-500 mx-auto mb-4" />
                <h3 className="text-xl font-bold text-white">System Diagnostic & Repair</h3>
                <p className="text-cyan-300 my-2">Agent Q will perform an advanced scan of all QCOS subsystems to detect and resolve anomalies using its QNN core.</p>
                <div className="flex items-center gap-4 mt-4">
                    {onClose && (
                         <button
                          onClick={onClose}
                          title="Return to the main dashboard."
                          className="holographic-button px-6 py-2 bg-slate-500/30 border border-slate-500/50 text-slate-200 font-bold rounded-md"
                        >
                            Cancel
                        </button>
                    )}
                    {onMaximize && (
                         <button
                          onClick={onMaximize}
                          title="Open diagnostic in full view."
                          className="holographic-button px-6 py-2 bg-blue-500/30 border border-blue-500/50 text-blue-200 font-bold rounded-md"
                        >
                            Full View
                        </button>
                    )}
                    <button
                      onClick={startDiagnostic}
                      title="Begin a comprehensive scan of all QCOS subsystems."
                      className="holographic-button px-6 py-2 bg-cyan-500/30 border border-cyan-500/50 text-cyan-200 font-bold rounded-md"
                    >
                        Start System Diagnostic
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="w-full h-full max-w-4xl mx-auto flex flex-col gap-4 animate-fade-in p-2">
            <h3 className="text-xl font-bold text-white text-center flex-shrink-0">
                {overallStatus === 'running' && 'System Diagnostic in Progress...'}
                {overallStatus === 'fixing' && 'Anomaly Detected - Applying AI Fix...'}
                {overallStatus === 'complete' && 'Diagnostic Complete'}
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 flex-shrink-0">
                {phases.map(phase => (
                    <div key={phase.name} className="bg-black/20 p-3 rounded-lg border border-cyan-900/50">
                        <h4 className="flex items-center text-sm font-semibold text-cyan-200 mb-2 pb-2 border-b border-cyan-800/50">
                            {phase.icon && <phase.icon className="w-4 h-4 mr-2" />} {phase.name}
                        </h4>
                        <div className="space-y-2">
                            {phase.stages.map(stage => (
                                <div key={stage.name} className="flex items-center justify-between text-xs">
                                    <span className={stage.status !== 'pending' ? 'text-white' : 'text-cyan-700'}>{stage.name}</span>
                                    {getStatusIndicator(stage.status)}
                                </div>
                            ))}
                        </div>
                    </div>
                ))}
            </div>
            <div className="flex-grow flex flex-col min-h-0">
                <h4 className="text-sm font-semibold text-cyan-200 mb-1">Live Activity Log</h4>
                <div ref={logRef} className="flex-grow bg-black/50 p-2 rounded-md border border-cyan-900 text-xs font-mono overflow-y-auto">
                    {activeLog.map((log, i) => <p key={i}>{log}</p>)}
                </div>
            </div>
            {overallStatus === 'complete' && report && (
                <div className="mt-2 p-3 bg-green-900/50 border border-green-700 rounded-lg text-sm animate-fade-in flex-shrink-0">
                    <p className="font-bold text-green-300 mb-2">{report.summary}</p>
                    {report.findings.length > 0 && (
                        <div>
                            <p className="font-semibold text-white">Findings:</p>
                            <ul className="list-disc list-inside text-yellow-300 text-xs">
                                {report.findings.map((f, i) => <li key={i}>{f}</li>)}
                            </ul>
                        </div>
                    )}
                     {report.actions.length > 0 && (
                        <div className="mt-2">
                            <p className="font-semibold text-white">Actions Taken:</p>
                            <ul className="list-disc list-inside text-green-400 text-xs">
                                {report.actions.map((a, i) => <li key={i}>{a}</li>)}
                            </ul>
                        </div>
                    )}
                    <div className="flex items-center justify-center gap-4 mt-3">
                        <button onClick={startDiagnostic} title="Start a new diagnostic scan." className="holographic-button text-xs px-4 py-1 rounded-md">
                            Run Again
                        </button>
                        {onClose && (
                            <button onClick={onClose} title="Close the diagnostic interface." className="holographic-button text-xs px-4 py-1 rounded-md bg-slate-500/30 border-slate-500/50 text-slate-200">
                                Close
                            </button>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default SystemDiagnostic;
