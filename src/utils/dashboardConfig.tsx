
import React from 'react';
import { 
    BrainCircuitIcon, GlobeIcon, BoxIcon, CodeBracketIcon, 
    NetworkIcon, CpuChipIcon, MessageSquareIcon, ServerCogIcon,
    SparklesIcon, GalaxyIcon, BanknotesIcon, ShieldCheckIcon, RocketLaunchIcon,
    AtomIcon
} from '../components/Icons';
import { LogEntry, SystemHealth } from '../types';

export interface PanelData {
    id: string;
    title: React.ReactNode;
    description: string;
    className?: string;
    minAdminLevel: number;
    content?: React.ReactNode;
}

export interface FaceData {
    layout: string;
    panels: PanelData[];
}

export type FaceMetadata = FaceData;

export const faceRotations = {
    0: { x: 0, y: 0 },       // Front
    1: { x: 0, y: -90 },     // Right
    2: { x: 0, y: 180 },     // Back
    3: { x: 0, y: 90 },      // Left
    4: { x: -90, y: 0 },     // Top
    5: { x: 90, y: 0 },      // Bottom
};

export const navigationTransitions: Record<number, Record<string, number>> = {
    0: { up: 5, down: 4, left: 1, right: 3 },
    1: { up: 5, down: 4, left: 2, right: 0 },
    2: { up: 5, down: 4, left: 3, right: 1 },
    3: { up: 5, down: 4, left: 0, right: 2 },
    4: { up: 0, down: 2, left: 1, right: 3 }, 
    5: { up: 2, down: 0, left: 1, right: 3 }, 
};

export const initialLogs: LogEntry[] = [
    { id: 1, time: '08:00:00', level: 'INFO', msg: 'System Boot Sequence Initiated.' },
    { id: 2, time: '08:00:01', level: 'SUCCESS', msg: 'Singularity State: LOCKED.' },
    { id: 3, time: '08:00:02', level: 'INFO', msg: 'Neural-Quantum Bridge Established.' },
];

export const initialSystemHealth: SystemHealth = {
    cognitiveEfficiency: 0.999,
    semanticIntegrity: 0.999,
    dataThroughput: 980,
    ipsThroughput: 850,
    powerEfficiency: 0.995,
    decoherenceFactor: 0.0001,
    processingSpeed: 9.9,
    qpuTempEfficiency: 1.0,
    qubitStability: 5000, 
    neuralLoad: 35,
    activeThreads: 128,
};

export const panelToFaceMap: { [key: string]: number } = {
    'agentq-core': 0, 'qcos-core-gateway': 0, 'agentq-self-evo': 0, 'quantum-reinforcement-learning': 0,
    'chips-quantum-network': 1,
    'chips-app-store': 2, 'grand-universe-simulator': 2,
    'chips-dev-platform': 3,
    'chips-back-office': 4, 'security-monitor': 4, 'chips-economy': 4,
    'qpu-health': 5, 'system-diagnostic': 5, 'quantum-large-language-model': 5, 'neural-programming': 5, 'quantum-machine-learning': 5, 'quantum-engineering-design': 5,
};

export const getPanelMetadata = (qcosVersion: number): Record<number, FaceData> => ({
  0: { // Front Face - Core / OS
    layout: 'grid grid-cols-4 grid-rows-3 gap-2',
    panels: [
      { id: 'agentq-core', title: <div className="flex items-center"><BrainCircuitIcon className="w-5 h-5 mr-2 text-cyan-400" />QIAI-IPS: Cognitive Architecture</div>, description: "Monitor Agent Q's cognitive core and QNN performance.", className: 'col-span-2 row-span-2', minAdminLevel: 1 },
      { id: 'qcos-core-gateway', title: <div className="flex items-center"><CpuChipIcon className="w-5 h-5 mr-2 text-red-500" />QCOS Core Gateway</div>, description: 'Direct neural link to Source Nexus and AGI Governance.', className: 'col-span-2 row-span-1', minAdminLevel: 3 },
      { id: 'agentq-self-evo', title: <div className="flex items-center"><SparklesIcon className="w-5 h-5 mr-2 text-purple-400" />QNN Evolution</div>, description: 'Visual pipeline of the QNN training process.', className: 'col-span-2 row-span-1', minAdminLevel: 2 },
      { id: 'quantum-reinforcement-learning', title: <div className="flex items-center"><RocketLaunchIcon className="w-5 h-5 mr-2 text-green-400" />QRL: Strategy Engine</div>, description: 'Variational Quantum Policy Gradient training loop.', className: 'col-span-4 row-span-1', minAdminLevel: 2 },
    ]
  },
  1: { // Right Face - Program 1: Chips Quantum Browser (CQB)
    layout: 'grid grid-cols-1', 
    panels: [
      { id: 'chips-quantum-network', title: <div className="flex items-center"><GlobeIcon className="w-5 h-5 mr-2 text-blue-400" />Chips Quantum Internet Network</div>, description: 'AI-Native browser for the CHIPS network and public web.', className: 'h-full', minAdminLevel: 1 },
    ]
  },
  2: { // Back Face - Program 2: Chips Quantum App Store, Plugins & Universe Sim
    layout: 'grid grid-cols-2 grid-rows-2 gap-2',
    panels: [
      { id: 'chips-app-store', title: <div className="flex items-center"><BoxIcon className="w-5 h-5 mr-2 text-cyan-400" />Chips Quantum App Store</div>, description: "Official application registry for the CHIPS Network.", className: 'col-span-2 row-span-1', minAdminLevel: 1 },
      { id: 'grand-universe-simulator', title: <div className="flex items-center"><GalaxyIcon className="w-5 h-5 mr-2 text-blue-300 animate-spin-slow" />Grand Universe Simulator</div>, description: "Predict future timelines via quantum parallelism.", className: 'col-span-2 row-span-1', minAdminLevel: 3 },
    ]
  },
  3: { // Left Face - Program 3: ChipsDev Platform (CDP)
    layout: 'grid grid-cols-1', 
    panels: [
      { id: 'chips-dev-platform', title: <div className="flex items-center"><CodeBracketIcon className="w-5 h-5 mr-2 text-yellow-400" />ChipsDev Platform (CQDP)</div>, description: 'Unified environment for community inspiration, planning, coding, and deployment.', className: 'h-full', minAdminLevel: 2 },
    ]
  },
  4: { // Top Face - Combined Back Office, Security & Economy
    layout: 'grid grid-cols-2 gap-2',
    panels: [
      { id: 'security-monitor', title: <div className="flex items-center"><ShieldCheckIcon className="w-6 h-6 mr-2 text-green-400" />Security Monitor & Simulator</div>, description: 'Real-time AI security oversight and defense protocol simulation.', className: 'h-full row-span-2', minAdminLevel: 3 },
      { id: 'chips-back-office', title: <div className="flex items-center"><NetworkIcon className="w-6 h-6 mr-2 text-red-400" />Chips Back Office</div>, description: 'Unified administration for CQB, CQAS, CDH, and CDP.', className: 'h-1/2', minAdminLevel: 2 },
      { id: 'chips-economy', title: <div className="flex items-center"><BanknotesIcon className="w-6 h-6 mr-2 text-green-400" />Chips Economy</div>, description: 'Control center for Digital Currency, Wallets, and Exchanges.', className: 'h-1/2', minAdminLevel: 2 }
    ]
  },
  5: { // Bottom Face - Vitals (Hardware & Neural)
    layout: 'grid grid-cols-4 grid-rows-2 gap-2', 
    panels: [
        { id: 'qpu-health', title: <div className="flex items-center"><CpuChipIcon className="w-5 h-5 mr-2 text-green-400" />Quantum Hardware Vitals</div>, description: 'Real-time QPU Health metrics.', className: 'col-span-2', minAdminLevel: 1 },
        { id: 'system-diagnostic', title: <div className="flex items-center"><ServerCogIcon className="w-5 h-5 mr-2 text-yellow-400" />System Diagnostic</div>, description: 'Run a full system diagnostic.', className: 'col-span-2', minAdminLevel: 2 },
        { id: 'neural-programming', title: <div className="flex items-center"><BrainCircuitIcon className="w-5 h-5 mr-2 text-purple-400 animate-pulse" />Neural Programming</div>, description: 'BCI Interface.', minAdminLevel: 2 },
        { id: 'quantum-large-language-model', title: <div className="flex items-center"><CodeBracketIcon className="w-5 h-5 mr-2 text-pink-400" />QLLM Engine</div>, description: 'Quantum Language Model.', minAdminLevel: 2 },
        { id: 'quantum-machine-learning', title: <div className="flex items-center"><BrainCircuitIcon className="w-5 h-5 mr-2 text-cyan-400" />QML Engine</div>, description: 'QNN training.', minAdminLevel: 2 },
        { id: 'quantum-engineering-design', title: <div className="flex items-center"><AtomIcon className="w-5 h-5 mr-2 text-blue-400" />Quantum Engineering Design</div>, description: 'NLP-to-CAD Physics Pipeline.', minAdminLevel: 2 },
    ]
  }
});
