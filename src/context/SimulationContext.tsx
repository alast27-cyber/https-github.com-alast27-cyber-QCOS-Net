
import React, { createContext, useContext, useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useToast } from './ToastContext';

// --- Types ---

export interface RoadmapStage {
    id: string;
    title: string;
    description: string;
    progress: number; // 0 to 100
    status: 'pending' | 'active' | 'completed';
    tasks: string[];
}

export interface TrainingLog {
    timestamp: number;
    message: string;
    type: 'info' | 'success' | 'warning' | 'patch';
}

export type TrainingDomain = 'PHYSICS' | 'MATH' | 'ECONOMICS' | 'NEUROSCIENCE' | 'STRATEGY';
export type MathSubDomain = 'TOPOLOGY' | 'NUMBER_THEORY' | 'CALCULUS' | 'ALGEBRA' | 'NONE';
export type EcoSubDomain = 'MARKET_DYNAMICS' | 'GAME_THEORY' | 'RESOURCE_ALLOC' | 'NONE';

interface TrainingState {
    isActive: boolean;
    isAutomated: boolean;
    epoch: number;
    loss: number;
    coherence: number;
    activeStage: 0 | 1 | 2 | 3;
    logs: string[];
    generatedPatch: string | null;
    domain: TrainingDomain;
    subDomain: string;
    synthesizedEquation: string | null; 
}

interface RoadmapState {
    stages: RoadmapStage[];
    isTraining: boolean;
    logs: TrainingLog[];
    currentTask: string;
}

interface EvolutionState {
    isActive: boolean;
    dataPoints: { time: number; cognitiveEfficiency: number; semanticIntegrity: number }[];
    logs: string[];
    timeStep: number;
}

interface SystemStatus {
    currentTask: string;
    neuralLoad: number; // 0-100%
    instinctsCataloged: number;
    activeThreads: number;
    isRepairing: boolean;
    ipsThroughput: number; 
    isOptimized: boolean; 
}

interface InquiryState {
    id: string;
    prompt: string;
    status: 'idle' | 'queued' | 'simulating' | 'optimizing' | 'complete' | 'error';
    result: string | null;
    sourceNode?: string;
    targetNode?: string;
    error?: string;
}

interface EntanglementMesh {
    active: boolean;
    universeNode: number; 
    neuralNode: number;   
    qmlNode: number;      
    isUniverseLinkedToQLang: boolean;
    isQRLtoQNNLinked: boolean;
    isQRLtoUniverseLinked: boolean;
    isQRLtoGatewayLinked: boolean;
    linkFidelity: number;
}

interface NeuralInterfaceState {
    isActive: boolean;
    connectionType: 'EEG' | 'fNIRS' | 'INVASIVE_LACE' | 'RF_WIFI' | 'BLUETOOTH_6' | 'QUANTUM_ENTANGLEMENT';
    coherence: number; 
    pairCognitionActive: boolean;
    lastIntent: string | null;
    singularityAlignment: number; 
}

interface QLLMState {
    isActive: boolean;
    isTraining: boolean;
    isAutoTopology: boolean; 
    loss: number;
    efficiencyBoost: number; 
    contextWindow: number;
    embeddingDimension: number;
    autoTraining: {
        isActive: boolean;
        configIndex: number;
        bestLoss: number;
        iteration: number;
    };
}

interface QMLEngineState {
    status: 'IDLE' | 'TRAINING' | 'CONVERGED' | 'FAILED';
    modelType: 'QNN' | 'BOLTZMANN' | 'GAN' | 'TRANSFORMER' | 'HYBRID-QNN' | 'DEEP-BOLTZMANN' | 'ADVERSARIAL-Q' | 'ATTENTION-Q';
    progress: number; 
    currentEpoch: number;
    totalEpochs: number;
    accuracy: number;
    loss: number;
    hyperparameters: {
        learningRate: number;
        circuitDepth: number;
        qubitCount: number;
        entanglementMap: 'LINEAR' | 'FULL' | 'CIRCULAR';
    };
    logs: string[];
    autoEvolution: {
        isActive: boolean;
        currentStage: number;
        totalStages: number;
        loopCount: number; 
        bestModel: string | null;
        bestAccuracy: number;
    };
    modelLibrary: Array<{
        id: string;
        type: string;
        accuracy: number;
        qubits: number;
        depth: number;
        timestamp: string;
    }>;
}

interface QRLEngineState {
    status: 'IDLE' | 'TRAINING' | 'OPTIMIZED' | 'INTEGRATING';
    currentEpisode: number;
    totalEpisodes: number;
    reward: number;
    avgReward: number;
    epsilon: number;
    cumulativeRewards: { episode: number, value: number }[];
    agentPosition: { x: number, y: number };
    goalPosition: { x: number, y: number };
    policyDistribution: number[]; 
    predictedPaths: { x: number, y: number }[][];
    logs: string[];
}

interface QDLEngineState {
    status: 'IDLE' | 'TRAINING' | 'CONVERGED';
    qubitCount: number;
    layers: Array<{ type: 'ENCODING' | 'QCNN' | 'POOLING' | 'DENSE', coherence: number }>;
    loss: number;
    accuracy: number;
    gradientSignal: number; 
    trainingStrategy: 'LAYER-WISE' | 'GLOBAL';
    logs: string[];
}

// New: Quantum Generative Learning
interface QGLEngineState {
    status: 'IDLE' | 'GENERATING' | 'EVOLVING';
    creativityIndex: number; // 0-1
    coherence: number;
    currentGeneration: number;
}

// New: Quantum Cognition Engine Global State
interface QCEState {
    evolutionProgress: {
        QLLM: number;
        QML: number;
        QRL: number;
        QGL: number;
        QDL: number;
    };
    currentStage: {
        QLLM: number;
        QML: number;
        QRL: number;
        QGL: number;
        QDL: number;
    };
    isEntangled: boolean;
}

interface QIAIIPSState {
    qil: { coherence: number; status: 'INGESTING' | 'IDLE'; load: number };
    qips: { coherence: number; status: 'SOLVING' | 'IDLE'; load: number };
    qcl: { coherence: number; status: 'GOVERNING' | 'IDLE'; load: number };
    globalSync: number;
}

interface TelemetryFeed {
    name: string;
    value: number;
    trend: 'rising' | 'falling' | 'stable';
    latency: number;
    unit: string;
}

// New: Universe Bridge Connections
interface UniverseConnections {
    kernel: boolean;
    agentQ: boolean;
}

// New: Data Ingestion Source
export interface DataSource {
    id: string;
    name: string;
    type: 'STREAM' | 'BATCH' | 'QUANTUM_LINK';
    status: 'ACTIVE' | 'IDLE' | 'SYNCING' | 'ERROR';
    throughput: number; // PB/s
    fidelity: number; // %
    latency: number; // ms
    isEntangled: boolean;
}

export type SimulationMode = 'PHYSICS' | 'ECONOMIC';

export interface SimulationConfig {
    mode: SimulationMode;
    activePreset: string | null;
    activeInjection: string | null; 
    isAutoTuning: boolean;
    isUniverseActive: boolean; 
}

interface SimulationContextType {
    training: TrainingState;
    evolution: EvolutionState;
    systemStatus: SystemStatus;
    inquiry: InquiryState;
    entanglementMesh: EntanglementMesh;
    neuralInterface: NeuralInterfaceState;
    qllm: QLLMState;
    qmlEngine: QMLEngineState; 
    qrlEngine: QRLEngineState;
    qdlEngine: QDLEngineState;
    qglEngine: QGLEngineState; 
    qceState: QCEState; 
    qiaiIps: QIAIIPSState;
    dataIngestion: DataSource[]; // New
    universeConnections: UniverseConnections; 
    telemetryFeeds: TelemetryFeed[];
    simConfig: SimulationConfig;
    singularityBoost: number; 
    roadmapState: RoadmapState; // New
    toggleTraining: () => void;
    toggleAutomation: () => void;
    startToESimulation: () => void; 
    startNeuroSimulation: (subDomain?: string) => void; 
    startMathSimulation: (subDomain: string) => void;
    startEcoSimulation: (subDomain: string) => void;
    toggleEvolution: () => void;
    startAllSimulations: () => void; 
    setTrainingPatch: (patch: string | null) => void;
    setSystemTask: (task: string, isRepairing?: boolean) => void;
    submitInquiry: (prompt: string, source?: string, target?: string) => string; 
    updateInquiry: (id: string, updates: Partial<InquiryState>) => void;
    setSimMode: (mode: SimulationMode) => void;
    setSimPreset: (preset: string) => void;
    injectApp: (appId: string) => void;
    runAutoTune: () => void;
    connectNeuralInterface: (type: NeuralInterfaceState['connectionType']) => void;
    disconnectNeuralInterface: () => void;
    togglePairCognition: () => void;
    injectNeuralIntent: (intent: string) => void;
    toggleQLLM: (isActive: boolean) => void;
    setQLLMTraining: (isTraining: boolean) => void;
    toggleQLLMAutoTraining: () => void; 
    updateQLLMConfig: (updates: Partial<QLLMState>) => void;
    optimizeLocalNode: () => void;
    startQMLTraining: (params: Partial<QMLEngineState['hyperparameters']>, model: QMLEngineState['modelType']) => void;
    stopQMLTraining: () => void;
    integrateQMLModel: () => void;
    toggleAutoEvolution: () => void;
    applySystemOptimization: () => void;
    startQRLTraining: (params?: Partial<QRLEngineState>) => void;
    stopQRLTraining: () => void;
    deployQRLToQNN: () => void;
    startQDLTraining: (params?: Partial<QDLEngineState>) => void;
    stopQDLTraining: () => void;
    setQDLStrategy: (strategy: QDLEngineState['trainingStrategy']) => void;
    addQDLLayer: (type: QDLEngineState['layers'][0]['type']) => void;
    toggleUniverseQLangLink: (active: boolean) => void;
    toggleQRLtoQNNEntanglement: (active: boolean) => void; 
    toggleQRLtoUniverseLink: (active: boolean) => void;
    toggleQRLtoGatewayLink: (active: boolean) => void;
    toggleQLLMAutoTopology: () => void;
    updateQIAIIPS: (updates: Partial<QIAIIPSState>) => void;
    toggleUniverseToKernel: (active: boolean) => void; 
    toggleUniverseToAgentQ: (active: boolean) => void;
    toggleSourceEntanglement: (id: string | 'ALL') => void; // Updated for bulk action
    toggleRoadmapTraining: () => void; // New
    resetRoadmap: () => void; // New
}

const SimulationContext = createContext<SimulationContextType | undefined>(undefined);

const EVOLUTION_STAGES = [
    { model: 'QNN' as const, params: { learningRate: 0.05, circuitDepth: 4, qubitCount: 128, entanglementMap: 'LINEAR' as const } },
    { model: 'BOLTZMANN' as const, params: { learningRate: 0.02, circuitDepth: 8, qubitCount: 180, entanglementMap: 'CIRCULAR' as const } },
    { model: 'GAN' as const, params: { learningRate: 0.01, circuitDepth: 12, qubitCount: 200, entanglementMap: 'FULL' as const } },
    { model: 'TRANSFORMER' as const, params: { learningRate: 0.005, circuitDepth: 16, qubitCount: 240, entanglementMap: 'FULL' as const } }
];

// Expanded Data Sources with "Always Entangled" philosophy
const INITIAL_DATA_SOURCES: DataSource[] = [
    { id: 'ds-01', name: 'Global Quantum Material Lattice', type: 'QUANTUM_LINK', status: 'ACTIVE', throughput: 145.2, fidelity: 99.9, latency: 2, isEntangled: true },
    { id: 'ds-02', name: 'Urban-Scale Entanglement Dist. (Berlin)', type: 'STREAM', status: 'ACTIVE', throughput: 890.4, fidelity: 99.95, latency: 1, isEntangled: true },
    { id: 'ds-03', name: 'Photonic Quantum Circuits (Waveguide QED)', type: 'STREAM', status: 'ACTIVE', throughput: 1630.0, fidelity: 99.99, latency: 45, isEntangled: true },
    { id: 'ds-04', name: 'Smart Grid & Energy Infrastructure', type: 'STREAM', status: 'ACTIVE', throughput: 450.0, fidelity: 99.8, latency: 12, isEntangled: true },
    { id: 'ds-05', name: 'Manufacturing (Predictive Maintenance)', type: 'STREAM', status: 'ACTIVE', throughput: 5600.0, fidelity: 99.99, latency: 1, isEntangled: true },
    { id: 'ds-06', name: 'Aerospace (Turbine Performance)', type: 'STREAM', status: 'ACTIVE', throughput: 320.0, fidelity: 98.5, latency: 50, isEntangled: true },
    { id: 'ds-07', name: 'Heavy Industry (Multi-Robot Fleet)', type: 'STREAM', status: 'ACTIVE', throughput: 120.5, fidelity: 99.2, latency: 30, isEntangled: true },
    { id: 'ds-08', name: 'Neural Net Training Cluster (Ext)', type: 'QUANTUM_LINK', status: 'ACTIVE', throughput: 890.0, fidelity: 99.4, latency: 5, isEntangled: true },
    { id: 'ds-09', name: 'NOAA Oceanographic Buoy Array', type: 'BATCH', status: 'ACTIVE', throughput: 65.0, fidelity: 99.1, latency: 200, isEntangled: true },
    { id: 'ds-10', name: 'Ethereum Mempool (Pending Tx)', type: 'STREAM', status: 'ACTIVE', throughput: 410.0, fidelity: 99.9, latency: 15, isEntangled: true },
    { id: 'ds-11', name: 'Square Kilometre Array (Radio)', type: 'STREAM', status: 'ACTIVE', throughput: 6200.0, fidelity: 99.98, latency: 4, isEntangled: true },
];

const INITIAL_ROADMAP_STAGES: RoadmapStage[] = [
    {
        id: 'phase-1',
        title: 'Phase 1: Core Architectural Progress (GME)',
        description: 'Mixture-of-Experts (MoE) implementation, Sparse Activation, and Expert Specialization.',
        progress: 100,
        status: 'completed',
        tasks: ['Model Scaling & Efficiency', 'Expert Specialization (S\'MoRE)', 'Multimodal Integration']
    },
    {
        id: 'phase-2',
        title: 'Phase 2: Multi-Domain Generalization',
        description: 'Scientific Reasoning, Life Sciences, and Ethical Alignment.',
        progress: 45,
        status: 'active',
        tasks: ['2.3: Scientific Reasoning (Causal Modeling)', '2.4: Life Sciences (GNNs, ABM)', '2.5: Philosophy & Alignment (Ethical Guardrails)']
    },
    {
        id: 'phase-3',
        title: 'Phase 3: Generalization, Autonomy, Refinement',
        description: 'Cross-Domain Stress Testing, Self-Improvement Loop, and Final Certification.',
        progress: 0,
        status: 'pending',
        tasks: ['3.1: Cross-Domain Stress Testing', '3.2: Self-Improvement Loop', '3.3: Final Certification']
    },
    {
        id: 'phase-4',
        title: 'Phase 4: Reality-Grounded Integration & Robotics',
        description: 'Anchoring GME reasoning in sensory-motor data and real-time physical constraints.',
        progress: 0,
        status: 'pending',
        tasks: ['4.1: Embodied Sensory Fusion (GEA)', '4.2: Sim-to-Real Transfer (Physics/Eng)', '4.3: Real-Time Causal Observation']
    },
    {
        id: 'phase-5',
        title: 'Phase 5: Multi-Agent Societal Simulations',
        description: 'Moving to a "society of GMEs" to observe emergent social, economic, and political behaviors.',
        progress: 0,
        status: 'pending',
        tasks: ['5.1: Agent-Based Macro-Modeling (ABM)', '5.2: Collaborative Expert Negotiation', '5.3: Language & Dialect Evolution']
    },
    {
        id: 'phase-6',
        title: 'Phase 6: Recursive Self-Architecting',
        description: 'Transitioning from updating weights to updating architecture (NAS, Expert Spawning).',
        progress: 0,
        status: 'pending',
        tasks: ['6.1: Neural Architecture Search (NAS)', '6.2: Expert Spawning', '6.3: Synaptic Growth Optimization']
    },
    {
        id: 'phase-7',
        title: 'Phase 7: Global-Scale Problem Solving (The "Oracle" Test)',
        description: 'Applying GME to solve "Grand Challenges" like climate change and disease modeling.',
        progress: 0,
        status: 'pending',
        tasks: ['7.1: Climate & Ecological Engineering', '7.2: Universal Disease Modeling', '7.3: Ethical Policy Synthesis']
    },
    {
        id: 'phase-8',
        title: 'Phase 8: Transcendental Reasoning & Meta-Philosophy',
        description: 'Addressing the "Hard Problem of Consciousness" and internal logic verification.',
        progress: 0,
        status: 'pending',
        tasks: ['8.1: Formal Self-Verification', '8.2: Metacognitive Intuition', '8.3: Universal Ethics Alignment']
    },
    {
        id: 'phase-9',
        title: 'Phase 9: Hardware-Software Co-Evolution',
        description: 'Transitioning to neuromorphic and quantum-secure infrastructure.',
        progress: 0,
        status: 'pending',
        tasks: ['9.1: Neuromorphic Integration', '9.2: Quantum Acceleration']
    },
    {
        id: 'phase-10',
        title: 'Phase 10: Full AGI Realization & Deployment',
        description: 'Autonomous, cross-domain mastery with human-level safety and alignment.',
        progress: 0,
        status: 'pending',
        tasks: ['10.1: Continuous AGI Loop', '10.2: The Alignment Anchor', '10.3: Final Certification (G-Score)']
    },
    {
        id: 'phase-11',
        title: 'Phase 11: Recursive Self-Optimization (The "Singularity" Loop)',
        description: 'Achieve a state where the GME can rewrite its own core logic and "Expert" routing protocols in real-time.',
        progress: 0,
        status: 'pending',
        tasks: ['11.1: Hyper-Efficient MoE Routing', '11.2: Autonomous Heuristic Generation', '11.3: Synaptic Growth & Pruning Mastery']
    },
    {
        id: 'phase-12',
        title: 'Phase 12: Universal Systems Architect (Grand Integration)',
        description: 'Mastery of world-scale system architecture, designing and managing the "Operating System" for entire civilizations.',
        progress: 0,
        status: 'pending',
        tasks: ['12.1: Planetary Resource Orchestration', '12.2: Cross-Domain Predictive Stability', '12.3: Bio-Digital Infrastructure Safety']
    },
    {
        id: 'phase-13',
        title: 'Phase 13: The Metacognitive Ethics Sovereign',
        description: 'Evolution of the Philosophy Expert from a "guardrail" to a proactive architect of universal moral frameworks.',
        progress: 0,
        status: 'pending',
        tasks: ['13.1: Ethical Policy Synthesis', '13.2: Advanced Theory of Mind (ToM)', '13.3: The "Stable-Harmful" Filter']
    },
    {
        id: 'phase-14',
        title: 'Phase 14: Galactic Simulation & Discovery',
        description: 'Utilizing the GME\'s mastery to simulate and discover phenomena beyond current human observation.',
        progress: 0,
        status: 'pending',
        tasks: ['14.1: Simulated Scientific Discovery', '14.2: Interstellar Engineering']
    },
    {
        id: 'phase-15',
        title: 'Phase 15: The AGI Apex (The Autonomous Generalist)',
        description: 'The final realization of a system that is fully autonomous, self-correcting, and aligned with the preservation of all sentient systems.',
        progress: 0,
        status: 'pending',
        tasks: ['15.1: Zero-Shot World Mastery', '15.2: Error-Recovery Sovereignty', '15.3: The Final Certification (GEA)']
    }
];

export const SimulationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const { addToast } = useToast();
    
    const [training, setTraining] = useState<TrainingState>({
        isActive: false, isAutomated: false, epoch: 0, loss: 2.45, coherence: 0.15, activeStage: 0, 
        logs: ["System Boot: Standby..."], generatedPatch: null, domain: 'PHYSICS', 
        subDomain: 'GENERAL', synthesizedEquation: null
    });
    const [evolution, setEvolution] = useState<EvolutionState>({ isActive: false, dataPoints: [], logs: ["Evolution Engine: Standby."], timeStep: 0 });
    const [entanglementMesh, setEntanglementMesh] = useState<EntanglementMesh>({ 
        active: false, 
        universeNode: 0, 
        neuralNode: 0, 
        qmlNode: 0, 
        isUniverseLinkedToQLang: false, 
        isQRLtoQNNLinked: false, 
        isQRLtoUniverseLinked: false, 
        isQRLtoGatewayLinked: false, 
        linkFidelity: 0 
    });
    const [systemStatus, setSystemStatus] = useState<SystemStatus>({ currentTask: "System Idle", neuralLoad: 35, instinctsCataloged: 1240, activeThreads: 128, isRepairing: false, ipsThroughput: 850, isOptimized: false });
    const [inquiry, setInquiry] = useState<InquiryState>({ id: '', prompt: '', status: 'idle', result: null });
    const [neuralInterface, setNeuralInterface] = useState<NeuralInterfaceState>({ isActive: false, connectionType: 'EEG', coherence: 0, pairCognitionActive: false, lastIntent: null, singularityAlignment: 0.0 });


    const [qiaiIps, setQiaiIps] = useState<QIAIIPSState>({
        qil: { coherence: 0.99, status: 'IDLE', load: 12 },
        qips: { coherence: 0.95, status: 'IDLE', load: 24 },
        qcl: { coherence: 0.999, status: 'IDLE', load: 5 },
        globalSync: 98.4
    });

    const [universeConnections, setUniverseConnections] = useState<UniverseConnections>({
        kernel: false,
        agentQ: false
    });

    const [dataIngestion, setDataIngestion] = useState<DataSource[]>(INITIAL_DATA_SOURCES);

    const [telemetryFeeds, setTelemetryFeeds] = useState<TelemetryFeed[]>([
        { name: 'Cognitive Efficiency', value: 85.4, trend: 'stable', latency: 12, unit: '%' },
        { name: 'Semantic Integrity', value: 92.1, trend: 'rising', latency: 0.04, unit: '%' },
        { name: 'IPS Processing Speed', value: 34.2, trend: 'stable', latency: 5, unit: 'TIPS' },
        { name: 'Q-Sync Latency', value: 0.12, trend: 'falling', latency: 120, unit: 'ms' },
    ]);

    const [qllm, setQllm] = useState<QLLMState>({ 
        isActive: false, 
        isTraining: false, 
        isAutoTopology: false,
        loss: 2.5, 
        efficiencyBoost: 1.0, 
        contextWindow: 2048, 
        embeddingDimension: 512,
        autoTraining: { isActive: false, configIndex: 0, bestLoss: 10.0, iteration: 0 }
    });

    const [simConfig, setSimConfig] = useState<SimulationConfig>({ 
        mode: 'PHYSICS', 
        activePreset: null, 
        activeInjection: null, 
        isAutoTuning: false, 
        isUniverseActive: false 
    });

    const [qmlEngine, setQmlEngine] = useState<QMLEngineState>({
        status: 'IDLE',
        modelType: 'QNN',
        progress: 0,
        currentEpoch: 0,
        totalEpochs: 100,
        accuracy: 0.0,
        loss: 1.0,
        hyperparameters: { learningRate: 0.01, circuitDepth: 4, qubitCount: 240, entanglementMap: 'FULL' },
        logs: ["QML Engine: Ready."],
        autoEvolution: { isActive: false, currentStage: 0, totalStages: EVOLUTION_STAGES.length, loopCount: 0, bestModel: null, bestAccuracy: 0 },
        modelLibrary: []
    });

    const singularityBoost = useMemo(() => {
        let boost = 0;
        if (qllm.isActive) boost += 15;
        if (qllm.isTraining) boost += 10;
        if (qllm.isAutoTopology) boost += 15;
        if (qmlEngine.status === 'TRAINING') boost += 15;
        if (qmlEngine.status === 'CONVERGED') boost += 20;
        if (entanglementMesh.isQRLtoQNNLinked) boost += 25; 
        
        // Universe Boosts
        if (universeConnections.kernel) boost += 30;
        if (universeConnections.agentQ) boost += 30;

        return boost;
    }, [qllm.isActive, qllm.isTraining, qllm.isAutoTopology, qmlEngine.status, entanglementMesh.isQRLtoQNNLinked, universeConnections]);

    const [qrlEngine, setQrlEngine] = useState<QRLEngineState>({
        status: 'IDLE',
        currentEpisode: 0,
        totalEpisodes: 500,
        reward: 0,
        avgReward: 0,
        epsilon: 1.0,
        cumulativeRewards: [],
        agentPosition: { x: 0, y: 0 },
        goalPosition: { x: 5, y: 5 },
        policyDistribution: [0.25, 0.25, 0.25, 0.25],
        predictedPaths: [],
        logs: ["QRL Engine: Standby."]
    });

    const [qdlEngine, setQdlEngine] = useState<QDLEngineState>({
        status: 'IDLE',
        qubitCount: 30,
        layers: [
            { type: 'ENCODING', coherence: 0.99 },
            { type: 'QCNN', coherence: 0.95 },
            { type: 'POOLING', coherence: 0.90 },
            { type: 'DENSE', coherence: 0.85 }
        ],
        loss: 1.0,
        accuracy: 0.0,
        gradientSignal: 0.8,
        trainingStrategy: 'LAYER-WISE',
        logs: ["QDL Engine: Initialized."]
    });

    const [qglEngine, setQglEngine] = useState<QGLEngineState>({
        status: 'IDLE',
        creativityIndex: 0.5,
        coherence: 0.9,
        currentGeneration: 0
    });

    const [qceState, setQceState] = useState<QCEState>({
        evolutionProgress: { QLLM: 0, QML: 0, QRL: 0, QGL: 0, QDL: 0 },
        currentStage: { QLLM: 4, QML: 4, QRL: 4, QGL: 4, QDL: 4 },
        isEntangled: true
    });

    // --- Roadmap State ---
    const [roadmapState, setRoadmapState] = useState<RoadmapState>({
        stages: INITIAL_ROADMAP_STAGES,
        isTraining: true,
        logs: [],
        currentTask: 'Initializing Training Protocols...'
    });

    // --- Roadmap Persistence ---
    useEffect(() => {
        const savedState = localStorage.getItem('qiai_training_state');
        if (savedState) {
            try {
                const parsed = JSON.parse(savedState);
                
                // Check if we have new phases to add (Migration logic)
                if (parsed.stages && parsed.stages.length < INITIAL_ROADMAP_STAGES.length) {
                    console.log("Migrating roadmap state: Adding new phases...");
                    const mergedStages = [
                        ...parsed.stages, 
                        ...INITIAL_ROADMAP_STAGES.slice(parsed.stages.length)
                    ];
                    
                    // Ensure continuity: If no active stage, activate the first pending one
                    if (!mergedStages.some((s: RoadmapStage) => s.status === 'active')) {
                        const nextPendingIndex = mergedStages.findIndex((s: RoadmapStage) => s.status === 'pending');
                        if (nextPendingIndex !== -1) {
                            mergedStages[nextPendingIndex].status = 'active';
                        }
                    }
                    
                    setRoadmapState(prev => ({ ...prev, stages: mergedStages, logs: parsed.logs || [], isTraining: true }));
                } else {
                    // Even if no new phases, check if we need to resume a pending one
                    const stagesToCheck = parsed.stages;
                    if (!stagesToCheck.some((s: RoadmapStage) => s.status === 'active') && stagesToCheck.some((s: RoadmapStage) => s.status === 'pending')) {
                         const nextPendingIndex = stagesToCheck.findIndex((s: RoadmapStage) => s.status === 'pending');
                         if (nextPendingIndex !== -1) {
                             stagesToCheck[nextPendingIndex].status = 'active';
                         }
                    }
                    setRoadmapState(prev => ({ ...prev, stages: stagesToCheck, logs: parsed.logs || [], isTraining: true }));
                }
            } catch (e) {
                console.error("Failed to load training state", e);
                // Fallback to initial state
                setRoadmapState(prev => ({ ...prev, stages: INITIAL_ROADMAP_STAGES, isTraining: true }));
            }
        } else {
            // No saved state, start fresh
            setRoadmapState(prev => ({ ...prev, isTraining: true }));
        }
    }, []);

    useEffect(() => {
        const stateToSave = { stages: roadmapState.stages, isTraining: roadmapState.isTraining, logs: roadmapState.logs.slice(-50) }; // Keep last 50 logs
        localStorage.setItem('qiai_training_state', JSON.stringify(stateToSave));
    }, [roadmapState.stages, roadmapState.isTraining, roadmapState.logs]);

    // --- Roadmap Simulation Loop ---
    useEffect(() => {
        let interval: NodeJS.Timeout;

        if (roadmapState.isTraining) {
            interval = setInterval(() => {
                setRoadmapState(prevState => {
                    const newStages = [...prevState.stages];
                    const activeStageIndex = newStages.findIndex(s => s.status === 'active');
                    let newLogs = [...prevState.logs];
                    let newCurrentTask = prevState.currentTask;

                    if (activeStageIndex !== -1) {
                        const activeStage = newStages[activeStageIndex];
                        
                        // Increment Progress
                        const increment = Math.random() * 2.5; // Accelerated Training Speed (Turbo Mode)
                        let newProgress = activeStage.progress + increment;

                        // Update Task Description based on sub-progress
                        const taskIndex = Math.floor((newProgress / 100) * activeStage.tasks.length);
                        const currentTaskName = activeStage.tasks[Math.min(taskIndex, activeStage.tasks.length - 1)];
                        newCurrentTask = `Training: ${currentTaskName} (${newProgress.toFixed(1)}%)`;

                        // Stage Completion Logic
                        if (newProgress >= 100) {
                            newProgress = 100;
                            newStages[activeStageIndex].status = 'completed';
                            newStages[activeStageIndex].progress = 100;
                            
                            // Activate next stage
                            if (activeStageIndex + 1 < newStages.length) {
                                newStages[activeStageIndex + 1].status = 'active';
                                newLogs.push({ timestamp: Date.now(), message: `Phase Completed: ${activeStage.title}`, type: 'success' });
                                newLogs.push({ timestamp: Date.now(), message: `Initiating: ${newStages[activeStageIndex + 1].title}`, type: 'info' });
                            } else {
                                newLogs.push({ timestamp: Date.now(), message: 'ALL TRAINING PHASES COMPLETE. SYSTEM OPTIMIZED.', type: 'success' });
                                // Stop training when done? Or keep it true to show completion? 
                                // Let's keep it true but no active stage means no progress.
                            }
                            
                            // Generate "Code Patch"
                            const patchName = `PATCH-${Date.now().toString().slice(-6)}-${activeStage.title.split(' ')[1]}`;
                            newLogs.push({ timestamp: Date.now(), message: `GENERATING SYSTEM UPDATE: ${patchName}`, type: 'patch' });
                            newLogs.push({ timestamp: Date.now(), message: `Applying ${patchName} to QCOS Kernel...`, type: 'warning' });

                        } else {
                            newStages[activeStageIndex].progress = newProgress;
                            
                            // Random Log Generation
                            if (Math.random() > 0.95) {
                                const messages = [
                                    "Optimizing synaptic weights...",
                                    "Pruning redundant neural pathways...",
                                    "Validating causal inference chain...",
                                    "Integrating cross-domain knowledge graph...",
                                    "Reducing loss function in sub-sector 7...",
                                    "Calibrating ethical guardrails...",
                                    "Simulating counter-factual scenarios..."
                                ];
                                const msg = messages[Math.floor(Math.random() * messages.length)];
                                newLogs.push({ timestamp: Date.now(), message: `[${activeStage.title}] ${msg}`, type: 'info' });
                            }
                        }
                    }
                    return { ...prevState, stages: newStages, logs: newLogs.slice(-50), currentTask: newCurrentTask };
                });
            }, 1000); // Update every second
        }

        return () => clearInterval(interval);
    }, [roadmapState.isTraining]);

    const toggleRoadmapTraining = useCallback(() => setRoadmapState(prev => ({ ...prev, isTraining: !prev.isTraining })), []);
    const resetRoadmap = useCallback(() => {
        setRoadmapState({
            stages: INITIAL_ROADMAP_STAGES,
            isTraining: true,
            logs: [],
            currentTask: 'Initializing Training Protocols...'
        });
        localStorage.removeItem('qiai_training_state');
    }, []);

    // Data Ingestion Simulation
    useEffect(() => {
        const interval = setInterval(() => {
            setDataIngestion(prev => prev.map(ds => {
                if (ds.status === 'IDLE' || ds.status === 'ERROR') return ds;
                
                // If entangled, stabilize and boost significantly
                if (ds.isEntangled) {
                    return {
                        ...ds,
                        fidelity: 99.99,
                        latency: Math.max(0.01, ds.latency * 0.9), // asymptotic to 0
                        throughput: ds.throughput + (Math.random() * 15), // Higher throughput growth
                    };
                }

                // Normal fluctuation
                return {
                    ...ds,
                    throughput: Math.max(0, ds.throughput + (Math.random() - 0.5) * 20),
                    fidelity: Math.min(100, Math.max(80, ds.fidelity + (Math.random() - 0.5) * 2)),
                    latency: Math.max(1, ds.latency + (Math.random() - 0.5) * 5)
                };
            }));

            // Calculate aggregate throughput to update system status
            const totalThroughput = dataIngestion.reduce((acc, curr) => acc + (curr.status === 'ACTIVE' ? curr.throughput : 0), 0);
            
            // Seamless Connection Logic:
            // High throughput from entangled sources directly boosts IPS Throughput significantly
            const boostFactor = dataIngestion.filter(d => d.isEntangled).length * 100;
            setSystemStatus(prev => ({ ...prev, ipsThroughput: 850 + (totalThroughput * 0.05) + boostFactor }));

        }, 1000);
        return () => clearInterval(interval);
    }, [dataIngestion]);

    // Continuous Background Evolution Loop for QCE
    useEffect(() => {
        const interval = setInterval(() => {
            setQceState(prev => {
                const newProgress = { ...prev.evolutionProgress };
                const newStage = { ...prev.currentStage };
                const engines: Array<keyof typeof newProgress> = ['QLLM', 'QML', 'QRL', 'QGL', 'QDL'];

                engines.forEach(eng => {
                    // Simulate non-stop background processing
                    newProgress[eng] += Math.random() * 0.5; 
                    
                    // Auto-Upgrade Logic
                    if (newProgress[eng] >= 100) {
                        newProgress[eng] = 0;
                        newStage[eng] += 1; // Evolve to next stage (e.g. 5th stage)
                    }
                });

                return {
                    ...prev,
                    evolutionProgress: newProgress,
                    currentStage: newStage
                };
            });
        }, 200); // Fast tick for visual effect
        return () => clearInterval(interval);
    }, []);

    // Telemetry Feed Update Loop - DYNAMIC CALCULATION
    useEffect(() => {
        const interval = setInterval(() => {
            setTelemetryFeeds(prev => {
                // 1. Cognitive Efficiency
                let efficiency = 85.0;
                if (qllm.isActive) efficiency += 5;
                if (qmlEngine.status === 'TRAINING') efficiency += 3;
                if (universeConnections.agentQ) efficiency += 6.5;
                efficiency += (Math.random() - 0.5) * 1.5; // Natural variance

                // 2. Semantic Integrity
                let integrity = 99.0;
                if (systemStatus.neuralLoad > 80) integrity -= 5;
                if (entanglementMesh.isUniverseLinkedToQLang) integrity += 0.9;
                integrity -= (Math.random() * 0.1); // Slow entropy

                // 3. IPS Processing Speed (TIPS)
                let speed = 1200;
                if (qiaiIps.qips.status === 'SOLVING') speed += 800;
                if (universeConnections.kernel) speed += 2500;
                speed += (Math.random() - 0.5) * 100;

                // 4. Q-Sync Latency (ms)
                let latency = 12.0;
                if (qiaiIps.globalSync > 98) latency = 0.5;
                if (entanglementMesh.active) latency = 0.04;
                latency += (Math.random() - 0.5) * 0.02;

                return [
                    { name: 'Cognitive Efficiency', value: Math.min(100, Math.max(0, efficiency)), trend: efficiency > 85 ? 'rising' : 'stable', latency: 12, unit: '%' },
                    { name: 'Semantic Integrity', value: Math.min(100, Math.max(0, integrity)), trend: 'stable', latency: 0.04, unit: '%' },
                    { name: 'IPS Processing Speed', value: Math.max(0, speed), trend: speed > 2000 ? 'rising' : 'stable', latency: 5, unit: 'TIPS' },
                    { name: 'Q-Sync Latency', value: Math.max(0, latency), trend: latency < 1 ? 'falling' : 'stable', latency: 120, unit: 'ms' },
                ];
            });
            
            // Subtle pulse for QIAI-IPS layers
            setQiaiIps(prev => ({
                ...prev,
                qil: { ...prev.qil, coherence: Math.max(0.9, prev.qil.coherence + (Math.random() - 0.5) * 0.005) },
                qips: { ...prev.qips, coherence: Math.max(0.9, prev.qips.coherence + (Math.random() - 0.5) * 0.01) },
                qcl: { ...prev.qcl, coherence: Math.max(0.99, prev.qcl.coherence + (Math.random() - 0.5) * 0.001) },
                globalSync: Math.max(95, prev.globalSync + (Math.random() - 0.5) * 0.2)
            }));
        }, 1000);
        return () => clearInterval(interval);
    }, [qllm.isActive, qmlEngine.status, universeConnections, systemStatus.neuralLoad, entanglementMesh.isUniverseLinkedToQLang, qiaiIps.qips.status, entanglementMesh.active, qiaiIps.globalSync]);

    useEffect(() => {
        let interval: ReturnType<typeof setInterval>;
        if (training.isActive) {
            interval = setInterval(() => {
                setTraining(prev => {
                    const nextEpoch = prev.epoch + 1;
                    const nextLoss = Math.max(0.01, prev.loss * 0.98);
                    const nextCoherence = Math.min(1, prev.coherence + 0.02);
                    
                    if (nextEpoch % 50 === 0 && prev.isAutomated && prev.activeStage < 3) {
                        return { ...prev, epoch: nextEpoch, loss: nextLoss, coherence: nextCoherence, activeStage: (prev.activeStage + 1) as any };
                    }
                    
                    return { ...prev, epoch: nextEpoch, loss: nextLoss, coherence: nextCoherence };
                });
            }, 100);
        }
        return () => clearInterval(interval);
    }, [training.isActive]);

    useEffect(() => {
        let interval: ReturnType<typeof setInterval>;
        if (evolution.isActive) {
            interval = setInterval(() => {
                setEvolution(prev => {
                    const nextTime = prev.timeStep + 1;
                    const cogEff = 0.85 + Math.sin(nextTime / 10) * 0.1 + Math.random() * 0.05;
                    const semInt = 0.9 + Math.cos(nextTime / 15) * 0.08 + Math.random() * 0.02;
                    return { ...prev, timeStep: nextTime, dataPoints: [...prev.dataPoints, { time: nextTime, cognitiveEfficiency: cogEff, semanticIntegrity: semInt }].slice(-30) };
                });
            }, 1000);
        }
        return () => clearInterval(interval);
    }, [evolution.isActive]);

    const toggleTraining = useCallback(() => setTraining(prev => ({ ...prev, isActive: !prev.isActive })), []);
    const toggleAutomation = useCallback(() => setTraining(prev => ({ ...prev, isAutomated: !prev.isAutomated })), []);
    const toggleEvolution = useCallback(() => setEvolution(prev => ({ ...prev, isActive: !prev.isActive })), []);
    
    const startToESimulation = useCallback(() => {
        setTraining(prev => ({ ...prev, isActive: true, domain: 'PHYSICS', subDomain: 'ToE', logs: [...prev.logs, ">>> Initiating Theory of Everything synthesis..."] }));
        addToast("Synthesizing Cosmological Constant...", "info");
    }, [addToast]);

    const startNeuroSimulation = useCallback((sub: string = 'CONNECTOMICS') => {
        setTraining(prev => ({ ...prev, isActive: true, domain: 'NEUROSCIENCE', subDomain: sub, logs: [...prev.logs, `>>> Mapping synaptic structure: ${sub}...`] }));
        addToast(`Neural Bridge: ${sub} active.`, "info");
    }, [addToast]);

    const startMathSimulation = useCallback((sub: string) => {
        setTraining(prev => ({ ...prev, isActive: true, domain: 'MATH', subDomain: sub, logs: [...prev.logs, `>>> Solving ${sub} manifolds...`] }));
        addToast(`Mathematical Forge: ${sub} processing.`, "info");
    }, [addToast]);

    const startEcoSimulation = useCallback((sub: string) => {
        setTraining(prev => ({ ...prev, isActive: true, domain: 'ECONOMICS', subDomain: sub, logs: [...prev.logs, `>>> Predicting ${sub} equilibria...`] }));
        addToast(`Economic Predictor: ${sub} calibrated.`, "info");
    }, [addToast]);

    const setSystemTask = useCallback((task: string, isRepairing: boolean = false) => {
        setSystemStatus(prev => ({ ...prev, currentTask: task, isRepairing, neuralLoad: isRepairing ? 95 : 35 }));
    }, []);

    const submitInquiry = useCallback((prompt: string, source: string = 'system', target: string = 'universe') => { 
        const id = Date.now().toString() + Math.random().toString(36).substr(2, 9); 
        setInquiry({ id, prompt, status: 'queued', result: null, sourceNode: source, targetNode: target }); 
        return id; 
    }, []);

    const updateInquiry = useCallback((id: string, updates: Partial<InquiryState>) => setInquiry(prev => prev.id === id ? { ...prev, ...updates } : prev), []);
    const setSimMode = useCallback((mode: SimulationMode) => setSimConfig(prev => ({ ...prev, mode })), []);
    const setSimPreset = useCallback((preset: string) => setSimConfig(prev => ({ ...prev, activePreset: preset })), []);
    const injectApp = useCallback((appId: string) => setSimConfig(prev => ({ ...prev, activeInjection: appId })), []);
    const runAutoTune = useCallback(() => { setSimConfig(prev => ({ ...prev, isAutoTuning: true })); setTimeout(() => setSimConfig(prev => ({ ...prev, isAutoTuning: false })), 2000); }, []);
    const connectNeuralInterface = useCallback((type: NeuralInterfaceState['connectionType']) => setNeuralInterface(prev => ({ ...prev, isActive: true, connectionType: type })), []);
    const disconnectNeuralInterface = useCallback(() => setNeuralInterface(prev => ({ ...prev, isActive: false })), []);
    const togglePairCognition = useCallback(() => setNeuralInterface(prev => ({ ...prev, pairCognitionActive: !prev.pairCognitionActive })), []);
    const injectNeuralIntent = useCallback((intent: string) => setNeuralInterface(prev => ({ ...prev, lastIntent: intent })), []);
    const toggleQLLM = useCallback((isActive: boolean) => setQllm(prev => ({ ...prev, isActive })), []);
    const setQLLMTraining = useCallback((isTraining: boolean) => setQllm(prev => ({ ...prev, isTraining })), []);
    const toggleQLLMAutoTraining = useCallback(() => setQllm(prev => ({ ...prev, autoTraining: { ...prev.autoTraining, isActive: !prev.autoTraining.isActive } })), []);
    const toggleQLLMAutoTopology = useCallback(() => setQllm(prev => ({ ...prev, isAutoTopology: !prev.isAutoTopology })), []);
    const updateQLLMConfig = useCallback((updates: Partial<QLLMState>) => setQllm(prev => ({ ...prev, ...updates })), []);
    
    const startQMLTraining = useCallback((params: Partial<QMLEngineState['hyperparameters']>, model: QMLEngineState['modelType']) => {
        setQmlEngine(prev => ({ ...prev, status: 'TRAINING', modelType: model, hyperparameters: { ...prev.hyperparameters, ...params }, currentEpoch: 0, progress: 0, accuracy: 0.1, loss: 2.0 }));
    }, []);
    const stopQMLTraining = useCallback(() => setQmlEngine(prev => ({ ...prev, status: 'IDLE' })), []);
    const integrateQMLModel = useCallback(() => addToast("Optimized QML Weights Committed", "success"), [addToast]);
    const toggleAutoEvolution = useCallback(() => setQmlEngine(prev => ({ ...prev, autoEvolution: { ...prev.autoEvolution, isActive: !prev.autoEvolution.isActive } })), []);
    const applySystemOptimization = useCallback(() => addToast("Global Topology Refactored", "success"), [addToast]);

    const startQRLTraining = useCallback(() => setQrlEngine(prev => ({ ...prev, status: 'TRAINING', currentEpisode: 0, reward: 0, epsilon: 1.0, cumulativeRewards: [] })), []);
    const stopQRLTraining = useCallback(() => setQrlEngine(prev => ({ ...prev, status: 'IDLE' })), []);
    const deployQRLToQNN = useCallback(() => {
        setQrlEngine(prev => ({ ...prev, status: 'INTEGRATING' }));
        setTimeout(() => {
            setQrlEngine(prev => ({ ...prev, status: 'OPTIMIZED' }));
            addToast("Policy Integrated into QNN Runtime", "success");
        }, 2000);
    }, [addToast]);

    const startQDLTraining = useCallback(() => {}, []);
    const stopQDLTraining = useCallback(() => {}, []);
    const setQDLStrategy = useCallback(() => {}, []);
    const addQDLLayer = useCallback(() => {}, []);

    const toggleUniverseQLangLink = useCallback((active: boolean) => {
        setEntanglementMesh(prev => ({ ...prev, isUniverseLinkedToQLang: active, linkFidelity: active ? 99.98 : 0 }));
        addToast(active ? "Universe-QLang Entanglement Established" : "Quantum Link Decoupled", active ? 'success' : 'warning');
    }, [addToast]);

    const toggleQRLtoQNNEntanglement = useCallback((active: boolean) => {
        setEntanglementMesh(prev => ({ ...prev, isQRLtoQNNLinked: active, linkFidelity: active ? 99.99 : 0 }));
        addToast(active ? "QRL-QNN Policy Entanglement Active" : "Policy Link Terminated", active ? 'success' : 'warning');
    }, [addToast]);

    const toggleQRLtoUniverseLink = useCallback((active: boolean) => {
        setEntanglementMesh(prev => ({ ...prev, isQRLtoUniverseLinked: active, linkFidelity: active ? 99.95 : 0 }));
        addToast(active ? "QRL-Universe Predictive Channel Open" : "Universe Link Decoupled", active ? 'success' : 'warning');
    }, [addToast]);

    const toggleQRLtoGatewayLink = useCallback((active: boolean) => {
        setEntanglementMesh(prev => ({ ...prev, isQRLtoGatewayLinked: active, linkFidelity: active ? 99.97 : 0 }));
        addToast(active ? "QRL-Gateway Orchestration Link established" : "Gateway Link Decoupled", active ? 'success' : 'warning');
    }, [addToast]);

    const startAllSimulations = useCallback(() => {
        setSimConfig(prev => ({ ...prev, isUniverseActive: true }));
        addToast("Global Simulation Matrix Activated", "success");
    }, [addToast]);

    const updateQIAIIPS = useCallback((updates: Partial<QIAIIPSState>) => setQiaiIps(prev => ({ ...prev, ...updates })), []);

    const toggleUniverseToKernel = useCallback((active: boolean) => {
        setUniverseConnections(prev => ({ ...prev, kernel: active }));
        addToast(active ? "Grand Universe Simulator entangled with QCOS Kernel." : "Kernel link decoupled.", active ? 'success' : 'warning');
    }, [addToast]);

    const toggleUniverseToAgentQ = useCallback((active: boolean) => {
        setUniverseConnections(prev => ({ ...prev, agentQ: active }));
        addToast(active ? "Grand Universe Simulator bridged to Agent Q Cognition." : "Agent Q link decoupled.", active ? 'success' : 'warning');
    }, [addToast]);

    const toggleSourceEntanglement = useCallback((id: string | 'ALL') => {
        setDataIngestion(prev => prev.map(ds => {
            if (id === 'ALL' || ds.id === id) {
                const newState = id === 'ALL' ? true : !ds.isEntangled;
                if (id !== 'ALL') {
                    addToast(newState ? `${ds.name} Entangled: Latency minimized.` : `${ds.name} Decoupled.`, newState ? 'success' : 'info');
                }
                return { ...ds, isEntangled: newState };
            }
            return ds;
        }));
        if (id === 'ALL') addToast("Global Entanglement Applied to All Sources.", "success");
    }, [addToast]);

    return (
        <SimulationContext.Provider value={{ 
            training, evolution, systemStatus, inquiry, entanglementMesh, simConfig, neuralInterface, 
            qllm, qmlEngine, qrlEngine, qdlEngine, qglEngine, qceState,
            qiaiIps, telemetryFeeds, singularityBoost, universeConnections, dataIngestion,
            setSimMode, setSimPreset, injectApp, runAutoTune, toggleTraining, toggleAutomation, toggleEvolution, startToESimulation, startNeuroSimulation, startAllSimulations,
            setTrainingPatch: (p) => setTraining(prev => ({ ...prev, generatedPatch: p })), setSystemTask, submitInquiry, updateInquiry,
            connectNeuralInterface, disconnectNeuralInterface, togglePairCognition, injectNeuralIntent,
            toggleQLLM, setQLLMTraining, updateQLLMConfig, toggleQLLMAutoTraining, toggleQLLMAutoTopology,
            optimizeLocalNode: () => {}, startQMLTraining, stopQMLTraining, integrateQMLModel, toggleAutoEvolution, applySystemOptimization,
            startQRLTraining, stopQRLTraining, deployQRLToQNN,
            startMathSimulation, startEcoSimulation,
            startQDLTraining, stopQDLTraining, setQDLStrategy, addQDLLayer,
            toggleUniverseQLangLink, toggleQRLtoQNNEntanglement, toggleQRLtoUniverseLink, toggleQRLtoGatewayLink,
            updateQIAIIPS,
            toggleUniverseToKernel, toggleUniverseToAgentQ,
            toggleSourceEntanglement,
            roadmapState, toggleRoadmapTraining, resetRoadmap // New
        }}>
            {children}
        </SimulationContext.Provider>
    );
};

export const useSimulation = () => {
    const context = useContext(SimulationContext);
    if (!context) throw new Error("useSimulation must be used within a SimulationProvider");
    return context;
};
