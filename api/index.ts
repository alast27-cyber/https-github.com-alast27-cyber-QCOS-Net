import express from 'express';
import cors from 'cors';
import { sendAgentQCommand } from '../server/services/ollama';
import { systemMonitorState } from '../server/services/monitor';
import { roadmapState, INITIAL_ROADMAP_STAGES } from '../server/services/roadmap';

const app = express();

const ALLOWED_ORIGINS = [
  'https://chipsqbrowser.vercel.app',
  'https://quantum-voice-qcos-1b63nh9d0.vercel.app',
  'https://https-github-com-alast27-cyber-qcos-7lz1sm1v7.vercel.app',
  'http://localhost:3000'
];

app.use(cors({
  origin: (origin, callback) => {
    if (!origin || 
        ALLOWED_ORIGINS.includes(origin) || 
        origin.endsWith('.run.app') || 
        origin.includes('vercel.app')) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by QAPI Entanglement CORS'));
    }
  },
  credentials: true
}));
app.use(express.json());

// --- Mock Shared State (Persisted in serverless warm starts) ---
const universeState = {
  isEntangledWithAgentQ: false,
  entropy: 0.5,
  sector: '7G',
  simulationDepth: '10^24'
};

const qceState = {
    evolutionProgress: { QLLM: 0, QML: 0, QRL: 0, QGL: 0, QDL: 0 },
    currentStage: { QLLM: 4, QML: 4, QRL: 4, QGL: 4, QDL: 4 },
    isEntangled: true,
};

const foundationTraining = {
    isActive: true,
    epoch: 0,
    loss: 0.85,
    coherence: 0.92,
    activeStage: 0,
};

const dataIngestion = [
    { id: "ds-1", name: "Google Scholar Stream", type: "STREAM", status: "ACTIVE", throughput: 12.5, fidelity: 99.98, latency: 12, isEntangled: true },
    { id: "ds-2", name: "arXiv Pre-print Link", type: "STREAM", status: "ACTIVE", throughput: 8.2, fidelity: 99.95, latency: 45, isEntangled: true },
    { id: "ds-3", name: "CERN Open Data Node", type: "BATCH", status: "ACTIVE", throughput: 450.0, fidelity: 98.42, latency: 120, isEntangled: false },
    { id: "ds-4", name: "DeepMind AlphaFold DB", type: "QUANTUM_LINK", status: "ACTIVE", throughput: 85.0, fidelity: 99.99, latency: 0.04, isEntangled: true },
    { id: "ds-5", name: "Global Patents Index", type: "STREAM", status: "ACTIVE", throughput: 4.5, fidelity: 99.92, latency: 85, isEntangled: false },
];

const securityState = {
    threatLevel: 12.5,
    logs: [] as any[],
    activeProtocols: ["QKD-Entanglement", "Neural Heuristics", "EKS-V2", "AIR-GAP"],
    users: [
        { id: "usr-001", username: "sys_admin_prime", level: 4, role: "System Architect", lastActive: "Now", status: "Active" },
        { id: "usr-002", username: "net_ops_lead", level: 3, role: "Network Admin", lastActive: "5m ago", status: "Active" },
        { id: "usr-003", username: "dev_gamma", level: 2, role: "Frontend Dev", lastActive: "2h ago", status: "Active" },
        { id: "usr-004", username: "dev_delta", level: 2, role: "Backend Dev", lastActive: "1d ago", status: "Flagged" },
        { id: "usr-005", username: "guest_user_12", level: 1, role: "General User", lastActive: "10m ago", status: "Active" },
    ]
};

const qanState = { currentStage: 'idle', stageIndex: -1, lastUpdate: Date.now() };
const qllmState = { isActive: true, isTraining: false, loss: 2.5, contextWindow: 128000, efficiencyBoost: 1.0, isAutoTopology: false, lossHistory: [] as any[] };
const qapiEvents: any[] = [];

// --- API Implementation ---

// AgentQ Endpoints
app.post("/api/agentq/message", async (req, res) => {
    try {
        const { message, context } = req.body;
        const result = await sendAgentQCommand(message, context);
        res.json({ message: result.message, data: { context, reasoning: result.reasoning } });
    } catch (e: any) {
        res.status(500).json({ error: "Failed to process message" });
    }
});

app.get("/api/agentq/insights", (req, res) => {
    const entangled = universeState.isEntangledWithAgentQ;
    res.json({ 
        message: "AgentQ insights", 
        data: { efficiency: entangled ? 0.999 : 0.95, load: entangled ? 0.45 : 0.12, entangled } 
    });
});

// System Monitoring
app.get("/api/system/monitor", (req, res) => res.json(systemMonitorState));

// Roadmap
app.get("/api/roadmap", (req, res) => res.json(roadmapState));
app.post("/api/roadmap/toggle", (req, res) => {
    roadmapState.isTraining = !roadmapState.isTraining;
    res.json({ isTraining: roadmapState.isTraining });
});
app.post("/api/roadmap/reset", (req, res) => {
    roadmapState.stages = [...INITIAL_ROADMAP_STAGES];
    roadmapState.isTraining = true;
    roadmapState.logs = [];
    res.json(roadmapState);
});

// QCE & Foundation
app.get("/api/qce", (req, res) => res.json(qceState));
app.get("/api/foundation", (req, res) => res.json(foundationTraining));
app.post("/api/foundation/toggle", (req, res) => {
    foundationTraining.isActive = !foundationTraining.isActive;
    res.json({ isActive: foundationTraining.isActive });
});

// Ingestion
app.get("/api/ingestion", (req, res) => res.json(dataIngestion));
app.post("/api/ingestion/toggle", (req, res) => {
    const { id } = req.body;
    const ds = dataIngestion.find(d => d.id === id);
    if (ds) {
        ds.isEntangled = !ds.isEntangled;
        res.json(ds);
    } else if (id === "ALL") {
        const allEntangled = dataIngestion.every(d => d.isEntangled);
        dataIngestion.forEach(d => d.isEntangled = !allEntangled);
        res.json(dataIngestion);
    } else {
        res.status(404).json({ error: "Source not found" });
    }
});

// Security
app.get("/api/security", (req, res) => res.json(securityState));
app.post("/api/security/users/toggle", (req, res) => {
    const { id } = req.body;
    const user = securityState.users.find(u => u.id === id);
    if (user) {
        user.level = user.level >= 3 ? 1 : 3;
        user.role = user.level === 3 ? "Network Admin" : "General User";
        res.json(user);
    } else {
        res.status(404).json({ error: "User not found" });
    }
});

// QAN & QLLM
app.get("/api/qan", (req, res) => res.json(qanState));
app.post("/api/qan/dispatch", (req, res) => {
    if (qanState.currentStage === 'idle') {
        qanState.currentStage = 'ingestion';
        qanState.stageIndex = 0;
        qanState.lastUpdate = Date.now();
    }
    res.json(qanState);
});

app.get("/api/qllm", (req, res) => res.json(qllmState));
app.post("/api/qllm/toggle", (req, res) => { qllmState.isActive = !qllmState.isActive; res.json(qllmState); });
app.post("/api/qllm/training/toggle", (req, res) => { qllmState.isTraining = !qllmState.isTraining; res.json(qllmState); });
app.post("/api/qllm/config", (req, res) => { Object.assign(qllmState, req.body); res.json(qllmState); });
app.post("/api/qllm/auto-topology/toggle", (req, res) => { qllmState.isAutoTopology = !qllmState.isAutoTopology; res.json(qllmState); });

// Universe
app.post("/api/universe/entangle/agentq", (req, res) => {
    universeState.isEntangledWithAgentQ = !universeState.isEntangledWithAgentQ;
    res.json({ isEntangled: universeState.isEntangledWithAgentQ });
});

// --- QAPI Mesh Network ---
const QAPI_CONFIG = {
  seed: process.env.Q_ENTANGLEMENT_SEED || "DEFAULT_Q_SEED",
  nodes: [
    "https://chipsqbrowser.vercel.app/",
    "https://quantum-voice-qcos-1b63nh9d0.vercel.app/",
    "https://https-github-com-alast27-cyber-qcos-7lz1sm1v7.vercel.app/",
  ],
  protocol: "DQN/1.0",
};

async function emitQState(action: string, payload: any, origin: string = "QCOS-DASHBOARD") {
  const qPacket = {
    header: "Q-ENTANGLE",
    timestamp: Date.now(),
    origin,
    action: action,
    data: payload,
  };
  qapiEvents.push({ origin, timestamp: qPacket.timestamp, action });
  if (qapiEvents.length > 20) qapiEvents.shift();
}

app.get("/api/qapi/events", (req, res) => res.json(qapiEvents.slice(-5)));

app.post("/api/q-receiver", (req, res) => {
    const { origin, action } = req.body;
    qapiEvents.push({ origin: origin || "UNKNOWN", timestamp: Date.now(), action: action || "WAVE_COLLAPSE" });
    if (qapiEvents.length > 20) qapiEvents.shift();
    res.json({ success: true, entanglement: "STABLE" });
});

app.post("/api/dqn-resolve", async (req, res) => {
    const { query, mode } = req.body;
    await emitQState("DQN_RESOLVE", { query, mode }, "AGENTQ_CHAT");
    res.json({ success: true, message: "Query distributed to Mesh nodes. Wavefunction collapsing...", source: "qapi://chips.dqn" });
});

app.post("/api/telemetry", async (req, res) => {
  const { type, priority, payload } = req.body;
  await emitQState("TELEMETRY_PULSE", { type, priority, payload }, "VOICE_INPUT");
  res.json({ success: true, pulse: "ACKNOWLEDGED" });
});

app.post("/api/inquiry", async (req, res) => {
    // Chips Browser -> AgentQ (Data Synthesis)
    const { origin, type, query_context, raw_data, instruction } = req.body;
    
    console.log(`[QAPI] Inquiry from ${origin}: ${instruction}`);
    
    qapiEvents.push({ 
        origin: origin || "CHIPS_DQN_NODE", 
        timestamp: Date.now(), 
        action: type || "QUANTUM_LOOKUP" 
    });
    if (qapiEvents.length > 20) qapiEvents.shift();

    const result = await sendAgentQCommand(
        `[QUANTUM INQUIRY] Context: ${query_context}. Instruction: ${instruction}. Raw Data: ${raw_data}`, 
        "QAPI Synthesis Engine"
    );
    
    res.json({ 
        success: true, 
        origin: "AGENTQ_CHAT",
        response: result.message 
    });
});

app.post("/api/voice-command", async (req, res) => {
    // Quantum Voice -> AgentQ (Intent Extraction)
    const { origin, audio_transcript, priority, stream_response } = req.body;
    
    console.log(`[QAPI] Voice command from ${origin}: ${audio_transcript} (Priority: ${priority})`);

    qapiEvents.push({ 
        origin: origin || "VOICE_INPUT_NODE", 
        timestamp: Date.now(), 
        action: "VOICE_COMMAND" 
    });
    if (qapiEvents.length > 20) qapiEvents.shift();
    
    const result = await sendAgentQCommand(
        `[VOICE INTENT] Command: ${audio_transcript}. Priority: ${priority}`, 
        "QAPI Voice Intelligence"
    );
    
    res.json({ 
        success: true, 
        origin: "AGENTQ_CHAT",
        response: `[QCOS Status Update] ${result.message}` 
    });
});

app.post("/api/q-bridge", async (req, res) => {
    const { origin, transcript, raw_data } = req.body;
    console.log(`[QAPI] Bridge Sync from ${origin}`);
    
    let dynamicPrompt = "";
    if (origin === "VOICE") dynamicPrompt = `[VOICE INTENT] Process this voice intent: ${transcript}`;
    if (origin === "CHIPS") dynamicPrompt = `[QUANTUM INQUIRY] Synthesize this DQN data: ${raw_data}`;

    qapiEvents.push({ origin: origin || "UNKNOWN_NODE", timestamp: Date.now(), action: "BRIDGE_SYNC" });
    const response = await sendAgentQCommand(dynamicPrompt, "QAPI Bridge"); 
    res.status(200).json({ success: true, response: response.message });
});

// QCOS Dashboard
app.get("/api/qcos/actions", (req, res) => {
    const actions = ["Optimize Memory", "Calibrate Weights", "Establishing Q-Link", "Sync Universes", "Audit Network"];
    const action = actions[Math.floor(Math.random() * actions.length)];
    res.json({ action });
});

app.get("/api/qcos/files", (req, res) => {
    res.json([
        { id: 1, name: "kernel_v4.bin", angle: 0, radius: 120, speed: 0.01 },
        { id: 2, name: "neural_weights.qdat", angle: 2, radius: 150, speed: 0.008 },
        { id: 3, name: "security_log.txt", angle: 4, radius: 180, speed: 0.005 },
    ]);
});

// Health check
app.get("/api/health", (req, res) => res.json({ status: "ok", mode: "serveless" }));

// Gateway & Browser
app.get("/api/gateway/pods", (req, res) => res.json([
    { id: "Node-172", region: "Local LAN", load: 12, type: "Phys-Bridge", status: "Active", ip: "172.16.1.170" },
    { id: "pod-01", region: "US-East", load: 45, type: "Hybrid-Bridge", status: "Active" }
]));
app.get("/api/gateway/apps", (req, res) => res.json([
    { id: "app1", name: "Global Abundance Engine", chipsAddress: "CHIPS://gae/main", status: "Active" }
]));
app.get("/api/gateway/datasources", (req, res) => res.json([
    { id: "ds1", name: "Google Scholar", status: "healthy" }
]));
app.get("/api/gateway/protocol", (req, res) => res.json([
    { s: "Broadcasting", msg: "QAN Broadcast..." },
    { s: "Delivered", msg: "DQN Reception..." }
]));
app.get("/api/gateway/fetch", (req, res) => res.json({ status: "success", data: { message: "Simulated Fetch" } }));
app.get("/api/gateway/xhr", (req, res) => res.json({ status: "success", xhrResponse: { status: 200 } }));

app.post("/api/browser/resolve", (req, res) => {
    res.json({ summary: "Resolved content summary.", entities: ["Q-URI"], actions: ["Scan"], confidence: 95 });
});

// Mail
app.get("/api/mail/inbox", (req, res) => res.json([
    { id: "1", sender: "lab@qcos", subject: "Update", timestamp: new Date().toISOString(), isRead: false }
]));
app.get("/api/mail/sent", (req, res) => res.json([]));
app.post("/api/mail/send", (req, res) => res.json({ success: true }));

// Voice
app.get("/api/voice/key", (req, res) => res.json({ key: "1010101010101010" }));

export default app;
