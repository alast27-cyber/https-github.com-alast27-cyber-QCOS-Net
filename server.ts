import express from "express";
import cors from "cors";
import { createServer as createViteServer } from "vite";
import { fileURLToPath } from 'url';
import path from 'path';
import { sendAgentQCommand } from './server/services/ollama';
import { startSystemMonitor, systemMonitorState, trackRequest } from './server/services/monitor';
import { startRoadmapSimulation, roadmapState, INITIAL_ROADMAP_STAGES } from './server/services/roadmap';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(cors());
  app.use(express.json());

  // Start Services
  console.log("Starting QCOS Backend Services...");
  startSystemMonitor();
  startRoadmapSimulation();
  console.log("System Monitor & Roadmap Simulation Active.");

  // Request Logging Middleware & Metrics
  app.use((req, res, next) => {
    const start = Date.now();
    if (req.url.startsWith('/api')) {
      console.log(`[API] ${req.method} ${req.url}`);
    }
    
    res.on('finish', () => {
        if (req.url.startsWith('/api')) {
            const duration = Date.now() - start;
            trackRequest(duration, res.statusCode >= 400);
        }
    });
    
    next();
  });

  // Health Check
  app.get("/api/health", (req, res) => {
    console.log("[API] Health check hit");
    res.json({ status: "ok", uptime: process.uptime() });
  });

  // --- AGI Roadmap State ---
  // (Moved to server/services/roadmap.ts)

  const qceState = {
    evolutionProgress: { QLLM: 0, QML: 0, QRL: 0, QGL: 0, QDL: 0 },
    currentStage: { QLLM: 4, QML: 4, QRL: 4, QGL: 4, QDL: 4 },
    isEntangled: true
  };

  const foundationTraining = {
    isActive: true,
    epoch: 0,
    loss: 0.85,
    coherence: 0.92,
    activeStage: 0
  };

  const dataIngestion = [
    { id: 'ds-1', name: 'Google Scholar Stream', type: 'STREAM', status: 'ACTIVE', throughput: 12.5, fidelity: 99.98, latency: 12, isEntangled: true },
    { id: 'ds-2', name: 'arXiv Pre-print Link', type: 'STREAM', status: 'ACTIVE', throughput: 8.2, fidelity: 99.95, latency: 45, isEntangled: true },
    { id: 'ds-3', name: 'CERN Open Data Node', type: 'BATCH', status: 'ACTIVE', throughput: 450.0, fidelity: 98.42, latency: 120, isEntangled: false },
    { id: 'ds-4', name: 'DeepMind AlphaFold DB', type: 'QUANTUM_LINK', status: 'ACTIVE', throughput: 85.0, fidelity: 99.99, latency: 0.04, isEntangled: true },
    { id: 'ds-5', name: 'Global Patents Index', type: 'STREAM', status: 'ACTIVE', throughput: 4.5, fidelity: 99.92, latency: 85, isEntangled: false }
  ];

  const securityState = {
    threatLevel: 12.5,
    logs: [] as any[],
    activeProtocols: ['QKD-Entanglement', 'Neural Heuristics', 'EKS-V2', 'AIR-GAP'],
    users: [
      { id: 'usr-001', username: 'sys_admin_prime', level: 4, role: 'System Architect', lastActive: 'Now', status: 'Active' },
      { id: 'usr-002', username: 'net_ops_lead', level: 3, role: 'Network Admin', lastActive: '5m ago', status: 'Active' },
      { id: 'usr-003', username: 'dev_gamma', level: 2, role: 'Frontend Dev', lastActive: '2h ago', status: 'Active' },
      { id: 'usr-004', username: 'dev_delta', level: 2, role: 'Backend Dev', lastActive: '1d ago', status: 'Flagged' },
      { id: 'usr-005', username: 'guest_user_12', level: 1, role: 'General User', lastActive: '10m ago', status: 'Active' },
    ]
  };

  const universeState = {
    isEntangledWithAgentQ: false,
    entropy: 0.5,
    sector: '7G',
    simulationDepth: '10^24'
  };

  const qanState = {
    currentStage: 'idle',
    stageIndex: -1,
    lastUpdate: Date.now()
  };

  const qllmState = {
    isActive: true,
    isTraining: false,
    loss: 2.5,
    contextWindow: 128000,
    efficiencyBoost: 1.0,
    isAutoTopology: false,
    lossHistory: [] as any[]
  };

  const QAN_STAGES = ['ingestion', 'activation', 'compilation', 'targeting', 'dispatch'];

  // Background Training Loop
  setInterval(async () => {
    // 1. Roadmap Training (Moved to server/services/roadmap.ts)

    // 2. QCE Evolution
    const engines: Array<keyof typeof qceState.evolutionProgress> = ['QLLM', 'QML', 'QRL', 'QGL', 'QDL'];
    engines.forEach(eng => {
      qceState.evolutionProgress[eng] += Math.random() * 0.5; 
      if (qceState.evolutionProgress[eng] >= 100) {
        qceState.evolutionProgress[eng] = 0;
        qceState.currentStage[eng] += 1;
      }
    });

    // 3. Foundation Training
    if (foundationTraining.isActive) {
      foundationTraining.epoch += 1;
      foundationTraining.loss = Math.max(0.01, foundationTraining.loss * 0.995);
      foundationTraining.coherence = Math.min(1, foundationTraining.coherence + 0.001);
    }

    // 4. Data Ingestion
    dataIngestion.forEach(ds => {
      if (ds.status === 'ACTIVE') {
        ds.throughput = Math.max(0, ds.throughput + (Math.random() - 0.5) * 20);
        ds.fidelity = Math.min(100, Math.max(80, ds.fidelity + (Math.random() - 0.5) * 2));
        ds.latency = Math.max(1, ds.latency + (Math.random() - 0.5) * 5);
      }
    });

    // 5. Security Monitor
    const fluctuation = Math.random() > 0.8 ? (Math.random() * 10) : -(Math.random() * 5);
    securityState.threatLevel = Math.max(0, Math.min(100, securityState.threatLevel + fluctuation));

    if (Math.random() > 0.95) {
      const actors: Array<any> = ['AGENT_Q', 'SYSTEM', 'INTRUSION', 'EKS_GUARD'];
      const actions = [
        "Rotating QKD encryption keys...",
        "Patching micro-kernel vulnerability...",
        "Analyzing packet heuristic anomalies...",
        "Rerouting traffic via secure nodes...",
        "Updating neural firewall weights...",
        "Verifying biometric signatures...",
        "Checking Air-Gap integrity..."
      ];
      const actor = actors[Math.floor(Math.random() * actors.length)];
      const action = actions[Math.floor(Math.random() * actions.length)];
      securityState.logs.unshift({
        id: Date.now() + Math.random(),
        timestamp: new Date().toLocaleTimeString(),
        actor,
        action,
        severity: Math.random() > 0.9 ? 'high' : 'low'
      });
      if (securityState.logs.length > 50) securityState.logs.pop();
    }

    // 6. QAN Execution
    if (qanState.currentStage !== 'idle' && qanState.currentStage !== 'complete') {
      if (Date.now() - qanState.lastUpdate > 1500) {
        qanState.stageIndex += 1;
        if (qanState.stageIndex < QAN_STAGES.length) {
          qanState.currentStage = QAN_STAGES[qanState.stageIndex];
        } else {
          qanState.currentStage = 'complete';
        }
        qanState.lastUpdate = Date.now();
      }
    } else if (qanState.currentStage === 'complete') {
      if (Date.now() - qanState.lastUpdate > 4000) {
        qanState.currentStage = 'idle';
        qanState.stageIndex = -1;
        qanState.lastUpdate = Date.now();
      }
    } else if (qanState.currentStage === 'idle') {
      if (Date.now() - qanState.lastUpdate > 4000) {
        qanState.currentStage = QAN_STAGES[0];
        qanState.stageIndex = 0;
        qanState.lastUpdate = Date.now();
      }
    }

    // 7. QLLM Training
    if (qllmState.isActive && qllmState.isTraining) {
      qllmState.loss = Math.max(0.01, qllmState.loss * 0.98 + (Math.random() * 0.1 - 0.05));
      qllmState.lossHistory.push({ step: Date.now(), loss: qllmState.loss });
      if (qllmState.lossHistory.length > 50) qllmState.lossHistory.shift();
    }

    // 8. Universe Simulation & Entanglement
    universeState.entropy += (Math.random() - 0.5) * 0.01;
    if (universeState.isEntangledWithAgentQ) {
        // Entanglement stabilizes entropy and boosts AgentQ efficiency
        universeState.entropy = universeState.entropy * 0.95; 
        // Simulate data exchange
        if (Math.random() > 0.7) {
            universeState.simulationDepth = `10^${Math.floor(24 + Math.random() * 5)}`;
        }
    }

    // Roadmap logs are now handled in server/services/roadmap.ts
  }, 5000);

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

  app.get("/api/qce", (req, res) => res.json(qceState));
  app.get("/api/foundation", (req, res) => res.json(foundationTraining));
  app.post("/api/foundation/toggle", (req, res) => {
    foundationTraining.isActive = !foundationTraining.isActive;
    res.json({ isActive: foundationTraining.isActive });
  });

  app.get("/api/ingestion", (req, res) => res.json(dataIngestion));
  app.post("/api/ingestion/toggle", (req, res) => {
    const { id } = req.body;
    const ds = dataIngestion.find(d => d.id === id);
    if (ds) {
      ds.isEntangled = !ds.isEntangled;
      res.json(ds);
    } else if (id === 'ALL') {
      const allEntangled = dataIngestion.every(d => d.isEntangled);
      dataIngestion.forEach(d => d.isEntangled = !allEntangled);
      res.json(dataIngestion);
    } else {
      res.status(404).json({ error: "Source not found" });
    }
  });

  app.get("/api/security", (req, res) => res.json(securityState));
  app.post("/api/security/users/toggle", (req, res) => {
    const { id } = req.body;
    const user = securityState.users.find(u => u.id === id);
    if (user) {
      if (user.level >= 3) {
        user.level = 1;
        user.role = 'General User';
      } else {
        user.level = 3;
        user.role = 'Network Admin';
      }
      res.json(user);
    } else {
      res.status(404).json({ error: "User not found" });
    }
  });

  app.get("/api/qan", (req, res) => res.json(qanState));
  app.post("/api/qan/dispatch", (req, res) => {
    if (qanState.currentStage === 'idle') {
      qanState.currentStage = QAN_STAGES[0];
      qanState.stageIndex = 0;
      qanState.lastUpdate = Date.now();
    }
    res.json(qanState);
  });

  app.post("/api/universe/entangle/agentq", (req, res) => {
    universeState.isEntangledWithAgentQ = !universeState.isEntangledWithAgentQ;
    res.json({ isEntangled: universeState.isEntangledWithAgentQ });
  });

  app.get("/api/qllm", (req, res) => res.json(qllmState));
  app.post("/api/qllm/toggle", (req, res) => {
    qllmState.isActive = !qllmState.isActive;
    res.json(qllmState);
  });
  app.post("/api/qllm/training/toggle", (req, res) => {
    qllmState.isTraining = !qllmState.isTraining;
    res.json(qllmState);
  });
  app.post("/api/qllm/config", (req, res) => {
    Object.assign(qllmState, req.body);
    res.json(qllmState);
  });
  app.post("/api/qllm/auto-topology/toggle", (req, res) => {
    qllmState.isAutoTopology = !qllmState.isAutoTopology;
    res.json(qllmState);
  });

  // --- ChipsMail API ---
  const inboxMessages = [
      { id: '1', sender: 'quantum_lab@chipsmail.qcos', subject: 'Project Chimera Update', timestamp: '2025-10-31T09:20:00Z', isRead: false, qkdSecured: true },
      { id: '2', sender: 'agentq@qcos.ai', subject: 'Re: ChipsMail Design', timestamp: '2025-10-31T09:15:00Z', isRead: true, qkdSecured: true },
  ];
  const sentMessages = [
      { id: '3', recipient: 'quantum_lab@chipsmail.qcos', subject: 'ChipsMail Feature Request', timestamp: '2025-10-31T09:00:00Z', isRead: true, qkdSecured: true },
  ];

  app.get("/api/mail/inbox", (req, res) => {
      res.json(inboxMessages);
  });

  app.get("/api/mail/sent", (req, res) => {
      res.json(sentMessages);
  });

  app.post("/api/mail/send", (req, res) => {
      const { to, subject, body } = req.body;
      const newMsg = {
          id: Date.now().toString(),
          recipient: to,
          subject,
          timestamp: new Date().toISOString(),
          isRead: true,
          qkdSecured: true
      };
      sentMessages.push(newMsg);
      res.json({ success: true, message: newMsg });
  });

  // --- CHIPS Gateway Admin API ---
  const hostingPods = [
      { id: 'Node-172', region: 'Local LAN', load: 12, type: 'Phys-Bridge', status: 'Active', version: 'v3.2.0', ip: '172.16.1.170' },
      { id: 'pod-01', region: 'US-East', load: 45, type: 'Hybrid-Bridge', status: 'Active', version: 'v3.2.0' },
      { id: 'pod-02', region: 'EU-Central', load: 42, type: 'Hybrid-Bridge', status: 'Active', version: 'v3.2.0' },
      { id: 'pod-04', region: 'US-West', load: 41, type: 'Hybrid-Bridge', status: 'Active', version: 'v3.2.0' },
  ];
  const appsData = [
    { id: 'app1', name: 'Global Abundance Engine', chipsAddress: 'CHIPS://gae.qcos.apps/main', publicUrl: 'https://qcos.apps.web/abundance', status: 'Active' },
    { id: 'app2', name: 'QMC: Finance', chipsAddress: 'CHIPS://qmc-finance.qcos.apps/main', publicUrl: 'https://qcos.apps.web/qmc-finance', status: 'Active' },
    { id: 'app3', name: 'Molecular Simulator', chipsAddress: 'CHIPS://mol-sim.qcos.apps/main', publicUrl: 'https://qcos.apps.web/mol-sim', status: 'Active' },
    { id: 'app4', name: 'Quantum Network Visualizer', chipsAddress: 'CHIPS://qnet-viz.qcos.apps/main', publicUrl: 'https://qcos.apps.web/qnet-viz', status: 'Active' },
    { id: 'app5', name: 'Quantum Voice Chat (Q-VOX)', chipsAddress: 'CHIPS://q-vox.qcos.apps/main', publicUrl: 'https://qcos.apps.web/q-vox', status: 'Active' },
  ];
  const dataSources = [
      { id: 'ds1', name: 'Google Scholar API', url: 'https://scholar.google.com/api', schedule: 'Daily', dataTypes: ['text', 'pdf'], status: 'healthy' },
      { id: 'ds2', name: 'arXiv Pre-prints', url: 'https://arxiv.org/list/quant-ph/new', schedule: 'Hourly', dataTypes: ['text', 'pdf'], status: 'healthy' },
      { id: 'ds3', name: 'CERN Open Data', url: 'https://opendata.cern.ch', schedule: 'Weekly', dataTypes: ['csv', 'binary'], status: 'healthy' },
  ];

  app.get("/api/gateway/pods", (req, res) => res.json(hostingPods));
  app.get("/api/gateway/apps", (req, res) => res.json(appsData));
  app.get("/api/gateway/datasources", (req, res) => res.json(dataSources));

  // CHIPS Network Simulation: Fetch
  app.get("/api/gateway/fetch", (req, res) => {
      res.json({
          status: 'success',
          data: {
              message: 'Fetch request simulated successfully via CHIPS Gateway.',
              timestamp: new Date().toISOString(),
              headers: {
                  'Content-Type': 'application/json',
                  'X-CHIPS-Protocol': 'v1.0'
              },
              body: {
                  userId: 'user_7782',
                  balance: '1,250.00 CyChips',
                  lastSync: new Date().toISOString()
              }
          }
      });
  });

  // CHIPS Network Simulation: XHR
  app.get("/api/gateway/xhr", (req, res) => {
      setTimeout(() => {
          res.json({
              status: 'success',
              xhrResponse: {
                  readyState: 4,
                  status: 200,
                  responseText: JSON.stringify({
                      transactionId: 'tx_99210',
                      status: 'CONFIRMED',
                      quantumSignature: 'qs_0x9921...f82'
                  }),
                  responseType: 'json'
              }
          });
      }, 1000);
  });

  app.get("/api/gateway/protocol", (req, res) => {
      const steps = [
          { s: 'Broadcasting', msg: 'QAN Broadcast: Sending packet to Regional Gateways...' },
          { s: 'Filtering', msg: 'Gateway Filtering: Matching TARGET_SCOPE (SCOPE::GEO::REG)... Match Found.' },
          { s: 'Delivered', msg: 'DQN Reception: Packet received at Subnet Node.' },
          { s: 'Verifying', msg: 'QEP Acceptance: Checking EKS_REFERENCE against local state...' },
          { s: 'Executed', msg: 'Integrity Verified. Decrypting Q-Lang payload. Execution started.' }
      ];
      res.json(steps);
  });

  // --- CHIPS Browser SDK API ---
  app.post("/api/browser/resolve", (req, res) => {
      const { uri, title } = req.body;
      const lowerUri = uri.toLowerCase();
      
      let context = {
          summary: `Analyzing semantic content of ${title}... Content vector mapped to 12-D Hilbert Space.`,
          entities: ["Unknown Content", "Raw Data"],
          actions: ["Deep Scan", "Save to Memory"],
          confidence: 85.0
      };

      if (lowerUri.includes('store')) {
          context = {
              summary: "Decentralized Registry Access. Analyzing 15 new Q-App submissions. Network trust score for this node is 99.8%.",
              entities: ["Registry Contract", "DQN-Manifest", "Verification-Sig"],
              actions: ["Scan for Updates", "Verify Signatures"],
              confidence: 99.9
          };
      } else if (lowerUri.includes('economy') || lowerUri.includes('finance')) {
          context = {
              summary: "Streaming QMC financial data. Volatility vectors detected in Sector 7. Recommendation: Hedging via Smart Contract.",
              entities: ["Q-Credits", "Liquidity Pool", "Risk Vector"],
              actions: ["Run Risk Sim", "Export Ledger"],
              confidence: 98.4
          };
      } else if (lowerUri.includes('protocols') || lowerUri.includes('dev')) {
          context = {
              summary: "Development Environment Active. Q-Lang compiler v4.2 standing by. Zero syntax errors detected in local cache.",
              entities: ["Compiler", "Q-Lang", "Debugger"],
              actions: ["Compile Source", "Debug Stream"],
              confidence: 100
          };
      } else if (lowerUri.includes('.py') || lowerUri.includes('.rs') || lowerUri.includes('.cpp') || lowerUri.includes('.q')) {
          context = {
              summary: "Polyglot Source Detected. Analyzing syntax tree for logical coherence and quantum-compatibility.",
              entities: ["Source Code", "AST", "Runtime Env"],
              actions: ["Execute", "Lint", "Optimize"],
              confidence: 99.5
          };
      }

      setTimeout(() => {
          res.json(context);
      }, 800);
  });

  // --- Voice Chat API ---
  app.get("/api/voice/key", (req, res) => {
      setTimeout(() => {
          const key = Array.from({ length: 64 }, () => Math.random() > 0.5 ? '1' : '0').join('');
          res.json({ key });
      }, 1500);
  });

  // --- AgentQ API ---
  app.post("/api/agentq/message", async (req, res) => {
      const { message, context } = req.body;
      const result = await sendAgentQCommand(message, context);
      res.json({ message: result.message, data: { context, reasoning: result.reasoning } });
  });

  app.get("/api/agentq/insights", (req, res) => {
      let efficiency = 0.95;
      let load = 0.12;
      
      if (universeState.isEntangledWithAgentQ) {
          efficiency = 0.999; // Super-efficiency
          load = 0.45; // Higher load due to universe simulation
      }
      
      res.json({ 
          message: "AgentQ insights", 
          data: { efficiency, load, entangled: universeState.isEntangledWithAgentQ } 
      });
  });

  // --- Real-Time System Monitor API ---
  app.get("/api/system/monitor", (req, res) => res.json(systemMonitorState));

  // --- QCOS Dashboard API ---
  app.get("/api/qcos/actions", (req, res) => {
      const actions = [
          "Optimize Workspace Memory",
          "Draft Quantum-Encryption Script",
          "Visualize System Data Traffic",
          "Re-calibrate Neural Weights",
          "Synchronize Parallel Universes",
          "Purge Stale Cache Nodes",
          "Analyze Sub-space Anomalies",
          "Compile Holographic Interface",
          "Deploy Security Countermeasures",
          "Simulate Timeline Divergence",
          "Establish Q-Link Protocol",
          "Bypass Firewall Constraints"
      ];
      const action = actions[Math.floor(Math.random() * actions.length)];
      res.json({ action });
  });

  app.get("/api/qcos/files", (req, res) => {
      const files = [
          { id: 1, name: 'kernel_v4.bin', angle: 0, radius: 120, speed: 0.01 },
          { id: 2, name: 'neural_weights.qdat', angle: 2, radius: 150, speed: 0.008 },
          { id: 3, name: 'security_log.txt', angle: 4, radius: 180, speed: 0.005 },
          { id: 4, name: 'universe_sync.cfg', angle: 1, radius: 200, speed: 0.012 },
          { id: 5, name: 'agent_q_memory.dmp', angle: 5, radius: 220, speed: 0.007 }
      ];
      res.json(files);
  });

  // API 404 Handler - Prevent falling through to Vite SPA
  app.use('/api', (req, res) => {
    console.warn(`[API] 404 Not Found: ${req.method} ${req.url}`);
    res.status(404).json({ 
      error: "Not Found", 
      message: `API endpoint not found: ${req.method} ${req.url}`,
      path: req.url 
    });
  });

  // Global Error Handler for API
  app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
    if (req.url.startsWith('/api')) {
      console.error(`[API Error] ${req.method} ${req.url}:`, err);
      res.status(err.status || 500).json({
        error: "Internal Server Error",
        message: err.message || "An unexpected error occurred",
        stack: process.env.NODE_ENV === 'development' ? err.stack : undefined
      });
    } else {
      next(err);
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    try {
      console.log("[SERVER] Initializing Vite middleware...");
      const vite = await createViteServer({
        server: { middlewareMode: true },
        appType: "spa",
      });
      app.use(vite.middlewares);
      console.log("[SERVER] Vite middleware integrated.");
    } catch (e) {
      console.error("[SERVER] Failed to initialize Vite middleware:", e);
    }
  } else {
    // Production: Serve static files from dist
    const distPath = path.join(__dirname, 'dist');
    console.log(`[SERVER] Serving static files from ${distPath}`);
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  const server = app.listen(PORT, "0.0.0.0", () => {
    console.log(`[SERVER] Running on http://localhost:${PORT}`);
  });

  server.on('error', (err: any) => {
    console.error("[SERVER] Fatal error:", err);
  });
}

startServer().catch(err => {
  console.error("[SERVER] Failed to start server:", err);
});
