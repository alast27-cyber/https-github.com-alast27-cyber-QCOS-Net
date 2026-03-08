import express from "express";
import cors from "cors";
import { createServer as createViteServer } from "vite";

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(cors());
  app.use(express.json());

  // Request Logging Middleware
  app.use((req, res, next) => {
    if (req.url.startsWith('/api')) {
      console.log(`[API] ${req.method} ${req.url}`);
    }
    next();
  });

  // --- AGI Roadmap State ---
  const INITIAL_ROADMAP_STAGES = [
    {
        id: 'phase-1',
        title: 'Phase 1: Foundation & MoE Architecture',
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
    }
  ];

  const roadmapState = {
    stages: [...INITIAL_ROADMAP_STAGES],
    isTraining: true,
    logs: [] as any[],
    currentTask: 'Initializing Training Protocols...'
  };

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
    // 1. Roadmap Training
    if (roadmapState.isTraining) {
      const activeIndex = roadmapState.stages.findIndex(s => s.status === 'active');
      if (activeIndex !== -1) {
        const activeStage = roadmapState.stages[activeIndex];
        const increment = 0.5 + Math.random() * 1.5;
        activeStage.progress = Math.min(100, activeStage.progress + increment);

        if (activeStage.progress >= 100) {
          activeStage.status = 'completed';
          const nextIndex = activeIndex + 1;
          if (nextIndex < roadmapState.stages.length) {
            roadmapState.stages[nextIndex].status = 'active';
            roadmapState.currentTask = `Starting ${roadmapState.stages[nextIndex].title}...`;
          } else {
            roadmapState.isTraining = false;
            roadmapState.currentTask = 'AGI Training Complete.';
          }
        } else {
          roadmapState.currentTask = `Training ${activeStage.title}: ${activeStage.progress.toFixed(1)}%`;
        }
      }
    }

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

    // Generate a "genuine" log using local heuristic every few steps
    if (Math.random() > 0.8) {
      const activeIndex = roadmapState.stages.findIndex(s => s.status === 'active');
      const activeStage = activeIndex !== -1 ? roadmapState.stages[activeIndex] : null;
      if (activeStage) {
        const technicalTerms = ["Gradient", "Tensor", "Lattice", "Heuristic", "Entropy", "Weights", "Topology", "Vector", "Hilbert", "Qubit", "Entanglement", "Backprop", "Transformer", "MoE", "Expert"];
        const actions = ["Optimizing", "Refining", "Converging", "Stabilizing", "Calibrating", "Initializing", "Validating", "Synchronizing", "Pruning", "Normalizing"];
        const components = ["Cluster-7", "Node-Alpha", "Core-Lattice", "Semantic-Buffer", "Logic-Gate", "Neural-Fabric", "Memory-Array", "Compute-Grid"];
        
        const term = technicalTerms[Math.floor(Math.random() * technicalTerms.length)];
        const action = actions[Math.floor(Math.random() * actions.length)];
        const component = components[Math.floor(Math.random() * components.length)];
        
        const logMsg = `${action} ${term} on ${component} for ${activeStage.id.toUpperCase()}.`;
        
        roadmapState.logs.push({
          timestamp: Date.now(),
          message: logMsg,
          type: Math.random() > 0.9 ? 'warning' : 'info'
        });
        if (roadmapState.logs.length > 50) roadmapState.logs.shift();
      }
    }
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

  // --- QCOS Dashboard API ---
  const POSSIBLE_ACTIONS = [
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

  app.get("/api/qcos/actions", (req, res) => {
      const action = POSSIBLE_ACTIONS[Math.floor(Math.random() * POSSIBLE_ACTIONS.length)];
      res.json({ action });
  });

  app.get("/api/qcos/files", (req, res) => {
      const files = [
          'quantum_architecture.q',
          'qiai_ips_core.rs',
          'grand_universe_sim.py',
          'qcos_kernel.ts',
          'neural_link_config.json'
      ];
      
      const orbiters = files.map((name, i) => ({
          id: i,
          name,
          angle: Math.random() * Math.PI * 2,
          radius: 80 + Math.random() * 60,
          speed: 0.005 + Math.random() * 0.01
      }));
      res.json(orbiters);
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

  app.get("/api/gateway/xhr", (req, res) => {
      setTimeout(() => {
          res.json({ data: "Stream buffer complete", size: "45kb" });
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
  app.post("/api/agentq/message", (req, res) => {
      const { message, context } = req.body;
      res.json({ message: `AgentQ processed: ${message}`, data: { context } });
  });

  app.get("/api/agentq/insights", (req, res) => {
      res.json({ message: "AgentQ insights", data: { efficiency: 0.95, load: 0.12 } });
  });

  // API 404 Handler - Prevent falling through to Vite SPA
  app.use('/api/*', (req, res) => {
    console.log(`[API] 404 Not Found: ${req.method} ${req.url}`);
    res.status(404).json({ error: `API endpoint not found: ${req.method} ${req.url}` });
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
