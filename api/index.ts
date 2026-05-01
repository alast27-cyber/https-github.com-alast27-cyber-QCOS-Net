import express from 'express';
import cors from 'cors';
import { sendAgentQCommand } from '../server/services/ollama';
import { systemMonitorState } from '../server/services/monitor';

const app = express();

app.use(cors());
app.use(express.json());

// Minimal state needed for standalone
const universeState = {
  isEntangledWithAgentQ: false,
  entropy: 0.5,
  sector: '7G',
  simulationDepth: '10^24'
};

app.post("/api/agentq/message", async (req, res) => {
    try {
        const { message, context } = req.body;
        const result = await sendAgentQCommand(message, context);
        res.json({ message: result.message, data: { context, reasoning: result.reasoning } });
    } catch (e: any) {
        console.error("AgentQ message error:", e);
        res.status(500).json({ error: "Failed to process message" });
    }
});

app.get("/api/agentq/insights", (req, res) => {
    let efficiency = 0.95;
    let load = 0.12;
    if (universeState.isEntangledWithAgentQ) {
        efficiency = 0.999;
        load = 0.45;
    }
    res.json({ 
        message: "AgentQ insights", 
        data: { efficiency, load, entangled: universeState.isEntangledWithAgentQ } 
    });
});

app.get("/api/system/monitor", (req, res) => {
    res.json(systemMonitorState);
});

export default app;
