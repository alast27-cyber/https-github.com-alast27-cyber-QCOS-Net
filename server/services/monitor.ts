
export interface SystemMonitorState {
    cpu: { usage: number; cores: number[] };
    memory: { total: number; used: number; free: number };
    network: { rx: number; tx: number; packets: number };
    processes: { pid: number; name: string; cpu: string; mem: string; status: string }[];
    disk: { total: number; used: number; free: number };
    uptime: number;
    api: { requests: number; errors: number; avgLatency: number };
}

export const systemMonitorState: SystemMonitorState = {
    cpu: { usage: 0, cores: [0, 0, 0, 0, 0, 0, 0, 0] },
    memory: { total: 32768, used: 0, free: 0 },
    network: { rx: 0, tx: 0, packets: 0 },
    processes: [],
    disk: { total: 1024, used: 0, free: 0 },
    uptime: 0,
    api: { requests: 0, errors: 0, avgLatency: 0 }
};

export function startSystemMonitor() {
    setInterval(() => {
        // CPU Simulation
        systemMonitorState.cpu.usage = 20 + Math.random() * 30;
        systemMonitorState.cpu.cores = systemMonitorState.cpu.cores.map(() => 10 + Math.random() * 50);

        // Memory Simulation
        systemMonitorState.memory.used = 12000 + Math.random() * 4000;
        systemMonitorState.memory.free = systemMonitorState.memory.total - systemMonitorState.memory.used;

        // Network Simulation
        systemMonitorState.network.rx = Math.random() * 50; // MB/s
        systemMonitorState.network.tx = Math.random() * 30; // MB/s
        systemMonitorState.network.packets = Math.floor(Math.random() * 1000);

        // Disk Simulation
        systemMonitorState.disk.used = 450 + Math.random() * 2;
        systemMonitorState.disk.free = systemMonitorState.disk.total - systemMonitorState.disk.used;

        // Uptime
        systemMonitorState.uptime += 1;

        // Process Simulation
        const processNames = ['node', 'python', 'docker', 'nginx', 'postgres', 'redis', 'agent_q_core', 'qcos_kernel'];
        systemMonitorState.processes = processNames.map((name, i) => ({
            pid: 1000 + i,
            name,
            cpu: (Math.random() * 5).toFixed(1),
            mem: (Math.random() * 200).toFixed(0),
            status: Math.random() > 0.05 ? 'Running' : 'Sleeping'
        }));

        // Reset API metrics periodically (simulated reset for demo)
        if (systemMonitorState.uptime % 60 === 0) {
            systemMonitorState.api.requests = 0;
            systemMonitorState.api.errors = 0;
        }

    }, 1000);
}

export function trackRequest(duration: number, isError: boolean) {
    systemMonitorState.api.requests++;
    if (isError) systemMonitorState.api.errors++;
    
    // Simple moving average for latency
    const currentAvg = systemMonitorState.api.avgLatency;
    const count = systemMonitorState.api.requests;
    systemMonitorState.api.avgLatency = currentAvg + (duration - currentAvg) / count;
}
