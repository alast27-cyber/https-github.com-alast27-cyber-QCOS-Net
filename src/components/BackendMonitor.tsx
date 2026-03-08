import React, { useState, useEffect } from 'react';
import { 
    ActivityIcon, ServerStackIcon as ServerIcon, ShieldCheckIcon, DatabaseIcon, 
    CpuChipIcon as CpuIcon, NetworkIcon, RefreshCwIcon, AlertTriangleIcon,
    CheckCircle2Icon, UsersIcon, LayersIcon, ZapIcon
} from './Icons';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, BarChart, Bar, Cell } from 'recharts';

const BackendMonitor: React.FC = () => {
    const [roadmap, setRoadmap] = useState<any>(null);
    const [qce, setQce] = useState<any>(null);
    const [ingestion, setIngestion] = useState<any[]>([]);
    const [security, setSecurity] = useState<any>(null);
    const [pods, setPods] = useState<any[]>([]);
    const [insights, setInsights] = useState<any>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [lastUpdate, setLastUpdate] = useState(new Date());

    const [system, setSystem] = useState<any>(null);

    const fetchData = async () => {
        setIsLoading(true);
        try {
            const [roadmapRes, qceRes, ingestionRes, securityRes, podsRes, insightsRes, systemRes] = await Promise.all([
                fetch('/api/roadmap').then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); }),
                fetch('/api/qce').then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); }),
                fetch('/api/ingestion').then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); }),
                fetch('/api/security').then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); }),
                fetch('/api/gateway/pods').then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); }),
                fetch('/api/agentq/insights').then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); }),
                fetch('/api/system/monitor').then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); })
            ]);

            setRoadmap(roadmapRes);
            setQce(qceRes);
            setIngestion(ingestionRes);
            setSecurity(securityRes);
            setPods(podsRes);
            setInsights(insightsRes);
            setSystem(systemRes);
            setLastUpdate(new Date());
        } catch (error) {
            console.error("Failed to fetch backend data:", error);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 10000); // Refresh every 10s
        return () => clearInterval(interval);
    }, []);

    if (isLoading && !roadmap) {
        return (
            <div className="flex flex-col items-center justify-center h-full text-cyan-500 animate-pulse">
                <RefreshCwIcon className="w-12 h-12 animate-spin mb-4" />
                <p className="text-sm font-bold tracking-widest uppercase">Synchronizing with QCOS Backend...</p>
            </div>
        );
    }

    const activeStage = roadmap?.stages.find((s: any) => s.status === 'active');

    return (
        <div className="flex flex-col h-full gap-4 p-4 overflow-y-auto custom-scrollbar bg-black/40">
            {/* Header Status Bar */}
            <div className="flex flex-wrap justify-between items-center gap-4 bg-black/60 p-4 rounded-xl border border-cyan-900/50">
                <div className="flex items-center gap-3">
                    <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse shadow-[0_0_10px_green]"></div>
                    <div>
                        <h2 className="text-lg font-black text-white uppercase tracking-tighter">QCOS Backend Monitor</h2>
                        <p className="text-[10px] text-cyan-600 font-mono uppercase">Last Sync: {lastUpdate.toLocaleTimeString()}</p>
                    </div>
                </div>
                <div className="flex gap-6">
                    <div className="text-center">
                        <p className="text-[8px] text-gray-500 uppercase font-black">Threat Level</p>
                        <p className={`text-sm font-mono font-bold ${security?.threatLevel > 50 ? 'text-red-400' : 'text-green-400'}`}>
                            {security?.threatLevel.toFixed(1)}%
                        </p>
                    </div>
                    <div className="text-center">
                        <p className="text-[8px] text-gray-500 uppercase font-black">AgentQ Efficiency</p>
                        <p className="text-sm font-mono font-bold text-purple-400">
                            {(insights?.data?.efficiency * 100).toFixed(1)}%
                        </p>
                    </div>
                    <div className="text-center">
                        <p className="text-[8px] text-gray-500 uppercase font-black">Active Pods</p>
                        <p className="text-sm font-mono font-bold text-cyan-400">{pods.length}</p>
                    </div>
                </div>
                <button 
                    onClick={fetchData}
                    className="p-2 bg-cyan-900/30 border border-cyan-800 rounded-lg text-cyan-400 hover:bg-cyan-800/50 transition-all"
                >
                    <RefreshCwIcon className="w-4 h-4" />
                </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {/* Roadmap Progress */}
                <div className="lg:col-span-2 bg-black/60 p-4 rounded-xl border border-cyan-900/30 flex flex-col gap-4">
                    <div className="flex items-center justify-between border-b border-cyan-900/30 pb-2">
                        <h3 className="text-xs font-black text-cyan-300 flex items-center gap-2 uppercase tracking-widest">
                            <LayersIcon className="w-4 h-4" /> AGI Roadmap Progress
                        </h3>
                        <span className="text-[10px] text-cyan-600 font-mono">{roadmap?.currentTask}</span>
                    </div>
                    <div className="space-y-4">
                        {roadmap?.stages.map((stage: any) => (
                            <div key={stage.id} className="space-y-1">
                                <div className="flex justify-between text-[10px]">
                                    <span className={`font-bold ${stage.status === 'active' ? 'text-cyan-400' : stage.status === 'completed' ? 'text-green-400' : 'text-gray-600'}`}>
                                        {stage.title}
                                    </span>
                                    <span className="text-gray-500">{stage.progress.toFixed(1)}%</span>
                                </div>
                                <div className="h-1.5 w-full bg-gray-900 rounded-full overflow-hidden">
                                    <div 
                                        className={`h-full transition-all duration-1000 ${stage.status === 'active' ? 'bg-cyan-500 shadow-[0_0_10px_cyan]' : stage.status === 'completed' ? 'bg-green-500' : 'bg-gray-800'}`}
                                        style={{ width: `${stage.progress}%` }}
                                    ></div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* QCE Evolution */}
                <div className="bg-black/60 p-4 rounded-xl border border-purple-900/30 flex flex-col gap-4">
                    <h3 className="text-xs font-black text-purple-300 flex items-center gap-2 uppercase tracking-widest border-b border-purple-900/30 pb-2">
                        <ZapIcon className="w-4 h-4" /> Cognitive Evolution
                    </h3>
                    <div className="flex-grow">
                        <ResponsiveContainer width="100%" height={180}>
                            <BarChart data={Object.entries(qce?.evolutionProgress || {}).map(([name, value]) => ({ name, value }))}>
                                <XAxis dataKey="name" stroke="#6b7280" fontSize={10} tickLine={false} axisLine={false} />
                                <Tooltip 
                                    contentStyle={{ backgroundColor: '#000', border: '1px solid #581c87', fontSize: '10px' }}
                                    itemStyle={{ color: '#a855f7' }}
                                />
                                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                                    {Object.entries(qce?.evolutionProgress || {}).map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={index % 2 === 0 ? '#a855f7' : '#7c3aed'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="grid grid-cols-5 gap-1">
                        {Object.entries(qce?.currentStage || {}).map(([name, stage]: [string, any]) => (
                            <div key={name} className="text-center">
                                <p className="text-[8px] text-gray-600 uppercase">{name}</p>
                                <p className="text-[10px] font-bold text-purple-400">S-{stage}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* Data Ingestion Streams */}
                <div className="bg-black/60 p-4 rounded-xl border border-cyan-900/30 flex flex-col gap-4">
                    <h3 className="text-xs font-black text-cyan-300 flex items-center gap-2 uppercase tracking-widest border-b border-cyan-900/30 pb-2">
                        <DatabaseIcon className="w-4 h-4" /> Data Ingestion Streams
                    </h3>
                    <div className="space-y-2 overflow-y-auto max-h-64 custom-scrollbar">
                        {ingestion.map((ds: any) => (
                            <div key={ds.id} className="flex items-center justify-between p-2 bg-black/40 border border-white/5 rounded-lg hover:border-cyan-500/30 transition-all">
                                <div className="flex items-center gap-3">
                                    <div className={`w-2 h-2 rounded-full ${ds.status === 'ACTIVE' ? 'bg-green-500 animate-pulse' : 'bg-gray-600'}`}></div>
                                    <div>
                                        <p className="text-[10px] font-bold text-white">{ds.name}</p>
                                        <p className="text-[8px] text-gray-500 uppercase">{ds.type} • {ds.latency.toFixed(1)}ms</p>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <p className="text-[10px] font-mono text-cyan-400">{ds.throughput.toFixed(1)} MB/s</p>
                                    <p className="text-[8px] text-gray-600">FIDELITY: {ds.fidelity.toFixed(2)}%</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Gateway Pods */}
                <div className="bg-black/60 p-4 rounded-xl border border-cyan-900/30 flex flex-col gap-4">
                    <h3 className="text-xs font-black text-cyan-300 flex items-center gap-2 uppercase tracking-widest border-b border-cyan-900/30 pb-2">
                        <ServerIcon className="w-4 h-4" /> Gateway Infrastructure
                    </h3>
                    <div className="grid grid-cols-2 gap-3">
                        {pods.map((pod: any) => (
                            <div key={pod.id} className="p-3 bg-black/40 border border-white/5 rounded-lg flex flex-col gap-2">
                                <div className="flex justify-between items-start">
                                    <span className="text-[10px] font-black text-white uppercase">{pod.id}</span>
                                    <span className="text-[8px] px-1.5 rounded-full border border-green-500 text-green-400 uppercase">{pod.status}</span>
                                </div>
                                <div className="flex justify-between text-[8px] text-gray-500">
                                    <span>{pod.region}</span>
                                    <span>{pod.version}</span>
                                </div>
                                <div className="space-y-1">
                                    <div className="flex justify-between text-[8px]">
                                        <span className="text-gray-600">LOAD</span>
                                        <span className="text-cyan-400">{pod.load}%</span>
                                    </div>
                                    <div className="h-1 w-full bg-gray-900 rounded-full overflow-hidden">
                                        <div className="h-full bg-cyan-500" style={{ width: `${pod.load}%` }}></div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* System Monitor */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="bg-black/60 p-4 rounded-xl border border-cyan-900/30 flex flex-col gap-4">
                    <h3 className="text-xs font-black text-cyan-300 flex items-center gap-2 uppercase tracking-widest border-b border-cyan-900/30 pb-2">
                        <CpuIcon className="w-4 h-4" /> System Resources
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                        <div className="bg-black/40 p-3 rounded-lg border border-white/5">
                            <p className="text-[10px] text-gray-500 uppercase font-bold">CPU Usage</p>
                            <div className="flex items-end gap-2">
                                <span className="text-2xl font-mono text-cyan-400">{system?.cpu?.usage.toFixed(1)}%</span>
                                <span className="text-[10px] text-gray-600 mb-1">{system?.cpu?.cores.length} Cores</span>
                            </div>
                            <div className="h-1.5 w-full bg-gray-900 rounded-full overflow-hidden mt-2">
                                <div className="h-full bg-cyan-500 transition-all duration-500" style={{ width: `${system?.cpu?.usage}%` }}></div>
                            </div>
                        </div>
                        <div className="bg-black/40 p-3 rounded-lg border border-white/5">
                            <p className="text-[10px] text-gray-500 uppercase font-bold">Memory Usage</p>
                            <div className="flex items-end gap-2">
                                <span className="text-2xl font-mono text-purple-400">{(system?.memory?.used / 1024).toFixed(1)}GB</span>
                                <span className="text-[10px] text-gray-600 mb-1">/ {(system?.memory?.total / 1024).toFixed(1)}GB</span>
                            </div>
                            <div className="h-1.5 w-full bg-gray-900 rounded-full overflow-hidden mt-2">
                                <div className="h-full bg-purple-500 transition-all duration-500" style={{ width: `${(system?.memory?.used / system?.memory?.total) * 100}%` }}></div>
                            </div>
                        </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        <div className="bg-black/40 p-3 rounded-lg border border-white/5">
                            <p className="text-[10px] text-gray-500 uppercase font-bold">Network I/O</p>
                            <div className="flex justify-between items-center mt-1">
                                <span className="text-xs text-green-400">RX: {system?.network?.rx.toFixed(1)} MB/s</span>
                                <span className="text-xs text-blue-400">TX: {system?.network?.tx.toFixed(1)} MB/s</span>
                            </div>
                        </div>
                        <div className="bg-black/40 p-3 rounded-lg border border-white/5">
                            <p className="text-[10px] text-gray-500 uppercase font-bold">System Uptime</p>
                            <p className="text-lg font-mono text-white mt-1">{new Date(system?.uptime * 1000).toISOString().substr(11, 8)}</p>
                        </div>
                    </div>
                    <div className="grid grid-cols-3 gap-4">
                        <div className="bg-black/40 p-3 rounded-lg border border-white/5">
                            <p className="text-[10px] text-gray-500 uppercase font-bold">API Requests</p>
                            <p className="text-lg font-mono text-cyan-400 mt-1">{system?.api?.requests || 0}</p>
                        </div>
                        <div className="bg-black/40 p-3 rounded-lg border border-white/5">
                            <p className="text-[10px] text-gray-500 uppercase font-bold">Avg Latency</p>
                            <p className="text-lg font-mono text-yellow-400 mt-1">{(system?.api?.avgLatency || 0).toFixed(1)}ms</p>
                        </div>
                        <div className="bg-black/40 p-3 rounded-lg border border-white/5">
                            <p className="text-[10px] text-gray-500 uppercase font-bold">Errors</p>
                            <p className={`text-lg font-mono mt-1 ${system?.api?.errors > 0 ? 'text-red-500' : 'text-green-500'}`}>{system?.api?.errors || 0}</p>
                        </div>
                    </div>
                </div>

                <div className="bg-black/60 p-4 rounded-xl border border-cyan-900/30 flex flex-col gap-4">
                    <h3 className="text-xs font-black text-cyan-300 flex items-center gap-2 uppercase tracking-widest border-b border-cyan-900/30 pb-2">
                        <ActivityIcon className="w-4 h-4" /> Active Processes
                    </h3>
                    <div className="overflow-y-auto max-h-48 custom-scrollbar space-y-1">
                        <div className="grid grid-cols-4 text-[8px] text-gray-500 uppercase font-bold px-2 mb-1">
                            <span>PID</span>
                            <span>Name</span>
                            <span>CPU%</span>
                            <span>Status</span>
                        </div>
                        {system?.processes.map((proc: any) => (
                            <div key={proc.pid} className="grid grid-cols-4 text-[10px] p-2 bg-black/40 border border-white/5 rounded hover:bg-white/5 transition-colors">
                                <span className="font-mono text-gray-400">{proc.pid}</span>
                                <span className="text-cyan-300 font-bold">{proc.name}</span>
                                <span className="font-mono text-white">{proc.cpu}%</span>
                                <span className={`text-[8px] uppercase ${proc.status === 'Running' ? 'text-green-400' : 'text-gray-500'}`}>{proc.status}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Security Logs */}
            <div className="bg-black/60 p-4 rounded-xl border border-red-900/30 flex flex-col gap-4">
                <h3 className="text-xs font-black text-red-300 flex items-center gap-2 uppercase tracking-widest border-b border-red-900/30 pb-2">
                    <ShieldCheckIcon className="w-4 h-4" /> Security Event Log
                </h3>
                <div className="font-mono text-[9px] space-y-1 overflow-y-auto max-h-48 custom-scrollbar">
                    {security?.logs.map((log: any) => (
                        <div key={log.id} className={`flex gap-3 p-1 border-b border-white/5 ${log.severity === 'high' ? 'bg-red-950/20 text-red-200' : 'text-gray-400'}`}>
                            <span className="text-gray-600 flex-shrink-0">[{log.timestamp}]</span>
                            <span className={`font-bold flex-shrink-0 w-24 ${log.severity === 'high' ? 'text-red-400' : 'text-cyan-600'}`}>{log.actor}</span>
                            <span className="flex-grow">{log.action}</span>
                            {log.severity === 'high' && <AlertTriangleIcon className="w-3 h-3 text-red-500 animate-pulse" />}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default BackendMonitor;
