
import React, { useState, useEffect, useRef } from 'react';
import GlassPanel from './GlassPanel';
import { 
    ShieldCheckIcon, LockIcon, ZapIcon, ActivityIcon, 
    AlertTriangleIcon, CheckCircle2Icon, RefreshCwIcon, 
    ServerCogIcon, EyeIcon, PlayIcon, StopIcon, BugAntIcon,
    CpuChipIcon, GlobeIcon, LoaderIcon, FastForwardIcon,
    UsersIcon, KeyIcon, FileCodeIcon
} from './Icons';
import { ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend } from 'recharts';

// --- Types ---
interface SecurityLog {
    id: number;
    timestamp: string;
    actor: 'AGENT_Q' | 'SYSTEM' | 'INTRUSION' | 'EKS_GUARD';
    action: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
}

interface UserRecord {
    id: string;
    username: string;
    level: 1 | 2 | 3 | 4;
    role: string;
    lastActive: string;
    status: 'Active' | 'Locked' | 'Flagged';
}

const ATTACK_VECTORS = [
    'Quantum Decryption', 'Polymorphic Worm', 'DDoS Volumetric', 'Social Engineering', 'Supply Chain Injection', 'EKS Replay Attack', 'Key Entanglement Collapse'
];

const DEFENSE_PROTOCOLS = [
    'Adaptive Firewall', 'QKD-Entanglement', 'Neural Heuristics', 'Air-Gap Isolation', 'Active Counter-Strike', 'EKS-Rotation (Dilithium)'
];

interface SecurityMonitorProps {
    onMaximize?: () => void;
}

// --- User Registry Sub-Component ---
const UserRegistryView: React.FC = () => {
    const [users, setUsers] = useState<UserRecord[]>([
        { id: 'usr-001', username: 'sys_admin_prime', level: 4, role: 'System Architect', lastActive: 'Now', status: 'Active' },
        { id: 'usr-002', username: 'net_ops_lead', level: 3, role: 'Network Admin', lastActive: '5m ago', status: 'Active' },
        { id: 'usr-003', username: 'dev_gamma', level: 2, role: 'Frontend Dev', lastActive: '2h ago', status: 'Active' },
        { id: 'usr-004', username: 'dev_delta', level: 2, role: 'Backend Dev', lastActive: '1d ago', status: 'Flagged' },
        { id: 'usr-005', username: 'guest_user_12', level: 1, role: 'General User', lastActive: '10m ago', status: 'Active' },
    ]);

    const getAccessDescription = (level: number) => {
        switch (level) {
            case 4: return "FULL CAPABILITY: Read/Write/Execute on Core, Network, and User Mgmt.";
            case 3: return "NETWORK ADMIN: R/W Chips Network. Read-Only QCOS Dashboard.";
            case 2: return "DEVELOPER: Read-Only Chips Network. Sandbox Write.";
            case 1: return "GENERAL: App usage only. No System Access.";
            default: return "Unknown";
        }
    };

    const getLevelColor = (level: number) => {
        switch (level) {
            case 4: return "text-red-400 border-red-500 bg-red-900/20";
            case 3: return "text-orange-400 border-orange-500 bg-orange-900/20";
            case 2: return "text-yellow-400 border-yellow-500 bg-yellow-900/20";
            case 1: return "text-blue-400 border-blue-500 bg-blue-900/20";
            default: return "text-gray-400";
        }
    };

    const toggleAdminRights = (userId: string) => {
        setUsers(prev => prev.map(user => {
            if (user.id === userId) {
                // If Level 3 or 4, Revoke to Level 1
                if (user.level >= 3) {
                    return { ...user, level: 1, role: 'General User' };
                } else {
                    // Otherwise, Promote to Level 3 (Network Admin)
                    return { ...user, level: 3, role: 'Network Admin' };
                }
            }
            return user;
        }));
    };

    return (
        <div className="h-full flex flex-col gap-4 overflow-hidden animate-fade-in">
            <div className="flex gap-4">
                <div className="flex-1 bg-black/40 border border-cyan-900/50 rounded-lg p-3">
                    <h4 className="text-xs font-bold text-cyan-300 uppercase mb-2 flex items-center gap-2">
                        <UsersIcon className="w-4 h-4" /> Active Personnel
                    </h4>
                    <div className="text-2xl font-mono text-white">{users.length}</div>
                </div>
                <div className="flex-1 bg-black/40 border border-cyan-900/50 rounded-lg p-3">
                    <h4 className="text-xs font-bold text-red-400 uppercase mb-2 flex items-center gap-2">
                        <LockIcon className="w-4 h-4" /> Admin Nodes
                    </h4>
                    <div className="text-2xl font-mono text-white">{users.filter(u => u.level >= 3).length}</div>
                </div>
            </div>

            <div className="flex-grow bg-black/30 border border-cyan-800/30 rounded-lg overflow-hidden flex flex-col">
                <div className="p-2 bg-cyan-950/30 border-b border-cyan-900/50 grid grid-cols-12 gap-2 text-[9px] font-bold text-cyan-500 uppercase tracking-wider">
                    <div className="col-span-3">User ID</div>
                    <div className="col-span-2">Level</div>
                    <div className="col-span-3">Role</div>
                    <div className="col-span-2">Status</div>
                    <div className="col-span-2 text-right">Action</div>
                </div>
                <div className="flex-grow overflow-y-auto custom-scrollbar p-1">
                    {users.map(user => (
                        <div key={user.id} className="grid grid-cols-12 gap-2 items-center p-2 hover:bg-white/5 rounded transition-colors text-[10px] border-b border-white/5 last:border-0 group">
                            <div className="col-span-3 font-mono text-cyan-100 flex items-center gap-2">
                                <div className={`w-1.5 h-1.5 rounded-full ${user.status === 'Active' ? 'bg-green-500 shadow-[0_0_5px_#22c55e]' : 'bg-red-500'}`}></div>
                                {user.username}
                            </div>
                            <div className="col-span-2">
                                <span className={`px-1.5 py-0.5 rounded border text-[9px] font-bold ${getLevelColor(user.level)}`}>
                                    LVL {user.level}
                                </span>
                            </div>
                            <div className="col-span-3 text-gray-300">{user.role}</div>
                            <div className="col-span-2 text-gray-400">{user.status}</div>
                            <div className="col-span-2 text-right opacity-0 group-hover:opacity-100 transition-opacity">
                                <button 
                                    onClick={() => toggleAdminRights(user.id)}
                                    className={`px-2 py-1 rounded text-[9px] font-bold border transition-colors ${
                                        user.level >= 3 
                                        ? 'text-red-300 border-red-500/50 hover:bg-red-900/40 hover:text-white' 
                                        : 'text-green-300 border-green-500/50 hover:bg-green-900/40 hover:text-white'
                                    }`}
                                >
                                    {user.level >= 3 ? 'Revoke Admin' : 'Grant Admin'}
                                </button>
                            </div>
                            
                            {/* Permission Tooltip (Visual only for layout) */}
                            <div className="col-span-12 mt-1 pl-4 text-[9px] text-gray-500 italic border-l-2 border-gray-700 hidden group-hover:block animate-fade-in">
                                ↳ {getAccessDescription(user.level)}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="p-2 bg-yellow-900/10 border border-yellow-700/30 rounded text-[9px] text-yellow-500/80 flex items-start gap-2">
                <AlertTriangleIcon className="w-3 h-3 flex-shrink-0 mt-0.5" />
                <p>Warning: Modifying Level 4 (System Admin) permissions requires multi-sig authentication from the QCOS Kernel.</p>
            </div>
        </div>
    );
};

const SecurityMonitorAndSimulator: React.FC<SecurityMonitorProps> = ({ onMaximize }) => {
    // --- View State ---
    const [activeView, setActiveView] = useState<'monitor' | 'registry'>('monitor');

    // --- Monitor State ---
    const [threatLevel, setThreatLevel] = useState(0); // 0-100
    const [logs, setLogs] = useState<SecurityLog[]>([]);
    const [activeProtocols, setActiveProtocols] = useState<string[]>(['QKD-Entanglement', 'Neural Heuristics', 'EKS-V2', 'AIR-GAP']);
    
    // --- Simulator State ---
    const [selectedAttack, setSelectedAttack] = useState(ATTACK_VECTORS[0]);
    const [selectedDefense, setSelectedDefense] = useState(DEFENSE_PROTOCOLS[0]);
    const [isSimulating, setIsSimulating] = useState(false);
    const [isAutoSimulating, setIsAutoSimulating] = useState(false);
    const [simProgress, setSimProgress] = useState(0);
    const [simResult, setSimResult] = useState<{ success: boolean; score: number; notes: string } | null>(null);
    const [humanOverride, setHumanOverride] = useState(false);

    // --- Monitor Logic (AI Agent Simulation) ---
    useEffect(() => {
        const interval = setInterval(() => {
            // Randomly fluctuate threat level
            const fluctuation = Math.random() > 0.8 ? (Math.random() * 10) : -(Math.random() * 5);
            setThreatLevel(prev => Math.max(0, Math.min(100, prev + fluctuation)));

            // EKS Health Check (New from PDF)
            if (Math.random() > 0.85) {
                 addLog('EKS_GUARD', 'Scanning channel for eavesdroppers (Eve)...', 'low');
                 setTimeout(() => {
                     addLog('EKS_GUARD', 'INTEGRITY_HASH Verified. Project Files Secure.', 'low');
                 }, 800);
            }

            // Agent Q Actions
            if (Math.random() > 0.7) {
                const actions = [
                    "Rotating QKD encryption keys...",
                    "Patching micro-kernel vulnerability...",
                    "Analyzing packet heuristic anomalies...",
                    "Rerouting traffic via secure nodes...",
                    "Updating neural firewall weights...",
                    "Verifying biometric signatures...",
                    "Checking Air-Gap integrity..."
                ];
                const action = actions[Math.floor(Math.random() * actions.length)];
                addLog('AGENT_Q', action, 'low');
            }

            // Simulate Attack Detection
            if (Math.random() > 0.98) {
                addLog('INTRUSION', `Unauthorized access attempt on Sector ${Math.floor(Math.random() * 9)}`, 'high');
                setThreatLevel(prev => Math.min(100, prev + 25));
                setTimeout(() => {
                    addLog('AGENT_Q', 'Threat neutralized. Counter-measures deployed.', 'medium');
                    setThreatLevel(prev => Math.max(0, prev - 20));
                }, 2000);
            }

        }, 1500);
        return () => clearInterval(interval);
    }, []);

    // --- Auto-Simulation Loop Logic ---
    useEffect(() => {
        let timer: ReturnType<typeof setTimeout>;

        if (isAutoSimulating && !isSimulating) {
            timer = setTimeout(() => {
                // 1. Implement previous result if exists
                if (simResult) {
                    const impact = simResult.success ? -15 : 10;
                    setThreatLevel(prev => Math.max(0, Math.min(100, prev + impact)));
                    addLog('AGENT_Q', `Auto-Implementing: ${simResult.success ? 'Optimization Applied' : 'Patching Vulnerability'}`, simResult.success ? 'low' : 'high');
                }

                // 2. Setup next scenario
                const nextAttack = ATTACK_VECTORS[Math.floor(Math.random() * ATTACK_VECTORS.length)];
                const nextDefense = DEFENSE_PROTOCOLS[Math.floor(Math.random() * DEFENSE_PROTOCOLS.length)];
                setSelectedAttack(nextAttack);
                setSelectedDefense(nextDefense);

                // 3. Trigger Run
                runSimulation();

            }, 2000); // 2 second delay to read results before next run
        }

        return () => clearTimeout(timer);
    }, [isAutoSimulating, isSimulating, simResult]);


    const addLog = (actor: SecurityLog['actor'], action: string, severity: SecurityLog['severity']) => {
        setLogs(prev => [{
            id: Date.now(),
            timestamp: new Date().toLocaleTimeString(),
            actor,
            action,
            severity
        }, ...prev].slice(0, 50));
    };

    // --- Simulator Logic ---
    const runSimulation = () => {
        setIsSimulating(true);
        setSimResult(null);
        setSimProgress(0);
        
        let p = 0;
        const interval = setInterval(() => {
            p += 5;
            setSimProgress(p);
            if (p >= 100) {
                clearInterval(interval);
                setIsSimulating(false);
                
                // Calculate Result
                const baseDefense = Math.random() * 50 + 30;
                const bonus = humanOverride ? 15 : 0; // Human insight bonus
                // EKS Bonus
                const eksBonus = selectedDefense.includes('EKS') ? 20 : 0;

                const score = Math.min(99.9, baseDefense + bonus + eksBonus);
                const success = score > 60;
                
                setSimResult({
                    success,
                    score,
                    notes: success 
                        ? `Protocol ${selectedDefense} effectively mitigated ${selectedAttack}. EKS Verification confirmed integrity.`
                        : `Defense failed. ${selectedAttack} bypassed logic gates. Recommendation: Enable Active Counter-Strike.`
                });
            }
        }, 100);
    };

    // --- Radar Chart Data ---
    const radarData = [
        { subject: 'Network', A: 90, B: threatLevel, fullMark: 100 },
        { subject: 'Identity', A: 98, B: threatLevel * 0.5, fullMark: 100 },
        { subject: 'QPU', A: 86, B: 20, fullMark: 100 },
        { subject: 'Data', A: 99, B: threatLevel * 0.8, fullMark: 100 },
        { subject: 'Physical', A: 85, B: 10, fullMark: 100 },
        { subject: 'App Layer', A: 65, B: threatLevel * 1.2, fullMark: 100 },
    ];

    return (
        <GlassPanel 
            onMaximize={onMaximize}
            title={
                <div className="flex items-center justify-between w-full">
                    <div className="flex items-center">
                        <ShieldCheckIcon className="w-5 h-5 mr-2 text-green-400" />
                        <span>Security Monitor</span>
                    </div>
                    <div className="flex items-center gap-2 mr-6">
                        <button 
                            onClick={(e) => { e.stopPropagation(); setActiveView('monitor'); }}
                            className={`px-2 py-0.5 text-[10px] rounded border transition-colors ${activeView === 'monitor' ? 'bg-cyan-700 text-white border-cyan-500' : 'bg-black/40 text-gray-400 border-gray-700 hover:text-white'}`}
                        >
                            Monitor
                        </button>
                        <button 
                            onClick={(e) => { e.stopPropagation(); setActiveView('registry'); }}
                            className={`px-2 py-0.5 text-[10px] rounded border transition-colors flex items-center gap-1 ${activeView === 'registry' ? 'bg-purple-700 text-white border-purple-500' : 'bg-black/40 text-gray-400 border-gray-700 hover:text-white'}`}
                        >
                            <UsersIcon className="w-3 h-3" /> Registry
                        </button>
                    </div>
                </div>
            }
        >
            {activeView === 'registry' ? (
                <UserRegistryView />
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full p-2 overflow-hidden">
                    
                    {/* --- LEFT PANEL: REAL-TIME MONITOR (The AI) --- */}
                    <div className="flex flex-col gap-3 bg-black/20 p-3 rounded-lg border border-cyan-800/30 overflow-hidden relative">
                        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-green-500 to-transparent opacity-50"></div>
                        
                        <div className="flex justify-between items-start">
                            <div>
                                <h3 className="text-sm font-bold text-cyan-200 flex items-center">
                                    <ActivityIcon className="w-4 h-4 mr-2 text-cyan-400 animate-pulse" /> Live Status Assessment
                                </h3>
                                <p className="text-[10px] text-cyan-600">Continuous 24/7 Heuristic Scan</p>
                            </div>
                            <div className="text-right">
                                <p className={`text-2xl font-mono font-bold ${threatLevel > 50 ? 'text-red-500' : 'text-green-400'}`}>
                                    {threatLevel > 0 ? `DEFCON ${threatLevel > 75 ? '1' : threatLevel > 50 ? '2' : '3'}` : 'SAFE'}
                                </p>
                                <p className="text-[10px] text-cyan-500">Threat Level: {threatLevel.toFixed(1)}%</p>
                            </div>
                        </div>

                        {/* Radar Chart */}
                        <div className="h-40 w-full relative">
                            <ResponsiveContainer width="100%" height="100%">
                                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                                    <PolarGrid stroke="#1e293b" />
                                    <PolarAngleAxis dataKey="subject" tick={{ fill: '#06b6d4', fontSize: 10 }} />
                                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                                    <Radar name="Integrity" dataKey="A" stroke="#22c55e" fill="#22c55e" fillOpacity={0.3} />
                                    <Radar name="Threat" dataKey="B" stroke="#ef4444" fill="#ef4444" fillOpacity={0.4} />
                                    <Legend wrapperStyle={{ fontSize: '10px' }} />
                                </RadarChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Active Protocols */}
                        <div className="flex gap-2 overflow-x-auto no-scrollbar pb-1">
                            {activeProtocols.map(p => (
                                <span key={p} className="flex-shrink-0 text-[10px] bg-cyan-900/30 border border-cyan-700/50 px-2 py-1 rounded text-cyan-300 flex items-center">
                                    <LockIcon className="w-3 h-3 mr-1" /> {p}
                                </span>
                            ))}
                        </div>

                        {/* Live Log */}
                        <div className="flex-grow bg-black/40 rounded border border-cyan-900/50 p-2 overflow-y-auto font-mono text-[10px] custom-scrollbar">
                            {logs.map((log) => (
                                <div key={log.id} className="mb-1 flex gap-2">
                                    <span className="text-gray-500">[{log.timestamp}]</span>
                                    <span className={`font-bold ${log.actor === 'AGENT_Q' ? 'text-purple-400' : log.actor === 'EKS_GUARD' ? 'text-yellow-400' : 'text-red-400'}`}>{log.actor}:</span>
                                    <span className={log.severity === 'high' ? 'text-red-200' : 'text-cyan-100'}>{log.action}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* --- RIGHT PANEL: WAR ROOM SIMULATOR (The Human) --- */}
                    <div className="flex flex-col gap-3 bg-black/20 p-3 rounded-lg border border-purple-900/30 relative">
                        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-purple-500 to-transparent opacity-50"></div>

                        <div className="flex justify-between items-center">
                            <h3 className="text-sm font-bold text-white flex items-center">
                                <BugAntIcon className="w-4 h-4 mr-2 text-purple-400" /> Protocol Simulator
                            </h3>
                            <div className="flex items-center gap-2">
                                <label className="text-[10px] text-cyan-400">AGI-Human Pair Mode</label>
                                <button 
                                    onClick={() => setHumanOverride(!humanOverride)} 
                                    className={`w-8 h-4 rounded-full p-0.5 transition-colors ${humanOverride ? 'bg-purple-600' : 'bg-gray-700'}`}
                                >
                                    <div className={`w-3 h-3 bg-white rounded-full transition-transform ${humanOverride ? 'translate-x-4' : ''}`}></div>
                                </button>
                            </div>
                        </div>

                        <div className="space-y-3 flex-grow">
                            <div className="grid grid-cols-2 gap-3">
                                <div>
                                    <label className="text-[10px] text-red-400 uppercase font-bold block mb-1">Attack Vector</label>
                                    <select 
                                        value={selectedAttack} 
                                        onChange={(e) => { setSelectedAttack(e.target.value); setIsAutoSimulating(false); }}
                                        disabled={isSimulating}
                                        className="w-full bg-red-950/20 border border-red-900/50 text-red-200 text-xs rounded p-2 focus:outline-none focus:border-red-500"
                                    >
                                        {ATTACK_VECTORS.map(v => <option key={v} value={v}>{v}</option>)}
                                    </select>
                                </div>
                                <div>
                                    <label className="text-[10px] text-green-400 uppercase font-bold block mb-1">Defense Protocol</label>
                                    <select 
                                        value={selectedDefense} 
                                        onChange={(e) => { setSelectedDefense(e.target.value); setIsAutoSimulating(false); }}
                                        disabled={isSimulating}
                                        className="w-full bg-green-950/20 border border-green-900/50 text-green-200 text-xs rounded p-2 focus:outline-none focus:border-green-500"
                                    >
                                        {DEFENSE_PROTOCOLS.map(v => <option key={v} value={v}>{v}</option>)}
                                    </select>
                                </div>
                            </div>

                            <div className="bg-black/40 border border-cyan-900/50 rounded p-3 h-32 flex flex-col items-center justify-center text-center relative overflow-hidden">
                                {isSimulating ? (
                                    <div className="w-full max-w-xs">
                                        <p className="text-cyan-400 text-xs mb-2 animate-pulse">Running Monte Carlo Simulations...</p>
                                        <div className="w-full h-1 bg-gray-800 rounded-full overflow-hidden">
                                            <div className="h-full bg-cyan-500 transition-all duration-100" style={{width: `${simProgress}%`}}></div>
                                        </div>
                                        <p className="text-[10px] text-gray-500 mt-2 font-mono">Iteration: {Math.floor(simProgress * 12.4)}</p>
                                    </div>
                                ) : simResult ? (
                                    <div className="animate-fade-in-up">
                                        <div className={`text-2xl font-bold mb-1 ${simResult.success ? 'text-green-400' : 'text-red-400'}`}>
                                            {simResult.score.toFixed(1)}% Success
                                        </div>
                                        <p className="text-xs text-gray-300 leading-relaxed px-2">{simResult.notes}</p>
                                        {isAutoSimulating && <p className="text-[9px] text-yellow-500 mt-2 animate-pulse">Auto-Implementing Fix...</p>}
                                    </div>
                                ) : (
                                    <p className="text-xs text-gray-500">Configure parameters and run simulation to test defense viability.</p>
                                )}
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-2">
                             <button 
                                onClick={() => setIsAutoSimulating(!isAutoSimulating)}
                                disabled={isSimulating && !isAutoSimulating}
                                className={`holographic-button py-2 rounded text-xs font-bold flex items-center justify-center gap-2 transition-all ${isAutoSimulating ? 'bg-purple-600/30 border-purple-500 text-purple-200 animate-pulse shadow-[0_0_10px_theme(colors.purple.500/50%)]' : 'bg-black/40 border-purple-500/30 text-purple-400'}`}
                            >
                                {isAutoSimulating ? <StopIcon className="w-4 h-4"/> : <FastForwardIcon className="w-4 h-4"/>}
                                {isAutoSimulating ? 'Auto-Loop Active' : 'Auto-Simulate'}
                            </button>
                            <button 
                                onClick={runSimulation}
                                disabled={isSimulating || isAutoSimulating}
                                className={`holographic-button py-2 rounded text-xs font-bold flex items-center justify-center gap-2 ${isSimulating || isAutoSimulating ? 'opacity-50 cursor-not-allowed' : ''}`}
                            >
                                {isSimulating ? <LoaderIcon className="w-4 h-4 animate-spin"/> : <PlayIcon className="w-4 h-4"/>}
                                {isSimulating ? 'Simulating...' : 'Run Once'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </GlassPanel>
    );
};

export default SecurityMonitorAndSimulator;
