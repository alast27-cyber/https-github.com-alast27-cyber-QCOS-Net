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

interface SecurityLog {
    id: string; // Changed from number to string for crypto-unique IDs
    timestamp: string;
    actor: 'AGENT_Q' | 'SYSTEM' | 'INTRUSION' | 'EKS_GUARD';
    action: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
}

// ... (UserRecord and Constants remain same as original)

const UserRegistryView: React.FC = () => {
    const [users, setUsers] = useState<UserRecord[]>([
        { id: 'usr-001', username: 'sys_admin_prime', level: 4, role: 'System Architect', lastActive: 'Now', status: 'Active' },
        { id: 'usr-002', username: 'net_ops_lead', level: 3, role: 'Network Admin', lastActive: '5m ago', status: 'Active' },
        { id: 'usr-003', username: 'dev_gamma', level: 2, role: 'Frontend Dev', lastActive: '2h ago', status: 'Active' },
        { id: 'usr-004', username: 'dev_delta', level: 2, role: 'Backend Dev', lastActive: '1d ago', status: 'Flagged' },
        { id: 'usr-005', username: 'guest_user_12', level: 1, role: 'General User', lastActive: '10m ago', status: 'Active' },
    ]);

    // ... (Helpers remain same)

    return (
        <div className="h-full flex flex-col gap-4 overflow-hidden animate-fade-in">
            {/* ... stats ... */}
            <div className="flex-grow bg-black/30 border border-cyan-800/30 rounded-lg overflow-hidden flex flex-col">
                <div className="flex-grow overflow-y-auto custom-scrollbar p-1">
                    {users.map((user, idx) => (
                        <div key={`${user.id}-${idx}`} className="grid grid-cols-12 gap-2 items-center p-2 hover:bg-white/5 rounded transition-colors text-[10px] border-b border-white/5 last:border-0 group">
                            {/* ... user row content ... */}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

const SecurityMonitorAndSimulator: React.FC<SecurityMonitorProps> = ({ onMaximize }) => {
    const [logs, setLogs] = useState<SecurityLog[]>([]);
    
    // Fixed Log Generation to prevent duplicate keys
    const addLog = (entry: Omit<SecurityLog, 'id'>) => {
        const newLog = {
            ...entry,
            id: `log-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
        };
        setLogs(prev => [newLog, ...prev].slice(0, 50));
    };

    // ... (Remainder of simulator logic utilizing addLog)

    return (
        <GlassPanel title="SECURITY_MONITOR" onMaximize={onMaximize}>
             <div className="space-y-2 overflow-y-auto custom-scrollbar pr-2 h-full">
                {logs.map((log) => (
                    <div key={log.id} className="p-2 border border-white/5 rounded bg-white/5 animate-fade-in">
                        {/* ... log rendering ... */}
                    </div>
                ))}
            </div>
        </GlassPanel>
    );
};

export default SecurityMonitorAndSimulator;