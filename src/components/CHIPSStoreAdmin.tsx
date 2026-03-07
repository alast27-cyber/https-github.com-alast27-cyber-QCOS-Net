import React, { useState } from 'react';
import { 
    BoxIcon, CheckCircle2Icon, XCircleIcon, ClockIcon, 
    ShieldCheckIcon, DownloadCloudIcon, SearchIcon, ActivityIcon 
} from './Icons';
import { AppDefinition } from '../types';

interface StoreAppEntry {
    id: string;
    name: string;
    developer: string;
    version: string;
    category: string;
    status: 'Pending' | 'Live' | 'Rejected';
    permissions: string[];
    submissionDate: string;
    installs: number;
}

const staticPendingApps: StoreAppEntry[] = [
    { 
        id: 'q-chess-pro', 
        name: 'Quantum Chess Pro', 
        developer: '0x7A...9F2', 
        version: '1.0.2', 
        category: 'Game', 
        status: 'Pending', 
        permissions: ['QPU:Low', 'Net:Public'], 
        submissionDate: '2025-10-30',
        installs: 0
    },
    { 
        id: 'secure-vote-dao', 
        name: 'SecureVote DAO', 
        developer: '0xB3...11C', 
        version: '0.9.5', 
        category: 'Governance', 
        status: 'Pending', 
        permissions: ['QKD:High', 'Storage:Persistent'], 
        submissionDate: '2025-10-31',
        installs: 0
    }
];

interface CHIPSStoreAdminProps {
    liveApps?: AppDefinition[];
}

const CHIPSStoreAdmin: React.FC<CHIPSStoreAdminProps> = ({ liveApps = [] }) => {
    const [pendingApps, setPendingApps] = useState<StoreAppEntry[]>(staticPendingApps);
    const [processingId, setProcessingId] = useState<string | null>(null);

    const handleAction = (id: string, action: 'approve' | 'reject') => {
        setProcessingId(id);
        setTimeout(() => {
            setPendingApps(prev => prev.filter(app => app.id !== id));
            setProcessingId(null);
            // In a real app, this would trigger an update to the global liveApps list
        }, 1500);
    };

    return (
        <div className="flex flex-col h-full space-y-6 animate-fade-in">
            
            {/* Top Metrics */}
            <div className="grid grid-cols-3 gap-4">
                <div className="bg-black/20 p-3 rounded-lg border border-cyan-800/50 flex flex-col items-center">
                    <span className="text-xs text-cyan-500 uppercase tracking-wider">Store Inventory</span>
                    <span className="text-2xl font-mono text-white">{liveApps.length + pendingApps.length}</span>
                </div>
                <div className="bg-black/20 p-3 rounded-lg border border-cyan-800/50 flex flex-col items-center">
                    <span className="text-xs text-yellow-500 uppercase tracking-wider">Queue</span>
                    <span className="text-2xl font-mono text-yellow-300">{pendingApps.length}</span>
                </div>
                <div className="bg-black/20 p-3 rounded-lg border border-cyan-800/50 flex flex-col items-center">
                    <span className="text-xs text-green-500 uppercase tracking-wider">Verified Live</span>
                    <span className="text-2xl font-mono text-green-300">{liveApps.length}</span>
                </div>
            </div>

            {/* Pending Review Section */}
            <div className="bg-black/20 p-3 rounded-lg border border-cyan-800/50 flex-grow min-h-0 flex flex-col">
                <h3 className="flex items-center text-sm font-semibold text-yellow-300 mb-3 border-b border-cyan-900 pb-2">
                    <ClockIcon className="w-4 h-4 mr-2" /> Pending Submissions
                </h3>
                
                <div className="overflow-y-auto pr-2 -mr-2 space-y-2 custom-scrollbar">
                    {pendingApps.length === 0 ? (
                        <p className="text-xs text-cyan-600 italic text-center py-4">Verification queue is empty.</p>
                    ) : (
                        pendingApps.map(app => (
                            <div key={app.id} className="bg-cyan-950/30 p-3 rounded border border-cyan-900/50 flex flex-col sm:flex-row justify-between gap-3 animate-fade-in">
                                <div className="flex-grow">
                                    <div className="flex items-center gap-2">
                                        <span className="font-bold text-white text-sm">{app.name}</span>
                                        <span className="text-[10px] text-cyan-500 bg-cyan-900/50 px-1.5 py-0.5 rounded border border-cyan-800">v{app.version}</span>
                                    </div>
                                    <p className="text-xs text-gray-400 font-mono mt-1">Provider: {app.developer}</p>
                                    <div className="flex gap-2 mt-2">
                                        {app.permissions.map(p => (
                                            <span key={p} className="text-[10px] text-cyan-300 bg-black/40 px-1.5 rounded flex items-center border border-cyan-900">
                                                <ShieldCheckIcon className="w-3 h-3 mr-1" /> {p}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                                <div className="flex items-center gap-2 flex-shrink-0">
                                    <button 
                                        onClick={() => handleAction(app.id, 'reject')}
                                        disabled={!!processingId}
                                        className="p-2 rounded bg-red-900/20 hover:bg-red-900/40 text-red-400 border border-red-800/50 transition-colors disabled:opacity-50"
                                        title="Reject"
                                    >
                                        <XCircleIcon className="w-5 h-5" />
                                    </button>
                                    <button 
                                        onClick={() => handleAction(app.id, 'approve')}
                                        disabled={!!processingId}
                                        className="holographic-button px-4 py-2 bg-green-600/20 hover:bg-green-600/40 text-green-300 border-green-500/50 flex items-center gap-2 disabled:opacity-50"
                                    >
                                        {processingId === app.id ? <ActivityIcon className="w-4 h-4 animate-spin"/> : <CheckCircle2Icon className="w-4 h-4" />}
                                        <span className="text-xs font-bold">{processingId === app.id ? 'Attesting EKS...' : 'Verify & Approve'}</span>
                                    </button>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>

            {/* Live Catalog Section */}
            <div className="bg-black/20 p-3 rounded-lg border border-cyan-800/50 flex-grow min-h-0 flex flex-col">
                <h3 className="flex items-center text-sm font-semibold text-green-300 mb-3 border-b border-cyan-900 pb-2">
                    <BoxIcon className="w-4 h-4 mr-2" /> Store Registry (Live)
                </h3>
                <div className="overflow-x-auto custom-scrollbar flex-grow">
                    <table className="w-full text-xs text-left">
                        <thead className="text-cyan-500 border-b border-cyan-900 bg-black/40">
                            <tr>
                                <th className="p-2 font-bold uppercase tracking-wider">Application</th>
                                <th className="p-2 font-bold uppercase tracking-wider">Q-URI</th>
                                <th className="p-2 font-bold uppercase tracking-wider text-right">Fidelity</th>
                                <th className="p-2 font-bold uppercase tracking-wider text-center">Protocol</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-cyan-900/30">
                            {liveApps.length === 0 ? (
                                <tr>
                                    <td colSpan={4} className="py-4 text-center text-gray-500 italic">No apps currently live in store.</td>
                                </tr>
                            ) : (
                                liveApps.map(app => (
                                    <tr key={app.id} className="hover:bg-cyan-900/10 transition-colors">
                                        <td className="p-2 font-bold text-white flex items-center gap-2">
                                            <app.icon className="w-4 h-4 text-cyan-400" />
                                            {app.name}
                                        </td>
                                        <td className="p-2 font-mono text-cyan-600 truncate max-w-[150px]">{app.q_uri || `CHIPS://${app.id}.apps`}</td>
                                        <td className="p-2 text-right font-mono text-green-300">99.99%</td>
                                        <td className="p-2 text-center">
                                            <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-cyan-900/30 text-cyan-400 border border-cyan-800 text-[9px] uppercase font-bold">
                                                CHIPS/EKS
                                            </span>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

        </div>
    );
};

export default CHIPSStoreAdmin;