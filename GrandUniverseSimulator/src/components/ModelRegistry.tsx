
import React, { useState } from 'react';
import { 
    CircleStackIcon, CheckCircle2Icon, AlertTriangleIcon, 
    RefreshCwIcon, TrashIcon, ClockIcon, ArrowPathIcon
} from './Icons';

interface Model {
    id: string;
    name: string;
    version: string;
    status: 'Deployed' | 'Staging' | 'Deprecated' | 'Training';
    accuracy: string;
    lastUpdated: string;
}

const initialModels: Model[] = [
    { id: 'm-001', name: 'Q-Fin Predictor', version: 'v2.1.0', status: 'Deployed', accuracy: '94.2%', lastUpdated: '2025-10-25' },
    { id: 'm-002', name: 'SwineFlu-Net', version: 'v1.0.4', status: 'Deployed', accuracy: '98.5%', lastUpdated: '2025-10-20' },
    { id: 'm-003', name: 'Traffic-Q-Opt', version: 'v0.9.0-beta', status: 'Staging', accuracy: '88.1%', lastUpdated: '2025-10-28' },
    { id: 'm-004', name: 'Legacy-Risk-Engine', version: 'v1.0.0', status: 'Deprecated', accuracy: '76.0%', lastUpdated: '2024-12-15' },
    { id: 'm-005', name: 'Alpha-Fold-Q', version: 'v3.0.1', status: 'Training', accuracy: '---', lastUpdated: '2025-10-31' },
];

const ModelRegistry: React.FC = () => {
    const [models, setModels] = useState<Model[]>(initialModels);

    const getStatusStyles = (status: Model['status']) => {
        switch(status) {
            case 'Deployed': return { color: 'text-green-400', bg: 'bg-green-900/30', border: 'border-green-800', icon: CheckCircle2Icon };
            case 'Staging': return { color: 'text-yellow-400', bg: 'bg-yellow-900/30', border: 'border-yellow-800', icon: ClockIcon };
            case 'Deprecated': return { color: 'text-red-400', bg: 'bg-red-900/30', border: 'border-red-800', icon: AlertTriangleIcon };
            case 'Training': return { color: 'text-blue-400', bg: 'bg-blue-900/30', border: 'border-blue-800', icon: ArrowPathIcon };
            default: return { color: 'text-gray-400', bg: 'bg-gray-900/30', border: 'border-gray-800', icon: CircleStackIcon };
        }
    };

    const handleDelete = (id: string) => {
        if(window.confirm('Are you sure you want to remove this model from the registry?')) {
            setModels(prev => prev.filter(m => m.id !== id));
        }
    };

    return (
        <div className="flex flex-col h-full animate-fade-in text-cyan-100 space-y-4">
            
            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-black/20 p-3 rounded-lg border border-cyan-800/50 text-center">
                    <p className="text-xs text-cyan-500 uppercase tracking-wider">Total Models</p>
                    <p className="text-2xl font-mono text-white">{models.length}</p>
                </div>
                <div className="bg-black/20 p-3 rounded-lg border border-cyan-800/50 text-center">
                    <p className="text-xs text-green-500 uppercase tracking-wider">Deployed</p>
                    <p className="text-2xl font-mono text-green-400">{models.filter(m => m.status === 'Deployed').length}</p>
                </div>
                <div className="bg-black/20 p-3 rounded-lg border border-cyan-800/50 text-center">
                    <p className="text-xs text-yellow-500 uppercase tracking-wider">Staging</p>
                    <p className="text-2xl font-mono text-yellow-400">{models.filter(m => m.status === 'Staging').length}</p>
                </div>
                <div className="bg-black/20 p-3 rounded-lg border border-cyan-800/50 text-center">
                    <p className="text-xs text-blue-500 uppercase tracking-wider">Training</p>
                    <p className="text-2xl font-mono text-blue-400">{models.filter(m => m.status === 'Training').length}</p>
                </div>
            </div>

            {/* Registry Table */}
            <div className="flex-grow bg-black/20 rounded-lg border border-cyan-800/50 overflow-hidden flex flex-col">
                <div className="p-3 border-b border-cyan-800/50 flex justify-between items-center bg-cyan-950/20">
                    <div className="flex items-center gap-2">
                        <CircleStackIcon className="w-5 h-5 text-cyan-400" />
                        <h3 className="font-bold text-sm text-cyan-200">Model Artifacts</h3>
                    </div>
                    <button className="holographic-button px-3 py-1.5 text-xs flex items-center gap-1 rounded">
                        <RefreshCwIcon className="w-3 h-3" /> Sync Registry
                    </button>
                </div>
                <div className="flex-grow overflow-y-auto">
                    <table className="w-full text-xs text-left">
                        <thead className="text-cyan-500 bg-cyan-950/30 sticky top-0 z-10">
                            <tr>
                                <th className="p-3 font-semibold">Model Name</th>
                                <th className="p-3 font-semibold">Version</th>
                                <th className="p-3 font-semibold">Status</th>
                                <th className="p-3 font-semibold">Accuracy</th>
                                <th className="p-3 font-semibold">Last Updated</th>
                                <th className="p-3 font-semibold text-right">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-cyan-900/30">
                            {models.map(model => {
                                const styles = getStatusStyles(model.status);
                                const StatusIcon = styles.icon;
                                return (
                                    <tr key={model.id} className="hover:bg-cyan-900/10 transition-colors group">
                                        <td className="p-3 font-medium text-white">{model.name}</td>
                                        <td className="p-3 font-mono text-cyan-600">{model.version}</td>
                                        <td className="p-3">
                                            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded border text-[10px] uppercase font-bold ${styles.color} ${styles.bg} ${styles.border}`}>
                                                <StatusIcon className="w-3 h-3" />
                                                {model.status}
                                            </span>
                                        </td>
                                        <td className="p-3 font-mono text-cyan-200">{model.accuracy}</td>
                                        <td className="p-3 text-gray-400">{model.lastUpdated}</td>
                                        <td className="p-3 text-right">
                                            <button 
                                                onClick={() => handleDelete(model.id)}
                                                className="text-red-500/50 hover:text-red-400 p-1.5 rounded hover:bg-red-900/20 transition-colors opacity-0 group-hover:opacity-100"
                                                title="Delete Model"
                                            >
                                                <TrashIcon className="w-4 h-4" />
                                            </button>
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default ModelRegistry;
