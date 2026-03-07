import React, { useState } from 'react';
import GlassPanel from './GlassPanel';
import { BoxIcon } from './Icons';

export interface DeployAppModalProps {
    code: string;
    onDeploy: (details: { name: string, description: string, code: string }) => void;
    onClose: () => void;
}

const DeployAppModal: React.FC<DeployAppModalProps> = ({ code, onDeploy, onClose }) => {
    const [name, setName] = useState('');
    const [description, setDescription] = useState('');

    const handleSubmit = () => {
        if (name.trim() && description.trim()) {
            onDeploy({ name, description, code });
            onClose();
        }
    };

    return (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center animate-fade-in-up" onClick={onClose}>
            <div className="w-[500px] max-w-[95vw] pointer-events-auto" onClick={(e) => e.stopPropagation()}>
                <GlassPanel title="Deploy New Application">
                    <div className="p-4 flex flex-col gap-4">
                        <h3 className="text-cyan-200 text-lg">Confirm Deployment</h3>
                        <p className="text-xs text-cyan-400 -mt-3">Provide the following details for your new application to be hosted on the Google App Store.</p>
                        <div>
                            <label htmlFor="app-name" className="block text-cyan-400 mb-1 text-sm">App Name</label>
                            <input
                                id="app-name"
                                type="text"
                                value={name}
                                onChange={(e) => setName(e.target.value)}
                                className="w-full bg-black/30 border border-blue-500/50 rounded-md p-2 text-white placeholder:text-gray-500 focus:ring-1 focus:ring-cyan-400 focus:outline-none"
                                placeholder="e.g., Quantum Weather Forecaster"
                            />
                        </div>
                         <div>
                            <label htmlFor="app-desc" className="block text-cyan-400 mb-1 text-sm">Description</label>
                            <textarea
                                id="app-desc"
                                value={description}
                                onChange={(e) => setDescription(e.target.value)}
                                rows={3}
                                className="w-full bg-black/30 border border-blue-500/50 rounded-md p-2 text-white placeholder:text-gray-500 focus:ring-1 focus:ring-cyan-400 focus:outline-none resize-none"
                                placeholder="A brief summary of what this app does."
                            />
                        </div>
                        <div className="flex justify-end gap-2 mt-2">
                             <button onClick={onClose} title="Close this dialog without deploying." className="px-4 py-2 bg-slate-500/30 hover:bg-slate-500/50 border border-slate-500/50 text-slate-200 font-bold rounded transition-colors">
                                Cancel
                            </button>
                            <button onClick={handleSubmit} title="Deploy the application to the App Exchange." disabled={!name.trim() || !description.trim()} className="px-4 py-2 bg-green-500/30 hover:bg-green-500/50 border border-green-500/50 text-green-200 font-bold rounded transition-colors disabled:opacity-50">
                                Confirm & Deploy
                            </button>
                        </div>
                    </div>
                </GlassPanel>
            </div>
        </div>
    );
};

export default DeployAppModal;