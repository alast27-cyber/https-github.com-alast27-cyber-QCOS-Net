import React from 'react';
import GlassPanel from './GlassPanel';
import { UploadCloudIcon, GlobeIcon, RocketLaunchIcon } from './Icons';

interface DeployedModel {
  id: string;
  name: string;
  status: 'Deployed' | 'Pending' | 'Draft';
  publicUrl?: string;
}

const PublicDeploymentPanel: React.FC = () => {
  const models: DeployedModel[] = [
    { id: 'abundance-v1', name: 'Abundance Pathways v1.0', status: 'Deployed', publicUrl: 'https://qcos.apps.web/abundance-v1' },
    { id: 'resilience-index', name: 'Economic Resilience Index', status: 'Deployed', publicUrl: 'https://qcos.apps.web/resilience-index' },
    { id: 'resource-alloc-sim', name: 'Resource Allocation Simulator (Dev)', status: 'Draft' },
    { id: 'supply-chain-optim', name: 'Quantum Supply Chain Optimizer', status: 'Pending' },
  ];

  const handleDeploy = (modelId: string) => {
    console.log(`Initiating deployment for model: ${modelId}`);
    alert(`Deployment initiated for ${modelId}. Check 'Public Deployment & Optimization Hub' (g-config) for status.`);
  };

  return (
    <GlassPanel title="Public Model Deployment">
      <div className="p-4 text-cyan-300 space-y-4 h-full flex flex-col">
        <p className="text-sm flex-shrink-0">Manage public access for GAE models via the Quantum-to-Web Gateway.</p>
        <ul className="space-y-3 flex-grow overflow-y-auto pr-2 -mr-2">
          {models.map((model) => (
            <li key={model.id} className="bg-black/30 p-3 rounded-lg shadow-inner flex flex-col sm:flex-row sm:items-center justify-between">
              <div className="flex-grow">
                <p className="font-semibold text-white flex items-center gap-2">
                  <RocketLaunchIcon className="h-5 w-5 text-cyan-400" />
                  {model.name}
                </p>
                <p className={`text-sm mt-1 ${model.status === 'Deployed' ? 'text-green-300' : model.status === 'Pending' ? 'text-yellow-300' : 'text-gray-400'}`}>
                  Status: {model.status}
                </p>
                {model.publicUrl && (
                  <a href={model.publicUrl} target="_blank" rel="noopener noreferrer" className="text-cyan-200 hover:text-cyan-100 text-xs flex items-center gap-1 mt-1">
                    <GlobeIcon className="h-4 w-4" />
                    {model.publicUrl}
                  </a>
                )}
              </div>
              <div className="mt-3 sm:mt-0 sm:ml-4 flex-shrink-0">
                {model.status === 'Draft' && (
                  <button
                    onClick={() => handleDeploy(model.id)}
                    className="flex items-center px-4 py-2 bg-cyan-600/30 hover:bg-cyan-700/50 text-white rounded-md text-sm transition-colors duration-200 holographic-button"
                  >
                    <UploadCloudIcon className="h-5 w-5 mr-2" />
                    Deploy
                  </button>
                )}
                {model.status === 'Deployed' && (
                    <span className="px-4 py-2 bg-green-700/50 text-green-300 rounded-md text-sm cursor-not-allowed flex items-center">
                        <GlobeIcon className="h-5 w-5 inline mr-1" /> Live
                    </span>
                )}
                {model.status === 'Pending' && (
                    <span className="px-4 py-2 bg-yellow-700/50 text-yellow-300 rounded-md text-sm cursor-not-allowed flex items-center">
                        <UploadCloudIcon className="h-5 w-5 inline mr-1" /> Pending
                    </span>
                )}
              </div>
            </li>
          ))}
        </ul>
      </div>
    </GlassPanel>
  );
};

export default PublicDeploymentPanel;
