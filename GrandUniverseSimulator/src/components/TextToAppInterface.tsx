import React, { useState, useEffect } from 'react';
import GlassPanel from './GlassPanel';
import { CpuChipIcon, ArrowPathIcon, CodeBracketSquareIcon, GlobeIcon, UploadCloudIcon, CheckCircle2Icon, XCircleIcon } from './Icons';

interface CHIPSAgentWeaverProps {
  initialAgentInput?: string;
  onPublishToExchange: (details: { name: string; description: string; code: string; }) => void;
}

const CHIPSAgentWeaver: React.FC<CHIPSAgentWeaverProps> = ({ initialAgentInput, onPublishToExchange }) => {
  const [appName, setAppName] = useState<string>('');
  const [appDescription, setAppDescription] = useState<string>('');
  const [agentInput, setAgentInput] = useState<string>('');
  const [protocolStatus, setProtocolStatus] = useState<'idle' | 'encoding' | 'encoded' | 'error'>('idle');
  const [deploymentStatus, setDeploymentStatus] = useState<'idle' | 'deploying' | 'deployed' | 'error'>('idle');
  const [publishStatus, setPublishStatus] = useState<'idle' | 'published'>('idle');
  const [publicUrlEnabled, setPublicUrlEnabled] = useState<boolean>(true);
  const [generatedPublicUrl, setGeneratedPublicUrl] = useState<string | null>(null);
  const [deploymentLog, setDeploymentLog] = useState<string[]>([]);

  useEffect(() => {
    if (initialAgentInput) {
        setTimeout(() => {
            setAgentInput(initialAgentInput);
            setProtocolStatus('idle');
            setDeploymentStatus('idle');
            setGeneratedPublicUrl(null);
            setPublishStatus('idle');
            setAppName('');
            setAppDescription('');
            setDeploymentLog(['[System] Agent definition received from QML Forge. Ready for protocol encoding.']);
        }, 0);
    }
  }, [initialAgentInput]);

  const handleEncodeProtocols = () => {
    setProtocolStatus('encoding');
    setDeploymentLog(prev => [...prev.slice(-10), `[${new Date().toLocaleTimeString()}] Initiating CHIPS protocol encoding...`]);
    setTimeout(() => {
      if (agentInput.trim() === '') {
        setProtocolStatus('error');
        setDeploymentLog(prev => [...prev.slice(-10), `[${new Date().toLocaleTimeString()}] Error: Agent input cannot be empty.`]);
      } else {
        setProtocolStatus('encoded');
        setDeploymentLog(prev => [...prev.slice(-10), `[${new Date().toLocaleTimeString()}] Protocols encoded for agent...`]);
      }
    }, 2000);
  };

  const handleDeployAgent = () => {
    if (protocolStatus !== 'encoded') {
      setDeploymentLog(prev => [...prev.slice(-10), `[${new Date().toLocaleTimeString()}] Error: Protocols must be encoded before deployment.`]);
      return;
    }
    setDeploymentStatus('deploying');
    setDeploymentLog(prev => [...prev.slice(-10), `[${new Date().toLocaleTimeString()}] Deploying AI-Native Agent to CHIPS Network...`]);

    setTimeout(() => {
      const success = Math.random() > 0.1;
      if (success) {
        setDeploymentStatus('deployed');
        setDeploymentLog(prev => [...prev.slice(-10), `[${new Date().toLocaleTimeString()}] Agent deployed successfully!`]);
        if (publicUrlEnabled) {
          const nameForUrl = appName.toLowerCase().replace(/[^a-z0-9]/g, '-') || 'agent';
          const uniqueId = Math.random().toString(36).substring(2, 8);
          const url = `https://qcos.apps.web/${nameForUrl}-${uniqueId}`;
          setGeneratedPublicUrl(url);
          setDeploymentLog(prev => [...prev.slice(-10), `[${new Date().toLocaleTimeString()}] Quantum-to-Web Gateway assigned: ${url}`]);
        }
      } else {
        setDeploymentStatus('error');
        setDeploymentLog(prev => [...prev.slice(-10), `[${new Date().toLocaleTimeString()}] Deployment failed. Retrying...`]);
      }
    }, 3000);
  };
  
  const handlePublish = () => {
    if (!appName || !appDescription || !agentInput) {
        setDeploymentLog(prev => [...prev.slice(-10), `[${new Date().toLocaleTimeString()}] Error: App Name, Description, and Agent Definition are required to publish.`]);
        return;
    }
    onPublishToExchange({
        name: appName,
        description: appDescription,
        code: agentInput,
    });
    setPublishStatus('published');
    setDeploymentLog(prev => [...prev.slice(-10), `[${new Date().toLocaleTimeString()}] Agent successfully published to the Quantum App Exchange!`]);
};

  const getStatusIcon = (status: 'idle' | 'encoding' | 'encoded' | 'error' | 'deploying' | 'deployed') => {
    switch (status) {
      case 'encoded':
      case 'deployed':
        return <CheckCircle2Icon className="w-5 h-5 text-green-400" />;
      case 'encoding':
      case 'deploying':
        return <ArrowPathIcon className="w-5 h-5 text-cyan-400 animate-spin" />;
      case 'error':
        return <XCircleIcon className="w-5 h-5 text-red-400" />;
      default:
        return <CodeBracketSquareIcon className="w-5 h-5 text-gray-400" />;
    }
  };

  return (
    <GlassPanel title={<div className="flex items-center"><CpuChipIcon className="w-6 h-6 text-cyan-300 mr-2" />CHIPS Agent Weaver</div>}>
      <div className="flex flex-col h-full p-2 space-y-4 text-cyan-100">
        <p className="text-sm text-cyan-200">
          Develop, protocol-encode, and deploy AI-Native Agents as decentralized CHIPS browser nodes. Integrate with the Quantum-to-Web Gateway for public access.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
                <label htmlFor="appName" className="block text-sm font-medium text-cyan-300">Application Name</label>
                <input id="appName" type="text" className="mt-1 w-full p-2 bg-black/30 border border-cyan-700 rounded-md text-cyan-50 focus:ring-cyan-500 focus:border-cyan-500 font-mono text-xs" placeholder="e.g., Quantum Market Predictor" value={appName} onChange={(e) => setAppName(e.target.value)} />
            </div>
            <div>
                <label htmlFor="appDescription" className="block text-sm font-medium text-cyan-300">Application Description</label>
                <input id="appDescription" type="text" className="mt-1 w-full p-2 bg-black/30 border border-cyan-700 rounded-md text-cyan-50 focus:ring-cyan-500 focus:border-cyan-500 font-mono text-xs" placeholder="A brief description of the agent's purpose." value={appDescription} onChange={(e) => setAppDescription(e.target.value)} />
            </div>
        </div>

        <div className="flex-grow flex flex-col space-y-2">
            <label htmlFor="agentInput" className="block text-sm font-medium text-cyan-300">
              Agent Definition (from QML Forge / Q-Lang):
            </label>
            <textarea
              id="agentInput"
              rows={6}
              className="w-full p-2 bg-black/30 border border-cyan-700 rounded-md text-cyan-50 focus:ring-cyan-500 focus:border-cyan-500 font-mono text-xs"
              placeholder="Paste Q-Lang code, Agent ID, or configuration from QML Forge here..."
              value={agentInput}
              onChange={(e) => {
                  setAgentInput(e.target.value);
                  setProtocolStatus('idle');
                  setDeploymentStatus('idle');
                  setGeneratedPublicUrl(null);
                  setPublishStatus('idle');
                  setDeploymentLog([]);
              }}
            />
        </div>

        <div className="flex items-center space-x-4">
          <button
            onClick={handleEncodeProtocols}
            disabled={protocolStatus === 'encoding' || deploymentStatus === 'deploying'}
            className="holographic-button flex items-center justify-center px-4 py-2 bg-cyan-600/30 text-white font-medium rounded-md shadow-sm disabled:opacity-50"
            title="Convert agent definition into CHIPS-compatible protocols"
          >
            {protocolStatus === 'encoding' ? (
              <><ArrowPathIcon className="w-5 h-5 mr-2 animate-spin" /> Encoding...</>
            ) : (
              <><CodeBracketSquareIcon className="w-5 h-5 mr-2" /> Encode Protocols</>
            )}
          </button>
          <div className="flex items-center text-sm text-cyan-200">
            Status: {getStatusIcon(protocolStatus)}
            <span className="ml-2">{protocolStatus.toUpperCase()}</span>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
            <input
                type="checkbox"
                id="publicUrlToggle"
                checked={publicUrlEnabled}
                onChange={() => setPublicUrlEnabled(!publicUrlEnabled)}
                className="h-4 w-4 text-cyan-600 bg-gray-700 border-gray-600 rounded focus:ring-cyan-500"
            />
            <label htmlFor="publicUrlToggle" className="text-sm font-medium text-cyan-300 flex items-center">
                <GlobeIcon className="w-5 h-5 mr-1 text-cyan-400" /> Assign Public HTTPS URL
            </label>
        </div>

        <div className="flex items-center space-x-4">
          <button
            onClick={handleDeployAgent}
            disabled={deploymentStatus === 'deploying' || protocolStatus !== 'encoded'}
            className="holographic-button flex items-center justify-center px-4 py-2 bg-purple-600/30 text-white font-medium rounded-md shadow-sm disabled:opacity-50"
            title="Deploy the encoded agent to the CHIPS network"
          >
            {deploymentStatus === 'deploying' ? (
              <><UploadCloudIcon className="w-5 h-5 mr-2 animate-bounce" /> Deploying...</>
            ) : (
              <><UploadCloudIcon className="w-5 h-5 mr-2" /> Deploy AI-Native Agent</>
            )}
          </button>
          <div className="flex items-center text-sm text-cyan-200">
            Status: {getStatusIcon(deploymentStatus)}
            <span className="ml-2">{deploymentStatus.toUpperCase()}</span>
          </div>
        </div>

        {deploymentStatus === 'deployed' && (
            <div className="p-3 mt-2 bg-green-900/30 border border-green-700 rounded-md text-green-300 text-sm animate-fade-in">
                <div className="flex justify-between items-center">
                    <div>
                        <p className="font-bold">Deployment Successful!</p>
                        {generatedPublicUrl && (<a href={generatedPublicUrl} target="_blank" rel="noopener noreferrer" className="text-xs underline hover:text-green-200 break-all">{generatedPublicUrl}</a>)}
                    </div>
                    <button onClick={handlePublish} disabled={publishStatus === 'published' || !appName || !appDescription} className="holographic-button flex-shrink-0 flex items-center justify-center px-3 py-1.5 bg-green-600/40 text-white font-medium rounded-md shadow-sm disabled:opacity-60 disabled:cursor-not-allowed ml-4" title={!appName || !appDescription ? "Please provide an App Name and Description to publish" : "Publish this agent to the App Exchange"}>
                        {publishStatus === 'published' ? (<><CheckCircle2Icon className="w-4 h-4 mr-2" /> Published</>) : (<><UploadCloudIcon className="w-4 h-4 mr-2" /> Publish to Exchange</>)}
                    </button>
                </div>
                {publishStatus === 'published' && <p className="text-xs mt-2">This agent is now listed in the Quantum App Exchange.</p>}
            </div>
        )}

        <div className="flex-grow bg-black/40 p-3 rounded-md border border-cyan-800/50 text-xs overflow-auto font-mono min-h-[100px]">
          <p className="text-cyan-300 font-medium mb-1">Deployment Log:</p>
          {deploymentLog.map((log, index) => (
            <p key={index} className="text-gray-200">{log}</p>
          ))}
          {deploymentLog.length === 0 && <p className="text-gray-400">Awaiting actions...</p>}
        </div>
      </div>
    </GlassPanel>
  );
};

export default CHIPSAgentWeaver;