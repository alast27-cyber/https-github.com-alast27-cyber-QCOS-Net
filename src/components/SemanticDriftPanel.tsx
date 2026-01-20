
import React, { useState } from 'react';
import GlassPanel from './GlassPanel';
import { MagnifyingGlassIcon, ShieldCheckIcon, BugAntIcon, ZapIcon } from './Icons'; // Replaced BoltIcon

const SemanticDriftPanel: React.FC = () => {
  const [driftStatus, setDriftStatus] = useState('Nominal'); // 'Nominal', 'Minor Drift Detected', 'Critical Drift Detected'
  const [remediationStatus, setRemediationStatus] = useState('Idle'); // 'Idle', 'Initiating', 'Remediating', 'Complete'

  // Mock effect to change drift status for demo
  React.useEffect(() => {
      const timer = setTimeout(() => {
          setDriftStatus('Minor Drift Detected');
      }, 10000);
      return () => clearTimeout(timer);
  }, []);

  const initiateRemediation = () => {
    setRemediationStatus('Initiating QNN Remediation...');
    setTimeout(() => {
      setRemediationStatus('QNN Remediation in Progress...');
      setTimeout(() => {
        setDriftStatus('Nominal');
        setRemediationStatus('Remediation Complete');
      }, 4000); // Simulate remediation time
    }, 1500);
  };

  const getStatusColor = (status: string) => {
    if (status.includes('Critical')) return 'text-red-400';
    if (status.includes('Minor')) return 'text-yellow-300';
    if (status.includes('Nominal') || status.includes('Complete')) return 'text-green-300';
    return 'text-cyan-100';
  };

  return (
    <GlassPanel title='Semantic Drift & Remediation'>
      <div className="p-4 space-y-4 text-cyan-200 h-full flex flex-col">
        <div className="flex items-center space-x-2">
          <MagnifyingGlassIcon className="h-6 w-6 text-cyan-400" />
          <h3 className="text-lg font-semibold text-cyan-300">AI Model Semantic Integrity</h3>
        </div>
        <div className="bg-cyan-900/30 p-3 rounded-lg flex items-center justify-between">
          <span className="flex items-center"><BugAntIcon className="h-5 w-5 mr-3 text-cyan-500" /> Drift Status:</span>
          <span className={`font-medium ${getStatusColor(driftStatus)}`}>{driftStatus}</span>
        </div>
        <p className="text-xs text-cyan-300 opacity-80">
          Last Check: {new Date().toLocaleTimeString()} (QNN-powered anomaly detection)
        </p>

        <div className="flex-grow"></div>
        
        {driftStatus !== 'Nominal' && (
          <div className="border-t border-cyan-800 pt-4 mt-4 animate-fade-in">
            <div className="flex items-center space-x-2 mb-3">
              <ShieldCheckIcon className="h-6 w-6 text-cyan-400" />
              <h3 className="text-lg font-semibold text-cyan-300">Remediation Protocol</h3>
            </div>
            <p className="text-sm mb-3">Recommended: QNN-accelerated semantic re-alignment.</p>
            <p className="text-sm mb-3">Remediation Status: <span className={`font-medium ${getStatusColor(remediationStatus)}`}>{remediationStatus}</span></p>
            <button
              onClick={initiateRemediation}
              disabled={remediationStatus !== 'Idle' && remediationStatus !== 'Remediation Complete'}
              className="w-full holographic-button bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <span className="flex items-center justify-center">
                <ZapIcon className="h-5 w-5 mr-2" />
                {remediationStatus.includes('Progress') || remediationStatus.includes('Initiating') ? 'Remediating...' : 'Initiate QNN Remediation'}
              </span>
            </button>
          </div>
        )}
      </div>
    </GlassPanel>
  );
};

export default SemanticDriftPanel;
