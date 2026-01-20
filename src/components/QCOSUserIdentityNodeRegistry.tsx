import React from 'react';
import GlassPanel from './GlassPanel';
import {
  CpuChipIcon,
  ShieldCheckIcon,
  SparklesIcon,
  KeyIcon,
  Bars3BottomLeftIcon,
  ArrowPathIcon,
} from './Icons';

interface QCOSUserIdentityNodeRegistryProps {
    onLogin: () => void;
    onRegister: () => void;
}

const QCOSUserIdentityNodeRegistry: React.FC<QCOSUserIdentityNodeRegistryProps> = ({ onLogin, onRegister }) => {
  return (
    <GlassPanel title="QCOS User Identity & Node Registry">
      <div className="p-4 h-full overflow-y-auto">
        <div className="mb-6 border-b border-cyan-800/50 pb-4">
          <p className="text-sm text-cyan-200 mb-3">
            Your CHIPS Browser acts as your personal QCOS node. Register or log in to activate your decentralized quantum identity and access network resources, including deploying applications with public HTTPS URLs via the Quantum-to-Web Gateway.
          </p>
          <h3 className="text-lg font-semibold text-cyan-300 mb-3">User Authentication</h3>
          
          <div className="mb-5 p-4 bg-black/30 rounded-lg border border-cyan-900">
            <h4 className="font-medium text-cyan-200 mb-2">Register Your QCOS Node</h4>
            <input
              type="text"
              placeholder="Desired Username"
              className="w-full p-2 rounded bg-cyan-900/50 text-white placeholder-cyan-500 mb-2 border border-cyan-700 focus:ring-1 focus:ring-cyan-500 focus:border-cyan-500"
            />
            <input
              type="email"
              placeholder="Email (for node verification)"
              className="w-full p-2 rounded bg-cyan-900/50 text-white placeholder-cyan-500 mb-2 border border-cyan-700 focus:ring-1 focus:ring-cyan-500 focus:border-cyan-500"
            />
            <input
              type="password"
              placeholder="Password"
              className="w-full p-2 rounded bg-cyan-900/50 text-white placeholder-cyan-500 mb-3 border border-cyan-700 focus:ring-1 focus:ring-cyan-500 focus:border-cyan-500"
            />
            <button onClick={onRegister} className="w-full p-2 rounded bg-cyan-600/50 hover:bg-cyan-700/70 text-white font-bold transition-colors duration-200 border border-cyan-500">
              Register QCOS Node
            </button>
            <p className="text-xs text-cyan-500 mt-2">
              Verification ensures one unique QCOS node per person, contributing to network security.
            </p>
          </div>

          <div className="p-4 bg-black/30 rounded-lg border border-cyan-900">
            <h4 className="font-medium text-cyan-200 mb-2">Login to QCOS Network</h4>
            <input
              type="text"
              placeholder="Username or Node ID"
              className="w-full p-2 rounded bg-cyan-900/50 text-white placeholder-cyan-500 mb-2 border border-cyan-700 focus:ring-1 focus:ring-cyan-500 focus:border-cyan-500"
            />
            <input
              type="password"
              placeholder="Password"
              className="w-full p-2 rounded bg-cyan-900/50 text-white placeholder-cyan-500 mb-3 border border-cyan-700 focus:ring-1 focus:ring-cyan-500 focus:border-cyan-500"
            />
            <button onClick={onLogin} className="w-full p-2 rounded bg-cyan-600/50 hover:bg-cyan-700/70 text-white font-bold transition-colors duration-200 border border-cyan-500">
              Login
            </button>
          </div>
        </div>

        <div className="mb-6 border-b border-cyan-800/50 pb-4">
          <h3 className="text-lg font-semibold text-cyan-300 mb-3">Your QCOS Node Status</h3>
          <div className="flex items-center justify-between bg-black/30 p-3 rounded-lg mb-2 border border-cyan-900">
            <span className="text-cyan-200 flex items-center">
              <CpuChipIcon className="h-5 w-5 mr-2 text-cyan-400" /> Node ID:
            </span>
            <span className="text-white font-mono text-sm">QCOS-USR-01A2C3F5</span>
          </div>
          <div className="flex items-center justify-between bg-black/30 p-3 rounded-lg border border-cyan-900">
            <span className="text-cyan-200 flex items-center">
              <ShieldCheckIcon className="h-5 w-5 mr-2 text-cyan-400" /> Safety Rating:
            </span>
            <span className="text-green-400 font-bold flex items-center text-sm">
              Excellent (5.0)
              <ArrowPathIcon className="h-4 w-4 ml-2 text-gray-400 animate-spin-slow" />
            </span>
          </div>
        </div>

        <div>
          <h3 className="text-lg font-semibold text-cyan-300 mb-3">Integrated Quantum Protocols</h3>
          <p className="text-sm text-cyan-400 mb-3">
            Your QCOS node is equipped with the following core quantum capabilities:
          </p>
          <ul className="space-y-2">
            <li className="flex items-center bg-black/30 p-3 rounded-lg border border-cyan-900">
              <CpuChipIcon className="h-5 w-5 text-cyan-400 mr-3" />
              <span className="text-white">Quantum Simulator (3 Qubits)</span>
              <span className="ml-auto px-2 py-1 text-xs font-semibold rounded-full bg-green-900/50 text-green-300 flex items-center border border-green-700">
                <span className="relative flex h-2 w-2 mr-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                </span>
                Active
              </span>
            </li>
            <li className="flex items-center bg-black/30 p-3 rounded-lg border border-cyan-900">
              <SparklesIcon className="h-5 w-5 text-cyan-400 mr-3" />
              <span className="text-white">Quantum Entanglement Protocol</span>
              <span className="ml-auto px-2 py-1 text-xs font-semibold rounded-full bg-green-900/50 text-green-300 flex items-center border border-green-700">
                 <span className="relative flex h-2 w-2 mr-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                </span>
                Active
              </span>
            </li>
            <li className="flex items-center bg-black/30 p-3 rounded-lg border border-cyan-900">
              <Bars3BottomLeftIcon className="h-5 w-5 text-cyan-400 mr-3" />
              <span className="text-white">Quantum Superposition Engine</span>
              <span className="ml-auto px-2 py-1 text-xs font-semibold rounded-full bg-green-900/50 text-green-300 flex items-center border border-green-700">
                 <span className="relative flex h-2 w-2 mr-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                </span>
                Active
              </span>
            </li>
            <li className="flex items-center bg-black/30 p-3 rounded-lg border border-cyan-900">
              <KeyIcon className="h-5 w-5 text-cyan-400 mr-3" />
              <span className="text-white">BB84 QKD Protocol</span>
              <span className="ml-auto px-2 py-1 text-xs font-semibold rounded-full bg-green-900/50 text-green-300 flex items-center border border-green-700">
                 <span className="relative flex h-2 w-2 mr-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                </span>
                Active
              </span>
            </li>
          </ul>
          <p className="text-xs text-cyan-600 mt-3">
            These protocols are deeply integrated into your QCOS node's core functionality, enabling full participation in the quantum network.
          </p>
        </div>
      </div>
    </GlassPanel>
  );
};

export default QCOSUserIdentityNodeRegistry;