import React from 'react';
import GlassPanel from './GlassPanel';
import {
  GlobeIcon,
  BuildingFarmIcon,
  UsersIcon,
  CpuChipIcon,
} from './Icons';

interface QuantumSwineIntelligenceProps {
    onOpenApp: (appId: string) => void;
}

const QuantumSwineIntelligence: React.FC<QuantumSwineIntelligenceProps> = ({ onOpenApp }) => {

  return (
    <GlassPanel
      title={<div className="flex items-center"><CpuChipIcon className='w-6 h-6 text-cyan-300 mr-2' /> Quantum Swine Intelligence</div>}
    >
      <div className='flex flex-col space-y-4 p-4 h-full overflow-y-auto'>
        <p className='text-sm text-gray-300 mb-4'>
          The Quantum Swine Intelligence (QSI) ecosystem leverages Agent Q's quantum core
          to deliver advanced insights and optimization across global, national, and
          consumer-facing aspects of the swine industry. Each module is a
          separate, quantum-powered application deployable via the new
          <span className='text-cyan-400 font-medium'> Quantum-to-Web Gateway</span>.
        </p>

        <div className='grid grid-cols-1 md:grid-cols-3 gap-6'>

          {/* Global Swine Foresight Card */}
          <div className='bg-gray-800 bg-opacity-70 border border-cyan-700 rounded-lg p-6 flex flex-col items-start shadow-lg hover:shadow-cyan-500/30 transition-shadow duration-300'>
            <GlobeIcon className='w-12 h-12 text-cyan-400 mb-4' />
            <h3 className='text-xl font-semibold text-cyan-300 mb-2'>Global Swine Foresight</h3>
            <p className='text-gray-300 text-sm mb-4 flex-grow'>
              Strategic insights for global markets, commodity forecasting (QMC-Finance),
              and predictive biosecurity modeling (QNN) to optimize intercontinental logistics
              and respond to global threats.
            </p>
            <button
              onClick={() => onOpenApp('global-swine-foresight')}
              title="Launch the Global Swine Foresight application"
              className='mt-4 px-6 py-2 bg-cyan-600 text-white font-semibold rounded-md hover:bg-cyan-700 transition-colors duration-200 shadow-md holographic-button'
            >
              Open App
            </button>
            <p className='text-xs text-gray-400 mt-2'>https://qsi-global.apps.web</p>
          </div>

          {/* Philippine Swine Resilience Card */}
          <div className='bg-gray-800 bg-opacity-70 border border-cyan-700 rounded-lg p-6 flex flex-col items-start shadow-lg hover:shadow-cyan-500/30 transition-shadow duration-300'>
            <BuildingFarmIcon className='w-12 h-12 text-cyan-400 mb-4' />
            <h3 className='text-xl font-semibold text-cyan-300 mb-2'>Philippine Swine Resilience</h3>
            <p className='text-gray-300 text-sm mb-4 flex-grow'>
              Quantum-optimized solutions targeting Philippine bottlenecks: QML for genetics,
              Q-Lang for dynamic feed optimization, quantum supply chain logistics,
              and QNN for management and capital access.
            </p>
            <button
              onClick={() => onOpenApp('philippine-swine-resilience')}
              title="Launch the Philippine Swine Resilience application"
              className='mt-4 px-6 py-2 bg-cyan-600 text-white font-semibold rounded-md hover:bg-cyan-700 transition-colors duration-200 shadow-md holographic-button'
            >
              Open App
            </button>
            <p className='text-xs text-gray-400 mt-2'>https://qsi-ph.apps.web</p>
          </div>

          {/* PigHaven Consumer Trust Card */}
          <div className='bg-gray-800 bg-opacity-70 border border-cyan-700 rounded-lg p-6 flex flex-col items-start shadow-lg hover:shadow-cyan-500/30 transition-shadow duration-300'>
            <UsersIcon className='w-12 h-12 text-cyan-400 mb-4' />
            <h3 className='text-xl font-semibold text-cyan-300 mb-2'>PigHaven Consumer Trust</h3>
            <p className='text-gray-300 text-sm mb-4 flex-grow'>
              Empowering general users with quantum-secured traceability (QKD principles),
              real-time availability, and pricing insights, fostering transparency and food security.
            </p>
            <button
              onClick={() => onOpenApp('pighaven-consumer-trust')}
              title="Launch the PigHaven Consumer Trust application"
              className='mt-4 px-6 py-2 bg-cyan-600 text-white font-semibold rounded-md hover:bg-cyan-700 transition-colors duration-200 shadow-md holographic-button'
            >
              Open App
            </button>
            <p className='text-xs text-gray-400 mt-2'>https://qsi-consumer.apps.web</p>
          </div>

        </div>
      </div>
    </GlassPanel>
  );
};

export default QuantumSwineIntelligence;