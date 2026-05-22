
import React, { useState } from 'react';
import GlassPanel from './GlassPanel';
import { ClockIcon, ChartBarIcon, ServerCogIcon } from './Icons'; // Replaced ServerStackIcon

const PredictiveTaskOrchestrationPanel: React.FC = () => {
  const [scheduleStatus, setScheduleStatus] = useState('Awaiting Analysis');

  const analyzeAndOptimize = () => {
    setScheduleStatus('Analyzing and Optimizing...');
    setTimeout(() => {
      setScheduleStatus('Optimized Schedule Applied');
    }, 2500); // Simulate analysis time
  };

  return (
    <GlassPanel title='Predictive Task Orchestration'>
      <div className="p-4 space-y-4 text-cyan-200 h-full flex flex-col">
        <div className="flex items-center space-x-2">
          <ClockIcon className="h-6 w-6 text-cyan-400" />
          <h3 className="text-lg font-semibold text-cyan-300">AI Workload Forecast (Next 48h)</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
          <div className="bg-cyan-900/30 p-3 rounded-lg flex items-center">
            <ChartBarIcon className="h-5 w-5 mr-3 text-cyan-500" />
            <span>Expected Peaks: <span className="font-medium text-cyan-100">14:00, 08:00</span></span>
          </div>
          <div className="bg-cyan-900/30 p-3 rounded-lg flex items-center">
            <ServerCogIcon className="h-5 w-5 mr-3 text-cyan-500" />
            <span>Predicted Bottlenecks: <span className="font-medium text-yellow-300">QPU-03</span></span>
          </div>
        </div>

        <div className="border-t border-cyan-800 pt-4 mt-4 flex-grow flex flex-col">
          <div className="flex items-center space-x-2 mb-3">
            <ChartBarIcon className="h-6 w-6 text-cyan-400" />
            <h3 className="text-lg font-semibold text-cyan-300">Optimization Control</h3>
          </div>
          <p className="text-sm mb-3">Current Status: <span className={`font-medium ${scheduleStatus.includes('Optimized') ? 'text-green-300' : 'text-yellow-300'}`}>{scheduleStatus}</span></p>
          <div className="flex-grow"></div>
          <button
            onClick={analyzeAndOptimize}
            disabled={scheduleStatus.includes('Analyzing')}
            className="w-full holographic-button bg-cyan-600 hover:bg-cyan-700 text-white font-bold py-2 px-4 rounded transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {scheduleStatus.includes('Analyzing') ? 'Optimizing...' : 'Apply Optimized Schedule'}
          </button>
        </div>
      </div>
    </GlassPanel>
  );
};

export default PredictiveTaskOrchestrationPanel;
