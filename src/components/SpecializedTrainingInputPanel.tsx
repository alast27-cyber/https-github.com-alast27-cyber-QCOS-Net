import React, { useState } from 'react';
import GlassPanel from './GlassPanel';
import { AcademicCapIcon } from './Icons';

const SpecializedTrainingInputPanel: React.FC = () => {
  const [trainingInput, setTrainingInput] = useState<string>('');

  const handleSimulate = () => {
    if (trainingInput.trim()) {
      console.log('Initiating specialized training simulation with input:', trainingInput);
      // In a real scenario, this would trigger an API call or internal QNN function
      // to start the simulation using the provided text.
      alert('Training simulation initiated. Check AI Operations logs for status.');
      setTrainingInput(''); // Clear the input after submission
    } else {
      alert('Please enter simulation parameters.');
    }
  };

  return (
    <GlassPanel
      title={<div className="flex items-center"><AcademicCapIcon className='h-5 w-5 text-cyan-400 mr-2' /><span>Specialized Training Simulation</span></div>}
    >
      <div className='flex flex-col space-y-4 p-4 h-full'>
        <p className='text-sm text-gray-300'>
          Admins can input detailed scenarios or parameters here to run specialized training simulations for Agent Q's QNN core.
        </p>
        <textarea
          className='w-full p-3 rounded-md bg-black/30 border border-cyan-600 focus:ring-cyan-500 focus:border-cyan-500 shadow-inner text-cyan-200 flex-grow resize-none'
          placeholder='Describe the specialized training scenario or input Q-Lang based parameters for QNN evolution (e.g., "Simulate quantum entanglement anomaly detection on network traffic from the CHIPS network during a solar flare event." or "LOAD QNN_MODEL_V7; TRAIN_ON DATASET_CHIPS_SECURITY_V3 WITH 1000 EPOCHS; OPTIMIZE USING ADAPTIVE_QUANTUM_GRADIENT_DESCENT;").'
          value={trainingInput}
          onChange={(e) => setTrainingInput(e.target.value)}
        ></textarea>
        <button
          className='flex-shrink-0 mt-2 px-6 py-2 bg-cyan-600 text-white font-semibold rounded-md hover:bg-cyan-700 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-gray-800 transition-colors duration-200 holographic-button'
          onClick={handleSimulate}
        >
          Initiate Simulation
        </button>
      </div>
    </GlassPanel>
  );
};

export default SpecializedTrainingInputPanel;
