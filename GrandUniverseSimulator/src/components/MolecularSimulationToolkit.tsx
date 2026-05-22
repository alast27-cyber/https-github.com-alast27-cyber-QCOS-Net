
import React, { useState } from 'react';
import GlassPanel from './GlassPanel';
import { AtomIcon, PlayIcon, StopIcon, GlobeIcon, CubeTransparentIcon, CodeBracketIcon, ChartBarIcon, LoaderIcon } from './Icons';

// Define the structure for simulation results
interface SimulationResult {
  molecule: string;
  energy: number;
  bondLengths: { [key: string]: string };
  bondAngles: { [key: string]: string };
  qubitState: string;
}

// Controller Panel Component
const MolecularSimulationController: React.FC<{
    isRunning: boolean;
    status: string;
    startSimulation: () => void;
    stopSimulation: () => void;
    moleculeName: string;
    setMoleculeName: (name: string) => void;
    qubits: number;
    setQubits: (q: number) => void;
    steps: number;
    setSteps: (s: number) => void;
}> = ({
    isRunning, status, startSimulation, stopSimulation,
    moleculeName, setMoleculeName, qubits, setQubits, steps, setSteps
}) => {
    return (
        <GlassPanel title='Simulation Controller'>
          <div className='p-4 space-y-4 text-cyan-200 h-full flex flex-col'>
            <div>
              <label htmlFor='moleculeName' className='block text-sm font-medium text-cyan-300'>Molecule Name:</label>
              <input
                type='text' id='moleculeName' value={moleculeName} onChange={(e) => setMoleculeName(e.target.value)}
                className='mt-1 block w-full bg-cyan-900/50 border border-cyan-700 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-1 focus:ring-cyan-500 text-cyan-100'
                disabled={isRunning}
              />
            </div>
            <div>
              <label htmlFor='qubits' className='block text-sm font-medium text-cyan-300'>Number of Qubits:</label>
              <input
                type='number' id='qubits' value={qubits} onChange={(e) => setQubits(parseInt(e.target.value))} min='1' max='32'
                className='mt-1 block w-full bg-cyan-900/50 border border-cyan-700 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-1 focus:ring-cyan-500 text-cyan-100'
                disabled={isRunning}
              />
            </div>
            <div>
              <label htmlFor='steps' className='block text-sm font-medium text-cyan-300'>Simulation Steps:</label>
              <input
                type='number' id='steps' value={steps} onChange={(e) => setSteps(parseInt(e.target.value))} min='1'
                className='mt-1 block w-full bg-cyan-900/50 border border-cyan-700 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-1 focus:ring-cyan-500 text-cyan-100'
                disabled={isRunning}
              />
            </div>
            <div className='flex-grow' />
            <div className='flex-shrink-0'>
                <p className="text-xs text-cyan-400 pt-4 flex items-center">
                    <GlobeIcon className="w-4 h-4 mr-1 text-cyan-400" />
                    Application can be assigned a public HTTPS URL.
                </p>
                <div className='flex justify-between items-center pt-2 mt-2 border-t border-cyan-500/30'>
                  <span className={`text-sm font-medium truncate pr-2 ${isRunning ? 'text-yellow-400 animate-pulse' : 'text-green-400'}`}>
                    {status}
                  </span>
                  {!isRunning ? (
                    <button onClick={startSimulation} className='holographic-button flex items-center px-4 py-2 border border-cyan-500/50 text-sm font-medium rounded-md shadow-sm text-white bg-cyan-600/30 hover:bg-cyan-700/50'>
                      <PlayIcon className='-ml-1 mr-2 h-5 w-5' aria-hidden='true' />
                      Start
                    </button>
                  ) : (
                    <button onClick={stopSimulation} className='holographic-button flex items-center px-4 py-2 border border-red-500/50 text-sm font-medium rounded-md shadow-sm text-white bg-red-600/30 hover:bg-red-700/50'>
                      <StopIcon className='-ml-1 mr-2 h-5 w-5' aria-hidden='true' />
                      Stop
                    </button>
                  )}
                </div>
            </div>
          </div>
        </GlassPanel>
    );
};

// Visualizer Panel Component
const MolecularStructureVisualizer: React.FC<{
    results: SimulationResult | null;
    isRunning: boolean;
}> = ({ results, isRunning }) => {
    const renderContent = () => {
        if (isRunning) {
            return (
                <div className="text-center text-cyan-400 h-full flex flex-col items-center justify-center">
                    <LoaderIcon className="w-8 h-8 animate-spin" />
                    <p className="mt-2">Simulation in progress...</p>
                </div>
            )
        }
        if (!results) {
            return (
                 <div className="text-center text-cyan-600 h-full flex flex-col items-center justify-center">
                    <AtomIcon className="w-12 h-12" />
                    <p className="mt-2">Run simulation to view results.</p>
                </div>
            )
        }
        return (
            <div className="space-y-4">
                <div className='w-full h-48 bg-black/30 border border-cyan-900 rounded-md flex items-center justify-center text-cyan-700 text-sm italic'>
                    <p>3D Molecular Model</p>
                    <CodeBracketIcon className="w-10 h-10 ml-2" />
                </div>
                <div>
                    <h3 className='text-lg font-medium text-cyan-200 flex items-center mb-2'><ChartBarIcon className="w-5 h-5 mr-2" />Simulation Output</h3>
                    <ul className='space-y-1 text-sm'>
                        <li className="flex justify-between"><span className='font-semibold text-cyan-300'>Molecule:</span> <span className="font-mono">{results.molecule}</span></li>
                        <li className="flex justify-between"><span className='font-semibold text-cyan-300'>Ground State Energy:</span> <span className="font-mono">{results.energy.toFixed(3)} Hartrees</span></li>
                        <li><span className='font-semibold text-cyan-300'>Final Qubit State:</span> <span className="font-mono">{results.qubitState}</span></li>
                    </ul>
                </div>
            </div>
        )
    }

    return (
        <GlassPanel title='Structure Visualizer'>
            <div className='p-4 text-cyan-200 h-full'>{renderContent()}</div>
        </GlassPanel>
    );
};


// Main Toolkit Component
const MolecularSimulationToolkit: React.FC = () => {
  const [moleculeName, setMoleculeName] = useState('H2O');
  const [qubits, setQubits] = useState(4);
  const [steps, setSteps] = useState(10);
  const [status, setStatus] = useState('Idle');
  const [isRunning, setIsRunning] = useState(false);
  const [simulationResults, setSimulationResults] = useState<SimulationResult | null>(null);
  const timeoutRef = React.useRef<number | null>(null);

  const startSimulation = () => {
    setStatus(`Simulating ${moleculeName}...`);
    setIsRunning(true);
    setSimulationResults(null);
    
    timeoutRef.current = window.setTimeout(() => {
      const newResults: SimulationResult = {
        molecule: `${moleculeName} (Simulated)`,
        energy: -76.321 + (Math.random() - 0.5) * 5,
        bondLengths: { 'O-H1': '0.958 Å', 'O-H2': '0.958 Å' },
        bondAngles: { 'H-O-H': '104.45°' },
        qubitState: `|${[...Array(qubits)].map(() => Math.round(Math.random())).join('')}>`
      };
      setSimulationResults(newResults);
      setStatus('Simulation Complete!');
      setIsRunning(false);
    }, 5000);
  };

  const stopSimulation = () => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setStatus('Simulation Stopped.');
    setIsRunning(false);
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
      <MolecularSimulationController 
        isRunning={isRunning} 
        status={status}
        startSimulation={startSimulation}
        stopSimulation={stopSimulation}
        moleculeName={moleculeName}
        setMoleculeName={setMoleculeName}
        qubits={qubits}
        setQubits={setQubits}
        steps={steps}
        setSteps={setSteps}
      />
      <MolecularStructureVisualizer 
        results={simulationResults}
        isRunning={isRunning}
      />
    </div>
  );
};

export default MolecularSimulationToolkit;
