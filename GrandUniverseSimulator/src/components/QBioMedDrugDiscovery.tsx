
import React, { useState } from 'react';
import GlassPanel from './GlassPanel';
import { AtomIcon, BeakerIcon, ChartBarIcon, PlayIcon, StopIcon, MagnifyingGlassIcon, UploadCloudIcon, AlertTriangleIcon, LoaderIcon } from './Icons';
import LoadingSkeleton from './LoadingSkeleton';

// --- Detail View for Simulation Report ---
const SimulationReportDetail = ({ result }: { result: any }) => {
    return (
        <div className="bg-black/20 p-4 animate-fade-in">
            <h4 className="font-semibold text-cyan-200 mb-3 text-base">Detailed Simulation Report</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                {/* Binding Affinity Profile */}
                <div className="md:col-span-1 bg-cyan-950/50 p-3 rounded-md">
                    <h5 className="font-bold text-cyan-300 mb-2">Binding Affinity Profile</h5>
                    <div className="space-y-1">
                        {result.bindingDistribution.map((value: number, i: number) => (
                            <div key={i} className="flex items-center">
                                <span className="w-10 text-cyan-500">{`-${8 + i * 0.5}`}</span>
                                <div className="flex-grow bg-cyan-800/50 rounded-full h-2">
                                    <div className="bg-cyan-400 h-2 rounded-full" style={{ width: `${value}%` }}></div>
                                </div>
                            </div>
                        ))}
                    </div>
                    <p className="text-cyan-600 mt-2 text-center">Energy Distribution (kcal/mol)</p>
                </div>

                {/* Quantum & Computational Details */}
                <div className="md:col-span-2 grid grid-cols-2 gap-4">
                    <div className="bg-cyan-950/50 p-3 rounded-md">
                        <h5 className="font-bold text-cyan-300 mb-2">Quantum State Analysis</h5>
                        <p className="font-mono text-cyan-100 whitespace-nowrap">{result.finalState}</p>
                        <p className="text-cyan-600 mt-1">Final Qubit State Vector</p>
                    </div>
                     <div className="bg-cyan-950/50 p-3 rounded-md">
                        <h5 className="font-bold text-cyan-300 mb-2">Computational Details</h5>
                        <ul className="font-mono text-cyan-100 space-y-1">
                            <li>Qubits Utilized: {result.qubitsUsed}</li>
                            <li>Circuit Depth: {result.circuitDepth}</li>
                            <li>Sim Time (QPU): {result.simulationTime}</li>
                        </ul>
                    </div>
                    <div className="col-span-2 bg-cyan-950/50 p-3 rounded-md">
                        <h5 className="font-bold text-cyan-300 mb-2">Conformer Analysis</h5>
                        <p className="text-cyan-200">{result.conformerAnalysis}</p>
                    </div>
                </div>
            </div>
        </div>
    );
};


const QBioMedDrugDiscovery: React.FC = () => {
    const [molecularStructure, setMolecularStructure] = useState<string | null>(null); // Stores actual structure data
    const [structureSource, setStructureSource] = useState<string | null>(null); // e.g., "PDB: 1ABC", "File: molecule.pdb"
    const [pdbIdInput, setPdbIdInput] = useState<string>('');
    const [simulationStatus, setSimulationStatus] = useState<'idle' | 'running' | 'completed'>('idle');
    const [simulationResults, setSimulationResults] = useState<any[]>([]);
    const [loading, setLoading] = useState<boolean>(false);
    const [dataLoading, setDataLoading] = useState<boolean>(false); // For loading molecular data
    const [isDraggingOver, setIsDraggingOver] = useState(false);
    const [expandedResultIndex, setExpandedResultIndex] = useState<number | null>(null);
    const [dataError, setDataError] = useState<string | null>(null);

    const processFile = (file: File | undefined) => {
        setDataError(null);
        if (file && (file.name.toLowerCase().endsWith('.pdb') || file.name.toLowerCase().endsWith('.xyz') || file.name.toLowerCase().endsWith('.mol'))) {
            setDataLoading(true);
            const reader = new FileReader();
            reader.onload = (e) => {
                // Simulate processing time
                setTimeout(() => {
                    setMolecularStructure(e.target?.result as string);
                    setStructureSource(`File: ${file.name}`);
                    setDataLoading(false);
                }, 1000);
            };
            reader.readAsText(file);
        } else if (file) {
            setDataError("Unsupported file type. Please upload a .pdb, .xyz, or .mol file.");
        }
    };

    const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        processFile(event.target.files?.[0]);
    };

    const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        setIsDraggingOver(true);
    };

    const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        setIsDraggingOver(false);
    };

    const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        setIsDraggingOver(false);
        processFile(event.dataTransfer.files?.[0]);
    };


    const handleLoadFromPDB = async () => {
        if (!pdbIdInput.trim()) {
            setDataError("Please enter a PDB ID.");
            return;
        }
        setDataLoading(true);
        setDataError(null);
        setMolecularStructure(null); // Clear previous data
        setStructureSource(null);

        try {
            // Simulate fetching from PDB. In a real application, you'd make an API call.
            console.log(`Attempting to load PDB ID: ${pdbIdInput}`);
            await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate network delay

            const dummyPDBContent = `
HEADER    DRUG TARGET COMPLEX
TITLE     SIMULATED PDB STRUCTURE FOR ${pdbIdInput.toUpperCase()}
... (full PDB content would be here) ...
ATOM      1  N   ALA A   1      21.220  17.770  19.530  1.00 12.00           N
ATOM      2  CA  ALA A   1      20.670  18.960  18.980  1.00 12.00           C
...
`;
            setMolecularStructure(dummyPDBContent); // Store the "fetched" content
            setStructureSource(`PDB ID: ${pdbIdInput.toUpperCase()}`);

        } catch (error) {
            console.error("Failed to load PDB data:", error);
            setDataError(`Failed to load PDB ID ${pdbIdInput.toUpperCase()}. Please check the ID or network connection.`);
            setMolecularStructure(null);
            setStructureSource(null);
        } finally {
            setDataLoading(false);
        }
    };

    const startSimulation = async () => {
        setLoading(true);
        setSimulationStatus('running');
        setSimulationResults([]); // Clear previous results
        setExpandedResultIndex(null); // Collapse any open reports
        // Simulate an asynchronous quantum simulation
        await new Promise(resolve => setTimeout(resolve, 5000)); // Simulate QPU processing time

        const dummyResults = [
            { 
                molecule: structureSource || "Unknown Molecule", target: "Protein X", bindingEnergy: -8.5, quantumScore: 0.92,
                qubitsUsed: 58, circuitDepth: 120, simulationTime: '4.2s', finalState: '0.82|01⟩ - 0.57|10⟩',
                bindingDistribution: [10, 30, 45, 15],
                conformerAnalysis: "Lowest energy conformer shows strong hydrophobic interaction at active site C. Pi-stacking observed with Phenylalanine residue."
            },
            { 
                molecule: "Drug B", target: "Protein X", bindingEnergy: -7.1, quantumScore: 0.78,
                qubitsUsed: 46, circuitDepth: 98, simulationTime: '3.1s', finalState: '0.61|01⟩ - 0.79|10⟩',
                bindingDistribution: [40, 35, 20, 5],
                conformerAnalysis: "Multiple stable conformers found. The primary conformation shows moderate hydrogen bonding potential."
            },
            { 
                molecule: "Drug C", target: "Protein X", bindingEnergy: -9.1, quantumScore: 0.95,
                qubitsUsed: 62, circuitDepth: 155, simulationTime: '5.8s', finalState: '0.91|01⟩ - 0.41|10⟩',
                bindingDistribution: [5, 15, 30, 50],
                conformerAnalysis: "Single, highly stable conformer identified. Excellent geometric and electrostatic complementarity to the binding pocket."
            },
        ];
        setSimulationResults(dummyResults);
        setLoading(false);
        setSimulationStatus('completed');
    };

    return (
        <GlassPanel title={<div className="flex items-center"><BeakerIcon className="w-5 h-5 mr-2 text-pink-400" /> Q-BioMed Drug Discovery</div>}>
            <div className="flex flex-col h-full gap-4 p-4 overflow-y-auto custom-scrollbar">
                {/* Input Section */}
                <div className="bg-black/30 p-4 rounded-lg border border-cyan-800/50">
                    <div className="flex flex-col md:flex-row gap-4 items-end">
                         {/* PDB Input */}
                         <div className="flex-grow w-full">
                            <label className="text-xs font-bold text-cyan-500 uppercase mb-1 block">Target Protein (PDB ID)</label>
                            <div className="flex gap-2">
                                <input 
                                    type="text" 
                                    value={pdbIdInput} 
                                    onChange={(e) => setPdbIdInput(e.target.value)} 
                                    placeholder="e.g., 6LU7" 
                                    className="bg-black/50 border border-cyan-700 text-white text-sm rounded px-3 py-2 flex-grow outline-none focus:border-cyan-400"
                                    disabled={dataLoading || loading}
                                />
                                <button 
                                    onClick={handleLoadFromPDB}
                                    disabled={dataLoading || loading}
                                    className="p-2 bg-cyan-900/30 border border-cyan-600 rounded text-cyan-400 hover:text-white disabled:opacity-50 min-w-[40px] flex items-center justify-center"
                                >
                                    {dataLoading ? <LoaderIcon className="w-5 h-5 animate-spin"/> : <MagnifyingGlassIcon className="w-5 h-5"/>}
                                </button>
                            </div>
                         </div>
                         
                         {/* File Upload */}
                         <div className="flex-grow w-full">
                            <label className="text-xs font-bold text-cyan-500 uppercase mb-1 block">Ligand Structure</label>
                            <div 
                                className={`border-2 border-dashed rounded-lg p-2 text-center transition-colors cursor-pointer ${isDraggingOver ? 'border-green-400 bg-green-900/20' : 'border-cyan-800 hover:border-cyan-600'}`}
                                onDragOver={handleDragOver}
                                onDragLeave={handleDragLeave}
                                onDrop={handleDrop}
                            >
                                <label className="cursor-pointer flex items-center justify-center gap-2 text-xs text-gray-400 w-full h-full">
                                    {dataLoading ? (
                                        <><LoaderIcon className="w-4 h-4 animate-spin"/> Processing...</>
                                    ) : (
                                        <>
                                            <UploadCloudIcon className="w-4 h-4" />
                                            <span>{structureSource ? structureSource : "Drop .pdb/.mol file"}</span>
                                        </>
                                    )}
                                    <input type="file" className="hidden" onChange={handleFileUpload} accept=".pdb,.mol,.xyz" disabled={dataLoading || loading} />
                                </label>
                            </div>
                         </div>
                    </div>
                    {dataError && <p className="text-xs text-red-400 mt-2 flex items-center gap-1"><AlertTriangleIcon className="w-3 h-3"/> {dataError}</p>}
                </div>

                {/* Simulation Control */}
                <div className="flex justify-center">
                    <button 
                        onClick={startSimulation}
                        disabled={loading || !molecularStructure}
                        className={`holographic-button px-8 py-3 rounded-lg font-bold text-sm flex items-center gap-2 transition-all ${loading || !molecularStructure ? 'bg-cyan-900/50 border-cyan-800 text-gray-400 cursor-not-allowed opacity-50' : 'bg-pink-600/30 border-pink-500 text-pink-200 hover:bg-pink-600/50'}`}
                    >
                        {loading ? <LoaderIcon className="w-4 h-4 animate-spin" /> : <PlayIcon className="w-4 h-4" />}
                        {loading ? 'Running Quantum Simulation...' : 'Run Binding Simulation'}
                    </button>
                </div>

                {/* Results Area */}
                <div className="flex-grow min-h-0 relative bg-black/40 border border-cyan-900/30 rounded-lg overflow-hidden flex flex-col">
                    <div className="p-2 border-b border-cyan-900/30 bg-cyan-950/20 text-xs font-bold text-cyan-300">
                        Simulation Results
                    </div>
                    <div className="flex-grow overflow-y-auto p-2 space-y-2 custom-scrollbar">
                        {loading ? (
                            <div className="p-4">
                                <LoadingSkeleton lines={4} className="h-20" />
                            </div>
                        ) : simulationResults.length === 0 ? (
                            <div className="h-full flex flex-col items-center justify-center text-gray-600 opacity-50">
                                <AtomIcon className="w-16 h-16 mb-2" />
                                <p className="text-xs">No simulation data generated.</p>
                            </div>
                        ) : (
                            simulationResults.map((res, idx) => (
                                <div key={idx} className="bg-black/40 border border-cyan-800/40 rounded p-3 hover:bg-cyan-900/10 transition-colors">
                                    <div 
                                        className="flex justify-between items-center cursor-pointer"
                                        onClick={() => setExpandedResultIndex(expandedResultIndex === idx ? null : idx)}
                                    >
                                        <div className="flex items-center gap-3">
                                            <div className="p-2 bg-pink-900/20 rounded text-pink-400"><BeakerIcon className="w-4 h-4"/></div>
                                            <div>
                                                <p className="text-sm font-bold text-white">{res.molecule}</p>
                                                <p className="text-xs text-gray-500">Target: {res.target}</p>
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <p className="text-sm font-mono text-green-400">{res.bindingEnergy} kcal/mol</p>
                                            <p className="text-[10px] text-cyan-600">Score: {res.quantumScore}</p>
                                        </div>
                                    </div>
                                    {expandedResultIndex === idx && (
                                        <div className="mt-3 pt-3 border-t border-gray-800">
                                            <SimulationReportDetail result={res} />
                                        </div>
                                    )}
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </GlassPanel>
    );
};

export default QBioMedDrugDiscovery;
