import React, { useState } from 'react';
import { MagnifyingGlassIcon, LoaderIcon } from './Icons';

const QuantumDataSearchPanel: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [searchResults, setSearchResults] = useState<string[]>([]);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [hasSearched, setHasSearched] = useState<boolean>(false);

  const handleSearch = async () => {
    if (!searchTerm.trim()) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    setHasSearched(true);
    setSearchResults([]);

    await new Promise(resolve => setTimeout(resolve, 1500));

    // Dummy results based on keywords
    const dummyResults: string[] = [];
    const lowerSearchTerm = searchTerm.toLowerCase();

    if (lowerSearchTerm.includes('superposition')) {
        dummyResults.push(`Found Q-Lang script: 'create_superposition.ql'`);
        dummyResults.push(`Academic Thesis Match: "Long-lived Superposition States in Ytterbium Atoms"`);
    } else if (lowerSearchTerm.includes('entanglement')) {
        dummyResults.push(`CHIPS Application Found: 'Quantum Network Visualizer'`);
        dummyResults.push(`Found Q-Lang script: 'bell_state_entanglement.ql'`);
        dummyResults.push(`Retrieved 32 results for "${searchTerm}" in Qubit Register Database.`);
    } else {
        dummyResults.push(`Found 5 relevant Q-Lang scripts related to "${searchTerm}".`);
        dummyResults.push(`Identified 2 CHIPS applications referencing "${searchTerm}".`);
    }

    setSearchResults(dummyResults);
    setIsSearching(false);
  };

  const renderResults = () => {
    if (isSearching) {
        return (
            <div className="text-center text-cyan-400 h-full flex flex-col items-center justify-center">
                <LoaderIcon className="w-8 h-8 animate-spin mb-2" />
                <p>Searching CHIPS network for "{searchTerm}"...</p>
            </div>
        );
    }
    if (searchResults.length > 0) {
        return (
            <div className="font-mono text-xs text-cyan-300 space-y-1">
                {searchResults.map((result, index) => (
                    <p key={index} className="animate-fade-in">
                        <span className="text-cyan-400">&gt; </span>
                        {result}
                    </p>
                ))}
            </div>
        );
    }
    if (hasSearched) {
        return <p className="text-center text-red-400">No results found for "{searchTerm}".</p>;
    }
    return <p className="text-center text-cyan-600">Enter a query to begin your quantum data search.</p>;
  };

  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="flex space-x-2">
        <input
          type="text"
          className="flex-grow p-2 rounded-md bg-black/30 border border-blue-500/50 text-white placeholder:text-gray-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
          placeholder="e.g., 'superposition', 'entanglement protocol'"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          disabled={isSearching}
        />
        <button
          onClick={handleSearch}
          disabled={isSearching || !searchTerm.trim()}
          className="w-32 flex items-center justify-center px-4 py-2 rounded-md bg-cyan-500/30 hover:bg-cyan-500/50 border border-cyan-500/50 text-cyan-200 font-bold transition-colors duration-200 disabled:opacity-50"
        >
          {isSearching ? <LoaderIcon className="w-5 h-5 animate-spin"/> : 'Search'}
        </button>
      </div>

      <div className="flex-grow overflow-auto p-3">
        {renderResults()}
      </div>
    </div>
  );
};

export default QuantumDataSearchPanel;