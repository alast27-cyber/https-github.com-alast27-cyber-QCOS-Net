
import React, { useState, useEffect } from 'react';
import GlassPanel from './GlassPanel';
import {
  CpuChipIcon, GlobeIcon, CodeBracketIcon, ChartBarIcon, RocketLaunchIcon,
  GitBranchIcon, ClipboardIcon, XIcon, ArrowPathIcon, PlayIcon, LoaderIcon
} from './Icons';
import LoadingSkeleton from './LoadingSkeleton';

interface AssetInput {
  id: number;
  name: string;
  initialPrice: string;
  volatility: string;
  correlation: string;
}

interface StockData {
  symbol: string;
  price: number;
  change: number;
  changePercent: string;
  volume: number;
  volatility: number;
}

const QuantumMonteCarloFinance: React.FC = () => {
  // --- STATE MANAGEMENT ---
  const [qaeEnabled, setQaeEnabled] = useState(false);
  const [qaePrecision, setQaePrecision] = useState('0.01');
  const [stressScenario, setStressScenario] = useState('');
  const [appName, setAppName] = useState('my-qmc-risk-model');
  const [publicUrl, setPublicUrl] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [showQlang, setShowQlang] = useState(false);
  const [generatedQlang, setGeneratedQlang] = useState('');
  const [assets, setAssets] = useState<AssetInput[]>([
    { id: 1, name: 'Stock A', initialPrice: '100', volatility: '0.2', correlation: 'N/A' }
  ]);
  const [nextAssetId, setNextAssetId] = useState(2);
  const [isSimulating, setIsSimulating] = useState(false);
  const [isDeploying, setIsDeploying] = useState(false);
  const [simulationResult, setSimulationResult] = useState<string | null>(null);

  // Live feed state
  const [liveStockData, setLiveStockData] = useState<StockData | null>(null);
  const [isFeedLoading, setIsFeedLoading] = useState<boolean>(true);
  const [feedError, setFeedError] = useState<string | null>(null);

  // --- STYLING CONSTANTS ---
  const inputClasses = "w-full p-2 bg-black/30 border border-cyan-800 rounded-md text-white placeholder:text-cyan-600 focus:ring-1 focus:ring-cyan-400 focus:outline-none transition duration-150 ease-in-out disabled:opacity-50";
  const subPanelClasses = "p-4 bg-black/20 rounded-lg shadow-inner border border-cyan-800/50";
  const holographicButtonClasses = "holographic-button w-full flex items-center justify-center font-bold py-2 px-4 rounded-md transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed";

  // --- LOGIC ---
  const fetchStockData = async () => {
    setIsFeedLoading(true);
    setFeedError(null);
    setLiveStockData(null); // Clear previous data to show skeleton
    
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    try {
      const mockData: StockData = {
        symbol: "QBIT",
        price: 175.68 + (Math.random() - 0.5) * 10,
        change: 1.23 + (Math.random() - 0.5) * 2,
        changePercent: `${(0.71 + (Math.random() - 0.5) * 0.5).toFixed(2)}%`,
        volume: 1234567 + Math.floor(Math.random() * 100000),
        volatility: 0.25 + (Math.random() - 0.5) * 0.05
      };
      setLiveStockData(mockData);
    } catch (err) {
      setFeedError("Failed to load stock data. Please refresh.");
    } finally {
      setIsFeedLoading(false);
    }
  };

  useEffect(() => {
    fetchStockData();
    const interval = setInterval(fetchStockData, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleImportData = () => {
    if (liveStockData) {
      setAssets(prevAssets => {
        const newAssets = [...prevAssets];
        if (newAssets[0]) {
          newAssets[0] = {
            ...newAssets[0],
            name: liveStockData.symbol,
            initialPrice: liveStockData.price.toFixed(2),
            volatility: liveStockData.volatility.toFixed(3)
          };
        }
        return newAssets;
      });
    }
  };
  
  const handleAddAsset = () => {
    setAssets([...assets, { id: nextAssetId, name: ``, initialPrice: '', volatility: '', correlation: '' }]);
    setNextAssetId(nextAssetId + 1);
  };

  const handleRemoveAsset = (id: number) => {
    setAssets(assets.filter(a => a.id !== id));
  };

  const handleAssetChange = (id: number, field: keyof Omit<AssetInput, 'id'>, value: string) => {
    setAssets(assets.map(asset =>
      asset.id === id ? { ...asset, [field]: value } : asset
    ));
  };

  const generateQlangScript = () => {
    const scenarioDescription = stressScenario || "Standard market simulation";
    const qlang = `// Q-Lang for scenario: ${scenarioDescription}
// --- Input Parameters ---
PARAM risk_free_rate = 0.05;
PARAM time_to_maturity = 1.0;

// --- Asset Definitions ---
${assets.map((asset, i) => `
// Asset ${i}: ${asset.name || 'Untitled'}
PARAM price_${i} = ${asset.initialPrice || 100};
PARAM volatility_${i} = ${asset.volatility || 0.2};
`).join('')}

// --- Quantum Registers ---
QREG asset_q[${assets.length * 2}]; // 2 qubits per asset (price dist, volatility)
CREG result_c[${assets.length}];

ALLOC asset_q, result_c;

EXECUTE {
    // 1. Encode financial parameters into quantum states
    ${assets.map((asset, i) => `// Initialize Asset ${i}
    OP::H asset_q[${i*2}]; // Prepare superposition for price distribution
    OP::RY(volatility_${i}) asset_q[${i*2 + 1}]; // Encode volatility
    `).join('')}

    // 2. Entangle assets for correlation modeling
    ${assets.length > 1 ? `// Example: Entangle Asset 0 and Asset 1
    OP::CNOT asset_q[0], asset_q[2];` : '// Single asset, no entanglement needed.'}

    // 3. Construct the Quantum Monte Carlo Circuit
    ${qaeEnabled ? `// --- QAE Accelerated Monte Carlo ---
    // The QAE subroutine would find the expected value of a payoff function.
    // This is a conceptual representation.
    OP::QAE(asset_q, target_precision=${qaePrecision});
    ` : `// --- Standard Quantum Monte Carlo ---
    // Simulate multiple paths using quantum walks.
    LOOP(1024) {
        OP::QUANTUM_WALK(asset_q);
    }
    `}

    // 4. Measure final states to get simulation output
    FOR i FROM 0 TO ${assets.length - 1} {
        MEASURE asset_q[i*2] -> result_c[i];
    }
}
`;
    setGeneratedQlang(qlang);
  };

  const handleRunSimulation = () => {
    setIsSimulating(true);
    setSimulationResult(null);
    generateQlangScript();
    setTimeout(() => {
      setSimulationResult(`Simulation successful. VaR (95%): ${(Math.random() * 5 + 2).toFixed(2)}%. Expected Return: ${(Math.random() * 10 - 2).toFixed(2)}%.`);
      setIsSimulating(false);
    }, 3000);
  };

  const handleDeployToGateway = () => {
    if (appName) {
      setIsDeploying(true);
      setTimeout(() => {
        const newUrl = `https://qcos.apps.web/${appName.toLowerCase().replace(/[^a-z0-9]/g, '-')}-qmc`;
        setPublicUrl(newUrl);
        setApiKey(Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15));
        setIsDeploying(false);
      }, 2000);
    } else {
      alert("Please provide an Application Name.");
    }
  };

  return (
    <GlassPanel title={<div className="flex items-center"><CpuChipIcon className="h-6 w-6 text-cyan-400 mr-2" />Quantum Monte Carlo: Finance</div>}>
      <div className="p-4 space-y-6 text-sm h-full overflow-y-auto">

        {/* Live Stock Feed */}
        <div className={subPanelClasses}>
          <h3 className="flex items-center text-lg font-semibold text-cyan-300 mb-3">
            <ChartBarIcon className="h-5 w-5 mr-2" /> Live Stock Feed & QMC Input
          </h3>
          {isFeedLoading ? (
            <div className="space-y-3 p-2">
                <div className="flex justify-between">
                    <LoadingSkeleton className="h-6 w-1/3" />
                    <LoadingSkeleton className="h-6 w-8" />
                </div>
                <div className="grid grid-cols-2 gap-4">
                     <LoadingSkeleton lines={2} className="h-4" />
                     <LoadingSkeleton lines={2} className="h-4" />
                </div>
                <LoadingSkeleton className="h-8 w-full mt-2" />
            </div>
          ) : feedError ? (
            <div className="text-red-400 text-center py-4">{feedError} <button onClick={fetchStockData} className="ml-2 text-cyan-400 hover:text-cyan-200"><ArrowPathIcon className="h-5 w-5 inline-block" /></button></div>
          ) : liveStockData ? (
            <div className="space-y-4 animate-fade-in">
              <div className="flex justify-between items-center mb-2">
                <h3 className="text-lg font-semibold text-cyan-300">{liveStockData.symbol} Live Data</h3>
                <button onClick={fetchStockData} className="p-1 rounded-full text-cyan-400 hover:bg-cyan-900 hover:text-cyan-200 transition-colors" title="Refresh Data">
                  <ArrowPathIcon className="h-5 w-5" />
                </button>
              </div>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="text-cyan-400">Price:</div><div className="text-cyan-200 font-medium">${liveStockData.price.toFixed(2)}</div>
                <div className="text-cyan-400">Change:</div><div className={`font-medium ${liveStockData.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>{liveStockData.change >= 0 ? '+' : ''}{liveStockData.change.toFixed(2)} ({liveStockData.changePercent})</div>
                <div className="text-cyan-400">Volume:</div><div className="text-cyan-200">{liveStockData.volume.toLocaleString()}</div>
                <div className="text-cyan-400">Volatility (Implied):</div><div className="text-cyan-200">{(liveStockData.volatility * 100).toFixed(2)}%</div>
              </div>
              <button onClick={handleImportData} className={`${holographicButtonClasses} bg-purple-600/30 hover:bg-purple-700/50 text-white mt-4`}>
                <ArrowPathIcon className="h-5 w-5 mr-2" /> Import Data to Scenario Builder
              </button>
            </div>
          ) : (
            <div className="text-red-400 text-center py-4">No stock data available.</div>
          )}
        </div>
        
        {/* QAE Module */}
        <div className={subPanelClasses}>
          <h3 className="flex items-center text-lg font-semibold text-cyan-300 mb-3"><RocketLaunchIcon className="h-5 w-5 mr-2" /> Quantum Amplitude Estimation (QAE)</h3>
          <div className="flex items-center justify-between mb-4">
            <span className="text-cyan-200">Enable QAE Acceleration:</span>
            <label htmlFor="qae-toggle" className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" id="qae-toggle" className="sr-only peer" checked={qaeEnabled} onChange={() => setQaeEnabled(!qaeEnabled)} />
              <div className="w-11 h-6 bg-gray-600 rounded-full peer peer-focus:ring-2 peer-focus:ring-cyan-500 peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-cyan-600"></div>
            </label>
          </div>
          <div className="mb-4">
            <label htmlFor="qae-precision" className="block text-cyan-400 mb-1">Target Precision:</label>
            <input type="number" id="qae-precision" className={inputClasses} value={qaePrecision} onChange={(e) => setQaePrecision(e.target.value)} step="0.001" min="0.001" max="0.1" disabled={!qaeEnabled} />
          </div>
        </div>

        {/* Real-time Scenario Builder */}
        <div className={subPanelClasses}>
          <h3 className="flex items-center text-lg font-semibold text-cyan-300 mb-3"><GitBranchIcon className="h-5 w-5 mr-2" /> Real-time Scenario Builder</h3>
          <div className="space-y-3 mb-4">
            {assets.map((asset) => (
              <div key={asset.id} className="grid grid-cols-1 md:grid-cols-5 gap-2 items-center bg-black/20 p-2 rounded-md">
                <input type="text" placeholder="Asset Name" className={`${inputClasses} md:col-span-1`} value={asset.name} onChange={(e) => handleAssetChange(asset.id, 'name', e.target.value)} />
                <input type="number" placeholder="Price" className={`${inputClasses} md:col-span-1`} value={asset.initialPrice} onChange={(e) => handleAssetChange(asset.id, 'initialPrice', e.target.value)} />
                <input type="number" placeholder="Volatility" className={`${inputClasses} md:col-span-1`} value={asset.volatility} onChange={(e) => handleAssetChange(asset.id, 'volatility', e.target.value)} />
                <input type="number" placeholder="Correlation" className={`${inputClasses} md:col-span-1`} value={asset.correlation} onChange={(e) => handleAssetChange(asset.id, 'correlation', e.target.value)} disabled={asset.id === 1} />
                <button onClick={() => handleRemoveAsset(asset.id)} className="text-red-400 hover:text-red-500 disabled:opacity-30 md:col-span-1 p-2 flex items-center justify-center" title="Remove Asset" disabled={asset.id === 1} ><XIcon className="w-5 h-5" /></button>
              </div>
            ))}
          </div>
          <button onClick={handleAddAsset} className={`${holographicButtonClasses} bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-200 border border-cyan-800 text-xs py-1`}>Add Asset</button>
        </div>

        {/* Dynamic Risk Visualization & Stress Testing */}
        <div className={subPanelClasses}>
          <h3 className="flex items-center text-lg font-semibold text-cyan-300 mb-3"><ChartBarIcon className="h-5 w-5 mr-2" /> Simulation & Stress Testing</h3>
          <div className="mb-4">
            <label htmlFor="stress-scenario" className="block text-cyan-400 mb-1">Stress Test Scenario (Optional):</label>
            <input type="text" id="stress-scenario" className={inputClasses} placeholder="e.g., Global Market Crash (20% drop)" value={stressScenario} onChange={(e) => setStressScenario(e.target.value)} />
          </div>
          <button onClick={handleRunSimulation} className={`${holographicButtonClasses} bg-cyan-600/30 hover:bg-cyan-700/50 text-white`} disabled={isSimulating}>
            {isSimulating ? <LoaderIcon className="w-5 h-5 mr-2 animate-spin" /> : <PlayIcon className="w-5 w-5 mr-2" />}
            {isSimulating ? 'Simulating...' : 'Run Scenario'}
          </button>
          {isSimulating && (
               <div className="mt-4 p-2 bg-black/40 rounded border border-cyan-900/30">
                    <LoadingSkeleton lines={2} />
               </div>
          )}
          {simulationResult && !isSimulating && <div className="mt-4 p-2 text-center bg-green-900/50 border border-green-700 rounded-md text-green-300 animate-fade-in">{simulationResult}</div>}
        </div>

        {/* Quantum-to-Web Gateway Deployment */}
        <div className={subPanelClasses}>
          <h3 className="flex items-center text-lg font-semibold text-cyan-300 mb-3"><GlobeIcon className="h-5 w-5 mr-2" /> Quantum-to-Web Gateway</h3>
          <div className="mb-4">
            <label htmlFor="app-name" className="block text-cyan-400 mb-1">Application Name:</label>
            <input type="text" id="app-name" className={inputClasses} placeholder="e.g., my-qmc-risk-model" value={appName} onChange={(e) => setAppName(e.target.value)} />
          </div>
          <button onClick={handleDeployToGateway} className={`${holographicButtonClasses} bg-purple-600/30 hover:bg-purple-700/50 text-white`} disabled={isDeploying}>
            {isDeploying ? <LoaderIcon className="w-5 h-5 mr-2 animate-spin" /> : <RocketLaunchIcon className="w-5 h-5 mr-2" />}
            Deploy to Gateway
          </button>
          
          {isDeploying && <div className="mt-2"><LoadingSkeleton /></div>}

          {publicUrl && apiKey && !isDeploying && (
            <div className="mt-4 space-y-2 animate-fade-in">
              <div className="mb-2">
                <label className="block text-cyan-400 mb-1">Public HTTPS URL:</label>
                <div className="flex items-center space-x-2">
                  <input type="text" className={`${inputClasses} cursor-not-allowed`} value={publicUrl} readOnly />
                  <button onClick={() => navigator.clipboard.writeText(publicUrl)} className="p-2 bg-black/30 hover:bg-black/50 rounded-md text-white holographic-button border border-cyan-800" title="Copy URL"><ClipboardIcon className="h-5 w-5" /></button>
                </div>
              </div>
              <div>
                <label className="block text-cyan-400 mb-1">API Key:</label>
                <div className="flex items-center space-x-2">
                  <input type="text" className={`${inputClasses} cursor-not-allowed`} value={apiKey} readOnly />
                  <button onClick={() => navigator.clipboard.writeText(apiKey)} className="p-2 bg-black/30 hover:bg-black/50 rounded-md text-white holographic-button border border-cyan-800" title="Copy API Key"><ClipboardIcon className="h-5 w-5" /></button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Q-Lang Script Generation & Customization */}
        <div className={subPanelClasses}>
          <h3 className="flex items-center text-lg font-semibold text-cyan-300 mb-3"><CodeBracketIcon className="h-5 w-5 mr-2" /> Q-Lang Script</h3>
          <div className="flex items-center justify-between mb-4">
            <span className="text-cyan-200">Show Generated Q-Lang:</span>
            <label htmlFor="qlang-toggle" className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" id="qlang-toggle" className="sr-only peer" checked={showQlang} onChange={() => setShowQlang(!showQlang)} />
              <div className="w-11 h-6 bg-gray-600 rounded-full peer peer-focus:ring-2 peer-focus:ring-cyan-500 peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-cyan-600"></div>
            </label>
          </div>
          <button onClick={generateQlangScript} className={`${holographicButtonClasses} bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-200 border border-cyan-800`}>Generate/Update Q-Lang</button>
          {showQlang && (
            <div className="mt-4"><textarea className={`${inputClasses} h-64 font-mono text-xs resize-y`} value={generatedQlang || "// Q-Lang script will appear here after generation."} readOnly /></div>
          )}
        </div>

      </div>
    </GlassPanel>
  );
};

export default QuantumMonteCarloFinance;
