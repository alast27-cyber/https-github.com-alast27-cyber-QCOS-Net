import React, { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ToastProvider, useToast } from './context/ToastContext';
import { SimulationProvider, useSimulation } from './context/SimulationContext';
import { useQuantumApps } from './hooks/useQuantumApps';
import { useAgentQ } from './hooks/useAgentQ';
import { useAdminChat } from './hooks/useAdminChat';
import { usePanelContent } from './hooks/usePanelContent';
import { useVoiceCommands } from './hooks/useVoiceCommands';

// Core Components
import LoginScreen from './components/LoginScreen';
import ResourceSteward from './components/ResourceSteward';
import ToastContainer from './components/ToastContainer';
import AgentQ from './components/AgentQ';
import AdminChat from './components/AdminChat';
import AnimatedBackground from './components/AnimatedBackground';
import EditorView from './components/EditorView';
import GlassPanel from './components/GlassPanel';
import FullScreenSwitcher from './components/FullScreenSwitcher';
import OnboardingFlow from './components/OnboardingFlow'; 
import ChipsBrowserInstaller from './components/ChipsBrowserInstaller'; 

// Icons
import { 
  XIcon, BrainCircuitIcon, CpuChipIcon, ServerCogIcon, GlobeIcon, 
  ShieldCheckIcon, LayersIcon, BeakerIcon, AtomIcon, GridIcon,
  RefreshCwIcon, ActivityIcon, SparklesIcon, NetworkIcon, 
  RocketLaunchIcon, BoxIcon, MaximizeIcon, MinimizeIcon
} from './components/Icons';

// Utils & Types
import { initialLogs, initialSystemHealth, getPanelMetadata, FaceData } from './utils/dashboardConfig';
import { LogEntry } from './types';

// Panel Components
import IAIKernelStatus from './components/IAIKernelStatus';
import DistributedCognitiveArchitecture from './components/DistributedCognitiveArchitecture'; 
import UtilityHubPanel from './components/UtilityHubPanel';
import QLangCoreEvolutionPanel from './components/QLangCoreEvolutionPanel';
import SecurityMonitorAndSimulator from './components/SecurityMonitorAndSimulator';
import QPUHealth from './components/QPUHealth';
import QuantumDataIngestion from './components/QuantumDataIngestion';
import SystemDiagnostic from './components/SystemDiagnostic';
import CHIPSBackOffice from './components/CHIPSBackOffice'; 
import NeuralProgrammingPanel from './components/NeuralProgrammingPanel';
import QuantumCognitiveArchitecture from './components/QuantumCognitiveArchitecture';
import QuantumDeepLearning from './components/QuantumDeepLearning';
import QuantumExecutionFlow from './components/QuantumExecutionFlow';
import QuantumNeuroNetworkVisualizer from './components/QuantumNeuroNetworkVisualizer';
import QuantumProtocolSimulator from './components/QuantumProtocolSimulator';
import QuantumSpeedometer from './components/QuantumSpeedometer';
import QuantumSystemSimulator from './components/QuantumSystemSimulator';
import QubitSimulator from './components/QubitSimulator';
import QuantumMachineLearning from './components/QuantumMachineLearning';
import QBioMedDrugDiscovery from './components/QBioMedDrugDiscovery';
import MolecularSimulationToolkit from './components/MolecularSimulationToolkit';
import QOSKernelConsole from './components/QOSKernelConsole';
import CHIPSBrowserSDK from './components/CHIPSBrowserSDK'; 
import QuantumEngineeringDesign from './components/QuantumEngineeringDesign';
import AGISingularityInterface from './components/AGISingularityInterface';
import QuantumSwineIntelligence from './components/QuantumSwineIntelligence';
import QuantumReinforcementLearning from './components/QuantumReinforcementLearning';
import QuantumProgrammingInterface from './components/QuantumProgrammingInterface';
import QuantumLargeLanguageModel from './components/QuantumLargeLanguageModel';
import GrandUniverseSimulator from './components/GrandUniverseSimulator';
import QuantumMonteCarloFinance from './components/QuantumMonteCarloFinance';
import QuantumNetworkVisualizer from './components/QuantumNetworkVisualizer';
import VQEToolkit from './components/VQEToolkit';
import QcosDashboard from './components/QcosDashboard';

const QCOS_VERSION = 4.5;

const EntanglementBeams: React.FC<{ active: boolean; isLinked?: boolean }> = ({ active, isLinked = false }) => {
    if (!active) return null;
    return (
        <svg className="fixed inset-0 w-full h-full pointer-events-none z-10 opacity-40">
            <defs>
                <linearGradient id="entangleGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="rgba(0, 255, 255, 0)" />
                    <stop offset="50%" stopColor="rgba(0, 255, 255, 0.4)" />
                    <stop offset="100%" stopColor="rgba(0, 255, 255, 0)" />
                </linearGradient>
            </defs>
            <path d="M 0,200 Q 400,200 1200,500" stroke="url(#entangleGrad)" strokeWidth="1" fill="none" className="animate-pulse" />
            <path d="M 0,800 Q 400,800 1200,500" stroke="url(#entangleGrad)" strokeWidth="1" fill="none" className="animate-pulse" style={{ animationDelay: '0.5s' }} />
        </svg>
    );
};

const DashboardContent: React.FC = () => {
  const { isAuthenticated, adminLevel } = useAuth();
  const { addToast } = useToast();
  const { systemStatus, startAllSimulations, qllm, entanglementMesh } = useSimulation();
  
  const [showOnboarding, setShowOnboarding] = useState(() => {
    const hasOnboarded = localStorage.getItem('qcos_onboarded') === 'true';
    const isChipsInstalled = localStorage.getItem('chips_browser_installed') === 'true';
    return isChipsInstalled && !hasOnboarded;
  });
  const [isInstalled, setIsInstalled] = useState(() => localStorage.getItem('chips_browser_installed') === 'true'); 
  const [isImmersive, setIsImmersive] = useState(false);

  const [tilt, setTilt] = useState({ x: 0, y: 0 });
  const dashboardRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!dashboardRef.current) return;
    const { innerWidth: w, innerHeight: h } = window;
    const x = (e.clientX - w / 2) / (w / 2);
    const y = (e.clientY - h / 2) / (h / 2);
    setTilt({ x: x * 2.5, y: -y * 2.5 }); 
  }, []);

  useEffect(() => {
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [handleMouseMove]);

  useEffect(() => {
    if (isAuthenticated && isInstalled) {
        const hasOnboarded = localStorage.getItem('qcos_onboarded') === 'true';
        if (!hasOnboarded && !showOnboarding) {
            setTimeout(() => setShowOnboarding(true), 0);
        } else {
            const timer = setTimeout(() => {
                startAllSimulations();
            }, 1000);
            return () => clearTimeout(timer);
        }
    }
  }, [isAuthenticated, isInstalled, startAllSimulations, showOnboarding]);
  
  const handleInstallationComplete = () => {
      localStorage.setItem('chips_browser_installed', 'true');
      setIsInstalled(true);
      setShowOnboarding(true); 
  };

  const handleOnboardingComplete = () => {
      setShowOnboarding(false);
      localStorage.setItem('qcos_onboarded', 'true');
      startAllSimulations();
  };

  const [logs, setLogs] = useState<LogEntry[]>(initialLogs);
  const [maximizedPanelId, setMaximizePanelId] = useState<string | null>(null);
  const [agentWeaverInput, setAgentWeaverInput] = useState('');
  const [isEditorOpen, setIsEditorOpen] = useState(false);
  const [isSwitcherOpen, setIsSwitcherOpen] = useState(false);
  const [connectedAppId, setConnectedAppId] = useState<string | null>(null);

  const [activeDataStreams] = useState<string[]>([
      'GQML Stream (Live)', 'Quantum Entanglement Feed', 'LHC Collision Data', 'NASA Exoplanet Archive'
  ]);


  const addLog = useCallback((level: LogEntry['level'], msg: string) => {
    setLogs(prev => [{ id: Date.now() + Math.random(), time: new Date().toLocaleTimeString(), level, msg }, ...prev].slice(0, 50));
  }, []);

  const handlePanelSelect = useCallback((panelId: string) => {
      setMaximizePanelId(panelId);
      setIsSwitcherOpen(false);
  }, []);

  const { marketApps, uriAssignments, handleInstallApp, handleFullDeployment, handleTogglePublic } = useQuantumApps(addLog, handlePanelSelect);
  const { isOpen: isAdminChatOpen, onToggle: toggleAdminChat, adminChatProps } = useAdminChat();

  const panelInfoMap = useMemo(() => {
      const metadata = getPanelMetadata(QCOS_VERSION);
      const map: any = {};
      Object.values(metadata).forEach((face: any) => {
          if (face.panels) {
              face.panels.forEach((p: any) => {
                  map[p.id] = { title: p.title, description: p.description };
              });
          }
      });
      return map;
  }, []);

  const { isAgentQOpen, toggleAgentQ, generateApp, updateAppForChips, debugAndFixApp, editCode, agentQProps } = useAgentQ({
      focusedPanelId: maximizedPanelId || 'agentq-core',
      panelInfoMap,
      qcosVersion: QCOS_VERSION,
      systemHealth: { ...initialSystemHealth, ...systemStatus },
      onDashboardControl: (action, target) => {
          if (action === 'modify_panel' && target) {
              try {
                  const data = JSON.parse(target);
                  addLog('INFO', `[Q-NATIVE] Panel ${data.action}: ${data.panelName}`);
                  addToast(`System Panel ${data.action} initiated for ${data.panelName}`, 'info');
              } catch (e) { console.error("Error processing modify_panel action:", e); }
          } else if (action === 'trigger_evolution' && target) {
              try {
                  const data = JSON.parse(target);
                  addLog('INFO', `[Q-NATIVE] ${data.evolutionType.toUpperCase()} Evolution Triggered: ${data.description}`);
                  addToast(`Evolution Protocol Initiated: ${data.evolutionType}`, 'warning');
              } catch (e) { console.error("Error processing trigger_evolution action:", e); }
          }
      }
  });

  const { listeningState, toggleListening, isSupported } = useVoiceCommands([
      { command: ['open agent q', 'agent q', 'help'], callback: () => !isAgentQOpen && toggleAgentQ() },
      { command: ['reset view', 'minimize'], callback: () => setMaximizePanelId(null) },
      { command: ['immersive mode', 'cinema mode'], callback: () => setIsImmersive(true) },
      { command: ['standard mode', 'exit immersive'], callback: () => setIsImmersive(false) }
  ]);

  const faceData = usePanelContent({
      systemHealth: { ...initialSystemHealth, ...systemStatus },
      isRecalibrating: false,
      isUpgrading: false,
      activeDataStreams, 
      uriAssignments,
      agentWeaverInput,
      onPublishToExchange: handleFullDeployment,
      onConnectToGateway: (source) => addLog('INFO', `Connecting to gateway source: ${source}`),
      logs,
      addLog,
      codebase: {}, 
      onApplyPatch: (f, c) => addLog('SUCCESS', `Patch applied to ${f}`),
      marketApps,
      onInstallApp: handleInstallApp,
      onLaunchApp: (id) => setMaximizePanelId(id),
      onGenerateApp: generateApp,
      onUpdateApp: updateAppForChips,
      onDebugApp: debugAndFixApp,
      onTogglePublic: handleTogglePublic,
      onSendToWeaver: setAgentWeaverInput,
      onCloseSystemDiagnostic: () => setMaximizePanelId(null),
      adminLevel,
      qcosVersion: QCOS_VERSION,
      agentQProps,
      editCode,
      onOpenAppCreator: (p) => {},
      onSimulateApp: (id) => setConnectedAppId(id),
      connectedAppId,
      toggleAgentQ
  });

  if (!isAuthenticated) {
      return (
          <>
            <AnimatedBackground />
            <div className="absolute inset-0 flex items-center justify-center">
              <LoginScreen />
            </div>
            <ToastContainer />
          </>
      );
  }

  if (!isInstalled) {
      return <ChipsBrowserInstaller onInstallComplete={handleInstallationComplete} />;
  }

  if (showOnboarding) {
      return <OnboardingFlow onComplete={handleOnboardingComplete} />;
  }

  if (isEditorOpen) {
      return <EditorView agentQProps={agentQProps} onToggleView={() => setIsEditorOpen(false)} />;
  }

  let maximizedContent = null;
  if (maximizedPanelId) {
      let foundPanel: any = null;
      Object.values(faceData).some((face: FaceData) => {
          foundPanel = face.panels.find(p => p.id === maximizedPanelId);
          return !!foundPanel;
      });

      if (!foundPanel) {
          const directMap: any = {
             'qiai-kernel-status': { title: 'Kernel Node Status', content: <IAIKernelStatus isRecalibrating={false} /> },
             'qiai-cognitive-mesh': { title: 'Neural Topology', content: <DistributedCognitiveArchitecture activeDataStreams={activeDataStreams} /> },
             'utility-hub': { title: 'Service Hub', content: <UtilityHubPanel onMaximizeSubPanel={setMaximizePanelId} /> },
             'security-monitor': { title: 'Security Monitor', content: <SecurityMonitorAndSimulator /> },
             'qpu-health': { title: 'QPU Vitals', content: <QPUHealth systemHealth={{ ...initialSystemHealth, ...systemStatus }} /> },
             'data-ingestion': { title: 'Quantum Data Ingestion', content: <QuantumDataIngestion /> },
             'system-diagnostic': { title: 'System Diagnostics', content: <SystemDiagnostic onClose={() => setMaximizePanelId(null)} /> },
             'q-lang-evolution': { title: 'Evolution Matrix', content: <QLangCoreEvolutionPanel /> },
             'chips-back-office': { title: 'Chips Back Office', content: <CHIPSBackOffice uriAssignments={uriAssignments} marketApps={marketApps} /> },
             'neural-programming': { title: 'Neural Programming', content: <NeuralProgrammingPanel /> },
             'quantum-cognitive-architecture': { title: 'Cognitive Architecture', content: <QuantumCognitiveArchitecture /> },
             'quantum-deep-learning': { title: 'Quantum Deep Learning', content: <QuantumDeepLearning /> },
             'quantum-engineering-design': { title: 'Quantum Engineering Design', content: <QuantumEngineeringDesign /> },
             'quantum-machine-learning': { title: 'Quantum MachineLearning', content: <QuantumMachineLearning /> },
             'quantum-execution-flow': { title: 'Execution Flow', content: <QuantumExecutionFlow ipsThroughput={systemStatus.ipsThroughput || 850} /> },
             'quantum-neuro-network': { title: 'QNN Topology', content: <QuantumNeuroNetworkVisualizer /> },
             'quantum-protocol-simulator': { title: 'Protocol Simulator', content: <QuantumProtocolSimulator /> },
             'quantum-speedometer': { title: 'Quantum Velocity', content: <QuantumSpeedometer /> },
             'quantum-system-simulator': { title: 'System Simulator', content: <QuantumSystemSimulator /> },
             'qubit-simulator': { title: 'Qubit Array Sim', content: <QubitSimulator /> },
             'q-biomed': { title: 'BioMed Discovery', content: <QBioMedDrugDiscovery /> },
             'mol-sim': { title: 'Molecular Toolkit', content: <MolecularSimulationToolkit /> },
             'qos-kernel-manager': { title: 'QOS Kernel Console', content: <QOSKernelConsole /> },
             'agi-singularity': { title: 'AGI Singularity', content: <AGISingularityInterface /> },
             'quantum-swine': { title: 'Swine Intelligence', content: <QuantumSwineIntelligence onOpenApp={() => {}} /> },
             'quantum-rl': { title: 'Reinforcement Learning', content: <QuantumReinforcementLearning /> },
             'quantum-programming': { title: 'Programming Interface', content: <QuantumProgrammingInterface /> },
             'quantum-llm': { title: 'Large Language Model', content: <QuantumLargeLanguageModel /> },
             'grand-universe-simulator': { title: 'Grand Universe Simulator', content: <GrandUniverseSimulator /> },
             'quantum-monte-carlo': { title: 'Monte Carlo Finance', content: <QuantumMonteCarloFinance /> },
             'quantum-network-visualizer': { title: 'Quantum Network Visualizer', content: <QuantumNetworkVisualizer /> },
             'vqe-toolkit': { title: 'VQE Toolkit', content: <VQEToolkit /> },
             'chips-quantum-network': { 
                 title: 'Chips Quantum Browser', 
                 content: <CHIPSBrowserSDK initialApp={undefined} onToggleAgentQ={() => {}} apps={marketApps} onInstallApp={handleInstallApp} /> 
             }
          };
          foundPanel = directMap[maximizedPanelId];
      }
      
      if (foundPanel) {
          maximizedContent = (
              <GlassPanel title={foundPanel.title} onMaximize={() => setMaximizePanelId(null)} isMaximized={true}>
                  {foundPanel.content}
              </GlassPanel>
          );
      }
  }

  const hudVisibility = maximizedPanelId ? 'opacity-0 pointer-events-none' : 'opacity-100 pointer-events-auto';

  return (
    <div ref={dashboardRef} className="relative w-screen h-screen bg-black text-cyan-100 font-mono overflow-hidden flex flex-col perspective-viewport">
      <AnimatedBackground />
      <EntanglementBeams active={qllm.efficiencyBoost > 1} isLinked={entanglementMesh.isUniverseLinkedToQLang} />
      
      <div className="lattice-overlay"></div>

      <FullScreenSwitcher 
          isOpen={isSwitcherOpen}
          onToggle={() => setIsSwitcherOpen(!isSwitcherOpen)}
          onPanelSelect={handlePanelSelect}
          corePanels={[
            { id: 'agentq-core', title: 'QIAI-IPS Core', icon: BrainCircuitIcon },
            { id: 'chips-quantum-network', title: 'Chips Browser', icon: GlobeIcon },
            { id: 'qcos-core-gateway', title: 'Gateway', icon: ServerCogIcon },
            { id: 'utility-hub', title: 'Vitals', icon: ShieldCheckIcon },
            { id: 'neural-programming', title: 'Neural Prog', icon: LayersIcon },
            { id: 'quantum-engineering-design', title: 'Engineering', icon: AtomIcon },
            { id: 'q-biomed', title: 'Material Sci', icon: BeakerIcon },
          ]}
          appPanels={marketApps.filter(app => app.status === 'installed').map(app => ({ id: app.id, title: app.name, icon: app.icon }))}
          className={`fixed bottom-4 left-[32rem] z-50 pointer-events-auto transition-opacity duration-500 ${hudVisibility}`}
      />

      <div 
        className={`
            flex-grow p-6 transition-all duration-700 overflow-hidden z-20 floating-hologram ${hudVisibility}
            ${isImmersive ? 'h-[35vh] mt-auto' : 'h-full'}
        `}
        style={{ transform: `rotateY(${tilt.x}deg) rotateX(${tilt.y}deg)` }}
      >
          <QcosDashboard />
      </div>

      {maximizedPanelId && (
          <div className="absolute inset-4 z-[90] bg-black/95 backdrop-blur-3xl border border-cyan-500/30 rounded-2xl shadow-2xl animate-fade-in-up flex flex-col overflow-hidden">
              <div className="absolute top-4 right-4 z-[95]">
                  <button onClick={() => setMaximizePanelId(null)} className="p-2 bg-black/50 hover:bg-red-900/50 rounded-full text-cyan-400 border border-cyan-800 transition-all">
                      <XIcon className="w-6 h-6" />
                  </button>
              </div>
              {maximizedContent}
          </div>
      )}

      <div className={`h-16 flex-shrink-0 flex items-center justify-between px-8 bg-black/60 border-t border-cyan-900/30 z-50 transition-opacity duration-500 backdrop-blur-md ${hudVisibility}`}>
          <div className="flex items-center gap-6">
              <ResourceSteward listeningState={listeningState} onToggleListen={toggleListening} isVoiceSupported={isSupported} />
              <div className="h-8 w-px bg-cyan-900/40"></div>
              <button onClick={() => setIsEditorOpen(true)} className="holographic-projection px-5 py-2 text-[10px] font-black uppercase tracking-widest bg-cyan-900/10 border-cyan-500/40 rounded-full hover:bg-cyan-500/20 transition-all">
                  Source Nexus
              </button>

              <button 
                onClick={() => setIsImmersive(!isImmersive)}
                className={`holographic-projection px-4 py-2 text-[10px] font-black uppercase tracking-widest border rounded-full transition-all flex items-center gap-2 ${isImmersive ? 'bg-green-900/20 border-green-500/40 text-green-300' : 'bg-cyan-900/10 border-cyan-500/40 text-cyan-300'}`}
              >
                  {isImmersive ? <MinimizeIcon className="w-3 h-3" /> : <MaximizeIcon className="w-3 h-3" />}
                  {isImmersive ? 'Expand View' : 'Cinematic View'}
              </button>
          </div>

          <div className="flex items-center gap-6">
               <div className="flex gap-6 text-[10px] font-mono text-cyan-600 bg-black/40 px-6 py-2 rounded-full border border-cyan-900/30">
                   <div className="flex items-center gap-2">
                       <CpuChipIcon className="w-3 h-3" />
                       LOAD: <span className={systemStatus.neuralLoad > 80 ? 'text-red-400' : 'text-green-400'}>{(systemStatus.neuralLoad || 0).toFixed(1)}%</span>
                   </div>
                   <div className="flex items-center gap-2">
                       <ActivityIcon className="w-3 h-3" />
                       DIM_SHIFT: <span className="text-white">ACTIVE</span>
                   </div>
               </div>

              <AgentQ 
                isOpen={isAgentQOpen} 
                onToggleOpen={toggleAgentQ} 
                triggerClassName={`fixed bottom-4 left-[24rem] z-50 pointer-events-auto group transition-opacity duration-500 ${hudVisibility}`}
                {...agentQProps} 
              />
          </div>
      </div>

      <AdminChat 
        triggerClassName={`fixed bottom-4 left-[28rem] z-50 pointer-events-auto group transition-opacity duration-500 ${hudVisibility}`}
        {...adminChatProps} 
      />

      <ToastContainer />
      
      <style>{`
        .transform-style-preserve-3d { transform-style: preserve-3d; }
        .backface-hidden { backface-visibility: hidden; }
        .rotate-y-180 { transform: rotateY(180deg); }
      `}</style>
    </div>
  );
};

const App: React.FC = () => (
  <AuthProvider>
    <ToastProvider>
      <SimulationProvider>
        <DashboardContent />
      </SimulationProvider>
    </ToastProvider>
  </AuthProvider>
);

export default App;