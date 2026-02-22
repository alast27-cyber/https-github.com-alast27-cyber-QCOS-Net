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
import QuantumAppExchange from './components/QuantumAppExchange';
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
            setShowOnboarding(true);
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

  const [dashboardGroup, setDashboardGroup] = useState<'group1' | 'group2' | 'group3' | 'group4' | 'group5'>('group1');
  
  const [flipP1, setFlipP1] = useState(false);
  const [flipP2, setFlipP2] = useState(false);
  const [flipP3, setFlipP3] = useState(false);

  const [flipP7, setFlipP7] = useState(false);
  const [flipP8, setFlipP8] = useState(false);
  const [flipP9, setFlipP9] = useState(false);

  const [flipP13, setFlipP13] = useState(false);
  const [flipP14, setFlipP14] = useState(false);
  const [flipP15, setFlipP15] = useState(false);

  const [flipP19, setFlipP19] = useState(false);
  const [flipP20, setFlipP20] = useState(false);
  const [flipP21, setFlipP21] = useState(false);

  const [flipP25, setFlipP25] = useState(false);
  const [flipP26, setFlipP26] = useState(false);
  const [flipP27, setFlipP27] = useState(false);
  
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
      onDashboardControl: () => {}
  });

  const { listeningState, toggleListening, isSupported } = useVoiceCommands([
      { command: ['open agent q', 'agent q', 'help'], callback: () => !isAgentQOpen && toggleAgentQ() },
      { command: ['reset view', 'minimize'], callback: () => setMaximizePanelId(null) },
      { command: ['switch deck', 'next page', 'swap view', 'switch group'], callback: () => setDashboardGroup(prev => prev === 'group1' ? 'group2' : 'group1') },
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
             'quantum-exchange': { title: 'App Exchange', content: <QuantumAppExchange /> },
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
            grid grid-cols-12 gap-6 p-6 transition-all duration-700 overflow-hidden z-20 floating-hologram ${hudVisibility}
            ${isImmersive ? 'h-[35vh] mt-auto items-end' : 'flex-grow h-0'}
        `}
        style={{ transform: `rotateY(${tilt.x}deg) rotateX(${tilt.y}deg)` }}
      >
          {dashboardGroup === 'group1' ? (
            <>
              <div className="col-span-12 lg:col-span-4 h-full relative" style={{ perspective: '3000px' }}>
                  <div className={`relative w-full h-full transition-transform duration-1000 transform-style-preserve-3d ${flipP1 ? 'rotate-y-180' : ''}`}>
                      <div className={`absolute inset-0 backface-hidden ${!flipP1 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                          <GlassPanel onMaximize={() => setMaximizePanelId('agentq-core')} title="QIAI-IPS Cognitive Architecture">
                              <div className="absolute top-2 right-14 z-50">
                                <button onClick={(e) => { e.stopPropagation(); setFlipP1(true); }} className="p-1.5 hover:bg-cyan-500/20 rounded-md border border-cyan-800 text-cyan-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                              </div>
                              {faceData[0].panels.find(p => p.id === 'agentq-core')?.content}
                          </GlassPanel>
                      </div>
                      <div className={`absolute inset-0 backface-hidden rotate-y-180 ${flipP1 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                          <GlassPanel onMaximize={() => setMaximizePanelId('q-lang-evolution')} title="Evolution Matrix">
                              <div className="absolute top-2 right-14 z-50">
                                <button onClick={(e) => { e.stopPropagation(); setFlipP1(false); }} className="p-1.5 hover:bg-purple-500/20 rounded-md border border-purple-800 text-purple-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                              </div>
                              <QLangCoreEvolutionPanel />
                          </GlassPanel>
                      </div>
                  </div>
              </div>

              <div className="col-span-12 lg:col-span-5 h-full relative" style={{ perspective: '3000px' }}>
                  <div className={`relative w-full h-full transition-transform duration-1000 transform-style-preserve-3d ${flipP2 ? 'rotate-y-180' : ''}`}>
                      <div className={`absolute inset-0 backface-hidden ${!flipP2 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                          <GlassPanel onMaximize={() => setMaximizePanelId('chips-quantum-network')} title="Chips Quantum Network">
                              <div className="absolute top-2 right-14 z-50">
                                <button onClick={(e) => { e.stopPropagation(); setFlipP2(true); }} className="p-1.5 hover:bg-blue-500/20 rounded-md border border-blue-800 text-blue-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                              </div>
                              <div className="h-full overflow-hidden">{faceData[1].panels.find(p => p.id === 'chips-quantum-network')?.content}</div>
                          </GlassPanel>
                      </div>
                      <div className={`absolute inset-0 backface-hidden rotate-y-180 ${flipP2 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                          <GlassPanel onMaximize={() => setMaximizePanelId('chips-back-office')} title="Chips Back Office">
                              <div className="absolute top-2 right-14 z-50">
                                <button onClick={(e) => { e.stopPropagation(); setFlipP2(false); }} className="p-1.5 hover:bg-purple-500/20 rounded-md border border-purple-800 text-purple-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                              </div>
                              <CHIPSBackOffice uriAssignments={uriAssignments} marketApps={marketApps} />
                          </GlassPanel>
                      </div>
                  </div>
              </div>

              <div className="col-span-12 lg:col-span-3 h-full relative" style={{ perspective: '3000px' }}>
                  <div className={`relative w-full h-full transition-transform duration-1000 transform-style-preserve-3d ${flipP3 ? 'rotate-y-180' : ''}`}>
                      <div className={`absolute inset-0 backface-hidden ${!flipP3 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                          <GlassPanel onMaximize={() => setMaximizePanelId('qcos-core-gateway')} title="Engine Hub">
                              <div className="absolute top-2 right-14 z-50">
                                <button onClick={(e) => { e.stopPropagation(); setFlipP3(true); }} className="p-1.5 hover:bg-red-500/20 rounded-md border border-red-800 text-red-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                              </div>
                              <div className="h-full overflow-hidden">{faceData[0].panels.find(p => p.id === 'qcos-core-gateway')?.content}</div>
                          </GlassPanel>
                      </div>
                      <div className={`absolute inset-0 backface-hidden rotate-y-180 ${flipP3 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                          <GlassPanel onMaximize={() => setMaximizePanelId('utility-hub')} title="Utility Nexus">
                              <div className="absolute top-2 right-14 z-50">
                                <button onClick={(e) => { e.stopPropagation(); setFlipP3(false); }} className="p-1.5 hover:bg-green-500/20 rounded-md border border-green-800 text-green-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                              </div>
                              <div className="h-full overflow-hidden">
                                <UtilityHubPanel onMaximizeSubPanel={setMaximizePanelId} />
                              </div>
                          </GlassPanel>
                      </div>
                  </div>
              </div>
            </>
          ) : dashboardGroup === 'group2' ? (
            <>
              <div className="col-span-12 lg:col-span-4 h-full relative" style={{ perspective: '3000px' }}>
                  <div className={`relative w-full h-full transition-transform duration-1000 transform-style-preserve-3d ${flipP7 ? 'rotate-y-180' : ''}`}>
                      <div className={`absolute inset-0 backface-hidden ${!flipP7 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                          <GlassPanel onMaximize={() => setMaximizePanelId('quantum-execution-flow')} title="Quantum Execution Flow">
                              <div className="absolute top-2 right-14 z-50">
                                <button onClick={(e) => { e.stopPropagation(); setFlipP7(true); }} className="p-1.5 hover:bg-blue-500/20 rounded-md border border-blue-800 text-blue-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                              </div>
                              <QuantumExecutionFlow ipsThroughput={systemStatus.ipsThroughput || 850} />
                          </GlassPanel>
                      </div>
                      <div className={`absolute inset-0 backface-hidden rotate-y-180 ${flipP7 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                          <GlassPanel onMaximize={() => setMaximizePanelId('quantum-protocol-simulator')} title="Protocol Simulator">
                              <div className="absolute top-2 right-14 z-50">
                                <button onClick={(e) => { e.stopPropagation(); setFlipP7(false); }} className="p-1.5 hover:bg-purple-500/20 rounded-md border border-purple-800 text-purple-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                              </div>
                              <QuantumProtocolSimulator />
                          </GlassPanel>
                      </div>
                  </div>
              </div>

              <div className="col-span-12 lg:col-span-5 h-full relative" style={{ perspective: '3000px' }}>
                  <div className={`relative w-full h-full transition-transform duration-1000 transform-style-preserve-3d ${flipP8 ? 'rotate-y-180' : ''}`}>
                      <div className={`absolute inset-0 backface-hidden ${!flipP8 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                          <GlassPanel onMaximize={() => setMaximizePanelId('quantum-neuro-network')} title="QNN Topology">
                              <div className="absolute top-2 right-14 z-50">
                                <button onClick={(e) => { e.stopPropagation(); setFlipP8(true); }} className="p-1.5 hover:bg-cyan-500/20 rounded-md border border-cyan-800 text-cyan-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                              </div>
                              <QuantumNeuroNetworkVisualizer />
                          </GlassPanel>
                      </div>
                      <div className={`absolute inset-0 backface-hidden rotate-y-180 ${flipP8 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                          <GlassPanel onMaximize={() => setMaximizePanelId('qubit-simulator')} title="Qubit Array Sim">
                              <div className="absolute top-2 right-14 z-50">
                                <button onClick={(e) => { e.stopPropagation(); setFlipP8(false); }} className="p-1.5 hover:bg-green-500/20 rounded-md border border-green-800 text-green-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                              </div>
                              <QubitSimulator />
                          </GlassPanel>
                      </div>
                  </div>
              </div>

              <div className="col-span-12 lg:col-span-3 h-full relative" style={{ perspective: '3000px' }}>
                  <div className={`relative w-full h-full transition-transform duration-1000 transform-style-preserve-3d ${flipP9 ? 'rotate-y-180' : ''}`}>
                      <div className={`absolute inset-0 backface-hidden ${!flipP9 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                          <GlassPanel onMaximize={() => setMaximizePanelId('quantum-speedometer')} title="Quantum Velocity">
                              <div className="absolute top-2 right-14 z-50">
                                <button onClick={(e) => { e.stopPropagation(); setFlipP9(true); }} className="p-1.5 hover:bg-yellow-500/20 rounded-md border border-yellow-800 text-yellow-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                              </div>
                              <QuantumSpeedometer />
                          </GlassPanel>
                      </div>
                      <div className={`absolute inset-0 backface-hidden rotate-y-180 ${flipP9 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                          <GlassPanel onMaximize={() => setMaximizePanelId('quantum-system-simulator')} title="System Simulator">
                              <div className="absolute top-2 right-14 z-50">
                                <button onClick={(e) => { e.stopPropagation(); setFlipP9(false); }} className="p-1.5 hover:bg-pink-500/20 rounded-md border border-pink-800 text-pink-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                              </div>
                              <QuantumSystemSimulator />
                          </GlassPanel>
                      </div>
                  </div>
              </div>
            </>
          ) : dashboardGroup === 'group3' ? (
            <>
              <div className="col-span-12 lg:col-span-4 h-full relative" style={{ perspective: '3000px' }}>
                  <div className={`relative w-full h-full transition-transform duration-1000 transform-style-preserve-3d ${flipP13 ? 'rotate-y-180' : ''}`}>
                      <div className={`absolute inset-0 backface-hidden ${!flipP13 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                           <GlassPanel onMaximize={() => setMaximizePanelId('agi-singularity')} title="AGI Singularity Interface">
                               <div className="absolute top-2 right-14 z-50">
                                 <button onClick={(e) => { e.stopPropagation(); setFlipP13(true); }} className="p-1.5 hover:bg-red-500/20 rounded-md border border-red-800 text-red-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                               </div>
                               <AGISingularityInterface />
                           </GlassPanel>
                       </div>
                       <div className={`absolute inset-0 backface-hidden rotate-y-180 ${flipP13 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                           <GlassPanel onMaximize={() => setMaximizePanelId('quantum-swine')} title="Quantum Swine Intelligence">
                               <div className="absolute top-2 right-14 z-50">
                                 <button onClick={(e) => { e.stopPropagation(); setFlipP13(false); }} className="p-1.5 hover:bg-green-500/20 rounded-md border border-green-800 text-green-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                               </div>
                               <QuantumSwineIntelligence onOpenApp={() => {}} />
                           </GlassPanel>
                       </div>
                   </div>
               </div>

               <div className="col-span-12 lg:col-span-5 h-full relative" style={{ perspective: '3000px' }}>
                   <div className={`relative w-full h-full transition-transform duration-1000 transform-style-preserve-3d ${flipP14 ? 'rotate-y-180' : ''}`}>
                       <div className={`absolute inset-0 backface-hidden ${!flipP14 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                           <GlassPanel onMaximize={() => setMaximizePanelId('quantum-rl')} title="Quantum Reinforcement Learning">
                               <div className="absolute top-2 right-14 z-50">
                                 <button onClick={(e) => { e.stopPropagation(); setFlipP14(true); }} className="p-1.5 hover:bg-orange-500/20 rounded-md border border-orange-800 text-orange-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                               </div>
                               <QuantumReinforcementLearning />
                           </GlassPanel>
                       </div>
                       <div className={`absolute inset-0 backface-hidden rotate-y-180 ${flipP14 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                           <GlassPanel onMaximize={() => setMaximizePanelId('quantum-programming')} title="Quantum Programming Interface">
                               <div className="absolute top-2 right-14 z-50">
                                 <button onClick={(e) => { e.stopPropagation(); setFlipP14(false); }} className="p-1.5 hover:bg-teal-500/20 rounded-md border border-teal-800 text-teal-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                               </div>
                               <QuantumProgrammingInterface />
                           </GlassPanel>
                       </div>
                   </div>
               </div>

               <div className="col-span-12 lg:col-span-3 h-full relative" style={{ perspective: '3000px' }}>
                   <div className={`relative w-full h-full transition-transform duration-1000 transform-style-preserve-3d ${flipP15 ? 'rotate-y-180' : ''}`}>
                       <div className={`absolute inset-0 backface-hidden ${!flipP15 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                           <GlassPanel onMaximize={() => setMaximizePanelId('quantum-llm')} title="Quantum Large Language Model">
                               <div className="absolute top-2 right-14 z-50">
                                 <button onClick={(e) => { e.stopPropagation(); setFlipP15(true); }} className="p-1.5 hover:bg-indigo-500/20 rounded-md border border-indigo-800 text-indigo-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                               </div>
                               <QuantumLargeLanguageModel />
                           </GlassPanel>
                       </div>
                       <div className={`absolute inset-0 backface-hidden rotate-y-180 ${flipP15 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                           <GlassPanel onMaximize={() => setMaximizePanelId('quantum-exchange')} title="Quantum App Exchange">
                               <div className="absolute top-2 right-14 z-50">
                                 <button onClick={(e) => { e.stopPropagation(); setFlipP15(false); }} className="p-1.5 hover:bg-pink-500/20 rounded-md border border-pink-800 text-pink-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                               </div>
                               <QuantumAppExchange />
                           </GlassPanel>
                       </div>
                   </div>
               </div>
            </>
          ) : dashboardGroup === 'group4' ? (
            <>
              <div className="col-span-12 lg:col-span-4 h-full relative" style={{ perspective: '3000px' }}>
                  <div className={`relative w-full h-full transition-transform duration-1000 transform-style-preserve-3d ${flipP19 ? 'rotate-y-180' : ''}`}>
                      <div className={`absolute inset-0 backface-hidden ${!flipP19 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                           <GlassPanel onMaximize={() => setMaximizePanelId('qcos-dashboard')} title="QCOS Dashboard">
                               <div className="absolute top-2 right-14 z-50">
                                 <button onClick={(e) => { e.stopPropagation(); setFlipP19(true); }} className="p-1.5 hover:bg-red-500/20 rounded-md border border-red-800 text-red-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                               </div>
                               <QcosDashboard />
                           </GlassPanel>
                       </div>
                       <div className={`absolute inset-0 backface-hidden rotate-y-180 ${flipP19 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                           <GlassPanel onMaximize={() => setMaximizePanelId('quantum-rl')} title="Quantum Reinforcement Learning">
                               <div className="absolute top-2 right-14 z-50">
                                 <button onClick={(e) => { e.stopPropagation(); setFlipP19(false); }} className="p-1.5 hover:bg-green-500/20 rounded-md border border-green-800 text-green-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                               </div>
                               <QuantumReinforcementLearning />
                           </GlassPanel>
                       </div>
                   </div>
               </div>

               <div className="col-span-12 lg:col-span-5 h-full relative" style={{ perspective: '3000px' }}>
                   <div className={`relative w-full h-full transition-transform duration-1000 transform-style-preserve-3d ${flipP20 ? 'rotate-y-180' : ''}`}>
                       <div className={`absolute inset-0 backface-hidden ${!flipP20 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                           <GlassPanel onMaximize={() => setMaximizePanelId('quantum-programming')} title="Quantum Programming Interface">
                               <div className="absolute top-2 right-14 z-50">
                                 <button onClick={(e) => { e.stopPropagation(); setFlipP20(true); }} className="p-1.5 hover:bg-orange-500/20 rounded-md border border-orange-800 text-orange-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                               </div>
                               <QuantumProgrammingInterface />
                           </GlassPanel>
                       </div>
                       <div className={`absolute inset-0 backface-hidden rotate-y-180 ${flipP20 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                           <GlassPanel onMaximize={() => setMaximizePanelId('quantum-llm')} title="Quantum Large Language Model">
                               <div className="absolute top-2 right-14 z-50">
                                 <button onClick={(e) => { e.stopPropagation(); setFlipP20(false); }} className="p-1.5 hover:bg-teal-500/20 rounded-md border border-teal-800 text-teal-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                               </div>
                               <QuantumLargeLanguageModel />
                           </GlassPanel>
                       </div>
                   </div>
               </div>

               <div className="col-span-12 lg:col-span-3 h-full relative" style={{ perspective: '3000px' }}>
                   <div className={`relative w-full h-full transition-transform duration-1000 transform-style-preserve-3d ${flipP21 ? 'rotate-y-180' : ''}`}>
                       <div className={`absolute inset-0 backface-hidden ${!flipP21 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                           <GlassPanel onMaximize={() => setMaximizePanelId('quantum-exchange')} title="Quantum App Exchange">
                               <div className="absolute top-2 right-14 z-50">
                                 <button onClick={(e) => { e.stopPropagation(); setFlipP21(true); }} className="p-1.5 hover:bg-indigo-500/20 rounded-md border border-indigo-800 text-indigo-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                               </div>
                               <QuantumAppExchange />
                           </GlassPanel>
                       </div>
                       <div className={`absolute inset-0 backface-hidden rotate-y-180 ${flipP21 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                           <GlassPanel onMaximize={() => setMaximizePanelId('agi-singularity')} title="AGI Singularity Interface">
                               <div className="absolute top-2 right-14 z-50">
                                 <button onClick={(e) => { e.stopPropagation(); setFlipP21(false); }} className="p-1.5 hover:bg-pink-500/20 rounded-md border border-pink-800 text-pink-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                               </div>
                               <AGISingularityInterface />
                           </GlassPanel>
                       </div>
                   </div>
               </div>
            </>
          ) : dashboardGroup === 'group5' ? (
            <>
              <div className="col-span-12 lg:col-span-4 h-full relative" style={{ perspective: '3000px' }}>
                  <div className={`relative w-full h-full transition-transform duration-1000 transform-style-preserve-3d ${flipP25 ? 'rotate-y-180' : ''}`}>
                      <div className={`absolute inset-0 backface-hidden ${!flipP25 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                           <GlassPanel onMaximize={() => setMaximizePanelId('q-biomed')} title="BioMed Discovery">
                               <div className="absolute top-2 right-14 z-50">
                                 <button onClick={(e) => { e.stopPropagation(); setFlipP25(true); }} className="p-1.5 hover:bg-red-500/20 rounded-md border border-red-800 text-red-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                               </div>
                               <QBioMedDrugDiscovery />
                           </GlassPanel>
                       </div>
                       <div className={`absolute inset-0 backface-hidden rotate-y-180 ${flipP25 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                           <GlassPanel onMaximize={() => setMaximizePanelId('mol-sim')} title="Molecular Toolkit">
                               <div className="absolute top-2 right-14 z-50">
                                 <button onClick={(e) => { e.stopPropagation(); setFlipP25(false); }} className="p-1.5 hover:bg-green-500/20 rounded-md border border-green-800 text-green-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                               </div>
                               <MolecularSimulationToolkit />
                           </GlassPanel>
                       </div>
                   </div>
               </div>

               <div className="col-span-12 lg:col-span-5 h-full relative" style={{ perspective: '3000px' }}>
                   <div className={`relative w-full h-full transition-transform duration-1000 transform-style-preserve-3d ${flipP26 ? 'rotate-y-180' : ''}`}>
                       <div className={`absolute inset-0 backface-hidden ${!flipP26 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                           <GlassPanel onMaximize={() => setMaximizePanelId('quantum-cognitive-architecture')} title="Cognitive Architecture">
                               <div className="absolute top-2 right-14 z-50">
                                 <button onClick={(e) => { e.stopPropagation(); setFlipP26(true); }} className="p-1.5 hover:bg-orange-500/20 rounded-md border border-orange-800 text-orange-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                               </div>
                               <QuantumCognitiveArchitecture />
                           </GlassPanel>
                       </div>
                       <div className={`absolute inset-0 backface-hidden rotate-y-180 ${flipP26 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                           <GlassPanel onMaximize={() => setMaximizePanelId('quantum-deep-learning')} title="Quantum Deep Learning">
                               <div className="absolute top-2 right-14 z-50">
                                 <button onClick={(e) => { e.stopPropagation(); setFlipP26(false); }} className="p-1.5 hover:bg-teal-500/20 rounded-md border border-teal-800 text-teal-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                               </div>
                               <QuantumDeepLearning />
                           </GlassPanel>
                       </div>
                   </div>
               </div>

               <div className="col-span-12 lg:col-span-3 h-full relative" style={{ perspective: '3000px' }}>
                   <div className={`relative w-full h-full transition-transform duration-1000 transform-style-preserve-3d ${flipP27 ? 'rotate-y-180' : ''}`}>
                       <div className={`absolute inset-0 backface-hidden ${!flipP27 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                           <GlassPanel onMaximize={() => setMaximizePanelId('qos-kernel-manager')} title="QOS Kernel Console">
                               <div className="absolute top-2 right-14 z-50">
                                 <button onClick={(e) => { e.stopPropagation(); setFlipP27(true); }} className="p-1.5 hover:bg-indigo-500/20 rounded-md border border-indigo-800 text-indigo-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                               </div>
                               <QOSKernelConsole />
                           </GlassPanel>
                       </div>
                       <div className={`absolute inset-0 backface-hidden rotate-y-180 ${flipP27 ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-500`}>
                           <GlassPanel onMaximize={() => setMaximizePanelId('system-diagnostic')} title="System Diagnostics">
                               <div className="absolute top-2 right-14 z-50">
                                 <button onClick={(e) => { e.stopPropagation(); setFlipP27(false); }} className="p-1.5 hover:bg-pink-500/20 rounded-md border border-pink-800 text-pink-400 transition-all hover:scale-110 bg-black/60 backdrop-blur-sm"><RefreshCwIcon className="w-4 h-4"/></button>
                               </div>
                               <SystemDiagnostic onClose={() => setMaximizePanelId(null)} />
                           </GlassPanel>
                       </div>
                   </div>
               </div>
            </>
          ) : null}
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
                onClick={() => setDashboardGroup(prev => {
                   if (prev === 'group1') return 'group2';
                   if (prev === 'group2') return 'group3';
                   if (prev === 'group3') return 'group4';
                   if (prev === 'group4') return 'group5';
                   return 'group1';
                 })}
                className={`holographic-projection px-5 py-2 text-[10px] font-black uppercase tracking-widest border rounded-full transition-all flex items-center gap-2 ${dashboardGroup === 'group2' ? 'bg-purple-900/20 border-purple-500/40 text-purple-300' : 'bg-cyan-900/10 border-cyan-500/40 text-cyan-300'}`}
              >
                 <GridIcon className="w-3 h-3" />
                 {dashboardGroup === 'group1' ? 'Shift to Panel Group 2' : dashboardGroup === 'group2' ? 'Shift to Panel Group 3' : dashboardGroup === 'group3' ? 'Shift to Panel Group 4' : dashboardGroup === 'group4' ? 'Shift to Panel Group 5' : 'Shift to Panel Group 1'}
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